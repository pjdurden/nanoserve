"""The read rectangle, kept on the device instead of rebuilt on the host. Day 51.

Day 50 moved a decode step's addressing out of the forward and then priced it. The
`[rows, max_ctx]` read rectangle is int64 built from scratch on the host every
step, and `max_ctx` grows by one per step, so step i writes `rows * (start + i)`
cells to append `rows`:

    steps    cells rebuilt   cells appended    ratio    host bytes
        8              656               32      20x        5.2 KB
      512          558,080            2,048     272x        4.5 MB

Half a million int64 written to say where two thousand tokens are, and
`check_rebuild_bounded` is the gate that fails it. The design was correct and
traceable and doing O(n^2) host work to append n tokens.

**So the rectangle stops being built.** A `SlotTable` is one
`[max_batch_size, max_model_len]` int64 buffer allocated once. A decode step writes
exactly one cell per row into it, at column `length`, and the thing the forward
reads is `slots[:rows, :width]`: a *window*, the same storage at the same address,
with the buffer's row stride. Per-step host work drops from `rows * max_ctx` to
`rows`, and the per-step allocation drops to nothing.

**A window is only a window when the rows are a prefix.** `slots[:n, :w]` is basic
slicing and therefore a view. `slots[[0, 2], :w]` is advanced indexing and
therefore a copy, and a scheduler hands out whichever row slots are free, so a
real batch is often not a prefix. The gather is still much cheaper than the
rebuild (it is `rows * width` on the *device*, from data already there, with no
host loop and no transfer), but it is not the same object, and the difference
matters for exactly one reason: a CUDA graph replays kernels bound to fixed
addresses. `check_mapping_is_window` is the gate that says whether this step could
have been captured, and `window_share` is how often a real run manages it.

**The lengths could have been persistent too, and deliberately are not.** This is
the day's actual lesson and it took a broken gate to find. Appends write at column
`length`, which is past the real region of any window somebody is still holding, so
a held rectangle keeps saying what it said. The lengths are the opposite: the
append moves the very number a previous plan is holding. Day 50's
`check_plan_current` compares a plan's `context_lens` against the cache's tables,
so a `context_lens` that read live out of the table would *always* agree with them:
the gate would stop being able to fail rather than start failing. The lengths are
`[rows]` and linear, so a copy costs nothing and keeps the gate armed. One buffer
is safe to share and the other is not, and the difference is whether a later step
writes inside the window or past its edge.

**The mirror is not a second source of truth.** `BlockTable` still owns where a
token lives. This table copies it: `sync` catches up any row whose length
disagrees, which happens after a prefill and after a row changes tenant, and
`check_slots_agree` is the gate that the copy has not drifted. `rebuild_mapping`
below is the Day-50 construction, extracted, and it is both the resync path and the
oracle the table is graded against.

**And the price is a fixed allocation.** `[max_batch_size, max_model_len]` int64 is
reserved for the process: 256 rows at 131,072 tokens is 268 MB of pure addressing.
vLLM does not keep this table. It keeps *block ids*, `[max_seqs, max_blocks]`,
which is the same table divided by the block size, and computes
`block_id * block_size + offset` in the kernel. That is a 16x saving at
`block_size=16` and it is the reason vLLM can afford one at full context.
`block_table_bytes` is that arithmetic, `check_table_fits` is the gate, and the
reason nanoserve stores slots today is that its reference read gathers per token
and has nowhere to do the multiply. Naming what the next version does differently,
in the module that does not do it, is the same order Day 50 used.
"""

from __future__ import annotations

import torch

#: How many times a row may be rebuilt from its block table over a run before the
#: mirror is not a mirror but a rebuild with extra steps. One is the prefill; the
#: allowance is for the recompute a preempted request comes back through.
DEFAULT_RESYNC_LIMIT = 2.0

#: Bytes per entry. An index tensor is int64, which is what makes the rectangle
#: expensive in bytes as well as in cells: 8 bytes to say where one token lives.
SLOT_ITEMSIZE = 8


class SlotTableFull(RuntimeError):
    """A row asked for more addressing than the table was sized for.

    A `RuntimeError` and not an assertion, because it is the same class of event as
    `KVCacheExhausted`: a real limit reached by a legal request, which a server
    answers by refusing that request rather than by crashing. `max_model_len` is a
    serving decision and this is where it becomes a fact.
    """


class SlotsUnsound(AssertionError):
    """The persistent table does not describe the cache it is supposed to mirror.

    An `AssertionError` for the same reason `PlanUnsound` is one: these are checks
    a test suite runs and a benchmark asserts, not conditions an engine recovers
    from. A mirror that has drifted does not raise on its own. It hands one
    sequence another sequence's slots and the run keeps producing plausible text.
    """


# --- the construction this module replaces --------------------------------------


def rebuild_mapping(tables, rows, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a `[rows, max_ctx]` rectangle from block tables. The Day-50 path.

    Extracted rather than deleted, and it has three jobs now. It is what a row's
    first sync copies in, it is the oracle the persistent table is graded against
    in the tests, and it is the arm `slotbench.py` times the incremental one
    against. Cost is `rows * max_ctx` Python-level `slot()` calls plus one
    host-to-device build of that many int64, every time it is called.

    Padding is slot 0 for the reason `BatchedPagedKVCache.slot_mapping` documents:
    the reference gathers the whole rectangle before it masks, so a padded entry is
    really dereferenced and has to be a legal index. A persistent table pads with
    whatever the buffer already held, which is a legal index for the same reason
    (nothing but a real slot is ever written into it) and is not zero.
    """
    rows = tuple(int(r) for r in rows)
    if not rows:
        raise ValueError("a rectangle needs at least one row")
    lens = [tables[r].num_tokens for r in rows]
    max_ctx = max(lens)
    if max_ctx < 1:
        raise ValueError(f"nothing is cached yet in rows {list(rows)}: prefill first")
    grid = [
        [tables[r].slot(p) for p in range(n)] + [0] * (max_ctx - n)
        for r, n in zip(rows, lens)
    ]
    return (
        torch.tensor(grid, dtype=torch.long, device=device),
        torch.tensor(lens, dtype=torch.long, device=device),
    )


# --- the table --------------------------------------------------------------------


class SlotTable:
    """`[max_batch_size, max_model_len]` slots on the device, written one at a time.

    Row `r`'s first `length(r)` entries are its own pool slots, oldest first.
    Everything past that is whatever the buffer last held there, which is padding:
    it is gathered by the reference read and then masked away by `context_lens`, so
    the only thing it ever owed anyone is being a legal index into the pool, and
    every value ever written here is one.

    Nothing is zeroed on reset and nothing is compacted. A row that finishes and is
    handed to the next request just has its length set to 0, and the next tenant
    overwrites from column 0 as it goes. Clearing would be `max_model_len` of device
    writes to hide values that are already masked.

    The counters are the day's evidence and are meant to be read: `appended_cells`
    is what an incremental design writes, `resynced_cells` is what it still has to
    rebuild, and `windows` against `gathers` is how often the rectangle came back as
    a view rather than a copy.
    """

    def __init__(self, max_batch_size: int, max_model_len: int, device=None):
        if max_batch_size < 1:
            raise ValueError(f"a table has at least one row; got {max_batch_size}")
        if max_model_len < 1:
            raise ValueError(f"a row holds at least one token; got {max_model_len}")
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        # The one allocation. Zero-filled because zero is a legal slot, so an
        # untouched cell is already valid padding on the very first read.
        self.slots = torch.zeros(
            (max_batch_size, max_model_len), dtype=torch.long, device=device
        )
        # Host-side lengths. Deliberately *not* a device buffer: see the module
        # docstring. They are `[rows]`, so persistence would buy nothing, and a
        # length the cache can move underneath a plan disarms Day 50's staleness
        # gate instead of tripping it.
        self._lengths = [0] * max_batch_size
        self.appended_cells = 0
        self.resynced_cells = 0
        self.resyncs = 0
        self.windows = 0
        self.gathers = 0
        self.moves = 0
        # Day 52. Reads that were widened past the rows the caller asked for, and
        # the cells that cost. Padding a batch up to a row bucket is what keeps the
        # forward's shape constant; these two say how often and how much.
        self.pads = 0
        self.padded_cells = 0

    # --- what it is ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.slots.device

    @property
    def cells(self) -> int:
        """Entries in the buffer, occupied and reserved alike."""
        return self.max_batch_size * self.max_model_len

    @property
    def bytes(self) -> int:
        """What the buffer weighs. Allocated once and held for the process."""
        return table_bytes(self.max_batch_size, self.max_model_len)

    @property
    def window_share(self) -> float:
        """Fraction of reads that came back as a view on the buffer.

        The fraction of steps a CUDA graph could have replayed without the
        rectangle moving. Below 1.0 means the batch's rows were not a prefix of the
        table's, which is a scheduler property, not a cache one.
        """
        reads = self.windows + self.gathers
        return self.windows / reads if reads else 0.0

    def length(self, row: int) -> int:
        """Tokens addressed in this row. A host int, so nobody waits on it."""
        return self._lengths[self._row(row)]

    def render(self) -> str:
        """One line for a log: shape, price, and how the run has used it."""
        return (
            f"{self.max_batch_size} rows x {self.max_model_len} tokens = "
            f"{self.cells} cells, {self.bytes} bytes, "
            f"{self.appended_cells} appended + {self.resynced_cells} resynced, "
            f"{self.window_share:.0%} windows"
        )

    # --- rows -----------------------------------------------------------------

    def _row(self, row) -> int:
        row = int(row)
        if row < 0 or row >= self.max_batch_size:
            raise ValueError(
                f"row {row} is out of range for a table of {self.max_batch_size}"
            )
        return row

    def _rows(self, rows) -> tuple[int, ...]:
        rows = tuple(self._row(r) for r in rows)
        if not rows:
            raise ValueError("a read covers at least one row")
        if len(set(rows)) != len(rows):
            raise ValueError(f"rows must be distinct, got {list(rows)}")
        return rows

    def to(self, device) -> SlotTable:
        """Put the buffer on `device`. A no-op there, a reallocation anywhere else.

        Worth being explicit about: moving is the one operation that changes the
        buffer's address, so any window handed out before a move points at freed
        storage. In practice this happens once, on the first decode of a process,
        because the cache learns its device from the first K/V it is given.
        """
        if device is None:
            return self
        target = torch.device(device)
        current = self.slots.device
        if target.type == current.type and (
            target.index is None or target.index == current.index
        ):
            return self
        self.slots = self.slots.to(target)
        self.moves += 1
        return self

    # --- writing --------------------------------------------------------------

    def append(self, rows, slots) -> None:
        """Give each named row one more token, at that row's own next column.

        The whole point of the module in four lines: `rows` cells written, whatever
        the histories are worth. No rectangle is built, nothing is copied, and the
        column each row writes at is its own length, which is why a row that is
        shorter than the batch writes *inside* a rectangle an earlier step handed
        out. That cell is padding as far as that rectangle's `context_lens` is
        concerned, and it stays padding because those lengths are a snapshot.
        """
        rows = self._rows(rows)
        slots = [int(s) for s in slots]
        if len(slots) != len(rows):
            raise ValueError(
                f"an append is one slot per row: {len(rows)} rows {list(rows)} and "
                f"{len(slots)} slots"
            )
        for row in rows:
            if self._lengths[row] >= self.max_model_len:
                raise SlotTableFull(
                    f"row {row} already holds {self._lengths[row]} tokens and the "
                    f"table is {self.max_model_len} wide: this is max_model_len "
                    "reached, and the answer is to finish the request, not to grow "
                    "the buffer"
                )
        index = torch.tensor(
            [self._lengths[r] for r in rows], dtype=torch.long, device=self.slots.device
        )
        rows_index = torch.tensor(rows, dtype=torch.long, device=self.slots.device)
        values = torch.tensor(slots, dtype=torch.long, device=self.slots.device)
        self.slots[rows_index, index] = values
        for row in rows:
            self._lengths[row] += 1
        self.appended_cells += len(rows)

    def resync(self, table, row: int) -> None:
        """Copy a whole row in from its block table. The path that is not cheap.

        Called when the mirror has fallen behind, which happens on exactly two
        events: a prefill wrote many tokens at once, and a row was reset and handed
        to a new request. Both are once-per-request, so the amortised cost over a
        generation is the prompt divided by the number of steps, and
        `check_resyncs_bounded` is the gate that it stays that way.
        """
        row = self._row(row)
        n = table.num_tokens
        if n > self.max_model_len:
            raise SlotTableFull(
                f"row {row} holds {n} tokens and the table is {self.max_model_len} "
                "wide: this row was grown past max_model_len somewhere else"
            )
        if n:
            self.slots[row, :n] = torch.tensor(
                [table.slot(p) for p in range(n)],
                dtype=torch.long,
                device=self.slots.device,
            )
        self._lengths[row] = n
        self.resyncs += 1
        self.resynced_cells += n

    def sync(self, tables, rows) -> tuple[int, ...]:
        """Resync every named row whose length disagrees with its block table.

        Length is the whole test, and it is enough: a row's slots only ever change
        by being appended to (which the mirror does itself) or by the row changing
        tenant (which goes through `reset`, so the length drops to zero first).
        Returns the rows it rewrote, so a caller can log or gate on how often this
        happens rather than only on what it cost.
        """
        rows = self._rows(rows)
        moved = tuple(r for r in rows if self._lengths[r] != tables[r].num_tokens)
        for row in moved:
            self.resync(tables[row], row)
        return moved

    def reset(self, row: int) -> None:
        """Hand a row back empty. Nothing is cleared; what is past a length is padding."""
        self._lengths[self._row(row)] = 0

    def reset_all(self) -> None:
        self._lengths = [0] * self.max_batch_size

    # --- reading --------------------------------------------------------------

    def read(
        self, rows, width: int, *, pad_rows: int = 0, lengths_writer=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The rectangle and the lengths, for a forward over `rows`.

        The rectangle is `slots[:n, :width]` when `rows` is a prefix of the table's,
        which is a view: the buffer's own storage, its own address, its own row
        stride. Otherwise it is `slots[:, :width].index_select(0, rows)`, a device
        gather of `rows * width` cells, narrowed *first* so the copy is the
        rectangle and not the whole `max_model_len`-wide buffer.

        The lengths are always a fresh tensor and that is deliberate, not an
        oversight: see the module docstring. They are `[rows]`, so it costs `rows`
        int64, and it is what keeps a plan checkable after the cache has moved on.

        `lengths_writer` is Day 53 and it is the one thing about the lengths that
        does change. It takes the host list and returns the tensor, so a caller that
        already keeps a persistent `[max_batch]` buffer can have them written in
        place instead of allocating a fresh vector here every step. The table says
        what the lengths are; where they land is the caller's business, and the
        caller is the one that has to carry a snapshot once they land somewhere
        shared. See `nanoserve.inputs`.

        `pad_rows` is Day 52. It widens the rectangle to `len(rows) + pad_rows` so
        that a batch can be padded up to a bucket and the forward's shape stops
        moving. The padded rows are the table's *next* rows, which costs nothing
        over a prefix (`slots[:n + pad, :width]` is the same basic slice, so the
        result is still a window at the buffer's own address) and costs a duplicated
        index over a gather. What is in those rows is another tenant's slots or
        nothing at all, and neither is read: their length is 0, so the whole row is
        masked. The length is what makes the padding safe, and it is the reason the
        lengths are built here rather than pointed at.
        """
        rows = self._rows(rows)
        width = int(width)
        pad_rows = int(pad_rows)
        if pad_rows < 0:
            raise ValueError(f"a read pads a non-negative number of rows; got {pad_rows}")
        if len(rows) + pad_rows > self.max_batch_size:
            raise ValueError(
                f"a read over {len(rows)} rows padded by {pad_rows} needs "
                f"{len(rows) + pad_rows} rows and the table has {self.max_batch_size}: "
                "a row bucket bigger than the cache is a bucket with nothing to pad into"
            )
        if width < 1:
            raise ValueError(f"a rectangle is at least one column wide; got {width}")
        if width > self.max_model_len:
            raise ValueError(
                f"a rectangle is at most {self.max_model_len} wide, the table's "
                f"max_model_len; got {width}"
            )
        for row in rows:
            if self._lengths[row] > width:
                raise ValueError(
                    f"row {row} holds {self._lengths[row]} tokens and the rectangle "
                    f"is {width} wide: the read would not see the whole history"
                )
        if rows == tuple(range(len(rows))):
            mapping = self.slots[: len(rows) + pad_rows, :width]
            self.windows += 1
        else:
            # A gather has no "next rows" to widen into, so the padding repeats the
            # first row's index. It reads real slots and is masked away by a length
            # of 0, exactly as the prefix case is.
            index = torch.tensor(
                rows + (rows[0],) * pad_rows, dtype=torch.long, device=self.slots.device
            )
            mapping = self.slots[:, :width].index_select(0, index)
            self.gathers += 1
        values = [self._lengths[r] for r in rows] + [0] * pad_rows
        if lengths_writer is None:
            lengths = torch.tensor(values, dtype=torch.long, device=self.slots.device)
        else:
            lengths = lengths_writer(values)
        if pad_rows:
            self.pads += 1
            self.padded_cells += pad_rows * width
        return mapping, lengths

    def is_window(self, tensor: torch.Tensor) -> bool:
        """Whether `tensor` is this buffer's storage rather than a copy of it."""
        return (
            tensor.untyped_storage().data_ptr() == self.slots.untyped_storage().data_ptr()
        )


# --- what it costs -----------------------------------------------------------------


def table_cells(max_batch_size: int, max_model_len: int) -> int:
    """Entries in the persistent table. Reserved at construction, all of them."""
    if max_batch_size < 1:
        raise ValueError(f"a table has at least one row; got {max_batch_size}")
    if max_model_len < 1:
        raise ValueError(f"a row holds at least one token; got {max_model_len}")
    return max_batch_size * max_model_len


def table_bytes(
    max_batch_size: int, max_model_len: int, itemsize: int = SLOT_ITEMSIZE
) -> int:
    """What the table weighs. int64, because that is what an index tensor is."""
    if itemsize < 1:
        raise ValueError(f"an entry is at least one byte; got {itemsize}")
    return table_cells(max_batch_size, max_model_len) * itemsize


def block_table_cells(max_batch_size: int, max_model_len: int, block_size: int) -> int:
    """Entries in the table vLLM actually keeps: block ids, not slots.

    A slot is `block_id * block_size + offset`, and the offset is `position %
    block_size`, which the kernel already knows because it is walking positions. So
    storing the block id per *block* rather than the slot per *token* is the same
    information `block_size` times smaller, and the multiply moves into the read.
    """
    if block_size < 1:
        raise ValueError(f"a block holds a positive number of tokens; got {block_size}")
    blocks = -(-max_model_len // block_size)  # ceil, a partial block still needs an id
    return table_cells(max_batch_size, blocks)


def block_table_bytes(
    max_batch_size: int,
    max_model_len: int,
    block_size: int,
    itemsize: int = SLOT_ITEMSIZE,
) -> int:
    if itemsize < 1:
        raise ValueError(f"an entry is at least one byte; got {itemsize}")
    return block_table_cells(max_batch_size, max_model_len, block_size) * itemsize


def incremental_cells(steps: int, rows: int, start_ctx: int) -> int:
    """Cells the host writes over a run that appends instead of rebuilding.

    One pass over each row's prompt when the mirror first catches up, then one cell
    per row per step forever: `rows * (start_ctx + steps)`. Linear where
    `plan.rebuild_cells` is quadratic, and the whole of the day.
    """
    if steps < 0:
        raise ValueError(f"a run has a non-negative number of steps; got {steps}")
    if rows < 1:
        raise ValueError(f"a step covers at least one row; got {rows}")
    if start_ctx < 1:
        raise ValueError(f"a decode row starts with a prefill; got start_ctx={start_ctx}")
    return rows * (start_ctx + steps)


def resident_share(table_bytes_: int, pool_bytes: int) -> float:
    """What fraction of a KV budget the addressing table takes before a token lands."""
    if pool_bytes <= 0:
        raise ValueError(f"a pool is a positive number of bytes; got {pool_bytes}")
    return table_bytes_ / pool_bytes


# --- gates ---------------------------------------------------------------------------


def check_slots_agree(table: SlotTable, tables, rows) -> None:
    """Refuse a mirror that has drifted from the block tables it copies.

    The one failure mode a mirror has. `BlockTable` owns where a token lives and
    this table repeats it, so two things now say the same thing and only one of them
    is right when they differ. Nothing raises on its own when they do: the slots are
    legal, the pool is legal, and one row reads another row's K/V.
    """
    for row in table._rows(rows):
        want = tables[row].num_tokens
        got = table.length(row)
        if got != want:
            raise SlotsUnsound(
                f"row {row}: the block table holds {want} tokens and the slot table "
                f"has {got} tokens for it"
            )
        for position in range(want):
            mirrored = int(table.slots[row, position])
            actual = tables[row].slot(position)
            if mirrored != actual:
                raise SlotsUnsound(
                    f"row {row} position {position}: the block table says slot "
                    f"{actual} and the slot table says {mirrored}"
                )


def check_mapping_is_window(table: SlotTable, plan) -> None:
    """Refuse a step whose rectangle is not the persistent buffer's own storage.

    Not a correctness gate, and it is important to say which one it is: a gathered
    rectangle computes exactly the same attention. It is a *capture* gate. A CUDA
    graph replays kernels bound to the addresses they were recorded with, so a
    rectangle that is a fresh allocation every step cannot be replayed, and neither
    can one that is a fresh allocation only on the steps where the scheduler's rows
    were not a prefix. Which is why the message names the row set: the fix is on the
    scheduler's side, not here.
    """
    rows = plan.rows
    if rows != tuple(range(len(rows))):
        raise SlotsUnsound(
            f"rows {list(rows)} are not a prefix of the table's, so the rectangle is "
            "an index_select and not a view: only a prefix narrows to a window at the "
            "buffer's own address"
        )
    if not table.is_window(plan.slot_mapping):
        raise SlotsUnsound(
            "this rectangle is its own allocation and not a window on the persistent "
            "table: a graph captured over it would replay against freed storage"
        )


def check_window_intact(plan) -> None:
    """Refuse a held rectangle that something has since written inside.

    A window is live storage, which is the trade the day makes. Ordinary steps
    cannot disturb one: an append lands at column `length`, which is past the real
    region of every window handed out before it. A row that changes tenant is the
    case that can, because its resync rewrites from column 0, and any plan still
    holding that row's window is then pointing at somebody else's slots. Legal
    indices, a legal pool, and the wrong sequence's keys.

    The witnesses are the plan's own copies. `write_slots` and `context_lens` were
    taken at build time and cannot move (Day 53 makes both of them buffers the next
    step writes through, and that is exactly why a plan over one carries a host-side
    snapshot: `plan.context_list` and `plan.write_list` are the copies, wherever
    they now live); `slot_mapping` is the view. So the check is
    the one Day 50 already wrote, "the write slot is the row's last real entry in
    the rectangle", read the other way round: there it was a statement about how a
    rectangle had been *constructed*, and over a window it is a statement about
    whether it still means anything. Same two lines, different question, and the
    second question only exists because the storage is shared.

    Comparing the window against the block tables instead would prove nothing: the
    window tracks them, so both sides move together and the check always passes.
    """
    lengths, slots = plan.context_list, plan.write_list
    for i, row in enumerate(plan.rows):
        n = lengths[i]
        last = int(plan.slot_mapping[i, n - 1])
        recorded = slots[i]
        if last != recorded:
            raise SlotsUnsound(
                f"row {row} was planned to write slot {recorded} and its window now "
                f"ends at slot {last}: something rewrote this rectangle underneath a "
                "plan that is still holding it"
            )


def check_appends_incremental(table: SlotTable, *, rows: int, steps: int) -> None:
    """Refuse a run that wrote more than one cell per row per step.

    The definition of the design, as an equality rather than a bound. Anything
    other than `rows * steps` means the rectangle went back to being built, which is
    a regression that shows up in a profile long before it shows up in a test.
    """
    if rows < 1 or steps < 0:
        raise ValueError(f"got rows={rows}, steps={steps}")
    want = rows * steps
    if table.appended_cells != want:
        raise SlotsUnsound(
            f"{steps} steps over {rows} rows appended {table.appended_cells} cells, "
            f"not {want}: an incremental table writes exactly one cell per row per "
            "step and anything else is a rebuild wearing this class"
        )


def check_resyncs_bounded(
    table: SlotTable, *, rows: int, limit: float = DEFAULT_RESYNC_LIMIT
) -> None:
    """Refuse a run that keeps rebuilding rows from their block tables.

    The resync is the expensive path and it is supposed to be rare: once when a
    prefill lands, once more if the request is preempted and recomputed. A resync
    per step is a mirror that is not mirroring, and it costs the whole rebuild the
    day was written to remove while looking like it does not.
    """
    if rows < 1:
        raise ValueError(f"a run covers at least one row; got {rows}")
    if limit <= 0:
        raise ValueError(f"a resync limit is a positive multiple; got {limit}")
    allowed = rows * limit
    if table.resyncs > allowed:
        raise SlotsUnsound(
            f"{rows} rows took {table.resyncs} resyncs against an allowance of "
            f"{allowed:.0f}: a row is rebuilt when it is prefilled and when it is "
            "recomputed, and a table that rebuilds more often than that is paying "
            "Day 50's bill through a different door"
        )


def check_table_fits(
    max_batch_size: int,
    max_model_len: int,
    *,
    budget_bytes: int,
    itemsize: int = SLOT_ITEMSIZE,
    block_size: int | None = None,
) -> None:
    """Refuse a persistent table that costs more than the addressing is worth.

    The new price. Rebuilding was quadratic in the run length and allocated nothing;
    this is constant in the run length and allocates up front, so the failure mode
    moved from "the profile fills up with host work" to "the process reserves 268 MB
    to address a pool it has not sized yet". `max_model_len` is the knob, and
    passing `block_size` prints the table vLLM would have kept instead.
    """
    if budget_bytes <= 0:
        raise ValueError(f"a budget is a positive number of bytes; got {budget_bytes}")
    cost = table_bytes(max_batch_size, max_model_len, itemsize)
    if cost <= budget_bytes:
        return
    message = (
        f"a {max_batch_size} x {max_model_len} slot table is {cost} bytes against a "
        f"budget of {budget_bytes}: lower max_model_len, lower max_batch_size, or "
        "store something smaller"
    )
    if block_size is not None:
        smaller = block_table_bytes(max_batch_size, max_model_len, block_size, itemsize)
        message += (
            f". The same addressing as block ids at block_size={block_size} is "
            f"{smaller} bytes, {cost // max(smaller, 1)}x smaller, which is what vLLM "
            "keeps"
        )
    raise SlotsUnsound(message)
