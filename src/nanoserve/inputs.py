"""The decode step's inputs, in buffers that do not move. Week 13, Day 53.

Four days have been spent turning a decode step into something a capture can hold.
Day 49 removed the graph breaks and found the guard was on `table.num_tokens`. Day
50 moved the addressing out of the forward so the only thing left to guard on is a
tensor shape. Day 51 made the read rectangle a window on a persistent table, so its
address stopped moving. Day 52 rounded both axes of that rectangle into a closed
bucket set, so the shape stopped moving.

**What is left is the other three tensors, and they are all still new every step.**
`positions` and `write_slots` are built out of Python lists, `context_lens` is built
inside `SlotTable.read`, and the engine's padded `input_ids` is a `torch.cat`. Every
one of them is a fresh allocation at a fresh address, and a replayed CUDA graph does
not take arguments: it re-runs the kernels it recorded, reading the buffers those
kernels were recorded against. An input that moves is an input the graph never sees,
and the failure is silent in the worst possible way, because the graph happily
replays over whatever the old address now holds.

So the inputs get allocated once. A `DecodeInputs` is four `[max_batch_size]` int64
buffers, a step writes cells into them, and what the forward is handed is
`buffer[:graph_rows]`: a window, the same storage at the same address, exactly the
trade Day 51 made for the rectangle. `check_addresses_stable` is the gate, and it is
the whole property stated as an address rather than as a shape.

**The write is in place, which means it needs somewhere to write from.** Putting a
Python list into a device tensor allocates the tensor you are trying not to
allocate. So each buffer keeps a *staging* mirror on the host, pinned when the
buffer is on CUDA, written through its numpy view (no torch allocation at all) and
copied down with one `copy_(non_blocking=True)`. On CPU the staging tensor is the
buffer, so the copy does not exist. This is what vLLM does and it is the reason its
input preparation does not show up as allocation traffic in a profile.

**One value already lives on the device and it is the interesting one.** The token a
decode row forwards is the token the previous step sampled, and Day 47 left it on
the device on purpose, so the fast path was `tokens.unsqueeze(1)`: a view, zero copy,
no journey home. A replay cannot use that. The address belongs to the sampler and it
is new every step, so the buffer turns a zero-copy view into a one-copy
device-to-device write. Day 53 makes a step do *more* work than Day 47's, and the
thing it buys is not throughput, it is an address.

**Sharing storage disarms two gates, and this is the part that took a broken test to
find.** Day 51 kept `context_lens` a fresh copy on purpose: `check_plan_current`
compares a plan's lengths against the cache's tables, so a length that lives in a
buffer the next step writes through *always* agrees with them and the gate stops
being able to fail rather than starting to. `check_window_intact` has the same
problem one level along: its witnesses were `write_slots` and `context_lens`, and
they were witnesses precisely because they were copies. So a plan built over shared
buffers carries `context_snapshot` and `write_snapshot`, host-side tuples taken from
the lists the tensors were written from. They are free (the host already had the
numbers) and they are not optional, which is what `check_snapshots_present` says.
The general rule the day cost me: **when a witness becomes shared storage, it stops
being a witness, and the copy has to become explicit.**

**And the arithmetic, which produced the opposite of the finding it was meant to.**
The question was how many buffer sets a bucket set needs, since Day 52 ends with a
capture list of up to 36 shapes. The answer is one: a graph is recorded per shape,
but a buffer is indexed by *batch position*, so every shape's window is a prefix of
the same storage and the largest row bucket covers all of them. That is
`check_one_set_covers`. The follow-on is that it barely mattered. The whole input
side of a decode step is four `[max_batch]` vectors, which is 8 KB at 256 rows,
against the slot table's 16 MB at 8192 tokens and a per-graph workspace that is
neither. These buffers are worth having for their address and not for their size,
and `per_shape_bytes` is here to say so in numbers rather than in a claim.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Bytes per entry. Everything here is an index or a token id, so everything here
#: is int64, for the same reason `nanoserve.slots` gives.
INPUT_ITEMSIZE = 8

#: The tensors a planned decode step hands the forward, and the shape of one row of
#: each. `input_ids` and `positions` are `[rows, 1]` because a decode query is one
#: token long and the model wants the sequence axis; the other two are `[rows]`.
#: Written down here rather than spelled out four times, because the arithmetic
#: below counts them and the count is the interesting number.
DECODE_INPUTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("input_ids", (1,)),
    ("positions", (1,)),
    ("write_slots", ()),
    ("context_lens", ()),
)


class InputsUnsound(AssertionError):
    """A step's inputs are not the fixed, self-describing things a replay needs.

    An `AssertionError` for the same reason `PlanUnsound`, `SlotsUnsound` and
    `BucketsUnsound` are: these are checks a test suite runs and a benchmark
    asserts, not conditions an engine recovers from. Every failure here is silent
    on its own. An input that moved means a captured graph reads a stale address
    and the run produces fluent, wrong text; a plan with no snapshot means two
    correctness gates pass forever without being able to fail.
    """


# --- one buffer ---------------------------------------------------------------------


class InputBuffer:
    """One `[max_rows, *tail]` int64 buffer, allocated once and written in place.

    name:     what this input is called in an error message. The gates below name
              the buffer that failed, because "an input moved" is not actionable
              and "positions moved" is.
    max_rows: the largest batch the forward will ever run, which is the cache's row
              count. Not a rounding target: a batch past it has nowhere to go.
    tail:     the shape of one row. `()` for a `[rows]` vector, `(1,)` for the
              `[rows, 1]` the model's embedding and rotary want.

    The staging tensor is the half that makes "written in place" true rather than
    approximately true. `buffer[:n] = torch.tensor(values)` allocates the tensor the
    whole class exists to avoid, so the values go into a host mirror through its
    numpy view, which writes into existing storage and allocates nothing, and then
    into the buffer with one copy. On CPU there is nothing to copy to, so the mirror
    *is* the buffer and the write ends there.
    """

    def __init__(self, name: str, max_rows: int, tail: Sequence[int] = (), device=None):
        if max_rows < 1:
            raise ValueError(f"a buffer holds at least one row; got {max_rows}")
        tail = tuple(int(t) for t in tail)
        if any(t < 1 for t in tail):
            raise ValueError(f"a row holds at least one cell; got a tail of {list(tail)}")
        self.name = name
        self.max_rows = max_rows
        self.tail = tail
        self.writes = 0
        self.written_cells = 0
        self.staged_writes = 0
        self.device_writes = 0
        self.moves = 0
        self._allocate(device)

    # --- what it is -----------------------------------------------------------

    def _allocate(self, device) -> None:
        """The one allocation, and the only place a buffer's address is decided.

        Nothing is carried across a reallocation and nothing needs to be: every cell
        a step reads is a cell that step wrote, and the cells past the batch are
        padding whose only duty is to hold a legal value, which zero is.
        """
        target = torch.device("cpu") if device is None else torch.device(device)
        shape = (self.max_rows, *self.tail)
        self.buffer = torch.zeros(shape, dtype=torch.long, device=target)
        if target.type == "cpu":
            # No transfer, so no second tensor: staging into a mirror and copying it
            # to itself would be a memcpy per input per step in exchange for nothing.
            self.staging = self.buffer
        else:
            self.staging = torch.zeros(shape, dtype=torch.long, pin_memory=True)
        self._flat = self.staging.view(-1).numpy()

    @property
    def row_cells(self) -> int:
        """Cells in one row. The divisor a flat write is measured against."""
        cells = 1
        for size in self.tail:
            cells *= size
        return cells

    @property
    def cells(self) -> int:
        """Entries in the buffer, occupied and reserved alike."""
        return self.max_rows * self.row_cells

    @property
    def bytes(self) -> int:
        return self.cells * INPUT_ITEMSIZE

    @property
    def device(self) -> torch.device:
        return self.buffer.device

    @property
    def address(self) -> int:
        """Where this buffer lives. The number a recorded graph bakes in."""
        return self.buffer.untyped_storage().data_ptr()

    def owns(self, tensor: torch.Tensor) -> bool:
        """Whether `tensor` is this buffer's storage rather than a copy of it."""
        return tensor.untyped_storage().data_ptr() == self.address

    def render(self) -> str:
        return (
            f"{self.name}[{self.max_rows}{''.join(f', {t}' for t in self.tail)}] "
            f"{self.bytes} bytes at {self.address:#x}, {self.writes} writes "
            f"({self.staged_writes} staged, {self.device_writes} from the device)"
        )

    # --- moving ---------------------------------------------------------------

    def to(self, device) -> InputBuffer:
        """Put the buffer on `device`. A no-op there, a reallocation anywhere else.

        The one operation that changes the address, which is why it is counted. In
        practice it happens once, on the first decode of a process, because the
        cache learns its device from the first K/V it is handed. A capture taken
        before a move replays against freed storage, so `moves` is a number a
        capture path has to be able to read.
        """
        if device is None:
            return self
        target = torch.device(device)
        current = self.buffer.device
        if target.type == current.type and (
            target.index is None or target.index == current.index
        ):
            return self
        self._allocate(target)
        self.moves += 1
        return self

    # --- writing --------------------------------------------------------------

    def window(self, rows: int) -> torch.Tensor:
        """`buffer[:rows]`: basic slicing, so the buffer's own storage and address."""
        rows = int(rows)
        if rows < 1:
            raise ValueError(f"a window covers at least one row; got {rows}")
        if rows > self.max_rows:
            raise ValueError(
                f"{self.name} holds {self.max_rows} rows and a window of {rows} was "
                "asked for: a batch past the buffer is a batch with nowhere to go"
            )
        return self.buffer[:rows]

    def write(self, values, window: int | None = None) -> torch.Tensor:
        """Put `values` in the buffer's first rows and hand back a window.

        `values` is a flat sequence of `rows * row_cells` numbers, row-major, or a
        tensor already holding them. A list goes through the staging mirror; a
        tensor is copied straight into the buffer, which is the path the previous
        step's sampled tokens take and the one place the values never visit the
        host at all.

        `window` widens what comes back past what was written, and exactly one input
        uses it: a bucketed step's padded `input_ids`. Those rows keep whatever the
        buffer last held, which is a legal token id for the same reason Day 51's
        slot-table padding is a legal slot, because nothing but a legal value is
        ever written here. The other three inputs must not do this. A padded row's
        `write_slots` entry has to be the sink and its `context_lens` entry has to
        be 0, and a stale value in either is a made-up token landing in a real
        sequence's history.
        """
        if isinstance(values, torch.Tensor):
            rows = self._rows_for(values.numel())
            self.buffer[:rows].copy_(values.reshape(rows, *self.tail))
            self.device_writes += 1
        else:
            values = list(values)
            rows = self._rows_for(len(values))
            self._flat[: rows * self.row_cells] = values
            if self.staging is not self.buffer:
                self.buffer[:rows].copy_(self.staging[:rows], non_blocking=True)
            self.staged_writes += 1
        self.writes += 1
        self.written_cells += rows * self.row_cells
        if window is None:
            return self.buffer[:rows]
        window = int(window)
        if window < rows:
            raise ValueError(
                f"{self.name} was written {rows} rows and asked for a window of "
                f"{window}: a window narrower than the write hides rows somebody "
                "just computed"
            )
        return self.window(window)

    def _rows_for(self, cells: int) -> int:
        """How many rows this many cells is, or why it is not a whole number of them."""
        if cells < self.row_cells:
            raise ValueError(f"a write covers at least one row; got {cells} cells")
        if cells % self.row_cells:
            raise ValueError(
                f"{self.name} is {self.row_cells} cells to a row and {cells} cells "
                "does not divide into rows"
            )
        rows = cells // self.row_cells
        if rows > self.max_rows:
            raise ValueError(
                f"{self.name} holds {self.max_rows} rows and {rows} were written: "
                "the buffers are sized from the cache's row count and a batch cannot "
                "be bigger than the cache"
            )
        return rows


# --- the set of them ----------------------------------------------------------------


class DecodeInputs:
    """The four buffers one cache's decode steps write, allocated together.

    Four separate allocations and not one tensor sliced four ways, which is a choice
    worth naming: a fused buffer would make each input's stride the next one's
    problem, and the whole point is that the forward can be handed
    `positions[:rows]` without anybody reasoning about what is next to it.

    Sized from the cache's `max_batch_size`, because that is the largest window any
    shape in a bucket set can ask for and therefore the only number that makes one
    set cover the whole capture list. See `check_one_set_covers`.
    """

    def __init__(self, max_batch_size: int, device=None):
        if max_batch_size < 1:
            raise ValueError(f"a batch has at least one row; got {max_batch_size}")
        self.max_batch_size = max_batch_size
        self.buffers = tuple(
            InputBuffer(name, max_batch_size, tail=tail, device=device)
            for name, tail in DECODE_INPUTS
        )
        self._by_name = {b.name: b for b in self.buffers}

    # --- what it is -----------------------------------------------------------

    @property
    def input_ids(self) -> InputBuffer:
        return self._by_name["input_ids"]

    @property
    def positions(self) -> InputBuffer:
        return self._by_name["positions"]

    @property
    def write_slots(self) -> InputBuffer:
        return self._by_name["write_slots"]

    @property
    def context_lens(self) -> InputBuffer:
        return self._by_name["context_lens"]

    @property
    def device(self) -> torch.device:
        return self.buffers[0].device

    @property
    def cells(self) -> int:
        return sum(b.cells for b in self.buffers)

    @property
    def bytes(self) -> int:
        return sum(b.bytes for b in self.buffers)

    @property
    def writes(self) -> int:
        return sum(b.writes for b in self.buffers)

    @property
    def written_cells(self) -> int:
        return sum(b.written_cells for b in self.buffers)

    @property
    def moves(self) -> int:
        return sum(b.moves for b in self.buffers)

    @property
    def addresses(self) -> tuple[int, ...]:
        """Where all four live. Take this before a run and compare it after."""
        return tuple(b.address for b in self.buffers)

    def owns(self, tensor: torch.Tensor) -> bool:
        return any(b.owns(tensor) for b in self.buffers)

    def render(self) -> str:
        """One line for a log: the shape, the price, and how the run has used it."""
        return (
            f"{self.max_batch_size} rows x {len(self.buffers)} inputs = {self.cells} "
            f"cells, {self.bytes} bytes, {self.writes} writes, "
            f"{self.written_cells} cells written, {self.moves} moves"
        )

    # --- using it -------------------------------------------------------------

    def to(self, device) -> DecodeInputs:
        for buffer in self.buffers:
            buffer.to(device)
        return self

    def set_input_ids(self, values, window: int | None = None) -> torch.Tensor:
        return self.input_ids.write(values, window=window)

    def set_positions(self, values, window: int | None = None) -> torch.Tensor:
        return self.positions.write(values, window=window)

    def set_write_slots(self, values, window: int | None = None) -> torch.Tensor:
        return self.write_slots.write(values, window=window)

    def set_context_lens(self, values, window: int | None = None) -> torch.Tensor:
        return self.context_lens.write(values, window=window)


# --- what it costs ------------------------------------------------------------------


def input_cells(max_batch_size: int, buffers: Sequence = DECODE_INPUTS) -> int:
    """Entries in one buffer set. Reserved at construction, all of them."""
    if max_batch_size < 1:
        raise ValueError(f"a buffer set holds at least one row; got {max_batch_size}")
    return max_batch_size * sum(
        _row_cells(tail) for _, tail in buffers
    )


def input_bytes(
    max_batch_size: int,
    itemsize: int = INPUT_ITEMSIZE,
    buffers: Sequence = DECODE_INPUTS,
) -> int:
    """What one buffer set weighs. int64, because these are indices and token ids."""
    if itemsize < 1:
        raise ValueError(f"an entry is at least one byte; got {itemsize}")
    return input_cells(max_batch_size, buffers) * itemsize


def fresh_cells(steps: int, rows: int, buffers: Sequence = DECODE_INPUTS) -> int:
    """Cells the per-step build wrote before there were buffers to write into.

    The same number a persistent set writes, which is the point: the day does not
    make the step write less, it makes the step write into the same place twice.
    What goes away is the allocation, and `fresh_allocations` is that half.
    """
    if steps < 0:
        raise ValueError(f"a run has a non-negative number of steps; got {steps}")
    if rows < 1:
        raise ValueError(f"a step covers at least one row; got {rows}")
    return steps * rows * sum(_row_cells(tail) for _, tail in buffers)


def fresh_allocations(steps: int, buffers: Sequence = DECODE_INPUTS) -> int:
    """Tensors the old path allocated over a run: one per input per step.

    Small in bytes and not small in consequence. Every one of them is at a new
    address, and an address that changes is the one thing a replayed graph cannot
    survive, so this count is the number that has to reach zero rather than the
    number that has to get small.
    """
    if steps < 0:
        raise ValueError(f"a run has a non-negative number of steps; got {steps}")
    return steps * len(buffers)


def per_shape_bytes(shapes: Sequence, itemsize: int = INPUT_ITEMSIZE) -> int:
    """What a buffer set for every captured shape would cost. The road not taken.

    Sized per shape's row count, since that is the window each graph would record
    against. The number is here to be compared with `input_bytes` and to make the
    point that neither of them is where a capture's memory goes.
    """
    if itemsize < 1:
        raise ValueError(f"an entry is at least one byte; got {itemsize}")
    return sum(input_bytes(int(s.rows), itemsize) for s in shapes)


def sharing_ratio(
    shapes: Sequence, max_batch_size: int, itemsize: int = INPUT_ITEMSIZE
) -> float:
    """How many times over a set per shape would pay for what one set covers."""
    shared = input_bytes(max_batch_size, itemsize)
    return per_shape_bytes(shapes, itemsize) / shared


def _row_cells(tail: Sequence[int]) -> int:
    cells = 1
    for size in tail:
        cells *= int(size)
    return cells


# --- gates --------------------------------------------------------------------------


def check_inputs_fit(rows: int, inputs: DecodeInputs) -> None:
    """Refuse a batch the buffers have no window for."""
    if rows > inputs.max_batch_size:
        raise InputsUnsound(
            f"a step of {rows} rows against buffers sized for "
            f"{inputs.max_batch_size}: the buffers are sized from the cache's row "
            "count, so this batch is bigger than the cache it came from"
        )


def check_one_set_covers(shapes: Sequence, inputs: DecodeInputs) -> None:
    """Refuse a capture list a single buffer set cannot serve.

    The day's second question, as a gate. A graph is recorded per shape and a buffer
    is indexed by batch position, so one set covers every shape whose row count fits
    in it, and the largest row bucket is the only size that matters. When this
    passes, `count` graphs share four buffers; when it fails, the bucket set was
    built against a different cache than the buffers were.
    """
    for shape in shapes:
        if int(shape.rows) > inputs.max_batch_size:
            raise InputsUnsound(
                f"a captured shape of {int(shape.rows)} rows against buffers sized "
                f"for {inputs.max_batch_size}: one set covers a capture list only "
                "when every shape's window is a prefix of it"
            )


def check_addresses_stable(inputs: DecodeInputs, before: Sequence[int]) -> None:
    """Refuse a run that replaced a buffer instead of writing into one.

    The gate of the day, and the only one whose subject is an address rather than a
    value. A replayed graph re-runs kernels bound to the pointers it recorded, so a
    buffer that was reallocated between the capture and the replay is a graph
    reading storage that now belongs to somebody else. Nothing raises when that
    happens: the kernels run, the shapes are right, and the numbers are whatever was
    there.
    """
    before = tuple(int(a) for a in before)
    if len(before) != len(inputs.buffers):
        raise ValueError(
            f"this set has {len(inputs.buffers)} buffers and {len(before)} addresses "
            "were recorded"
        )
    for buffer, was in zip(inputs.buffers, before):
        if buffer.address != was:
            raise InputsUnsound(
                f"{buffer.name} moved from {was:#x} to {buffer.address:#x}: a graph "
                "recorded against the old address would replay over storage that is "
                f"no longer this buffer ({buffer.moves} moves)"
            )


def check_plan_inputs_persistent(plan, inputs: DecodeInputs) -> None:
    """Refuse a plan whose addressing is its own allocation rather than a window.

    A capture gate and not a correctness one, in the same sense as Day 51's
    `check_mapping_is_window`: a freshly built `positions` computes exactly the same
    attention. It cannot be replayed, which is a different complaint.
    """
    for name in ("positions", "write_slots", "context_lens"):
        tensor = getattr(plan, name)
        if not inputs.owns(tensor):
            raise InputsUnsound(
                f"this plan's {name} is its own allocation and not a window on the "
                "persistent buffers: a graph captured over it would replay against "
                "an address the next step does not write"
            )


def check_step_inputs_persistent(input_ids: torch.Tensor, plan, inputs: DecodeInputs) -> None:
    """The same check over everything a decode forward is actually handed.

    `input_ids` is the one the plan does not own, because the token comes from the
    sampler and not from the cache, and it is the one most likely to be a view on
    somebody else's tensor: Day 47's fast path returned exactly that.
    """
    if not inputs.owns(input_ids):
        raise InputsUnsound(
            "this step's input_ids is not a window on the persistent buffers: last "
            "step's sampled tokens are a fresh allocation at a fresh address, so "
            "they have to be copied in rather than pointed at"
        )
    check_plan_inputs_persistent(plan, inputs)


def check_snapshots_present(plan, inputs: DecodeInputs) -> None:
    """Refuse a plan that shares storage and kept no copy of what it said.

    The correctness gate of the day. Two gates written before this one used the
    plan's own tensors as witnesses: Day 50's `check_plan_current` compares its
    lengths against the cache's tables, and Day 51's `check_window_intact` compares
    its write slots against its rectangle. Both were valid because those tensors
    were copies. Over a shared buffer they are not copies, they are storage the next
    step writes through, so both gates would agree with everything forever. A plan
    built this way carries host-side snapshots instead, and this is the line that
    says it has to.

    The second half checks the snapshot against its own tensor, which is only
    interesting for a plan somebody assembled by hand: taken at build time off the
    list the tensor was written from, they cannot disagree.
    """
    shared = inputs.owns(plan.context_lens) or inputs.owns(plan.write_slots)
    if not shared:
        return
    if not plan.context_snapshot or not plan.write_snapshot:
        raise InputsUnsound(
            "this plan's lengths and slots live in buffers the next step overwrites "
            "and it carries no snapshot of them: the staleness gate and the window "
            "gate would both pass forever, because their witness now tracks the "
            "thing it was supposed to be checked against"
        )
    for name, snapshot, tensor in (
        ("context_snapshot", plan.context_snapshot, plan.context_lens),
        ("write_snapshot", plan.write_snapshot, plan.write_slots),
    ):
        if list(snapshot) != tensor.tolist():
            raise InputsUnsound(
                f"this plan's {name} says {list(snapshot)} and its tensor says "
                f"{tensor.tolist()}: a snapshot is a copy taken at build time and "
                "one that disagrees with the buffer it was copied from was built by "
                "hand"
            )
