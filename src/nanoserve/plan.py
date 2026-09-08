"""The decode step's addressing, worked out on the host before the forward. Day 50.

Day 49 put the decode forward behind `torch.compile`, removed twelve graph breaks,
got one captured graph with no breaks in it, and measured a forty-six-fold
regression. The breaks were gone and the *guard* was not, and the guard that kept
failing was not on a shape:

    kwargs['cache'].cache.tables[0].num_tokens == 13
      # [table.slot(p) for p in range(start, table.num_tokens)], cache.py:729

`num_tokens` is a plain Python int on a plain Python object, read inside the traced
region. A tracer cannot make an attribute of an arbitrary object symbolic, so it
bakes the value in as a constant and guards on it. That integer grows by one every
decode step, for the whole run, so the graph was invalidated every step, rebuilt
every step, and after eight rebuilds abandoned to the interpreter for the rest of
the process with nothing raised and nothing logged.

`dynamic=True` does not touch this. It makes *tensor dimensions* symbolic, and the
thing being specialised here is not a dimension.

**So the addressing moves out of the forward.** A `DecodePlan` is everything the
decode pass needs to know about where things live, computed once on the host before
the call and handed in as tensors:

  - `write_slots`  `[rows]`      where this step's K/V goes, one flat pool slot a row
  - `slot_mapping` `[rows, ctx]` the read rectangle, oldest token first, padded with 0
  - `context_lens` `[rows]`      how much of each row is real
  - `positions`    `[rows, 1]`   the absolute position of each row's new token

With one of these in hand the forward reads no Python attribute of the cache at
all. `write` is a single scatter into the plan's slots, `slot_mapping` hands back
the tensors it was given, and the read's `context_lens` validation was done on the
host by whoever built the plan. What is left for a tracer to guard on is tensor
shapes, which is exactly the thing `dynamic=True` was built for.

**The bounds are the trap repeating itself, one level down.** Day 49's fix was to
stop reading `int(context_lens.min())` in the kernel and hand the two numbers down
from the cache, where they were already Python ints. Handing those same ints
*into* a traced region is the identical specialisation in different clothing:
`min_ctx` and `max_ctx` change every step, so a guard on their values rebuilds
every step. That is why the plan's bounds are checked when the plan is built and
`paged_attention_batched_reference` takes `validated=True` rather than a pair of
integers. A bool that is True for the whole run is a guard that holds for the whole
run.

**Three consequences fall out that are not about compilation at all.**

*The forward becomes a function again.* A forward over a KV cache normally is not
one: the paged read writes this step's K/V into the row before it attends, so
calling it twice attends over a history one token longer the second time and
returns different logits. Day 49's first equality test failed by whole units for
exactly that reason. With the addressing fixed before the call, replaying the same
plan writes the same slots and reads the same rectangle, which is what makes a
compiled forward comparable to an eager one and what a replayed CUDA graph is going
to require outright.

*The per-row Python loop in the write disappears.* Rows used to write different
counts to different places, so the scatter was a loop. A decode step is one token
per row by definition, so a plan's write is one `index_put` over the whole batch.

*And the plan is not free.* The read rectangle is `[rows, max_ctx]` int64 rebuilt
from scratch on the host every step, and `max_ctx` grows by one per step, so a
512-step generation rebuilds about 140,000 cells per row in order to append 512.
`rebuild_cells` against `appended_cells` is that ratio and `check_rebuild_bounded`
is the gate. The answer is a persistent block table that lives on the device and
takes one write per row per step, which is what vLLM keeps and what this module
prices rather than implements. Naming the cost the day the design incurs it is the
same order Day 49 used for bucketing: write the arithmetic first, and let the
measurement disagree with it later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: How many times over a run may the host rebuild the read rectangle, measured
#: against the cells an incremental table would have touched, before the rebuild is
#: the dominant cost of the step. Deliberately generous: at 4 rows and a 16-token
#: prompt this is passed by an 8-step run and failed by a 512-step one, which is
#: the range where the quadratic starts to be the whole bill.
DEFAULT_REBUILD_LIMIT = 32.0

#: Bytes per entry of the addressing tensors. `torch.long` is what an index tensor
#: has to be, and it is the reason the rectangle is expensive in bytes as well as in
#: cells: 8 bytes to say where one token lives.
SLOT_ITEMSIZE = 8


class PlanUnsound(AssertionError):
    """A decode plan does not describe the step it is about to be used for.

    An `AssertionError` for the same reason `CompileUnsound` is one: these are
    checks a test suite runs and a benchmark asserts, not conditions an engine
    recovers from. A plan that is wrong about where a token goes does not raise
    anywhere on its own; it writes into another sequence's slot and the run keeps
    producing plausible text.
    """


@dataclass(frozen=True)
class DecodePlan:
    """One decode step's addressing, fixed before the forward is called.

    rows:         the cache rows this step covers, in the order the batch presents
                  them. `write_slots[i]` belongs to `rows[i]`.
    positions:    `[rows, 1]` long, each row's new token's absolute position, which
                  is the number of tokens that preceded it. Read *before* the tables
                  grew, which is why the plan supplies it: computing it afterwards
                  is off by one, and computing it separately is a second host-side
                  build of the same list.
    write_slots:  `[rows]` long, the flat pool slot this step's K/V goes to.
    slot_mapping: `[rows, max_ctx]` long, the read rectangle. Row i's first
                  `context_lens[i]` entries are its own slots, oldest first; the
                  rest are slot 0, which is padding that really is dereferenced
                  (the reference gathers before it masks) and so has to be legal.
    context_lens: `[rows]` long, how many entries of a row are real. Includes this
                  step's own token: it is written before it is attended.
    pad_rows:     rows appended to the batch purely to keep the forward's shape on
                  a bucket. Day 52. They sit after the real ones, their
                  `context_lens` entry is 0 so they attend over nothing, and their
                  `write_slots` entry is `sink_slot`. Zero means the plan is the
                  Day-51 one and every tensor below is exactly `[rows, ...]`.
    sink_slot:    the pool slot a padded row's K/V is thrown at: one past the end of
                  the pool the allocator hands out, so it is a legal address that no
                  block maps to and no `BlockTable` can name. `None` when nothing is
                  padded. See `nanoserve.buckets` for why the write is not simply
                  sliced off instead.
    context_snapshot,
    write_snapshot:
                  host-side copies of `context_lens` and `write_slots` as they were
                  when the plan was built. Day 53, and empty unless the two tensors
                  live in buffers somebody else writes through. They exist because
                  a gate needs a witness that cannot move: `check_plan_current` and
                  `check_window_intact` both read a plan's own tensors, and that was
                  only sound while those tensors were copies. See `nanoserve.inputs`.
    min_ctx,
    max_ctx:      the bounds of `context_lens`, as Python ints, *for the host*.
                  Nothing inside a traced forward may read them; that is the whole
                  point of the plan and the reason the kernel takes `validated=True`
                  instead. They are here so the caller that built the plan can check
                  them, and so a log line can print them.

    Frozen, because by the time anyone else holds one it has already been handed to
    a forward, and addressing that changes underneath a call is the bug this class
    exists to remove.
    """

    rows: tuple[int, ...]
    positions: torch.Tensor
    write_slots: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    min_ctx: int
    max_ctx: int
    pad_rows: int = 0
    sink_slot: int | None = None
    context_snapshot: tuple[int, ...] = ()
    write_snapshot: tuple[int, ...] = ()

    @property
    def batch_size(self) -> int:
        """Cache rows in this step. Padding is not one of them."""
        return len(self.rows)

    @property
    def context_list(self) -> list[int]:
        """The lengths this plan was built with, as host ints. Day 53.

        The snapshot when there is one and the tensor otherwise, and every gate that
        asks a *held* plan what it used to say goes through here. The two sources
        agree at build time by construction; they stop agreeing the moment the
        tensor is a buffer somebody else writes through, and that is the only case
        where the snapshot exists.
        """
        return list(self.context_snapshot) if self.context_snapshot else self.context_lens.tolist()

    @property
    def write_list(self) -> list[int]:
        """The write slots this plan was built with, as host ints. Day 53."""
        return list(self.write_snapshot) if self.write_snapshot else self.write_slots.tolist()

    @property
    def graph_rows(self) -> int:
        """Rows the forward actually runs, padding included. Day 52.

        The dimension a shape guard sees, which is the one that has to stop moving.
        `batch_size` is the dimension the *scheduler* moved, and the whole point of
        padding is that the two are allowed to disagree.
        """
        return self.batch_size + self.pad_rows

    @property
    def graph_width(self) -> int:
        """The rectangle's context axis as the forward sees it, rounding included.

        Read off the tensor rather than recomputed, because this is the number a
        guard keys on and the tensor is the only thing that cannot be wrong about
        it. `max_ctx` is the longest real history and is what `context_lens` is
        checked against; past it the columns are padding the mask covers.
        """
        return int(self.slot_mapping.shape[1])

    @property
    def is_bucketed(self) -> bool:
        """Whether anything about this step's shape was rounded rather than measured."""
        return self.pad_rows > 0 or self.graph_width != self.max_ctx

    @property
    def width(self) -> int:
        """The rectangle's context axis: the longest history in the batch."""
        return self.max_ctx

    @property
    def device(self) -> torch.device:
        return self.slot_mapping.device

    @property
    def cells(self) -> int:
        """Entries the batch itself asked for: real rows, as wide as the longest one."""
        return self.batch_size * self.max_ctx

    @property
    def graph_cells(self) -> int:
        """Entries the kernel actually reads, after both axes have been rounded up."""
        return self.graph_rows * self.graph_width

    @property
    def pad_cells(self) -> int:
        """Cells that exist only to keep the shape on a bucket. The price of Day 52."""
        return self.graph_cells - self.cells

    @property
    def pad_share(self) -> float:
        """What fraction of the computed rectangle bucketing added."""
        return self.pad_cells / self.graph_cells if self.graph_cells else 0.0

    @property
    def real_cells(self) -> int:
        """Entries that address a token. Everything else is slot 0."""
        return int(self.context_lens.sum())

    @property
    def padding_cells(self) -> int:
        return self.cells - self.real_cells

    @property
    def padding_share(self) -> float:
        """What fraction of the rectangle exists only because the batch is ragged.

        The same waste Day 49 priced for bucketing, arriving here for a different
        reason: the rectangle is as wide as the longest row, so a batch of one long
        sequence and three short ones computes four long rows and masks three.
        """
        return self.padding_cells / self.cells if self.cells else 0.0

    @property
    def mapping_bytes(self) -> int:
        """What the read rectangle costs to hold, in bytes. The padded one, since
        that is the tensor the forward is handed."""
        return mapping_bytes(self.graph_rows, self.graph_width)

    def render(self) -> str:
        """One line for a log: shape, occupancy, and price."""
        line = (
            f"{self.batch_size} rows x {self.max_ctx} wide = {self.cells} cells, "
            f"{self.real_cells} real ({self.padding_share:.0%} padding), "
            f"{self.mapping_bytes} bytes, ctx {self.min_ctx}..{self.max_ctx}"
        )
        if self.is_bucketed:
            line += (
                f", bucketed to {self.graph_rows} x {self.graph_width} "
                f"(+{self.pad_cells} cells, {self.pad_share:.0%})"
            )
        return line


# --- building one ---------------------------------------------------------------


def plan_decode(cache, rows=None, device=None, buckets=None) -> DecodePlan:
    """Grow every named row by one token and hand back this step's addressing.

    `cache` is a `BatchedPagedKVCache` or one of its row views; a view supplies its
    own rows unless it is overridden. The work itself lives on the cache, because
    growing the tables is the cache's business and doing it from here would mean
    reaching into three of its attributes from another module. What lives here is
    the type, the arithmetic and the gates.

    This is the mutating call of the step, and it is the *only* one: the tables grow
    exactly once, here, atomically across the batch, before anything is traced.

    `buckets` is Day 52: a `DecodeBuckets` that rounds this step's two dimensions up
    so the forward's shape stays in a closed set. `None` falls back to whatever the
    cache was built with, which is `None` again unless somebody asked for it.
    """
    inner = getattr(cache, "cache", None)
    if inner is not None:
        rows = cache.rows if rows is None else rows
        cache = inner
    return cache.plan_decode(rows=rows, device=device, buckets=buckets)


# --- what it costs ----------------------------------------------------------------


def mapping_cells(rows: int, max_ctx: int) -> int:
    """Entries in a `[rows, max_ctx]` read rectangle."""
    if rows < 1 or max_ctx < 1:
        raise ValueError(f"a rectangle has at least one row and column; got {rows}x{max_ctx}")
    return rows * max_ctx


def mapping_bytes(rows: int, max_ctx: int, itemsize: int = SLOT_ITEMSIZE) -> int:
    """What that rectangle weighs. Index tensors are int64, so a slot is 8 bytes."""
    if itemsize < 1:
        raise ValueError(f"an entry is at least one byte; got {itemsize}")
    return mapping_cells(rows, max_ctx) * itemsize


def rebuild_cells(steps: int, rows: int, start_ctx: int) -> int:
    """Cells the host writes over a run that rebuilds the rectangle every step.

    Step i (1-based) builds `rows * (start_ctx + i)` of them, because the context
    grew by one on every step before it. Summing gives
    `rows * (steps * start_ctx + steps * (steps + 1) / 2)`, which is quadratic in
    the length of the generation. This is the honest price of the design in this
    module: correct, traceable, and doing O(n^2) host work to append n tokens.
    """
    if steps < 0:
        raise ValueError(f"a run has a non-negative number of steps; got {steps}")
    if rows < 1:
        raise ValueError(f"a step covers at least one row; got {rows}")
    if start_ctx < 1:
        raise ValueError(f"a decode row starts with a prefill; got start_ctx={start_ctx}")
    return rows * (steps * start_ctx + steps * (steps + 1) // 2)


def appended_cells(steps: int, rows: int) -> int:
    """Cells a persistent device-side table would touch: one per row per step.

    The alternative this module does not implement. A block table that lives on the
    device and is written in place needs exactly one new entry per row per step, so
    the host work is linear and, once the table is on the device, is not host work
    at all. That is what vLLM keeps, and it is the reason its slot mapping does not
    show up in a profile the way this one will.
    """
    if steps < 0:
        raise ValueError(f"a run has a non-negative number of steps; got {steps}")
    if rows < 1:
        raise ValueError(f"a step covers at least one row; got {rows}")
    return steps * rows


def rebuild_ratio(steps: int, rows: int, start_ctx: int) -> float:
    """How many cells this design writes for every cell it actually needed to."""
    if steps < 1:
        raise ValueError(f"a ratio needs at least one step; got {steps}")
    return rebuild_cells(steps, rows, start_ctx) / appended_cells(steps, rows)


# --- gates -------------------------------------------------------------------------


def check_plan_addressing(plan: DecodePlan) -> None:
    """Refuse a plan whose two halves disagree about where this step's token is.

    Two claims, and both of them are silent when they break. The write slot has to
    be the row's last real entry in the read rectangle, or the step writes its token
    somewhere the same step's read will not look at. And no two rows may name one
    slot, which is the invariant a shared pool lives on: a collision is one sequence
    reading another's K/V, and it produces fluent, wrong text.
    """
    if tuple(plan.write_slots.shape) != (plan.graph_rows,):
        raise PlanUnsound(
            f"a plan has one write slot per row ({plan.graph_rows}); got "
            f"{tuple(plan.write_slots.shape)}"
        )
    if tuple(plan.slot_mapping.shape) != (plan.graph_rows, plan.graph_width):
        raise PlanUnsound(
            f"the read rectangle is [graph_rows, graph_width] = "
            f"{(plan.graph_rows, plan.graph_width)}; got {tuple(plan.slot_mapping.shape)}"
        )
    lens = plan.context_lens.tolist()
    if len(lens) != plan.graph_rows:
        raise PlanUnsound(
            f"a plan has one context length per row ({plan.graph_rows}); got {len(lens)}"
        )
    for i, n in enumerate(lens[plan.batch_size :], start=plan.batch_size):
        if n != 0:
            raise PlanUnsound(
                f"padded row {i} claims {n} cached tokens: a padded row is there to "
                "hold the shape still and must attend over nothing"
            )
    for i, n in enumerate(lens[: plan.batch_size]):
        if n < 1:
            raise PlanUnsound(
                f"row {plan.rows[i]} claims {n} cached tokens: a decode query has at "
                "least its own token to attend to"
            )
        last = int(plan.slot_mapping[i, n - 1])
        if last != int(plan.write_slots[i]):
            raise PlanUnsound(
                f"row {plan.rows[i]} writes to slot {int(plan.write_slots[i])} but its "
                f"write slot in the read rectangle is {last}: the step would store this "
                "token where the same step's read does not look"
            )
    live = [int(s) for i, n in enumerate(lens[: plan.batch_size]) for s in plan.slot_mapping[i, :n]]
    if len(set(live)) != len(live):
        raise PlanUnsound(
            f"{len(live) - len(set(live))} of {len(live)} addressed slots are named by "
            "more than one row: rows share a pool and must never share a slot"
        )


def check_plan_rows(plan: DecodePlan, rows) -> None:
    """Refuse a plan built for a different set of rows than the forward covers."""
    rows = tuple(int(r) for r in rows)
    if rows != plan.rows:
        raise PlanUnsound(
            f"this forward covers rows {list(rows)} and the plan addresses "
            f"{list(plan.rows)}: the plan is what says where each row's token goes"
        )


def check_plan_current(plan: DecodePlan, cache) -> None:
    """Refuse a plan the cache has moved on from.

    A plan is addressing for the state it was built in, and `plan_decode` is what
    grows the tables, so planning twice and using the first plan writes the second
    step's token over the first step's slot. Nothing raises when that happens: the
    tables are consistent, the pool is legal, and one token is simply gone.

    Day 53 changes where the lengths come from and not what is compared. Over a
    persistent input buffer the plan's `context_lens` is storage the next step
    writes through, so it would always agree with the tables and this gate would
    stop being able to fail; `context_list` reads the plan's own snapshot instead.
    """
    inner = getattr(cache, "cache", None)
    cache = inner if inner is not None else cache
    now = [cache.tables[r].num_tokens for r in plan.rows]
    want = plan.context_list[: plan.batch_size]
    if now != want:
        raise PlanUnsound(
            f"this plan is stale: it was built when rows {list(plan.rows)} held "
            f"{want} tokens and they now hold {now}"
        )


def check_rebuild_bounded(
    steps: int, rows: int, start_ctx: int, *, limit: float = DEFAULT_REBUILD_LIMIT
) -> None:
    """Refuse a run long enough that rebuilding the rectangle is the step.

    The gate this day ends on, and it is a gate against this module's own design.
    Rebuilding is quadratic and appending is linear, so there is always a run length
    past which the addressing costs more than the attention it addresses. Saying
    where that is, in the module that causes it, is cheaper than finding it in a
    profile six days later.
    """
    if limit <= 0:
        raise ValueError(f"a rebuild limit is a positive multiple; got {limit}")
    ratio = rebuild_ratio(steps, rows, start_ctx)
    if ratio > limit:
        raise PlanUnsound(
            f"{steps} steps over {rows} rows from a {start_ctx}-token context rebuilds "
            f"{rebuild_cells(steps, rows, start_ctx)} cells to append "
            f"{appended_cells(steps, rows)}, a rebuild ratio of {ratio:.0f}x against a "
            f"limit of {limit:.0f}x: the host is spending more on saying where the "
            "tokens are than on the tokens"
        )
