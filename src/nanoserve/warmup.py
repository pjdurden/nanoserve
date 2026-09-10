"""The capture list recorded before the server accepts anything. Week 13, Day 55.

Day 54 got a decode step into a graph and did it in the wrong place. The first
sighting of every shape takes a recording, and a recording happens *mid-run*, in
front of a client who is waiting for a token. It is not a one-off cost either:
`lazy_captures` below is the arithmetic, and it says a run takes one recording per
`width_multiple` tokens it generates, per row bucket it lands in, for as long as it
runs. Day 54's own benchmark caught it and I read it as a bug for a minute: a
16-token prompt and 128 generated tokens recorded two graphs, because 144 crosses
the 128-token width bucket.

vLLM does not do this. It walks its capture list at startup, off dummy batches, and
the first real decode of the process is already a replay. This is that day.

**The hard part is what a dummy batch *is*.** A recorded graph is bound to the
addresses it was launched with, so a warm batch cannot be built somewhere quiet and
thrown away: it has to be driven through the real slot table, the real input
buffers and the real K/V pool, or the replay is bound to storage no real step
writes. Which means the fake batch is about to write K/V into a pool full of real
sequences, before anything has been served, using rows the scheduler has not handed
out.

**And Day 52 already answered it.** A *padded* row exists precisely to hold a shape
still while touching nothing: its `context_lens` entry is 0 so the whole row is
masked, and its `write_slots` entry is `sink_slot`, one past the pool the allocator
hands out, so its K/V lands at a legal address no `BlockTable` can name. A warm
batch is a step whose rows are **all** padding: `rows=()`, `pad_rows=graph_rows`,
every write at the sink. There is no sequence, so there is nothing to corrupt, and
the plan is not a special case of a real one, it is the limit of one.

**Every gate Day 54 spends passes on it unchanged**, which is the second day in a
row that has been the reward for writing them as functions. `check_pad_inert`'s
third clause is "no *real* row writes to the sink" and is vacuous when there are no
real rows. `check_mapping_is_window` wants the rows to be a prefix of the table's,
and the empty tuple is a prefix of everything. `check_plan_addressing` iterates the
real rows and finds none.

**A warm graph is unbound, and that is strictly safer than a warm one.** Day 54
ended with `check_replay_rows`, the only gate on this arc that guards a field
instead of a pointer: a mid-run capture freezes step 1's `plan.rows` tuple and
holds it for the process, so a replay over a different set of sequences would write
this step's tokens into the other batch's slots. A capture recorded on a warm batch
freezes the *empty* tuple. There is no set of sequences it can be wrong about.
Moving the recording to startup deletes the one thing the held plan carried that
was not storage, and I did not expect the two halves of this week to meet.

**What a replay actually reads.** Everything else the recorded call holds is a
window on storage a real step writes through: `slot_mapping` is Day 51's table,
`positions`, `write_slots`, `context_lens` and `input_ids` are Day 53's buffers. So
a graph recorded over a batch of nothing replays over a batch of something and
computes the right answer, and `test_a_graph_recorded_on_nothing_computes_a_real_step`
is that claim as an assertion rather than as this paragraph.

**One hole showed up while doing it.** Day 54 replays behind `check_addresses_stable`,
which covers the four input buffers, and nothing covers the slot table those
rectangles are windows on. It is the larger of the two by three orders of magnitude
(16 MB against 8 KB at 256 rows) and it has a `to()` that reallocates. Warming is
what makes the hole reachable, because warming is now the first thing in the
process to hand out a window, and it is therefore the first thing that can hand one
out on the wrong device. `check_table_stable` is the missing line.

**And the budget has a number at last.** Day 54's `check_pool_budget` wanted "what
is left after the weights and the K/V pool" and there was nobody to ask.
`warm_budget_bytes` asks `nanoserve.launch`'s probe, after both are resident, and
`width_ceiling` turns the answer into the only thing a shared pool's budget is
actually about: how wide the widest capture in the list may be.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .cache import BatchedCacheRows
from .captured import (
    ACTIVATION_ITEMSIZE,
    CapturedDecode,
    check_capture_preconditions,
    check_capture_ready,
    check_pool_budget,
    score_cells,
)
from .compiled import DecodeShape
from .plan import DecodePlan

#: The token id a synthetic row forwards. Any id in the vocabulary would do: the
#: row's logits are discarded and its K/V goes to the sink. Zero is the one that
#: cannot be mistaken for a real sample while reading a log, which is the same
#: reason Day 52 puts padded rows at position 0.
DEFAULT_WARM_TOKEN = 0


class WarmupUnsound(AssertionError):
    """A warm-up did not do what warming up is for.

    An `AssertionError` for the same reason `CaptureUnsound`, `InputsUnsound`,
    `BucketsUnsound` and `SlotsUnsound` are: these are checks on the engine's own
    claims. Every failure here is quiet. A warm-up that grew a row has spent a real
    sequence's block on a token nobody asked for; a warm-up that missed a shape
    leaves a recording in the decode loop where a client will pay for it; a warm-up
    that recorded against a table that then moved leaves a replay reading storage
    that is no longer the table.
    """


# --- one warm batch -----------------------------------------------------------------


@dataclass(frozen=True)
class WarmupBatch:
    """The arguments of one synthetic decode step, and nothing behind them.

    shape:     the dimensions this batch presents. It comes off the bucket set, so
               the graph recorded over it is keyed on a shape real steps will land
               on exactly.
    input_ids: `[shape.rows, 1]`, a window on the persistent buffer. The tokens are
               made up; the *address* is the whole point.
    plan:      the synthetic plan. `rows` is empty and `pad_rows` is the whole
               batch, which is what makes this batch own nothing.
    view:      the cache wearing no rows at all. Built here rather than through
               `cache.view`, which refuses an empty selection on purpose: a
               *scheduled* view over no rows is a forward with nothing in it, and
               this is the one caller that means it.
    """

    shape: DecodeShape
    input_ids: torch.Tensor
    plan: DecodePlan
    view: BatchedCacheRows

    @property
    def rows(self) -> int:
        """Real cache rows in this batch. Zero, by construction and by design."""
        return self.plan.batch_size

    def render(self) -> str:
        return (
            f"{self.shape.rows} x {self.shape.context_width} over {self.rows} real "
            f"rows, {self.plan.pad_rows} at sink slot {self.plan.sink_slot}"
        )


@dataclass(frozen=True)
class WarmupReport:
    """What one walk of the capture list did, as a value a boot line can print.

    shapes:   the list that was walked, in the order it was walked.
    captures: graphs this walk recorded. Zero on a second walk, which is what makes
              warming idempotent rather than merely harmless.
    graphs:   graphs the capture holds afterwards, warm and otherwise.
    seconds:  wall time of the walk. The number a startup budget is spent against,
              and the number that is *not* in anybody's inter-token latency, which
              is the entire trade the day makes.
    cold:     shapes that were asked for and are still not recorded. Empty unless
              something fell through to eager, and a warm-up whose whole job was to
              leave nothing cold should say so rather than be assumed.
    """

    shapes: tuple[DecodeShape, ...]
    captures: int
    graphs: int
    seconds: float
    cold: tuple[DecodeShape, ...] = ()

    @property
    def count(self) -> int:
        return len(self.shapes)

    @property
    def per_capture_s(self) -> float:
        """Seconds a recording costs, measured rather than assumed. Feeds
        `stall_seconds`, which is what the same recording would have cost a client
        had it happened mid-run."""
        if not self.captures:
            return 0.0
        return self.seconds / self.captures

    def render(self) -> str:
        return (
            f"{self.captures} of {self.count} shapes recorded in {self.seconds:.3f}s "
            f"({self.per_capture_s * 1e3:.1f} ms a graph), {self.graphs} graphs held, "
            f"{len(self.cold)} cold"
        )


# --- building one -------------------------------------------------------------------


def warm_shapes(
    buckets, *, max_rows: int | None = None, max_width: int | None = None
) -> tuple[DecodeShape, ...]:
    """The bucket set as a capture list, biggest first.

    The order is not cosmetic. Every capture after the first is handed the pool the
    first one made (Day 54's `check_pool_shared`), and an arena is sized by what is
    allocated out of it. Record the largest shape first and every later graph fits
    inside an arena that is already the right size; record ascending and the arena
    is grown once per shape on the way up.

    `max_rows` and `max_width` trim the list to what a server will actually present.
    A cache sized for 256 rows that is deployed behind a scheduler admitting 8 has
    no reason to hold 248 rows' worth of graphs, and the width axis is where the
    product gets expensive: the list is `len(rows) * len(widths)` and only one of
    those two grows with the context length.
    """
    shapes = [
        shape
        for shape in buckets.shapes
        if (max_rows is None or shape.rows <= max_rows)
        and (max_width is None or shape.context_width <= max_width)
    ]
    if not shapes:
        raise WarmupUnsound(
            f"no shape in this bucket set is within max_rows={max_rows} and "
            f"max_width={max_width}: the set rounds *up*, so a ceiling below the "
            "smallest bucket excludes every shape a step could present"
        )
    return tuple(sorted(shapes, key=lambda s: (s.cells, s.rows, s.context_width), reverse=True))


def warmup_plan(cache, shape: DecodeShape, *, device=None) -> DecodePlan:
    """One shape's addressing, over no sequences at all.

    Every field is the limit of what `BatchedPagedKVCache.plan_decode` builds when
    the batch is empty and the whole shape is padding, and it is written here rather
    than reached by passing `rows=()` there because `plan_decode` is the *mutating*
    call of a decode step: it reserves blocks, grows every table by one and appends
    to the slot table. A warm-up must do none of those, and a flag threaded through
    that method to skip all three would be a second function wearing the first one's
    name.

    What it does share is every buffer. The rectangle is a window on the real slot
    table, the three addressing tensors are written into the real input buffers, and
    the write slots all name the real sink. That is the requirement, not a shortcut:
    a graph recorded against storage a real step does not write is a graph that
    replays yesterday's numbers forever.
    """
    inner = getattr(cache, "cache", cache)
    check_capture_ready(inner)
    if shape.query_len != 1:
        raise WarmupUnsound(
            f"a warm batch is a decode step: one token per row, got query_len="
            f"{shape.query_len}. A prefill is a different shape and a different path"
        )
    table = inner.slot_table
    rows, width = int(shape.rows), int(shape.context_width)
    if rows > table.max_batch_size:
        raise WarmupUnsound(
            f"a warm batch of {rows} rows is wider than the cache's "
            f"{table.max_batch_size}: there is no row to pad into past the last one"
        )
    if width > table.max_model_len:
        raise WarmupUnsound(
            f"a warm rectangle of {width} columns is wider than max_model_len "
            f"{table.max_model_len}: a rectangle wider than the table it is a window "
            "on is not a window"
        )
    # Both moves happen before anything is read, and both are the reason the day
    # needed `check_table_stable`: this is the first call in the process to hand out
    # a window, so it is the first that can hand one out on the wrong device.
    table.to(device)
    inputs = inner.decode_inputs
    inputs.to(device)

    sink = inner.sink_slot
    # `rows=()` and `pad_rows=rows`: the read is `slots[:rows, :width]`, a basic
    # slice and therefore the buffer's own storage, and the lengths come back as
    # `rows` zeros written into the persistent buffer.
    slot_mapping, context_lens = table.read(
        (), width, pad_rows=rows, lengths_writer=inputs.set_context_lens
    )
    positions = inputs.set_positions([0] * rows)
    write_slots = inputs.set_write_slots([sink] * rows)
    return DecodePlan(
        rows=(),
        positions=positions,
        write_slots=write_slots,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        # The bounds of an empty batch. Nothing inside the forward reads them, and
        # the two host-side gates that do (`check_plan_current`, `check_pad_inert`)
        # iterate the real rows, of which there are none.
        min_ctx=0,
        max_ctx=0,
        pad_rows=rows,
        sink_slot=sink,
        context_snapshot=(0,) * rows,
        write_snapshot=(sink,) * rows,
    )


def warmup_batch(
    cache, shape: DecodeShape, *, device=None, token: int = DEFAULT_WARM_TOKEN
) -> WarmupBatch:
    """A whole synthetic step: the plan, the tokens, and the view that owns nothing."""
    inner = getattr(cache, "cache", cache)
    plan = warmup_plan(inner, shape, device=device)
    input_ids = inner.decode_inputs.set_input_ids([token] * int(shape.rows))
    return WarmupBatch(
        shape=shape,
        input_ids=input_ids,
        plan=plan,
        # Straight to the class, because `cache.view(())` is refused and should stay
        # refused: an empty *scheduled* view is a forward the scheduler built with
        # nothing in it. This is the one caller that means the empty tuple.
        view=BatchedCacheRows(inner, (), plan),
    )


def warm_decode(
    captured: CapturedDecode,
    cache,
    shapes: Sequence[DecodeShape] | None = None,
    *,
    device=None,
    token: int = DEFAULT_WARM_TOKEN,
    clock=None,
    precheck: bool = True,
) -> WarmupReport:
    """Record every shape in the list, off synthetic batches, before anything serves.

    The loop is four lines and the checks around it are the day. Each shape gets a
    batch that names no rows, the batch goes through Day 54's preconditions exactly
    as a real step would, and the capture records it because it is that shape's first
    sighting. Afterwards the cache is asked whether anything moved, and the answer
    has to be no: a warm-up that grew a row has spent a sequence's block, and a
    warm-up that moved the slot table has invalidated every window it just recorded.

    `clock` is injected so a test can price a walk without one, the same seam the
    benchmarks use.
    """
    if captured.mode == "off":
        raise ValueError(
            "this capture is off, so warming it walks the whole bucket set through "
            "the forward and records nothing: build the engine with capture_decode "
            "before warming it"
        )
    inner = getattr(cache, "cache", cache)
    check_capture_ready(inner)
    shapes = tuple(warm_shapes(inner.decode_buckets) if shapes is None else shapes)
    clock = time.perf_counter if clock is None else clock

    before_lens = tuple(inner.seq_lens)
    before_free = inner.allocator.num_free
    inner.slot_table.to(device)
    table_address = inner.slot_table.address
    started = captured.captures

    elapsed = clock()
    for shape in shapes:
        batch = warmup_batch(inner, shape, device=device, token=token)
        check_warm_rows_empty(batch.plan)
        check_warm_writes_sink(batch.plan)
        if precheck:
            check_capture_preconditions(batch.input_ids, batch.plan, batch.view)
        captured(batch.input_ids, batch.plan.positions, cache=batch.view)
    elapsed = clock() - elapsed

    check_warm_touches_nothing(inner, before_lens, free_blocks=before_free)
    check_table_stable(inner.slot_table, table_address)
    return WarmupReport(
        shapes=shapes,
        captures=captured.captures - started,
        graphs=captured.count,
        seconds=elapsed,
        cold=tuple(s for s in shapes if s not in captured.graphs),
    )


# --- what it costs, and what not doing it costs -------------------------------------


def lazy_captures(*, steps: int, start_width: int, width_multiple: int) -> int:
    """Recordings a run takes *inside* the decode loop when nothing was warmed.

    One per width bucket the run crosses, which is the reading of Day 54's benchmark
    that took a minute to see: a 16-token prompt generating 128 tokens reaches a
    context of 144, crosses the 128-token boundary, and legitimately presents two
    shapes. `width_multiple` is not only a padding knob, it is a *rate*: one capture
    per that many tokens of generation, for the whole life of the request.

    Counted over one row bucket. A run that also gains or loses enough rows to change
    row bucket multiplies this, which is the other axis and the reason the capture
    list is a product.
    """
    if steps < 1:
        raise ValueError(f"a run is at least one step; got {steps}")
    if start_width < 0:
        raise ValueError(f"a context starts at zero tokens or more; got {start_width}")
    if width_multiple < 1:
        raise ValueError(f"a width multiple is at least one; got {width_multiple}")
    first = -(-(start_width + 1) // width_multiple)
    last = -(-(start_width + steps) // width_multiple)
    return last - first + 1


def stall_seconds(
    *, steps: int, start_width: int, width_multiple: int, per_capture_s: float
) -> float:
    """What those recordings cost the client they happen in front of.

    The honest framing of the day. This is not throughput lost, it is latency landed
    on whichever token happened to be the one that crossed a bucket boundary, so it
    shows up in a p99 inter-token number and in nothing else. Warming moves the same
    seconds to startup, where nobody is waiting on them.
    """
    if per_capture_s < 0:
        raise ValueError(f"a recording does not take negative time; got {per_capture_s}")
    return (
        lazy_captures(steps=steps, start_width=start_width, width_multiple=width_multiple)
        * per_capture_s
    )


def warmup_seconds(count: int, per_capture_s: float) -> float:
    """What warming the whole list costs at startup.

    The other side of the trade, and it is a worse number by construction: warming
    pays for every shape in the closed set, and a lazy run pays only for the ones it
    reaches. What it buys is that the payment is not in a request's latency, and
    that the number is knowable before the door opens rather than discovered in a
    tail.
    """
    if count < 0:
        raise ValueError(f"a capture list holds no negative number of shapes; got {count}")
    if per_capture_s < 0:
        raise ValueError(f"a recording does not take negative time; got {per_capture_s}")
    return count * per_capture_s


def warm_budget_bytes(
    device, *, reserved_bytes: int = 0, utilization: float = 0.90, probe=None
) -> int:
    """Bytes left for the capture pool, once the weights and the K/V pool are resident.

    The connection Day 54 asked for and could not make: `check_pool_budget` wanted a
    number for what was left over and there was nobody to ask. This is
    `launch.kv_budget_bytes` one step later in the boot order, and "later" is the
    whole content of it. Call it *after* the pool is allocated and `free` already has
    the weights, the CUDA context, the allocator's fragmentation and the K/V pool
    subtracted by the driver, so none of them need modelling here.

    `reserved_bytes` is what the caller still intends to allocate and the driver
    cannot know about yet. `utilization` is the same reserve against the device total
    that the KV budget takes, for the same reason: a co-tenant should eat into the
    budget and leave the safety margin the size it was chosen to be.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            f"cannot size a capture pool by probing a {device.type} device: there is "
            "no VRAM to divide. Pass an explicit budget to check_pool_budget instead"
        )
    if reserved_bytes < 0:
        raise ValueError(f"a reservation is not negative; got {reserved_bytes}")
    probe = torch.cuda.mem_get_info if probe is None else probe
    free, total = probe(device)
    return max(0, free - int(total * (1.0 - utilization)) - reserved_bytes)


def width_ceiling(
    *, rows: int, num_heads: int, budget_bytes: int, itemsize: int = ACTIVATION_ITEMSIZE
) -> int:
    """The widest context a shape of `rows` rows may be captured at, given a budget.

    The inverse of `captured.workspace_bytes`, and the reason it is the useful form:
    a shared pool is sized by its *largest* member, so a budget is not a statement
    about the length of the capture list at all. It is a statement about one number,
    and that number is a width. `warm_shapes(..., max_width=width_ceiling(...))` is
    the whole use.
    """
    if rows < 1 or num_heads < 1:
        raise ValueError(f"a shape has at least one row and one head; got {rows}x{num_heads}")
    if itemsize < 1:
        raise ValueError(f"an activation is at least one byte; got {itemsize}")
    if budget_bytes < 0:
        raise ValueError(f"a budget is not negative; got {budget_bytes}")
    return int(budget_bytes) // (rows * num_heads * itemsize)


# --- gates ---------------------------------------------------------------------------


def check_warm_rows_empty(plan: DecodePlan) -> None:
    """Refuse a warm batch that named a cache row.

    The definition of the day as a check. A plan with a real row in it is a plan that
    will write that row's K/V into the pool and grow nothing to match, so the
    sequence's own next read looks for a token in a slot its table never learned
    about. It is the same failure as writing to the wrong slot, arrived at from the
    other direction.
    """
    if plan.rows:
        raise WarmupUnsound(
            f"this warm plan names rows {list(plan.rows)}: a warm batch is every row "
            "padding, because a padded row is the only row in this engine that reads "
            "nothing and writes nowhere"
        )


def check_warm_writes_sink(plan: DecodePlan) -> None:
    """Refuse a warm batch whose K/V would land anywhere a block maps to.

    Day 52's `check_pad_inert` says this about a plan's *padded* rows, and on a warm
    plan every row is padded, so this is that gate with the qualifier removed. It is
    worth its own line because a warm batch is the one place a wrong answer here
    would corrupt a sequence that has not been admitted yet, in a process that has
    not served anything, which is as far from the failure as a log can get.
    """
    if plan.sink_slot is None:
        raise WarmupUnsound(
            "this warm plan names no sink slot: a warm row's made-up K/V has to go "
            "somewhere, and without a sink it goes into the pool"
        )
    for i, slot in enumerate(plan.write_list):
        if slot != plan.sink_slot:
            raise WarmupUnsound(
                f"warm row {i} writes to slot {slot} and the sink is "
                f"{plan.sink_slot}: a made-up token would land in the pool, in "
                "whatever sequence owns that slot once the server opens"
            )


def check_warm_touches_nothing(cache, before: Sequence[int], *, free_blocks: int = None) -> None:
    """Refuse a warm-up that moved the cache it warmed against.

    The claim the whole design rests on, checked afterwards rather than argued for.
    Nothing in a warm batch may grow a table (a warm row is a token the sequence did
    not generate) and nothing may take a block (a block spent on a fake row is a
    block a real request is refused for). `plan_decode` does both, which is exactly
    why `warmup_plan` does not call it.
    """
    inner = getattr(cache, "cache", cache)
    before = tuple(int(n) for n in before)
    now = tuple(inner.seq_lens)
    if now != before:
        grew = [i for i, (was, is_) in enumerate(zip(before, now)) if was != is_]
        raise WarmupUnsound(
            f"rows {grew} grew during a warm-up: {list(before)} became {list(now)}. A "
            "warm batch is all padding and must not reach a block table, so this "
            "walked the real planning path instead of the synthetic one"
        )
    if free_blocks is not None and inner.allocator.num_free != free_blocks:
        raise WarmupUnsound(
            f"the pool had {free_blocks} free blocks before this warm-up and "
            f"{inner.allocator.num_free} after: a block spent on a synthetic row is a "
            "block a real request gets refused for"
        )


def check_table_stable(table, address: int) -> None:
    """Refuse a slot table that moved between a recording and now.

    The hole Day 54 left, and it is the bigger of the two buffers by three orders of
    magnitude. `check_addresses_stable` guards the four `[max_batch]` input vectors,
    8 KB at 256 rows; the rectangles those graphs read are windows on
    `[max_batch, max_model_len]`, 16 MB at 8192 tokens, and nothing was watching it.
    `SlotTable.to` reallocates, and warming is the first thing in the process to hand
    out a window, so it is the first thing that can hand one out on a device the
    engine is about to leave.
    """
    address = int(address)
    if table.address != address:
        raise WarmupUnsound(
            f"the slot table moved from {address:#x} to {table.address:#x} "
            f"({table.moves} moves): every rectangle recorded before the move is a "
            "window on storage that is no longer this table, and a replay over one "
            "reads whatever is there now"
        )


def check_warm_graphs_unbound(captured: CapturedDecode) -> None:
    """Refuse a capture list holding a graph that was recorded over real sequences.

    The positive statement of what warming buys. Day 54's `check_replay_rows` is the
    only gate on this arc that guards a field rather than a pointer, and it exists
    because a mid-run capture freezes step 1's row tuple for the life of the process.
    A warm graph freezes the empty tuple, so the field it guards cannot say anything
    wrong. This is how you know every graph in the list came from the warm-up and not
    from a step that slipped past it.
    """
    bound = {shape: graph.rows for shape, graph in captured.graphs.items() if graph.rows}
    if bound:
        shape, rows = next(iter(bound.items()))
        raise WarmupUnsound(
            f"the graph for {shape.rows} x {shape.context_width} was recorded over "
            f"rows {list(rows)}, so it was recorded mid-run rather than warmed: it "
            "holds that batch's row tuple forever, and only a warm graph is free of "
            f"a row set ({len(bound)} of {captured.count} graphs are bound)"
        )


def check_no_cold_captures(captured: CapturedDecode, *, warmed: int) -> None:
    """Refuse a run that recorded a graph after the door opened.

    What the day is *for*, as one number. `warmed` is what the warm-up reported;
    anything past it is a recording that happened in front of a client, and it is
    silent: the tokens are right, the run is merely slower on exactly the step
    somebody was measuring.
    """
    if warmed < 0:
        raise ValueError(f"a warm-up records no negative number of graphs; got {warmed}")
    cold = captured.captures - warmed
    if cold > 0:
        raise WarmupUnsound(
            f"{cold} graph(s) were recorded after the list was warmed: a shape the "
            "warm-up did not cover takes its recording inside the decode loop, so a "
            "client waits for it. Widen the warm list, or find out why a step "
            "presented a shape outside the bucket set"
        )


def check_all_warm(captured: CapturedDecode, shapes: Sequence[DecodeShape]) -> None:
    """Refuse a warm-up that left a shape in the list without a graph."""
    cold = tuple(s for s in shapes if s not in captured.graphs)
    if cold:
        shape = cold[0]
        raise WarmupUnsound(
            f"{len(cold)} of {len(shapes)} warm shapes were never recorded, starting "
            f"with {shape.rows} x {shape.context_width}: a shape that fell through to "
            "eager during a warm-up will be recorded by the first real step that "
            "presents it"
        )


def check_warm_budget(
    shapes: Sequence[DecodeShape],
    num_heads: int,
    *,
    device,
    reserved_bytes: int = 0,
    utilization: float = 0.90,
    probe=None,
    itemsize: int = ACTIVATION_ITEMSIZE,
) -> None:
    """Day 54's pool budget, priced against what the driver says is actually free.

    Two functions that were written a day apart and could not be joined until there
    was a boot order to join them in. The list is sized by its widest shape and the
    device is asked what is left; `width_ceiling` is what to do with a refusal.
    """
    check_pool_budget(
        shapes,
        num_heads,
        budget_bytes=max(
            1,
            warm_budget_bytes(
                device, reserved_bytes=reserved_bytes, utilization=utilization, probe=probe
            ),
        ),
        itemsize=itemsize,
    )


def widest(shapes: Sequence[DecodeShape], num_heads: int) -> DecodeShape:
    """The shape a shared pool is sized by. A max, not a sum, which is Day 54's point."""
    if not shapes:
        raise ValueError("a capture list holds at least one shape")
    return max(shapes, key=lambda s: score_cells(s, num_heads))
