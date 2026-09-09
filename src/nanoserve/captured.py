"""The decode step recorded once and replayed. Week 13, Day 54.

Five days have been spent making a decode step something a capture can hold, and
every one of them ended in a gate written as a function rather than as a comment.
Day 49 removed the graph breaks. Day 50 moved the addressing out of the forward, so
the only thing left to guard on is a tensor shape. Day 51 made the read rectangle a
window on a persistent table, so its address stopped moving. Day 52 rounded the
shape into a closed bucket set, so the shape stopped moving. Day 53 put the four
inputs into buffers that do not move either. This is the day that spends all of it:
those gates are the *preconditions* a capture checks before it records anything.

**What a recorded graph is, in one sentence.** A list of kernel launches bound to
the addresses they were launched with. `replay()` takes no arguments: it re-runs
those kernels over whatever those addresses now hold, and writes the result into
whatever address the output was at. That is the whole model, and every design
decision below falls out of it.

**Which means the held plan is the interesting object, not the held tensors.** The
recorded call keeps a `DecodePlan` and a cache view from the step it was recorded
on, and it keeps them forever. Step 500 replays with step 1's plan. That is sound
for exactly one reason: everything the forward reads off a plan is a window on
persistent storage, so the Python object is frozen and the numbers in it are not.
`write_slots`, `context_lens` and `positions` are Day 53's buffers; `slot_mapping`
is Day 51's table. `rows` is a Python tuple, and it is the one field that is not
storage, which is why `check_replay_rows` exists.

**The gates split into two kinds and the split is a performance decision.** A gate
that reads a tensor's *values* (`check_pad_inert`, `check_window_intact`) has to
bring them to the host, which on a device is a synchronisation: exactly the thing
Day 47 and Day 48 spent two days removing. So the value gates run once, at capture,
where a sync is already unavoidable. The gates that run on every replay are the ones
that read an *address* or a Python tuple, and neither of those touches the device:
`check_addresses_stable` and `check_replay_rows`.

**The output is a fixed buffer too, and that half is the one nobody warns you
about.** Every argument to `check_step_inputs_persistent` is about the input side,
because that is where the loud failure is. The mirror image is quieter: a replay
writes into the same storage every time, so a captured step's result has a lifetime
of exactly one step. Anything holding last step's logits is holding this step's.
`check_output_not_held` is that written down, and Day 48's deferred window is what
it is aimed at: that window keeps token tensors across step boundaries, and it
survives a capture only because the sampler runs *outside* the recorded region. Move
the sampling inside and a window of N steps becomes N views on one buffer, all of
them holding the newest token.

**One memory pool, and the arithmetic said something I did not expect.** Every
capture in this process is handed the same pool handle, which is what vLLM does and
for the obvious reason: a private pool per graph would multiply the workspace by the
size of the capture list. The number is not the multiple it looks like. A bucket set
is geometric on both axes, so the largest shape is most of the bill, and sharing a
pool over the 36-shape list Day 52 ends on saves about **5x**, not 36x. The number
that actually hurts is the absolute one. `workspace_bytes` prices the
`[rows, heads, 1, ctx]` score rectangle that `paged_attention_batched_reference`
materialises before it masks, and at 256 rows and an 8192-token width that single
intermediate is 268 MB. That is not the capture's fault and no pool sharing touches
it: it is the reference read, and it is the same reason Day 52 has to bucket the
width axis at all when vLLM's capture list is over batch sizes only.

**And on CPU none of this is fast.** `eager_recorder` records nothing. It reproduces
a capture's *semantics* (fixed input addresses, one fixed output buffer, a replay
that takes no arguments) and none of its speed, which is what makes every
correctness property here testable on a box with no device. The real recorder is
`cuda_graph_recorder`, it is four lines, and the four lines are warmup on a side
stream, `torch.cuda.CUDAGraph()`, the pool handle, and `graph.replay`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from .buckets import check_pad_inert, check_shape_in_set
from .compiled import DecodeShape, check_single_graph, decode_shape
from .inputs import (
    check_addresses_stable,
    check_snapshots_present,
    check_step_inputs_persistent,
)
from .slots import check_mapping_is_window, check_window_intact

#: Calls into the forward before anything is recorded. Not decoration: the first
#: call into a kernel picks an algorithm and allocates a workspace, and recording
#: that records the allocation into the graph. vLLM warms up on a side stream for
#: the same reason, and three is its number too.
DEFAULT_WARMUP = 3

#: How many graphs one process will record before it refuses. A capture list is
#: supposed to be a list; a run that keeps finding new shapes has lost the closed
#: set Day 52 bought and is spending device memory to find that out slowly.
DEFAULT_CAPTURE_LIMIT = 64

#: Bytes per entry of the activations a captured region holds live. fp32, which is
#: what this engine runs on CPU; halve it for a bf16 device and the conclusion does
#: not change, because the conclusion is about a ratio and an order of magnitude.
ACTIVATION_ITEMSIZE = 4

#: The modes `CapturedDecode` accepts. Two, and "off" exists so the engine has one
#: call site rather than a branch, the same way Day 49's "off" does.
MODES = ("off", "capture")


class CaptureUnsound(AssertionError):
    """A recorded step is not the fixed, self-describing thing a replay needs.

    An `AssertionError` for the same reason `CompileUnsound`, `PlanUnsound`,
    `SlotsUnsound`, `BucketsUnsound` and `InputsUnsound` are: these are checks on
    the engine's own claims. Every failure in this module is silent on its own. A
    replay over a moved buffer runs the right kernels over the wrong memory, a run
    that falls through to eager is a run that is merely slower than it promised, and
    a held output is a value that changed while somebody was reading it.
    """


# --- one recorded shape ---------------------------------------------------------------


@dataclass
class CapturedGraph:
    """One shape, recorded, with everything a replay of it needs to be checked.

    shape:           the dimensions this graph was recorded at. A replay is only
                     legal for a step that presents exactly these, which is what
                     Day 52's bucket set exists to make possible.
    replay:          `() -> None`. Takes no arguments, because there is nothing to
                     pass: the kernels are bound to the addresses they were recorded
                     with. `torch.cuda.CUDAGraph.replay` is this exactly.
    output:          the tensor those kernels write into, every time. Fixed storage,
                     which is the half of the day that surprises people. See
                     `check_output_not_held`.
    input_addresses: where the persistent input buffers were when this was recorded.
                     Compared on every replay, because it is the one check that
                     catches a reallocation and costs nothing to run.
    rows:            the cache rows the recorded plan covers, as a Python tuple. The
                     only thing carried across a replay that is not storage, which
                     makes it the only thing that can silently be about a different
                     set of sequences. See `check_replay_rows`.
    pool:            the memory pool this graph's workspace came out of, shared with
                     every other graph in the process.
    """

    shape: DecodeShape
    replay: Callable[[], None]
    output: torch.Tensor
    input_addresses: tuple[int, ...] = ()
    rows: tuple[int, ...] = ()
    pool: object = None
    replays: int = 0
    recorded_output_address: int = field(default=0)

    def __post_init__(self) -> None:
        self.input_addresses = tuple(int(a) for a in self.input_addresses)
        self.rows = tuple(int(r) for r in self.rows)
        if not self.recorded_output_address:
            self.recorded_output_address = self.output_address

    @property
    def output_address(self) -> int:
        """Where the result lands. The same number on every replay, by construction."""
        return self.output.untyped_storage().data_ptr()

    @property
    def output_bytes(self) -> int:
        return self.output.numel() * self.output.element_size()

    def owns(self, tensor: torch.Tensor) -> bool:
        """Whether `tensor` is this graph's output storage rather than a copy of it."""
        return tensor.untyped_storage().data_ptr() == self.output_address

    def run(self) -> torch.Tensor:
        """Replay, and hand back the buffer the replay just wrote into."""
        self.replay()
        self.replays += 1
        return self.output

    def render(self) -> str:
        return (
            f"{self.shape.rows} x {self.shape.context_width} over rows "
            f"{list(self.rows)}, {self.output_bytes} bytes out at "
            f"{self.output_address:#x}, {self.replays} replays"
        )


# --- recording one --------------------------------------------------------------------


class EagerReplay:
    """A replay with a capture's semantics and none of its speed. The CPU stand-in.

    It records nothing. What it reproduces is the three properties that make a
    capture hard to use correctly, and they are all about storage rather than about
    kernels: the call keeps the exact argument objects it was recorded with, so it
    reads whatever those now hold; it takes no arguments; and it writes into one
    output buffer allocated at record time.

    That is enough to make every gate in this module testable on a box with no
    device, and it is enough to make the day's failure modes *happen* rather than be
    described: rebind an input instead of writing into it and the replay quietly
    computes the previous step's answer, exactly as a real graph would.
    """

    def __init__(self, fn: Callable, args, kwargs, *, warmup: int = DEFAULT_WARMUP):
        if warmup < 0:
            raise ValueError(f"a warmup count is not negative; got {warmup}")
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.warmups = warmup
        self.replays = 0
        for _ in range(warmup):
            self.fn(*self.args, **self.kwargs)
        # The recorded call. Its result is copied into storage this object owns for
        # the rest of the process, which is what `torch.cuda.graph` does with the
        # tensor its recorded kernels write to.
        self.output = self.fn(*self.args, **self.kwargs).detach().clone()

    def __call__(self) -> None:
        self.output.copy_(self.fn(*self.args, **self.kwargs))
        self.replays += 1


class MemoryPool:
    """A stand-in for `torch.cuda.graph_pool_handle()`: an identity and nothing else.

    A pool handle is not a buffer. It is a token that says "allocate this capture's
    workspace out of the same arena as the last one", and the only thing any of this
    module does with it is pass it to the next capture and check that every graph
    got the same one. So the stand-in is an object, and that is the whole type.
    """

    __slots__ = ()


def eager_recorder(
    fn: Callable, args, kwargs, *, pool=None, warmup: int = DEFAULT_WARMUP
) -> tuple[Callable[[], None], torch.Tensor, object]:
    """Record nothing, hand back the same three things a real capture does."""
    replay = EagerReplay(fn, args, kwargs, warmup=warmup)
    return replay, replay.output, pool if pool is not None else MemoryPool()


def cuda_graph_recorder(
    fn: Callable, args, kwargs, *, pool=None, warmup: int = DEFAULT_WARMUP
) -> tuple[Callable[[], None], torch.Tensor, object]:
    """The real one: warm up on a side stream, then record into a shared pool.

    The side stream is not a flourish. Warmup allocations made on the capturing
    stream can end up inside the graph, and the point of warming up is that the
    kernel's own lazy setup happens somewhere the recording will not see it. The
    pool is the other half: the first capture creates one and every later capture is
    handed it back, so the whole list shares an arena instead of holding a private
    workspace each. See `check_pool_shared` and `pool_sharing_ratio`.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        output = fn(*args, **kwargs)
    return graph.replay, output, graph.pool()


def default_recorder(
    fn: Callable, args, kwargs, *, pool=None, warmup: int = DEFAULT_WARMUP
) -> tuple[Callable[[], None], torch.Tensor, object]:
    """`cuda_graph_recorder` on a device, `eager_recorder` anywhere else.

    Dispatching on the tensors rather than making the caller choose, because the
    caller is an engine that already knows where its weights are and should not have
    to know what a capture is. What it must not do is pretend: on CPU the returned
    thing is a stand-in with a capture's semantics and no speedup at all, and nothing
    in this repo reports a number that says otherwise.
    """
    on_cuda = any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)
    record = cuda_graph_recorder if on_cuda else eager_recorder
    return record(fn, args, kwargs, pool=pool, warmup=warmup)


# --- the wrapper ------------------------------------------------------------------------


class CapturedDecode:
    """The decode forward, recorded per shape and replayed after that.

    fn:       what to record. In this engine it is Day 49's `CompiledDecode`, so a
              capture wraps a compile rather than replacing it: dynamo produces the
              kernels and the graph records the launches.
    mode:     "off" hands the call straight through, which is how the engine keeps
              one call site. "capture" records the first sighting of each shape and
              replays every sighting after it.
    recorder: `(fn, args, kwargs, pool, warmup) -> (replay, output, pool)`. The seam,
              for the same reason `CompiledDecode` has one: the whole of the policy
              here is testable against something that builds nothing.
    warmup:   calls before the recorded one. See `DEFAULT_WARMUP`.
    limit:    graphs this process will record before refusing.
    precheck: whether to run the five days' gates before recording. On by default and
              off in exactly one place: a test that wants to *measure* what an
              unbucketed run would cost in graphs, which is a thing the gates exist
              to refuse.

    A step with no plan on its cache view falls through to eager and is counted
    rather than refused. That is a prefill, or a caller who did not plan, and there
    is nothing fixed about an unplanned step's addressing to record.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        mode: str = "off",
        recorder: Callable | None = None,
        warmup: int = DEFAULT_WARMUP,
        limit: int = DEFAULT_CAPTURE_LIMIT,
        precheck: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"capture mode is one of {MODES}; got {mode!r}")
        if limit < 1:
            raise ValueError(f"a capture limit is at least one graph; got {limit}")
        self.fn = fn
        self.mode = mode
        self.warmup = warmup
        self.limit = limit
        self.precheck = precheck
        self.recorder = recorder if recorder is not None else default_recorder
        self.graphs: dict[DecodeShape, CapturedGraph] = {}
        self.pool = None
        self.calls = 0
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0

    # --- running it -----------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Replay this step's shape, recording it first if this is its first sighting.

        The recording branch ends in a replay rather than in the recorded call's own
        result. On a real capture the values produced *while* recording are not a
        result anybody should read, and doing it this way means the CPU stand-in and
        the device agree about what a capture step returns: one forward's worth of
        logits, out of the graph's output buffer.
        """
        self.calls += 1
        if self.mode == "off":
            return self.fn(*args, **kwargs)
        cache = kwargs.get("cache")
        plan = getattr(cache, "plan", None)
        if plan is None or not args:
            self.eager_calls += 1
            return self.fn(*args, **kwargs)
        input_ids = args[0]
        shape = decode_shape(input_ids, cache)
        graph = self.graphs.get(shape)
        if graph is None:
            graph = self._record(shape, input_ids, plan, cache, args, kwargs)
        else:
            # The two gates cheap enough to run every step. Both read a number the
            # host already has: a storage pointer and a tuple of row indices.
            # Everything else the five days wrote reads tensor *values*, which on a
            # device is a synchronisation, so those ran once at capture.
            inputs = _decode_inputs(cache)
            if inputs is not None:
                check_addresses_stable(inputs, graph.input_addresses)
            check_replay_rows(graph, plan)
        self.replays += 1
        return graph.run()

    def _record(self, shape, input_ids, plan, cache, args, kwargs) -> CapturedGraph:
        """Check the five days' preconditions, then record this shape once."""
        if self.precheck:
            check_capture_preconditions(input_ids, plan, cache)
        if len(self.graphs) >= self.limit:
            raise CaptureUnsound(
                f"this run has recorded {len(self.graphs)} graphs against a limit of "
                f"{self.limit} and has found shape {shape.rows} x "
                f"{shape.context_width}, which is a new one: a capture list that "
                "keeps growing is a shape set that was never closed"
            )
        replay, output, pool = self.recorder(
            self.fn, args, kwargs, pool=self.pool, warmup=self.warmup
        )
        self.pool = pool
        inputs = _decode_inputs(cache)
        graph = CapturedGraph(
            shape=shape,
            replay=replay,
            output=output,
            input_addresses=inputs.addresses if inputs is not None else (),
            rows=plan.rows,
            pool=pool,
        )
        self.graphs[shape] = graph
        self.captures += 1
        return graph

    # --- what it did ----------------------------------------------------------

    @property
    def shapes(self) -> tuple[DecodeShape, ...]:
        """The recorded shapes, in the order they were first seen."""
        return tuple(self.graphs)

    @property
    def count(self) -> int:
        """Graphs held. What the pool is paying for."""
        return len(self.graphs)

    @property
    def reuse(self) -> float:
        """Share of calls that landed on a graph that already existed."""
        if not self.calls:
            return 0.0
        return (self.calls - self.captures) / self.calls

    @property
    def output_bytes(self) -> int:
        """What the recorded outputs weigh. One buffer a shape, held for the run."""
        return sum(g.output_bytes for g in self.graphs.values())

    def owns_output(self, tensor: torch.Tensor) -> bool:
        """Whether `tensor` is some graph's output storage. A one-step value."""
        return any(g.owns(tensor) for g in self.graphs.values())

    def render(self) -> str:
        return (
            f"{self.count} graphs over {self.calls} calls ({self.captures} captures, "
            f"{self.replays} replays, {self.eager_calls} eager, {self.reuse:.0%} "
            f"reuse), {self.output_bytes} bytes of output buffers"
        )


def _decode_inputs(cache):
    """The persistent input buffers behind a cache or a row view of one."""
    inner = getattr(cache, "cache", cache)
    return getattr(inner, "decode_inputs", None)


# --- what it costs --------------------------------------------------------------------


def score_cells(shape: DecodeShape, num_heads: int) -> int:
    """Entries in the `[rows, heads, query, ctx]` score rectangle of one layer.

    The intermediate `paged_attention_batched_reference` materialises before it
    masks, and the largest live tensor a captured decode region holds. It is
    proportional to `rows * context_width`, which is why bucketing the width axis
    costs memory as well as arithmetic, and it is the thing vLLM's paged kernel never
    builds: it walks the block table and accumulates, so its capture list is over
    batch sizes only and its pool is smaller by the width.
    """
    if num_heads < 1:
        raise ValueError(f"a forward has at least one head; got {num_heads}")
    return shape.rows * num_heads * shape.query_len * shape.context_width


def workspace_bytes(
    shape: DecodeShape, num_heads: int, itemsize: int = ACTIVATION_ITEMSIZE
) -> int:
    """What that rectangle weighs, which is what one captured shape reserves."""
    if itemsize < 1:
        raise ValueError(f"an entry is at least one byte; got {itemsize}")
    return score_cells(shape, num_heads) * itemsize


def shared_pool_bytes(
    shapes: Sequence[DecodeShape], num_heads: int, itemsize: int = ACTIVATION_ITEMSIZE
) -> int:
    """One pool, sized by the largest shape in the list. The `graph_pool_handle` bill.

    A max and not a sum, because that is what sharing an arena means: the graphs are
    replayed one at a time, so a workspace big enough for the biggest of them is big
    enough for all of them.
    """
    return max((workspace_bytes(s, num_heads, itemsize) for s in shapes), default=0)


def private_pool_bytes(
    shapes: Sequence[DecodeShape], num_heads: int, itemsize: int = ACTIVATION_ITEMSIZE
) -> int:
    """A pool per graph. The road not taken, and it is what you get by default."""
    return sum(workspace_bytes(s, num_heads, itemsize) for s in shapes)


def pool_sharing_ratio(
    shapes: Sequence[DecodeShape], num_heads: int, itemsize: int = ACTIVATION_ITEMSIZE
) -> float:
    """How many times over private pools would pay for what one shared pool covers.

    The number this day expected to be the length of the capture list and is not. A
    bucket set is geometric on the row axis and arithmetic on the width one, so the
    sum is dominated by its largest term: 36 shapes share a pool for about 5x, not
    36x. Sharing is still right and the reason is not the multiple.
    """
    shared = shared_pool_bytes(shapes, num_heads, itemsize)
    if not shared:
        return 1.0
    return private_pool_bytes(shapes, num_heads, itemsize) / shared


def capture_cost_s(count: int, per_capture_s: float) -> float:
    """Seconds spent recording. Linear in the graphs, paid once at startup.

    vLLM does this in `capture_model` before the server accepts a request, which is
    the only sensible place for it: a capture in the middle of a run is a step that
    took a hundred times as long as its neighbours for no reason the client can see.
    """
    if count < 0:
        raise ValueError(f"a graph count is not negative; got {count}")
    if per_capture_s < 0:
        raise ValueError(f"a per-capture cost is not negative; got {per_capture_s}")
    return count * per_capture_s


def launch_saving_s(kernels: int, per_launch_s: float) -> float:
    """Seconds a step stops spending on launches: one submission where there were n.

    What a replay actually buys, and it is a host-side saving rather than a device
    one. The kernels are the same kernels and they take the same time; what goes away
    is the CPU walking the graph and submitting each of them, which is why the win
    shows up on small batches and short kernels and vanishes on large ones.
    """
    if kernels < 1:
        raise ValueError(f"a step launches at least one kernel; got {kernels}")
    if per_launch_s < 0:
        raise ValueError(f"a per-launch cost is not negative; got {per_launch_s}")
    return (kernels - 1) * per_launch_s


def capture_breakeven_steps(capture_s: float, saving_per_step_s: float) -> float:
    """Steps before the recording has paid for itself. Infinite when it never does."""
    if saving_per_step_s <= 0:
        return math.inf
    return capture_s / saving_per_step_s


# --- gates ------------------------------------------------------------------------------


def check_capture_ready(cache) -> None:
    """Refuse a cache that has not had the last two days switched on.

    The first precondition and the cheapest, and it names the constructor argument
    rather than the symptom, because the fix is one keyword at the other end of the
    program.
    """
    inner = getattr(cache, "cache", cache)
    if getattr(inner, "decode_buckets", None) is None:
        raise CaptureUnsound(
            "this cache was built without bucket_decode, so its decode shape is "
            "whatever the batch and the longest history happened to be: a graph is "
            "recorded per shape, and an open shape set is a capture list with no end"
        )
    if getattr(inner, "decode_inputs", None) is None:
        raise CaptureUnsound(
            "this cache was built without persist_inputs, so its decode step builds "
            "fresh input tensors rather than writing into buffers a replay can read: "
            "a graph recorded over them would replay against an address nobody writes"
        )


def check_capture_preconditions(input_ids, plan, cache, *, report=None) -> None:
    """Every gate the last five days wrote, run in order, before anything is recorded.

    This is the payoff of having written them as functions. Nothing here is new: it
    is Day 52's shape set and inert padding, Day 51's window and its intactness, Day
    53's persistent inputs and their snapshots, and optionally Day 49's single graph
    when a `CompileReport` is supplied.

    The exception that comes out is the one the failing day raises, not a
    `CaptureUnsound` wrapping it. That is deliberate: "the rectangle is a gather"
    should say `SlotsUnsound` and point at `nanoserve.slots`, because that is the
    module whose promise broke and the module where the fix is.
    """
    inner = getattr(cache, "cache", cache)
    check_capture_ready(inner)
    check_shape_in_set(
        DecodeShape(
            rows=plan.graph_rows, context_width=plan.graph_width, query_len=1
        ),
        inner.decode_buckets,
    )
    check_pad_inert(plan)
    check_mapping_is_window(inner.slot_table, plan)
    check_window_intact(plan)
    check_step_inputs_persistent(input_ids, plan, inner.decode_inputs)
    check_snapshots_present(plan, inner.decode_inputs)
    if report is not None:
        check_single_graph(report)


def check_replay_rows(graph: CapturedGraph, plan) -> None:
    """Refuse a replay for a different set of sequences than it was recorded over.

    The one thing a graph carries across a replay that is not storage. Everything
    else in the held plan is a window and therefore says whatever this step wrote;
    `rows` is a Python tuple and says what step 1 said forever. Day 51 already
    requires the rows to be a prefix of the table's, which makes a mismatch rare, and
    "rare" is the reason to check it in a line rather than argue about it in a
    comment.
    """
    rows = tuple(int(r) for r in plan.rows)
    if graph.rows and rows != graph.rows:
        raise CaptureUnsound(
            f"this graph was recorded over rows {list(graph.rows)} and this step "
            f"covers {list(rows)}: the recorded plan names its rows in Python, so a "
            "replay would write this step's tokens into the other batch's slots"
        )


def check_all_shapes_captured(captured: CapturedDecode) -> None:
    """Refuse a run that quietly went eager for some of its steps.

    The failure with no symptom, again, and it is the third time it has turned up in
    this week: Day 49's dynamo fallback, Day 52's shape outside the set, and now a
    step whose view carried no plan. The process keeps producing the right tokens and
    the capture bought nothing for those steps.
    """
    if captured.eager_calls:
        raise CaptureUnsound(
            f"{captured.eager_calls} of {captured.calls} decode calls ran eager "
            "rather than replaying a graph: a step whose cache view carries no plan "
            "has no fixed addressing to record, so the capture is not covering the "
            "whole loop"
        )


def check_replays_dominate(captured: CapturedDecode, *, min_reuse: float = 0.5) -> None:
    """Refuse a run that recorded about as often as it replayed.

    The same shape of gate as Day 49's `check_graph_reused` and for the same reason.
    A capture is worth having when one recording serves many steps; a run whose
    captures track its calls has paid the recording cost every step, holds a graph
    per step in the pool, and got the replay speed on none of them.
    """
    if captured.calls < 1:
        raise ValueError(f"a run has at least one call; got {captured.calls}")
    if captured.reuse < min_reuse:
        raise CaptureUnsound(
            f"{captured.captures} captures over {captured.calls} calls is "
            f"{captured.reuse:.0%} reuse, under the {min_reuse:.0%} this path is "
            "worth recording at: the shape set is still open, so every step records "
            "a graph and none of them get replayed"
        )


def check_pool_shared(captured: CapturedDecode) -> None:
    """Refuse a capture list whose graphs each hold a private workspace.

    One line of policy with a memory bill behind it. Every capture after the first is
    handed the pool the first one made, so the arena is sized by the largest shape
    instead of by all of them. `pool_sharing_ratio` is what the difference is worth,
    and it is smaller than the length of the list, which is the day's surprise.
    """
    pools = {id(g.pool) for g in captured.graphs.values()}
    if len(pools) > 1:
        raise CaptureUnsound(
            f"{captured.count} graphs are spread over {len(pools)} memory pools: a "
            "capture that is not handed the previous one's pool allocates its whole "
            "workspace privately, so the arena is the sum of the shapes rather than "
            "the largest of them"
        )


def check_pool_budget(
    shapes: Sequence[DecodeShape],
    num_heads: int,
    *,
    budget_bytes: int,
    itemsize: int = ACTIVATION_ITEMSIZE,
) -> None:
    """Refuse a capture list whose shared pool does not fit in what is left over.

    Priced off the largest shape, because that is what a shared pool costs, and it is
    still the number that hurts: this engine's read materialises a
    `[rows, heads, 1, ctx]` rectangle, so the top of a 256-row, 8192-wide bucket set
    is hundreds of megabytes of workspace that the weights and the K/V pool do not
    get to use.
    """
    if budget_bytes < 1:
        raise ValueError(f"a pool budget is positive; got {budget_bytes}")
    need = shared_pool_bytes(shapes, num_heads, itemsize)
    if need > budget_bytes:
        biggest = max(shapes, key=lambda s: score_cells(s, num_heads))
        raise CaptureUnsound(
            f"a shared capture pool for {len(shapes)} shapes needs {need} bytes, "
            f"sized by {biggest.rows} x {biggest.context_width}, against a budget of "
            f"{budget_bytes}: the width axis is what makes this large, and it is a "
            "property of the reference read rather than of the capture"
        )


def check_output_not_held(captured: CapturedDecode, tensor: torch.Tensor) -> None:
    """Refuse a value from a captured step that is about to outlive its step.

    The mirror image of every input gate on this page, and the quieter half. A replay
    writes into the storage it was recorded against, so a graph's output holds this
    step's answer until the next replay starts and then holds that one. Day 48's
    deferred window keeps tensors across step boundaries on purpose; it survives a
    capture only because the sampler runs outside the recorded region, so what it
    holds is the sampler's allocation. Move the sampling inside the graph and the
    window becomes N views of one buffer.
    """
    if captured.owns_output(tensor):
        raise CaptureUnsound(
            "this tensor is a captured graph's output buffer, which holds one step's "
            "worth of values: the next replay writes through it, so anything carried "
            "past the end of this step is reading the wrong step. Clone it, or "
            "consume it before the next call"
        )
