"""The decode forward behind a compiler, and the scissors hiding in it. Week 13, Day 49.

Day 46 profiled a step and found the host loop was almost all of it. Day 47 and Day
48 spent two days on the loop *around* the forward, moving one synchronisation and
then removing k-1 of every k. This day is the forward itself, and the tool is
`torch.compile`: trace the Python, fuse what fuses, emit one kernel where there were
twenty, and hand back a callable with the same signature.

Four things about that turn out to matter far more than the speedup, and this
module is mostly about them.

**Dynamo does not fail. It breaks.** When TorchDynamo meets a line it cannot trace,
it does not raise and it does not refuse: it ends the graph there, drops back into
the interpreter for that line, and starts a new graph after it. A function with
twelve breaks in it is still "compiled", still returns the right answer, and is
still made of thirteen fragments with Python and a device synchronisation between
each pair. It looks exactly like success. The only way to know is to ask, which is
`explain_forward`, and the only honest gate is `check_single_graph`.

This engine had twelve of them in a two-layer decode, and every one was its own
code: `paged_attention_batched_reference` validated `context_lens` with
`int(context_lens.min())` and `int(context_lens.max())` on every layer of every
step. Day 48 already tripped over those, as synchronisations, and scoped its
no-readback claim around them. Seen through a compiler they are worse than slow:
they are scissors. The fix is not to delete the validation but to move it to where
the numbers are already Python ints, which is the cache: `BlockTable.num_tokens` is
an int on the host, so `BatchedPagedKVCache.context_bounds` hands the bounds down
and the kernel never touches the device to check them. Thirteen graphs became one.

**What the graph is specialised on is not only its shapes, and that is the finding
that cost the most.** A decode call presents two moving dimensions: `[rows, 1]`
input ids, and a `[rows, max_ctx]` slot mapping whose width is the longest history
in the batch and therefore grows by one *every step, for the whole run*. That
alone is 100 distinct shapes over a 100-token generation, and `dynamic=True`
answers it: both dimensions go symbolic and the shape stops mattering.

It did not help, and the reason is the sentence to remember. The guard that fails
is not on a shape at all:

    kwargs['cache'].cache.tables[0].num_tokens == 13
      # [table.slot(p) for p in range(start, table.num_tokens)], cache.py:729

`num_tokens` is a plain Python int read off a plain Python object inside the
traced region, and dynamo specialises on its *value*, as a constant. `dynamic=True`
makes tensor dimensions symbolic and does nothing whatever for that. So the graph
is invalidated on every decode step, is rebuilt on every decode step, and after
`cache_size_limit` (8, by default) rebuilds dynamo stops compiling the frame and
runs eager for the rest of the process without raising. Measured on Llama-3.2-1B,
cpu fp32, 2 rows, 9 decode steps: 628ms a step eager against 28,983ms compiled,
with 8 builds over those 9 steps in *both* modes. A forty-six-fold regression, from
a guard on an integer.

The shape of the fix is the same as the readback fix above, and Day 50 is where it
got written: the host-side bookkeeping (`table.slot(p)` over a `range` that moves)
happens *outside* the traced region and arrives as a tensor. `nanoserve.plan` is
that, and with a `DecodePlan` in hand the guard that fails is a slot mapping's
*size* rather than an integer's value, which is the thing `dynamic=True` was built
for. Measured on the tiny two-layer model over 13 decode steps: 8 builds in 9 steps
before, 2 builds in 13 after, and the second of those two is the symbolic one that
then holds for the rest of the run. `check_graph_reused` is the gate that says so
and it now passes.

**Bucketing is the other answer, and it is the one a captured graph needs.**
Padding rows to the next power of two and the context width to the next multiple
turns an unbounded shape set into a closed one, at the price of cells the kernel
computes and nobody wants. `bucketed`, `bucket_padding_waste` and
`check_shapes_bucketed` are that trade written down. A compiler does not need it,
because symbolic shapes are free; a *CUDA graph* does, because a replayed capture
holds fixed pointers and fixed sizes, and vLLM's list of capture batch sizes is
exactly this. That is the next day, and the arithmetic is here first so the
measurement has something to disagree with.

And the whole thing is on credit. A compile is seconds and a saving is
microseconds a step, so `breakeven_steps` is in the thousands and a short run is
better off eager. `step_speedup` is Amdahl over the forward's share of the step,
which is the ceiling, and the ceiling is a fact about the machine rather than about
the change: Day 46's CSV puts `loop_share` at 0.0008 on this CPU box, so the model
*is* the step and there is nothing for Amdahl to subtract. On a GPU that profile
inverts and the same change is capped at `1/(1 - loop_share)`. Quoting a speedup
without saying which share it was measured against is quoting a property of a
laptop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

from .profiler import amdahl

#: Dynamo's default `torch._dynamo.config.cache_size_limit`. Distinct shapes past
#: this many and the frame is abandoned to the interpreter, without a raise and
#: without a log at default verbosity. It is a small number and a decode loop walks
#: past it in eight steps.
RECOMPILE_LIMIT = 8

#: Row counts a capture would be specialised on. Powers of two because the batch
#: is bounded by `max_batch_size` and doubling keeps the set logarithmic; this is
#: the same shape as vLLM's cudagraph capture sizes and for the same reason.
ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: Context width is rounded up to a multiple rather than to a power of two. Widths
#: run to thousands and doubling would pad a 600-token history to 1024; a multiple
#: keeps the overshoot bounded by the multiple itself, whatever the length.
WIDTH_MULTIPLE = 128

#: The modes `CompiledDecode` accepts. Three, and they are three different bets.
MODES = ("off", "dynamic", "static")


class CompileUnsound(AssertionError):
    """A compiled path that is not doing what its name says it is doing.

    An `AssertionError` for the same reason the other gate exceptions in this repo
    are: these are checks on the engine's own claims, and a failure means a
    benchmark is about to report a number that is not true.
    """


# --- what a decode step looks like to a shape guard --------------------------------


@dataclass(frozen=True)
class DecodeShape:
    """The dimensions a compiled decode graph is specialised on.

    rows:          how many sequences are in this forward. Changes when the
                   scheduler admits or reaps, which under continuous batching is
                   often but not every step.
    context_width: the width of the `[rows, max_ctx]` slot mapping, which is the
                   longest history in the batch. This one grows by one every step
                   for the whole run, so a graph specialised on it is rebuilt every
                   step. It is not the only thing that forces a rebuild here, and
                   it turned out not to be the binding one: see the module
                   docstring on the guard over `table.num_tokens`.
    query_len:     always 1. A decode step is one new token per row; a prefill is a
                   different shape and a different graph and goes down the dense
                   masked path. It is a field rather than an assumption so that the
                   refusal below has something to check.

    Frozen and hashable because the whole point is to put these in a set and count
    it: a compiler's cache is keyed on exactly this tuple, so the number of distinct
    values a run presents *is* the number of builds it asks for.
    """

    rows: int
    context_width: int
    query_len: int = 1

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError(f"a decode forward has at least one row; got {self.rows}")
        if self.context_width < 1:
            raise ValueError(
                f"a decode row attends over at least its own token; got a context "
                f"width of {self.context_width}"
            )
        if self.query_len != 1:
            raise ValueError(
                f"a decode step is one new token per row; got query_len="
                f"{self.query_len}. A ragged prefill is a different shape and a "
                "different graph"
            )

    @property
    def cells(self) -> int:
        """Entries in the slot mapping: what the reference read actually gathers."""
        return self.rows * self.context_width


def decode_shape(input_ids: torch.Tensor, cache) -> DecodeShape:
    """The shape a decode call would present to a compiler, off the real arguments.

    `input_ids` gives the rows and the query length; the cache view gives the
    context width, and it gives it from `seq_lens`, which is a Python list of ints
    off the block tables. No tensor is touched to work this out, which matters: a
    shape probe that synchronised would cost more than the recompile it is trying
    to predict.

    The width is the longest history *plus this step's token*, which is one more
    than `seq_lens` reports, because the read builds its mapping after the write:
    by the time `slot_mapping` is asked for a rectangle, the token being forwarded
    is already in the row. Predicting the shape a guard will see means counting it.

    Unless the view carries a Day-50 `DecodePlan`, in which case there is nothing to
    predict. The plan grew the tables and built the rectangle before this wrapper
    was called, so `seq_lens` is already the post-write length and adding one to it
    overshoots; the plan's dimensions are not an estimate of the guard's, they are
    the guard's. Which two they are is Day 52: `graph_rows` and `graph_width` are
    what the tensors handed to the forward really measure, and on a bucketed plan
    they are the rounded pair rather than the batch the scheduler admitted.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"decode input ids are [rows, seq]; got {tuple(input_ids.shape)}")
    plan = getattr(cache, "plan", None)
    if plan is not None:
        return DecodeShape(
            rows=plan.graph_rows,
            context_width=plan.graph_width,
            query_len=int(input_ids.shape[1]),
        )
    lens = list(cache.seq_lens)
    if not lens:
        raise ValueError("a decode forward over no rows has no shape")
    query_len = int(input_ids.shape[1])
    return DecodeShape(
        rows=int(input_ids.shape[0]),
        context_width=max(lens) + query_len,
        query_len=query_len,
    )


def shape_history(
    row_counts: Iterable[int], context_widths: Iterable[int]
) -> tuple[DecodeShape, ...]:
    """The shapes a run presented, step by step, from two columns of a benchmark.

    Kept separate from `decode_shape` because most of the reasoning about
    recompiles is about runs that have not happened yet: a sweep, a projection, an
    argument about whether a static graph could ever work here.
    """
    rows = list(row_counts)
    widths = list(context_widths)
    if len(rows) != len(widths):
        raise ValueError(
            f"a shape history has one width per step: got {len(rows)} row counts "
            f"and {len(widths)} widths"
        )
    return tuple(DecodeShape(r, w) for r, w in zip(rows, widths))


def distinct_shapes(shapes: Sequence[DecodeShape]) -> int:
    """How many different shapes a run presented, which is how many graphs it needs."""
    return len(set(shapes))


def recompiles(shapes: Sequence[DecodeShape]) -> int:
    """Builds past the first. The first one is a compile and is not free either.

    Off by one on purpose: the cost of running under a compiler is one build you
    always pay plus one per shape you did not think about, and the second term is
    the one that decides whether this is an optimisation or a regression.
    """
    return max(0, distinct_shapes(shapes) - 1)


def falls_back(shapes: Sequence[DecodeShape], limit: int = RECOMPILE_LIMIT) -> bool:
    """Whether dynamo would give up on this run and go back to the interpreter.

    The failure mode this predicts is the nastiest one in the module, because it
    has no symptom. Past `cache_size_limit` distinct shapes the frame is marked and
    runs eager forever after, so the process keeps working, keeps returning the
    right tokens, and has quietly paid for every compile it did before giving up.
    """
    return distinct_shapes(shapes) > limit


# --- bucketing: turning an open shape set into a closed one -------------------------


def bucket_for(value: int, buckets: Sequence[int] = ROW_BUCKETS) -> int:
    """The smallest bucket that holds `value`. Padding, expressed as rounding.

    Refuses rather than clamps when nothing is big enough. A clamp here would
    silently truncate a batch, and a batch that is quietly one row short is the
    kind of bug that reads as a model quality problem three weeks later.
    """
    if value < 1:
        raise ValueError(f"a bucket holds at least one; got {value}")
    for size in sorted(buckets):
        if size >= value:
            return size
    raise ValueError(f"no bucket in {tuple(sorted(buckets))} holds {value}")


def round_up(value: int, multiple: int) -> int:
    """`value` rounded up to a multiple. The context width's bucketing rule."""
    if multiple < 1:
        raise ValueError(f"a multiple is at least one; got {multiple}")
    if value < 1:
        raise ValueError(f"a width is at least one; got {value}")
    return math.ceil(value / multiple) * multiple


def bucketed(
    shape: DecodeShape,
    *,
    row_buckets: Sequence[int] = ROW_BUCKETS,
    width_multiple: int = WIDTH_MULTIPLE,
) -> DecodeShape:
    """The shape a bucketing capture would actually run, given the one asked for.

    Both guarded dimensions are padded, because a graph is specialised on all of
    them and leaving one free leaves the shape set open. The rows go up to a power
    of two and the width to a multiple, which is the asymmetry described at the top
    of the file: rows are bounded and small, widths are unbounded and large.
    """
    return DecodeShape(
        rows=bucket_for(shape.rows, row_buckets),
        context_width=round_up(shape.context_width, width_multiple),
        query_len=shape.query_len,
    )


def bucket_padding_waste(
    shapes: Sequence[DecodeShape],
    *,
    row_buckets: Sequence[int] = ROW_BUCKETS,
    width_multiple: int = WIDTH_MULTIPLE,
) -> float:
    """Share of the padded cells that nobody wanted. The price of a closed set.

    In the same currency as Day 29's `waste_fraction` and Day 34's
    `prefill_padding_waste`, and it is the third time this engine has bought a
    fixed rectangle and paid for the corners. A padded row attends over a padded
    history: the kernel computes it, the mask throws it away, and the only thing it
    bought is a graph that did not have to be built again.
    """
    real = sum(s.cells for s in shapes)
    if not real:
        return 0.0
    padded = sum(
        bucketed(s, row_buckets=row_buckets, width_multiple=width_multiple).cells
        for s in shapes
    )
    return (padded - real) / padded


# --- graph breaks -------------------------------------------------------------------


def fragments(breaks: int) -> int:
    """Compiled regions a function with `breaks` graph breaks is really made of.

    n breaks is n+1 graphs, and the "+1" is the reason a break count of zero is the
    only number worth aiming at. Each boundary is a return to the interpreter, a
    fresh set of guards to evaluate, and on a device a point where the queue is
    allowed to drain.
    """
    if breaks < 0:
        raise ValueError(f"a graph break count is not negative; got {breaks}")
    return breaks + 1


def fragment_overhead_s(breaks: int, per_break_s: float) -> float:
    """Seconds a step spends going back to Python, at a cost per boundary.

    `per_break_s` is not a constant of nature: it is guard evaluation plus a return
    into the interpreter plus, on a device, whatever the boundary made the host wait
    for. Supply it from a measurement rather than believing a default.
    """
    if per_break_s < 0:
        raise ValueError(f"a per-break cost is not negative; got {per_break_s}")
    return (fragments(breaks) - 1) * per_break_s


@dataclass(frozen=True)
class CompileReport:
    """What a compiler actually built, as opposed to what it was asked to build.

    graphs: compiled regions produced for one call.
    breaks: places the tracer gave up and handed the line back to Python.
    ops:    operations captured, which is the only number here that says how much
            of the work is inside a graph rather than around one.
    """

    graphs: int
    breaks: int
    ops: int

    @property
    def fragments(self) -> int:
        return fragments(self.breaks)

    @property
    def is_one_graph(self) -> bool:
        return self.graphs == 1 and self.breaks == 0

    @property
    def ops_per_graph(self) -> float:
        """Captured ops per region. Small means the fusion had nothing to work with."""
        return self.ops / self.graphs if self.graphs else 0.0


def explain_forward(fn: Callable, *args, **kwargs) -> CompileReport:
    """Ask dynamo what it would make of this call, without keeping the result.

    `torch._dynamo.explain` traces exactly what `torch.compile` would and reports
    the breaks instead of hiding them behind a working callable. It is the only
    honest way to answer "is this compiled?", and it is cheap enough to run in a
    test, which is why the no-breaks claim in this repo is a test and not a comment.
    """
    import torch._dynamo as dynamo

    explanation = dynamo.explain(fn)(*args, **kwargs)
    return CompileReport(
        graphs=int(explanation.graph_count),
        breaks=int(explanation.graph_break_count),
        ops=int(explanation.op_count),
    )


def dynamo_unique_graphs() -> int:
    """Distinct graph *structures* dynamo has built in this process.

    Not the recompile count, and finding that out cost a benchmark run. A frame
    recompiled eight times because a guard on a Python int kept failing has the
    same structure every time, so this counter can sit still while the process
    burns minutes rebuilding. Useful for "how many different things did it build",
    useless for "how often did it build".
    """
    from torch._dynamo.utils import counters

    return int(counters["stats"].get("unique_graphs", 0))


def dynamo_frames_compiled() -> int:
    """Frames dynamo has compiled successfully in this process. The build count.

    This is the one to subtract across a run: it goes up once per compile, whether
    or not the result is a graph it has built before, so a run that recompiles the
    same frame every step reports every one of them. It is the default counter for
    `CompiledDecode` because the number a compile budget is spent in is builds, not
    distinct shapes of build.
    """
    from torch._dynamo.utils import counters

    return int(counters["frames"].get("ok", 0))


# --- the arithmetic of paying for a compile ------------------------------------------


def compile_cost_s(graphs: int, per_graph_s: float) -> float:
    """Seconds spent building. Linear in the graphs, which is the whole hazard.

    A run that presents one shape pays this once. A run that presents ninety-nine
    pays it ninety-nine times, minus whatever dynamo refused to build after it gave
    up, and the refusal is the cheaper of the two outcomes.
    """
    if graphs < 0:
        raise ValueError(f"a graph count is not negative; got {graphs}")
    if per_graph_s < 0:
        raise ValueError(f"a per-graph compile cost is not negative; got {per_graph_s}")
    return graphs * per_graph_s


def step_speedup(forward_share: float, forward_speedup: float) -> float:
    """Amdahl: the step's speedup when only the forward got faster.

    The ceiling on this whole day, and it is worth computing before starting rather
    than after. Day 46's profile is where `forward_share` comes from, and on this
    engine it is not close to 1: the step also schedules, syncs rows, builds inputs,
    samples and collects, and a compiler touches none of those.
    """
    return amdahl(forward_share, forward_speedup)


def saving_per_step_s(*, step_s: float, forward_share: float, forward_speedup: float) -> float:
    """Seconds one step stops costing. The numerator of the breakeven."""
    if step_s < 0:
        raise ValueError(f"a step time is not negative; got {step_s}")
    return step_s - step_s / step_speedup(forward_share, forward_speedup)


def breakeven_steps(compile_s: float, saving_per_step_s: float) -> float:
    """Steps before the compile has paid for itself. Infinite when it never does.

    A float, and it is allowed to be enormous. Two seconds of compile against ten
    microseconds a step is two hundred thousand steps, which at a few hundred
    tokens a request is a server that has to stay up rather than a script.
    """
    if saving_per_step_s <= 0:
        return math.inf
    return compile_s / saving_per_step_s


def net_saving_s(steps: int, *, compile_s: float, saving_per_step_s: float) -> float:
    """Seconds the whole run saved, compile included. Negative before breakeven."""
    if steps < 0:
        raise ValueError(f"a step count is not negative; got {steps}")
    return steps * saving_per_step_s - compile_s


def worth_compiling(steps: int, *, compile_s: float, saving_per_step_s: float) -> bool:
    """Whether this run is long enough to earn its compile back."""
    return net_saving_s(steps, compile_s=compile_s, saving_per_step_s=saving_per_step_s) > 0


# --- the wrapper ----------------------------------------------------------------------


def torch_compiler(fn: Callable, dynamic: bool | None) -> Callable:
    """The default builder: `torch.compile`, with the dynamic decision passed in.

    A named function rather than a lambda inside the class because it is the seam.
    Everything about *policy* in `CompiledDecode` is tested against a fake that
    builds nothing, and this is the one line that has to be swapped to do that.
    """
    return torch.compile(fn, dynamic=dynamic)


class CompiledDecode:
    """The decode forward, wrapped, counted, and honest about which bet it took.

    fn:              the eager callable, `LlamaModel.forward` in this engine.
    mode:            "off" hands `fn` back untouched, and exists so the engine has
                     one code path rather than a branch at the call site.
                     "dynamic" compiles once with both guarded dimensions symbolic,
                     which is the only mode that survives a decode loop unattended.
                     "static" compiles per shape, which is what a capture would do
                     and what makes the recompile bill visible.
    compiler:        `(fn, dynamic) -> callable`. Injectable so the policy above can
                     be tested in milliseconds.
    graph_counter:   `() -> int`, the process-wide count of graphs built. Used to
                     report what was really compiled next to what was predicted.
    recompile_limit: where dynamo stops compiling this frame and goes back to the
                     interpreter forever.

    What it does *not* do is pad, and that stayed true. Bucketing is priced in this
    module and applied in `nanoserve.buckets` (Day 52), one layer down where the
    rectangle is built: a padded batch needs a cache row that is not a scheduler
    slot and a pool address that is not a block, and neither is something a wrapper
    around a forward can invent. What this class does with it is report the shape
    the guard really sees, through `decode_shape`, so `distinct` collapses to the
    bucket count on a bucketed run and stays one-per-step on an unbucketed one.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        mode: str = "dynamic",
        compiler: Callable[[Callable, bool | None], Callable] | None = None,
        graph_counter: Callable[[], int] | None = None,
        recompile_limit: int = RECOMPILE_LIMIT,
    ):
        if mode not in MODES:
            raise ValueError(f"compile mode is one of {MODES}; got {mode!r}")
        self.fn = fn
        self.mode = mode
        self.recompile_limit = recompile_limit
        self._compiler = compiler if compiler is not None else torch_compiler
        # Defaulted to dynamo's own counter, so `compiles` is a measurement rather
        # than a restatement of `expected_compiles`. Left unset in "off" mode: that
        # mode's promise is that the eager path is untouched, and reaching into
        # `torch._dynamo` to count nothing would be the first thing to break it.
        if graph_counter is None and mode != "off":
            graph_counter = dynamo_frames_compiled
        self._graph_counter = graph_counter
        self.calls = 0
        self._shapes: dict[DecodeShape, None] = {}
        self._graphs_at_start: int | None = None
        if mode == "off":
            self._call = fn
        else:
            self._call = self._compiler(fn, mode == "dynamic")

    # --- running it ------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Record the shape, then forward the call unchanged.

        Recording first and forwarding second, so that a call that raises is still
        a shape the compiler was asked about. The forwarding is `*args, **kwargs`
        rather than a fixed signature because the thing being wrapped is
        `LlamaModel.forward` and this class has no business knowing its parameters.
        """
        self._note(args, kwargs)
        self.calls += 1
        if self._graphs_at_start is None and self._graph_counter is not None:
            self._graphs_at_start = self._graph_counter()
        return self._call(*args, **kwargs)

    def _note(self, args, kwargs) -> None:
        """Work out what shape a guard would key this call on, and remember it.

        Two ways in, because the dimension that matters most is not in the
        arguments this wrapper can see. When a cache view is passed, the width is
        `decode_shape`'s: the longest history plus this step's token, which is the
        `[rows, max_ctx]` rectangle the read builds inside the forward. Without one
        the fallback is the tensors themselves, which is what makes this class
        testable over a function that is not a model.
        """
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        tensors += [a for _, a in sorted(kwargs.items()) if isinstance(a, torch.Tensor)]
        if not tensors:
            return
        cache = kwargs.get("cache")
        if getattr(cache, "seq_lens", None):
            shape = decode_shape(tensors[0], cache)
        else:
            rows = int(tensors[0].shape[0])
            width = max((int(d) for t in tensors for d in t.shape[1:]), default=1)
            shape = DecodeShape(rows=max(rows, 1), context_width=max(width, 1))
        self._shapes.setdefault(shape, None)

    # --- what it saw -----------------------------------------------------------

    @property
    def shapes(self) -> tuple[DecodeShape, ...]:
        """The distinct call shapes, in first-seen order.

        A dict rather than a set, because order is information: the first entry is
        the shape the compile was paid for and every later one is a shape that was
        not planned. `recompiles(forward.shapes)` reads the same over a live engine
        as it does over a benchmark's two columns, which is the point of sharing
        the type.
        """
        return tuple(self._shapes)

    @property
    def distinct(self) -> int:
        """Distinct shapes seen, which is builds asked for in static mode."""
        return len(self._shapes)

    @property
    def reuses(self) -> int:
        """Calls that landed on a shape already seen. The point of the exercise."""
        return self.calls - self.distinct

    @property
    def expected_compiles(self) -> int:
        """Builds this mode implies. Zero off, one dynamic, one per shape static."""
        if self.mode == "off":
            return 0
        if self.mode == "dynamic":
            return 1 if self.calls else 0
        return min(self.distinct, self.recompile_limit)

    @property
    def compiles(self) -> int:
        """Builds the compiler really did, when a counter was supplied.

        Falls back to `expected_compiles` when nobody is counting, and the two
        disagreeing is worth knowing about: dynamic mode that recompiles is a
        symbolic dimension that got specialised behind your back.
        """
        if self._graph_counter is None or self._graphs_at_start is None:
            return self.expected_compiles
        return self._graph_counter() - self._graphs_at_start

    @property
    def fell_back(self) -> bool:
        """Whether this wrapper has presented more shapes than dynamo will build.

        Only ever true in static mode, and that is a limitation of this property
        rather than a fact about the engine. It predicts the fallback from *shapes*,
        which is the whole story only when shapes are the whole guard. On Day 49 they
        were not: a `dynamic=True` run rebuilt on a guard over `table.num_tokens` and
        hit the same limit while this returned False. Day 50's `DecodePlan` is what
        made shapes the whole guard, so the prediction and the counter now agree on
        the planned path. The measured version is still `check_graph_reused` against
        `dynamo_frames_compiled`, and where the two disagree, believe the counter.
        """
        return self.mode == "static" and self.distinct > self.recompile_limit


# --- gates ---------------------------------------------------------------------------


def check_single_graph(report: CompileReport) -> None:
    """Refuse a "compiled" function that is really a chain of fragments.

    The gate this whole day turns on. A break count above zero means the tracer
    stopped somewhere, and it is worth failing loudly over because every other
    symptom of it is a benchmark that quietly did not improve.
    """
    if not report.is_one_graph:
        raise CompileUnsound(
            f"the traced call is {report.graphs} graphs with {report.breaks} graph "
            f"breaks in it, not one captured region: every break is a return to the "
            "interpreter, and the code between two of them is not compiled at all"
        )


def check_no_fallback(compiled: CompiledDecode) -> None:
    """Refuse a run that walked past dynamo's cache limit and went back to eager."""
    if compiled.fell_back:
        raise CompileUnsound(
            f"the decode forward presented {compiled.distinct} distinct shapes "
            f"against a recompile limit of {compiled.recompile_limit}: dynamo fell "
            "back to the interpreter for this frame and will not compile it again "
            "in this process"
        )


def check_graph_reused(calls: int, builds: int, *, min_reuse: float = 0.5) -> None:
    """Refuse a compiled run that rebuilt its graph nearly every call.

    The gate this day ends on. A compiler is worth having when one build serves
    many calls; a run where `builds` tracks `calls` has bought the compile cost
    once per step and the compiled speed never, which is a regression however good
    the kernels are. `min_reuse` is the share of calls that must land on a graph
    that already existed.

    Written as `calls` and `builds` rather than taking a `CompiledDecode`, because
    the build count that matters is dynamo's (`dynamo_frames_compiled`) and not the
    wrapper's opinion of what its mode implies.
    """
    if calls < 1:
        raise ValueError(f"a run has at least one call; got {calls}")
    if builds < 0:
        raise ValueError(f"a build count is not negative; got {builds}")
    reuse = (calls - builds) / calls
    if reuse < min_reuse:
        raise CompileUnsound(
            f"{builds} builds over {calls} calls is {reuse:.0%} reuse, under the "
            f"{min_reuse:.0%} this path is worth compiling at: the graph is being "
            "invalidated about as often as it is used, so every step pays a compile "
            "and none of them get the compiled speed"
        )


def check_shapes_bucketed(
    shapes: Sequence[DecodeShape],
    *,
    row_buckets: Sequence[int] = ROW_BUCKETS,
    width_multiple: int = WIDTH_MULTIPLE,
    limit: int = RECOMPILE_LIMIT,
) -> None:
    """Refuse a bucketing that still leaves more shapes than a cache can hold.

    The check a capture list has to pass before it is worth capturing anything: if
    the padded set is still open, the padding bought nothing and was paid for.
    """
    padded = tuple(
        bucketed(s, row_buckets=row_buckets, width_multiple=width_multiple) for s in shapes
    )
    count = distinct_shapes(padded)
    if count > limit:
        raise CompileUnsound(
            f"bucketing leaves {count} distinct shapes against a limit of {limit}: "
            f"the shape set is still open, so the padding is a cost with no graph "
            "reuse to pay for it"
        )


def check_compile_amortised(
    steps: int, *, compile_s: float, saving_per_step_s: float
) -> None:
    """Refuse a compile that this run is too short to earn back."""
    if not worth_compiling(steps, compile_s=compile_s, saving_per_step_s=saving_per_step_s):
        needed = breakeven_steps(compile_s, saving_per_step_s)
        raise CompileUnsound(
            f"{steps} steps do not pay for {compile_s:.2f}s of compile at "
            f"{saving_per_step_s * 1e6:.1f}us saved per step: breakeven is "
            f"{needed:.0f} steps"
        )


# --- the table -------------------------------------------------------------------------


def render(
    *,
    step_s: float,
    forward_share: float,
    forward_speedup: float,
    compile_s: float,
    steps: Sequence[int] = (100, 1_000, 10_000, 100_000),
    title: str | None = None,
) -> str:
    """The three modes against a run length, in seconds saved and seconds spent.

    `compile_s` is the cost of *one* build, because the number of builds is the
    thing that differs between the rows: `dynamic` pays it once, and `static` is
    priced at one build per step, which is what an engine whose context width grows
    every step actually presents. (Dynamo would stop building long before that and
    run eager instead, which is cheaper and worse; the row prices the bill that
    the limit exists to refuse.)

    One row per mode and one column per run length, because the answer genuinely
    changes with the run: the same compile that is a clear win on a server that
    stays up is a ten-second regression on a script that generates forty tokens.
    """
    lines = [title] if title else []
    saved = saving_per_step_s(
        step_s=step_s, forward_share=forward_share, forward_speedup=forward_speedup
    )
    header = f"{'mode':>9}{'speedup':>10}{'saved/step':>13}" + "".join(
        f"{n:>12}" for n in steps
    )
    lines.append(header)
    for mode in MODES:
        if mode == "off":
            per_step, builds = 0.0, 0
        elif mode == "dynamic":
            per_step, builds = saved, 1
        else:
            per_step, builds = saved, None
        speed = 1.0 if mode == "off" else step_speedup(forward_share, forward_speedup)
        row = f"{mode:>9}{speed:>10.2f}x{per_step * 1e6:>10.1f}us"
        for n in steps:
            graphs = n if builds is None else builds
            net = net_saving_s(
                n, compile_s=compile_cost_s(graphs, compile_s), saving_per_step_s=per_step
            )
            row += f"{net:>11.2f}s"
        lines.append(row)
    return "\n".join(lines)
