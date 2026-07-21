"""Day 20: the paged-read microbenchmark core, the number the kernel must beat.

Day 16 stored the KV cache in scattered blocks but read it the slow way: gather
every block back into one contiguous history each step, then run a normal SDPA over
it. Day 18 wrote the torch reference that reads through the block table directly,
and Day 19 fused that read into attention. All three are still *torch*: the fused
read is the correctness oracle, not the fast path. The speed only arrives with the
hand-written Triton kernel, and before you write a kernel you need a target: how
long does the current read take, and how does that cost grow as the history gets
longer? This module measures exactly that, so the kernel has a fixed number to beat
instead of a vibe.

The timing lives here, stdlib-only and model-free, with the clock injected. The
runner in ``readbench.py`` builds a real `PagedKVCache`, prefills it to a length,
and hands this module two zero-arg closures (the gather read and the fused read)
plus a real ``time.perf_counter``; the tests hand it a scripted fake clock and
plain callables and pin best-of, mean, median, and the speedup ratio to the decimal.
The split is the same one Day 13 drew: a benchmark whose own math is unverified is
just a confident guess, and this one exists to size the kernel's job, so its
arithmetic is pinned the way the kernels are.

Why best-of (``min_s``) as the headline number? A microbenchmark on a shared CPU
is contaminated upward by every scheduler hiccup and never downward: nothing makes
the same work finish faster than the hardware allows. So the fastest sample is the
cleanest estimate of the real cost, and it is what perf reports (and the kernel
comparison) should quote. Mean and median stay available to show the spread.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

Clock = Callable[[], float]


@dataclass
class ReadTiming:
    """Per-call timings for one read path, plus the summaries derived from them.

    label:     which read this is ("gather" or "fused"), carried so a comparison
               and its printout never mix the two up.
    samples_s: one wall-clock duration per *timed* call, warmups excluded. Order is
               call order, so the raw jitter is inspectable, not just the summary.
    """

    label: str
    samples_s: list[float] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        """How many timed samples were collected (warmups do not count)."""
        return len(self.samples_s)

    @property
    def min_s(self) -> float:
        """Best-of: the fastest sample, the headline microbenchmark number.

        Cleanest estimate of the true cost, because noise on a shared box only ever
        pushes a sample slower, never faster. Zero for an empty run.
        """
        return min(self.samples_s) if self.samples_s else 0.0

    @property
    def mean_s(self) -> float:
        """Average sample; sensitive to outliers, kept to show the spread."""
        return statistics.fmean(self.samples_s) if self.samples_s else 0.0

    @property
    def median_s(self) -> float:
        """Middle sample; robust to the occasional slow call."""
        return statistics.median(self.samples_s) if self.samples_s else 0.0


@dataclass
class ReadComparison:
    """The two read paths timed side by side, with the ratio that matters.

    gather: the Day-16 read, rebuild the contiguous history then attend.
    fused:  the Day-19 read, attend over the scattered blocks in place.

    On CPU both are torch and both do an O(history) index gather, so the honest
    result is roughly a tie: the reference is not where the speedup lives. The
    number this comparison exists to expose is each path's *absolute* per-step cost
    and how it scales with history length, which is the baseline the Triton kernel
    has to beat.
    """

    gather: ReadTiming
    fused: ReadTiming

    @property
    def speedup(self) -> float:
        """How many times faster the fused read is than the gather (best-of / best-of).

        > 1 means fused wins, < 1 means gather wins. Infinity if the fused read
        measured as zero time (degenerate, but better than a ZeroDivisionError
        mid-sweep), matching how `benchmark.speedup` guards the same case.
        """
        if self.fused.min_s == 0.0:
            return math.inf
        return self.gather.min_s / self.fused.min_s

    @property
    def faster(self) -> str:
        """Which path won best-of: "fused", "gather", or "tie" on an exact match."""
        if self.fused.min_s == self.gather.min_s:
            return "tie"
        return "fused" if self.fused.min_s < self.gather.min_s else "gather"


@dataclass
class ScalingFit:
    """How one read path's cost grows with history, read off the sweep's points.

    Day 20 timed each read at a handful of history lengths and printed a table; a
    table does not tell you whether the curve is flat or climbing, which is the one
    thing the kernel is supposed to change. This is the instrument that reads the
    slope: given the per-length best-of times, it fits ``t = c * L**p`` and reports
    the exponent ``p``. That is the empirical big-O of the read, the number that
    separates "grows with the history" from "does not".

    label:   which read this is ("gather", "fused", ...), so a printed fit and its
             curve never get crossed.
    lengths: the history lengths sampled, one per point, in the order given.
    seconds: the best-of time at each of those lengths, aligned by index.

    The fit is a least-squares line through the points in log-log space. Slope is
    scale-invariant, so it does not matter whether the times are seconds or
    milliseconds; only the ``per_token_s`` readout below carries a unit, and it is
    taken straight from the raw seconds.
    """

    label: str
    lengths: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    @property
    def exponent(self) -> float:
        """The log-log least-squares slope: the empirical exponent ``p`` in ``c*L**p``.

        Fit ``log t = log c + p * log L`` by ordinary least squares over the points,
        so ``p`` is the growth the sweep actually measured. A perfectly linear read
        (double the history, double the time) returns 1.0; a read whose cost does not
        move with the history returns 0.0; an O(L^2) blow-up returns 2.0. The log
        turns every power law into a straight line, which is why the one slope
        describes the whole curve instead of a single doubling.
        """
        xs = [math.log(length) for length in self.lengths]
        ys = [math.log(sec) for sec in self.seconds]
        n = len(xs)
        x_bar = sum(xs) / n
        y_bar = sum(ys) / n
        num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        den = sum((x - x_bar) ** 2 for x in xs)
        return num / den

    @property
    def per_token_s(self) -> float:
        """Seconds per history token at the *largest* length sampled.

        The absolute cost the slope hides: two reads can share an exponent and still
        differ by an order of magnitude in constant, and on CPU that constant is
        where the fused read's whole win lives (same O(L), less memory moved). Taken
        at the longest history because that is where the per-step cost matters and
        where fixed overheads are most diluted.
        """
        i = max(range(len(self.lengths)), key=lambda j: self.lengths[j])
        return self.seconds[i] / self.lengths[i]

    @property
    def regime(self) -> str:
        """A word for the exponent: "flat", "sublinear", "linear", or "superlinear".

        Bands around the integer exponents so a noisy real sweep still classifies
        cleanly: <=0.25 reads as flat (cost independent of history), (0.25, 0.75] as
        sublinear, (0.75, 1.25] as linear (the gather, and every honest per-step
        attention, lives here), and anything past 1.25 as superlinear. The raw
        ``exponent`` is the measurement; this is the label for a glance.
        """
        p = self.exponent
        if p <= 0.25:
            return "flat"
        if p <= 0.75:
            return "sublinear"
        if p <= 1.25:
            return "linear"
        return "superlinear"


def fit_scaling(label: str, points: list[tuple[int, float]]) -> ScalingFit:
    """Fit ``t = c * L**p`` to (history_len, seconds) points; return the `ScalingFit`.

    points: one ``(history_len, seconds)`` per sampled length, the best-of time at
            each. Order is free; the fit and the per-token readout sort it out.

    Raises `ValueError` on anything the log-log fit cannot answer: fewer than two
    points (no line through one point), fewer than two *distinct* lengths (a vertical
    line has no slope), or a non-positive length or time (``log`` is undefined, and a
    zero best-of is the degenerate case `speedup` already guards). Catching these
    here keeps a NaN out of the exponent, where it would quietly poison the regime.
    """
    if len(points) < 2:
        raise ValueError(f"scaling needs at least two points to fit a slope; got {len(points)}")
    lengths = [length for length, _ in points]
    seconds = [sec for _, sec in points]
    if any(length <= 0 for length in lengths) or any(sec <= 0.0 for sec in seconds):
        raise ValueError(
            "scaling is fit in log-log space, so every history length and time must be "
            f"positive; got lengths={lengths}, seconds={seconds}"
        )
    if len(set(lengths)) < 2:
        raise ValueError(
            f"scaling needs at least two distinct history lengths to fit a slope; got {lengths}"
        )
    return ScalingFit(label=label, lengths=lengths, seconds=seconds)


def time_read(
    fn: Callable[[], object],
    *,
    label: str,
    repeats: int,
    warmup: int = 0,
    clock: Clock = time.perf_counter,
) -> ReadTiming:
    """Time a zero-arg read `repeats` times, after `warmup` untimed calls.

    The warmup calls run the read without touching the clock, so the first-call
    costs that do not belong to steady state (lazy allocations, import-time caches,
    a cold CPU) land before timing starts. Then each timed call reads the clock
    once before and once after; the difference is one sample. The read's return
    value is discarded: this measures how long it takes, not what it produces (its
    output is already pinned byte-identical by the Day-18/19 tests).
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = clock()
        fn()
        samples.append(clock() - t0)
    return ReadTiming(label=label, samples_s=samples)


def compare_reads(
    gather_fn: Callable[[], object],
    fused_fn: Callable[[], object],
    *,
    repeats: int,
    warmup: int = 0,
    clock: Clock = time.perf_counter,
) -> ReadComparison:
    """Time the gather read then the fused read under identical settings.

    Same repeats and warmup for both so the comparison is apples to apples; the
    gather is timed first only because that is the order the story reads (the slow
    Day-16 path, then the fused Day-19 one). Returns the pair wrapped so the ratio
    and the winner are one attribute away.
    """
    gather = time_read(gather_fn, label="gather", repeats=repeats, warmup=warmup, clock=clock)
    fused = time_read(fused_fn, label="fused", repeats=repeats, warmup=warmup, clock=clock)
    return ReadComparison(gather=gather, fused=fused)


# --- Day 26: the closures wired to the cache's real dispatched read ----------
#
# Everything above is the timing core: stdlib-only, model-free, the clock
# injected, and the two read paths handed in as opaque zero-arg callables. The
# pure tests feed it `lambda: None`; Days 20 and 24 fed it standalone functions.
# That measures the harness, not the engine.
#
# The runner below closes the gap the Day-25 log named: it builds a *real*
# `PagedKVCache`, prefills it to a history length, and hands the core the two
# closures for the reads the engine actually runs. The gather closure rebuilds the
# contiguous history and scores it, the naive path a `NaiveKVCache` drives through
# `gqa_attention`. The fused closure is the *dispatched* read: `paged_attention`
# through `select_backend`, the tlsim model on a CPU and the Triton kernel on a
# card, exactly what `PagedKVCache.paged_attention` invokes. So the sweep now times
# the path the box would take, not a function called in isolation.
#
# torch and the cache are imported inside the functions, not at module top, so the
# timing core above stays importable without them and keeps its stdlib-only shape.


def build_paged_reads(
    config,
    history_len: int,
    *,
    block_size: int = 16,
    layer: int = 0,
    seed: int = 0,
) -> tuple[Callable[[], object], Callable[[], object]]:
    """Prefill a real `PagedKVCache` to `history_len` and return (gather_fn, fused_fn).

    Two zero-arg read closures over that one fixed history, the pair `compare_reads`
    expects:

    gather_fn: the Day-16 read, gather the scattered pool back into one contiguous
               ``[1, n_kv, L, d]`` history, GQA-repeat it, and score this step's
               single decode query against it. That is the work a `NaiveKVCache`
               drives inside `gqa_attention`; a single decode query sees the whole
               history, so the causal mask is all-visible and drops out.
    fused_fn:  the dispatched paged read, `paged_attention` through `select_backend`
               over the *same* pool and the real slot mapping. On a CPU that is the
               Day-22 tlsim model; on a card it is the Triton kernel. This is the
               path `PagedKVCache.paged_attention` takes, minus the O(1) write.

    Both return the decode step's attention output ``[1, n_q, 1, d]``. On CPU they
    agree to a few ulps rather than bit for bit, because the fused read streams the
    online softmax (the Day-25 trade); `test_readbench_cache` pins that closeness.

    Neither closure writes: the query's K/V is never appended, so the cache's history
    stays at `history_len` no matter how many times a closure is called. That is what
    makes a repeated timing sample honest, every repeat reads the identical length,
    where calling the engine's writing `paged_attention` in a loop would grow it.

    Raises `ValueError` for `history_len < 1`: there is no token to read, and the
    scaling fit downstream is over strictly-positive lengths anyway.
    """
    import torch

    from .cache import BlockAllocator, PagedKVCache
    from .kernels.triton_paged_attention import paged_attention as paged_attention_dispatch
    from .layers import repeat_kv

    if history_len < 1:
        raise ValueError(f"history_len must be at least 1 to read a token; got {history_len}")

    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    n_rep = config.num_kv_groups
    d = config.head_dim
    scale = d**-0.5

    # A pool sized to hold the whole history, then one prefill write to fill it.
    num_blocks = -(-history_len // block_size)  # ceil division: blocks to cover L
    cache = PagedKVCache(config, BlockAllocator(num_blocks=num_blocks, block_size=block_size))
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(1, n_kv, history_len, d, generator=gen)
    v = torch.randn(1, n_kv, history_len, d, generator=gen)
    cache.append(layer, k, v)  # write + gather; we keep only the pool it wrote

    # This step's single decode query, fixed so both reads score the same thing.
    q = torch.randn(1, n_q, 1, d, generator=gen)
    slot_mapping = cache._slots_for(range(history_len), q.device)
    k_pool = cache.k_pool[layer]
    v_pool = cache.v_pool[layer]

    def gather_fn() -> object:
        # Rebuild the contiguous history (the Day-16 read), then score it: exactly
        # the work a naive cache drives through `gqa_attention` on a decode step.
        # One query over the full history has an all-visible mask, so it is omitted.
        hist_k = repeat_kv(k_pool[slot_mapping].transpose(0, 1)[None], n_rep)
        hist_v = repeat_kv(v_pool[slot_mapping].transpose(0, 1)[None], n_rep)
        scores = torch.matmul(q, hist_k.transpose(2, 3)) * scale
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, hist_v)

    def fused_fn() -> object:
        # The dispatched paged read: `select_backend` over the same pool and slots.
        return paged_attention_dispatch(q, k_pool, v_pool, slot_mapping, n_rep, scale)

    return gather_fn, fused_fn


def time_paged_reads(
    config,
    history_len: int,
    *,
    repeats: int,
    warmup: int = 0,
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> ReadComparison:
    """Time the gather against the *dispatched* fused read at one history length.

    Builds the real cache reads (`build_paged_reads`) and hands them to
    `compare_reads` under identical settings. This is the Day-20 comparison run
    against the path the engine actually takes rather than two standalone functions:
    the fused side is `paged_attention` through `select_backend`, so on a CPU the
    number is the tlsim model's and on a card it is the Triton kernel's. `warmup`
    absorbs the first-call costs (lazy allocation, a cold cache) before timing.
    """
    gather_fn, fused_fn = build_paged_reads(config, history_len, **build_kwargs)
    return compare_reads(gather_fn, fused_fn, repeats=repeats, warmup=warmup, clock=clock)


def sweep_paged_reads(
    config,
    lengths: list[int],
    *,
    repeats: int,
    warmup: int = 0,
    clock: Clock = time.perf_counter,
    **build_kwargs,
) -> tuple[ScalingFit, ScalingFit]:
    """Sweep the real reads across `lengths` and fit each path's growth.

    Times both reads at every history length (`time_paged_reads`), takes each path's
    best-of, and fits ``t = c * L**p`` to the points (`fit_scaling`). Returns
    ``(gather_fit, fused_fit)``: the Day-24 scaling fit read off the *dispatched*
    read, so the exponent and the per-token constant describe the path the engine
    runs. On a CPU both fit roughly linear (both do an O(L) read); the fused
    constant is the tlsim model's, the number a card bends down when the Triton
    kernel replaces it, which is the whole claim Week 6 has been setting up.
    """
    gather_points: list[tuple[int, float]] = []
    fused_points: list[tuple[int, float]] = []
    for length in lengths:
        comparison = time_paged_reads(
            config, length, repeats=repeats, warmup=warmup, clock=clock, **build_kwargs
        )
        gather_points.append((length, comparison.gather.min_s))
        fused_points.append((length, comparison.fused.min_s))
    return fit_scaling("gather", gather_points), fit_scaling("fused", fused_points)
