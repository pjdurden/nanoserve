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
