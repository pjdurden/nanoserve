"""Day 20: pure tests for the paged-read microbenchmark core.

No model, no torch: a fake clock feeds `time_read`/`compare_reads` scripted
timestamps and plain callables stand in for the two read paths, so the timing
math (best-of, mean, median, speedup) is checked exactly, not approximately. Same
discipline as the Day-13 benchmark tests: a microbenchmark whose own arithmetic is
unverified is just a confident guess, and this one exists to hand the Triton kernel
a number to beat, so the number has to be trustworthy.
"""

from __future__ import annotations

import math

import pytest

from nanoserve.readbench import (
    ReadComparison,
    ReadTiming,
    compare_reads,
    time_read,
)


class FakeClock:
    """Returns a scripted, strictly increasing sequence of times.

    `time_read` reads the clock exactly twice per *timed* call (before and after),
    and never during warmup, so a run of R repeats with any warmup needs exactly
    2*R ticks. Each call pops the next value.
    """

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


def test_time_read_records_one_sample_per_repeat():
    # Three timed calls: (10.0->10.5), (20.0->20.2), (30.0->30.9).
    clock = FakeClock([10.0, 10.5, 20.0, 20.2, 30.0, 30.9])
    timing = time_read(lambda: None, label="fused", repeats=3, clock=clock)

    assert timing.label == "fused"
    assert timing.n_calls == 3
    assert timing.samples_s == pytest.approx([0.5, 0.2, 0.9])
    # best-of is the number a microbenchmark reports (least noise, no scheduler jitter).
    assert timing.min_s == pytest.approx(0.2)
    assert timing.mean_s == pytest.approx(1.6 / 3)
    assert timing.median_s == pytest.approx(0.5)


def test_warmup_calls_run_but_are_not_timed():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    # warmup=2 untimed + repeats=2 timed -> fn runs 4 times, clock only during the 2 timed.
    clock = FakeClock([1.0, 1.3, 5.0, 5.1])
    timing = time_read(fn, label="gather", repeats=2, warmup=2, clock=clock)

    assert calls["n"] == 4
    assert timing.n_calls == 2
    assert timing.samples_s == pytest.approx([0.3, 0.1])
    assert timing.min_s == pytest.approx(0.1)


def test_speedup_is_gather_over_fused_best_of():
    gather = ReadTiming(label="gather", samples_s=[0.8, 0.9, 1.0])  # min 0.8
    fused = ReadTiming(label="fused", samples_s=[0.2, 0.25, 0.3])  # min 0.2
    cmp = ReadComparison(gather=gather, fused=fused)

    # gather best / fused best: the fused read is 4x faster here.
    assert cmp.speedup == pytest.approx(4.0)
    assert cmp.faster == "fused"


def test_speedup_reports_gather_faster_when_fused_is_slower():
    gather = ReadTiming(label="gather", samples_s=[0.2])
    fused = ReadTiming(label="fused", samples_s=[0.5])
    cmp = ReadComparison(gather=gather, fused=fused)

    assert cmp.speedup == pytest.approx(0.4)
    assert cmp.faster == "gather"


def test_speedup_is_infinite_when_fused_measures_zero():
    gather = ReadTiming(label="gather", samples_s=[0.3])
    fused = ReadTiming(label="fused", samples_s=[0.0])
    cmp = ReadComparison(gather=gather, fused=fused)

    assert cmp.speedup == math.inf
    assert cmp.faster == "fused"


def test_compare_reads_times_both_paths_in_order():
    # gather timed first (two calls), then fused (two calls); 8 ticks total.
    clock = FakeClock([0.0, 1.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.3])
    cmp = compare_reads(lambda: None, lambda: None, repeats=2, clock=clock)

    assert isinstance(cmp, ReadComparison)
    assert cmp.gather.label == "gather"
    assert cmp.fused.label == "fused"
    assert cmp.gather.min_s == pytest.approx(1.0)
    assert cmp.fused.min_s == pytest.approx(0.3)
    assert cmp.speedup == pytest.approx(1.0 / 0.3)
