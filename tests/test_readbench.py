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
    ScalingFit,
    compare_reads,
    fit_scaling,
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


# --- Day 24: the scaling fit -----------------------------------------------
#
# The sweep hands per-length times; these read the *growth* off them. Synthetic
# series with a known closed form (t = c*L^p) pin the log-log slope to the exact
# exponent, so the instrument that decides "flat" vs "linear" is checked, not the
# noise it will later be run on. Same discipline as the timing math above: a
# scaling number no test constrains is a story, not a measurement.


def test_perfectly_linear_series_fits_exponent_one():
    # t = 1e-4 * L: doubling the history doubles the time, slope exactly 1.
    fit = fit_scaling("gather", [(10, 1e-3), (20, 2e-3), (40, 4e-3), (80, 8e-3)])

    assert isinstance(fit, ScalingFit)
    assert fit.label == "gather"
    assert fit.exponent == pytest.approx(1.0)
    assert fit.regime == "linear"
    # per-token cost is read off the *largest* length: 8e-3 / 80 = 1e-4 s.
    assert fit.per_token_s == pytest.approx(1e-4)


def test_constant_series_fits_flat():
    # Same time at every length: the read does not grow with history, slope 0.
    fit = fit_scaling("fused", [(16, 5e-3), (64, 5e-3), (256, 5e-3)])

    assert fit.exponent == pytest.approx(0.0)
    assert fit.regime == "flat"
    assert fit.per_token_s == pytest.approx(5e-3 / 256)


def test_quadratic_series_fits_exponent_two():
    # t = L**2 / 100: the classic superlinear blow-up, slope exactly 2.
    fit = fit_scaling("gather", [(10, 1.0), (20, 4.0), (40, 16.0)])

    assert fit.exponent == pytest.approx(2.0)
    assert fit.regime == "superlinear"


def test_sqrt_series_fits_sublinear():
    # t = sqrt(L): grows, but slower than the history, slope 0.5.
    fit = fit_scaling("fused", [(4, 2.0), (16, 4.0), (64, 8.0)])

    assert fit.exponent == pytest.approx(0.5)
    assert fit.regime == "sublinear"


def test_per_token_uses_the_largest_length_regardless_of_order():
    # Points handed out of order; the per-token cost still comes from L=80.
    fit = fit_scaling("gather", [(80, 8e-3), (10, 1e-3), (40, 4e-3)])

    assert fit.per_token_s == pytest.approx(8e-3 / 80)


def test_fit_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least two"):
        fit_scaling("fused", [(16, 1e-3)])


def test_fit_needs_two_distinct_lengths():
    # Two samples at the same history length have no slope to fit.
    with pytest.raises(ValueError, match="distinct"):
        fit_scaling("fused", [(16, 1e-3), (16, 2e-3)])


def test_fit_rejects_nonpositive_length_or_time():
    # log-log has no answer at zero; a zero best-of (the degenerate case
    # `speedup` guards) must raise here rather than feed NaN into the slope.
    with pytest.raises(ValueError, match="positive"):
        fit_scaling("fused", [(16, 1e-3), (64, 0.0)])
    with pytest.raises(ValueError, match="positive"):
        fit_scaling("fused", [(0, 1e-3), (64, 2e-3)])
