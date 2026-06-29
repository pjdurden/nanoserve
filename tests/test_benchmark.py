"""Day 13: pure tests for the benchmark timing core.

No model, no torch: a fake clock feeds `measure_stream`/`measure_call` scripted
timestamps so the latency math (TTFT, inter-token latencies, throughput) is
checked exactly, not approximately. The point of a benchmark is that its own
arithmetic is trustworthy, so the arithmetic gets pinned the same way the kernels
do.
"""

from __future__ import annotations

import math

import pytest

from nanoserve.benchmark import RunTiming, measure_call, measure_stream, speedup


class FakeClock:
    """A clock that returns a scripted, strictly increasing sequence of times.

    Each call pops the next value, so a test spells out exactly when the stream
    starts and when each token lands. `measure_stream` reads the clock once before
    the loop and once per yielded token, so a run of N tokens needs 1 + N ticks.
    """

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


def test_measure_stream_splits_ttft_from_inter_token_latencies():
    # start at t=0; token 1 at t=2.0 (TTFT), then tokens at 2.5, 2.9, 3.4.
    clock = FakeClock([0.0, 2.0, 2.5, 2.9, 3.4])
    timing = measure_stream(iter([10, 11, 12, 13]), clock=clock)

    assert timing.n_tokens == 4
    assert timing.ttft_s == pytest.approx(2.0)
    # three gaps after the first token, not four: the first token is TTFT.
    assert timing.itls == pytest.approx([0.5, 0.4, 0.5])


def test_run_timing_derived_metrics():
    timing = RunTiming(ttft_s=2.0, itls=[0.5, 0.4, 0.5], n_tokens=4)

    assert timing.total_s == pytest.approx(2.0 + 1.4)
    # decode throughput excludes the prefill: 3 decode tokens over 1.4s of itls.
    assert timing.decode_tps == pytest.approx(3 / 1.4)
    # end-to-end counts every token over the whole wall time.
    assert timing.tokens_per_s == pytest.approx(4 / 3.4)
    assert timing.median_itl_s == pytest.approx(0.5)


def test_single_token_run_has_no_inter_token_latencies():
    clock = FakeClock([0.0, 1.5])
    timing = measure_stream(iter([99]), clock=clock)

    assert timing.n_tokens == 1
    assert timing.ttft_s == pytest.approx(1.5)
    assert timing.itls == []
    # no decode steps means decode throughput is undefined -> reported as 0.
    assert timing.decode_tps == 0.0
    assert timing.median_itl_s == 0.0
    assert timing.tokens_per_s == pytest.approx(1 / 1.5)


def test_empty_run_is_all_zeros_not_a_crash():
    clock = FakeClock([0.0])
    timing = measure_stream(iter([]), clock=clock)

    assert timing.n_tokens == 0
    assert timing.total_s == 0.0
    assert timing.tokens_per_s == 0.0
    assert timing.decode_tps == 0.0


def test_measure_call_times_a_whole_callable():
    clock = FakeClock([10.0, 12.5])
    elapsed = measure_call(lambda: "done", clock=clock)
    assert elapsed == pytest.approx(2.5)


def test_speedup_is_naive_over_cached():
    # the headline number: how many times faster the cached path is.
    assert speedup(naive_s=87.3, cached_s=15.2) == pytest.approx(87.3 / 15.2)
    assert math.isinf(speedup(naive_s=10.0, cached_s=0.0))
