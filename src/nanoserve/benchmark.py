"""Day 13: the timing core that turns the Day-11 "5.74x" point into a curve.

Day 11 measured one number (40 tokens, 87s naive versus 15s cached). One point is
an anecdote; the story the KV cache tells is a *shape*: cached decode is O(n) so
its tokens/sec stays roughly flat as the sequence grows, while the naive recompute
path is O(n^2) so its tokens/sec collapses as length climbs. To draw that you have
to measure across lengths, and to trust the drawing the measurement arithmetic has
to be exactly right.

So the timing lives here, stdlib-only and model-free, with the clock injected. The
model loop in ``bench.py`` hands this module an iterator of tokens (or a callable)
and a real ``time.perf_counter``; the tests hand it a scripted fake clock and pin
TTFT, inter-token latency, and throughput to the decimal. The split matters: a
benchmark whose own math is unverified is just a confident guess.

Definitions, matching how serving systems report latency:
- TTFT (time to first token): prefill plus the first decode, the wait a user feels
  before anything appears.
- ITL (inter-token latency): the gap between each subsequent token, the streaming
  cadence.
- decode throughput: decode tokens divided by time spent decoding (TTFT excluded),
  the steady-state speed.
- end-to-end tokens/sec: every generated token over the whole wall clock.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

Clock = Callable[[], float]


@dataclass
class RunTiming:
    """Timings for one generation run, plus the metrics derived from them.

    ttft_s:   seconds from the start of the call to the first token (prefill +
              first decode).
    itls:     inter-token latencies in seconds, one per token *after* the first.
              A run of n tokens has n-1 entries here; an empty or single-token run
              has none.
    n_tokens: total tokens generated.
    """

    ttft_s: float
    itls: list[float] = field(default_factory=list)
    n_tokens: int = 0

    @property
    def total_s(self) -> float:
        """Whole wall time: prefill+first token, then every inter-token gap."""
        if self.n_tokens == 0:
            return 0.0
        return self.ttft_s + sum(self.itls)

    @property
    def decode_tps(self) -> float:
        """Steady-state decode speed: decode tokens per second of decoding.

        Excludes TTFT, because prefill is a one-time cost that should not be
        amortized into the per-token rate. Zero when there were no decode steps.
        """
        decode_time = sum(self.itls)
        if not self.itls or decode_time == 0.0:
            return 0.0
        return len(self.itls) / decode_time

    @property
    def tokens_per_s(self) -> float:
        """End-to-end throughput: all tokens over the whole wall time."""
        total = self.total_s
        return self.n_tokens / total if total > 0.0 else 0.0

    @property
    def median_itl_s(self) -> float:
        """Median inter-token latency; robust to the occasional slow step."""
        return statistics.median(self.itls) if self.itls else 0.0


def measure_stream(tokens: Iterable[Any], *, clock: Clock = time.perf_counter) -> RunTiming:
    """Consume a token iterator, timing TTFT and each inter-token latency.

    Reads the clock once before pulling anything (the start), then once per token
    as it arrives. The first token's gap from the start is TTFT; every later gap
    from the previous token is an inter-token latency. Works for any iterator, so
    the same function times ``model.generate_stream`` in the CLI and a plain list
    of ints in the tests.
    """
    start = clock()
    prev = start
    ttft = 0.0
    itls: list[float] = []
    n = 0
    it: Iterator[Any] = iter(tokens)
    for _ in it:
        now = clock()
        if n == 0:
            ttft = now - start
        else:
            itls.append(now - prev)
        prev = now
        n += 1
    return RunTiming(ttft_s=ttft, itls=itls, n_tokens=n)


def measure_call(fn: Callable[[], Any], *, clock: Clock = time.perf_counter) -> float:
    """Time a single callable end to end, returning elapsed seconds.

    Used for the naive recompute path, which returns the whole sequence at once
    rather than streaming, so only its total wall time is observable.
    """
    t0 = clock()
    fn()
    return clock() - t0


def speedup(*, naive_s: float, cached_s: float) -> float:
    """How many times faster the cached path is than the naive one.

    Infinity if the cached path measured as zero time (degenerate, but better than
    a ZeroDivisionError mid-sweep).
    """
    if cached_s == 0.0:
        return math.inf
    return naive_s / cached_s
