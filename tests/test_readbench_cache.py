"""Day 26: the read benchmark run against the cache's real dispatched read.

Day 20 built the timing core and Day 24 the scaling fit, both in `readbench.py`,
and both timed `gather` and `fused` as standalone functions: the pure tests in
`test_readbench.py` hand them `lambda: None`. That measures the harness, not the
engine. This file drives the other half: `build_paged_reads` prefills a *real*
`PagedKVCache` to a history length and hands back the two zero-arg read closures
the timing core expects, the gather (rebuild the contiguous history then score,
the naive path) and the *dispatched* fused read (`paged_attention` through
`select_backend`, the path the engine actually takes on a paged cache).

So these tests need torch and the cache; they stay out of `test_readbench.py`,
which is deliberately model-free. They pin the two things a benchmark's own math
cannot: that the dispatched read the sweep times agrees with the gather it is
compared against (to the streaming tolerance Day 25 established), and that neither
read grows the cache, so every per-length sample measures the same history.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.config import ModelConfig
from nanoserve.readbench import (
    build_paged_reads,
    sweep_paged_reads,
    time_paged_reads,
)


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the cache/model tests use."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


class FakeClock:
    """Returns a scripted, strictly increasing sequence of times (see test_readbench).

    `time_read` reads the clock exactly twice per *timed* call and never during
    warmup, so a `compare_reads` of R repeats needs 2*R ticks per path, gather
    first then fused, and a sweep needs that block once per history length.
    """

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


def test_build_paged_reads_returns_two_decode_reads():
    """Both closures produce a decode step's attention output [1, n_q, 1, d]."""
    cfg = _tiny_config()
    gather_fn, fused_fn = build_paged_reads(cfg, history_len=20)

    want = (1, cfg.num_attention_heads, 1, cfg.head_dim)
    assert tuple(gather_fn().shape) == want
    assert tuple(fused_fn().shape) == want


def test_dispatched_read_matches_the_gather_it_is_timed_against():
    """The fused read the sweep times agrees with the gather it is compared to.

    On CPU the dispatched read runs the tlsim model, which reassociates the online
    softmax and so lands a few ulps off the contiguous gather rather than bit for
    bit. Pin it at the same atol Day 25 held the cache read to: a benchmark that
    times two paths only means something if both compute the same answer.
    """
    cfg = _tiny_config()
    gather_fn, fused_fn = build_paged_reads(cfg, history_len=37, seed=3)
    assert torch.allclose(fused_fn(), gather_fn(), atol=1e-5)


def test_read_closures_are_pure_so_every_sample_times_the_same_history():
    """Calling a closure repeatedly reads the identical history: no write, no growth.

    A microbenchmark repeats the read many times; if the read appended its query's
    K/V (as the engine's `paged_attention` does) the history would grow every call
    and the per-length number would be a moving average of many lengths. These
    closures only read, so each of many calls returns byte-identical output.
    """
    cfg = _tiny_config()
    gather_fn, fused_fn = build_paged_reads(cfg, history_len=25, seed=1)
    assert torch.equal(gather_fn(), gather_fn())
    assert torch.equal(fused_fn(), fused_fn())


def test_history_len_below_one_is_rejected():
    """No token to read: the fit is over history lengths, all of them positive."""
    cfg = _tiny_config()
    with pytest.raises(ValueError):
        build_paged_reads(cfg, history_len=0)


def test_time_paged_reads_times_both_real_paths():
    """`time_paged_reads` runs the Day-20 comparison against the real cache reads.

    A scripted clock makes the timing exact: two repeats per path, gather timed
    first, so eight ticks. The comparison carries one sample per repeat on each
    side, and the fused side is the dispatched read, not a standalone function.
    """
    cfg = _tiny_config()
    clock = FakeClock([0.0, 0.9, 0.0, 0.4, 0.0, 0.5, 0.0, 0.2])
    cmp = time_paged_reads(cfg, history_len=16, repeats=2, clock=clock)

    assert cmp.gather.label == "gather"
    assert cmp.fused.label == "fused"
    assert cmp.gather.n_calls == 2
    assert cmp.fused.n_calls == 2
    # gather best-of 0.4, fused best-of 0.2 under this script: fused twice as fast.
    assert cmp.gather.min_s == pytest.approx(0.4)
    assert cmp.fused.min_s == pytest.approx(0.2)
    assert cmp.speedup == pytest.approx(2.0)


def test_sweep_paged_reads_fits_both_paths_over_the_real_reads():
    """The sweep reads each path's growth off the *dispatched* read, not a stub.

    Script the clock so both paths clock `t = c * L` at the three lengths (gather
    at 1e-4*L, fused at 0.5e-4*L), one timed call each. Both fit exponent 1, the
    linear regime, and the sweep returns the two fits labelled by path, ready to
    print the exponent and the per-token constant a card later bends down.
    """
    cfg = _tiny_config()
    # Per length: gather (t0, t1) then fused (t0, t1), repeats=1.
    ticks = [
        0.0, 16e-4, 0.0, 8e-4,      # L=16
        0.0, 64e-4, 0.0, 32e-4,     # L=64
        0.0, 256e-4, 0.0, 128e-4,   # L=256
    ]
    clock = FakeClock(ticks)
    gather_fit, fused_fit = sweep_paged_reads(
        cfg, lengths=[16, 64, 256], repeats=1, clock=clock
    )

    assert gather_fit.label == "gather"
    assert fused_fit.label == "fused"
    assert gather_fit.lengths == [16, 64, 256]
    assert gather_fit.exponent == pytest.approx(1.0)
    assert fused_fit.exponent == pytest.approx(1.0)
    assert gather_fit.regime == "linear"
    assert fused_fit.regime == "linear"
