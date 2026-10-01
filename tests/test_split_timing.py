"""Day 72: the split server times its own read before it serves, and says if it paid.

Day 71's probe says the two passes are *correct* on the device the server boots on.
It says nothing about whether they are *worth it*. The split read exists to shorten
the longest program in a decode launch, and `SplitPlan.wave_speedup` predicts how
much from tile counts alone: a row four tiles long cut into one-tile chunks has a
tail of one tile where it had four. That is a prediction about a fully resident grid,
and only a card can say whether it holds.

So `time_split_read` builds a case the prediction is about (one long row, a few
one-key rows, chunks one tile wide), runs the unsplit batched read and the split
read on it, checks they agree, and reports the median of each next to the
prediction. `measure_split_read` runs it on the engine's device with the engine's
geometry, `build_app` calls it right after the probe, and `/health` carries it.

The host result is the lesson. tlsim runs one program at a time, so it collects the
*work*, which the split leaves unchanged and adds a reduce to, and never the *wave*.
The predicted 4x shows up here as a measured ratio under 1, and that is not a bug: it
is the prediction being about a machine this box is not.
"""

from __future__ import annotations

import pytest
import torch
from reference import requires_triton_gpu
from test_split_boot import GRAPHS, _app, _engine, _tiny_config

from nanoserve.kernels import flash_decoding
from nanoserve.kernels.flash_decoding import (
    SplitTiming,
    SplitUnsound,
    split_plan,
    time_split_read,
)
from nanoserve.launch import BootUnsound, boot_info, check_boot_info, measure_split_read

CPU = torch.device("cpu")


class _Clock:
    """A clock that only moves when a planted read tells it to."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _plant(monkeypatch, clock, unsplit_costs, split_costs):
    """Wrap both reads so each call advances `clock` by the next cost in its list.

    The honest read still runs, so the agreement check sees real attention; only the
    time it is charged is planted. Each list's first entry is the warm call.
    """
    honest_unsplit = flash_decoding.paged_attention_batched
    honest_split = flash_decoding.paged_attention_split
    unsplit_costs, split_costs = list(unsplit_costs), list(split_costs)

    def unsplit(*args, **kwargs):
        out = honest_unsplit(*args, **kwargs)
        clock.now += unsplit_costs.pop(0)
        return out

    def split(*args, **kwargs):
        out = honest_split(*args, **kwargs)
        clock.now += split_costs.pop(0)
        return out

    monkeypatch.setattr(flash_decoding, "paged_attention_batched", unsplit)
    monkeypatch.setattr(flash_decoding, "paged_attention_split", split)


# --- the case and the prediction ------------------------------------------------------


def test_the_timing_runs_on_the_host_and_names_its_backend():
    report = time_split_read(CPU, block=4, head_dim=8, n_rep=2, repeats=2)
    assert isinstance(report, SplitTiming)
    assert report.backend == "tlsim"
    assert report.unsplit_ms > 0 and report.split_ms > 0
    assert report.repeats == 2


def test_the_prediction_is_the_plan_s_wave_speedup_for_the_timed_case():
    report = time_split_read(CPU, block=4, head_dim=8, long_tiles=4, short_rows=3,
                             repeats=1)
    plan = split_plan([16, 1, 1, 1], n_q=2, head_dim=8, block=4, splits=4)
    assert report.predicted == plan.wave_speedup == 4.0
    assert (report.tail_tiles, report.unsplit_tail_tiles) == (1, 4)
    assert report.rows == 4 and report.splits == 4


def test_the_case_is_cut_one_tile_per_chunk_whatever_the_block():
    for block in (1, 4, 16):
        report = time_split_read(CPU, block=block, head_dim=4, long_tiles=3, repeats=1)
        assert report.block == block
        assert report.predicted == 3.0


def test_a_case_with_nothing_to_split_is_refused():
    with pytest.raises(ValueError, match="tiles"):
        time_split_read(CPU, block=4, head_dim=8, long_tiles=1)


def test_a_timing_needs_at_least_one_timed_call():
    with pytest.raises(ValueError, match="repeats"):
        time_split_read(CPU, block=4, head_dim=8, repeats=0)


# --- what is measured -----------------------------------------------------------------


def test_the_measured_ratio_is_unsplit_over_split(monkeypatch):
    clock = _Clock()
    _plant(monkeypatch, clock, [0.0, 0.006, 0.006, 0.006], [0.0, 0.002, 0.002, 0.002])
    report = time_split_read(CPU, block=4, head_dim=8, repeats=3, clock=clock)
    assert report.unsplit_ms == pytest.approx(6.0)
    assert report.split_ms == pytest.approx(2.0)
    assert report.measured == pytest.approx(3.0)
    assert report.pays


def test_a_split_slower_than_the_read_it_replaces_does_not_pay(monkeypatch):
    clock = _Clock()
    _plant(monkeypatch, clock, [0.0, 0.001], [0.0, 0.004])
    report = time_split_read(CPU, block=4, head_dim=8, repeats=1, clock=clock)
    assert report.measured == pytest.approx(0.25)
    assert not report.pays


def test_the_warm_call_is_not_timed(monkeypatch):
    """The first call on a card compiles both kernels, which takes seconds and is
    paid once per process. Timing it would grade the compiler, not the read."""
    clock = _Clock()
    _plant(monkeypatch, clock, [9.0, 0.004, 0.004], [30.0, 0.002, 0.002])
    report = time_split_read(CPU, block=4, head_dim=8, repeats=2, clock=clock)
    assert report.measured == pytest.approx(2.0)


def test_one_slow_call_does_not_move_the_median(monkeypatch):
    clock = _Clock()
    _plant(monkeypatch, clock, [0.0, 0.004, 0.004, 0.004],
           [0.0, 0.002, 0.5, 0.002])
    report = time_split_read(CPU, block=4, head_dim=8, repeats=3, clock=clock)
    assert report.split_ms == pytest.approx(2.0)


def test_a_split_that_disagrees_with_the_read_it_replaces_is_not_timed(monkeypatch):
    """A fast wrong answer is not a speedup. The probe graded a different case, so
    the timed case is checked on its own before its numbers are believed."""
    honest = flash_decoding.paged_attention_split
    monkeypatch.setattr(flash_decoding, "paged_attention_split",
                        lambda *a, **k: torch.zeros_like(honest(*a, **k)))
    with pytest.raises(SplitUnsound, match="timed"):
        time_split_read(CPU, block=4, head_dim=8, repeats=1)


def test_a_serial_backend_collects_the_work_and_never_the_wave():
    """The host finding. tlsim runs one program at a time, so it pays for every tile
    and every empty chunk and the reduce on top, and the predicted 4x does not show
    up. The prediction is about a resident grid; this box does not have one."""
    report = time_split_read(CPU, block=4, head_dim=8, n_rep=2, long_tiles=4,
                             repeats=3)
    assert report.backend == "tlsim"
    assert report.measured < report.predicted


def test_the_timing_report_says_what_it_measured():
    report = time_split_read(CPU, block=4, head_dim=8, repeats=1)
    d = report.as_dict()
    assert d["backend"] == "tlsim"
    assert set(d) >= {"predicted", "measured", "pays", "unsplit_ms", "split_ms",
                      "repeats", "rows", "splits", "dtype"}
    assert d["measured"] == pytest.approx(report.measured)
    assert "tlsim" in report.render() and "predicted" in report.render()


# --- the boot path --------------------------------------------------------------------


def test_timing_a_non_split_engine_is_a_no_op():
    assert measure_split_read(_engine(**GRAPHS)) is None


def test_the_boot_timing_uses_the_read_s_tile_and_the_model_s_heads():
    engine = _engine(split_read=True, **GRAPHS)
    report = measure_split_read(engine)
    cfg = _tiny_config()
    assert report.block == engine.cache.read.block
    assert report.head_dim == cfg.head_dim
    assert report.n_rep == cfg.num_attention_heads // cfg.num_key_value_heads
    assert report.dtype == torch.float32


def test_a_timed_read_that_disagrees_is_a_refused_boot_before_the_arena(monkeypatch):
    honest = flash_decoding.paged_attention_split
    monkeypatch.setattr(flash_decoding, "paged_attention_split",
                        lambda *a, **k: torch.zeros_like(honest(*a, **k)))
    engine = _engine(split_read=True, **GRAPHS)
    with pytest.raises(BootUnsound, match="timed"):
        measure_split_read(engine)
    assert engine.cache.read.workspace is None


def test_a_split_server_boots_with_its_timing_beside_its_probe():
    app = _app(split_read=True, **GRAPHS)
    assert app.state.split_timing.backend == app.state.split_probe.backend == "tlsim"
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace, app.state.split_probe, app.state.split_timing)
    assert info["split_timing"]["predicted"] >= 2.0
    check_boot_info(info)


def test_a_server_on_another_read_times_nothing():
    app = _app(**GRAPHS)
    assert app.state.split_timing is None


def _split_info():
    app = _app(split_read=True, **GRAPHS)
    return boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace, app.state.split_probe, app.state.split_timing)


def test_a_payload_that_timed_a_read_nobody_graded_is_refused():
    info = _split_info()
    del info["split_probe"]
    with pytest.raises(BootUnsound, match="graded"):
        check_boot_info(info)


def test_a_payload_that_timed_one_backend_and_graded_another_is_refused():
    info = _split_info()
    info["split_timing"]["backend"] = "triton"
    with pytest.raises(BootUnsound, match="backend"):
        check_boot_info(info)


# --- the card -------------------------------------------------------------------------


@requires_triton_gpu
def test_the_timing_runs_the_jitted_reads_on_a_card():
    report = time_split_read(torch.device("cuda"), block=16, head_dim=64, n_rep=4,
                             dtype=torch.float16, long_tiles=8)
    assert report.backend == "triton"
    assert report.predicted == 8.0
    assert report.split_ms > 0
