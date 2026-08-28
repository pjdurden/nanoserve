"""Day 45 tests: the latency-versus-throughput curve, and the knee as arithmetic.

Four tiers, because the module has four jobs and they fail in different ways.

  1. **The arithmetic**, over points placed by hand on the two axes. Power is
     throughput over latency, dominance is a pair of inequalities, and the frontier
     and the knee are both defined in terms of those two and nothing else. No
     smoothing, no fitting, no eyeballing: every answer here is a comparison.
  2. **The analytic check.** An M/M/1 queue has a closed-form latency curve whose
     power is maximised at exactly half the service rate, so the knee finder can be
     pointed at a curve whose answer is known in advance rather than only at
     measurements whose answer is whatever the code says.
  3. **The gate.** A sweep that never bent backwards did not find the wall, two
     curves with different y quantities may not share a pair of axes, and a curve
     with a repeated knob setting is two runs of one experiment rather than a
     sweep. All three produce a plausible picture and a wrong reading.
  4. **The wiring and the plot**, over Day 42's `LoadReport` and Day 44's
     `SystemRun` built by hand, then the ASCII renderer, whose only real content is
     the cell arithmetic and the axes it refuses to truncate.
"""

from __future__ import annotations

import math

import pytest

from nanoserve.baseline import RunRecord, SystemRun
from nanoserve.curve import (
    CurveUnsound,
    OperatingPoint,
    TradeoffCurve,
    capacity_under_slo,
    cell_for,
    check_same_axes,
    check_swept_to_saturation,
    dominates,
    point_from_load_report,
    point_from_run,
    render,
)
from nanoserve.servebench import LoadReport, RequestRecord


def _pt(knob_value: float, tps: float, latency: float, *, knob: str = "rps") -> OperatingPoint:
    return OperatingPoint(
        label=f"{knob_value:g}",
        knob=knob,
        knob_value=knob_value,
        throughput_tps=tps,
        latency_s=latency,
    )


def _curve(*triples: tuple[float, float, float], name: str = "sweep") -> TradeoffCurve:
    return TradeoffCurve.from_points(
        name,
        [_pt(k, t, lat) for k, t, lat in triples],
        latency_name="ttft_p50",
    )


# --- tier 1: the arithmetic on one point and one pair ----------------------------------


def test_power_is_throughput_over_latency():
    assert _pt(1.0, 10.0, 2.0).power == pytest.approx(5.0)


def test_a_point_cannot_have_zero_latency():
    with pytest.raises(ValueError, match="latency"):
        _pt(1.0, 10.0, 0.0)


def test_a_point_cannot_have_negative_throughput():
    with pytest.raises(ValueError, match="throughput"):
        _pt(1.0, -1.0, 2.0)


def test_a_point_dominates_one_that_is_worse_on_both_axes():
    better = _pt(2.0, 10.0, 1.0)
    worse = _pt(4.0, 8.0, 3.0)
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_more_throughput_at_the_same_latency_dominates():
    assert dominates(_pt(2.0, 10.0, 1.0), _pt(4.0, 9.0, 1.0))


def test_the_same_throughput_at_lower_latency_dominates():
    assert dominates(_pt(2.0, 10.0, 1.0), _pt(4.0, 10.0, 2.0))


def test_a_point_does_not_dominate_itself():
    point = _pt(2.0, 10.0, 1.0)
    assert not dominates(point, point)


def test_neither_point_dominates_when_one_trades_latency_for_throughput():
    faster = _pt(4.0, 12.0, 5.0)
    snappier = _pt(2.0, 10.0, 1.0)
    assert not dominates(faster, snappier)
    assert not dominates(snappier, faster)


# --- tier 1: the curve ------------------------------------------------------------------


def test_a_curve_is_sorted_by_its_knob_whatever_order_it_was_given_in():
    curve = TradeoffCurve.from_points(
        "sweep",
        [_pt(4.0, 2.0, 9.0), _pt(1.0, 1.0, 2.0), _pt(2.0, 1.8, 4.0)],
        latency_name="ttft_p50",
    )
    assert [p.knob_value for p in curve.points] == [1.0, 2.0, 4.0]


def test_the_peak_is_the_highest_throughput_and_not_the_last_point():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    assert curve.peak.knob_value == 2.0
    assert curve.max_throughput == pytest.approx(2.5)


def test_the_knee_maximises_power():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.6, 20.0))
    assert [round(p.power, 4) for p in curve.points] == [0.5, 0.625, 0.13]
    assert curve.knee.knob_value == 2.0


def test_the_knee_is_the_first_of_two_settings_that_tie_on_power():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.0, 4.0), (4.0, 1.5, 9.0))
    assert curve.knee.knob_value == 1.0


def test_the_frontier_drops_a_point_that_is_worse_on_both_axes():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    assert [p.knob_value for p in curve.frontier] == [1.0, 2.0]
    assert [p.knob_value for p in curve.dominated] == [4.0]


def test_the_frontier_keeps_a_point_that_only_buys_latency():
    curve = _curve((1.0, 1.0, 1.0), (2.0, 2.5, 4.0), (4.0, 2.0, 2.0))
    assert [p.knob_value for p in curve.frontier] == [1.0, 2.0, 4.0]
    assert curve.dominated == ()


def test_every_point_is_on_the_frontier_or_has_a_dominator():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0), (8.0, 1.2, 30.0))
    on = {p.knob_value for p in curve.frontier}
    off = {p.knob_value for p in curve.dominated}
    assert on & off == set()
    assert on | off == {p.knob_value for p in curve.points}


def test_the_knee_is_always_on_the_frontier():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.6, 20.0), (8.0, 2.0, 40.0))
    assert curve.knee in curve.frontier


def test_the_peak_is_always_on_the_frontier():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    assert curve.peak in curve.frontier


def test_a_knee_in_the_middle_of_the_sweep_is_bracketed():
    assert _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.6, 20.0)).knee_bracketed


def test_a_knee_on_the_first_setting_was_never_bracketed():
    curve = _curve((1.0, 1.0, 1.0), (2.0, 2.5, 8.0), (4.0, 2.6, 20.0))
    assert curve.knee is curve.points[0]
    assert not curve.knee_bracketed
    assert "first setting swept" in curve.summary()


def test_a_knee_on_the_last_setting_was_never_bracketed_either():
    curve = _curve((1.0, 1.0, 9.0), (2.0, 2.0, 6.0), (4.0, 2.6, 2.0))
    assert curve.knee is curve.points[-1]
    assert not curve.knee_bracketed
    assert "last setting swept" in curve.summary()


def test_a_curve_that_bent_backwards_found_the_wall():
    assert _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0)).found_the_wall


def test_a_curve_that_only_ever_climbed_did_not():
    assert not _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 3.0, 9.0)).found_the_wall


# --- tier 1: the segments between settings ----------------------------------------------


def test_a_segment_reports_what_the_next_setting_cost_and_bought():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0))
    (seg,) = curve.segments
    assert seg.d_throughput == pytest.approx(1.5)
    assert seg.d_latency == pytest.approx(2.0)
    assert seg.latency_per_tps == pytest.approx(2.0 / 1.5)
    assert not seg.backward


def test_a_backward_segment_has_an_infinite_price():
    curve = _curve((1.0, 2.5, 4.0), (2.0, 2.0, 9.0))
    (seg,) = curve.segments
    assert seg.backward
    assert seg.latency_per_tps == math.inf


def test_a_step_that_also_gave_latency_back_is_free_rather_than_cheap():
    curve = _curve((1.0, 1.0, 9.0), (2.0, 2.5, 4.0))
    (seg,) = curve.segments
    assert seg.free
    assert not seg.backward
    assert dominates(seg.hi, seg.lo)


def test_a_step_that_bought_throughput_with_latency_is_not_free():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0))
    assert not curve.segments[0].free


def test_a_backward_step_is_not_free_however_much_latency_it_gave_back():
    curve = _curve((1.0, 2.5, 9.0), (2.0, 2.0, 1.0))
    (seg,) = curve.segments
    assert not seg.free
    assert seg.backward


def test_a_free_step_is_priced_as_free_in_the_summary():
    assert "free" in _curve((1.0, 1.0, 9.0), (2.0, 2.5, 4.0)).summary()


def test_the_segments_telescope_back_to_the_ends_of_the_curve():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    assert sum(s.d_throughput for s in curve.segments) == pytest.approx(1.0)
    assert sum(s.d_latency for s in curve.segments) == pytest.approx(7.0)


def test_a_single_point_has_no_segments_but_still_needs_two_to_be_a_curve():
    with pytest.raises(CurveUnsound, match="one point"):
        TradeoffCurve.from_points("sweep", [_pt(1.0, 1.0, 2.0)], latency_name="ttft_p50")


# --- tier 1: the SLO question -----------------------------------------------------------


def test_capacity_under_an_slo_is_the_fastest_setting_that_still_meets_it():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 3.0, 9.0))
    assert capacity_under_slo(curve, 5.0).knob_value == 2.0


def test_capacity_under_a_generous_slo_is_the_peak():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 3.0, 9.0))
    assert capacity_under_slo(curve, 60.0) is curve.peak


def test_no_setting_meets_an_impossible_slo():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0))
    assert capacity_under_slo(curve, 0.5) is None


def test_capacity_under_an_slo_is_on_the_frontier():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.5, 9.0), (8.0, 2.0, 30.0))
    for slo in (2.0, 4.5, 9.5, 100.0):
        found = capacity_under_slo(curve, slo)
        assert found is None or found in curve.frontier


def test_capacity_prefers_the_snappier_of_two_settings_with_the_same_throughput():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.5, 9.0))
    assert capacity_under_slo(curve, 20.0).knob_value == 2.0


# --- tier 2: the curve whose answer is known in advance ---------------------------------


def _mm1(service_rate: float = 1.0) -> TradeoffCurve:
    """M/M/1 at utilisations 0.1 to 0.9: throughput is lambda, latency is 1/(mu-lambda)."""
    points = []
    for i in range(1, 10):
        rho = i / 10.0
        lam = rho * service_rate
        points.append(
            OperatingPoint(
                label=f"rho={rho:.1f}",
                knob="rho",
                knob_value=rho,
                throughput_tps=lam,
                latency_s=1.0 / (service_rate - lam),
            )
        )
    return TradeoffCurve.from_points("m/m/1", points, latency_name="sojourn")


def test_the_knee_of_an_mm1_queue_is_at_half_the_service_rate():
    assert _mm1().knee.knob_value == pytest.approx(0.5)


def test_the_mm1_knee_does_not_move_when_the_server_gets_faster():
    assert _mm1(service_rate=8.0).knee.knob_value == pytest.approx(0.5)


def test_the_mm1_knee_is_bracketed_by_the_utilisations_either_side_of_it():
    assert _mm1().knee_bracketed


def test_an_mm1_queue_has_no_dominated_points_because_it_never_saturates():
    assert _mm1().dominated == ()
    assert not _mm1().found_the_wall


def test_the_mm1_peak_is_the_last_point_and_is_not_the_knee():
    curve = _mm1()
    assert curve.peak.knob_value == pytest.approx(0.9)
    assert curve.peak is not curve.knee


# --- tier 3: the gate -------------------------------------------------------------------


def test_check_swept_to_saturation_accepts_a_curve_that_bent_back():
    check_swept_to_saturation(_curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0)))


def test_check_swept_to_saturation_refuses_a_sweep_that_stopped_early():
    with pytest.raises(CurveUnsound, match="never bent back"):
        check_swept_to_saturation(_curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 3.0, 9.0)))


def test_a_repeated_knob_setting_is_not_a_sweep():
    with pytest.raises(CurveUnsound, match="twice"):
        TradeoffCurve.from_points(
            "sweep",
            [_pt(2.0, 1.0, 2.0), _pt(2.0, 2.5, 4.0)],
            latency_name="ttft_p50",
        )


def test_a_curve_cannot_mix_two_knobs():
    with pytest.raises(CurveUnsound, match="knob"):
        TradeoffCurve.from_points(
            "sweep",
            [_pt(1.0, 1.0, 2.0), _pt(2.0, 2.5, 4.0, knob="slots")],
            latency_name="ttft_p50",
        )


def test_a_point_that_delivered_nothing_is_not_an_operating_point():
    with pytest.raises(CurveUnsound, match="no throughput"):
        TradeoffCurve.from_points(
            "sweep",
            [_pt(1.0, 0.0, 2.0), _pt(2.0, 2.5, 4.0)],
            latency_name="ttft_p50",
        )


def test_two_curves_measuring_the_same_latency_share_a_pair_of_axes():
    check_same_axes([_curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0)), _curve((1.0, 2.0, 3.0), (4.0, 3.0, 8.0))])


def test_a_ttft_curve_and_an_end_to_end_curve_may_not_share_a_pair_of_axes():
    ttft = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0))
    e2e = TradeoffCurve.from_points(
        "other",
        [_pt(1.0, 1.0, 12.0), _pt(2.0, 2.5, 16.0)],
        latency_name="e2e_p50",
    )
    with pytest.raises(CurveUnsound, match="e2e_p50"):
        check_same_axes([ttft, e2e])


# --- tier 4: the adapters ---------------------------------------------------------------


def _load_report(rate: float, *, ttfts: list[float], tokens: int, span: float) -> LoadReport:
    records = []
    for i, ttft in enumerate(ttfts):
        records.append(
            RequestRecord(
                client_id=f"c{i}",
                scheduled_at=0.0,
                sent_at=0.0,
                frame_times=(ttft, ttft + 1.0),
                done_at=ttft + 1.0,
                prompt_tokens=4,
                output_tokens=tokens,
                status=200,
            )
        )
    return LoadReport(
        records=tuple(records), started_at=0.0, finished_at=span, target_rate=rate
    )


def test_a_load_report_becomes_a_point_on_the_throughput_ttft_axes():
    report = _load_report(0.4, ttfts=[1.0, 3.0, 5.0], tokens=10, span=10.0)
    point = point_from_load_report(report)
    assert point.knob == "offered_rps"
    assert point.knob_value == pytest.approx(0.4)
    assert point.throughput_tps == pytest.approx(3.0)
    assert point.latency_s == pytest.approx(3.0)


def test_a_load_report_can_be_read_on_the_end_to_end_axis_instead():
    report = _load_report(0.4, ttfts=[1.0, 3.0, 5.0], tokens=10, span=10.0)
    point = point_from_load_report(report, latency="e2e_p50")
    assert point.latency_s == pytest.approx(4.0)


def test_an_unknown_latency_name_is_refused_rather_than_guessed():
    report = _load_report(0.4, ttfts=[1.0], tokens=10, span=10.0)
    with pytest.raises(CurveUnsound, match="ttft_p50"):
        point_from_load_report(report, latency="p50")


def test_a_closed_loop_report_has_no_offered_rate_to_put_on_the_knob():
    report = LoadReport(records=(), started_at=0.0, finished_at=1.0, concurrency=4)
    with pytest.raises(CurveUnsound, match="knob"):
        point_from_load_report(report)


def _run(name: str, *, n: int, span: float, tokens: int) -> SystemRun:
    records = tuple(
        RunRecord(
            request_id=f"r{i}",
            prompt_tokens=4,
            output_token_ids=tuple(range(tokens)),
            started_s=0.0,
            finished_s=span,
            first_token_s=span / 2.0,
        )
        for i in range(n)
    )
    return SystemRun.from_records(name, records)


def test_a_system_run_becomes_a_point_with_the_slot_count_as_its_knob():
    run = _run("nanoserve", n=4, span=8.0, tokens=16)
    point = point_from_run(run, knob_value=4)
    assert point.knob == "slots"
    assert point.knob_value == 4
    assert point.throughput_tps == pytest.approx(64.0 / 8.0)
    assert point.latency_s == pytest.approx(8.0)


def test_a_system_run_can_be_read_on_the_ttft_axis_instead():
    run = _run("nanoserve", n=4, span=8.0, tokens=16)
    assert point_from_run(run, knob_value=4, latency="mean_ttft").latency_s == pytest.approx(4.0)


def test_the_two_sweeps_land_on_one_pair_of_axes():
    rate = TradeoffCurve.from_points(
        "rate sweep",
        [
            point_from_load_report(_load_report(0.1, ttfts=[2.0], tokens=10, span=10.0)),
            point_from_load_report(_load_report(0.4, ttfts=[9.0], tokens=10, span=10.0)),
        ],
        latency_name="ttft_p50",
    )
    slots = TradeoffCurve.from_points(
        "slot sweep",
        [
            point_from_run(_run("a", n=1, span=8.0, tokens=16), knob_value=1, latency="mean_ttft"),
            point_from_run(_run("b", n=4, span=8.0, tokens=16), knob_value=4, latency="mean_ttft"),
        ],
        latency_name="ttft_p50",
    )
    check_same_axes([rate, slots])
    assert rate.knob != slots.knob


# --- tier 4: the plot -------------------------------------------------------------------


def test_the_origin_is_the_bottom_left_cell():
    assert cell_for(_pt(1.0, 0.001, 0.001), width=40, height=10, max_tps=10.0, max_latency=5.0) == (
        9,
        0,
    )


def test_the_busiest_point_is_the_top_right_cell():
    assert cell_for(_pt(1.0, 10.0, 5.0), width=40, height=10, max_tps=10.0, max_latency=5.0) == (0, 39)


def test_a_point_at_half_of_both_axes_lands_in_the_middle():
    row, col = cell_for(_pt(1.0, 5.0, 2.5), width=41, height=11, max_tps=10.0, max_latency=5.0)
    assert (row, col) == (5, 20)


def test_the_axes_start_at_zero_so_a_small_difference_stays_small():
    near = cell_for(_pt(1.0, 9.5, 1.0), width=41, height=11, max_tps=10.0, max_latency=5.0)
    far = cell_for(_pt(2.0, 10.0, 1.0), width=41, height=11, max_tps=10.0, max_latency=5.0)
    assert far[1] - near[1] <= 2


def test_the_plot_draws_a_marker_for_every_point_and_names_the_axes():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    text = render([curve], width=40, height=12)
    assert "ttft_p50" in text
    assert "output tok/s" in text
    assert text.count(curve.marker) >= 3
    assert "sweep" in text


def test_the_plot_marks_the_knee_apart_from_the_other_points():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    text = render([curve], width=40, height=12)
    assert "knee" in text
    assert "#" in text


def test_the_plot_refuses_to_overlay_two_different_latencies():
    ttft = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0))
    e2e = TradeoffCurve.from_points(
        "other", [_pt(1.0, 1.0, 12.0), _pt(2.0, 2.5, 16.0)], latency_name="e2e_p50"
    )
    with pytest.raises(CurveUnsound, match="e2e_p50"):
        render([ttft, e2e], width=40, height=12)


def test_two_curves_on_one_plot_get_different_markers():
    a = TradeoffCurve.from_points(
        "a", [_pt(1.0, 1.0, 2.0), _pt(2.0, 2.5, 4.0)], latency_name="ttft_p50", marker="o"
    )
    b = TradeoffCurve.from_points(
        "b", [_pt(1.0, 2.0, 3.0), _pt(2.0, 3.5, 6.0)], latency_name="ttft_p50", marker="x"
    )
    text = render([a, b], width=40, height=12)
    assert "o" in text and "x" in text


def test_an_slo_line_is_drawn_when_one_is_given():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    text = render([curve], width=40, height=12, slo_s=5.0)
    assert "slo" in text.lower()


def test_rendering_nothing_is_refused():
    with pytest.raises(CurveUnsound, match="no curves"):
        render([], width=40, height=12)


# --- tier 4: the summary a report prints -------------------------------------------------


def test_the_summary_names_the_knee_the_peak_and_the_wall():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 2.0, 9.0))
    text = curve.summary()
    assert "knee" in text
    assert "peak" in text
    assert "dominated" in text


def test_the_summary_says_so_when_the_sweep_never_found_the_wall():
    curve = _curve((1.0, 1.0, 2.0), (2.0, 2.5, 4.0), (4.0, 3.0, 9.0))
    assert "no wall" in curve.summary()
