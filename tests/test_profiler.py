"""Day 46 tests: the decode step under a stopwatch, and where the seconds go.

Four tiers, because the module has four jobs and they fail in different ways.

  1. **One step's arithmetic.** A phase is a host time and the device time it
     launched, overhead is the difference, and a step is the sum of its phases.
     Nothing here is measured or estimated: every number is a subtraction over
     values placed by hand, which is what makes the aggregate worth reading.
  2. **The aggregate.** Warmup steps dropped, phases totalled across steps,
     hotspots ranked by overhead rather than by host time, and the per-token
     amortisation that says why the same overhead hurts a batch of one and not a
     batch of thirty-two.
  3. **The models, against closed forms.** Amdahl's law has an exact answer for
     every input, so `amdahl` is checked against the formula rather than against
     itself, including the two limits (infinite speedup of a fraction, finite
     speedup of the whole). The overlap models are the day's real content: under
     `overlapped`, deleting overhead that was already hidden behind the device buys
     exactly 1.0x, and the test asserts that equality rather than a range.
  4. **The gates, the plot and the bridge.** A profile whose phases do not add up
     to its steps, one that still contains its own warmup, one whose host timings
     do not contain the work they launched, and a caller claiming overlap on a step
     that synchronises. All four produce a believable hotspot table and a wrong
     conclusion, which is the only reason they are assertions.

Then two sections that are not tiers. The recorder is driven on a clock the test
advances by hand, so its durations are exact rather than approximate, which is the
only way a timing module gets tested at all. And the last section runs the real
`Engine` loop on a tiny random-weight model and asserts the phases it records are
the step it actually took: no `./weights` needed, and it is what stops the phase
names in `engine.py` and the ones this module reasons about from drifting apart.
"""

from __future__ import annotations

import math

import pytest

from nanoserve.curve import OperatingPoint, dominates
from nanoserve.profiler import (
    NULL_RECORDER,
    Phase,
    ProfileUnsound,
    StepProfile,
    StepRecorder,
    StepSample,
    amdahl,
    bar,
    check_accounted,
    check_attributable,
    check_device_timed,
    check_model_applies,
    check_warmed_up,
    device_timer_for,
    loop_overhead_s,
    project_point,
    recommended_model,
    render,
    speedup_if,
    step_time_under,
    step_with_compute,
)

MS = 1e-3

#: One synthetic decode step, in milliseconds, as (name, host, device, syncs).
#: The shape is the real one: `forward` owns nearly all the device time and a fifth
#: of the overhead, while `sample` owns almost nothing on the device and the largest
#: single block of host time, because that is where the tokens come back to Python.
DECODE = (
    ("schedule", 0.5, 0.0, False),
    ("sync_rows", 0.2, 0.0, False),
    ("build_inputs", 1.3, 0.1, False),
    ("forward", 5.0, 4.0, False),
    ("sample", 2.2, 0.4, True),
    ("collect", 0.8, 0.0, False),
)
# host 10.0ms, device 4.5ms, overhead 5.5ms per step.


def _phases(spec=DECODE, *, scale: float = 1.0) -> tuple[Phase, ...]:
    return tuple(
        Phase(name=n, host_s=h * MS * scale, device_s=d * MS * scale, syncs=s)
        for n, h, d, s in spec
    )


def _step(index: int = 0, *, batch_size: int = 4, scale: float = 1.0, **kw) -> StepSample:
    return StepSample(
        index=index, kind="decode", batch_size=batch_size, phases=_phases(scale=scale), **kw
    )


def _profile(steps: int = 8, *, warmup: int = 0, batch_size: int = 4) -> StepProfile:
    return StepProfile.from_samples(
        "decode",
        [_step(i, batch_size=batch_size) for i in range(steps)],
        warmup=warmup,
    )


# --- tier 1: one phase and one step ---------------------------------------------------


def test_overhead_is_host_minus_device():
    assert Phase("forward", 5.0 * MS, 4.0 * MS).overhead_s == pytest.approx(1.0 * MS)


def test_a_phase_that_launched_nothing_is_all_overhead():
    phase = Phase("schedule", 0.5 * MS, 0.0)
    assert phase.overhead_s == pytest.approx(phase.host_s)


def test_a_phase_cannot_have_negative_host_time():
    with pytest.raises(ValueError, match="negative"):
        Phase("forward", -1.0, 0.0)


def test_a_phase_cannot_have_negative_device_time():
    with pytest.raises(ValueError, match="negative"):
        Phase("forward", 1.0, -1.0)


def test_a_phase_needs_a_name():
    with pytest.raises(ValueError, match="name"):
        Phase("  ", 1.0, 0.0)


def test_a_phase_whose_device_exceeds_its_host_did_not_wait():
    assert Phase("forward", 0.1 * MS, 4.0 * MS).hidden


def test_a_phase_the_host_waited_for_is_not_hidden():
    assert not Phase("forward", 5.0 * MS, 4.0 * MS).hidden


def test_step_host_is_the_sum_of_its_phases():
    assert _step().host_s == pytest.approx(10.0 * MS)


def test_step_device_is_the_sum_of_its_phases():
    assert _step().device_s == pytest.approx(4.5 * MS)


def test_step_overhead_is_host_minus_device():
    assert _step().overhead_s == pytest.approx(5.5 * MS)


def test_step_utilisation_is_device_over_host():
    assert _step().device_utilisation == pytest.approx(0.45)


def test_a_step_with_more_overhead_than_device_is_overhead_bound():
    assert _step().bound == "overhead"


def test_a_step_with_more_device_than_overhead_is_compute_bound():
    heavy = StepSample(
        index=0,
        kind="decode",
        batch_size=4,
        phases=(Phase("forward", 50.0 * MS, 48.0 * MS),),
    )
    assert heavy.bound == "compute"


def test_an_unmeasured_wall_is_the_sum_of_the_phases():
    assert _step().wall == pytest.approx(_step().host_s)


def test_unaccounted_time_is_wall_minus_the_phases():
    step = _step(wall_s=12.0 * MS)
    assert step.unaccounted_s == pytest.approx(2.0 * MS)


def test_phases_cannot_sum_to_more_than_the_step():
    with pytest.raises(ValueError, match="more than the step"):
        _step(wall_s=9.0 * MS)


def test_a_step_needs_at_least_one_phase():
    with pytest.raises(ValueError, match="no phases"):
        StepSample(index=0, kind="decode", batch_size=1, phases=())


def test_a_step_needs_a_positive_batch():
    with pytest.raises(ValueError, match="batch"):
        StepSample(index=0, kind="decode", batch_size=0, phases=_phases())


def test_a_step_counts_its_sync_points():
    assert _step().sync_points == 1


def test_a_phase_appearing_twice_in_one_step_is_a_mistake():
    with pytest.raises(ValueError, match="twice"):
        StepSample(
            index=0,
            kind="decode",
            batch_size=1,
            phases=(Phase("forward", MS, 0.0), Phase("forward", MS, 0.0)),
        )


# --- tier 2: the aggregate ------------------------------------------------------------


def test_a_profile_totals_host_over_its_steps():
    assert _profile(8).total_host_s == pytest.approx(80.0 * MS)


def test_a_profile_totals_device_over_its_steps():
    assert _profile(8).total_device_s == pytest.approx(36.0 * MS)


def test_a_profile_totals_overhead_over_its_steps():
    assert _profile(8).total_overhead_s == pytest.approx(44.0 * MS)


def test_idle_fraction_is_the_share_of_the_step_the_device_sat_out():
    assert _profile(8).idle_fraction == pytest.approx(0.55)


def test_warmup_steps_are_dropped():
    profile = _profile(8, warmup=3)
    assert profile.steps == 5
    assert profile.dropped == 3


def test_dropping_warmup_drops_the_leading_steps_and_not_others():
    profile = _profile(8, warmup=3)
    assert [s.index for s in profile.samples] == [3, 4, 5, 6, 7]


def test_dropping_every_step_is_refused():
    with pytest.raises(ProfileUnsound, match="nothing left"):
        _profile(3, warmup=3)


def test_mean_step_time_is_the_wall_over_the_steps():
    assert _profile(8).mean_step_s == pytest.approx(10.0 * MS)


def test_steps_per_second_is_the_reciprocal_of_the_mean_step():
    assert _profile(8).steps_per_second == pytest.approx(100.0)


def test_tokens_are_one_per_row_per_step():
    assert _profile(8, batch_size=4).tokens == 32


def test_throughput_is_tokens_over_the_wall():
    assert _profile(8, batch_size=4).tokens_per_second == pytest.approx(400.0)


def test_overhead_per_token_divides_the_fixed_cost_across_the_batch():
    assert _profile(8, batch_size=4).overhead_per_token_s == pytest.approx(5.5 * MS / 4)


def test_overhead_per_token_falls_as_one_over_the_batch():
    small = _profile(4, batch_size=1).overhead_per_token_s
    large = _profile(4, batch_size=32).overhead_per_token_s
    assert small / large == pytest.approx(32.0)


def test_phase_totals_sum_each_phase_across_the_steps():
    totals = _profile(8).phase_totals()
    assert totals["forward"].host_s == pytest.approx(40.0 * MS)
    assert totals["forward"].device_s == pytest.approx(32.0 * MS)


def test_phase_totals_keep_execution_order():
    assert list(_profile(2).phase_totals()) == [name for name, _, _, _ in DECODE]


def test_hotspots_rank_by_overhead_not_by_host_time():
    names = [h.name for h in _profile(8).hotspots()]
    assert names[:3] == ["sample", "build_inputs", "forward"]


def test_the_biggest_host_phase_is_not_the_biggest_hotspot():
    profile = _profile(8)
    slowest = max(profile.phase_totals().values(), key=lambda p: p.host_s)
    assert slowest.name == "forward"
    assert profile.hotspots()[0].name == "sample"


def test_hotspot_shares_are_of_the_total_overhead():
    top = _profile(8).hotspots()[0]
    assert top.share == pytest.approx(1.8 / 5.5)


def test_hotspot_shares_sum_to_one():
    assert sum(h.share for h in _profile(8).hotspots()) == pytest.approx(1.0)


def test_the_last_cumulative_share_is_one():
    assert _profile(8).hotspots()[-1].cumulative_share == pytest.approx(1.0)


def test_the_top_three_hotspots_carry_their_cumulative_share():
    assert _profile(8).hotspots()[2].cumulative_share == pytest.approx(4.0 / 5.5)


def test_hotspots_can_be_limited():
    assert len(_profile(8).hotspots(limit=3)) == 3


def test_a_hotspot_reports_its_cost_per_step():
    assert _profile(8).hotspots()[0].per_step_s == pytest.approx(1.8 * MS)


def test_a_profile_of_mixed_kinds_can_be_narrowed_to_one():
    prefill = StepSample(
        index=0, kind="prefill", batch_size=2, phases=(Phase("forward", 40.0 * MS, 38.0 * MS),)
    )
    profile = StepProfile.from_samples("mixed", [prefill, _step(1), _step(2)])
    assert profile.select("decode").steps == 2
    assert profile.select("prefill").steps == 1


def test_narrowing_to_a_kind_that_is_not_there_is_refused():
    with pytest.raises(ProfileUnsound, match="no prefill"):
        _profile(4).select("prefill")


def test_an_empty_profile_is_refused():
    with pytest.raises(ProfileUnsound, match="no steps"):
        StepProfile.from_samples("empty", [])


def test_a_profile_is_overhead_bound_when_most_of_the_step_is_not_the_device():
    assert _profile(8).bound == "overhead"


# --- tier 3: the models ---------------------------------------------------------------


def test_amdahl_of_nothing_is_no_speedup():
    assert amdahl(0.0, 100.0) == pytest.approx(1.0)


def test_amdahl_of_everything_is_the_factor():
    assert amdahl(1.0, 8.0) == pytest.approx(8.0)


def test_amdahl_matches_its_closed_form():
    assert amdahl(0.95, 100.0) == pytest.approx(1.0 / (0.05 + 0.95 / 100.0))


def test_amdahl_with_an_infinite_factor_is_the_serial_ceiling():
    assert amdahl(0.55, math.inf) == pytest.approx(1.0 / 0.45)


def test_amdahl_of_everything_infinitely_fast_is_unbounded():
    assert math.isinf(amdahl(1.0, math.inf))


def test_amdahl_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError, match="fraction"):
        amdahl(1.5, 2.0)


def test_amdahl_rejects_a_slowdown_dressed_as_a_speedup():
    with pytest.raises(ValueError, match="factor"):
        amdahl(0.5, 0.0)


def test_the_serial_model_adds_the_two():
    assert step_time_under(5.5 * MS, 4.5 * MS, "serial") == pytest.approx(10.0 * MS)


def test_the_overlapped_model_takes_the_larger():
    assert step_time_under(5.5 * MS, 4.5 * MS, "overlapped") == pytest.approx(5.5 * MS)


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="model"):
        step_time_under(1.0, 1.0, "magic")


def test_the_serial_model_reproduces_the_measured_step():
    profile = _profile(8)
    modelled = step_time_under(profile.total_overhead_s, profile.total_device_s, "serial")
    assert modelled == pytest.approx(profile.total_host_s)


def test_deleting_all_overhead_serially_leaves_the_device_time():
    assert speedup_if(_profile(8), overhead_factor=math.inf) == pytest.approx(10.0 / 4.5)


def test_deleting_one_phase_serially_buys_exactly_its_overhead():
    assert speedup_if(_profile(8), eliminate=("sample",)) == pytest.approx(10.0 / 8.2)


def test_deleting_all_overhead_under_overlap_stops_at_the_device_time():
    assert speedup_if(
        _profile(8), overhead_factor=math.inf, model="overlapped"
    ) == pytest.approx(5.5 / 4.5)


def test_under_overlap_deleting_hidden_overhead_buys_nothing():
    # A step whose device time already covers all the host work: every second of
    # Python here is spent while the GPU is busy, so removing it changes no wall.
    hidden = StepProfile.from_samples(
        "hidden",
        [
            StepSample(
                index=0,
                kind="decode",
                batch_size=4,
                phases=(Phase("schedule", 1.0 * MS, 0.0), Phase("forward", 9.0 * MS, 9.0 * MS)),
            )
        ],
    )
    assert speedup_if(hidden, eliminate=("schedule",), model="overlapped") == pytest.approx(1.0)


def test_the_same_deletion_under_the_serial_model_does_buy_something():
    hidden = StepProfile.from_samples(
        "hidden",
        [
            StepSample(
                index=0,
                kind="decode",
                batch_size=4,
                phases=(Phase("schedule", 1.0 * MS, 0.0), Phase("forward", 9.0 * MS, 9.0 * MS)),
            )
        ],
    )
    assert speedup_if(hidden, eliminate=("schedule",)) == pytest.approx(10.0 / 9.0)


def test_a_finite_overhead_factor_only_shrinks_what_is_left():
    # Halve the overhead: 5.5ms becomes 2.75ms, device untouched.
    assert speedup_if(_profile(8), overhead_factor=2.0) == pytest.approx(10.0 / 7.25)


def test_a_device_factor_speeds_the_kernels_and_not_the_driver():
    assert speedup_if(_profile(8), device_factor=2.0) == pytest.approx(10.0 / 7.75)


def test_eliminating_a_phase_that_is_not_there_is_refused():
    with pytest.raises(ProfileUnsound, match="not in this profile"):
        speedup_if(_profile(8), eliminate=("cuda_graph",))


def test_the_ceiling_agrees_with_amdahl_on_the_overhead_fraction():
    profile = _profile(8)
    assert speedup_if(profile, overhead_factor=math.inf) == pytest.approx(
        amdahl(profile.idle_fraction, math.inf)
    )


def test_a_step_that_reads_tokens_back_cannot_be_modelled_as_overlapped():
    assert recommended_model(_profile(8)) == "serial"


def test_a_step_with_no_sync_can_overlap():
    quiet = StepProfile.from_samples(
        "quiet",
        [
            StepSample(
                index=0, kind="decode", batch_size=1, phases=(Phase("forward", MS, MS),)
            )
        ],
    )
    assert recommended_model(quiet) == "overlapped"


# --- tier 4: the gates, the plot and the bridge ---------------------------------------


def test_a_fully_accounted_profile_passes_the_gate():
    check_accounted(_profile(8))


def test_a_profile_with_a_hole_in_it_is_refused():
    leaky = StepProfile.from_samples(
        "leaky", [_step(i, wall_s=20.0 * MS) for i in range(4)]
    )
    with pytest.raises(ProfileUnsound, match="unaccounted"):
        check_accounted(leaky)


def test_a_small_hole_is_allowed_by_the_default_tolerance():
    check_accounted(StepProfile.from_samples("ok", [_step(i, wall_s=10.2 * MS) for i in range(4)]))


def test_a_profile_still_holding_its_warmup_is_refused():
    samples = [_step(0, scale=4.0)] + [_step(i) for i in range(1, 6)]
    with pytest.raises(ProfileUnsound, match="warmup"):
        check_warmed_up(StepProfile.from_samples("cold", samples))


def test_dropping_the_warmup_satisfies_the_gate():
    samples = [_step(0, scale=4.0)] + [_step(i) for i in range(1, 6)]
    check_warmed_up(StepProfile.from_samples("warm", samples, warmup=1))


def test_the_warmup_gate_needs_enough_steps_to_have_a_median():
    with pytest.raises(ProfileUnsound, match="too few"):
        check_warmed_up(_profile(2))


def test_host_timings_that_do_not_contain_their_kernels_are_refused():
    launched = StepProfile.from_samples(
        "async",
        [
            StepSample(
                index=0,
                kind="decode",
                batch_size=1,
                phases=(Phase("forward", 0.1 * MS, 4.0 * MS), Phase("sample", 4.0 * MS, 0.0)),
            )
        ],
    )
    with pytest.raises(ProfileUnsound, match="forward"):
        check_attributable(launched)


def test_a_synchronised_profile_is_attributable():
    check_attributable(_profile(4))


def test_claiming_overlap_on_a_step_that_synchronises_is_refused():
    with pytest.raises(ProfileUnsound, match="synchronis"):
        check_model_applies(_profile(4), "overlapped")


def test_the_serial_model_always_applies():
    check_model_applies(_profile(4), "serial")


def test_a_bar_draws_overhead_then_device():
    assert bar(2.0, 3.0, per_cell_s=1.0) == "##==="


def test_a_bar_is_blank_only_when_the_quantity_is_zero():
    assert bar(0.0, 0.0, per_cell_s=1.0) == ""
    assert bar(0.01, 0.0, per_cell_s=1.0) == "#"


def test_a_bar_is_clipped_to_its_width():
    assert len(bar(100.0, 100.0, per_cell_s=1.0, width=10)) == 10


def test_a_bar_needs_a_positive_cell():
    with pytest.raises(ValueError, match="per_cell_s"):
        bar(1.0, 1.0, per_cell_s=0.0)


def test_the_plot_names_every_phase_in_execution_order():
    lines = render(_profile(8)).splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("  phase"))
    rows = [line.split()[0] for line in lines[header + 1 :]]
    assert rows[: len(DECODE)] == [name for name, _, _, _ in DECODE]


def test_the_plot_reports_the_device_share():
    assert "45%" in render(_profile(8))


def test_the_plot_totals_the_step():
    assert "total" in render(_profile(8))


def test_a_projected_point_dominates_the_one_it_came_from():
    before = OperatingPoint(
        label="4 slots", knob="slots", knob_value=4, throughput_tps=4.01, latency_s=10.14
    )
    after = project_point(before, 2.0)
    assert dominates(after, before)


def test_a_projected_point_scales_both_axes_by_the_speedup():
    before = OperatingPoint(
        label="4 slots", knob="slots", knob_value=4, throughput_tps=4.0, latency_s=10.0
    )
    after = project_point(before, 2.0)
    assert after.throughput_tps == pytest.approx(8.0)
    assert after.latency_s == pytest.approx(5.0)


def test_a_projection_can_hold_the_latency_still():
    before = OperatingPoint(
        label="4 slots", knob="slots", knob_value=4, throughput_tps=4.0, latency_s=10.0
    )
    after = project_point(before, 2.0, latency_factor=1.0)
    assert after.latency_s == pytest.approx(10.0)


def test_a_projection_cannot_be_a_slowdown():
    before = OperatingPoint(
        label="4 slots", knob="slots", knob_value=4, throughput_tps=4.0, latency_s=10.0
    )
    with pytest.raises(ValueError, match="speedup"):
        project_point(before, 0.0)


# --- the recorder, on a clock that does not tick by itself ----------------------------


class FakeClock:
    """A stopwatch the test advances by hand, so the timings are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_the_recorder_times_a_phase_off_its_clock():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with recorder.step("decode", batch_size=2) as step:
        with step.phase("schedule"):
            clock.advance(0.5 * MS)
    sample = recorder.samples[0]
    assert sample.phases[0].host_s == pytest.approx(0.5 * MS)


def test_the_recorder_takes_a_device_time_from_the_caller():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with recorder.step("decode", batch_size=2) as step:
        with step.phase("forward") as phase:
            clock.advance(5.0 * MS)
            phase.device_s = 4.0 * MS
    assert recorder.samples[0].phases[0].overhead_s == pytest.approx(1.0 * MS)


def test_the_recorder_walls_the_whole_step_including_untimed_gaps():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with recorder.step("decode", batch_size=2) as step:
        with step.phase("schedule"):
            clock.advance(1.0 * MS)
        clock.advance(0.5 * MS)  # nobody's phase
    assert recorder.samples[0].unaccounted_s == pytest.approx(0.5 * MS)


def test_the_recorder_numbers_its_steps():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    for _ in range(3):
        with recorder.step("decode", batch_size=1) as step:
            with step.phase("forward"):
                clock.advance(MS)
    assert [s.index for s in recorder.samples] == [0, 1, 2]


def test_the_recorder_marks_a_phase_that_syncs():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with recorder.step("decode", batch_size=1) as step:
        with step.phase("sample", syncs=True):
            clock.advance(MS)
    assert recorder.samples[0].sync_points == 1


def test_the_recorder_builds_a_profile_and_drops_the_warmup():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    for _ in range(4):
        with recorder.step("decode", batch_size=1) as step:
            with step.phase("forward"):
                clock.advance(MS)
    assert recorder.profile("decode", warmup=1).steps == 3


def test_a_step_that_raised_is_not_recorded():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with pytest.raises(RuntimeError):
        with recorder.step("decode", batch_size=1) as step:
            with step.phase("forward"):
                clock.advance(MS)
            raise RuntimeError("boom")
    assert recorder.samples == ()


def test_the_recorder_lets_a_step_name_itself_once_it_knows():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with recorder.step() as step:
        with step.phase("schedule"):
            clock.advance(MS)
        step.describe("decode", 6)
    sample = recorder.samples[0]
    assert (sample.kind, sample.batch_size) == ("decode", 6)


def test_a_dropped_step_is_not_recorded():
    clock = FakeClock()
    recorder = StepRecorder(clock=clock)
    with recorder.step() as step:
        with step.phase("schedule"):
            clock.advance(MS)
        step.drop()
    assert recorder.samples == ()


def test_a_step_with_no_phases_records_nothing():
    recorder = StepRecorder(clock=FakeClock())
    with recorder.step("decode", batch_size=1):
        pass
    assert recorder.samples == ()


def test_the_recorder_brackets_a_device_phase_with_its_timer():
    class FakeTimer:
        def __init__(self) -> None:
            self.started = 0

        def start(self) -> None:
            self.started += 1

        def stop(self) -> float:
            return 4.0 * MS

    clock, timer = FakeClock(), FakeTimer()
    recorder = StepRecorder(clock=clock, device_timer=timer)
    with recorder.step("decode", batch_size=1) as step:
        with step.phase("forward", device=True):
            clock.advance(5.0 * MS)
        with step.phase("collect"):
            clock.advance(MS)
    assert timer.started == 1
    assert recorder.samples[0].phases[0].device_s == pytest.approx(4.0 * MS)
    assert recorder.samples[0].phases[1].device_s == 0.0


def test_a_cpu_device_has_no_separate_clock():
    assert device_timer_for("cpu") is None


def test_a_profile_with_no_device_times_cannot_price_anything():
    host_only = StepProfile.from_samples(
        "cpu", [StepSample(index=i, kind="decode", batch_size=1, phases=(Phase("f", MS),))
                for i in range(3)]
    )
    with pytest.raises(ProfileUnsound, match="no phase recorded any device time"):
        check_device_timed(host_only)


def test_a_profile_with_device_times_passes_that_gate():
    check_device_timed(_profile(4))


# --- the off switch an uninstrumented engine carries ---------------------------------


def test_the_null_recorder_has_the_shape_of_the_real_one():
    with NULL_RECORDER.step("decode", batch_size=4) as step:
        with step.phase("forward", syncs=True, device=True) as phase:
            phase.device_s = 1.0
        step.describe("decode", 4)
        step.drop()


def test_the_null_recorder_keeps_nothing():
    assert not hasattr(NULL_RECORDER, "samples")


# --- the engine loop, instrumented ----------------------------------------------------


def _tiny_engine():
    """The same tiny random-weight model the engine tests use, so no ./weights."""
    import torch

    from nanoserve.config import ModelConfig
    from nanoserve.engine import Engine
    from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
    from nanoserve.model import LlamaModel

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    model = LlamaModel(cfg, Weights(tensors, cfg))
    return Engine.build(model, num_blocks=64, block_size=4, max_batch_size=4)


def _run_profiled(max_new_tokens: int = 5):
    from nanoserve.scheduler import Request

    engine = _tiny_engine()
    engine.recorder = StepRecorder()
    for i, prompt in enumerate([[1, 2, 3], [4, 5], [6]]):
        engine.add_request(
            Request(
                request_id=f"r{i}", prompt_token_ids=prompt, max_new_tokens=max_new_tokens
            )
        )
    engine.run_to_completion()
    return engine


def test_an_engine_is_not_profiled_until_it_is_given_a_recorder():
    assert _tiny_engine().recorder is NULL_RECORDER


def test_an_instrumented_engine_records_one_sample_per_step():
    engine = _run_profiled()
    assert len(engine.recorder.samples) == engine.iterations


def test_the_recorded_phases_are_the_decode_step_in_order():
    profile = _run_profiled().recorder.profile("engine").select("decode")
    assert list(profile.phase_totals()) == [
        "schedule",
        "sync_rows",
        "build_inputs",
        "forward",
        "sample",
        "collect",
    ]


def test_the_recorded_batch_size_is_the_rows_that_stepped():
    engine = _run_profiled()
    assert engine.recorder.profile("engine").tokens == engine.issued_tokens


def test_the_engine_marks_exactly_one_phase_as_the_step_s_sync_point():
    """On Day 46 that phase was `sample`; on Day 47 it is `collect`.

    The count is what the overlap model reads, and it did not change: the tokens
    stopped coming home one row at a time, but a stop rule still needs a Python
    int, so the host still stops once a step and `serial` is still the model that
    applies. `nanoserve.output` is where the move is measured.
    """
    profile = _run_profiled().recorder.profile("engine").select("decode")
    assert profile.sync_points == 1
    syncing = [p.name for p in profile.samples[0].phases if p.syncs]
    assert syncing == ["collect"]
    assert recommended_model(profile) == "serial"


def test_a_profiled_engine_step_accounts_for_nearly_all_of_its_own_wall():
    check_accounted(_run_profiled().recorder.profile("engine").select("decode"))


def test_a_cpu_profile_of_the_engine_cannot_price_an_optimisation():
    profile = _run_profiled().recorder.profile("engine").select("decode")
    with pytest.raises(ProfileUnsound, match="device time"):
        check_device_timed(profile)


# --- what a profile with no device in it can still say --------------------------------


def test_the_loop_is_everything_that_is_not_the_forward():
    assert loop_overhead_s(_profile(8)) == pytest.approx(5.0 * MS)


def test_the_loop_can_be_told_which_phases_are_arithmetic():
    assert loop_overhead_s(_profile(8), compute=("forward", "sample")) == pytest.approx(2.8 * MS)


def test_separating_the_loop_needs_the_compute_phase_to_exist():
    with pytest.raises(ProfileUnsound, match="not a phase"):
        loop_overhead_s(_profile(8), compute=("matmul",))


def test_substituting_a_faster_forward_keeps_the_loop():
    assert step_with_compute(_profile(8), 2.0 * MS) == pytest.approx(7.0 * MS)


def test_a_free_forward_leaves_exactly_the_loop():
    assert step_with_compute(_profile(8), 0.0) == pytest.approx(loop_overhead_s(_profile(8)))


def test_a_negative_forward_is_refused():
    with pytest.raises(ValueError, match="not negative"):
        step_with_compute(_profile(8), -1.0)
