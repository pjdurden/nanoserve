"""Day 29: pure tests for the static-batching measurement core.

No model, no torch: a fake clock feeds `time_batched_decode` scripted timestamps
and plain callables stand in for the prefill and the decode step, so every number
the sweep will quote (goodput, issued throughput, the waste fraction, the
head-of-line delay, the scaling efficiency) is checked exactly rather than
approximately. Same discipline as the Day-13 benchmark and the Day-20 read
benchmark: a measurement whose own arithmetic is unverified is just a confident
guess, and this one exists to say what static batching buys and where it stops
paying, which is the argument Week 8's scheduler is built on.

The distinction the whole file turns on: an *issued* token is one the forward
computed, a *useful* token is one a row actually collected. A static batch issues
`batch_size` tokens every step until its slowest row finishes, so those two counts
come apart the moment the rows are ragged, and quoting the first as throughput is
the flattering lie this module exists to refuse.
"""

from __future__ import annotations

import pytest

from nanoserve.batchbench import (
    BatchScaling,
    BatchTiming,
    fit_batch_scaling,
    time_batched_decode,
)


class FakeClock:
    """Returns a scripted, strictly increasing sequence of times.

    `time_batched_decode` reads the clock exactly twice for the prefill (before and
    after) and twice per decode step, so a run of S steps needs exactly 2 + 2*S
    ticks. Each call pops the next value.
    """

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


def _timing(**overrides) -> BatchTiming:
    """A hand-built timing: 3 rows, prefill 1.0s, four 0.5s steps, ragged finishes."""
    kwargs = dict(
        batch_size=3,
        prefill_s=1.0,
        step_s=[0.5, 0.5, 0.5, 0.5],
        finished_at=[1, 4, 2],
    )
    kwargs.update(overrides)
    return BatchTiming(**kwargs)


# --- the timing driver: what the clock saw -----------------------------------


def test_prefill_and_each_step_are_timed_separately():
    # prefill 10.0->11.0, then steps (11.0->11.5), (11.5->12.1), (12.1->12.4).
    clock = FakeClock([10.0, 11.0, 11.0, 11.5, 11.5, 12.1, 12.1, 12.4])
    timing = time_batched_decode(
        lambda: [False, False],
        lambda: [False, False],
        batch_size=2,
        max_steps=3,
        clock=clock,
    )

    assert timing.prefill_s == pytest.approx(1.0)
    assert timing.step_s == pytest.approx([0.5, 0.6, 0.3])
    assert timing.n_steps == 3
    assert timing.decode_s == pytest.approx(1.4)
    assert timing.total_s == pytest.approx(2.4)


def test_a_row_is_recorded_at_the_step_it_first_reported_done():
    """finished_at is the row's own step count, not the batch's."""
    dones = [[False, False, False], [True, False, False], [True, False, True]]
    clock = FakeClock([0.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5])
    timing = time_batched_decode(
        lambda: [False, False, False],
        lambda: dones.pop(0),
        batch_size=3,
        max_steps=3,
        clock=clock,
    )

    # Row 0 stopped after step 2, row 2 after step 3, row 1 never did (capped).
    assert timing.finished_at == [2, 3, 3]


def test_a_row_done_at_the_prefill_needs_zero_decode_steps():
    clock = FakeClock([0.0, 1.0, 1.0, 1.5])
    timing = time_batched_decode(
        lambda: [True, False],
        lambda: [True, True],
        batch_size=2,
        max_steps=4,
        clock=clock,
    )

    assert timing.finished_at == [0, 1]


def test_the_loop_stops_when_every_row_is_done():
    """Static means the batch runs until its *slowest* row finishes, then stops."""
    steps = {"n": 0}

    def step():
        steps["n"] += 1
        return [True, True]

    clock = FakeClock([0.0, 1.0, 1.0, 1.5])
    timing = time_batched_decode(
        lambda: [False, False], step, batch_size=2, max_steps=50, clock=clock
    )

    assert steps["n"] == 1  # not 50: the cap is a cap, not a schedule
    assert timing.n_steps == 1


def test_all_done_at_the_prefill_runs_no_decode_step_at_all():
    steps = {"n": 0}

    def step():
        steps["n"] += 1
        return [True]

    clock = FakeClock([0.0, 1.0])
    timing = time_batched_decode(lambda: [True], step, batch_size=1, max_steps=9, clock=clock)

    assert steps["n"] == 0
    assert timing.n_steps == 0
    assert timing.decode_s == 0.0
    assert timing.total_s == pytest.approx(1.0)
    # No decode step happened, so there is no rate to report rather than a division.
    assert timing.goodput_tps == 0.0
    assert timing.issued_tps == 0.0


def test_a_step_that_reports_the_wrong_number_of_rows_is_an_error():
    """A done vector shorter than the batch would silently drop a row's finish."""
    clock = FakeClock([0.0, 1.0, 1.0, 1.5])
    with pytest.raises(ValueError, match="one done flag per row"):
        time_batched_decode(
            lambda: [False, False],
            lambda: [True],
            batch_size=2,
            max_steps=2,
            clock=clock,
        )


def test_max_steps_must_be_non_negative():
    with pytest.raises(ValueError, match="max_steps"):
        time_batched_decode(lambda: [False], lambda: [True], batch_size=1, max_steps=-1)


# --- issued versus useful: the waste a static batch pays ---------------------


def test_issued_counts_every_row_every_step_and_useful_counts_only_collected():
    timing = _timing()  # 3 rows, 4 steps, finishes at 1, 4, 2

    assert timing.issued_tokens == 12  # 3 rows * 4 steps, the forward's real work
    assert timing.useful_tokens == 7  # 1 + 4 + 2, the tokens anyone asked for
    assert timing.wasted_tokens == 5


def test_waste_fraction_is_the_share_of_the_forward_nobody_wanted():
    timing = _timing()

    assert timing.waste_fraction == pytest.approx(5 / 12)


def test_a_batch_of_equal_length_rows_wastes_nothing():
    """The waste is raggedness, not batching: same lengths, no head-of-line cost."""
    timing = _timing(finished_at=[4, 4, 4])

    assert timing.wasted_tokens == 0
    assert timing.waste_fraction == 0.0
    assert timing.max_hol_delay_s == 0.0


def test_goodput_and_issued_throughput_come_apart_on_a_ragged_batch():
    timing = _timing()  # decode_s = 2.0

    assert timing.goodput_tps == pytest.approx(7 / 2.0)
    assert timing.issued_tps == pytest.approx(12 / 2.0)
    # The flattering number is 1.7x the honest one; that gap is the whole point.
    assert timing.issued_tps > timing.goodput_tps


def test_step_summaries_report_the_spread_of_the_decode_steps():
    timing = _timing(step_s=[0.5, 0.9, 0.4, 0.6])

    assert timing.mean_step_s == pytest.approx(0.6)
    assert timing.median_step_s == pytest.approx(0.55)
    assert timing.min_step_s == pytest.approx(0.4)


def test_a_timing_whose_rows_outlast_the_batch_is_rejected():
    with pytest.raises(ValueError, match="cannot finish after"):
        _timing(finished_at=[1, 9, 2])


def test_a_timing_must_carry_one_finish_per_row():
    with pytest.raises(ValueError, match="one finish per row"):
        _timing(finished_at=[1, 2])


# --- head-of-line blocking: what the short rows pay -------------------------


def test_a_rows_own_work_ends_at_its_own_step_not_the_batchs():
    timing = _timing()  # prefill 1.0, steps 0.5 each

    assert timing.row_finish_s(0) == pytest.approx(1.5)  # prefill + one step
    assert timing.row_finish_s(1) == pytest.approx(3.0)  # the straggler: all four
    assert timing.row_finish_s(2) == pytest.approx(2.0)


def test_hol_delay_is_the_time_a_finished_row_spends_waiting_on_the_batch():
    """Static batching returns every row when the last one is done, so the wait is real."""
    timing = _timing()  # total 3.0

    assert timing.hol_delay_s(0) == pytest.approx(1.5)
    assert timing.hol_delay_s(1) == pytest.approx(0.0)  # nobody blocks the straggler
    assert timing.hol_delay_s(2) == pytest.approx(1.0)
    assert timing.max_hol_delay_s == pytest.approx(1.5)
    assert timing.mean_hol_delay_s == pytest.approx(2.5 / 3)


def test_hol_inflation_is_how_many_times_longer_a_row_waited_than_it_worked():
    timing = _timing()

    assert timing.hol_inflation(0) == pytest.approx(3.0 / 1.5)  # 2x its own latency
    assert timing.hol_inflation(1) == pytest.approx(1.0)
    assert timing.max_hol_inflation == pytest.approx(2.0)


def test_the_straggler_is_the_row_that_sets_the_batchs_length():
    timing = _timing()

    assert timing.straggler == 1


def test_one_long_row_among_short_ones_is_the_worst_case_the_scheduler_exists_for():
    """The Week-8 pitch in one number: 7 rows of 8 tokens behind 1 row of 200."""
    timing = BatchTiming(
        batch_size=8,
        prefill_s=0.0,
        step_s=[0.01] * 200,
        finished_at=[200] + [8] * 7,
    )

    assert timing.useful_tokens == 256
    assert timing.issued_tokens == 1600
    assert timing.waste_fraction == pytest.approx(1 - 256 / 1600)
    assert timing.max_hol_inflation == pytest.approx(25.0)  # a short row waits 25x


# --- scaling: what another row in the batch is worth ------------------------


def test_scaling_reports_speedup_and_efficiency_against_the_single_row_baseline():
    scaling = fit_batch_scaling([(1, 10.0), (2, 19.0), (4, 32.0), (8, 40.0)])

    assert isinstance(scaling, BatchScaling)
    assert scaling.baseline_tps == pytest.approx(10.0)
    assert scaling.speedup(4) == pytest.approx(3.2)
    # Efficiency is the share of ideal linear scaling that survived.
    assert scaling.efficiency(2) == pytest.approx(0.95)
    assert scaling.efficiency(8) == pytest.approx(0.5)


def test_the_best_size_is_the_one_with_the_most_throughput_not_the_most_rows():
    """Past saturation another row buys nothing and can cost; the sweep should say so."""
    scaling = fit_batch_scaling([(1, 10.0), (2, 19.0), (4, 32.0), (8, 30.0)])

    assert scaling.best_size == 4
    assert scaling.best_tps == pytest.approx(32.0)


def test_the_knee_is_the_largest_size_still_paying_its_way():
    scaling = fit_batch_scaling([(1, 10.0), (2, 19.0), (4, 32.0), (8, 40.0)])

    assert scaling.knee(0.9) == 2  # 4 is at 0.8, 2 is at 0.95
    assert scaling.knee(0.5) == 8
    assert scaling.knee(0.99) == 1  # the baseline is always at efficiency 1.0


def test_asking_about_a_size_that_was_not_swept_is_an_error():
    scaling = fit_batch_scaling([(1, 10.0), (2, 19.0)])

    with pytest.raises(KeyError):
        scaling.speedup(3)


def test_scaling_needs_the_single_row_baseline_to_have_anything_to_divide_by():
    with pytest.raises(ValueError, match="batch size 1"):
        fit_batch_scaling([(2, 19.0), (4, 32.0)])


def test_scaling_rejects_a_repeated_batch_size():
    with pytest.raises(ValueError, match="repeated"):
        fit_batch_scaling([(1, 10.0), (2, 19.0), (2, 18.0)])


def test_scaling_rejects_non_positive_sizes_and_rates():
    with pytest.raises(ValueError, match="positive"):
        fit_batch_scaling([(1, 10.0), (2, 0.0)])
    with pytest.raises(ValueError, match="positive"):
        fit_batch_scaling([(1, 10.0), (0, 5.0)])


def test_scaling_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least two"):
        fit_batch_scaling([(1, 10.0)])
