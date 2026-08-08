"""Day 32: pure tests for the continuous-batching measurement, and its runner.

Day 29 measured static batching and reported two bills: `waste_fraction`, the share
of the decode forward computed for rows that were already done, and
`hol_inflation`, how many times longer a finished row waited than its own work took.
Day 31 built the loop that is supposed to remove both. This file is the acceptance
test for that claim, and it is deliberately two halves, the same split the static
side has.

The first half is the timing core (`time_continuous_run`), driven by a fake clock
and scripted iteration outcomes. Every number the comparison will quote is pinned
exactly here: issued against collected, goodput, occupancy, and per-request latency,
which is the number continuous batching exists to change. A benchmark whose own
arithmetic is unverified is a confident guess, and this one is about to be quoted as
evidence that a week's work paid off.

The second half drives a real `Engine` over a tiny model. Two things the arithmetic
cannot check live there:

  1. **It measures the engine, not a sketch of it.** The tokens the timed run
     collects are exactly the ones `Engine.generate` emits for the same prompts and
     budgets. A harness that has drifted from the engine is measuring fiction.
  2. **The batch really shrinks.** Under static batching the row count is a constant;
     here `batch_sizes` has to fall as requests finish, which is the mechanism the
     whole measurement is attributing the win to.

The head-to-head runs under a unit clock (every clock read advances by exactly one),
so "seconds" are iterations and the ratios are exact integers rather than whatever
this box happened to do while the test ran. Wall-clock numbers belong in the daily
log, not in an assertion.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batchbench import (
    BatchingComparison,
    ContinuousTiming,
    IterationOutcome,
    build_continuous_run,
    compare_batching,
    compare_model_batching,
    time_continuous_run,
    time_model_batch,
    time_model_continuous,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel


class FakeClock:
    """Scripted, strictly increasing times, one pop per read.

    `time_continuous_run` reads the clock exactly twice per iteration (before and
    after the step), so a run of N iterations needs exactly 2*N ticks. There is no
    prefill read: under continuous batching the prefill is just an iteration in
    which some rows happened to be newly admitted.
    """

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


def unit_clock():
    """A clock that advances by exactly 1.0 per read: iterations, wearing seconds.

    Two reads per iteration, so every iteration costs 2.0 and every latency is a
    count of iterations times two. Lets a real model run produce deterministic
    ratios, which is the only kind of timing assertion worth writing in a test.
    """
    counter = {"t": 0.0}

    def clock() -> float:
        counter["t"] += 1.0
        return counter["t"]

    return clock


class ScriptedLoop:
    """A fake engine loop: hand out one prepared outcome per iteration."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self._i = 0

    def unfinished(self) -> bool:
        return self._i < len(self._outcomes)

    def step(self) -> IterationOutcome:
        outcome = self._outcomes[self._i]
        self._i += 1
        return outcome


def _outcome(batch_size: int, collected: int | None = None, finished=()) -> IterationOutcome:
    return IterationOutcome(
        batch_size=batch_size,
        collected=batch_size if collected is None else collected,
        finished=tuple(finished),
    )


def _timing(**overrides) -> ContinuousTiming:
    """A hand-built timing: 3 requests over 2 slots, four iterations, ragged exits."""
    kwargs = dict(
        num_requests=3,
        max_batch_size=2,
        step_s=[1.0, 1.0, 2.0, 2.0],
        batch_sizes=[2, 2, 2, 1],
        collected=[2, 2, 2, 1],
        finished_at={"a": 1, "b": 2, "c": 3},
    )
    kwargs.update(overrides)
    return ContinuousTiming(**kwargs)


# --- the timing core ---------------------------------------------------------


def test_the_core_times_one_pair_of_clock_reads_per_iteration():
    loop = ScriptedLoop([_outcome(2), _outcome(2, finished=("a",)), _outcome(1, finished=("b",))])
    timing = time_continuous_run(
        loop.step,
        loop.unfinished,
        num_requests=2,
        max_batch_size=2,
        clock=FakeClock([0.0, 0.5, 0.5, 1.0, 1.0, 1.25]),
    )
    assert timing.n_iterations == 3
    assert timing.step_s == pytest.approx([0.5, 0.5, 0.25])
    assert timing.total_s == pytest.approx(1.25)


def test_the_core_stops_when_the_loop_reports_nothing_unfinished():
    loop = ScriptedLoop([_outcome(1), _outcome(1, finished=("a",))])
    timing = time_continuous_run(
        loop.step, loop.unfinished, num_requests=1, max_batch_size=1, clock=FakeClock(range(4))
    )
    assert timing.n_iterations == 2


def test_the_core_records_the_batch_size_of_every_iteration():
    loop = ScriptedLoop([_outcome(4), _outcome(3), _outcome(1)])
    timing = time_continuous_run(
        loop.step, loop.unfinished, num_requests=4, max_batch_size=4, clock=FakeClock(range(6))
    )
    assert timing.batch_sizes == [4, 3, 1]


def test_the_core_maps_each_request_to_the_iteration_it_finished_in():
    loop = ScriptedLoop(
        [_outcome(3), _outcome(3, finished=("a", "b")), _outcome(1, finished=("c",))]
    )
    timing = time_continuous_run(
        loop.step, loop.unfinished, num_requests=3, max_batch_size=3, clock=FakeClock(range(6))
    )
    assert timing.finished_at == {"a": 1, "b": 1, "c": 2}


def test_a_request_that_finishes_twice_is_a_bug_and_raises():
    # Its second finish would silently overwrite its latency with a later one, and
    # the run would report a slower engine than it is.
    loop = ScriptedLoop([_outcome(1, finished=("a",)), _outcome(1, finished=("a",))])
    with pytest.raises(ValueError, match="finished twice"):
        time_continuous_run(
            loop.step, loop.unfinished, num_requests=1, max_batch_size=1, clock=FakeClock(range(4))
        )


def test_a_loop_that_never_drains_raises_instead_of_hanging():
    class Forever:
        def unfinished(self):
            return True

        def step(self):
            return _outcome(1)

    forever = Forever()
    with pytest.raises(RuntimeError, match="did not drain"):
        time_continuous_run(
            forever.step,
            forever.unfinished,
            num_requests=1,
            max_batch_size=1,
            clock=FakeClock(range(100)),
            max_iterations=5,
        )


def test_an_iteration_that_collects_more_than_it_issued_is_rejected():
    # At the boundary, not in the timing: an outcome that cannot have happened
    # should not survive long enough to be averaged into a rate.
    with pytest.raises(ValueError, match="at most one token"):
        IterationOutcome(batch_size=2, collected=3)


def test_an_iteration_with_negative_counts_is_rejected():
    with pytest.raises(ValueError, match="negative rows or tokens"):
        IterationOutcome(batch_size=-1, collected=0)


# --- what the timing says ----------------------------------------------------


def test_issued_and_collected_are_the_two_counts_the_week_is_about():
    timing = _timing()
    assert timing.issued_tokens == 7
    assert timing.collected_tokens == 7
    assert timing.wasted_tokens == 0
    assert timing.waste_fraction == 0.0


def test_waste_fraction_still_counts_a_row_that_kept_nothing():
    # Not reachable through the Day-31 engine, where every scheduled row collects.
    # The metric is defined anyway, because a benchmark that can only express the
    # number it hopes to see is not a measurement.
    timing = _timing(collected=[2, 2, 1, 1])
    assert timing.issued_tokens == 7
    assert timing.collected_tokens == 6
    assert timing.waste_fraction == pytest.approx(1 / 7)


def test_goodput_and_issued_rates_are_over_the_whole_run():
    # One loop, so there is no prefill to hold out of the denominator the way the
    # static timing does: the prefill is an iteration like any other.
    timing = _timing(collected=[2, 2, 1, 1])
    assert timing.goodput_tps == pytest.approx(6 / 6.0)
    assert timing.issued_tps == pytest.approx(7 / 6.0)


def test_occupancy_is_the_share_of_the_slots_a_forward_actually_filled():
    timing = _timing()
    # 7 rows issued over 4 iterations of 2 slots.
    assert timing.occupancy == pytest.approx(7 / 8)
    assert timing.mean_batch_size == pytest.approx(7 / 4)


def test_occupancy_ignores_a_final_iteration_that_forwarded_nothing():
    # The engine's last step reaps and schedules an empty batch. It costs real time
    # and issues nothing, so it belongs in the makespan and not in the denominator
    # of "how full were the forwards".
    timing = _timing(
        step_s=[1.0, 1.0, 2.0, 2.0, 0.5],
        batch_sizes=[2, 2, 2, 1, 0],
        collected=[2, 2, 2, 1, 0],
    )
    assert timing.forward_iterations == 4
    assert timing.occupancy == pytest.approx(7 / 8)
    assert timing.total_s == pytest.approx(6.5)


def test_latency_is_the_elapsed_time_at_the_iteration_the_request_finished():
    timing = _timing()
    assert timing.latency_s("a") == pytest.approx(2.0)  # iterations 0 and 1
    assert timing.latency_s("b") == pytest.approx(4.0)
    assert timing.latency_s("c") == pytest.approx(6.0)


def test_the_latency_summary_is_over_the_requests_that_finished():
    timing = _timing()
    assert timing.min_latency_s == pytest.approx(2.0)
    assert timing.mean_latency_s == pytest.approx(4.0)
    assert timing.max_latency_s == pytest.approx(6.0)


def test_asking_for_an_unknown_request_raises_rather_than_inventing_a_latency():
    with pytest.raises(KeyError, match="never finished"):
        _timing().latency_s("nobody")


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"batch_sizes": [2, 2, 2]}, "one entry per iteration"),
        ({"collected": [2, 2, 2]}, "one entry per iteration"),
        ({"batch_sizes": [2, 2, 2, 3]}, "more rows than"),
        ({"collected": [2, 2, 2, 2]}, "at most one token"),
        ({"num_requests": 0}, "at least one request"),
        ({"max_batch_size": 0}, "at least one slot"),
        ({"finished_at": {"a": 1, "b": 2, "c": 3, "d": 4}}, "more requests finished"),
        ({"finished_at": {"a": 1, "b": 2, "c": 9}}, "outside the run"),
    ],
)
def test_a_timing_that_cannot_be_true_is_rejected_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        _timing(**overrides)


def test_an_empty_run_reports_zeroes_rather_than_dividing_by_zero():
    timing = _timing(step_s=[], batch_sizes=[], collected=[], finished_at={})
    assert timing.total_s == 0.0
    assert timing.waste_fraction == 0.0
    assert timing.goodput_tps == 0.0
    assert timing.issued_tps == 0.0
    assert timing.occupancy == 0.0
    assert timing.mean_batch_size == 0.0


# --- the head-to-head arithmetic ---------------------------------------------


def _static(**overrides):
    """A static run of a 3-request workload: rows of 2, 3 and 4 tokens, held to the end."""
    from nanoserve.batchbench import BatchTiming

    kwargs = dict(batch_size=3, prefill_s=1.0, step_s=[1.0, 1.0, 1.0], finished_at=[1, 2, 3])
    kwargs.update(overrides)
    return BatchTiming(**kwargs)


def _matching_continuous(**overrides) -> ContinuousTiming:
    """The same three requests through the scheduled loop: 9 tokens, ragged exits."""
    kwargs = dict(
        num_requests=3,
        max_batch_size=3,
        step_s=[1.0, 1.0, 2.0, 2.0],
        batch_sizes=[3, 3, 2, 1],
        collected=[3, 3, 2, 1],
        finished_at={"a": 1, "b": 2, "c": 3},
    )
    kwargs.update(overrides)
    return ContinuousTiming(**kwargs)


def test_the_static_timing_can_count_its_prefill_token_too():
    # Day 29 quoted goodput over the decode alone, because the prefill is a
    # different shape of forward. Comparing against a loop that has no separable
    # prefill needs the end-to-end rate, so it is defined here: every row collected
    # a token from the prefill as well.
    static = _static()
    assert static.useful_tokens == 6
    assert static.end_to_end_goodput_tps == pytest.approx((6 + 3) / 4.0)


def test_the_comparison_reports_the_work_the_scheduled_loop_did_not_do():
    static = _static()
    comparison = compare_batching(static, _matching_continuous())
    # Static issues 3 rows x 3 steps plus 3 prefill tokens for the same 9 collected.
    assert comparison.static_issued_tokens == 12
    assert comparison.continuous_issued_tokens == 9
    assert comparison.work_ratio == pytest.approx(12 / 9)
    assert comparison.waste_removed == pytest.approx(static.waste_fraction)


def test_the_comparison_reports_the_latency_every_row_stopped_paying_for():
    comparison = compare_batching(_static(), _matching_continuous())
    # Static hands every row back at 4.0s; the scheduled loop hands them back at
    # 2.0, 4.0 and 6.0 and takes 6.0 in total.
    assert comparison.makespan_speedup == pytest.approx(4.0 / 6.0)
    assert comparison.mean_latency_speedup == pytest.approx(4.0 / 4.0)
    assert comparison.first_answer_speedup == pytest.approx(4.0 / 2.0)


def test_the_comparison_reports_goodput_end_to_end_on_both_sides():
    comparison = compare_batching(_static(), _matching_continuous())
    assert comparison.goodput_speedup == pytest.approx((9 / 6.0) / ((6 + 3) / 4.0))


@pytest.mark.parametrize(
    "static_kwargs, match",
    [
        ({"batch_size": 2, "finished_at": [1, 2]}, "same number of requests"),
        ({"finished_at": [1, 1, 1]}, "same tokens"),
    ],
)
def test_comparing_two_different_workloads_is_refused(static_kwargs, match):
    # The only thing that makes a speedup meaningful is that both runs produced the
    # same answers. A comparison that cannot check that is a press release.
    with pytest.raises(ValueError, match=match):
        compare_batching(_static(**static_kwargs), _matching_continuous())


def test_comparing_a_run_with_no_measured_time_is_refused():
    with pytest.raises(ValueError, match="no measured time"):
        compare_batching(
            _static(prefill_s=0.0, step_s=[0.0, 0.0, 0.0]),
            _matching_continuous(),
        )


def test_the_comparison_keeps_both_timings_for_the_reader():
    comparison = compare_batching(_static(), _matching_continuous())
    assert isinstance(comparison, BatchingComparison)
    assert comparison.static.batch_size == 3
    assert comparison.continuous.num_requests == 3


# --- the runner, over a real engine ------------------------------------------


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the other model tests use."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(seed: int = 0) -> tuple[LlamaModel, ModelConfig]:
    """A tiny model on fixed random weights (seeded: greedy tokens must be stable)."""
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


PROMPTS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]


def test_the_runner_collects_exactly_what_the_engine_generates():
    model, _ = _model()
    run = build_continuous_run(model, PROMPTS, max_new_tokens=[3, 5, 2, 4], block_size=4)
    time_continuous_run(
        run.step_fn,
        run.unfinished_fn,
        num_requests=len(PROMPTS),
        max_batch_size=run.engine.cache.batch_size,
        clock=unit_clock(),
    )
    measured = [r.token_ids for r in run.requests]

    reference = Engine.build(
        _model()[0], num_blocks=64, block_size=4, max_batch_size=len(PROMPTS)
    )
    expected = [
        reference.generate([prompt], max_new_tokens=budget)[0]
        for prompt, budget in zip(PROMPTS, [3, 5, 2, 4])
    ]
    assert measured == expected


def test_every_request_finishes_at_its_own_budget_not_the_longest():
    model, _ = _model()
    timing = time_model_continuous(
        model, PROMPTS, max_new_tokens=[3, 5, 2, 4], block_size=4, clock=unit_clock()
    )
    # Budget b means one prefill iteration plus b-1 decode iterations, and every
    # row is admitted in iteration 0 here, so a row's finishing iteration is b-1.
    assert timing.finished_at == {"r0": 2, "r1": 4, "r2": 1, "r3": 3}
    assert timing.collected_tokens == 3 + 5 + 2 + 4


def test_the_batch_shrinks_as_requests_leave_it():
    model, _ = _model()
    timing = time_model_continuous(
        model, PROMPTS, max_new_tokens=[3, 5, 2, 4], block_size=4, clock=unit_clock()
    )
    # Four rows, then three (r2 gone), then three, then two, then one, then the
    # final reap with nothing left to forward.
    assert timing.batch_sizes == [4, 4, 3, 2, 1, 0]
    assert timing.waste_fraction == 0.0
    assert timing.forward_iterations == 5


def test_a_pool_too_small_for_everyone_queues_instead_of_failing():
    model, _ = _model()
    # Three blocks for prompts wanting five, so the queue really is holding somebody
    # back and the measured occupancy is a number about a busy engine, not 1.0. Day
    # 33 is why this needs a pool half the size it used to: admission buys the
    # prompt's blocks now, so the same six blocks fit every request at once.
    timing = time_model_continuous(
        model,
        PROMPTS,
        max_new_tokens=[3, 5, 2, 4],
        block_size=4,
        num_blocks=3,
        clock=unit_clock(),
    )
    assert timing.collected_tokens == 3 + 5 + 2 + 4
    assert max(timing.batch_sizes) < len(PROMPTS)


def test_the_runner_rejects_a_budget_that_is_not_one_per_prompt():
    model, _ = _model()
    with pytest.raises(ValueError, match="one token budget per prompt"):
        build_continuous_run(model, PROMPTS, max_new_tokens=[3, 5], block_size=4)


def test_one_budget_for_every_prompt_is_the_uniform_case():
    model, _ = _model()
    run = build_continuous_run(model, PROMPTS, max_new_tokens=3, block_size=4)
    assert [r.max_new_tokens for r in run.requests] == [3, 3, 3, 3]


# --- the acceptance test: the Day-29 shape, both ways ------------------------


def test_the_day_29_shape_costs_less_work_and_less_waiting_when_scheduled():
    """Seven short rows behind one long one, static and continuous, one clock.

    The unit clock makes a "second" one clock read, so every ratio below is a count
    of iterations and holds on any box. What it cannot show is the wall-clock win,
    since a smaller batch is also a cheaper forward; that is the daily log's job.
    """
    model, _ = _model()
    prompts = [[1, 2, 3, 4]] * 8
    budgets = [4] * 7 + [32]

    comparison = compare_model_batching(
        model, prompts, max_new_tokens=budgets, block_size=16, clock=unit_clock()
    )
    static, continuous = comparison.static, comparison.continuous

    # Same answers on both sides: 7 rows of 4 tokens and one of 32.
    assert continuous.collected_tokens == 7 * 4 + 32 == 60
    assert static.useful_tokens + static.batch_size == 60

    # The static batch runs to its slowest row and charges everyone for it.
    assert static.issued_tokens == 8 * 31
    assert static.waste_fraction > 0.75
    assert continuous.waste_fraction == 0.0

    # 8 rows for 32 iterations against 8 rows for 4 and then 1 for 28.
    assert comparison.continuous_issued_tokens == 8 * 4 + 28
    assert comparison.work_ratio > 4.0
    # Every static row waits for the straggler; here the short ones do not.
    assert comparison.first_answer_speedup > 7.0
    assert comparison.mean_latency_speedup > 3.0


def test_the_two_runs_are_the_same_workload_and_the_comparison_checks_it():
    model, _ = _model()
    prompts = [[1, 2, 3], [4, 5, 6]]
    budgets = [2, 6]
    static = time_model_batch(
        model,
        prompts,
        max_new_tokens=max(budgets),
        stop_steps=[b - 1 for b in budgets],
        block_size=4,
        clock=unit_clock(),
    )
    continuous = time_model_continuous(
        model, prompts, max_new_tokens=budgets, block_size=4, clock=unit_clock()
    )
    comparison = compare_batching(static, continuous)
    assert comparison.static_issued_tokens == 2 * 5 + 2
    assert comparison.continuous_issued_tokens == 2 * 2 + 4
