"""Day 34 tests: the preemption bill, reported where it is paid.

Day 33 built preemption by recompute and proved the only property that makes a
memory policy allowed to exist: a cramped engine and a roomy one return the same
text. What it did not build is any way to see what that cost. `batchbench` counted
rows per iteration, so a run that threw away half its K/V and bought it again looked
exactly like a run that scheduled well: the recompute landed inside a prefill, the
prefill was one iteration like any other, and the victim's extra wait averaged
quietly into the mean latency.

This file pins the three numbers that make the cost visible, in the same two halves
the rest of the benchmark uses.

  1. **Preemptions, per request.** A total is a property of the pool; a per-request
     count is the thing a caller was charged. One request evicted four times and
     four requests evicted once are the same total and very different services.
  2. **The recompute bill as a share of forward work.** Which forces the denominator
     to stop being rows. A decode row forwards one position and a resumed prefill
     forwards its whole context, so counting iterations or batch sizes hides the
     surcharge in exactly the place it is largest. Pad slots stay out of that
     denominator on purpose: padding is a separate bill, already measured, and
     mixing it in would make the recompute share depend on how the prompts happened
     to line up.
  3. **What a victim waited, against what everybody else waited.** The recompute is
     paid in FLOPs by the engine and in latency by one caller, and only the second
     one is felt.

The pure half scripts iterations against a fake clock and pins every ratio exactly.
The model half runs the real engine through shrinking pools and holds the harness to
the engine's own counters, because a benchmark that has drifted from the thing it
measures is fiction with decimal places.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batchbench import (
    ContinuousTiming,
    IterationOutcome,
    build_continuous_run,
    sweep_pool_sizes,
    time_continuous_run,
    time_model_continuous,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel


class FakeClock:
    """Scripted, strictly increasing times, one pop per read (see test_batchbench)."""

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


def unit_clock():
    """A clock that advances by exactly 1.0 per read: iterations, wearing seconds."""
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


def _timing(**overrides) -> ContinuousTiming:
    """One scripted run with a preemption in it, worked out by hand.

    Three requests over two slots, and a pool tight enough that b is evicted for a
    and comes back later:

      it 0: a and b admitted, two prefills of 4 tokens   -> 8 positions
      it 1: a and b decode                               -> 2
      it 2: b preempted for a, c admitted and prefilled  -> 5 (1 decode + 4 prompt)
      it 3: a and c decode, a finishes                   -> 2
      it 4: c decodes, b resumes over its 8 tokens       -> 9, of which 8 recomputed
      it 5: b and c decode, c finishes                   -> 2
      it 6: b decodes and finishes                       -> 1

    Every step is one second, so a latency is a count of iterations.
    """
    kwargs = dict(
        num_requests=3,
        max_batch_size=2,
        step_s=[1.0] * 7,
        batch_sizes=[2, 2, 2, 2, 2, 2, 1],
        collected=[2, 2, 2, 2, 2, 2, 1],
        forward_tokens=[8, 2, 5, 2, 9, 2, 1],
        recomputed=[0, 0, 0, 0, 8, 0, 0],
        preempted_at={"b": [2]},
        finished_at={"a": 3, "b": 6, "c": 5},
    )
    kwargs.update(overrides)
    return ContinuousTiming(**kwargs)


# --- the outcome an iteration reports ----------------------------------------


def test_an_iteration_reports_who_it_evicted_and_what_that_cost():
    outcome = IterationOutcome(
        batch_size=2, collected=2, preempted=("b",), forward_tokens=9, recomputed_tokens=8
    )
    assert outcome.preempted == ("b",)
    assert outcome.work_tokens == 9
    assert outcome.recomputed_tokens == 8


def test_an_iteration_that_does_not_report_positions_is_charged_one_per_row():
    # Which is exactly right for a decode-only iteration, and is the shape every
    # Day-32 caller fed this class before there was a prefill length to report.
    assert IterationOutcome(batch_size=3, collected=3).work_tokens == 3


def test_an_iteration_cannot_forward_fewer_positions_than_it_has_rows():
    with pytest.raises(ValueError, match="at least one position per row"):
        IterationOutcome(batch_size=4, collected=4, forward_tokens=3)


def test_an_iteration_cannot_recompute_more_than_it_forwarded():
    # The recompute is a subset of the forward, not an extra charge alongside it.
    with pytest.raises(ValueError, match="more positions than it forwarded"):
        IterationOutcome(batch_size=1, collected=1, forward_tokens=4, recomputed_tokens=5)


def test_negative_forward_or_recompute_counts_are_rejected():
    with pytest.raises(ValueError, match="negative forward or recomputed"):
        IterationOutcome(batch_size=1, collected=1, forward_tokens=-1)
    with pytest.raises(ValueError, match="negative forward or recomputed"):
        IterationOutcome(batch_size=1, collected=1, recomputed_tokens=-1)


# --- the timing core ---------------------------------------------------------


def test_the_core_maps_each_request_to_the_iterations_it_was_evicted_in():
    loop = ScriptedLoop(
        [
            IterationOutcome(batch_size=2, collected=2),
            IterationOutcome(batch_size=1, collected=1, preempted=("b",)),
            IterationOutcome(batch_size=1, collected=1, preempted=("b",)),
            IterationOutcome(batch_size=1, collected=1, finished=("a", "b")),
        ]
    )
    timing = time_continuous_run(
        loop.step, loop.unfinished, num_requests=2, max_batch_size=2, clock=FakeClock(range(8))
    )
    assert timing.preempted_at == {"b": [1, 2]}
    assert timing.preemptions("b") == 2
    assert timing.preemptions("a") == 0


def test_the_core_carries_the_forward_and_recompute_counts_through():
    loop = ScriptedLoop(
        [
            IterationOutcome(batch_size=2, collected=2, forward_tokens=8),
            IterationOutcome(batch_size=2, collected=2, forward_tokens=6, recomputed_tokens=5),
            IterationOutcome(batch_size=1, collected=1, finished=("a",)),
        ]
    )
    timing = time_continuous_run(
        loop.step, loop.unfinished, num_requests=1, max_batch_size=2, clock=FakeClock(range(6))
    )
    assert timing.forward_tokens == [8, 6, 1]
    assert timing.recomputed == [0, 5, 0]


def test_a_request_evicted_after_it_finished_is_a_bug_and_raises():
    # A finished request owns nothing to take, so this is an id mix-up, and it would
    # otherwise land as a phantom victim in the per-request counts.
    loop = ScriptedLoop(
        [
            IterationOutcome(batch_size=1, collected=1, finished=("a",)),
            IterationOutcome(batch_size=1, collected=1, preempted=("a",)),
        ]
    )
    with pytest.raises(ValueError, match="preempted after it finished"):
        time_continuous_run(
            loop.step, loop.unfinished, num_requests=1, max_batch_size=1, clock=FakeClock(range(4))
        )


# --- what the timing says ----------------------------------------------------


def test_the_preemption_count_is_per_request_and_in_total():
    timing = _timing()
    assert timing.num_preemptions == 1
    assert timing.preempted_requests == ("b",)
    assert timing.preemptions_per_request == pytest.approx(1 / 3)


def test_forward_work_counts_positions_so_a_resumed_prefill_is_not_one_row():
    timing = _timing()
    # 8 + 2 + 5 + 2 + 9 + 2 + 1, against 13 rows over the same seven iterations.
    assert timing.total_forward_tokens == 29
    assert timing.issued_tokens == 13


def test_the_recompute_bill_is_a_share_of_the_forward_not_of_the_rows():
    timing = _timing()
    assert timing.recomputed_tokens == 8
    assert timing.recompute_fraction == pytest.approx(8 / 29)


def test_a_run_that_never_preempted_reports_no_recompute_and_no_penalty():
    timing = _timing(recomputed=[0] * 7, preempted_at={})
    assert timing.num_preemptions == 0
    assert timing.recompute_fraction == 0.0
    assert timing.preemption_latency_penalty == 1.0


def test_the_victim_is_compared_against_the_requests_that_ran_straight_through():
    timing = _timing()
    # a finished at iteration 3 and c at 5, so 4s and 6s; b was evicted and took 7s.
    assert timing.straight_through_latencies_s == pytest.approx([4.0, 6.0])
    assert timing.preempted_latencies_s == pytest.approx([7.0])
    assert timing.mean_straight_through_latency_s == pytest.approx(5.0)
    assert timing.mean_preempted_latency_s == pytest.approx(7.0)
    assert timing.preemption_latency_penalty == pytest.approx(1.4)


def test_with_no_control_group_the_penalty_says_nothing_rather_than_inventing_it():
    # Everybody was preempted, so there is no straight-through latency to divide by
    # and the honest answer is not a number.
    timing = _timing(preempted_at={"a": [2], "b": [2], "c": [2]})
    assert timing.straight_through_latencies_s == []
    assert timing.preemption_latency_penalty == 0.0


def test_a_timing_built_without_the_new_series_reads_as_the_day_32_one():
    # One position per row and nothing recomputed, so every Day-32 timing keeps the
    # numbers it had and the new ones degrade to "no preemption happened here".
    timing = ContinuousTiming(
        num_requests=2,
        max_batch_size=2,
        step_s=[1.0, 1.0],
        batch_sizes=[2, 1],
        collected=[2, 1],
        finished_at={"a": 0, "b": 1},
    )
    assert timing.forward_tokens == [2, 1]
    assert timing.recomputed == [0, 0]
    assert timing.total_forward_tokens == 3
    assert timing.recompute_fraction == 0.0
    assert timing.num_preemptions == 0


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"forward_tokens": [8, 2, 5]}, "one entry per iteration"),
        ({"recomputed": [0, 0]}, "one entry per iteration"),
        ({"forward_tokens": [1, 2, 5, 2, 9, 2, 1]}, "at least one position per row"),
        ({"recomputed": [9, 0, 0, 0, 0, 0, 0]}, "more positions than it forwarded"),
        ({"preempted_at": {"b": [9]}}, "preempted outside the run"),
        ({"preempted_at": {"b": [-1]}}, "preempted outside the run"),
    ],
)
def test_a_preemption_record_that_cannot_be_true_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _timing(**overrides)


def test_an_empty_run_reports_zeroes_rather_than_dividing_by_zero():
    timing = ContinuousTiming(num_requests=1, max_batch_size=1)
    assert timing.total_forward_tokens == 0
    assert timing.recompute_fraction == 0.0
    assert timing.preemption_latency_penalty == 1.0


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
BUDGETS = [6, 5, 4, 6]


def test_a_pool_with_room_preempts_nobody_and_recomputes_nothing():
    model, _ = _model()
    timing = time_model_continuous(
        model, PROMPTS, max_new_tokens=BUDGETS, block_size=4, num_blocks=32, clock=unit_clock()
    )
    assert timing.num_preemptions == 0
    assert timing.recomputed_tokens == 0
    assert timing.recompute_fraction == 0.0
    assert timing.preemption_latency_penalty == 1.0


def test_a_cramped_pool_preempts_and_the_report_matches_the_engine():
    model, _ = _model()
    run = build_continuous_run(
        model, PROMPTS, max_new_tokens=BUDGETS, block_size=4, num_blocks=4
    )
    timing = time_continuous_run(
        run.step_fn,
        run.unfinished_fn,
        num_requests=len(PROMPTS),
        max_batch_size=run.engine.cache.batch_size,
        clock=unit_clock(),
    )
    engine = run.engine
    assert timing.num_preemptions == engine.scheduler.num_preemptions > 0
    assert timing.recomputed_tokens == engine.recomputed_tokens > 0
    # The harness's forward positions are the engine's own: every real prefill token
    # plus one per decode row.
    assert timing.total_forward_tokens == engine.forward_tokens
    assert engine.forward_tokens == (
        engine.prefill_tokens + engine.issued_tokens - engine.prefill_rows
    )


def test_the_recompute_share_is_a_real_share_of_a_bigger_denominator():
    model, _ = _model()
    timing = time_model_continuous(
        model, PROMPTS, max_new_tokens=BUDGETS, block_size=4, num_blocks=4, clock=unit_clock()
    )
    assert 0.0 < timing.recompute_fraction < 1.0
    # Rows would flatter it: the recompute arrives as one row carrying a whole
    # context, so dividing by iterations understates what the card actually did.
    assert timing.total_forward_tokens > timing.issued_tokens


def test_a_victim_waits_longer_than_the_requests_that_were_never_evicted():
    model, _ = _model()
    timing = time_model_continuous(
        model, PROMPTS, max_new_tokens=BUDGETS, block_size=4, num_blocks=4, clock=unit_clock()
    )
    assert timing.preempted_requests
    assert timing.straight_through_latencies_s
    assert timing.preemption_latency_penalty > 1.0


# --- the sweep: the same workload through a shrinking pool -------------------


def test_the_pool_sweep_returns_one_timing_per_pool_and_recompute_rises_as_it_shrinks():
    model, _ = _model()
    timings = sweep_pool_sizes(
        model,
        PROMPTS,
        max_new_tokens=BUDGETS,
        pool_sizes=[32, 6, 4],
        block_size=4,
        clock=unit_clock(),
    )
    assert len(timings) == 3
    assert timings[0].num_preemptions == 0
    assert timings[-1].num_preemptions > 0
    assert timings[-1].recompute_fraction > timings[0].recompute_fraction
    # Same answers, so the surcharge is the only thing that changed.
    assert {t.collected_tokens for t in timings} == {sum(BUDGETS)}


def test_the_sweep_refuses_pool_sizes_it_cannot_compare():
    model, _ = _model()
    with pytest.raises(ValueError, match="at least one pool size"):
        sweep_pool_sizes(model, PROMPTS, max_new_tokens=BUDGETS, pool_sizes=[], block_size=4)
    with pytest.raises(ValueError, match="two answers to one question"):
        sweep_pool_sizes(
            model, PROMPTS, max_new_tokens=BUDGETS, pool_sizes=[8, 8], block_size=4
        )


def test_the_text_is_the_same_at_every_pool_size():
    """The Day-33 property, now checked by the harness that quotes the surcharge.

    A benchmark that reports "recompute cost 18% of the forward" is only worth
    reading if the run it measured produced the right answer, so the sweep compares
    the generations across pools itself and refuses to hand back timings that
    disagree.
    """
    model, _ = _model()
    reference = Engine.build(_model()[0], num_blocks=64, block_size=4, max_batch_size=4)
    expected = [
        reference.generate([prompt], max_new_tokens=budget)[0]
        for prompt, budget in zip(PROMPTS, BUDGETS)
    ]
    for pool in (32, 6, 4):
        run = build_continuous_run(
            model, PROMPTS, max_new_tokens=BUDGETS, block_size=4, num_blocks=pool
        )
        time_continuous_run(
            run.step_fn,
            run.unfinished_fn,
            num_requests=len(PROMPTS),
            max_batch_size=run.engine.cache.batch_size,
            clock=unit_clock(),
        )
        assert [r.token_ids for r in run.requests] == expected
