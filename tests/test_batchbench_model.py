"""Day 29: the batching benchmark run against a real model and a real block pool.

The core in `test_batchbench.py` is deliberately model-free: a fake clock and two
scripted callables. That measures the harness, not the engine. This file drives the
other half. `build_batched_decode` stands up a `BatchedPagedKVCache`, runs the Day-27
padded prefill and the Day-28 batched decode step, and hands the timing core the two
closures it expects, so the sweep times the loop `greedy_generate_batch` actually
runs rather than a function called in isolation.

Three things a benchmark's own arithmetic cannot check, pinned here:

  1. **It drives the real decode.** The tokens the run collects are exactly the ones
     `greedy_generate_batch` emits for the same prompts. A benchmark that has quietly
     drifted from the engine is measuring fiction.
  2. **The waste it reports is really being paid.** A row that has reported done
     still gets a slot and a block every step, and the cache says so: every row's
     cached length grows on every step, finished or not. That is the static-batching
     bill in the one place it cannot be argued with.
  3. **The finishes are prescribed, and that is deliberate.** Rows stop on a step
     budget rather than on EOS, because tiny random weights emit no EOS and the real
     model's output lengths are not the thing under measurement. The timing is real;
     which token ended the row is not the question. The step budget makes the
     raggedness a knob, so head-of-line blocking can be measured at a chosen spread
     instead of whatever a prompt happened to produce.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batchbench import (
    build_batched_decode,
    sweep_batch_sizes,
    time_model_batch,
)
from nanoserve.config import ModelConfig
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel


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


def _model(seed: int = 0) -> tuple[LlamaModel, ModelConfig]:
    """A tiny model on fixed random weights (seeded: greedy tokens must be stable)."""
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


class FakeClock:
    """Scripted, strictly increasing times (see test_batchbench).

    `time_batched_decode` reads the clock twice for the prefill and twice per decode
    step, so a run of S steps needs exactly 2 + 2*S ticks.
    """

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def __call__(self) -> float:
        t = self._ticks[self._i]
        self._i += 1
        return t


PROMPTS = [[1, 2, 3], [4], [5, 6]]


# --- the closures drive the engine's own loop --------------------------------


def test_the_run_collects_exactly_what_greedy_generate_batch_emits():
    """The benchmark times the real decode, so its tokens must be the engine's."""
    model, _ = _model()
    expected = model.greedy_generate_batch(PROMPTS, max_new_tokens=5)

    run = build_batched_decode(model, PROMPTS, max_new_tokens=5)
    run.prefill_fn()
    for _ in range(4):
        run.step_fn()

    assert run.rows == expected


def test_the_prefill_emits_one_token_per_row_before_any_step():
    model, _ = _model()
    run = build_batched_decode(model, PROMPTS, max_new_tokens=4)

    done = run.prefill_fn()

    assert [len(r) for r in run.rows] == [4, 2, 3]  # each prompt plus one token
    assert list(done) == [False, False, False]
    # Ragged in the cache, rectangle in the input: the prefill stored the prompts and
    # nothing else, since the token it just emitted has not been forwarded yet.
    assert run.cache.seq_lens == [3, 1, 2]


def test_a_finished_row_still_gets_a_slot_and_a_block_every_step():
    """The static-batching bill, read off the cache rather than argued about."""
    model, _ = _model()
    run = build_batched_decode(model, PROMPTS, max_new_tokens=5, stop_steps=[0, 3, 1])

    done = run.prefill_fn()
    assert list(done) == [True, False, False]  # row 0 is finished at the prefill
    before = list(run.cache.seq_lens)
    run.step_fn()

    # Every row grew, including the one that is done and collecting nothing.
    assert run.cache.seq_lens == [n + 1 for n in before]
    assert len(run.rows[0]) == 4  # prompt + the one prefill token, no more


def test_the_step_budget_is_what_reports_a_row_done():
    model, _ = _model()
    run = build_batched_decode(model, PROMPTS, max_new_tokens=6, stop_steps=[1, 4, 2])

    assert list(run.prefill_fn()) == [False, False, False]
    assert list(run.step_fn()) == [True, False, False]
    assert list(run.step_fn()) == [True, False, True]


def test_rows_that_never_stop_run_to_the_cap():
    model, _ = _model()

    timing = time_model_batch(model, PROMPTS, max_new_tokens=4, clock=FakeClock(range(20)))

    assert timing.n_steps == 3  # max_new_tokens - 1
    assert timing.finished_at == [3, 3, 3]
    assert timing.waste_fraction == 0.0  # nothing ragged, nothing wasted


# --- the two halves together: real forwards, exact arithmetic ----------------


def test_a_ragged_run_reports_the_blocking_the_clock_was_scripted_to_show():
    """Real forwards, fake clock: the engine moves, the arithmetic stays exact."""
    model, _ = _model()
    # prefill 0->1 (1.0s), then four 1.0s steps.
    clock = FakeClock([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0])

    timing = time_model_batch(
        model, PROMPTS, max_new_tokens=5, stop_steps=[1, 4, 2], clock=clock
    )

    assert timing.finished_at == [1, 4, 2]
    assert timing.total_s == pytest.approx(5.0)
    assert timing.issued_tokens == 12  # 3 rows * 4 steps
    assert timing.useful_tokens == 7
    assert timing.goodput_tps == pytest.approx(7 / 4.0)
    assert timing.issued_tps == pytest.approx(12 / 4.0)
    assert timing.straggler == 1
    assert timing.max_hol_inflation == pytest.approx(5.0 / 2.0)  # row 0 waited 2.5x


def test_the_batch_stops_when_its_slowest_row_does_not_at_the_cap():
    model, _ = _model()
    clock = FakeClock([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])

    timing = time_model_batch(
        model, PROMPTS, max_new_tokens=9, stop_steps=[1, 2, 2], clock=clock
    )

    assert timing.n_steps == 2  # not 8


# --- the sweep ----------------------------------------------------------------


def test_the_sweep_measures_one_prompt_replicated_across_batch_sizes():
    model, _ = _model()

    scaling, timings = sweep_batch_sizes(model, [1, 2, 3], [1, 2, 4], max_new_tokens=3)

    assert scaling.sizes == [1, 2, 4]
    assert [t.batch_size for t in timings] == [1, 2, 4]
    # Uniform rows: every issued token is collected, so the two rates agree and the
    # curve is a clean read of what another row costs.
    assert all(t.waste_fraction == 0.0 for t in timings)
    assert all(t.goodput_tps == t.issued_tps for t in timings)
    assert all(t.useful_tokens == t.batch_size * 2 for t in timings)


def test_the_sweep_scales_goodput_with_the_rows_it_added():
    """Not a speed claim: on CPU the step grows too. Only that the count is right."""
    model, _ = _model()
    # Two steps per size, all 1.0s: goodput is then exactly rows*2/2 = rows.
    ticks = []
    t = 0.0
    for _ in range(3):  # three batch sizes
        for _ in range(3):  # prefill + two steps
            ticks += [t, t + 1.0]
            t += 1.0
    scaling, _ = sweep_batch_sizes(
        model, [1, 2, 3], [1, 2, 4], max_new_tokens=3, clock=FakeClock(ticks)
    )

    assert scaling.baseline_tps == pytest.approx(1.0)
    assert scaling.speedup(4) == pytest.approx(4.0)
    assert scaling.efficiency(4) == pytest.approx(1.0)
    assert scaling.best_size == 4


# --- refusals -----------------------------------------------------------------


def test_a_step_budget_must_cover_every_row():
    model, _ = _model()
    with pytest.raises(ValueError, match="one step budget per prompt"):
        build_batched_decode(model, PROMPTS, max_new_tokens=4, stop_steps=[1, 2])


def test_a_step_budget_past_the_cap_would_never_be_reached():
    model, _ = _model()
    with pytest.raises(ValueError, match="max_new_tokens"):
        build_batched_decode(model, PROMPTS, max_new_tokens=4, stop_steps=[1, 9, 2])


def test_max_new_tokens_must_be_at_least_one():
    model, _ = _model()
    with pytest.raises(ValueError, match="max_new_tokens"):
        build_batched_decode(model, PROMPTS, max_new_tokens=0)


def test_the_sweep_needs_the_single_row_baseline():
    model, _ = _model()
    with pytest.raises(ValueError, match="batch size 1"):
        sweep_batch_sizes(model, [1, 2, 3], [2, 4], max_new_tokens=3)
