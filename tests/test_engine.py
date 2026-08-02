"""Day 31 tests: the scheduler wired to the model, one iteration at a time.

Day 30 built the state machine and the two queues and ran them over integers,
because every decision in that file was bookkeeping. This file is the other half:
the same scheduler with a real `LlamaModel` (tiny random weights, so it is fast and
needs no `./weights`) and a real `BatchedPagedKVCache` behind it, driven one
`Engine.step()` at a time.

The claim under test is narrow and total. Continuous batching is a *throughput*
change, so every request must emit exactly the tokens it would have emitted alone,
no matter who it shared a forward with, who left the batch halfway through its
generation, or whose cache row it inherited. Anything else is not a faster engine,
it is a different model.

Three failure modes get their own tests, because all three are silent:

  1. **The inherited row.** A slot is a cache row, and the next request to land on
     it must start from an empty history. Miss the reset and the new tenant attends
     over the last tenant's tokens, which is a fluent, plausible, wrong answer.
  2. **The double reservation.** The scheduler already reserved this request's
     blocks; if the cache row allocates its own as it grows, the pool is booked
     twice and the second booking fails mid-flight, which is the one thing the
     worst-case reservation was supposed to make impossible.
  3. **The stale batch.** The forward covers whichever rows the scheduler picked,
     and that set changes every iteration. A row that is not in this step's batch
     must not grow, must not be read, and must not shift anyone else's addressing.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request, Scheduler

from reference import PROMPT_IDS, WEIGHTS_DIR, requires_weights


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


def _engine(model, num_blocks=64, block_size=4, max_batch_size=4) -> Engine:
    return Engine.build(
        model, num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def _req(request_id, prompt, max_new_tokens=4, eos_token_id=None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


PROMPTS = [[1, 2, 3], [4], [5, 6]]


# --- one iteration ------------------------------------------------------------


def test_an_engine_with_nothing_queued_steps_to_an_empty_output():
    engine = _engine(_model()[0])

    out = engine.step()

    assert out.is_empty
    assert engine.iterations == 0  # an empty step is not a forward pass
    assert not engine.has_unfinished()


def test_the_first_step_prefills_and_emits_one_token_per_admitted_row():
    model, _ = _model()
    engine = _engine(model)
    for i, prompt in enumerate(PROMPTS):
        engine.add_request(_req(f"r{i}", prompt))

    out = engine.step()

    assert [r.request_id for r in out.admitted] == ["r0", "r1", "r2"]
    assert all(r.num_output_tokens == 1 for r in out.scheduled)
    # The cache holds the prompts and nothing else: the token just emitted has not
    # been forwarded yet, so it has no K/V.
    assert engine.cache.seq_lens[:3] == [3, 1, 2]


def test_an_admitted_row_runs_on_the_blocks_the_scheduler_reserved():
    """One reservation per request, made by the scheduler, borrowed by the cache row."""
    model, _ = _model()
    engine = _engine(model, num_blocks=16, block_size=4)
    engine.add_request(_req("r0", [1, 2, 3], max_new_tokens=4))

    out = engine.step()
    request = out.admitted[0]

    assert engine.cache.tables[request.slot].block_ids == request.block_ids
    assert len(request.block_ids) == 2  # 3 + 4 tokens worst case, 4 per block
    assert engine.allocator.num_free == 14


def test_a_running_request_never_asks_the_pool_for_another_block():
    """The reservation's whole promise: nothing can fail once it is running."""
    model, _ = _model()
    engine = _engine(model, num_blocks=16, block_size=4)
    engine.add_request(_req("r0", [1, 2, 3], max_new_tokens=8))

    engine.step()
    free_while_running = engine.allocator.num_free
    while engine.has_unfinished():
        engine.step()
        assert engine.allocator.num_free >= free_while_running

    assert engine.allocator.num_free == 16  # and everything came back at the end


def test_a_scheduled_row_collects_exactly_one_token_per_iteration():
    model, _ = _model()
    engine = _engine(model)
    for i, prompt in enumerate(PROMPTS):
        engine.add_request(_req(f"r{i}", prompt, max_new_tokens=3))

    seen = []
    done_before: set[str] = set()
    while engine.has_unfinished():
        out = engine.step()
        for r in out.scheduled:
            assert r.request_id not in done_before, "a finished row stayed in the forward"
        # A request finishes *during* the step that emits its last token, so the
        # claim is about the next one: it must never be scheduled again.
        done_before.update(r.request_id for r in out.scheduled if r.is_finished)
        seen.append(out.batch_size)

    assert engine.issued_tokens == engine.collected_tokens == 9
    assert engine.waste_fraction == 0.0
    assert seen[:3] == [3, 3, 3]


def test_prefill_and_decode_happen_in_the_same_iteration():
    """A new arrival joins a batch that is already mid-generation."""
    model, _ = _model()
    engine = _engine(model, max_batch_size=4)
    engine.add_request(_req("early", [1, 2, 3], max_new_tokens=6))
    engine.step()

    engine.add_request(_req("late", [5, 6], max_new_tokens=3))
    out = engine.step()

    assert [r.request_id for r in out.prefill] == ["late"]
    assert [r.request_id for r in out.decode] == ["early"]
    assert engine.cache.seq_lens[:2] == [4, 2]  # early cached its first token, late its prompt


# --- the tokens are the tokens ------------------------------------------------


def test_the_engine_emits_what_static_batching_emits():
    """One batch, admitted together, decoded together: identical to Day 28's loop."""
    model, _ = _model()
    expected = model.greedy_generate_batch(PROMPTS, max_new_tokens=5)

    engine = _engine(model, max_batch_size=3)
    rows = engine.generate(PROMPTS, max_new_tokens=5)

    assert rows == expected


def test_a_request_emits_what_it_would_have_emitted_alone():
    """The invariant continuous batching must not break, at a batch size it changes."""
    model, _ = _model()
    engine = _engine(model, max_batch_size=2)

    rows = engine.generate(PROMPTS, max_new_tokens=4)

    for prompt, row in zip(PROMPTS, rows):
        assert row == model.greedy_generate_batch([prompt], max_new_tokens=4)[0]


def test_the_next_tenant_of_a_slot_generates_from_its_own_prompt_only():
    """The inherited-row bug: one slot, three requests in turn, no bleed between them.

    With a single slot every request runs alone in cache row 0 and inherits the row
    from the one before it. If the row is not reset, request 2's first decode step
    attends over request 1's K/V and emits something that looks entirely reasonable.
    """
    model, _ = _model()
    engine = _engine(model, max_batch_size=1)

    rows = engine.generate(PROMPTS, max_new_tokens=4)

    for prompt, row in zip(PROMPTS, rows):
        assert row == model.greedy_generate_batch([prompt], max_new_tokens=4)[0]
    assert engine.iterations == 12  # 3 requests x 4 tokens, one row per forward


def test_the_row_a_finished_request_leaves_is_empty_before_the_next_one_lands():
    model, _ = _model()
    engine = _engine(model, max_batch_size=1)
    engine.add_request(_req("first", [1, 2, 3], max_new_tokens=2))
    engine.add_request(_req("second", [7], max_new_tokens=2))

    engine.step()
    engine.step()  # "first" hits its budget here
    lengths_while_first_ran = engine.cache.seq_lens[0]
    engine.step()  # reset row 0, release "first", admit and prefill "second"

    assert lengths_while_first_ran == 4  # prompt + the first emitted token
    assert engine.cache.seq_lens[0] == 1  # "second" has only its own one-token prompt


def test_eos_stops_a_row_at_its_own_token():
    model, _ = _model()
    solo = model.greedy_generate_batch([[1, 2, 3]], max_new_tokens=4)[0]
    eos = solo[4]  # whatever this model emits second, made into a stop token

    engine = _engine(model)
    engine.add_request(_req("r0", [1, 2, 3], max_new_tokens=4, eos_token_id=eos))
    done = engine.run_to_completion()

    assert done[0].finish_reason == "stop"
    assert done[0].output_token_ids == solo[3:5]  # the EOS token is kept
    assert engine.issued_tokens == 2  # and nothing was issued after it


# --- what the scheduling buys -------------------------------------------------


def test_no_forward_computes_a_token_nobody_collects():
    """Day 29's waste fraction on the same ragged shape: 0.0, and 77% less work.

    Seven short rows behind one long one. A static batch runs all eight rows for as
    many steps as the longest needs, so it issues `8 * 32` token-slots to collect 60
    real tokens. Here a row is in the forward only while it still wants a token.
    """
    model, _ = _model()
    engine = _engine(model, num_blocks=64, block_size=4, max_batch_size=8)
    budgets = [4] * 7 + [32]
    for i, budget in enumerate(budgets):
        engine.add_request(_req(f"r{i}", [1, 2], max_new_tokens=budget))

    engine.run_to_completion()

    assert engine.collected_tokens == sum(budgets) == 60
    assert engine.issued_tokens == 60
    assert engine.waste_fraction == 0.0
    static_issued = len(budgets) * max(budgets)  # every row runs until the straggler
    assert static_issued == 256
    assert 1 - engine.issued_tokens / static_issued == pytest.approx(0.7656, abs=1e-3)


def test_a_short_request_is_returned_at_its_own_last_token():
    """The head-of-line half: r0 does not wait for r7's thirty-second token."""
    model, _ = _model()
    engine = _engine(model, num_blocks=64, block_size=4, max_batch_size=8)
    for i, budget in enumerate([4] * 7 + [32]):
        engine.add_request(_req(f"r{i}", [1, 2], max_new_tokens=budget))

    returned_at: dict[str, int] = {}
    while engine.has_unfinished():
        out = engine.step()
        for r in out.finished:
            returned_at[r.request_id] = engine.iterations

    assert returned_at["r0"] == 5  # its own 4 tokens, plus the step that reaps it
    assert returned_at["r7"] == 32
    assert engine.iterations == 32


def test_a_queue_drains_by_refilling_slots_not_by_waves():
    """Twelve requests through four slots, refilled the moment one comes free."""
    model, _ = _model()
    engine = _engine(model, num_blocks=64, block_size=4, max_batch_size=4)
    budgets = [2, 2, 2, 8] * 3
    for i, budget in enumerate(budgets):
        engine.add_request(_req(f"r{i}", [1, 2], max_new_tokens=budget))

    engine.run_to_completion()

    assert engine.collected_tokens == sum(budgets) == 42
    assert engine.issued_tokens == 42
    # Three static waves of four would each run for their own straggler's 8 steps:
    # 3 * 4 * 8 = 96 issued token-slots, and 24 iterations, for the same 42 tokens.
    assert engine.iterations < 24


def test_the_prefill_rectangle_is_still_padded_and_the_engine_says_so():
    """Continuous batching removes the decode bill, not the prefill one. Week 10's debt."""
    model, _ = _model()
    engine = _engine(model, max_batch_size=4)
    engine.generate([[1, 2, 3, 4, 5, 6, 7, 8], [9], [10, 11]], max_new_tokens=2)

    assert engine.prefill_tokens == 11  # the real prompt tokens
    assert engine.prefill_slots == 24  # the rectangle they were padded into: 3 x 8
    assert engine.prefill_padding_waste == pytest.approx(13 / 24)


# --- leaving early ------------------------------------------------------------


def test_an_aborted_running_request_gives_its_row_back_clean():
    model, _ = _model()
    engine = _engine(model, max_batch_size=2)
    engine.add_request(_req("a", [1, 2, 3], max_new_tokens=8))
    engine.add_request(_req("b", [4, 5], max_new_tokens=8))
    engine.step()

    engine.abort("a")
    engine.add_request(_req("c", [6], max_new_tokens=2))
    out = engine.step()

    assert [r.request_id for r in out.finished] == ["a"]
    assert out.finished[0].finish_reason == "abort"
    assert [r.request_id for r in out.prefill] == ["c"]
    assert engine.cache.seq_lens[0] == 1  # c's one-token prompt, not a's four


def test_every_block_comes_back_when_the_last_request_finishes():
    model, _ = _model()
    engine = _engine(model, num_blocks=32, block_size=4, max_batch_size=2)
    engine.generate(PROMPTS + PROMPTS, max_new_tokens=3)

    assert engine.allocator.num_free == 32
    assert engine.cache.seq_lens == [0, 0]
    assert not engine.has_unfinished()


def test_a_pool_too_small_for_the_batch_just_queues_instead_of_failing():
    """Backpressure, not a crash: what fits runs, the rest waits for a release."""
    model, _ = _model()
    engine = _engine(model, num_blocks=4, block_size=4, max_batch_size=4)
    for i in range(4):
        engine.add_request(_req(f"r{i}", [1, 2, 3], max_new_tokens=4))

    out = engine.step()
    assert out.batch_size == 2  # two blocks each, four blocks in the pool
    assert out.num_waiting == 2

    rows = [r.token_ids for r in engine.run_to_completion()]
    assert len(rows) == 4
    assert all(len(row) == 7 for row in rows)


# --- the caller's view --------------------------------------------------------


def test_generate_returns_rows_in_submission_order():
    model, _ = _model()
    engine = _engine(model, max_batch_size=2)

    rows = engine.generate(PROMPTS, max_new_tokens=3)

    assert [row[: len(p)] for row, p in zip(rows, PROMPTS)] == PROMPTS
    assert [len(row) for row in rows] == [6, 4, 5]


def test_run_to_completion_hands_requests_back_in_finish_order():
    model, _ = _model()
    engine = _engine(model, max_batch_size=4)
    engine.add_request(_req("slow", [1, 2], max_new_tokens=6))
    engine.add_request(_req("fast", [3, 4], max_new_tokens=2))

    done = engine.run_to_completion()

    assert [r.request_id for r in done] == ["fast", "slow"]
    assert [r.finish_reason for r in done] == ["length", "length"]


def test_a_runaway_loop_is_caught_rather_than_hung():
    model, _ = _model()
    engine = _engine(model)
    engine.add_request(_req("r0", [1, 2], max_new_tokens=8))

    with pytest.raises(RuntimeError, match="did not drain"):
        engine.run_to_completion(max_iterations=3)


# --- real weights -------------------------------------------------------------


@requires_weights
def test_the_scheduled_loop_matches_the_static_one_on_llama():
    """The same claim on the real 1B, where a leak across rows would be unmissable."""
    from nanoserve.loader import load_weights

    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    prompts = [PROMPT_IDS, PROMPT_IDS[:3]]

    static = model.greedy_generate_batch(prompts, max_new_tokens=4, pad_id=cfg.vocab_size - 1)

    allocator = BlockAllocator(num_blocks=64, block_size=16)
    engine = Engine(
        model,
        Scheduler(allocator, max_batch_size=1),  # one slot: row 0 is reused
        BatchedPagedKVCache(cfg, allocator, batch_size=1),
        pad_id=cfg.vocab_size - 1,
    )

    assert engine.generate(prompts, max_new_tokens=4) == static
    assert engine.issued_tokens == 8
