"""Day 35 tests: the invariants the engine is supposed to have, checked every step.

Week 9 built preemption (Day 33) and priced it (Day 34). What it never did is run
the thing hard. Every preemption test so far scripted a specific eviction and
asserted a specific outcome; this file does the opposite, and it is the shape of
test the last two days earned: put far more requests than slots through a pool too
small for them, keep going until requests have been evicted and resumed several
times each, and after *every* iteration ask the engine to prove it is still whole.

The four claims Day 34 listed, in the order they fail quietly:

  1. **The text never changes.** Recompute is a memory decision. If it is also a
     model decision the run still finishes, still reports plausible numbers, and
     returns different sentences than a roomy pool would.
  2. **No block goes missing.** The pool is a closed ledger: every block is free,
     or held by exactly one request, and never both and never neither. A stranded
     block is invisible until the pool is mysteriously smaller than it was.
  3. **Every slot comes back.** A slot is a cache row, so a slot that leaks costs
     concurrency and a slot handed out twice costs correctness.
  4. **The engine keeps making progress.** An iteration that emits nothing,
     releases nothing and still has work queued is a livelock, and the only
     difference between that and a slow run is how long you are willing to wait.

The file has the usual two halves. First `audit_*` over a scheduler of plain
integers, where each check is proved to fire by corrupting exactly one thing, since
an assertion nobody has ever seen fail is a comment. Then the real engine on a tiny
random `LlamaModel`, driven by `soak`, which is the audit in a loop.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.audit import (
    InvariantViolation,
    SoakReport,
    audit_engine,
    audit_pool,
    audit_requests,
    audit_rows,
    audit_scheduler,
    audit_slots,
    soak,
)
from nanoserve.cache import BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request, RequestState, Scheduler

from reference import PROMPT_IDS, WEIGHTS_DIR, requires_weights


def _req(request_id="r0", prompt=(1, 2, 3, 4), max_new_tokens=8, eos_token_id=None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


def _sched(num_blocks=8, block_size=4, max_batch_size=4, watermark=0.0) -> Scheduler:
    return Scheduler(
        BlockAllocator(num_blocks, block_size),
        max_batch_size=max_batch_size,
        watermark=watermark,
    )


def _one_running(prompt=(1, 2, 3, 4, 5), **kwargs) -> Scheduler:
    """A scheduler with a single admitted request, sitting at a clean observation point."""
    s = _sched(**kwargs)
    s.add_request(_req("a", prompt=prompt, max_new_tokens=8))
    s.schedule()
    return s


# --- the pool is a closed ledger ----------------------------------------------


def test_a_healthy_scheduler_passes_every_check():
    s = _one_running()
    s.add_request(_req("b", prompt=(6, 7), max_new_tokens=8))
    s.schedule()

    audit_scheduler(s)  # no raise is the assertion


def test_a_block_held_by_two_requests_is_caught():
    """The bug a double allocation makes: two sequences writing over each other."""
    s = _one_running()
    s.add_request(_req("b", prompt=(6, 7), max_new_tokens=8))
    s.schedule()
    a, b = s.running

    b.block_ids[0] = a.block_ids[0]

    with pytest.raises(InvariantViolation, match="held by both"):
        audit_pool(s)


def test_a_block_nobody_holds_is_caught():
    """A stranded block: out of the pool, owned by no request, never coming back."""
    s = _one_running()
    s.running[0].block_ids.pop()

    with pytest.raises(InvariantViolation, match="allocated but no request holds"):
        audit_pool(s)


def test_a_block_a_request_holds_but_the_pool_thinks_is_free_is_caught():
    """The other direction: the allocator is about to hand out a live sequence's block."""
    s = _one_running()
    s.allocator.free(s.running[0].block_ids[0])

    with pytest.raises(InvariantViolation, match="free and held at once"):
        audit_pool(s)


def test_a_block_that_is_free_twice_is_caught():
    s = _one_running()
    s.allocator._free.append(s.allocator._free[0])

    with pytest.raises(InvariantViolation, match="free twice"):
        audit_pool(s)


def test_a_pool_that_lost_a_block_entirely_is_caught():
    """Neither free nor held: the count says the pool shrank and nothing says why."""
    s = _one_running()
    s.allocator._free.pop()

    with pytest.raises(InvariantViolation, match="neither free nor held"):
        audit_pool(s)


# --- a slot is a cache row ----------------------------------------------------


def test_a_slot_held_by_two_requests_is_caught():
    s = _one_running()
    s.add_request(_req("b", prompt=(6, 7), max_new_tokens=8))
    s.schedule()
    a, b = s.running

    b.slot = a.slot

    with pytest.raises(InvariantViolation, match="slot 0 is held by both"):
        audit_slots(s)


def test_a_slot_that_never_came_back_is_caught():
    s = _one_running(max_batch_size=4)
    s._free_slots.remove(3)

    with pytest.raises(InvariantViolation, match="neither free nor held"):
        audit_slots(s)


def test_a_slot_handed_out_while_it_was_still_free_is_caught():
    s = _one_running(max_batch_size=4)
    s._free_slots.append(s.running[0].slot)

    with pytest.raises(InvariantViolation, match="free and held at once"):
        audit_slots(s)


def test_a_running_request_without_a_slot_is_caught():
    s = _one_running()
    s.running[0].slot = None

    with pytest.raises(InvariantViolation, match="is running with no slot"):
        audit_slots(s)


# --- requests, their state, and what that state is allowed to hold ------------


def test_a_waiting_request_still_holding_blocks_is_caught():
    """What a preemption that forgets to free looks like from the outside."""
    s = _one_running()
    s.add_request(_req("b", prompt=(6, 7), max_new_tokens=8))
    b = s.waiting[0]
    b.block_ids = [0]

    with pytest.raises(InvariantViolation, match="is waiting but holds 1 block"):
        audit_requests(s)


def test_a_waiting_request_still_holding_a_slot_is_caught():
    s = _one_running()
    s.add_request(_req("b", prompt=(6, 7), max_new_tokens=8))
    s.waiting[0].slot = 2

    with pytest.raises(InvariantViolation, match="is waiting but holds slot 2"):
        audit_requests(s)


def test_a_request_in_the_wrong_queue_is_caught():
    s = _one_running()
    s.running[0].state = RequestState.WAITING

    with pytest.raises(InvariantViolation, match="is in the running set but is waiting"):
        audit_requests(s)


def test_a_request_whose_blocks_do_not_cover_its_tokens_is_caught():
    """The failure that writes K/V into a block belonging to somebody else."""
    s = _one_running(prompt=tuple(range(9)))  # 9 tokens, 3 blocks of 4
    del s.running[0].block_ids[1:]

    with pytest.raises(InvariantViolation, match="tokens but its 1 block"):
        audit_requests(s)


def test_a_request_holding_more_blocks_than_its_tokens_need_is_caught():
    """Day 33's claim, as an invariant: nothing is held for a maybe."""
    s = _one_running(prompt=(1, 2, 3, 4, 5))
    s.running[0].block_ids.append(s.allocator.allocate())

    with pytest.raises(InvariantViolation, match="more blocks than its"):
        audit_requests(s)


def test_the_waiting_queue_must_be_in_arrival_order():
    """The invariant `_admit` assumes: the head of the queue is the oldest request."""
    s = _sched()
    s.add_request(_req("a", prompt=(1, 2)))
    s.add_request(_req("b", prompt=(3, 4)))
    s.waiting.reverse()

    with pytest.raises(InvariantViolation, match="out of arrival order"):
        audit_requests(s)


def test_a_younger_request_running_while_an_older_one_waits_is_caught():
    """FIFO admission does not skip a blocked head, and this is how you find out it did."""
    s = _one_running()
    s.add_request(_req("b", prompt=(6, 7), max_new_tokens=8))
    s.running[0].arrival, s.waiting[0].arrival = s.waiting[0].arrival, s.running[0].arrival

    with pytest.raises(InvariantViolation, match="jumped the queue"):
        audit_requests(s)


def test_the_id_index_must_match_the_queues():
    """A stale id is an abort that hits a dead object; a missing one is a KeyError."""
    s = _one_running()
    s._by_id.pop("a")

    with pytest.raises(InvariantViolation, match="not in the id index"):
        audit_requests(s)


def test_a_finished_request_awaiting_its_reap_is_not_a_violation():
    """The one legal moment a FINISHED request still holds a slot and blocks."""
    s = _one_running()
    s.running[0].finish("length")

    audit_scheduler(s)


def test_the_audit_holds_at_every_step_of_a_pressured_drain():
    """No corruption injected: just a small pool, and the ledger checked all the way down."""
    s = _sched(num_blocks=4, block_size=4, max_batch_size=3)
    for i in range(6):
        s.add_request(_req(f"r{i}", prompt=(1, 2, 3), max_new_tokens=5))

    steps = 0
    while s.has_unfinished():
        out = s.schedule()
        audit_scheduler(s)
        for request in out.scheduled:
            request.append_token(7)
        audit_scheduler(s)
        steps += 1
        assert steps < 500

    assert s.num_preemptions > 0, "this pool was supposed to be too small"
    assert s.allocator.num_free == 4


# --- the engine: rows, and what a slot's row is allowed to hold ---------------


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(seed: int = 0) -> LlamaModel:
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _engine(num_blocks=8, block_size=4, max_batch_size=4, model=None) -> Engine:
    return Engine.build(
        model if model is not None else _model(),
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=max_batch_size,
    )


def _reqs(n: int, prompt_len: int = 3, max_new_tokens: int = 10) -> list[Request]:
    return [
        _req(f"r{i}", prompt=tuple(range(1, prompt_len + 1)), max_new_tokens=max_new_tokens)
        for i in range(n)
    ]


def test_a_healthy_engine_passes_the_row_audit():
    engine = _engine()
    engine.add_request(_req("a", prompt=(1, 2, 3, 4, 5)))
    engine.step()

    audit_engine(engine)


def test_a_dirty_row_on_a_free_slot_is_caught():
    """Day 31's silent bug: the next tenant of this slot attends over a stranger."""
    engine = _engine(max_batch_size=4)
    engine.add_request(_req("a", prompt=(1, 2, 3, 4, 5)))
    engine.step()
    engine.cache.tables[3].adopt([7])

    with pytest.raises(InvariantViolation, match="slot 3 is free but its cache row"):
        audit_rows(engine)


def test_a_row_that_lags_its_request_by_more_than_one_block_is_caught():
    """The sync the engine owes: one block of drift is in flight, two is a lost write."""
    engine = _engine()
    engine.add_request(_req("a", prompt=tuple(range(9))))
    engine.step()
    table = engine.cache.tables[engine.scheduler.running[0].slot]
    del table.block_ids[1:]

    with pytest.raises(InvariantViolation, match="blocks behind"):
        audit_rows(engine)


def test_a_row_addressing_a_block_its_request_does_not_hold_is_caught():
    engine = _engine()
    engine.add_request(_req("a", prompt=(1, 2, 3, 4, 5)))
    engine.step()
    table = engine.cache.tables[engine.scheduler.running[0].slot]
    table.block_ids[0] = 7

    with pytest.raises(InvariantViolation, match="does not match the blocks"):
        audit_rows(engine)


def test_a_cache_row_holding_more_tokens_than_its_request_is_caught():
    """The cache trails the request by exactly one token; ahead of it means somebody else's."""
    engine = _engine()
    engine.add_request(_req("a", prompt=(1, 2, 3, 4, 5)))
    engine.step()
    engine.cache.tables[engine.scheduler.running[0].slot].num_tokens += 4

    with pytest.raises(InvariantViolation, match="ahead of"):
        audit_rows(engine)


def test_the_engine_audit_holds_after_every_step_under_pressure():
    engine = _engine(num_blocks=4, block_size=4, max_batch_size=3)
    for request in _reqs(6, prompt_len=3, max_new_tokens=6):
        engine.add_request(request)

    steps = 0
    while engine.has_unfinished():
        engine.step()
        audit_engine(engine)
        steps += 1
        assert steps < 500

    assert engine.scheduler.num_preemptions > 0


# --- the soak -----------------------------------------------------------------


def test_the_soak_drains_far_more_requests_than_there_are_slots():
    engine = _engine(num_blocks=10, block_size=4, max_batch_size=3)

    report = soak(engine, _reqs(12, prompt_len=4, max_new_tokens=8))

    assert isinstance(report, SoakReport)
    assert report.num_requests == 12
    assert len(report.finish_order) == 12
    assert all(len(text) == 12 for text in report.texts.values())


def test_the_soak_returns_every_block_and_every_slot():
    engine = _engine(num_blocks=10, block_size=4, max_batch_size=3)

    report = soak(engine, _reqs(12, prompt_len=4, max_new_tokens=8))

    assert report.pool_returned
    assert report.slots_returned
    assert engine.allocator.num_free == 10
    assert engine.cache.seq_lens == [0, 0, 0]


def test_the_soak_evicts_the_same_request_more_than_once():
    """Otherwise this is not a stress test, it is a long test."""
    engine = _engine(num_blocks=6, block_size=4, max_batch_size=4)

    report = soak(engine, _reqs(10, prompt_len=4, max_new_tokens=10))

    assert report.max_preemptions >= 2
    assert report.num_preemptions > report.num_requests
    assert report.recompute_fraction > 0.0


def test_the_soak_produces_the_text_a_roomy_pool_produces():
    """The claim the other three exist to protect."""
    model = _model()
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [2, 4], [1, 1, 1, 1, 1], [9, 8, 7]]
    roomy = _engine(num_blocks=64, block_size=4, max_batch_size=6, model=model)
    expected = roomy.generate(prompts, max_new_tokens=9)

    cramped = _engine(num_blocks=5, block_size=4, max_batch_size=3, model=model)
    requests = [
        _req(f"r{i}", prompt=tuple(p), max_new_tokens=9) for i, p in enumerate(prompts)
    ]
    report = soak(cramped, requests)

    assert report.num_preemptions > 0
    assert [report.texts[f"r{i}"] for i in range(len(prompts))] == expected


def test_the_soak_catches_a_leaked_block_the_step_after_it_leaks():
    """The harness has teeth: break the release and the run stops, it does not limp."""
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=3)
    engine.scheduler.allocator.free_all = lambda block_ids: None

    with pytest.raises(InvariantViolation, match="allocated but no request holds"):
        soak(engine, _reqs(6, prompt_len=4, max_new_tokens=6))


def test_the_soak_catches_a_row_that_was_not_emptied():
    """The dirty row again, this time found by the loop rather than by hand.

    One request per slot and a pool nobody has to share, so the only thing that can
    go wrong is the release: the rows fall out of the batch as their requests finish
    and none of them is ever emptied.
    """
    engine = _engine(num_blocks=16, block_size=4, max_batch_size=4)
    engine.scheduler.on_release = lambda slot: None

    with pytest.raises(InvariantViolation, match="cache row"):
        soak(engine, _reqs(4, prompt_len=4, max_new_tokens=6))


def test_a_soak_that_does_not_drain_is_caught_rather_than_run_forever():
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=2)

    with pytest.raises(InvariantViolation, match="did not drain"):
        soak(engine, _reqs(6, prompt_len=4, max_new_tokens=8), max_iterations=5)


def test_a_soak_that_stops_making_progress_is_caught():
    """An iteration that emits nothing, releases nothing, and still has work queued."""
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=2)
    requests = _reqs(4, prompt_len=4, max_new_tokens=6)
    # An engine that schedules and never forwards: the batch is decided, no token is
    # ever sampled, so nothing finishes, nothing is released, and the queue stays.
    engine.step = engine.scheduler.schedule

    with pytest.raises(InvariantViolation, match="made no progress"):
        soak(engine, requests, max_iterations=50)


def test_the_audit_interval_is_a_knob_and_the_default_is_every_step():
    engine = _engine(num_blocks=10, block_size=4, max_batch_size=3)
    every = soak(engine, _reqs(6, prompt_len=4, max_new_tokens=6))

    engine = _engine(num_blocks=10, block_size=4, max_batch_size=3)
    sparse = soak(engine, _reqs(6, prompt_len=4, max_new_tokens=6), audit_every=4)

    assert every.iterations == sparse.iterations
    assert every.audits == every.iterations + 1  # every step, plus the final state
    assert sparse.audits < every.audits


def test_the_soak_reports_what_the_pressure_cost():
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=3)

    report = soak(engine, _reqs(8, prompt_len=4, max_new_tokens=8))

    assert report.resumed_requests > 0
    assert 0.0 < report.resumed_fraction <= 1.0
    assert report.recomputed_tokens > 0
    assert report.forward_tokens > report.recomputed_tokens
    assert report.iterations > 8


def test_a_request_id_is_not_allowed_to_repeat_in_one_soak():
    engine = _engine()
    duplicated = _reqs(2)
    duplicated[1].request_id = duplicated[0].request_id

    with pytest.raises(ValueError, match="already known"):
        soak(engine, duplicated)


@requires_weights
def test_the_soak_does_not_change_llamas_text():
    """A tiny random model forgives a lot; the real 1B does not."""
    from nanoserve.loader import load_weights

    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    prompts = [PROMPT_IDS, PROMPT_IDS[:5], PROMPT_IDS[:3]]

    roomy = Engine.build(model, num_blocks=64, block_size=4, max_batch_size=3)
    expected = roomy.generate(prompts, max_new_tokens=5)

    cramped = Engine.build(model, num_blocks=4, block_size=4, max_batch_size=2)
    requests = [
        _req(f"r{i}", prompt=tuple(p), max_new_tokens=5) for i, p in enumerate(prompts)
    ]
    report = soak(cramped, requests)

    assert report.num_preemptions > 0
    assert report.pool_returned and report.slots_returned
    assert [report.texts[f"r{i}"] for i in range(len(prompts))] == expected
