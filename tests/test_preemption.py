"""Day 33 tests: blocks handed out a step at a time, and what happens when they run out.

Day 30 admitted a request by reserving `worst_case_tokens`, which made a running
request unkillable: it already owned every block it could ever need, so nothing
could fail mid-flight. Today that reservation is gone. Admission buys only the
blocks the request's *current* tokens need, growth happens one block at a time at
the top of an iteration, and the failure the worst case ruled out is now real and
has to be recovered from rather than avoided.

The file has the same two halves as Day 32's. First the scheduler over plain
integers, because allocation, victim choice and queue order are bookkeeping and a
model would only slow the tests down. Then the engine on a tiny random `LlamaModel`,
because the recompute is not bookkeeping: a preempted request throws away real K/V
and re-runs a prefill over prompt plus everything it has generated, and the only
claim that matters is that the text does not change.

Four failure modes get their own tests, and all four are quiet:

  1. **The stranded block.** A preempted request that frees its blocks but keeps
     its ids, or frees them twice, corrupts the pool rather than the answer.
  2. **The stolen block.** A newly admitted request taking the block a running one
     needs on its next step, which turns admission into a preemption.
  3. **The dirty row.** A preempted slot handed to somebody else while the cache
     row still holds the old tenant's tokens.
  4. **The livelock.** Preempting the request that was about to make progress, so
     the engine spends forever recomputing and never finishes anything.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.cache import BlockAllocator, KVCacheExhausted
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import IllegalTransition, Request, RequestState, Scheduler

from reference import PROMPT_IDS, WEIGHTS_DIR, requires_weights


def _req(request_id="r0", prompt=(1, 2, 3, 4), max_new_tokens=8, eos_token_id=None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


def _sched(num_blocks=3, block_size=4, max_batch_size=4, watermark=0.0) -> Scheduler:
    return Scheduler(
        BlockAllocator(num_blocks, block_size),
        max_batch_size=max_batch_size,
        watermark=watermark,
    )


def _drain(scheduler: Scheduler, limit: int = 200) -> dict[str, list[str]]:
    """Run the scheduler to empty, appending one token per scheduled row.

    Returns a small trace: who was preempted, and in what order requests finished.
    """
    trace: dict[str, list[str]] = {"preempted": [], "finished": []}
    steps = 0
    while scheduler.has_unfinished():
        out = scheduler.schedule()
        trace["preempted"].extend(r.request_id for r in out.preempted)
        trace["finished"].extend(r.request_id for r in out.finished)
        for request in out.scheduled:
            request.append_token(7)
        steps += 1
        assert steps < limit, "the scheduler did not drain: livelock"
    return trace


# --- admission buys the prompt, not the budget --------------------------------


def test_admission_reserves_what_the_request_holds_now_not_its_worst_case():
    """The Week-9 change, in one assertion."""
    s = _sched(num_blocks=8, block_size=4)
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=16))  # worst case 19 tokens

    out = s.schedule()

    # 3 prompt tokens is one block. The old policy took five (19 tokens).
    assert len(out.admitted[0].block_ids) == 1
    assert s.allocator.num_free == 7


def test_a_request_the_worst_case_would_have_blocked_is_admitted_now():
    """Same pool, same requests: reserving the prompt fits four where one fit before."""
    s = _sched(num_blocks=4, block_size=4, max_batch_size=4)
    for i in range(4):
        s.add_request(_req(f"r{i}", prompt=(1, 2, 3), max_new_tokens=8))  # 11 tokens each

    out = s.schedule()

    assert [r.request_id for r in out.admitted] == ["r0", "r1", "r2", "r3"]
    assert s.allocator.num_free == 0


def test_the_one_admission_rule_is_the_tokens_the_request_holds():
    """Fresh or resumed, the question is the same: how many blocks do its tokens need?"""
    s = _sched(num_blocks=8, block_size=4)
    fresh = _req("fresh", prompt=(1,) * 5, max_new_tokens=8)  # 5 tokens -> 2 blocks
    s.add_request(fresh)

    assert s.blocks_needed_for(fresh) == 2

    fresh.transition_to(RequestState.RUNNING)
    fresh.output_token_ids.extend([9, 9, 9])  # 8 tokens -> still 2 blocks
    assert s.blocks_needed_for(fresh) == 2
    fresh.output_token_ids.append(9)  # 9 tokens -> 3
    assert s.blocks_needed_for(fresh) == 3


# --- growth -------------------------------------------------------------------


def test_a_running_request_takes_a_block_only_when_it_crosses_a_boundary():
    s = _sched(num_blocks=8, block_size=4, max_batch_size=1)
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=8))

    out = s.schedule()
    request = out.scheduled[0]
    request.append_token(7)  # 4 tokens: exactly fills block 0

    s.schedule()
    assert len(request.block_ids) == 1  # the 4th token had a home already
    request.append_token(7)  # 5 tokens: block 0 is full

    s.schedule()
    assert len(request.block_ids) == 2
    assert s.allocator.num_free == 6


def test_growth_is_one_block_at_a_time_all_the_way_up():
    s = _sched(num_blocks=8, block_size=4, max_batch_size=1)
    s.add_request(_req("a", prompt=(1, 2), max_new_tokens=9))
    s.schedule()

    held = []
    while s.has_unfinished():
        out = s.schedule()
        for r in out.scheduled:
            held.append(len(r.block_ids))
            r.append_token(7)

    # 2 prompt + 9 generated = 11 tokens, three blocks, taken one at a time.
    assert held == [1, 1, 1, 2, 2, 2, 2, 3, 3]
    assert max(held) == 3
    assert s.allocator.num_free == 8


def test_a_running_row_is_grown_before_a_waiting_one_is_admitted():
    """Otherwise admission steals the block a running request needs this step."""
    s = _sched(num_blocks=2, block_size=4, max_batch_size=4)
    s.add_request(_req("old", prompt=(1, 2, 3, 4), max_new_tokens=4))
    first = s.schedule()
    first.scheduled[0].append_token(7)  # 5 tokens: needs a second block next step

    s.add_request(_req("new", prompt=(1, 2), max_new_tokens=2))
    out = s.schedule()

    assert [r.request_id for r in out.admitted] == []
    assert len(out.scheduled[0].block_ids) == 2
    assert out.num_waiting == 1


# --- preemption ---------------------------------------------------------------


def test_the_youngest_running_request_is_the_victim():
    """Seniority is the policy: the oldest request is the one that must finish."""
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b", "c"):
        s.add_request(_req(name, prompt=(1, 2, 3, 4), max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)  # every row now needs a second block, and the pool is dry

    out = s.schedule()

    assert [r.request_id for r in out.preempted] == ["c", "b"]
    assert [r.request_id for r in out.scheduled] == ["a"]
    assert len(out.scheduled[0].block_ids) == 2


def test_a_preempted_request_gives_back_everything_and_keeps_its_tokens():
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)
    victim = out.scheduled[-1]
    s.allocator.allocate()  # a third party takes the spare block: now the pool is dry

    s.schedule()

    assert victim.state is RequestState.WAITING
    assert victim.block_ids == []
    assert victim.slot is None
    assert victim.output_token_ids == [7]  # the text survives; only the K/V is dropped
    assert victim.num_preemptions == 1


def test_a_preempted_request_goes_back_to_the_head_of_the_queue():
    """Ahead of anything that has never run: it is older, and it has work invested."""
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)
    s.add_request(_req("late", prompt=(1,), max_new_tokens=2))
    s.allocator.allocate()

    s.schedule()

    assert [r.request_id for r in s.waiting] == ["b", "late"]


def test_several_victims_come_back_in_arrival_order():
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b", "c"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)

    out = s.schedule()

    # Preempted youngest first, requeued at the head, so the queue is oldest first.
    assert [r.request_id for r in out.preempted] == ["c", "b"]
    assert [r.request_id for r in s.waiting] == ["b", "c"]


def test_the_oldest_running_request_is_never_a_victim():
    """The forward-progress theorem: somebody always finishes, so the pool always drains."""
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for i in range(4):
        s.add_request(_req(f"r{i}", prompt=(1,) * 4, max_new_tokens=6))

    trace = _drain(s)

    assert "r0" not in trace["preempted"]
    assert trace["finished"][0] == "r0"


def test_a_request_with_nobody_younger_to_evict_preempts_itself():
    """The last resort, and the only one that does not charge somebody else for it.

    a is grown by evicting c. b then needs a block of its own with an empty pool, an
    older request in front of it and nothing behind it, so it is its own victim.
    """
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b", "c"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)
    b = out.scheduled[1]

    out = s.schedule()

    assert b in out.preempted
    assert b.state is RequestState.WAITING
    assert b.num_preemptions == 1
    assert [r.request_id for r in out.scheduled] == ["a"]


def test_the_oldest_request_never_has_to_preempt_itself():
    """It cannot: everyone else is evicted first, and then it holds a pool it fits in."""
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))  # worst case 3 blocks
    out = s.schedule()
    for _ in range(7):
        for r in out.scheduled:
            r.append_token(7)
        out = s.schedule()

    a = out.scheduled[0]
    assert a.request_id == "a"
    assert a.num_preemptions == 0
    assert len(a.block_ids) == 3  # it ended up holding the whole pool, and it fits


def test_running_to_waiting_is_a_legal_edge_now():
    r = _req()
    r.transition_to(RequestState.RUNNING)

    r.transition_to(RequestState.WAITING)

    assert r.state is RequestState.WAITING


def test_only_a_running_request_can_be_preempted():
    with pytest.raises(IllegalTransition, match="waiting"):
        _req().preempt()


def test_preemption_is_counted_along_with_what_it_will_cost():
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for name in ("a", "b", "c"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)

    s.schedule()

    assert s.num_preemptions == 2
    # Five tokens of K/V thrown away per victim, and they have to be computed again.
    assert s.preempted_tokens == 10


# --- the invariants under pressure --------------------------------------------


def test_a_pool_under_pressure_still_drains():
    s = _sched(num_blocks=2, block_size=4, max_batch_size=4)
    for i in range(4):
        s.add_request(_req(f"r{i}", prompt=(1, 2), max_new_tokens=5))

    trace = _drain(s)

    assert sorted(trace["finished"]) == ["r0", "r1", "r2", "r3"]
    assert trace["preempted"]  # the pool really was too small to hold them all


def test_every_block_comes_back_after_a_run_with_preemptions():
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for i in range(6):
        s.add_request(_req(f"r{i}", prompt=(1,) * 3, max_new_tokens=5))

    trace = _drain(s)

    assert trace["preempted"]
    assert s.allocator.num_free == 3
    assert s.num_running == 0 and s.num_waiting == 0


def test_no_block_is_ever_held_by_two_requests():
    s = _sched(num_blocks=4, block_size=4, max_batch_size=3)
    for i in range(6):
        s.add_request(_req(f"r{i}", prompt=(1,) * 5, max_new_tokens=4))

    steps = 0
    while s.has_unfinished():
        out = s.schedule()
        held = [b for r in s.running for b in r.block_ids]
        assert len(held) == len(set(held)), "a block is in two block tables"
        assert len(held) + s.allocator.num_free == 4, "a block was lost"
        for r in out.scheduled:
            r.append_token(7)
        steps += 1
        assert steps < 200

    assert s.allocator.num_free == 4


def test_a_request_too_large_for_the_whole_pool_is_still_rejected_at_the_door():
    """Now load-bearing twice over: it is what makes self-preemption terminate."""
    s = _sched(num_blocks=4, block_size=4)

    with pytest.raises(KVCacheExhausted, match="pool"):
        s.add_request(_req("huge", prompt=(1,) * 14, max_new_tokens=8))


# --- the watermark ------------------------------------------------------------


def test_the_watermark_holds_the_last_blocks_back_from_admission():
    """A newcomer that takes the pool's last block is a preemption waiting to happen."""
    s = _sched(num_blocks=10, block_size=4, max_batch_size=4, watermark=0.2)
    s.add_request(_req("a", prompt=(1,) * 20, max_new_tokens=8))  # 5 blocks
    s.add_request(_req("b", prompt=(1,) * 12, max_new_tokens=8))  # 3 blocks

    out = s.schedule()

    # 5 + 3 = 8 of 10 blocks would leave 2, which is exactly the watermark, so both
    # fit; a third request needing 2 more would not.
    assert [r.request_id for r in out.admitted] == ["a", "b"]
    s.add_request(_req("c", prompt=(1,) * 5, max_new_tokens=4))
    assert s.schedule().admitted == ()
    assert s.allocator.num_free == 2


def test_growth_may_dip_into_the_watermark():
    """It is an admission brake, not a reserve: a running row is never starved by it."""
    s = _sched(num_blocks=4, block_size=4, max_batch_size=2, watermark=0.5)
    s.add_request(_req("a", prompt=(1,) * 4, max_new_tokens=12))
    out = s.schedule()
    for _ in range(9):
        for r in out.scheduled:
            r.append_token(7)
        out = s.schedule()

    assert len(out.scheduled[0].block_ids) == 4
    assert s.allocator.num_free == 0


def test_an_impossible_watermark_is_rejected():
    allocator = BlockAllocator(8, 4)
    with pytest.raises(ValueError, match="watermark"):
        Scheduler(allocator, watermark=1.0)
    with pytest.raises(ValueError, match="watermark"):
        Scheduler(allocator, watermark=-0.1)


# --- what the pool is really holding ------------------------------------------


def test_reservation_waste_is_zero_under_incremental_allocation():
    """The Day-30 number this day exists to kill: no block is held for a maybe."""
    s = _sched(num_blocks=32, block_size=4, max_batch_size=2)
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=16))
    s.schedule()

    assert s.reserved_blocks == 1
    assert s.reservation_waste == pytest.approx(0.0)


def test_fragmentation_waste_is_the_unfilled_tail_of_the_last_block():
    """What paging cannot remove: the last block of a sequence is rarely full."""
    s = _sched(num_blocks=32, block_size=4, max_batch_size=2)
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=16))
    s.schedule()

    # One block of 4 token slots holding 3 tokens.
    assert s.fragmentation_waste == pytest.approx(0.25)


def test_fragmentation_waste_is_bounded_by_one_block_per_sequence():
    s = _sched(num_blocks=32, block_size=4, max_batch_size=4)
    for i in range(4):
        s.add_request(_req(f"r{i}", prompt=(1,) * 9, max_new_tokens=8))
    s.schedule()

    # Four rows of 9 tokens in 3 blocks each: 12 slots held, 9 filled.
    assert s.fragmentation_waste == pytest.approx(3 / 12)
    assert s.fragmentation_waste < 1.0


def test_fragmentation_waste_is_zero_with_nothing_running():
    assert _sched().fragmentation_waste == 0.0


# --- the slot release hook ----------------------------------------------------


def test_the_release_hook_fires_when_a_request_finishes():
    released: list[int] = []
    s = _sched(num_blocks=8, block_size=4, max_batch_size=2)
    s.on_release = released.append
    s.add_request(_req("a", prompt=(1, 2), max_new_tokens=1))
    out = s.schedule()
    out.scheduled[0].append_token(7)

    s.schedule()

    assert released == [0]


def test_the_release_hook_fires_before_the_slot_is_handed_on():
    """The cache row has to be emptied while the scheduler still knows whose it was."""
    seen: list[tuple[str, int]] = []
    s = _sched(num_blocks=3, block_size=4, max_batch_size=2)
    s.on_release = lambda slot: seen.append(("release", slot))
    for name in ("a", "b"):
        s.add_request(_req(name, prompt=(1,) * 4, max_new_tokens=8))
    out = s.schedule()
    for r in out.scheduled:
        r.append_token(7)
    s.allocator.allocate()

    out = s.schedule()
    for r in out.admitted:
        seen.append(("admit", r.slot))

    assert seen == [("release", 1)]


# --- the engine ---------------------------------------------------------------


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


def test_the_cache_row_sees_the_block_the_scheduler_just_added():
    """Two lists have to stay one truth: the request's reservation and the row's table."""
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=1)
    engine.add_request(_req("a", prompt=(1, 2, 3, 4), max_new_tokens=8))

    engine.step()
    for _ in range(3):
        engine.step()
        request = engine.scheduler.running[0]
        assert engine.cache.tables[request.slot].block_ids == request.block_ids

    assert len(engine.scheduler.running[0].block_ids) == 2


def test_a_row_grows_a_block_at_a_time_instead_of_reserving_the_budget():
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=1)
    engine.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=8))

    engine.step()
    assert engine.allocator.num_free == 7  # one block, not the three the budget needs

    while engine.has_unfinished():
        engine.step()

    assert engine.allocator.num_free == 8


def test_a_preempted_request_finishes_with_the_text_it_would_have_emitted_alone():
    """The whole point. Recompute is a memory decision; it must not be a model decision."""
    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    model = _model()

    roomy = _engine(num_blocks=64, block_size=4, max_batch_size=4, model=model)
    expected = roomy.generate(prompts, max_new_tokens=6)

    cramped = _engine(num_blocks=3, block_size=4, max_batch_size=4, model=model)
    got = cramped.generate(prompts, max_new_tokens=6)

    assert cramped.scheduler.num_preemptions > 0, "this pool was supposed to be too small"
    assert got == expected


def test_a_preempted_row_is_emptied_before_the_next_request_lands_on_it():
    """The silent one: a re-prefill onto a dirty row attends over its own stale K/V."""
    engine = _engine(num_blocks=3, block_size=4, max_batch_size=2)
    engine.add_request(_req("a", prompt=(1, 2, 3, 4), max_new_tokens=8))
    engine.add_request(_req("b", prompt=(5, 6, 7, 8), max_new_tokens=8))

    engine.run_to_completion()

    assert engine.scheduler.num_preemptions > 0
    assert engine.cache.seq_lens == [0, 0]
    assert engine.allocator.num_free == 3


def test_the_recompute_is_a_prefill_over_the_prompt_plus_what_was_generated():
    """One preemption, priced in both currencies: K/V dropped, and K/V paid for again."""
    engine = _engine(num_blocks=2, block_size=4, max_batch_size=2)
    engine.add_request(_req("a", prompt=(1, 2, 3, 4), max_new_tokens=4))
    engine.add_request(_req("b", prompt=(5, 6, 7, 8), max_new_tokens=4))

    done = engine.run_to_completion()

    assert engine.scheduler.num_preemptions == 1
    # b held 4 prompt tokens and 1 generated one when it lost its blocks, and its
    # re-prefill computed all five again. The engine counts what it paid; the
    # scheduler counts what it threw away.
    assert engine.scheduler.preempted_tokens == 5
    assert engine.recomputed_tokens == 5
    assert [r.request_id for r in done] == ["a", "b"]
    assert all(len(r.token_ids) == 8 for r in done)


def test_a_pool_too_small_for_the_batch_drains_instead_of_raising():
    engine = _engine(num_blocks=2, block_size=4, max_batch_size=4)
    for i in range(4):
        engine.add_request(_req(f"r{i}", prompt=(1, 2, 3), max_new_tokens=4))

    rows = [r.token_ids for r in engine.run_to_completion()]

    assert len(rows) == 4
    assert all(len(row) == 7 for row in rows)


def test_no_forward_computes_a_token_nobody_collects_even_under_preemption():
    """Day 29's waste fraction survives Week 9: a re-prefill still emits its token."""
    engine = _engine(num_blocks=3, block_size=4, max_batch_size=4)
    for i in range(5):
        engine.add_request(_req(f"r{i}", prompt=(1, 2), max_new_tokens=4))

    engine.run_to_completion()

    assert engine.scheduler.num_preemptions > 0
    assert engine.waste_fraction == 0.0
    assert engine.collected_tokens == 5 * 4


@requires_weights
def test_preemption_does_not_change_the_text_on_llama():
    """A tiny random model forgives a lot; the real 1B does not."""
    from nanoserve.loader import load_weights

    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    prompts = [PROMPT_IDS, PROMPT_IDS[:4]]

    roomy = Engine.build(model, num_blocks=64, block_size=4, max_batch_size=2)
    expected = roomy.generate(prompts, max_new_tokens=4)

    cramped = Engine.build(model, num_blocks=3, block_size=4, max_batch_size=2)
    got = cramped.generate(prompts, max_new_tokens=4)

    assert cramped.scheduler.num_preemptions > 0
    assert got == expected
