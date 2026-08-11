"""Day 36 tests: the scheduler edges Week 9 kept putting off, and the watermark measured.

Three things have been on the "tomorrow" line of the daily log since preemption
landed, and Day 35 finally built the harness that makes them worth writing down:

  1. **Aborting a request that is currently preempted.** It is in the middle of the
     waiting queue, its tokens are alive and its K/V is not, and the release path was
     written for a request that holds things. A request that holds nothing is the
     case a release path gets wrong quietly.
  2. **A prompt whose length is an exact multiple of the block size.** Its prefill
     fills its last block to the final slot, so its *first* decode crosses a block
     boundary, which is the growth path taken on iteration one instead of on
     iteration five. Off-by-one bugs in `blocks_for_length` live exactly here.
  3. **The watermark**, a 1% default since Day 30 that nobody had ever measured. It
     exists to stop a newcomer taking the pool's last block and being evicted for it
     one step later, which costs a prefill and buys nothing.

The first two turned out to hold, and they are here as regressions rather than as
fixes: an edge that works by accident and an edge that works by design look the same
until something asserts on it. The third did not hold, and the way it failed is the
day.

**The watermark could be a wall.** It is subtracted at admission and ignored during
growth, which is right: a running row must never be starved by an accounting rule.
But a request that grows past the watermark line and is then preempted has to be
*re-admitted*, and admission is where the subtraction happens. So it could need more
blocks than the watermark leaves, in a pool that is completely empty, forever. Both
queues stop moving, nobody is running to free anything, and the engine spins. The
fix is one line and the sentence it enforces is the invariant that was missing: the
watermark may delay a request, it may never exclude one.

The measurement is the other half. `sweep_watermarks` runs one workload through a
scheduler at several reserve sizes and counts what changed, with no model in it, the
same way the Day-14 allocator and the Day-29 timing core were measured: what the
watermark moves is preemptions and iterations, and both are integers.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.audit import audit_engine, audit_scheduler, soak
from nanoserve.batchbench import WatermarkPoint, sweep_watermarks
from nanoserve.cache import BlockAllocator, KVCacheExhausted
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request, RequestState, Scheduler

FAKE_TOKEN = 7


def _req(request_id="r0", prompt=(1, 2, 3, 4), max_new_tokens=8, eos_token_id=None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


def _sched(num_blocks=8, block_size=4, max_batch_size=4, **kwargs) -> Scheduler:
    return Scheduler(
        BlockAllocator(num_blocks, block_size), max_batch_size=max_batch_size, **kwargs
    )


def _drain(scheduler: Scheduler, max_iterations: int = 200) -> int:
    """Step a model-free scheduler to empty, appending one fake token per row.

    Returns the iteration count. Raises if it does not drain, which is the whole
    point of most of the tests below: the failure being tested is a loop that never
    ends, and a test that hangs is not a failing test.
    """
    iterations = 0
    while scheduler.has_unfinished():
        if iterations >= max_iterations:
            raise AssertionError(
                f"the scheduler did not drain in {max_iterations} iterations: "
                f"{scheduler.num_running} running, {scheduler.num_waiting} waiting, "
                f"{scheduler.allocator.num_free} blocks free"
            )
        out = scheduler.schedule()
        for request in out.scheduled:
            request.append_token(FAKE_TOKEN)
        iterations += 1
    return iterations


# --- the watermark as a wall ---------------------------------------------------


def test_a_resumed_request_that_outgrew_the_watermark_still_gets_back_in():
    """The deadlock: admitted small, grown past the line, preempted, never re-admitted.

    Nothing here is misuse. `b` was admitted holding one block, which the watermark
    was happy with; it grew to four, which the watermark is deliberately not consulted
    about; and then it was evicted. Coming back it needs those four blocks at once,
    and admission subtracts a reserve that leaves three. The pool is empty, nobody is
    running, and no future iteration can change any of those numbers.
    """
    s = _sched(num_blocks=6, block_size=4, max_batch_size=2, watermark_blocks=3)
    s.add_request(_req("a", prompt=(1, 2, 3, 4), max_new_tokens=20))
    s.add_request(_req("b", prompt=(1, 2, 3, 4), max_new_tokens=20))

    _drain(s, max_iterations=200)

    assert s.num_preemptions > 0, "this pool was supposed to be too small"
    assert s.allocator.num_free == 6


def test_a_prompt_bigger_than_the_watermark_leaves_is_admitted_rather_than_queued_forever():
    """The same wall for a request that never ran: the door said yes, admission said no.

    `add_request` refuses anything too large for the *whole* pool, so a request that
    passes the door is one the engine has promised to run. A watermark that then makes
    it unadmittable turns that promise into a hang, which is the worst of the three
    possible answers: a refusal is actionable and a slow run finishes.
    """
    s = _sched(num_blocks=10, block_size=4, max_batch_size=2, watermark_blocks=5)
    request = _req("big", prompt=tuple(range(24)), max_new_tokens=4)  # 6 blocks

    s.add_request(request)
    out = s.schedule()

    assert out.admitted == (request,)
    assert request.state is RequestState.RUNNING


def test_the_watermark_still_holds_back_a_newcomer_that_could_wait_instead():
    """The waiver is for the excluded request only, not a licence to ignore the reserve.

    `c` needs two blocks and the pool has two free, so without a reserve it would be
    admitted, and the whole design claim is that it should not be. Two blocks is well
    inside what the watermark leaves, so nothing about `c` is permanently excluded and
    the brake applies exactly as Day 30 wrote it.
    """
    s = _sched(num_blocks=10, block_size=4, max_batch_size=4, watermark_blocks=2)
    s.add_request(_req("a", prompt=(1,) * 20, max_new_tokens=8))  # 5 blocks
    s.add_request(_req("b", prompt=(1,) * 12, max_new_tokens=8))  # 3 blocks
    s.schedule()
    s.add_request(_req("c", prompt=(1,) * 5, max_new_tokens=4))  # 2 blocks

    assert s.allocator.num_free == 2
    assert s.schedule().admitted == ()


def test_the_waiver_does_not_let_a_request_take_blocks_that_are_not_there():
    """It drops the reserve, not the pool. A request still waits for real free blocks."""
    s = _sched(num_blocks=6, block_size=4, max_batch_size=2, watermark_blocks=3)
    s.add_request(_req("a", prompt=(1,) * 12, max_new_tokens=4))  # 3 blocks
    s.schedule()
    s.add_request(_req("big", prompt=(1,) * 16, max_new_tokens=4))  # 4 blocks, waived

    assert s.allocator.num_free == 3
    assert s.schedule().admitted == ()


def test_the_watermark_never_starves_the_engine_under_a_soak():
    """The general claim, with the audit watching: any reserve, and it still drains."""
    for reserve in range(0, 6):
        engine = _engine(num_blocks=6, block_size=4, max_batch_size=2)
        engine.scheduler.watermark_blocks = reserve
        report = soak(engine, _reqs(4, prompt_len=4, max_new_tokens=6))

        assert report.pool_returned and report.slots_returned
        assert report.num_requests == 4


# --- the watermark as a knob ---------------------------------------------------


def test_the_default_watermark_rounds_to_nothing_below_a_hundred_blocks():
    """The measurement nobody had made: at this project's pool sizes it is switched off.

    `int(0.01 * num_blocks)` is 0 for every pool smaller than a hundred blocks, which
    is every pool in this repo's tests and most of its benchmarks. The default is not
    wrong, it is inert, and the difference matters: a knob that has never moved has
    never been measured either.
    """
    for num_blocks in (8, 16, 64, 99):
        assert _sched(num_blocks=num_blocks).watermark_blocks == 0
    assert _sched(num_blocks=100).watermark_blocks == 1
    assert _sched(num_blocks=256).watermark_blocks == 2


def test_the_reserve_can_be_given_in_blocks_directly():
    """Blocks, not a share, because a share cannot say "one block of ten" honestly.

    `int(0.1 * 10)` is 1 and `int(0.3 * 10)` is 3, but the fraction is a float and the
    truncation is doing arithmetic the caller did not ask for. A measurement that
    sweeps the reserve needs to name the integer it is sweeping.
    """
    assert _sched(num_blocks=10, watermark_blocks=3).watermark_blocks == 3
    assert _sched(num_blocks=10, watermark_blocks=0).watermark_blocks == 0


def test_an_explicit_reserve_overrides_the_fraction():
    s = _sched(num_blocks=200, watermark=0.5, watermark_blocks=7)

    assert s.watermark_blocks == 7


def test_a_reserve_that_swallows_the_pool_is_rejected():
    """A scheduler that can never admit anything is a configuration error, not a queue."""
    with pytest.raises(ValueError, match="watermark_blocks"):
        _sched(num_blocks=8, watermark_blocks=8)
    with pytest.raises(ValueError, match="watermark_blocks"):
        _sched(num_blocks=8, watermark_blocks=-1)


def test_admissible_blocks_is_what_the_reserve_leaves():
    """The largest request admission will take without waiving anything."""
    assert _sched(num_blocks=10, watermark_blocks=3).admissible_blocks == 7
    assert _sched(num_blocks=10, watermark_blocks=0).admissible_blocks == 10


# --- aborting a request that is preempted --------------------------------------


def _preempted_scheduler() -> tuple[Scheduler, Request]:
    """Drive a small pool until somebody is evicted, and hand back the victim."""
    s = _sched(num_blocks=4, block_size=4, max_batch_size=3, watermark_blocks=0)
    for i in range(4):
        s.add_request(_req(f"r{i}", prompt=(1, 2, 3), max_new_tokens=6))
    for _ in range(50):
        out = s.schedule()
        for request in out.scheduled:
            request.append_token(FAKE_TOKEN)
        if out.preempted:
            return s, out.preempted[0]
    raise AssertionError("this pool was supposed to be too small to run four requests")


def test_a_preempted_request_holds_nothing_for_an_abort_to_release():
    """The reason this edge is safe, stated: preemption already did the releasing."""
    s, victim = _preempted_scheduler()
    free_before = s.allocator.num_free
    slots_before = s.free_slots

    s.abort(victim.request_id)

    assert victim.finish_reason == "abort"
    assert victim.block_ids == [] and victim.slot is None
    assert s.allocator.num_free == free_before
    assert s.free_slots == slots_before


def test_aborting_a_preempted_request_keeps_the_tokens_it_had_generated():
    """Its K/V is gone and its text is not, which is what recompute means. Abort is the
    one path where that text is never rebuilt, so it has to survive the abort intact
    for the caller to be handed a partial answer."""
    s, victim = _preempted_scheduler()
    generated = list(victim.output_token_ids)
    assert generated, "the victim was supposed to have run before it was evicted"

    s.abort(victim.request_id)

    assert victim.output_token_ids == generated
    assert victim.token_ids == victim.prompt_token_ids + generated


def test_a_preempted_request_is_reaped_out_of_the_middle_of_the_waiting_queue():
    """Release walks both queues, and a waiting request's release is a queue removal."""
    s, victim = _preempted_scheduler()
    s.add_request(_req("late", prompt=(1, 2, 3), max_new_tokens=4))
    assert victim in s.waiting

    s.abort(victim.request_id)
    out = s.schedule()

    assert victim in out.finished
    assert victim not in s.waiting
    assert victim.request_id not in s.known_ids
    audit_scheduler(s)


def test_the_queues_stay_in_arrival_order_after_a_preempted_request_is_aborted():
    """The invariant Day 35 wrote down, checked across the edge that reorders queues."""
    s, victim = _preempted_scheduler()
    s.abort(victim.request_id)

    while s.has_unfinished():
        out = s.schedule()
        audit_scheduler(s)
        for request in out.scheduled:
            request.append_token(FAKE_TOKEN)
        audit_scheduler(s)

    assert s.allocator.num_free == 4
    assert s.free_slots == (0, 1, 2)


def test_aborting_a_request_twice_keeps_the_first_reason():
    s = _sched()
    request = _req("a")
    s.add_request(request)
    s.schedule()

    s.abort("a")
    s.abort("a")

    assert request.finish_reason == "abort"
    assert request.state is RequestState.FINISHED


def test_aborting_everything_empties_the_engine_in_one_iteration():
    """Nothing left to schedule, every block back, and no exception on the way out."""
    engine = _engine(num_blocks=8, block_size=4, max_batch_size=3)
    for request in _reqs(5, prompt_len=5, max_new_tokens=8):
        engine.add_request(request)
    engine.step()
    for i in range(5):
        engine.abort(f"r{i}")

    out = engine.step()

    assert out.is_empty
    assert not engine.has_unfinished()
    assert engine.allocator.num_free == 8
    audit_engine(engine)


def test_the_engine_drains_after_a_preempted_request_is_aborted():
    """The tensor half of the same edge: the victim's row was emptied at eviction."""
    engine = _engine(num_blocks=4, block_size=4, max_batch_size=3)
    requests = _reqs(4, prompt_len=3, max_new_tokens=6)
    for request in requests:
        engine.add_request(request)

    aborted = None
    for _ in range(60):
        out = engine.step()
        audit_engine(engine)
        if out.preempted and aborted is None:
            aborted = out.preempted[0]
            engine.abort(aborted.request_id)
        if not engine.has_unfinished():
            break

    assert aborted is not None, "this pool was supposed to preempt"
    assert aborted.finish_reason == "abort"
    assert engine.allocator.num_free == 4
    assert engine.scheduler.free_slots == (0, 1, 2)


# --- a prompt that exactly fills its blocks ------------------------------------


def test_a_prompt_that_exactly_fills_its_blocks_holds_no_spare():
    """Four tokens in a block of four is one block, not two. The ceiling's boundary."""
    s = _sched(num_blocks=8, block_size=4, max_batch_size=2, watermark_blocks=0)
    request = _req("exact", prompt=(1, 2, 3, 4), max_new_tokens=6)
    s.add_request(request)

    s.schedule()

    assert len(request.block_ids) == 1
    assert s.blocks_needed_for(request) == 0


def test_the_first_decode_after_an_exact_fill_buys_a_block():
    """The growth path on iteration one instead of iteration five, which is the edge.

    A prompt of 5 tokens in blocks of 4 has three free slots to decode into and does
    not touch the allocator for another three iterations. A prompt of 4 has none, so
    the very first sampled token crosses a boundary, and every ordering question the
    grow-then-admit loop answers is asked immediately rather than eventually.
    """
    s = _sched(num_blocks=8, block_size=4, max_batch_size=2, watermark_blocks=0)
    request = _req("exact", prompt=(1, 2, 3, 4), max_new_tokens=6)
    s.add_request(request)
    out = s.schedule()
    for scheduled in out.scheduled:
        scheduled.append_token(FAKE_TOKEN)

    assert s.blocks_needed_for(request) == 1
    s.schedule()
    assert len(request.block_ids) == 2


def test_a_request_preempted_on_a_block_boundary_asks_for_exactly_its_blocks():
    """Resumption is `blocks_for_length(num_tokens)`, and on a boundary that is exact."""
    s = _sched(num_blocks=8, block_size=4, max_batch_size=2, watermark_blocks=0)
    request = _req("exact", prompt=(1, 2, 3, 4), max_new_tokens=6)
    s.add_request(request)
    out = s.schedule()
    for _ in range(4):
        for scheduled in out.scheduled:
            scheduled.append_token(FAKE_TOKEN)
        out = s.schedule()
    assert request.num_tokens == 8  # exactly two blocks again

    s._preempt(request)

    assert request.block_ids == []
    assert s.blocks_needed_for(request) == 2


def test_an_exact_fill_survives_a_pool_too_small_for_it():
    """The whole-engine version, audited every step, against a pool that never preempts."""
    engine = _engine(num_blocks=6, block_size=4, max_batch_size=2)
    tight = _reqs(4, prompt_len=8, max_new_tokens=6)
    report = soak(engine, tight)

    roomy = _engine(num_blocks=64, block_size=4, max_batch_size=4)
    reference = _reqs(4, prompt_len=8, max_new_tokens=6)
    for request in reference:
        roomy.add_request(request)
    roomy.run_to_completion()

    assert engine.scheduler.num_preemptions > 0, "this pool was supposed to be too small"
    assert [r.token_ids for r in tight] == [r.token_ids for r in reference]
    assert report.pool_returned


# --- the sweep: what the watermark actually buys --------------------------------


def _sweep(watermarks, num_blocks=8, block_size=4, max_batch_size=3, n=8, plen=6, budget=10):
    return sweep_watermarks(
        [list(range(1, plen + 1))] * n,
        max_new_tokens=budget,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=max_batch_size,
        watermarks=watermarks,
    )


def test_a_sweep_reports_one_point_per_setting():
    points = _sweep([0, 1, 2])

    assert [p.watermark_blocks for p in points] == [0, 1, 2]
    assert all(isinstance(p, WatermarkPoint) for p in points)
    assert all(p.num_requests == 8 and p.num_blocks == 8 for p in points)


def test_holding_blocks_back_cuts_preemptions():
    """The claim the watermark has been making since Day 30, finally with a number."""
    none, some = _sweep([0, 3])

    assert none.preemptions > 0
    assert some.preemptions < none.preemptions


def test_holding_blocks_back_costs_iterations():
    """And the price: the thrash becomes waiting, which is time the caller still pays."""
    none, some = _sweep([0, 3])

    assert some.iterations > none.iterations


def test_every_request_finishes_at_every_reserve():
    """The sweep's own safety check: a reserve that hangs is not a faster reserve."""
    for point in _sweep([0, 1, 2, 3, 4]):
        assert point.completed == point.num_requests


def test_the_sweep_counts_an_admission_evicted_before_it_filled_its_block():
    """The thrash the watermark exists to remove, in the window it really happens in.

    A general preemption count cannot make the watermark's case, because a preemption
    of a request that has been running for thirty iterations is the pool being small
    and has nothing to do with admission policy. This counts the specific shape: a
    prefill the engine bought less than one block of progress with.

    The window is a block wide and not one iteration wide, which is the measurement
    correcting the Day-30 comment. Every victim in this workload was admitted, spent
    the free tail of the block its admission bought at one token per iteration, and
    was evicted when it reached the boundary. Nobody was ever evicted a step after
    being let in.
    """
    none, some = _sweep([0, 3])

    assert none.thrashed_admissions == none.preemptions
    assert some.thrashed_admissions == 0
    assert none.thrash_fraction > 0.0


def test_a_point_reports_the_recompute_share():
    point = _sweep([0])[0]

    assert point.forward_tokens > 0
    assert 0.0 < point.recompute_fraction < 1.0
    assert point.recompute_fraction == point.recomputed_tokens / point.forward_tokens


def test_a_sweep_with_no_pressure_reports_zeros():
    point = _sweep([0], num_blocks=64)[0]

    assert point.preemptions == 0
    assert point.thrashed_admissions == 0
    assert point.recompute_fraction == 0.0
    assert point.admissions == point.num_requests


def test_a_repeated_reserve_is_rejected():
    with pytest.raises(ValueError, match="repeated"):
        _sweep([1, 1])


def test_an_empty_sweep_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        _sweep([])


def test_a_reserve_the_pool_cannot_hold_is_rejected():
    with pytest.raises(ValueError, match="watermark_blocks"):
        _sweep([0, 8], num_blocks=8)


def test_the_sweep_needs_one_budget_per_prompt():
    with pytest.raises(ValueError, match="one token budget per prompt"):
        sweep_watermarks(
            [[1, 2], [3, 4]],
            max_new_tokens=[4],
            num_blocks=8,
            block_size=4,
            max_batch_size=2,
            watermarks=[0],
        )


def test_a_sweep_that_does_not_drain_is_caught_rather_than_run_forever():
    with pytest.raises(RuntimeError, match="did not drain"):
        sweep_watermarks(
            [[1, 2, 3, 4]] * 4,
            max_new_tokens=8,
            num_blocks=8,
            block_size=4,
            max_batch_size=2,
            watermarks=[0],
            max_iterations=3,
        )


def test_a_pool_too_small_for_a_request_is_refused_at_the_door_by_the_sweep():
    with pytest.raises(KVCacheExhausted):
        _sweep([0], num_blocks=2, block_size=4, n=2, plen=16, budget=8)


# --- the tiny model the engine tests run on ------------------------------------


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


def _engine(num_blocks=8, block_size=4, max_batch_size=4) -> Engine:
    return Engine.build(
        _model(), num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def _reqs(n: int, prompt_len: int = 3, max_new_tokens: int = 10) -> list[Request]:
    return [
        _req(f"r{i}", prompt=tuple(range(1, prompt_len + 1)), max_new_tokens=max_new_tokens)
        for i in range(n)
    ]
