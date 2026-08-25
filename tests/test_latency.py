"""Day 43 tests: where a request's latency went, measured inside the server.

Day 42 measured TTFT from outside the socket and could only take it apart by
subtraction: one client alone got 1.64s, the same request under load got 22.26s,
so 93% of the wait "was" queueing. That subtraction assumes the unloaded number is
the loaded one's prefill, which is a guess, and it cannot see the two components
that live between the socket and the scheduler at all.

This file pins the measured version. Three tiers, the same split every measurement
day in this repo has used.

  1. **The arithmetic**, on a manual clock, where every timestamp is a number the
     test chose. It pins the five parts of TTFT and the one property that makes
     them a decomposition rather than five numbers printed together: they sum to
     the whole, exactly, including when preemption makes two of them non-zero.
  2. **The report**, which is means for the split and percentiles for the total,
     because a p99 does not decompose and printing it as if it did is the mistake
     the whole day is about.
  3. **The wiring**, over the real scheduler, the real engine and the real bridge,
     because a timeline nobody stamps is a dataclass with zeros in it.

Nothing here needs `./weights` or a GPU. The claim is about clocks and queues.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from nanoserve.cache import BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.latency import (
    TTFT_PARTS,
    LatencyReport,
    RequestTimeline,
    TimelineBroken,
    summarize,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request, RequestState, Scheduler
from nanoserve.serving import AsyncEngine


class ManualClock:
    """A clock the test moves by hand. Every span below is therefore exact.

    Wall time in a unit test buys nothing and costs flakes: a 3ms assertion fails
    on a loaded CI box and passes on a laptop, and the thing under test is
    arithmetic over timestamps, not the timestamps themselves.
    """

    def __init__(self, start: float = 100.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def _timeline(clock: ManualClock) -> RequestTimeline:
    return RequestTimeline(clock=clock)


def _clean_run(clock: ManualClock) -> RequestTimeline:
    """The happy path, with a different number for every part.

    inbox 0.1, queue 2.0, prefill 0.5, then 3 more tokens of decode at 0.25.
    """
    tl = _timeline(clock)
    clock.advance(0.1)
    tl.on_queued()
    clock.advance(2.0)
    tl.on_admitted()
    clock.advance(0.5)
    tl.on_first_token()
    clock.advance(0.75)
    tl.on_finished()
    return tl


# --- tier 1: the arithmetic -----------------------------------------------------


def test_a_fresh_timeline_has_only_a_creation_stamp():
    clock = ManualClock()
    tl = _timeline(clock)
    assert tl.created_at == 100.0
    assert tl.queued_at is None
    assert tl.first_token_at is None
    assert tl.ttft_s is None
    assert tl.total_s is None


def test_the_inbox_is_creation_to_queueing():
    """The bridge's own latency: built in a handler, seen by the loop later."""
    clock = ManualClock()
    tl = _timeline(clock)
    clock.advance(0.4)
    tl.on_queued()
    assert tl.inbox_s == pytest.approx(0.4)
    assert tl.queue_wait_s is None


def test_the_queue_wait_is_queueing_to_the_first_admission():
    clock = ManualClock()
    tl = _clean_run(clock)
    assert tl.queue_wait_s == pytest.approx(2.0)


def test_the_prefill_is_the_admission_that_produced_the_first_token():
    clock = ManualClock()
    tl = _clean_run(clock)
    assert tl.prefill_s == pytest.approx(0.5)


def test_ttft_is_creation_to_the_first_token():
    clock = ManualClock()
    tl = _clean_run(clock)
    assert tl.ttft_s == pytest.approx(2.6)


def test_the_five_parts_sum_to_ttft_exactly():
    """The property that makes this a decomposition. Every other test leans on it."""
    clock = ManualClock()
    tl = _clean_run(clock)
    parts = tl.ttft_parts
    assert set(parts) == set(TTFT_PARTS)
    assert sum(parts.values()) == pytest.approx(tl.ttft_s)


def test_a_clean_run_pays_nothing_for_preemption():
    clock = ManualClock()
    tl = _clean_run(clock)
    assert tl.requeue_s == 0.0
    assert tl.lost_prefill_s == 0.0
    assert tl.num_prefills == 1


def test_decode_is_the_first_token_to_the_finish():
    clock = ManualClock()
    tl = _clean_run(clock)
    assert tl.decode_s == pytest.approx(0.75)
    assert tl.total_s == pytest.approx(3.35)


def test_a_one_token_request_has_no_decode_span():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    tl.on_admitted()
    clock.advance(0.5)
    tl.on_first_token()
    tl.on_finished()
    assert tl.decode_s == 0.0
    assert tl.total_s == pytest.approx(tl.ttft_s)


def test_preemption_before_the_first_token_names_both_wasted_parts():
    """Admitted, prefilled halfway, evicted, requeued, admitted again, answered.

    The 0.3 of prefill that was thrown away and the 1.5 spent back in the queue
    are the two parts that exist only because the pool ran dry, and they are the
    difference between this request's TTFT and the one it would have had.
    """
    clock = ManualClock()
    tl = _timeline(clock)
    clock.advance(0.1)
    tl.on_queued()
    clock.advance(2.0)
    tl.on_admitted()
    clock.advance(0.3)
    tl.on_preempted()
    clock.advance(1.5)
    tl.on_admitted()
    clock.advance(0.5)
    tl.on_first_token()

    assert tl.queue_wait_s == pytest.approx(2.0)
    assert tl.requeue_s == pytest.approx(1.5)
    assert tl.lost_prefill_s == pytest.approx(0.3)
    assert tl.prefill_s == pytest.approx(0.5)
    assert tl.num_prefills == 2
    assert tl.ttft_s == pytest.approx(4.4)
    assert sum(tl.ttft_parts.values()) == pytest.approx(tl.ttft_s)


def test_preemption_after_the_first_token_leaves_the_ttft_split_alone():
    """The split is frozen at the first token. Later evictions are decode's bill."""
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    tl.on_admitted()
    clock.advance(0.5)
    tl.on_first_token()
    before = dict(tl.ttft_parts)
    clock.advance(0.2)
    tl.on_preempted()
    clock.advance(9.0)
    tl.on_admitted()
    clock.advance(0.4)
    tl.on_finished()

    assert tl.ttft_parts == before
    assert tl.ttft_s == pytest.approx(0.5)
    assert tl.decode_s == pytest.approx(9.6)
    assert tl.num_prefills == 2


def test_residency_accounts_for_every_second_of_the_life():
    """A request is queued or running, always, and the two must add to the whole."""
    clock = ManualClock()
    tl = _timeline(clock)
    clock.advance(0.1)
    tl.on_queued()
    clock.advance(2.0)
    tl.on_admitted()
    clock.advance(0.3)
    tl.on_preempted()
    clock.advance(1.5)
    tl.on_admitted()
    clock.advance(0.7)
    tl.on_first_token()
    clock.advance(0.4)
    tl.on_finished()

    assert tl.queued_s == pytest.approx(3.5)
    assert tl.running_s == pytest.approx(1.4)
    assert tl.inbox_s + tl.queued_s + tl.running_s == pytest.approx(tl.total_s)


def test_a_request_aborted_in_the_queue_is_all_queue_and_no_answer():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    clock.advance(3.0)
    tl.on_finished()
    assert tl.answered is False
    assert tl.ttft_s is None
    assert tl.queued_s == pytest.approx(3.0)
    assert tl.running_s == 0.0
    assert tl.decode_s is None


def test_queueing_twice_is_a_bug_and_says_so():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    with pytest.raises(TimelineBroken, match="already queued"):
        tl.on_queued()


def test_admitting_a_running_request_again_is_a_bug():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    tl.on_admitted()
    with pytest.raises(TimelineBroken, match="already running"):
        tl.on_admitted()


def test_preempting_a_waiting_request_is_a_bug():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    with pytest.raises(TimelineBroken, match="not running"):
        tl.on_preempted()


def test_a_second_first_token_is_a_bug():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    tl.on_admitted()
    tl.on_first_token()
    with pytest.raises(TimelineBroken, match="first token"):
        tl.on_first_token()


def test_an_admission_with_no_queueing_stamps_the_queue_at_that_instant():
    """Tolerated on purpose: the scheduler tests drive transitions by hand.

    It costs nothing to be honest about, since a request that was never queued
    waited zero seconds for a slot and the time before it lands in the inbox.
    """
    clock = ManualClock()
    tl = _timeline(clock)
    clock.advance(0.6)
    tl.on_admitted()
    assert tl.queued_at == 100.6
    assert tl.queue_wait_s == 0.0
    assert tl.inbox_s == pytest.approx(0.6)


def test_check_passes_a_real_timeline_and_catches_a_doctored_one():
    clock = ManualClock()
    tl = _clean_run(clock)
    tl.check()
    tl.first_token_at = tl.finished_at + 1.0
    with pytest.raises(TimelineBroken):
        tl.check()


def test_check_catches_residency_that_does_not_add_up():
    clock = ManualClock()
    tl = _clean_run(clock)
    tl.running_s += 0.5
    with pytest.raises(TimelineBroken, match="residency"):
        tl.check()


# --- tier 2: the report ---------------------------------------------------------


def _answered(clock: ManualClock, inbox: float, queue: float, prefill: float) -> RequestTimeline:
    tl = _timeline(clock)
    clock.advance(inbox)
    tl.on_queued()
    clock.advance(queue)
    tl.on_admitted()
    clock.advance(prefill)
    tl.on_first_token()
    clock.advance(1.0)
    tl.on_finished()
    return tl


def test_an_empty_report_is_zeros_and_not_an_exception():
    report = summarize([])
    assert isinstance(report, LatencyReport)
    assert report.n == 0
    assert report.n_answered == 0
    assert report.mean_ttft_s == 0.0
    assert report.queue_share == 0.0


def test_the_mean_components_sum_to_the_mean_ttft():
    """Means decompose. This is the only line of the report that is allowed to."""
    clock = ManualClock()
    timelines = [
        _answered(clock, 0.1, 1.0, 0.5),
        _answered(clock, 0.2, 3.0, 0.5),
        _answered(clock, 0.3, 8.0, 0.6),
    ]
    report = summarize(timelines)
    total = (
        report.mean_inbox_s
        + report.mean_queue_wait_s
        + report.mean_requeue_s
        + report.mean_lost_prefill_s
        + report.mean_prefill_s
    )
    assert total == pytest.approx(report.mean_ttft_s)
    assert report.mean_queue_wait_s == pytest.approx(4.0)
    assert report.n == 3
    assert report.n_answered == 3


def test_the_percentiles_are_nearest_rank_over_the_whole_ttft():
    clock = ManualClock()
    timelines = [_answered(clock, 0.0, float(q), 0.0) for q in range(1, 11)]
    report = summarize(timelines)
    assert report.ttft_p50_s == pytest.approx(5.0)
    assert report.ttft_p99_s == pytest.approx(10.0)
    assert report.queue_p99_s == pytest.approx(10.0)


def test_the_shares_are_the_headline_and_they_sum_to_one():
    clock = ManualClock()
    report = summarize([_answered(clock, 0.1, 9.0, 0.9)])
    assert report.queue_share == pytest.approx(0.9)
    assert report.prefill_share == pytest.approx(0.09)
    assert report.inbox_share == pytest.approx(0.01)
    assert report.waste_share == 0.0
    assert (
        report.inbox_share + report.queue_share + report.prefill_share + report.waste_share
        == pytest.approx(1.0)
    )


def test_a_request_with_no_first_token_is_counted_and_kept_out_of_the_ttft_stats():
    """A 500 or an abort has no TTFT, and a zero in the percentiles is a lie."""
    clock = ManualClock()
    good = _answered(clock, 0.0, 2.0, 0.0)
    dead = _timeline(clock)
    dead.on_queued()
    clock.advance(50.0)
    dead.on_finished()
    report = summarize([good, dead])
    assert report.n == 2
    assert report.n_answered == 1
    assert report.mean_ttft_s == pytest.approx(2.0)
    assert report.ttft_p99_s == pytest.approx(2.0)


def test_the_report_counts_the_prefills_it_paid_for_twice():
    clock = ManualClock()
    tl = _timeline(clock)
    tl.on_queued()
    tl.on_admitted()
    clock.advance(0.4)
    tl.on_preempted()
    clock.advance(1.0)
    tl.on_admitted()
    clock.advance(0.4)
    tl.on_first_token()
    tl.on_finished()
    report = summarize([tl, _answered(clock, 0.0, 0.0, 0.4)])
    assert report.n_preempted == 1
    assert report.reprefills == 1
    assert report.waste_share > 0.0


def test_the_summary_names_the_sample_count_next_to_every_percentile():
    clock = ManualClock()
    text = summarize([_answered(clock, 0.1, 1.0, 0.5)]).summary()
    assert "n=1" in text
    assert "queue" in text
    assert "prefill" in text


# --- tier 3: the wiring ---------------------------------------------------------


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


def _req(request_id="r0", prompt=(1, 2, 3, 4), max_new_tokens=4) -> Request:
    return Request(
        request_id=request_id, prompt_token_ids=list(prompt), max_new_tokens=max_new_tokens
    )


def test_the_scheduler_stamps_the_queue_and_the_admission():
    scheduler = Scheduler(BlockAllocator(16, 4), max_batch_size=2, watermark=0.0)
    request = _req()
    scheduler.add_request(request)
    assert request.timeline.queued_at is not None
    assert request.timeline.first_admitted_at is None

    scheduler.schedule()
    assert request.state is RequestState.RUNNING
    assert request.timeline.first_admitted_at is not None
    assert request.timeline.queue_wait_s >= 0.0


def test_the_engine_stamps_the_first_token_and_the_finish():
    engine = Engine.build(_model(), num_blocks=32, block_size=4, max_batch_size=2)
    request = engine.add_request(_req(max_new_tokens=3))
    engine.run_to_completion()
    tl = request.timeline
    tl.check()
    assert tl.answered is True
    assert tl.prefill_s > 0.0
    assert tl.decode_s > 0.0
    assert tl.ttft_s < tl.total_s


def test_a_request_that_waits_for_a_slot_says_so_in_its_split():
    """One slot, two requests: the second one's TTFT is mostly queue, and it knows.

    This is Day 42's subtraction, done properly. The second request's prefill is
    the same work as the first's; everything else it paid is the wait for a row.
    """
    engine = Engine.build(_model(), num_blocks=64, block_size=4, max_batch_size=1)
    first = engine.add_request(_req("r0", max_new_tokens=6))
    second = engine.add_request(_req("r1", max_new_tokens=6))
    engine.run_to_completion()

    for tl in (first.timeline, second.timeline):
        tl.check()
    assert second.timeline.queue_wait_s > first.timeline.queue_wait_s
    assert second.timeline.queue_wait_s > second.timeline.prefill_s
    assert second.timeline.ttft_s > first.timeline.ttft_s


def test_preemption_shows_up_as_a_reprefill_in_the_timeline():
    """A pool too small for both: the victim comes back and is prefilled twice."""
    engine = Engine.build(_model(), num_blocks=4, block_size=4, max_batch_size=2)
    engine.scheduler.watermark_blocks = 0
    a = engine.add_request(_req("r0", prompt=(1, 2, 3, 4, 5, 6), max_new_tokens=8))
    b = engine.add_request(_req("r1", prompt=(1, 2, 3, 4, 5, 6), max_new_tokens=8))
    engine.run_to_completion()

    assert engine.scheduler.num_preemptions > 0
    victim = b if b.num_preemptions else a
    assert victim.num_preemptions > 0
    assert victim.timeline.num_prefills == victim.num_preemptions + 1
    victim.timeline.check()


def _engine(num_blocks=64, block_size=4, max_batch_size=2) -> Engine:
    return Engine.build(
        _model(), num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def run(coro, timeout: float = 15.0):
    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


def test_the_bridge_keeps_the_timeline_of_everything_it_answered():
    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            await asyncio.gather(
                serving.submit([1, 2, 3], max_new_tokens=3),
                serving.submit([4, 5, 6], max_new_tokens=3),
            )
            return serving.latency_report()

    report = run(scenario())
    assert report.n == 2
    assert report.n_answered == 2
    assert report.mean_ttft_s > 0.0
    assert report.mean_prefill_s > 0.0


def test_the_inbox_wait_is_real_and_the_bridge_measures_it():
    """A request built in a handler is not queued until the loop's next drain.

    Day 42 could not see this at all: from the client's side it is part of TTFT
    and indistinguishable from prefill. It is a whole forward pass wide when the
    loop is busy, and it belongs to the bridge, not to the model.
    """

    async def scenario():
        async with AsyncEngine(_engine(max_batch_size=4)) as serving:
            await serving.submit([1, 2, 3], max_new_tokens=8)
            futures = [serving.submit([4, 5, 6], max_new_tokens=2) for _ in range(3)]
            await asyncio.gather(*futures)
            return list(serving.latencies)

    timelines = run(scenario())
    for tl in timelines:
        tl.check()
        assert tl.inbox_s >= 0.0
    assert any(tl.inbox_s > 0.0 for tl in timelines)


def test_stats_carries_the_split_so_health_does_too():
    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            await serving.submit([1, 2, 3], max_new_tokens=3)
            return serving.stats()

    stats = run(scenario())
    assert stats["answered"] == 1
    assert stats["ttft_p50_s"] > 0.0
    assert 0.0 <= stats["queue_share"] <= 1.0


def test_the_latency_window_is_bounded_so_a_long_run_does_not_grow_forever():
    async def scenario():
        async with AsyncEngine(_engine(), latency_window=2) as serving:
            for _ in range(4):
                await serving.submit([1, 2, 3], max_new_tokens=2)
            return serving.latency_report(), len(serving.latencies)

    report, held = run(scenario())
    assert held == 2
    assert report.n == 2
