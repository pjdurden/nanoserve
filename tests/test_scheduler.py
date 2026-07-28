"""Day 30: pure tests for the request state machine and the two queues.

No model and no forward pass: a real `BlockAllocator` and plain integer token ids
are enough to pin every decision the scheduler makes, because the decision is
bookkeeping. Which request is admitted, what it reserves, which slot it gets, when
its blocks come back, and what a request is allowed to do in each state.

The file is organised the way the day was built. First the `Request`: an object
with a state, replacing "row index into a fixed batch" as the thing the engine
tracks. Then the `Scheduler`: two queues over that object, admitting from the head
of the waiting queue while a slot and the blocks are both there. The last test is
the point of Week 8 stated in Day 29's vocabulary: on a scheduled loop every token
the forward issues is a token some unfinished row collects, so the waste fraction
that static batching paid is 0.0 by construction.
"""

from __future__ import annotations

import pytest

from nanoserve.cache import BlockAllocator, KVCacheExhausted
from nanoserve.scheduler import (
    IllegalTransition,
    Request,
    RequestState,
    Scheduler,
    SchedulerOutput,
)


def _req(request_id="r0", prompt=(1, 2, 3), max_new_tokens=4, eos_token_id=None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


def _sched(num_blocks=64, block_size=4, max_batch_size=4) -> Scheduler:
    return Scheduler(BlockAllocator(num_blocks, block_size), max_batch_size=max_batch_size)


# --- the request object -------------------------------------------------------


def test_a_new_request_is_waiting_and_owns_nothing():
    r = _req()

    assert r.state is RequestState.WAITING
    assert r.slot is None
    assert r.block_ids == []
    assert r.output_token_ids == []
    assert r.finish_reason is None
    assert not r.is_finished


def test_an_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="prompt"):
        _req(prompt=())


def test_a_request_that_may_not_emit_a_token_is_rejected():
    with pytest.raises(ValueError, match="max_new_tokens"):
        _req(max_new_tokens=0)


def test_length_counts_prompt_plus_what_has_been_generated():
    r = _req(prompt=(1, 2, 3), max_new_tokens=4)
    assert r.num_prompt_tokens == 3
    assert r.num_output_tokens == 0
    assert r.num_tokens == 3

    r.transition_to(RequestState.RUNNING)
    r.append_token(9)

    assert r.num_output_tokens == 1
    assert r.num_tokens == 4
    assert r.token_ids == [1, 2, 3, 9]


def test_worst_case_tokens_is_the_footprint_admission_reserves():
    """Prompt plus every token it is still allowed to emit: the longest it can get."""
    assert _req(prompt=(1, 2, 3), max_new_tokens=4).worst_case_tokens == 7


def test_only_a_running_request_may_be_appended_to():
    r = _req()
    with pytest.raises(IllegalTransition, match="waiting"):
        r.append_token(9)


def test_hitting_the_token_budget_finishes_the_request_with_reason_length():
    r = _req(max_new_tokens=2)
    r.transition_to(RequestState.RUNNING)

    r.append_token(9)
    assert not r.is_finished

    r.append_token(10)
    assert r.is_finished
    assert r.state is RequestState.FINISHED
    assert r.finish_reason == "length"


def test_the_eos_token_finishes_the_request_with_reason_stop():
    r = _req(max_new_tokens=8, eos_token_id=42)
    r.transition_to(RequestState.RUNNING)

    r.append_token(42)

    assert r.is_finished
    assert r.finish_reason == "stop"
    # The stop token is recorded. It was really sampled, and the detokenizer, not
    # the scheduler, decides whether to show it.
    assert r.output_token_ids == [42]


def test_a_finished_request_cannot_be_appended_to():
    r = _req(max_new_tokens=1)
    r.transition_to(RequestState.RUNNING)
    r.append_token(9)

    with pytest.raises(IllegalTransition, match="finished"):
        r.append_token(10)


def test_the_legal_edges_are_admit_finish_and_abort():
    admitted = _req()
    admitted.transition_to(RequestState.RUNNING)
    admitted.transition_to(RequestState.FINISHED)
    assert admitted.state is RequestState.FINISHED

    aborted = _req()
    aborted.transition_to(RequestState.FINISHED)  # never ran
    assert aborted.state is RequestState.FINISHED


@pytest.mark.parametrize(
    "path",
    [
        # No preemption yet: putting a running request back on the waiting queue is
        # Week 9's edge, and until the recompute path exists it would silently strand
        # blocks.
        (RequestState.RUNNING, RequestState.WAITING),
        # Finished is terminal in both directions.
        (RequestState.RUNNING, RequestState.FINISHED, RequestState.RUNNING),
        (RequestState.FINISHED, RequestState.WAITING),
        # A no-op transition is a bug in the caller, not a nothing.
        (RequestState.RUNNING, RequestState.RUNNING),
    ],
)
def test_illegal_edges_raise(path):
    r = _req()
    with pytest.raises(IllegalTransition):
        for state in path:
            r.transition_to(state)


def test_finish_records_its_reason():
    r = _req()
    r.transition_to(RequestState.RUNNING)
    r.finish("abort")

    assert r.state is RequestState.FINISHED
    assert r.finish_reason == "abort"


# --- the queues ---------------------------------------------------------------


def test_a_new_request_lands_on_the_waiting_queue():
    s = _sched()
    s.add_request(_req())

    assert s.num_waiting == 1
    assert s.num_running == 0
    assert s.has_unfinished()


def test_duplicate_request_ids_are_rejected():
    s = _sched()
    s.add_request(_req("a"))

    with pytest.raises(ValueError, match="a"):
        s.add_request(_req("a"))


def test_only_a_waiting_request_may_be_added():
    s = _sched()
    r = _req()
    r.transition_to(RequestState.RUNNING)

    with pytest.raises(ValueError, match="waiting"):
        s.add_request(r)


def test_a_request_too_large_for_the_whole_pool_is_rejected_at_the_door():
    """Admission would never fire, and it would block the queue behind it forever."""
    s = _sched(num_blocks=4, block_size=4)  # 16 tokens, all of it

    with pytest.raises(KVCacheExhausted, match="pool"):
        s.add_request(_req(prompt=(1,) * 14, max_new_tokens=8))


def test_schedule_admits_up_to_the_batch_size_and_leaves_the_rest_waiting():
    s = _sched(max_batch_size=2)
    for i in range(4):
        s.add_request(_req(f"r{i}"))

    out = s.schedule()

    assert isinstance(out, SchedulerOutput)
    assert [r.request_id for r in out.scheduled] == ["r0", "r1"]
    assert [r.request_id for r in out.admitted] == ["r0", "r1"]
    assert out.num_waiting == 2
    assert s.num_running == 2
    assert all(r.state is RequestState.RUNNING for r in out.scheduled)


def test_admission_reserves_the_worst_case_and_hands_out_a_slot():
    s = _sched(num_blocks=64, block_size=4, max_batch_size=2)
    free_before = s.allocator.num_free
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=4))  # 7 tokens -> 2 blocks
    s.add_request(_req("b", prompt=(1,) * 5, max_new_tokens=4))  # 9 tokens -> 3 blocks

    out = s.schedule()
    a, b = out.scheduled

    assert len(a.block_ids) == 2
    assert len(b.block_ids) == 3
    assert s.allocator.num_free == free_before - 5
    assert (a.slot, b.slot) == (0, 1)
    assert len(set(a.block_ids) & set(b.block_ids)) == 0


def test_admission_stops_when_the_pool_cannot_take_the_next_request():
    # 3 blocks of 4 tokens. Each request needs 8 tokens, i.e. 2 blocks.
    s = _sched(num_blocks=3, block_size=4, max_batch_size=4)
    for i in range(3):
        s.add_request(_req(f"r{i}", prompt=(1,) * 4, max_new_tokens=4))

    out = s.schedule()

    assert [r.request_id for r in out.admitted] == ["r0"]
    assert s.num_waiting == 2
    assert s.allocator.num_free == 1


def test_a_blocked_head_is_not_skipped_over():
    """FIFO, deliberately. Letting small requests overtake starves the large one."""
    s = _sched(num_blocks=4, block_size=4, max_batch_size=4)
    s.add_request(_req("big", prompt=(1,) * 8, max_new_tokens=8))  # 16 tokens, 4 blocks
    s.add_request(_req("small", prompt=(1,), max_new_tokens=1))  # 2 tokens, 1 block
    s.schedule()  # big takes the whole pool
    assert s.num_running == 1

    out = s.schedule()  # nothing freed, so small still waits behind big

    assert out.admitted == ()
    assert s.num_waiting == 1

    s.running[0].finish("length")
    out = s.schedule()

    assert [r.request_id for r in out.admitted] == ["small"]


def test_a_running_request_keeps_its_slot_across_steps():
    s = _sched(max_batch_size=2)
    s.add_request(_req("a"))
    s.add_request(_req("b"))

    first = s.schedule()
    slots = [r.slot for r in first.scheduled]
    second = s.schedule()

    assert [r.slot for r in second.scheduled] == slots
    assert second.admitted == ()


def test_a_finished_request_gives_back_its_blocks_and_a_waiting_one_takes_its_slot():
    """The whole point of Week 8, in one test."""
    s = _sched(num_blocks=8, block_size=4, max_batch_size=2)
    for i in range(3):
        s.add_request(_req(f"r{i}", prompt=(1, 2, 3), max_new_tokens=4))  # 2 blocks each

    s.schedule()
    free_mid = s.allocator.num_free
    done = s.running[0]
    done_slot = done.slot
    done.finish("length")

    out = s.schedule()

    assert [r.request_id for r in out.finished] == ["r0"]
    assert done.block_ids == []
    assert done.slot is None
    assert s.allocator.num_free == free_mid + 2 - 2  # r0 released 2, r2 reserved 2
    assert [r.request_id for r in out.admitted] == ["r2"]
    assert out.admitted[0].slot == done_slot
    # The batch is kept in slot order, because a row index has to mean the same
    # thing to the scheduler and to the cache. r2 inherited slot 0, so it is first.
    assert [(r.request_id, r.slot) for r in out.scheduled] == [("r2", 0), ("r1", 1)]


def test_a_request_aborted_while_waiting_never_runs():
    s = _sched(max_batch_size=4)
    s.add_request(_req("a"))
    s.add_request(_req("b"))

    s.abort("a")
    out = s.schedule()

    assert [r.request_id for r in out.admitted] == ["b"]
    assert [r.request_id for r in out.finished] == ["a"]
    assert s.allocator.num_free == s.allocator.num_blocks - 2  # only b reserved


def test_a_request_aborted_while_running_releases_everything():
    s = _sched(max_batch_size=4)
    s.add_request(_req("a"))
    s.schedule()

    s.abort("a")
    out = s.schedule()

    assert [r.request_id for r in out.finished] == ["a"]
    assert out.scheduled == ()
    assert s.allocator.num_free == s.allocator.num_blocks
    assert not s.has_unfinished()


def test_aborting_an_unknown_request_is_an_error():
    s = _sched()
    with pytest.raises(KeyError, match="ghost"):
        s.abort("ghost")


def test_the_output_splits_prefill_from_decode():
    """A newly admitted request needs its whole prompt run; the rest need one token."""
    s = _sched(max_batch_size=4)
    s.add_request(_req("a"))
    s.schedule()
    s.add_request(_req("b"))

    out = s.schedule()

    assert [r.request_id for r in out.prefill] == ["b"]
    assert [r.request_id for r in out.decode] == ["a"]
    assert out.batch_size == 2
    assert not out.is_empty


def test_an_empty_schedule_is_empty_rather_than_none():
    out = _sched().schedule()

    assert out.is_empty
    assert out.scheduled == ()
    assert out.batch_size == 0


def test_every_block_comes_back_when_the_queues_drain():
    s = _sched(num_blocks=16, block_size=4, max_batch_size=2)
    for i in range(5):
        s.add_request(_req(f"r{i}", prompt=(1, 2), max_new_tokens=3))

    steps = 0
    while s.has_unfinished():
        out = s.schedule()
        for r in out.scheduled:
            r.append_token(7)
        steps += 1
        assert steps < 100, "scheduler failed to drain"

    assert s.allocator.num_free == s.allocator.num_blocks
    assert s.num_running == 0
    assert s.num_waiting == 0


# --- what the reservation costs ----------------------------------------------


def test_reserved_and_used_blocks_are_reported_separately():
    s = _sched(num_blocks=32, block_size=4, max_batch_size=2)
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=4))  # 7 worst case, 2 blocks
    s.schedule()

    # 3 prompt tokens live in 1 block; the second block is reserved and empty.
    assert s.reserved_blocks == 2
    assert s.used_blocks == 1
    assert s.reservation_waste == pytest.approx(0.5)


def test_reservation_waste_falls_as_a_request_generates():
    s = _sched(num_blocks=32, block_size=4, max_batch_size=2)
    s.add_request(_req("a", prompt=(1, 2, 3), max_new_tokens=4))
    out = s.schedule()

    for _ in range(4):
        out.scheduled[0].append_token(7)

    assert s.used_blocks == 2  # 7 tokens now really occupy both blocks
    assert s.reservation_waste == pytest.approx(0.0)


def test_reservation_waste_is_zero_with_nothing_running():
    assert _sched().reservation_waste == 0.0


# --- the acceptance shape -----------------------------------------------------


def test_no_scheduled_row_is_a_finished_row():
    """Day 29's waste fraction, recomputed on a scheduled loop: 0.0 by construction.

    A static batch issues `batch_size` tokens every step until its slowest row is
    done, so a ragged batch spends most of its forward on rows that already
    finished (measured Day 29: 79%). Here the batch is rebuilt every iteration, so
    a row is in the forward only while it still wants a token. Issued equals
    useful, and the head-of-line delay it goes with is gone too: r0 leaves at its
    own last token rather than at r7's.
    """
    s = _sched(num_blocks=64, block_size=4, max_batch_size=4)
    lengths = [1, 1, 1, 1, 1, 1, 1, 8]  # the Day-29 ragged shape
    for i, n in enumerate(lengths):
        s.add_request(_req(f"r{i}", prompt=(1, 2), max_new_tokens=n))

    issued = 0
    returned_at: dict[str, int] = {}
    step = 0
    while s.has_unfinished():
        out = s.schedule()
        for r in out.finished:
            returned_at[r.request_id] = step
        for r in out.scheduled:
            assert not r.is_finished, "a finished row must never be in the forward"
            r.append_token(7)
        issued += out.batch_size
        step += 1
        assert step < 100, "scheduler failed to drain"

    useful = sum(lengths)
    assert issued == useful  # waste_fraction == 0.0
    # And the short rows are handed back long before the long one, which is the
    # head-of-line half: a static batch would return all eight together.
    assert returned_at["r0"] < returned_at["r7"]
