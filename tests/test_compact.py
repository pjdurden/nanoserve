"""Day 58 tests: the persistent batch, so a decode step's rows are always a prefix.

Day 57's acceptance run found that a recorded decode reads two things indexed from
zero (the persistent input buffers, written in batch order, and the window
`slots[:rows, :width]` on the slot table, in cache-row order) and that those are the
same index only while the scheduler's rows are `(0, 1, ... n-1)`. It fixed the
correctness half with `rows_are_a_prefix`: a step that fails it runs the forward
instead of replaying. Then it measured what that costs, and the table is the reason
this file exists. Four generation lengths over eight slots replayed 26% of its decode
steps. More slots made it worse.

This file is the other half. The scheduler hands out whichever row slot is free,
which is O(1) and arbitrary; a **persistent batch** keeps the running rows compacted
instead, so the question `rows_are_a_prefix` asks can only ever be answered yes.

Four claims:

  1. **The move is a relabel, not a recompute.** A row's K/V stays exactly where it
     is in the pool. What moves is the `BlockTable` (a host object holding block
     ids), the slot table's row (one device row copy of `length` cells), and the
     request's slot id. Nothing is read, nothing is written into the pool, and no
     token is computed twice. See `BatchedPagedKVCache.move_row`.
  2. **The plan is the fewest moves there are.** Every occupied row at or past the
     running count has to leave, and every hole below it has to be filled, so the
     number of moves is forced and the only freedom is the pairing. Sorted against
     sorted keeps the batch in the order it was already in. See `plan_compaction`.
  3. **The timing is the whole safety argument.** Compaction runs inside
     `Scheduler.schedule`, after the reap and the growth and *before* admission: the
     slot table row it copies is storage a captured graph reads, so the only moment
     it is safe to write is the one where no step is in flight, and running it
     before admission is what makes a newly admitted row land on top of the prefix
     rather than past a hole.
  4. **It is paid for by completions, not by steps.** A hole is made by a release,
     so the moves a run does are bounded by the rows it gave back: a decode step
     that finishes nobody compacts nothing and costs one tuple comparison.
     `check_moves_amortised` is the gate.

The number this is all for is `replay_share`, and the last test in this file is the
Day-57 workload with compaction on: 26% becomes 100%.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.captured import CaptureStats, eager_recorder, rows_are_a_prefix
from nanoserve.compact import (
    CompactionUnsound,
    RowCompactor,
    RowMove,
    check_batch_is_persistent,
    check_moves_amortised,
    check_moves_minimal,
    holes,
    is_compact,
    move_cells,
    moves_needed,
    plan_compaction,
    strays,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request, Scheduler
from nanoserve.slots import SlotTable

# --- what a compact batch is ---------------------------------------------------------


def test_a_prefix_is_compact_and_a_hole_is_not():
    assert is_compact(())
    assert is_compact((0,))
    assert is_compact((0, 1, 2))
    assert not is_compact((1,))
    assert not is_compact((0, 2))


def test_compactness_does_not_care_what_order_the_rows_arrive_in():
    """The caller's tuple is a batch order; this question is about the set."""
    assert is_compact((2, 0, 1))


def test_the_holes_are_the_free_rows_under_the_running_count():
    assert holes((0, 2, 5)) == (1,)
    assert holes((3, 4, 5)) == (0, 1, 2)
    assert holes((0, 1, 2)) == ()


def test_the_strays_are_the_occupied_rows_at_or_past_it():
    assert strays((0, 2, 5)) == (5,)
    assert strays((3, 4, 5)) == (3, 4, 5)
    assert strays((0, 1, 2)) == ()


def test_there_are_exactly_as_many_strays_as_holes():
    """Which is why a compaction is a permutation and never an allocation: every
    row that has to leave has somewhere to land, worked out by counting alone."""
    for rows in ((0, 2, 5), (3, 4, 5), (1, 2), (0, 1, 2), (7,)):
        assert len(holes(rows)) == len(strays(rows))


# --- the plan ------------------------------------------------------------------------


def test_a_compact_batch_is_planned_as_no_moves_at_all():
    assert plan_compaction((0, 1, 2)) == ()
    assert plan_compaction(()) == ()


def test_the_plan_fills_the_holes_from_the_top():
    assert plan_compaction((0, 2, 5)) == (RowMove(5, 1),)


def test_the_plan_moves_every_stray_and_nothing_else():
    assert plan_compaction((3, 4, 5)) == (RowMove(3, 0), RowMove(4, 1), RowMove(5, 2))


def test_applying_the_plan_leaves_a_prefix():
    for rows in ((0, 2, 5), (3, 4, 5), (1, 2), (0, 1, 2), (7,), (2, 3, 6, 9)):
        where = {m.src: m.dst for m in plan_compaction(rows)}
        assert is_compact(tuple(where.get(r, r) for r in rows))


def test_the_plan_is_the_fewest_moves_there_are():
    """Forced, not chosen: a stray has to leave and a hole has to be filled, and one
    move does both. The only freedom in a compaction is which stray takes which
    hole, and that is a readability decision, not a cost one."""
    for rows in ((0, 2, 5), (3, 4, 5), (2, 3, 6, 9)):
        moves = plan_compaction(rows)
        assert len(moves) == moves_needed(rows) == len(strays(rows))
        check_moves_minimal(rows, moves)


def test_the_pairing_keeps_the_batch_in_the_order_it_was_already_in():
    """Sorted holes against sorted strays. Any pairing is correct, and this one
    leaves a step's rows in the same relative order as the step before it, which is
    what makes two consecutive scheduling traces readable next to each other."""
    moves = plan_compaction((1, 4, 7))
    assert [m.src for m in moves] == [4, 7]
    assert [m.dst for m in moves] == [0, 2]


def test_a_plan_over_a_repeated_row_is_refused():
    with pytest.raises(ValueError, match="distinct"):
        plan_compaction((0, 1, 1))


def test_a_move_onto_itself_is_not_a_move():
    with pytest.raises(ValueError, match="same row"):
        RowMove(2, 2)


def test_a_move_renders_as_the_arrow_a_trace_reads():
    assert RowMove(5, 1).render() == "row 5 -> row 1"


def test_what_a_compaction_copies_is_the_moved_rows_lengths():
    """The price, in the only currency it is paid in: cells of the slot table. The
    K/V does not move, so a 2,000-token row costs 2,000 int64 of addressing and
    nothing at all of the megabytes its context actually occupies."""
    assert move_cells((RowMove(5, 1),), {5: 120}) == 120
    assert move_cells((RowMove(3, 0), RowMove(4, 1)), {3: 10, 4: 7}) == 17
    assert move_cells((), {}) == 0


# --- the table's half of a move ------------------------------------------------------


def _filled(table: SlotTable, row: int, slots: list[int]) -> None:
    for s in slots:
        table.append([row], [s])


def test_moving_a_table_row_carries_its_slots_and_its_length():
    table = SlotTable(4, 8)
    _filled(table, 2, [11, 12, 13])
    table.move_row(2, 0)
    assert table.length(0) == 3
    assert table.length(2) == 0
    assert table.slots[0, :3].tolist() == [11, 12, 13]


def test_a_moved_row_reads_back_as_a_window_the_next_step():
    """The whole point of the move: the same history, addressed from row zero, so
    the rectangle a graph was recorded against now covers it."""
    table = SlotTable(4, 8)
    _filled(table, 1, [7, 8])
    table.move_row(1, 0)
    mapping, lengths = table.read([0], width=2)
    assert table.is_window(mapping)
    assert mapping[0].tolist() == [7, 8]
    assert lengths.tolist() == [2]


def test_a_move_leaves_every_other_row_alone():
    table = SlotTable(4, 8)
    _filled(table, 0, [1, 2])
    _filled(table, 3, [9])
    table.move_row(3, 1)
    assert table.slots[0, :2].tolist() == [1, 2]
    assert table.length(0) == 2


def test_a_move_counts_itself_and_what_it_copied():
    table = SlotTable(4, 8)
    _filled(table, 2, [11, 12, 13])
    table.move_row(2, 0)
    assert table.row_moves == 1
    assert table.row_moved_cells == 3


def test_moving_an_empty_row_copies_nothing_and_is_still_a_move():
    table = SlotTable(4, 8)
    table.move_row(3, 0)
    assert table.row_moves == 1
    assert table.row_moved_cells == 0


def test_moving_onto_a_row_that_still_holds_tokens_is_refused():
    """The destination is a hole by construction, and if it is not, this is an
    overwrite of somebody's addressing rather than a compaction."""
    table = SlotTable(4, 8)
    _filled(table, 0, [1])
    _filled(table, 1, [2])
    with pytest.raises(ValueError, match="still holds"):
        table.move_row(1, 0)


def test_moving_a_row_onto_itself_is_refused():
    table = SlotTable(4, 8)
    with pytest.raises(ValueError, match="same row"):
        table.move_row(1, 1)


def test_a_move_does_not_reallocate_the_buffer():
    """It is a row copy inside storage a captured graph holds the address of. If
    this ever became an allocation, every recorded window would point at freed
    memory and the gate that would notice is `warmup.check_table_stable`."""
    table = SlotTable(4, 8)
    _filled(table, 2, [11, 12])
    before = table.address
    table.move_row(2, 0)
    assert table.address == before


# --- the cache's half ----------------------------------------------------------------


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _cache(batch_size: int = 4):
    from nanoserve.cache import BatchedPagedKVCache

    allocator = BlockAllocator(num_blocks=32, block_size=4)
    return BatchedPagedKVCache(
        _config(), allocator, batch_size=batch_size, max_model_len=32
    )


def test_moving_a_cache_row_carries_its_block_table():
    cache = _cache()
    blocks = cache.allocator.allocate_for(8)
    cache.adopt_row(2, blocks)
    cache.move_row(2, 0)
    assert cache.tables[0].block_ids == blocks
    assert cache.tables[2].block_ids == []


def test_moving_a_cache_row_does_not_touch_the_pool():
    """The K/V is where it was. A block id is a name, and this renames who holds it,
    which is why a compaction costs a row of int64 rather than a copy of a context."""
    cache = _cache()
    before = cache.allocator.num_free
    cache.adopt_row(2, cache.allocator.allocate_for(8))
    after_adopt = cache.allocator.num_free
    cache.move_row(2, 0)
    assert cache.allocator.num_free == after_adopt < before


def test_moving_a_cache_row_carries_the_slot_table_row_with_it():
    cache = _cache()
    cache.adopt_row(2, cache.allocator.allocate_for(8))
    cache.tables[2].append(3)
    cache.slot_table.sync(cache.tables, [2])
    slots = cache.slot_table.slots[2, :3].tolist()
    cache.move_row(2, 0)
    assert cache.slot_table.length(0) == 3
    assert cache.slot_table.slots[0, :3].tolist() == slots


def test_moving_a_cache_row_invalidates_the_cached_mapping():
    cache = _cache()
    cache.adopt_row(1, cache.allocator.allocate_for(4))
    cache.tables[1].append(1)
    cache.slot_mapping(rows=[1])
    assert cache._mapping is not None
    cache.move_row(1, 0)
    assert cache._mapping is None


def test_moving_onto_an_occupied_cache_row_is_refused():
    cache = _cache()
    cache.adopt_row(0, cache.allocator.allocate_for(4))
    cache.tables[0].append(1)
    cache.adopt_row(1, cache.allocator.allocate_for(4))
    cache.tables[1].append(1)
    with pytest.raises(ValueError, match="still holds"):
        cache.move_row(1, 0)


def test_the_moved_row_agrees_with_its_block_table_afterwards():
    """`check_slots_agree` is Day 51's gate that the mirror has not drifted, and a
    move is the third event that can drift it after a prefill and a reset."""
    from nanoserve.slots import check_slots_agree

    cache = _cache()
    cache.adopt_row(3, cache.allocator.allocate_for(8))
    for _ in range(5):
        cache.tables[3].append(1)
    cache.slot_table.sync(cache.tables, [3])
    cache.move_row(3, 0)
    check_slots_agree(cache.slot_table, cache.tables, [0])


# --- the compactor -------------------------------------------------------------------


def test_a_compactor_that_is_off_plans_nothing_and_moves_nothing():
    seen = []
    compactor = RowCompactor(mode="off", on_move=lambda s, d: seen.append((s, d)))
    assert compactor.compact((1, 3)) == ()
    assert seen == []
    assert compactor.moves == 0


def test_a_compactor_calls_the_mover_once_per_move_in_plan_order():
    seen = []
    compactor = RowCompactor(mode="on", on_move=lambda s, d: seen.append((s, d)))
    moves = compactor.compact((3, 4, 5))
    assert seen == [(3, 0), (4, 1), (5, 2)]
    assert moves == (RowMove(3, 0), RowMove(4, 1), RowMove(5, 2))


def test_a_compactor_counts_its_calls_its_compactions_and_its_moves():
    """Three different numbers and the distance between them is the day's claim:
    most calls are a tuple comparison that decides there is nothing to do."""
    compactor = RowCompactor(mode="on")
    compactor.compact((0, 1, 2))
    compactor.compact((0, 1, 2))
    compactor.compact((0, 2, 5))
    assert compactor.calls == 3
    assert compactor.compactions == 1
    assert compactor.moves == 1


def test_a_compactor_with_no_mover_is_a_planner():
    """Useful on its own: the coverage study asks what a workload *would* move
    without wiring a cache to it."""
    assert RowCompactor(mode="on").compact((2,)) == (RowMove(2, 0),)


def test_a_compactor_renders_what_it_has_done():
    compactor = RowCompactor(mode="on")
    compactor.compact((0, 2, 5))
    assert "1 move" in compactor.render()


def test_an_unknown_compactor_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        RowCompactor(mode="sometimes")


# --- the gates -----------------------------------------------------------------------


def test_the_persistence_gate_passes_a_prefix_and_names_the_hole_otherwise():
    check_batch_is_persistent((0, 1, 2))
    with pytest.raises(CompactionUnsound, match="row 0"):
        check_batch_is_persistent((1, 2))


def test_the_persistence_gate_takes_a_plan_as_readily_as_a_tuple():
    """Same shape as Day 54's gates taking a reading or an object: the caller with
    the plan in hand should not have to reach into it."""

    class _Plan:
        rows = (1, 2)

    with pytest.raises(CompactionUnsound):
        check_batch_is_persistent(_Plan())


def test_the_minimality_gate_refuses_a_plan_that_left_a_hole():
    with pytest.raises(CompactionUnsound, match="not compact"):
        check_moves_minimal((0, 2, 5), ())


def test_the_minimality_gate_refuses_a_plan_that_moved_more_than_it_had_to():
    """Compact at the end and two row copies to get there, where one would have
    done. The gate is on the count and not on the outcome, because the outcome is
    the same and the cost is the thing this module is defending."""
    with pytest.raises(CompactionUnsound, match="move"):
        check_moves_minimal((0, 2, 5), (RowMove(2, 1), RowMove(5, 2)))


def test_the_amortisation_gate_passes_a_run_that_moved_once_per_release():
    compactor = RowCompactor(mode="on")
    compactor.compact((0, 2))
    check_moves_amortised(compactor, releases=1)


def test_the_amortisation_gate_refuses_more_moves_than_releases():
    """The invariant behind it: a hole is made by a release and filled by one move,
    so a run that moved more rows than it gave back is compacting something other
    than a completion, which is work nobody asked for on every step."""
    compactor = RowCompactor(mode="on")
    compactor.compact((3, 4, 5))
    with pytest.raises(CompactionUnsound, match="release"):
        check_moves_amortised(compactor, releases=1)


def test_the_amortisation_gate_passes_a_run_that_never_compacted():
    check_moves_amortised(RowCompactor(mode="on"), releases=0)


# --- the scheduler -------------------------------------------------------------------


def _scheduler(slots: int = 4, compact: bool = True, moved=None) -> Scheduler:
    scheduler = Scheduler(
        BlockAllocator(num_blocks=64, block_size=4),
        max_batch_size=slots,
        compact_rows=compact,
    )
    if moved is not None:
        scheduler.compactor.on_move = lambda s, d: moved.append((s, d))
    return scheduler


def _run(scheduler: Scheduler, request: Request) -> None:
    scheduler.add_request(request)


def test_an_uncompacted_scheduler_leaves_the_survivor_where_it_was():
    """Day 57's picture, kept as the control: this is what every day before today
    did, and it is correct. It is just not a prefix."""
    scheduler = _scheduler(compact=False)
    a, b = Request("a", [1, 2], max_new_tokens=4), Request("b", [3, 4], max_new_tokens=4)
    _run(scheduler, a)
    _run(scheduler, b)
    scheduler.schedule()
    a.finish("stop")
    out = scheduler.schedule()
    assert [r.slot for r in out.scheduled] == [1]


def test_a_compacting_scheduler_moves_the_survivor_down_to_row_zero():
    scheduler = _scheduler()
    a, b = Request("a", [1, 2], max_new_tokens=4), Request("b", [3, 4], max_new_tokens=4)
    _run(scheduler, a)
    _run(scheduler, b)
    scheduler.schedule()
    a.finish("stop")
    out = scheduler.schedule()
    assert [r.slot for r in out.scheduled] == [0]
    assert b.slot == 0


def test_the_moves_are_reported_on_the_output():
    scheduler = _scheduler()
    a, b = Request("a", [1, 2], max_new_tokens=4), Request("b", [3, 4], max_new_tokens=4)
    _run(scheduler, a)
    _run(scheduler, b)
    scheduler.schedule()
    a.finish("stop")
    assert scheduler.schedule().moved == (RowMove(1, 0),)


def test_the_tensor_half_of_a_move_runs_before_the_request_is_relabelled():
    """The same ordering `on_release` has and for the same reason: the callback is
    told which physical row is moving where while the scheduler still knows, and it
    is the only component that can talk about the rows as tensors."""
    moved = []
    scheduler = _scheduler(moved=moved)
    a, b = Request("a", [1, 2], max_new_tokens=4), Request("b", [3, 4], max_new_tokens=4)
    _run(scheduler, a)
    _run(scheduler, b)
    scheduler.schedule()
    a.finish("stop")
    scheduler.schedule()
    assert moved == [(1, 0)]


def test_the_freed_row_goes_back_to_the_top_of_the_free_list():
    scheduler = _scheduler()
    a, b = Request("a", [1, 2], max_new_tokens=4), Request("b", [3, 4], max_new_tokens=4)
    _run(scheduler, a)
    _run(scheduler, b)
    scheduler.schedule()
    a.finish("stop")
    scheduler.schedule()
    assert scheduler.free_slots == (1, 2, 3)


def test_a_newcomer_lands_on_top_of_the_prefix_rather_than_in_the_hole_it_left():
    """Which is why compaction runs before admission and not after it. Admitted
    after a compaction, the new row is `n`; admitted before one, it takes the hole
    and the survivor above it still has to move, so the step is a prefix either way
    and the second order pays a move for a row that had just arrived."""
    scheduler = _scheduler()
    a, b = Request("a", [1, 2], max_new_tokens=4), Request("b", [3, 4], max_new_tokens=4)
    _run(scheduler, a)
    _run(scheduler, b)
    scheduler.schedule()
    a.finish("stop")
    _run(scheduler, Request("c", [5, 6], max_new_tokens=4))
    out = scheduler.schedule()
    assert [r.slot for r in out.scheduled] == [0, 1]
    assert [r.request_id for r in out.scheduled] == ["b", "c"]


def test_every_schedule_of_a_long_run_hands_back_a_prefix():
    """The invariant, over a workload built to break it: eight requests of four
    different lengths over four slots, which is the row of Day 57's table that
    replayed 26% of its steps."""
    scheduler = _scheduler()
    requests = [
        Request(f"r{i}", [1, 2], max_new_tokens=(2, 3, 5, 9)[i % 4]) for i in range(8)
    ]
    for request in requests:
        scheduler.add_request(request)
    while scheduler.has_unfinished():
        out = scheduler.schedule()
        check_batch_is_persistent(tuple(r.slot for r in out.scheduled))
        for request in out.scheduled:
            request.append_token(7)


def test_a_run_that_finishes_nobody_compacts_nothing():
    scheduler = _scheduler()
    for i in range(3):
        scheduler.add_request(Request(f"r{i}", [1, 2], max_new_tokens=6))
    for _ in range(4):
        for request in scheduler.schedule().scheduled:
            request.append_token(7)
    assert scheduler.compactor.compactions == 0
    assert scheduler.compactor.calls > 0


def test_a_compacting_scheduler_moves_no_more_rows_than_it_released():
    scheduler = _scheduler()
    releases = 0
    for i in range(8):
        scheduler.add_request(Request(f"r{i}", [1, 2], max_new_tokens=(2, 3, 5, 9)[i % 4]))
    while scheduler.has_unfinished():
        out = scheduler.schedule()
        releases += len(out.finished) + len(out.preempted)
        for request in out.scheduled:
            request.append_token(7)
    check_moves_amortised(scheduler.compactor, releases=releases)


def test_a_preempted_row_is_compacted_away_like_a_finished_one():
    """A preemption releases a slot in the middle of `schedule`, between the reap
    and the compaction, which is exactly why the compaction is placed after the
    growth rather than after the reap."""
    scheduler = Scheduler(
        BlockAllocator(num_blocks=8, block_size=4), max_batch_size=4, compact_rows=True
    )
    for i in range(3):
        scheduler.add_request(Request(f"r{i}", [1, 2, 3, 4], max_new_tokens=8))
    for _ in range(12):
        out = scheduler.schedule()
        check_batch_is_persistent(tuple(r.slot for r in out.scheduled))
        for request in out.scheduled:
            request.append_token(7)
    assert scheduler.num_preemptions > 0


def test_compaction_is_off_by_default():
    """Every day from 31 to 57 ran without it and is still correct. A scheduler that
    starts moving rows because it was constructed differently would be a silent
    change of what a slot id means to everything holding one."""
    assert Scheduler(BlockAllocator(num_blocks=8, block_size=4)).compactor.mode == "off"


# --- the engine ----------------------------------------------------------------------


def _weights(config: ModelConfig, seed: int = 0) -> Weights:
    torch.manual_seed(seed)
    tensors = {n: torch.randn(*s) for n, s in expected_shapes(config).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, config)


def _engine(**kw) -> Engine:
    config = _config()
    model = LlamaModel(config, _weights(config))
    defaults = dict(num_blocks=64, block_size=4, max_batch_size=4, max_model_len=32)
    defaults.update(kw)
    return Engine.build(model, **defaults)


def _graphed(compact: bool, warm: bool = True) -> Engine:
    engine = _engine(
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
        compact_rows=compact,
    )
    if warm:
        engine.warm_decode()
    return engine


def _uneven(engine: Engine) -> list[list[int]]:
    """Two short requests and a long one, so the long one is left holding row 2."""
    requests = [
        Request("r0", [1, 2, 3], max_new_tokens=2),
        Request("r1", [4, 5, 6], max_new_tokens=3),
        Request("r2", [7, 8, 9], max_new_tokens=9),
    ]
    for request in requests:
        engine.add_request(request)
    while engine.has_unfinished():
        engine.step()
    return [r.output_token_ids for r in requests]


def test_a_compacted_engine_answers_what_an_uncompacted_one_answers():
    """The claim that has to survive the day. Moving a row changes where a history
    is addressed from and not what is in it, so the tokens are the same tokens."""
    assert _uneven(_engine(compact_rows=True)) == _uneven(_engine())


def test_a_compacted_graphed_engine_answers_what_an_eager_one_answers():
    assert _uneven(_graphed(compact=True)) == _uneven(_engine())


def test_compaction_takes_the_scattered_steps_to_zero():
    """Day 57's number, on Day 57's shape of workload. Without it the survivor sits
    in row 2 and no graph in the list addresses a batch that starts there."""
    scattered = _graphed(compact=False)
    _uneven(scattered)
    assert scattered.decode_graphs.scattered_calls > 0

    compact = _graphed(compact=True)
    _uneven(compact)
    assert compact.decode_graphs.scattered_calls == 0
    assert compact.decode_graphs.replay_share == 1.0


def test_the_engine_moves_the_cache_row_the_scheduler_names():
    """The wiring, asserted as wiring: the scheduler owns slot lifetime and has no
    tensors, so the move it plans reaches the cache through the engine's callback."""
    engine = _engine(compact_rows=True)
    assert engine.scheduler.compactor.on_move == engine._move_row


def test_an_engine_built_without_the_flag_has_a_compactor_that_is_off():
    assert _engine().scheduler.compactor.mode == "off"


def test_a_compacted_run_counts_what_its_moves_copied():
    engine = _engine(compact_rows=True)
    _uneven(engine)
    assert engine.cache.slot_table.row_moves > 0
    assert engine.cache.slot_table.row_moved_cells > 0


def test_a_compacted_run_moves_no_more_rows_than_it_finished():
    engine = _engine(compact_rows=True)
    _uneven(engine)
    check_moves_amortised(engine.scheduler.compactor, releases=3)


def test_every_decode_plan_of_a_compacted_run_is_a_window():
    """`rows_are_a_prefix` is the question a replay asks every step, and with the
    persistent batch it has one answer. Asked here of the plan the engine actually
    built rather than of the scheduler's bookkeeping."""
    engine = _engine(compact_rows=True)
    seen = []
    original = engine.model.forward

    def spy(*args, **kwargs):
        view = kwargs.get("cache")
        plan = getattr(view, "plan", None)
        if plan is not None:
            seen.append(rows_are_a_prefix(plan))
        return original(*args, **kwargs)

    engine.model.forward = spy
    _uneven(engine)
    assert seen and all(seen)


# --- the number the week was for -----------------------------------------------------


def _coverage(slots: int, requests: int, lengths: tuple[int, ...], compact: bool) -> float:
    engine = _graphed(compact=compact)
    before = CaptureStats.of(engine.decode_graphs)
    for i in range(requests):
        engine.add_request(
            Request(f"r{i}", [1 + (i % 7), 2, 3], max_new_tokens=lengths[i % len(lengths)])
        )
    while engine.has_unfinished():
        engine.step()
    return CaptureStats.of(engine.decode_graphs).since(before).replay_share


def test_the_workload_that_replayed_a_quarter_of_its_steps_now_replays_all_of_them():
    """Day 57's worst row, reproduced and then fixed. Four generation lengths is not
    a pathological workload, it is what any server sees, and it was the shape that
    left three quarters of the decode loop unable to use any graph in the list."""
    lengths = (2, 3, 5, 9)
    assert _coverage(4, 8, lengths, compact=False) < 1.0
    assert _coverage(4, 8, lengths, compact=True) == 1.0


def test_more_slots_no_longer_makes_the_coverage_worse():
    """The direction that said the design was wrong. A wider batch has more chances
    to be something other than a contiguous run from zero, so without compaction the
    share fell as the slots rose; with it there is nothing left to fall."""
    lengths = (2, 3, 5, 9)
    assert _coverage(2, 8, lengths, compact=True) == 1.0
    assert _coverage(4, 8, lengths, compact=True) == 1.0


def test_a_block_table_is_a_host_object_and_a_move_does_not_copy_one():
    """Worth pinning down, because it is the reason the move is cheap: the table
    that lands in the destination row is the same object, holding the same block
    ids, that the source row was holding."""
    cache = _cache()
    cache.adopt_row(2, cache.allocator.allocate_for(8))
    table = cache.tables[2]
    cache.move_row(2, 0)
    assert cache.tables[0] is table
    assert isinstance(cache.tables[2], BlockTable)
