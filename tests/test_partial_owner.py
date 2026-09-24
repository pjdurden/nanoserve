"""Day 67: the partials get one owner, and the owner is the line with the right rows.

Day 64 priced the split's partials into `CapturePlan.pool_bytes`, on the argument
that an allocation made while a graph records is served from the graph's pool. That
was true for one day. Day 64 also moved the partials into an arena allocated once,
Day 65 made the read refuse a launch without one, and Day 66 put that arena on its
own boot line. So as of last night a split server printed the partials twice: once
folded into the capture's "MiB of workspace" and once as the arena's own line, and
`/health` carried both under two keys. A reader summing the lines counted the same
bytes twice.

Deciding which line owns them turned out to be a question with an answer rather than
a taste, and the answer found a second bug:

  1. **The graph pool never holds a partial.** `PagedRead.__call__` refuses a split
     launch with no arena, so under capture the kernel writes into the arena and the
     pool holds the score tiles and nothing else. `pool_bytes` is the tiles.
  2. **The plan priced the partials at the wrong row count.** The capture list is
     trimmed by `--warm-rows`; the arena is not, because an unrecorded shape still
     runs eagerly against it. `plan_capture` checked the partials against the probe
     at the *recorded* rows, and the cache allocated them at the *served* rows. A
     `--warm-rows 4` server with 256 slots probed for a 64th of its arena.
  3. **So the plan now carries `arena_rows`,** prices `partial_bytes` at it, checks
     `pool_bytes + partial_bytes` against the budget, and the boot path refuses an
     arena whose bytes are not exactly the number the probe was asked about.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from nanoserve.captured import (
    PARTIAL_ITEMSIZE,
    eager_recorder,
    split_partial_bytes,
    split_score_cells,
    split_tile_bytes,
    split_workspace_bytes,
    streamed_workspace_bytes,
)
from nanoserve.compiled import DecodeShape
from nanoserve.engine import Engine
from nanoserve.launch import (
    BootUnsound,
    CaptureTooSmall,
    KVPoolPlan,
    arm_split_read,
    boot_info,
    boot_lines,
    check_arena_matches_capture,
    plan_capture,
)
from nanoserve.model import LlamaModel

from tests.test_split_boot import _app, _launch_kwargs, _tiny_config, _weights


def _engine(**kw) -> Engine:
    defaults = dict(
        num_blocks=256,
        block_size=4,
        max_batch_size=4,
        max_model_len=1024,
        read_block=4,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        split_read=True,
        capture_recorder=eager_recorder,
    )
    defaults.update(kw)
    return Engine.build(LlamaModel(_tiny_config(), _weights()), **defaults)


def _pool_plan(max_batch_size: int = 4) -> KVPoolPlan:
    return KVPoolPlan(
        num_blocks=256,
        block_size=4,
        max_batch_size=max_batch_size,
        max_model_len=1024,
        bytes_per_block=1024,
        budget_bytes=1 << 24,
        dtype=torch.float32,
    )


def _split(**kw):
    engine = _engine()
    return engine, plan_capture(engine, _pool_plan(), split_read=True, **kw)


# --- the two halves, priced apart ------------------------------------------------------


def test_the_split_workspace_is_its_tiles_plus_its_partials():
    """The Day 64 total is still the right number for one question, what a split
    launch needs in all, so it stays and becomes the sum of the two it always was."""
    shape = DecodeShape(rows=4, context_width=1024)
    tiles = split_tile_bytes(shape, 8, 4, 2)
    partials = split_partial_bytes(shape, 8, 2, 4)

    assert tiles == split_score_cells(shape, 8, 4, 2) * 4
    assert partials == 4 * 8 * 2 * (4 + 2) * PARTIAL_ITEMSIZE
    assert split_workspace_bytes(shape, 8, 4, 2, 4) == tiles + partials


def test_split_tiles_are_the_streamed_tile_once_per_chunk():
    shape = DecodeShape(rows=4, context_width=1024)
    assert split_tile_bytes(shape, 8, 4, 2) == 2 * streamed_workspace_bytes(shape, 8, 4)


def test_the_halves_refuse_what_the_total_refused():
    shape = DecodeShape(rows=4, context_width=1024)
    with pytest.raises(ValueError, match="byte"):
        split_tile_bytes(shape, 8, 4, 2, itemsize=0)
    with pytest.raises(ValueError, match="partial"):
        split_partial_bytes(shape, 8, 2, 4, partial_itemsize=0)


# --- which line owns which bytes ---------------------------------------------------------


def test_the_capture_pool_of_a_split_plan_holds_the_tiles_and_no_partial():
    """The reversal of Day 64's test of the same name. The pool is what a recording
    allocates from, and a split launch under capture allocates nothing: the read
    refuses to run without an arena, so the partials were never the pool's."""
    engine, capture = _split()

    assert capture.splits == 2
    assert capture.pool_bytes == max(
        split_tile_bytes(s, capture.num_heads, capture.block, capture.splits)
        for s in capture.shapes
    )


def test_a_split_pool_is_the_streamed_pool_times_the_chunks():
    """No partial in it means the ratio is exact, which is a check a reader can do in
    their head: a split holds one tile per chunk and nothing else in the pool."""
    engine = _engine()
    streamed = plan_capture(engine, _pool_plan())
    split = plan_capture(engine, _pool_plan(), split_read=True)

    assert split.pool_bytes == split.splits * streamed.pool_bytes


def test_a_split_plan_prices_its_partials_on_their_own():
    engine, capture = _split()

    assert capture.arena_rows == 4
    assert capture.partial_bytes == split_partial_bytes(
        DecodeShape(rows=4, context_width=capture.max_width),
        capture.num_heads,
        capture.splits,
        capture.head_dim,
    )
    assert capture.reserved_bytes == capture.pool_bytes + capture.partial_bytes


def test_the_other_two_reads_plan_no_arena():
    for kw in (dict(split_read=False, streamed_read=True), dict(split_read=False)):
        engine = _engine(**kw)
        capture = plan_capture(engine, _pool_plan())
        assert capture.arena_rows == 0
        assert capture.partial_bytes == 0
        assert capture.reserved_bytes == capture.pool_bytes


def test_the_plan_prices_exactly_the_arena_the_cache_allocates():
    """The equality this day is for. Two callers, two moments, one number: the probe
    was asked about `partial_bytes` and the process holds `workspace.bytes`."""
    engine, capture = _split()
    workspace = arm_split_read(engine, capture)

    assert workspace.bytes == capture.partial_bytes


# --- the row count the partials are priced at ----------------------------------------------


def test_trimming_the_list_does_not_trim_the_arena_the_plan_prices():
    """The second bug. `--warm-rows 1` records one row bucket and the cache still
    allocates the arena for every slot, because the shapes the list skipped still
    run eagerly against it. The plan priced the partials at 1 row."""
    engine, full = _split()
    _, trimmed = _split(max_rows=1)

    assert trimmed.max_rows == 1
    assert trimmed.arena_rows == 4
    assert trimmed.partial_bytes == full.partial_bytes
    assert trimmed.pool_bytes < full.pool_bytes
    workspace = arm_split_read(engine, trimmed)
    assert workspace.bytes == trimmed.partial_bytes


def test_a_budget_that_holds_the_trimmed_arena_but_not_the_served_one_is_refused():
    """What the probe was missing. At 1 recorded row the old check asked about a
    quarter of the partials; the arena the cache then reserved was all four rows."""
    engine, full = _split()
    one_row = DecodeShape(rows=1, context_width=full.max_width)
    old_need = split_workspace_bytes(
        one_row, full.num_heads, full.block, full.splits, full.head_dim
    )

    with pytest.raises(CaptureTooSmall, match="4 rows"):
        plan_capture(engine, _pool_plan(), split_read=True, max_rows=1,
                     budget_bytes=old_need)


def test_a_budget_that_holds_both_halves_boots():
    engine, full = _split()
    capture = plan_capture(engine, _pool_plan(), split_read=True,
                           budget_bytes=full.reserved_bytes)
    assert capture.reserved_bytes == full.reserved_bytes


# --- the arena gate learns the byte count -----------------------------------------------------


def test_an_arena_with_other_rows_than_the_plan_priced_is_refused():
    """Day 66 bounded the rows (the plan cannot record more than the arena holds).
    Today the plan also says how many rows it *priced*, and that one is an equality:
    an arena of any other size is memory the probe was never asked about."""
    engine, capture = _split()
    workspace = engine.cache.allocate_split_workspace()

    check_arena_matches_capture(capture, workspace)
    with pytest.raises(BootUnsound, match="priced"):
        check_arena_matches_capture(dataclasses.replace(capture, arena_rows=2), workspace)


# --- what the boot log and /health say ------------------------------------------------------


def test_the_capture_line_hands_the_partials_to_the_arena_line():
    engine, capture = _split()
    line = capture.describe()

    assert "partials on the split workspace line" in line
    streamed = plan_capture(_engine(split_read=False, streamed_read=True), _pool_plan())
    assert "partials" not in streamed.describe()


def test_health_carries_each_byte_once():
    """`cuda_graphs.workspace_bytes` is the pool and `split_workspace.partial_bytes` is
    the arena, and the sum of the two is what the probe was asked to hold."""
    engine, capture = _split()
    workspace = arm_split_read(engine, capture)
    info = boot_info(_pool_plan(), capture, workspace=workspace)

    graphs, arena = info["cuda_graphs"], info["split_workspace"]
    assert graphs["workspace_bytes"] == capture.pool_bytes
    assert "partial_bytes" not in graphs
    assert graphs["workspace_bytes"] + arena["partial_bytes"] == capture.reserved_bytes


def test_the_real_boot_path_reserves_what_it_printed():
    """Through `build_app`, the way `serve.py` boots, with the list trimmed below the
    slot count: the arena is still the one the plan priced."""
    app = _app(**_launch_kwargs(split_read=True, bucket_decode=True,
                                persist_inputs=True, capture_decode=True), warm_rows=1)
    capture, workspace = app.state.capture, app.state.workspace

    assert capture.max_rows == 1
    assert workspace.max_rows == 4
    assert workspace.bytes == capture.partial_bytes
    lines = boot_lines(app.state.plan, capture, workspace=workspace)
    assert sum("allocated once" in line for line in lines) == 1
