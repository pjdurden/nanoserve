"""Day 64 tests: the split's workspace becomes a thing the plan owns.

Day 63 built the split decode read and ended on one honest caveat: the kernel calls
`torch.empty` three times per read. That is a real allocation per layer per step
outside a graph, and inside one it is worse in a quieter way. It is legal, because
an allocation made under capture is served from the graph's pool and replayed at the
address it was recorded at, like every other intermediate. What it is not is
*priced*. `CapturePlan.pool_bytes` was the score term and nothing else, so a process
running this read reserves an arena it has under-reported by the partials, once per
graph, at the largest shape in the list, and learns the real number from the
allocator.

So the partials become a buffer the plan allocates once and hands to the kernel, the
way Day 51's slot table hands it a mapping. Three things here are the day.

**One split count for a whole list.** `choose_splits` answers per launch and a
capture list is a graph per shape against one workspace, so the answers have to
collapse. `plan_splits` takes the max, which is the direction that keeps the split:
a bucket handed more chunks than it wanted gets empty ones, and Day 63 made those
free.

**Only the row axis narrows.** Both passes address the workspace flat, so a window
is legal on the outermost axis and nowhere else. `SplitWorkspace.rows` is that, and
`check_partials` is the refusal for everything that looks like it but is not.

**Nothing needs initialising.** Every program stores its slot whether it walked a
tile or not, which is Day 63's empty-chunk design, and the consequence shows up here
a day later: a buffer reused across steps carries no state, and the `-inf` fill the
tlsim path had was decorative.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.kernels.flash_decoding import (
    PARTIAL_DTYPE,
    SplitUnsound,
    choose_splits,
    plan_splits,
)
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.partials import SplitWorkspace, allocate_for, allocate_partials


def _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4, device=None):
    return allocate_partials(
        max_rows=max_rows,
        n_q=n_q,
        head_dim=head_dim,
        context_width=context_width,
        block=block,
        splits=splits,
        device=device,
    )


def _case(seed=64, lens=(9, 3, 16), width=16, n_q=2, d=8, n_rep=1):
    torch.manual_seed(seed)
    n_kv = n_q // n_rep
    k_pool = torch.randn(64, n_kv, d)
    v_pool = torch.randn(64, n_kv, d)
    q = torch.randn(len(lens), n_q, 1, d)
    generator = torch.Generator().manual_seed(seed)
    mapping = torch.zeros(len(lens), width, dtype=torch.long)
    for i, n in enumerate(lens):
        mapping[i, :n] = torch.randperm(64, generator=generator)[:n]
    return q, k_pool, v_pool, mapping, torch.tensor(list(lens))


# --- what it allocates ---------------------------------------------------------------


def test_the_workspace_is_the_three_buffers_the_two_passes_share():
    ws = _ws()

    assert tuple(ws.part_max.shape) == (4, 2, 4)
    assert tuple(ws.part_denom.shape) == (4, 2, 4)
    assert tuple(ws.part_acc.shape) == (4, 2, 4, 8)
    assert ws.part_acc.dtype is PARTIAL_DTYPE


def test_the_partials_are_fp32_whatever_the_pool_holds():
    """The one dtype in this engine that is not a deployment choice. A partial makes
    a round trip through memory between the program that writes it and the program
    that rescales it by `exp(m - M)`, and that trip is where the range is needed."""
    ws = _ws()

    assert ws.part_max.dtype is PARTIAL_DTYPE
    assert ws.part_denom.dtype is PARTIAL_DTYPE
    assert ws.dtype is PARTIAL_DTYPE


def test_the_split_count_comes_from_the_row_buckets_when_nobody_names_one():
    """Which is `plan_splits`, and it is the *narrowest* bucket that decides."""
    ws = allocate_partials(
        max_rows=8,
        n_q=32,
        head_dim=64,
        context_width=8192,
        block=32,
        row_buckets=(1, 2, 4, 8),
    )

    assert ws.splits == plan_splits((1, 2, 4, 8), 32, 8192, 32)
    assert ws.splits == choose_splits(1, 32, 8192, 32)
    assert ws.splits > choose_splits(8, 32, 8192, 32)


def test_a_workspace_with_no_bucket_list_is_planned_for_its_own_row_ceiling():
    """The honest single-shape answer, and it is smaller. A list is the reason to
    ask for more splits than the widest batch wants."""
    ws = allocate_partials(
        max_rows=8, n_q=32, head_dim=64, context_width=8192, block=32
    )

    assert ws.splits == choose_splits(8, 32, 8192, 32)


def test_the_chunk_is_a_whole_number_of_tiles():
    ws = _ws(context_width=16, block=4, splits=4)

    assert ws.keys_per_split == 4
    assert ws.keys_per_split % ws.block == 0


def test_a_split_count_the_width_cannot_hold_comes_back_smaller():
    """`partition_width`'s floor is one tile, and the workspace records what the
    launch will really use rather than what was asked for."""
    ws = _ws(context_width=8, block=4, splits=8)

    assert ws.splits == 2
    assert tuple(ws.part_max.shape)[2] == 2


def test_allocating_refuses_nonsense():
    with pytest.raises(ValueError, match="at least one row"):
        _ws(max_rows=0)
    with pytest.raises(ValueError, match="at least one query head"):
        _ws(n_q=0)
    with pytest.raises(ValueError, match="at least one channel"):
        _ws(head_dim=0)


# --- what a window is ----------------------------------------------------------------


def test_a_row_window_is_the_buffers_own_storage():
    ws = _ws(max_rows=4)
    held = ws.rows(2)

    assert all(ws.is_window(t) for t in held)
    assert held[0].data_ptr() == ws.part_max.data_ptr()


def test_a_row_window_is_contiguous_so_the_flat_addressing_still_holds():
    """The reason the row axis is the only one a launch may take less than all of."""
    ws = _ws(max_rows=4)

    assert all(t.is_contiguous() for t in ws.rows(3))


def test_the_window_is_exactly_the_shape_the_grid_addresses():
    ws = _ws(max_rows=4, n_q=2, head_dim=8, splits=4)
    part_max, part_denom, part_acc = ws.rows(3)

    assert tuple(part_max.shape) == (3, 2, 4)
    assert tuple(part_denom.shape) == (3, 2, 4)
    assert tuple(part_acc.shape) == (3, 2, 4, 8)


def test_two_windows_are_the_same_addresses():
    """Allocated once is the whole claim, so it is the thing with a test on it."""
    ws = _ws()
    first = [t.data_ptr() for t in ws.rows(2)]
    second = [t.data_ptr() for t in ws.rows(4)]

    assert first == second


def test_a_window_wider_than_the_arena_is_refused():
    ws = _ws(max_rows=4)

    with pytest.raises(SplitUnsound, match="rows"):
        ws.rows(5)


def test_a_window_of_no_rows_is_refused():
    ws = _ws(max_rows=4)

    with pytest.raises(SplitUnsound, match="at least one row"):
        ws.rows(0)


def test_a_tensor_of_the_right_shape_is_not_a_window():
    """The property `is_window` exists to distinguish, and a copy passes every other
    check a kernel could make: same shape, same dtype, same contiguity."""
    ws = _ws()
    copy = ws.part_max[:2].clone()

    assert not ws.is_window(copy)


# --- what it costs -------------------------------------------------------------------


def test_the_arena_is_the_max_a_denominator_and_an_accumulator_per_program():
    ws = _ws(max_rows=4, n_q=2, head_dim=8, splits=4)

    assert ws.programs == 4 * 2 * 4
    assert ws.cells == 4 * 2 * 4 * (8 + 2)
    assert ws.bytes == 4 * 2 * 4 * 10 * 4


def test_the_arena_grows_linearly_in_the_number_the_tail_shrinks_by():
    """The trade stated as two calls: doubling the splits doubles the workspace."""
    small = _ws(context_width=16, block=4, splits=2)
    large = _ws(context_width=16, block=4, splits=4)

    assert large.splits == 2 * small.splits
    assert large.bytes == 2 * small.bytes


def test_the_workspace_renders_the_trade_and_not_just_the_size():
    ws = _ws()

    line = ws.render()
    assert "4 rows" in line
    assert "splits" in line
    assert "MiB" in line


def test_as_dict_says_what_a_health_payload_would_want():
    ws = _ws()

    payload = ws.as_dict()
    assert payload["splits"] == 4
    assert payload["keys_per_split"] == 4
    assert payload["partial_bytes"] == ws.bytes


# --- what it is for ------------------------------------------------------------------


def test_reading_through_the_workspace_matches_the_oracle():
    q, k_pool, v_pool, mapping, lens = _case()
    ws = _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4)

    got = ws.read(q, k_pool, v_pool, mapping, lens, n_rep=1)
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep=1)

    assert torch.allclose(got, oracle, atol=1e-5)


def test_reading_twice_through_one_workspace_returns_the_same_answer():
    """No step leaves state behind, because pass one writes every slot it is given."""
    q, k_pool, v_pool, mapping, lens = _case()
    ws = _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4)

    first = ws.read(q, k_pool, v_pool, mapping, lens, n_rep=1)
    second = ws.read(q, k_pool, v_pool, mapping, lens, n_rep=1)

    assert torch.allclose(first, second, atol=1e-6)


def test_a_read_does_not_move_the_addresses_it_was_planned_with():
    """Which is the only property that makes this worth a day: a captured region
    replays a launch bound to these pointers."""
    q, k_pool, v_pool, mapping, lens = _case()
    ws = _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4)
    before = ws.addresses

    ws.read(q, k_pool, v_pool, mapping, lens, n_rep=1)

    assert ws.addresses == before


def test_a_read_over_fewer_rows_than_the_arena_holds_is_still_the_oracle():
    """The row window doing its job: a two-row step against a four-row arena."""
    q, k_pool, v_pool, mapping, lens = _case(lens=(9, 3))
    ws = _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4)

    got = ws.read(q, k_pool, v_pool, mapping, lens, n_rep=1)
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep=1)

    assert torch.allclose(got, oracle, atol=1e-5)


def test_a_read_refuses_a_mapping_of_a_width_the_arena_was_not_partitioned_for():
    """The chunk bounds were computed from the plan's width. A narrower mapping is a
    different partition, so the launch would walk chunks the plan never sized."""
    q, k_pool, v_pool, mapping, lens = _case(lens=(4, 3), width=8)
    ws = _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4)

    with pytest.raises(SplitUnsound, match="width"):
        ws.read(q, k_pool, v_pool, mapping, lens, n_rep=1)


def test_a_read_refuses_a_query_with_the_wrong_head_count():
    q, k_pool, v_pool, mapping, lens = _case(n_q=4, n_rep=2)
    ws = _ws(max_rows=4, n_q=2, head_dim=8, context_width=16, block=4, splits=4)

    with pytest.raises(SplitUnsound, match="head"):
        ws.read(q, k_pool, v_pool, mapping, lens, n_rep=2)


# --- the seam onto a capture plan -----------------------------------------------------


class _Plan:
    """A `CapturePlan`-shaped record, which is all `allocate_for` reads."""

    max_rows = 8
    num_heads = 2
    max_width = 16
    block = 4
    splits = 4


def test_allocate_for_takes_the_numbers_a_capture_plan_already_decided():
    ws = allocate_for(_Plan(), head_dim=8)

    assert isinstance(ws, SplitWorkspace)
    assert ws.max_rows == 8
    assert ws.n_q == 2
    assert ws.splits == 4
    assert ws.context_width == 16


def test_allocate_for_refuses_a_plan_that_did_not_plan_a_split():
    """A plan with `splits` of 0 is a plan for one of the other two reads, and a
    workspace built off it would be an arena nothing in the process ever addresses."""
    plan = _Plan()
    plan.splits = 0

    with pytest.raises(ValueError, match="split"):
        allocate_for(plan, head_dim=8)
