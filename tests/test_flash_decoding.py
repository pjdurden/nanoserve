"""Day 63 tests: the tail becomes parallelism, and the partials become a workspace.

Day 62 measured the thing this day is about. A batched decode read launches one
program per `(row, query head)`, every program is resident, and the launch ends when
its slowest one retires. `LaunchWork.imbalance` is how many times the longest row
outlasts the average one, and on a long-tail batch it is 4.27x: three quarters of the
grid finished and waited.

Flash-decoding is the answer vLLM and SGLang ship. Cut one row's history into chunks,
give each chunk its own program with its own running max, denominator and weighted-V
accumulator, and reduce the partials in a second pass. The tail stops being a wait
and becomes more parallelism.

The file splits the way Day 62's did, and for the same reason: this box has no GPU.

The *plan* half is `split_plan` and `choose_splits`, ordinary Python over ints. The
partition is in whole tiles, because a chunk that starts in the middle of a tile is a
program whose first `tl.arange` is misaligned with every other program's; the refusal
when a `keys_per_split` is not a multiple of the block is that, stated once. And the
split count has to come from the mapping's *width*, which is a shape, not from the
batch's longest row, which is a tensor: reading the longest row is Day 49's readback,
and a split count that changes per step is a new graph per step.

The *reduction* half is `reduce_partials`, plain torch, which is the oracle the second
jitted pass gets held to. The load-bearing case is the empty partial: a split past a
short row's end walks nothing and still launches and still stores, so it stores
`-inf`, `0`, `0`, and the reduction has to fold that into exactly nothing.

The *model* half is `paged_attention_split_kernel`, the two passes as tlsim programs,
held to `paged_attention_batched_reference`: the same oracle Day 59 and Day 62 used.

The *device* half is gated behind `requires_triton_gpu`. Nothing here claims the
jitted bodies are verified on a box that cannot run them.
"""

from __future__ import annotations

import math

import pytest
import torch
from reference import requires_triton_gpu

from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.config import ModelConfig
from nanoserve.kernels.flash_decoding import (
    DEFAULT_PARTITION,
    DEFAULT_TARGET_PROGRAMS,
    SplitPlan,
    choose_splits,
    paged_attention_split,
    paged_attention_split_kernel,
    paged_attention_split_triton,
    reduce_partials,
    split_plan,
)
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.kernels.triton_batched_attention import launch_work


def _random_pools(num_slots, n_kv, d, seed, device="cpu"):
    torch.manual_seed(seed)
    k_pool = torch.randn(num_slots, n_kv, d, device=device)
    v_pool = torch.randn(num_slots, n_kv, d, device=device)
    return k_pool, v_pool


def _mapping(lens, width, num_slots, seed, device="cpu", pad=0):
    """A `[rows, width]` slot mapping whose rows hold `lens` real, scattered slots."""
    generator = torch.Generator().manual_seed(seed)
    rows = torch.full((len(lens), width), pad, dtype=torch.long)
    for i, n in enumerate(lens):
        rows[i, :n] = torch.randperm(num_slots, generator=generator)[:n]
    return rows.to(device)


def _partials(lens_per_split, d, seed=0):
    """Partial softmax state for one row and one head, one entry per split.

    Each split is given its own random keys and values, scored against one shared
    query, so the union of the splits is a single softmax with a known answer.
    Returns `(part_max, part_denom, part_acc, whole)` where `whole` is that answer.
    """
    torch.manual_seed(seed)
    q = torch.randn(d)
    part_max, part_denom, part_acc, scores, values = [], [], [], [], []
    for n in lens_per_split:
        if n == 0:  # a split past the end of the row: it walks nothing and stores this
            part_max.append(float("-inf"))
            part_denom.append(0.0)
            part_acc.append(torch.zeros(d))
            continue
        k = torch.randn(n, d)
        v = torch.randn(n, d)
        s = (k * q[None, :]).sum(dim=1)
        m = float(s.max())
        p = torch.exp(s - m)
        part_max.append(m)
        part_denom.append(float(p.sum()))
        part_acc.append((p[:, None] * v).sum(dim=0))
        scores.append(s)
        values.append(v)
    whole_scores = torch.cat(scores)
    whole = (torch.softmax(whole_scores, dim=0)[:, None] * torch.cat(values)).sum(dim=0)
    return (
        torch.tensor(part_max)[None, None, :],
        torch.tensor(part_denom)[None, None, :],
        torch.stack(part_acc)[None, None],
        whole,
    )


# --- the partition is in whole tiles -----------------------------------------


def test_split_plan_partitions_the_width_into_whole_tiles():
    """A chunk boundary that falls inside a tile is a program whose first offset ramp
    is misaligned with every other program's, so the partition is in tiles and the
    keys follow. 256 keys in 32-key tiles is 8 tiles; 4 splits is 2 tiles each."""
    plan = split_plan([256, 100, 7], n_q=4, head_dim=8, block=32, splits=4, context_width=256)
    assert plan.keys_per_split == 64  # 2 tiles, not 256 / 4 by accident
    assert plan.keys_per_split % plan.block == 0
    assert plan.splits == 4
    assert plan.rows == 3


def test_split_plan_refuses_a_keys_per_split_that_is_not_whole_tiles():
    """The refusal the alignment buys. 48 keys is a tile and a half, so split 1 would
    start at key 48, which is lane 16 of tile 1, and its `arange` would straddle two
    tiles that another program also owns."""
    with pytest.raises(ValueError, match="multiple of the tile|whole tiles"):
        split_plan([100], n_q=1, head_dim=8, block=32, keys_per_split=48, context_width=128)


def test_split_plan_takes_a_split_count_or_a_chunk_size_and_not_both():
    """The same either/or the read's `validated` / `context_bounds` pair is: both
    arguments name the same partition, and only one of them can be the reason for it."""
    with pytest.raises(ValueError, match="either"):
        split_plan(
            [100], n_q=1, head_dim=8, block=32, splits=2, keys_per_split=64, context_width=128
        )
    with pytest.raises(ValueError, match="either"):
        split_plan([100], n_q=1, head_dim=8, block=32, context_width=128)


def test_a_chunk_size_names_the_split_count_directly():
    """Handed a chunk, the plan counts how many of them cover the width. This is the
    vLLM shape: a fixed partition size, and the split count falls out of the width."""
    plan = split_plan([500], n_q=1, head_dim=8, block=32, keys_per_split=256, context_width=1024)
    assert plan.splits == 4
    assert plan.keys_per_split == 256


def test_asking_for_more_splits_than_there_are_tiles_gets_one_tile_each():
    """The floor on a chunk is one tile, so a 4-tile width cannot be cut eight ways.
    The plan reports what the launch will really be, not what was asked for."""
    plan = split_plan([100], n_q=1, head_dim=8, block=32, splits=8, context_width=100)
    assert plan.keys_per_split == 32
    assert plan.splits == 4  # cdiv(100, 32), not 8


def test_asking_for_three_splits_of_four_tiles_gets_two():
    """Whole tiles do not divide evenly, and the plan rounds the chunk up rather than
    handing the last split a fraction. Four tiles in three pieces is two pieces of
    two, which is worth saying out loud because the requested number is not the one
    that launches."""
    plan = split_plan([128], n_q=1, head_dim=8, block=32, splits=3, context_width=128)
    assert plan.keys_per_split == 64
    assert plan.splits == 2


def test_one_split_is_the_unsplit_launch():
    """`splits=1` is Day 62's kernel exactly: one program per (row, head), each
    walking its whole row. The plan has to agree with `launch_work` there or the
    comparison the whole day rests on is between two different things."""
    lens = [100, 40, 7]
    plan = split_plan(lens, n_q=8, head_dim=16, block=32, splits=1, context_width=128)
    work = launch_work(lens, n_q=8, block=32, context_width=128)
    assert plan.splits == 1
    assert plan.programs == work.programs
    assert plan.tiles == work.tiles
    assert plan.tail_tiles == work.tail_tiles
    assert plan.wave_speedup == pytest.approx(1.0)


# --- what the split does and does not change ---------------------------------


def test_splitting_does_not_change_the_tiles_walked():
    """Conservation, and it is the reason the split is free in work terms. The same
    keys are read either way; they are read by more programs. Every ratio in the day
    is about *when* the tiles are walked, never how many."""
    lens = [1000, 400, 33, 7]
    unsplit = split_plan(lens, n_q=8, head_dim=16, block=32, splits=1, context_width=1024)
    split = split_plan(lens, n_q=8, head_dim=16, block=32, splits=8, context_width=1024)
    assert split.tiles == unsplit.tiles
    assert split.programs == 8 * unsplit.programs


def test_the_tail_becomes_the_longest_chunk_and_not_the_longest_row():
    """The whole day in one assertion. Unsplit, the slowest program walks the longest
    row: 1024 keys is 32 tiles. Cut into 8 chunks of 128 keys, the slowest program
    walks 4 tiles, and the launch is 8 times shorter for the same work."""
    lens = [1024, 64, 32]
    plan = split_plan(lens, n_q=8, head_dim=16, block=32, splits=8, context_width=1024)
    assert plan.unsplit_tail_tiles == 32
    assert plan.tail_tiles == 4
    assert plan.wave_speedup == pytest.approx(8.0)


def test_the_wave_speedup_stops_at_the_longest_row_not_the_width():
    """A split of the width is not a split of the batch. If every row is short, the
    chunks past its end are empty and the tail is already small, so more splits buy
    nothing: the speedup is capped by how many tiles the longest row actually holds."""
    plan = split_plan([64, 32], n_q=4, head_dim=8, block=32, splits=32, context_width=1024)
    assert plan.tail_tiles == 1  # the longest row is 2 tiles, cut at a 1-tile chunk
    assert plan.wave_speedup == pytest.approx(2.0)


def test_a_short_rows_later_splits_walk_nothing_and_still_launch():
    """The cost side, and it is not a rounding error. A split is a grid axis, so every
    row gets `splits` programs whether its history reaches them or not. Row 1 here has
    one tile of history and 7 of its 8 programs load their bound, find nothing to do,
    store an empty partial and retire."""
    plan = split_plan([1024, 20], n_q=2, head_dim=8, block=32, splits=8, context_width=1024)
    assert plan.split_tiles[0] == (4, 4, 4, 4, 4, 4, 4, 4)
    assert plan.split_tiles[1] == (1, 0, 0, 0, 0, 0, 0, 0)
    assert plan.idle_programs == 2 * 7  # n_q * the empty (row, split) pairs
    assert plan.idle_fraction == pytest.approx(7 / 16)


def test_a_uniform_batch_wastes_no_programs():
    """Nothing is idle when every row reaches every chunk, which is the case the split
    is unambiguously good in and also the case with the least imbalance to recover."""
    plan = split_plan([256, 256, 256], n_q=4, head_dim=8, block=32, splits=4, context_width=256)
    assert plan.idle_programs == 0
    assert plan.idle_fraction == pytest.approx(0.0)
    assert plan.wave_speedup == pytest.approx(4.0)


def test_the_imbalance_after_a_split_counts_the_empty_programs():
    """An idle program is still a scheduled program, so it belongs in the mean. That
    makes the post-split imbalance look worse than the tail alone suggests, and it
    should: `idle_fraction` is the other half of what the split cost."""
    plan = split_plan([1024, 32], n_q=1, head_dim=8, block=32, splits=8, context_width=1024)
    assert plan.tiles == 33  # 32 tiles of history, plus row 1's single tile
    assert plan.mean_tiles == pytest.approx(33 / 16)
    assert plan.imbalance == pytest.approx(4 / (33 / 16))


def test_at_a_one_tile_chunk_the_idle_programs_are_exactly_day_59s_skipped_tiles():
    """The identity the day turns on, and I did not expect it to be exact.

    Cut the width into one-tile chunks and a row of length L fills exactly
    `cdiv(L, block)` of its chunks, so the empty ones number
    `rectangle_tiles - tiles`: the *same* tiles Day 59's `ragged_saving` counts as
    skipped. The streamed read's whole win reappears as grid area. It did not go
    away, it changed what it is made of, and an idle program costs a launch slot
    rather than a memory read, which is why the trade is still worth making.
    """
    lens = [8192] + [1024] * 7
    work = launch_work(lens, n_q=32, block=32, context_width=8192)
    plan = split_plan(
        lens, n_q=32, head_dim=128, block=32, keys_per_split=32, context_width=8192
    )
    assert plan.splits == 256  # one tile each
    assert plan.programs == work.rectangle_tiles
    assert plan.idle_programs == work.rectangle_tiles - work.tiles
    assert plan.idle_fraction == pytest.approx(1 - 1 / work.work_saving)


# --- the workspace the partials need -----------------------------------------


def test_the_partials_are_a_real_workspace_and_this_is_its_size():
    """Every program stores a max, a denominator and a `head_dim`-wide accumulator, so
    the launch needs `[rows, heads, splits]` of `head_dim + 2` fp32 numbers. Day 59
    took the score rectangle away; this is the first thing since that gives memory
    back, and it is priced here so the capture plan can own it later."""
    plan = split_plan([512] * 4, n_q=32, head_dim=128, block=32, splits=8, context_width=512)
    assert plan.partial_elements == 4 * 32 * 8 * (128 + 2)
    assert plan.partial_bytes == plan.partial_elements * 4  # fp32, always
    assert plan.partial_mib == pytest.approx(plan.partial_bytes / 2**20)


def test_the_partial_workspace_is_linear_in_the_split_count():
    """Which is the trade in one line: the tail shrinks by the split count and the
    workspace grows by it, so the split count is a memory decision as much as a
    latency one."""
    small = split_plan([512], n_q=8, head_dim=64, block=32, splits=2, context_width=512)
    big = split_plan([512], n_q=8, head_dim=64, block=32, splits=8, context_width=512)
    assert big.partial_bytes == 4 * small.partial_bytes
    assert big.wave_speedup == pytest.approx(4 * small.wave_speedup)


def test_the_partials_are_fp32_whatever_the_pool_holds():
    """A partial is a max and a running sum that another program is about to rescale
    by `exp(m - M)`. In fp16 that rescale is where the accuracy of the whole read
    goes, so the workspace has one dtype and the pool's is not it."""
    plan = split_plan([64], n_q=1, head_dim=4, block=32, splits=2, context_width=64)
    assert plan.partial_dtype is torch.float32
    assert plan.partial_bytes == 1 * 1 * 2 * 6 * 4


# --- what a plan refuses -----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"block": 0}, "tile holds at least one key"),
        ({"n_q": 0}, "at least one query head"),
        ({"head_dim": 0}, "at least one channel"),
        ({"splits": 0}, "at least one split"),
    ],
)
def test_split_plan_refuses_a_launch_it_cannot_describe(kwargs, message):
    """Every one of these is an integer the kernel turns into a pointer or a loop
    bound, and a zero in any of them is a launch that reads something else."""
    args = {
        "context_lens": [64],
        "n_q": 1,
        "head_dim": 8,
        "block": 32,
        "splits": 2,
        "context_width": 64,
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        split_plan(**args)


def test_split_plan_refuses_an_empty_batch_and_an_empty_row():
    """The same two refusals every read in this repo makes, for the same reason: a
    query with no visible key softmaxes over nothing, and here it also walks no tiles
    in any split, so the reduction would divide zero by zero."""
    with pytest.raises(ValueError, match="at least one row"):
        split_plan([], n_q=1, head_dim=8, block=32, splits=2, context_width=64)
    with pytest.raises(ValueError, match="at least 1 for every row"):
        split_plan([4, 0], n_q=1, head_dim=8, block=32, splits=2, context_width=64)


def test_split_plan_refuses_a_row_longer_than_the_width():
    """The mapping's width is the row stride, so a row claiming more history than it
    holds is a program that walks into the next row's slots."""
    with pytest.raises(ValueError, match="width"):
        split_plan([100], n_q=1, head_dim=8, block=32, splits=2, context_width=64)


def test_split_plan_defaults_the_width_to_the_longest_row():
    """A tight mapping, which is what an unbucketed batch is."""
    plan = split_plan([100, 40], n_q=1, head_dim=8, block=32, splits=2)
    assert plan.context_width == 100
    assert plan.keys_per_split == 64  # cdiv(cdiv(100, 32), 2) * 32


def test_a_plan_renders_one_line_with_the_trade_in_it():
    """A bench row needs the tail, the speedup and the workspace next to each other,
    because the day's claim is a trade and not a win."""
    plan = split_plan([1024, 64], n_q=8, head_dim=64, block=32, splits=8, context_width=1024)
    line = plan.render()
    assert "8 splits" in line
    assert "8.00x" in line  # the wave speedup: a 32-tile tail became a 4-tile one
    assert "MiB" in line


# --- choosing the split count ------------------------------------------------


def test_a_long_context_and_a_small_batch_is_what_a_split_is_for():
    """One row of 8192 tokens on 32 heads is 32 programs, and a card that holds
    thousands is idle. The split is the only way that launch fills the machine."""
    assert choose_splits(rows=1, n_q=32, context_width=8192) == 16


def test_a_big_batch_does_not_split_because_the_grid_is_already_full():
    """256 rows on 32 heads is 8192 programs, which oversubscribes anything. Splitting
    there buys no parallelism and costs a workspace and a second pass, so it does not
    happen. This is why vLLM keeps both kernels rather than replacing one."""
    assert choose_splits(rows=256, n_q=32, context_width=8192) == 1


def test_a_split_never_goes_below_the_partition_floor():
    """A chunk of a handful of tiles pays a full reduction for almost no walking, so
    the partition is a floor on the chunk and not just a target. A 256-token width is
    below it and is never split at all."""
    assert choose_splits(rows=1, n_q=1, context_width=256) == 1
    assert choose_splits(rows=1, n_q=1, context_width=DEFAULT_PARTITION) == 1
    assert choose_splits(rows=1, n_q=1, context_width=2 * DEFAULT_PARTITION) == 2


def test_the_split_count_comes_from_the_width_and_never_from_the_batch():
    """Capture safety, and it is Day 49 and Day 61 in one argument. The longest row is
    a number inside a tensor, so asking for it is a readback the host waits for and a
    guard a tracer rebuilds on. The width is a *shape*, and under Day 61's streamed
    bucket set it is `max_model_len` on every step forever, so a split count derived
    from it is a launch constant for the life of the process."""
    early = choose_splits(rows=4, n_q=32, context_width=8192)
    late = choose_splits(rows=4, n_q=32, context_width=8192)
    assert early == late == 16  # nothing about the batch's lengths appears in the call
    assert choose_splits(rows=4, n_q=32, context_width=8192, target_programs=512) == 4


def test_choose_splits_fills_the_grid_and_stops():
    """The grid bound: enough splits to reach `target_programs` and not one more,
    because past that the card is a queue again and a split is pure overhead."""
    assert DEFAULT_TARGET_PROGRAMS == 2048  # a stand-in for the card's resident grid
    assert choose_splits(rows=8, n_q=32, context_width=8192) == 8
    assert choose_splits(rows=16, n_q=32, context_width=8192) == 4
    assert choose_splits(rows=64, n_q=32, context_width=8192) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rows": 0}, "at least one row"),
        ({"n_q": 0}, "at least one query head"),
        ({"context_width": 0}, "at least one key"),
        ({"block": 0}, "tile holds at least one key"),
        ({"target_programs": 0}, "at least one program"),
        ({"partition": 16}, "at least one tile"),
    ],
)
def test_choose_splits_refuses_what_it_cannot_answer(kwargs, message):
    args = {"rows": 2, "n_q": 4, "context_width": 4096, "block": 32}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        choose_splits(**args)


# --- the reduction -----------------------------------------------------------


def test_reduce_partials_rebuilds_the_softmax_the_splits_were_cut_from():
    """The claim the second pass makes: three programs that each saw a third of the
    history and never spoke to each other combine into the softmax over all of it.
    The rescale is `exp(m_s - M)`, which is the same renormalisation the online
    softmax does inside one program, hoisted one level out."""
    part_max, part_denom, part_acc, whole = _partials([5, 7, 3], d=8, seed=1)
    out = reduce_partials(part_max, part_denom, part_acc)
    assert out.shape == (1, 1, 1, 8)
    assert torch.allclose(out[0, 0, 0], whole, atol=1e-5)


def test_an_empty_partial_contributes_exactly_nothing():
    """The load-bearing case. A split past a short row's end stores `-inf`, `0`, `0`,
    and `exp(-inf - M)` is exactly zero for any finite `M`, so the empty programs fall
    out of the sum without a branch anywhere. That is why the kernel can store an
    empty partial instead of skipping the store, which it could not do anyway."""
    part_max, part_denom, part_acc, whole = _partials([6, 0, 0], d=8, seed=2)
    out = reduce_partials(part_max, part_denom, part_acc)
    assert torch.allclose(out[0, 0, 0], whole, atol=1e-6)


def test_the_reduction_does_not_care_what_order_the_splits_retire_in():
    """Programs retire in whatever order the card feels like. The reduction reads a
    workspace, so the answer cannot depend on that, and the max-then-rescale form is
    what makes it true rather than nearly true."""
    part_max, part_denom, part_acc, _ = _partials([4, 9, 2], d=8, seed=3)
    order = torch.tensor([2, 0, 1])
    shuffled = reduce_partials(
        part_max[:, :, order], part_denom[:, :, order], part_acc[:, :, order]
    )
    assert torch.allclose(reduce_partials(part_max, part_denom, part_acc), shuffled, atol=1e-6)


def test_the_reduction_survives_a_split_whose_scores_are_far_larger():
    """Why the max comes out first. If one chunk's scores are 60 larger than another's,
    a reduction that summed `exp(m_s) * denom_s` directly would overflow fp32 on the
    big one and flush the small one to zero. Rescaling to the joint max keeps both."""
    part_max = torch.tensor([[[-80.0, 80.0]]])
    part_denom = torch.tensor([[[2.0, 3.0]]])
    part_acc = torch.stack([torch.full((4,), 2.0), torch.full((4,), 9.0)])[None, None]
    out = reduce_partials(part_max, part_denom, part_acc)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 0, 0], torch.full((4,), 3.0), atol=1e-5)


def test_reduce_partials_refuses_shapes_that_do_not_line_up():
    """The workspace is three buffers that have to be indexed by the same
    `(row, head, split)`, and a kernel writing them has no shapes to check."""
    m = torch.zeros(2, 3, 4)
    denom = torch.ones(2, 3, 4)
    acc = torch.ones(2, 3, 4, 8)
    with pytest.raises(ValueError, match="same"):
        reduce_partials(m, denom[:, :, :3], acc)
    with pytest.raises(ValueError, match="one accumulator per split"):
        reduce_partials(m, denom, acc[:, :, :3])
    with pytest.raises(ValueError, match=r"\[rows, heads, splits\]"):
        reduce_partials(m[0], denom[0], acc[0])


def test_reduce_partials_refuses_a_row_whose_every_split_was_empty():
    """`-inf` everywhere means no program saw a key, so the joint max is `-inf`, the
    rescale is `-inf - -inf` and the answer is NaN rather than an error. Every row is
    required to have at least one key upstream, which makes this unreachable, so it is
    checked here where it is one comparison and not in the kernel where it is none."""
    m = torch.tensor([[[float("-inf"), float("-inf")]]])
    denom = torch.zeros(1, 1, 2)
    acc = torch.zeros(1, 1, 2, 4)
    with pytest.raises(ValueError, match="no split saw a key"):
        reduce_partials(m, denom, acc)


def test_one_non_empty_split_reduces_to_itself():
    """The degenerate case the unsplit launch is: one partial in, that partial
    normalised out, and no rescale anywhere."""
    acc = torch.arange(4, dtype=torch.float32)[None, None, None]
    out = reduce_partials(torch.zeros(1, 1, 1), torch.full((1, 1, 1), 2.0), acc)
    assert torch.allclose(out[0, 0, 0], acc[0, 0, 0] / 2.0)


# --- the two passes as a model -----------------------------------------------


@pytest.mark.parametrize("splits", [1, 2, 3, 7])
def test_the_split_read_matches_the_oracle_on_a_ragged_batch(splits):
    """The day's correctness claim, and the parametrisation is the point of it: the
    split count is a performance knob and every value returns the same attention.
    Held to `paged_attention_batched_reference`, the oracle Day 59 and Day 62 use."""
    lens = [17, 3, 40, 1]
    k_pool, v_pool = _random_pools(64, n_kv=2, d=8, seed=10)
    q = torch.randn(len(lens), 8, 1, 8)
    mapping = _mapping(lens, width=40, num_slots=64, seed=10)
    context_lens = torch.tensor(lens)

    got = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, block=8, splits=splits
    )
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep=4)
    assert got.shape == oracle.shape
    assert torch.allclose(got, oracle, atol=1e-5)


def test_the_split_read_matches_the_oracle_on_a_uniform_batch():
    """No row has an empty split here, so nothing is masked out of the reduction and a
    failure is the walk or the rescale rather than the empty-partial path."""
    lens = [16, 16, 16]
    k_pool, v_pool = _random_pools(32, n_kv=1, d=4, seed=11)
    q = torch.randn(3, 4, 1, 4)
    mapping = _mapping(lens, width=16, num_slots=32, seed=11)
    context_lens = torch.tensor(lens)

    got = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, block=8, splits=2
    )
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep=4)
    assert torch.allclose(got, oracle, atol=1e-5)


def test_one_split_is_the_day_59_walk_again():
    """A sanity rail on the refactor: with a single split there is one program per
    (row, head) and the reduction divides one accumulator by one denominator, so the
    answer has to match the unsplit streamed read and not merely the oracle."""
    from nanoserve.kernels.paged_attention import paged_attention_batched_kernel

    lens = [9, 2, 31]
    k_pool, v_pool = _random_pools(64, n_kv=2, d=8, seed=12)
    q = torch.randn(3, 4, 1, 8)
    mapping = _mapping(lens, width=32, num_slots=64, seed=12)
    context_lens = torch.tensor(lens)

    split = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=16, splits=1
    )
    streamed = paged_attention_batched_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=16
    )
    assert torch.allclose(split, streamed, atol=1e-6)


def test_the_split_read_never_loads_the_padding_past_a_row():
    """Ragged, not rectangular, and the split does not weaken it. The pad names slot
    63, which holds NaN, and a row's empty splits are exactly the programs most likely
    to read it: they compute a base offset past the row's end and must mask it off."""
    lens = [5, 2, 9]
    k_pool, v_pool = _random_pools(64, n_kv=1, d=4, seed=13)
    k_pool[63] = float("nan")
    v_pool[63] = float("nan")
    q = torch.randn(3, 2, 1, 4)
    mapping = _mapping(lens, width=32, num_slots=63, seed=13, pad=-1)
    mapping[mapping < 0] = 63
    context_lens = torch.tensor(lens)

    got = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=4, splits=4
    )
    assert torch.isfinite(got).all()


def test_the_split_read_uses_the_gqa_head_mapping_from_the_config():
    """Query head h reads KV head h // n_rep, and the split adds a third grid axis
    without touching that. A wrong head mapping is a real key from the wrong head,
    which is finite and plausible, so it is pinned against the oracle."""
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=384,  # 8 heads x 48 channels
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=48,  # not a power of two: the channel mask carries it
    )
    d = cfg.head_dim
    lens = [12, 4]
    k_pool, v_pool = _random_pools(32, cfg.num_key_value_heads, d, seed=14)
    q = torch.randn(2, cfg.num_attention_heads, 1, d)
    mapping = _mapping(lens, width=16, num_slots=32, seed=14)
    context_lens = torch.tensor(lens)

    got = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=cfg.num_kv_groups, block=8, splits=3
    )
    oracle = paged_attention_batched_reference(
        q, k_pool, v_pool, mapping, context_lens, n_rep=cfg.num_kv_groups
    )
    assert torch.allclose(got, oracle, atol=1e-5)


def test_the_split_read_refuses_what_its_oracle_refuses():
    """A kernel that accepts inputs its oracle rejects cannot be compared to it, so
    the split path runs the same `check_batched_inputs` the unsplit one does."""
    k_pool, v_pool = _random_pools(16, n_kv=1, d=4, seed=15)
    mapping = _mapping([4], width=8, num_slots=16, seed=15)
    with pytest.raises(ValueError, match="one new token per row"):
        paged_attention_split_kernel(
            torch.randn(1, 2, 3, 4), k_pool, v_pool, mapping, torch.tensor([4]), n_rep=2
        )
    with pytest.raises(ValueError, match="at least 1 for every row"):
        paged_attention_split_kernel(
            torch.randn(1, 2, 1, 4), k_pool, v_pool, mapping, torch.tensor([0]), n_rep=2
        )


def test_the_default_split_count_is_chosen_from_the_width():
    """No argument means `choose_splits` on the mapping's width and the grid, never on
    the batch's contents: the default must not be a reason to touch the tensor."""
    lens = [30, 6]
    k_pool, v_pool = _random_pools(64, n_kv=1, d=4, seed=16)
    q = torch.randn(2, 4, 1, 4)
    mapping = _mapping(lens, width=32, num_slots=64, seed=16)
    context_lens = torch.tensor(lens)

    default = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, block=8
    )
    explicit = paged_attention_split_kernel(
        q,
        k_pool,
        v_pool,
        mapping,
        context_lens,
        n_rep=4,
        block=8,
        splits=choose_splits(rows=2, n_q=4, context_width=32, block=8),
    )
    assert torch.equal(default, explicit)


def test_the_split_read_carries_the_scale_and_the_validated_flag():
    """Both are arguments the planned decode path already passes to the unsplit read,
    and a path that quietly drops either is a path that cannot replace it."""
    lens = [11, 5]
    k_pool, v_pool = _random_pools(32, n_kv=1, d=4, seed=17)
    q = torch.randn(2, 2, 1, 4)
    mapping = _mapping(lens, width=16, num_slots=32, seed=17)
    context_lens = torch.tensor(lens)

    got = paged_attention_split_kernel(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, scale=0.25, splits=2, validated=True
    )
    oracle = paged_attention_batched_reference(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, scale=0.25, validated=True
    )
    assert torch.allclose(got, oracle, atol=1e-5)
    with pytest.raises(ValueError, match="either validated"):
        paged_attention_split_kernel(
            q,
            k_pool,
            v_pool,
            mapping,
            context_lens,
            n_rep=2,
            validated=True,
            context_bounds=(5, 11),
        )


# --- the dispatch ------------------------------------------------------------


def test_the_split_read_falls_back_to_the_cpu_model_on_a_host_tensor():
    """Same contract as Day 62's dispatcher: a CUDA tensor with Triton gets the two
    jitted passes, everything else gets the tlsim model of the same two passes."""
    lens = [7, 3]
    k_pool, v_pool = _random_pools(32, n_kv=1, d=4, seed=18)
    q = torch.randn(2, 2, 1, 4)
    mapping = _mapping(lens, width=8, num_slots=32, seed=18)
    context_lens = torch.tensor(lens)

    got = paged_attention_split(q, k_pool, v_pool, mapping, context_lens, n_rep=2, splits=2)
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep=2)
    assert torch.allclose(got, oracle, atol=1e-5)


def test_the_jitted_split_read_refuses_host_tensors_rather_than_crashing():
    """Calling the Triton entry point directly means demanding the GPU, which is what
    a kernel test does; on this box it says so instead of failing in a launch."""
    lens = [4]
    k_pool, v_pool = _random_pools(16, n_kv=1, d=4, seed=19)
    q = torch.randn(1, 2, 1, 4)
    mapping = _mapping(lens, width=8, num_slots=16, seed=19)
    with pytest.raises((RuntimeError, ValueError), match="cuda|CUDA|Triton|triton"):
        paged_attention_split_triton(
            q, k_pool, v_pool, mapping, torch.tensor(lens), n_rep=2, splits=2
        )


def test_the_split_read_refuses_a_tile_of_no_keys():
    """`block` reaches the kernel as a tile extent and the plan as a divisor, and zero
    is a wrong answer in both."""
    lens = [4]
    k_pool, v_pool = _random_pools(16, n_kv=1, d=4, seed=20)
    q = torch.randn(1, 2, 1, 4)
    mapping = _mapping(lens, width=8, num_slots=16, seed=20)
    with pytest.raises(ValueError, match="tile holds at least one key"):
        paged_attention_split(q, k_pool, v_pool, mapping, torch.tensor(lens), n_rep=2, block=0)


def test_a_plan_is_frozen_because_it_describes_a_launch_that_is_about_to_happen():
    """Same reason `BatchedGeometry` and `LaunchWork` are: a description that can be
    edited after the launch is a description of nothing."""
    plan = split_plan([64], n_q=1, head_dim=8, block=32, splits=2, context_width=64)
    assert isinstance(plan, SplitPlan)
    with pytest.raises(AttributeError):
        plan.splits = 4


def test_the_split_plan_agrees_with_the_walk_the_model_actually_does():
    """The arithmetic and the loop are written separately and have to say the same
    thing, so the plan's per-split tile counts are checked against a direct count of
    the tiles the chunks imply."""
    lens = [1000, 400, 33, 7]
    plan = split_plan(lens, n_q=1, head_dim=8, block=32, splits=8, context_width=1024)
    for row, ctx in enumerate(lens):
        for s in range(plan.splits):
            lo = s * plan.keys_per_split
            hi = min(lo + plan.keys_per_split, ctx)
            assert plan.split_tiles[row][s] == max(0, math.ceil((hi - lo) / 32))


# --- the device half, gated --------------------------------------------------


@requires_triton_gpu
def test_triton_split_kernel_matches_the_oracle_on_a_ragged_batch():
    """The two jitted passes against the same oracle the model is held to."""
    lens = [17, 3, 40, 1]
    k_pool, v_pool = _random_pools(64, n_kv=2, d=8, seed=40, device="cuda")
    q = torch.randn(len(lens), 8, 1, 8, device="cuda")
    mapping = _mapping(lens, width=40, num_slots=64, seed=40, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    got = paged_attention_split_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, block=8, splits=4
    )
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep=4)
    assert torch.allclose(got, oracle, atol=1e-3)


@requires_triton_gpu
@pytest.mark.parametrize("splits", [1, 2, 5])
def test_triton_split_kernel_is_transparent_to_the_split_count(splits):
    """Every split count is the same attention, on hardware as in the model."""
    lens = [31, 8, 20]
    k_pool, v_pool = _random_pools(64, n_kv=1, d=8, seed=41, device="cuda")
    q = torch.randn(3, 4, 1, 8, device="cuda")
    mapping = _mapping(lens, width=32, num_slots=64, seed=41, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    got = paged_attention_split_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, block=16, splits=splits
    )
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep=4)
    assert torch.allclose(got, oracle, atol=1e-3)


@requires_triton_gpu
def test_triton_split_kernel_never_loads_the_padding_past_a_row():
    """A -1 pad naming a NaN slot, read by the empty programs a split manufactures."""
    lens = [5, 2, 9]
    k_pool, v_pool = _random_pools(64, n_kv=1, d=8, seed=42, device="cuda")
    k_pool[63] = float("nan")
    v_pool[63] = float("nan")
    q = torch.randn(3, 2, 1, 8, device="cuda")
    mapping = _mapping(lens, width=32, num_slots=63, seed=42, device="cuda", pad=-1)
    context_lens = torch.tensor(lens, device="cuda")

    got = paged_attention_split_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=4, splits=4
    )
    assert torch.isfinite(got).all()


@requires_triton_gpu
def test_triton_split_kernel_matches_the_unsplit_kernel():
    """The two jitted reads are the same computation associated differently, so they
    agree to a flash-attention tolerance and not bit for bit, and the difference is
    the reassociation and nothing else."""
    from nanoserve.kernels.triton_batched_attention import paged_attention_batched_triton

    lens = [64, 9, 33]
    k_pool, v_pool = _random_pools(128, n_kv=2, d=8, seed=43, device="cuda")
    q = torch.randn(3, 4, 1, 8, device="cuda")
    mapping = _mapping(lens, width=64, num_slots=128, seed=43, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    split = paged_attention_split_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=16, splits=4
    )
    unsplit = paged_attention_batched_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=16
    )
    assert torch.allclose(split, unsplit, atol=1e-3)


@requires_triton_gpu
def test_triton_split_kernel_reads_genuinely_scattered_blocks():
    """The slots come from a real allocator, so the rows interleave in the pool the
    way a live batch's do and a program that assumed contiguity would be caught."""
    block_size = 4
    allocator = BlockAllocator(num_blocks=32, block_size=block_size)
    lens = [11, 5, 18]
    tables = []
    for n in lens:
        table = BlockTable(allocator)
        table.grow(n)
        tables.append(table)
    width = max(lens)
    mapping = torch.zeros(len(lens), width, dtype=torch.long)
    for i, (table, n) in enumerate(zip(tables, lens)):
        for p in range(n):
            mapping[i, p] = table.slot(p)
    num_slots = 32 * block_size
    k_pool, v_pool = _random_pools(num_slots, n_kv=1, d=8, seed=44, device="cuda")
    q = torch.randn(len(lens), 2, 1, 8, device="cuda")
    mapping = mapping.to("cuda")
    context_lens = torch.tensor(lens, device="cuda")

    got = paged_attention_split_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep=2, block=8, splits=3
    )
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep=2)
    assert torch.allclose(got, oracle, atol=1e-3)
