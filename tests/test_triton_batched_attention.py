"""Day 62 tests: the batched decode read becomes a `triton.jit` kernel.

Day 59 wrote `paged_attention_batched_kernel`, the streamed decode read as a grid of
tlsim programs: one program per `(row, query head)`, each walking its own row's
`cdiv(ctx, block)` tiles and folding them into an online softmax. Day 60 wired it to
a flag, Day 61 spent the memory it frees. All three days ended on the same caveat:
the loop is Python, so nothing arrived faster. Today it becomes the kernel.

The file splits exactly the way Day 23's did, and for the same reason: this box has
no GPU and no Triton, so the half that can be checked here is checked here and the
half that cannot is gated and says so.

The *host* half is where the addressing bugs live, and the batched kernel has three
integers Day 23's did not. `stride_map` is the mapping's row stride, which under Day
61's streamed bucket set is `max_model_len` on every step and is the only place that
number appears at all. `ctx` is a loop bound loaded from device memory rather than
computed from a program id, which is what makes the walk ragged and what makes it
survive CUDA-graph capture. And `slot_index_dtype` picks the width of the address
arithmetic, because a 32-bit slot times a 32-bit stride wraps silently into a
finite, plausible key.

The *launch* half is `launch_work`, and it is not a kernel test at all: it is the
arithmetic that says how much of Day 59's `ragged_saving` a real grid can collect.
Tiles walked is a sum over programs; wall clock is a max over the resident ones.
Those are different numbers and the gap between them is the day's honest caveat.

The *device* half is gated behind `requires_triton_gpu` and holds the jitted body to
`paged_attention_batched_reference`, the same oracle Day 59 used, on the same toy
pools. Nothing here claims the body is verified on a box that cannot run it.
"""

from __future__ import annotations

import pytest
import torch
from reference import requires_triton_gpu

from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.kernels.triton_batched_attention import (
    INT32_MAX,
    BatchedGeometry,
    batched_launch_grid,
    check_batched_inputs,
    launch_work,
    paged_attention_batched,
    paged_attention_batched_triton,
    select_backend,
    slot_index_dtype,
)


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


def _geometry(num_slots, channels):
    """A geometry with only the two fields the address width depends on set."""
    return BatchedGeometry(
        batch=1,
        n_q=1,
        n_kv=1,
        head_dim=channels,
        num_slots=num_slots,
        channels=channels,
        max_ctx=1,
        stride_q_row=channels,
    )


# --- the geometry the kernel addresses with ---------------------------------


def test_check_batched_inputs_returns_the_geometry_the_kernel_addresses_with():
    """Every integer the kernel turns into a pointer, derived once by ordinary Python.

    `channels` is the stride from one pool slot to the next, `stride_q_row` is the
    stride from one row's queries to the next, and `max_ctx` is the mapping's row
    stride. A jitted body has no shapes, only these, so they are computed where they
    can be single stepped and tested where a failure is a failed assertion rather
    than a plausible number.
    """
    q = torch.randn(3, 8, 1, 4)  # 3 rows, 8 query heads, head_dim 4
    k_pool, v_pool = _random_pools(num_slots=32, n_kv=2, d=4, seed=0)
    slot_mapping = _mapping([5, 9, 2], width=16, num_slots=32, seed=0)
    context_lens = torch.tensor([5, 9, 2])

    geom = check_batched_inputs(q, k_pool, v_pool, slot_mapping, context_lens, n_rep=4)
    assert geom.batch == 3
    assert geom.n_q == 8
    assert geom.n_kv == 2
    assert geom.head_dim == 4
    assert geom.num_slots == 32
    assert geom.channels == 8  # n_kv * head_dim: one slot's worth of elements
    assert geom.max_ctx == 16  # the mapping's row stride, not the longest row
    assert geom.stride_q_row == 32  # n_q * head_dim: one row's worth of queries
    assert geom.pool_elements == 256  # num_slots * channels, the flat pool's length


def test_the_mapping_row_stride_is_the_width_and_never_the_longest_row():
    """Day 61's bucket set hands the read `max_model_len` on every step, and the
    kernel must address with that and not with what the batch happens to hold. The
    width is the only place the rounded-up number appears: it is an address
    multiplier, never a loop bound and never an allocation, which is the whole reason
    a streamed capture list can collapse to the row axis."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=64, n_kv=1, d=8, seed=1)
    slot_mapping = _mapping([3, 4], width=2048, num_slots=64, seed=1)

    geom = check_batched_inputs(q, k_pool, v_pool, slot_mapping, torch.tensor([3, 4]), n_rep=4)
    assert geom.max_ctx == 2048


def test_check_batched_inputs_rejects_a_multi_token_step():
    """The batched read is the decode read: exactly one new token per row."""
    q = torch.randn(2, 4, 3, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=2)
    with pytest.raises(ValueError, match="one new token"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 2), torch.tensor([4, 4]), n_rep=4
        )


def test_check_batched_inputs_rejects_mismatched_pools():
    """A token's key and value share a slot, so the two pools share a shape."""
    q = torch.randn(2, 4, 1, 8)
    k_pool = torch.randn(16, 1, 8)
    v_pool = torch.randn(8, 1, 8)
    with pytest.raises(ValueError, match="same shape"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4], 8, 8, 3), torch.tensor([4, 4]), n_rep=4
        )


def test_check_batched_inputs_rejects_a_one_dimensional_mapping():
    """`[batch, max_ctx]`: the single-sequence read's mapping is a different kernel."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=4)
    with pytest.raises(ValueError, match="batch, max_ctx"):
        check_batched_inputs(
            q, k_pool, v_pool, torch.arange(8), torch.tensor([4, 4]), n_rep=4
        )


def test_check_batched_inputs_rejects_a_mapping_and_lengths_that_disagree_with_the_batch():
    """One table and one length per sequence, or some program addresses another row."""
    q = torch.randn(3, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=5)
    with pytest.raises(ValueError, match="one row each"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 5), torch.tensor([4, 4, 4]), n_rep=4
        )
    with pytest.raises(ValueError, match="one row each"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4, 4], 8, 16, 5), torch.tensor([4, 4]), n_rep=4
        )


def test_check_batched_inputs_rejects_a_head_dim_the_pool_does_not_hold():
    """`channels` is derived from the pool's head_dim and the query's must match it.

    The oracle would die in a matmul and say so in torch's words. The kernel would
    not: `slot * channels + kv_head * head_dim` is arithmetic on two different
    numbers that both look fine, and the result is a real key belonging to another
    head. Stated here because here is the last place shapes still exist.
    """
    q = torch.randn(2, 4, 1, 16)  # head_dim 16
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=6)  # pool holds 8
    with pytest.raises(ValueError, match="head_dim"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 6), torch.tensor([4, 4]), n_rep=4
        )


def test_check_batched_inputs_rejects_an_n_rep_that_does_not_tile_the_query_heads():
    """`h // n_rep` must land inside the compact KV heads for every program.

    A query head that maps past `n_kv` reads the next *token's* channels, because
    the pool is flat and there is nothing at that offset to object. Finite, plausible,
    wrong, and only the host can see it.
    """
    q = torch.randn(2, 8, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=3, d=8, seed=7)
    with pytest.raises(ValueError, match="n_rep"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 7), torch.tensor([4, 4]), n_rep=4
        )


def test_check_batched_inputs_rejects_an_empty_row():
    """A query with no visible key softmaxes over nothing, and here it walks no tiles.

    Worse than the oracle's 0/0: the kernel's loop simply does not run, so `denom`
    is zero and `acc / denom` is a NaN produced by a program that never faulted.
    """
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=8)
    with pytest.raises(ValueError, match="at least 1"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 1], 8, 16, 8), torch.tensor([4, 0]), n_rep=4
        )


def test_check_batched_inputs_rejects_a_context_longer_than_the_mapping():
    """A row cannot claim more history than its table addresses."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=9)
    with pytest.raises(ValueError, match="more history"):
        check_batched_inputs(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 9), torch.tensor([4, 12]), n_rep=4
        )


def test_check_batched_inputs_trusts_the_bounds_it_is_handed():
    """Day 49's pair, unchanged: two Python ints instead of two device readbacks."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=10)
    mapping = _mapping([4, 6], 8, 16, 10)
    geom = check_batched_inputs(
        q, k_pool, v_pool, mapping, torch.tensor([4, 6]), n_rep=4, context_bounds=(4, 6)
    )
    assert geom.max_ctx == 8
    with pytest.raises(ValueError, match="more history"):
        check_batched_inputs(
            q, k_pool, v_pool, mapping, torch.tensor([4, 6]), n_rep=4, context_bounds=(4, 99)
        )


def test_check_batched_inputs_refuses_both_validated_and_bounds():
    """Day 50's refusal: only one of the two can be the reason the check is skipped."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=11)
    with pytest.raises(ValueError, match="validated"):
        check_batched_inputs(
            q,
            k_pool,
            v_pool,
            _mapping([4, 4], 8, 16, 11),
            torch.tensor([4, 4]),
            n_rep=4,
            context_bounds=(4, 4),
            validated=True,
        )


def test_check_batched_inputs_skips_the_length_check_when_the_caller_validated():
    """`validated=True` is a claim, and taking it means not reading the tensor.

    The shape checks stay, because they are free. Only the two that would cost a
    readback are dropped, which is why a length the guard would have refused gets
    through here: that is the contract, not a hole in it.
    """
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=12)
    geom = check_batched_inputs(
        q,
        k_pool,
        v_pool,
        _mapping([4, 4], 8, 16, 12),
        torch.tensor([4, 99]),
        n_rep=4,
        validated=True,
    )
    assert geom.batch == 2


# --- the launch grid and the width of its addresses -------------------------


def test_batched_launch_grid_is_one_program_per_row_and_query_head():
    """The grid vLLM's paged kernel launches, and the reason is the accumulator.

    One program keeps a running max, a denominator and one `d`-wide weighted-V sum,
    all registers. A row's heads share only the slot lookup, so giving one program
    all of them multiplies the state for nothing.
    """
    assert batched_launch_grid(batch=1, n_q=8) == (1, 8)
    assert batched_launch_grid(batch=256, n_q=32) == (256, 32)


def test_slot_index_dtype_stays_narrow_for_a_pool_whose_addresses_fit():
    """int32 is the cheaper address: fewer registers per lane, and every real toy
    pool is nowhere near the ceiling."""
    assert slot_index_dtype(_geometry(num_slots=65536, channels=1024)) is torch.int32


def test_slot_index_dtype_widens_exactly_where_the_multiply_would_wrap():
    """`slot * channels` is the whole risk, so the boundary is `num_slots * channels`.

    The largest address the kernel forms is `pool_elements - 1`. At exactly `2**31`
    elements that is `INT32_MAX` and still fits; one element more and an int32
    multiply wraps to a negative offset, which a masked load will happily follow into
    whatever the allocator put in front of the pool. It does not fault and it does
    not warn: it returns a key.
    """
    assert slot_index_dtype(_geometry(num_slots=2**21, channels=1024)) is torch.int32
    assert slot_index_dtype(_geometry(num_slots=2**21 + 1, channels=1024)) is torch.int64
    assert _geometry(num_slots=2**21, channels=1024).pool_elements - 1 == INT32_MAX


def test_a_geometry_knows_whether_its_addresses_fit():
    """The predicate is on the geometry, so a bench or a planner can ask it too."""
    assert _geometry(num_slots=1024, channels=64).addresses_fit_int32 is True
    assert _geometry(num_slots=2**22, channels=1024).addresses_fit_int32 is False


# --- what a real grid can collect from a ragged read ------------------------


def test_launch_work_counts_one_program_per_row_and_head():
    """Every head of a row walks the same tiles, so the grid's total is `n_q` times
    the row sum. The program count is the number `launch_grid` returns, multiplied
    out, and it is what decides whether the machine is oversubscribed."""
    work = launch_work([100, 100, 100], n_q=8, block=50)
    assert work.programs == 24
    assert work.tiles == 24 * 2  # two tiles a row, eight heads, three rows


def test_launch_work_tail_is_the_longest_row_and_not_the_mean():
    """The tail is what a resident grid waits for.

    Three rows of 8 and one of 800 average 206 tokens, and the wave still takes 800
    tokens' worth of tiles, because the long program is running while the short ones
    have retired. Summing tiles hides that; this is the number that does not.
    """
    work = launch_work([8, 8, 8, 800], n_q=4, block=8)
    assert work.tail_tiles == 100  # cdiv(800, 8)
    assert work.tiles == 4 * (1 + 1 + 1 + 100)
    assert work.mean_tiles == pytest.approx(103 / 4)
    assert work.imbalance == pytest.approx(100 / (103 / 4))


def test_the_work_saving_is_a_sum_and_the_wave_saving_is_a_max():
    """Day 59's `ragged_saving` is tiles, and tiles are not always seconds.

    Both numbers are real and they answer different questions. `work_saving` is what
    an oversubscribed grid collects, because there the machine is a queue and skipped
    tiles are skipped waves. `wave_saving` is what a grid small enough to be fully
    resident collects, because there the clock is the slowest program. The truth sits
    between them and nothing on this box can say where.
    """
    work = launch_work([8, 8, 8, 800], n_q=4, block=8, context_width=800)
    assert work.rectangle_tiles == 16 * 100
    assert work.work_saving == pytest.approx(1600 / 412)
    assert work.wave_saving == pytest.approx(1.0)


def test_on_an_unbucketed_mapping_a_resident_grid_saves_no_time_at_all():
    """The honest floor, and it is a floor of exactly 1.0x.

    An unbucketed mapping is as wide as its longest row, so the longest program walks
    the whole width and a fully resident grid finishes no sooner than the rectangle
    would have. Everything the read saves there is memory. That is not a
    disappointment, it is the same shape as Day 59's "a uniform batch has nothing to
    skip", one level out.
    """
    for lens in ([4, 4, 4], [1, 50, 900], [7, 7, 7, 4000]):
        assert launch_work(lens, n_q=8, block=32).wave_saving == pytest.approx(1.0)


def test_a_bucketed_width_is_where_the_wave_saving_actually_comes_from():
    """Day 61's streamed set rounds every context to `max_model_len`, and that is the
    width a rectangle read would have been handed. So against *that* mapping the
    longest program still skips almost everything, and the saving a resident grid
    collects is real. The two days compose: the set that made the capture list short
    is also what puts the latency back into the ragged read."""
    work = launch_work([12, 40, 61], n_q=8, block=32, context_width=8192)
    assert work.tail_tiles == 2  # cdiv(61, 32)
    assert work.wave_saving == pytest.approx(256 / 2)


def test_the_work_saving_factors_into_the_wave_saving_and_the_imbalance():
    """`work == wave * imbalance`, exactly, on every batch and every width.

    Both sides are `cdiv(width, block) / mean_tiles` once the `tail` cancels, so this
    is an identity and not a coincidence, and it is the shape of the whole day. The
    tiles a ragged read saves split into a part a fully resident grid collects and a
    part only an oversubscribed one does, and the second factor is precisely the
    length spread that made the read worth writing.
    """
    for lens, width in (
        ([8, 8, 8, 800], None),
        ([8, 8, 8, 800], 8192),
        ([12, 40, 61], 8192),
        ([64, 64, 64, 64], 4096),
        ([1, 2, 3, 5, 8, 13, 21], None),
    ):
        work = launch_work(lens, n_q=8, block=32, context_width=width)
        assert work.work_saving == pytest.approx(work.wave_saving * work.imbalance)


def test_on_a_tight_mapping_the_whole_saving_is_the_imbalance():
    """The corollary, and it is the uncomfortable one.

    A mapping as wide as its own longest row makes `wave_saving` exactly 1.0, so the
    identity collapses to `work == imbalance`: everything the ragged walk saves in
    tiles is exactly the tail it still has to wait for. That is why Day 61's bucket
    set matters to a kernel and not only to a capture list. Rounding the width up to
    `max_model_len` is what puts a `wave_saving` back into the product.
    """
    work = launch_work([8, 8, 8, 800], n_q=4, block=8)
    assert work.wave_saving == pytest.approx(1.0)
    assert work.work_saving == pytest.approx(work.imbalance)


def test_a_uniform_batch_is_perfectly_balanced_and_says_so():
    """1.0x imbalance is a measurement. Every program walks the same tiles, so the
    grid retires in one wave and there is nothing for a split to recover."""
    work = launch_work([64, 64, 64, 64], n_q=2, block=16)
    assert work.tail_tiles == 4
    assert work.imbalance == pytest.approx(1.0)


def test_launch_work_refuses_a_batch_it_cannot_describe():
    with pytest.raises(ValueError, match="at least one row"):
        launch_work([], n_q=4, block=8)
    with pytest.raises(ValueError, match="at least 1"):
        launch_work([4, 0], n_q=4, block=8)
    with pytest.raises(ValueError, match="at least one key"):
        launch_work([4, 4], n_q=4, block=0)
    with pytest.raises(ValueError, match="at least one query head"):
        launch_work([4, 4], n_q=0, block=8)
    with pytest.raises(ValueError, match="width"):
        launch_work([4, 40], n_q=4, block=8, context_width=8)


def test_launch_work_renders_one_line_with_both_savings():
    line = launch_work([12, 40, 61], n_q=8, block=32, context_width=8192).render()
    assert "24 programs" in line and "128.00x" in line


# --- the dispatcher ---------------------------------------------------------


def test_the_batched_read_falls_back_to_the_cpu_model_on_a_host_tensor():
    """A jitted kernel cannot address host memory, whatever is installed."""
    assert select_backend(torch.device("cpu")) == "tlsim"


def test_paged_attention_batched_on_cpu_matches_the_oracle():
    """The entry point is a real attention and not just a router.

    Routed to Day 59's tlsim loop, it must return what
    `paged_attention_batched_reference` returns to a few ulps, which is the same
    assertion Day 59 made of the loop directly. The point of repeating it through the
    dispatcher is that the dispatcher is now the only thing the engine calls.
    """
    lens = [3, 9, 5]
    n_q, n_kv, d, n_rep = 8, 2, 8, 4
    k_pool, v_pool = _random_pools(num_slots=32, n_kv=n_kv, d=d, seed=20)
    torch.manual_seed(20)
    q = torch.randn(len(lens), n_q, 1, d)
    mapping = _mapping(lens, width=12, num_slots=32, seed=20)
    context_lens = torch.tensor(lens)

    out = paged_attention_batched(q, k_pool, v_pool, mapping, context_lens, n_rep, block=4)
    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep)
    assert out.shape == ref.shape == (len(lens), n_q, 1, d)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_paged_attention_batched_on_cpu_carries_the_scale_and_the_validated_flag():
    """Both keywords have to survive the hop into the backend.

    A re-defaulted `scale` is a wrong softmax that still looks like attention, and a
    dropped `validated` puts Day 50's readback back in the graph under one flag and
    not the other.
    """
    lens = [6, 6]
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=21)
    torch.manual_seed(21)
    q = torch.randn(2, 4, 1, 8)
    mapping = _mapping(lens, width=8, num_slots=16, seed=21)
    context_lens = torch.tensor(lens)

    out = paged_attention_batched(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, scale=0.3, block=3, validated=True
    )
    ref = paged_attention_batched_reference(
        q, k_pool, v_pool, mapping, context_lens, n_rep=4, scale=0.3
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_paged_attention_batched_refuses_what_its_oracle_refuses():
    """A kernel that accepts inputs its oracle rejects cannot be compared to it."""
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=22)
    q = torch.randn(2, 4, 3, 8)  # three new tokens: not a decode step
    with pytest.raises(ValueError, match="one new token"):
        paged_attention_batched(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 22), torch.tensor([4, 4]), n_rep=4
        )


def test_paged_attention_batched_triton_refuses_to_run_on_host_tensors():
    """The direct entry point is public, and someone will call it on a laptop. The
    failure has to be a Python error at the boundary and not a crash in a launch."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=23)
    with pytest.raises((RuntimeError, ValueError), match="cuda|CUDA|Triton|triton"):
        paged_attention_batched_triton(
            q, k_pool, v_pool, _mapping([4, 4], 8, 16, 23), torch.tensor([4, 4]), n_rep=4
        )


def test_paged_attention_batched_refuses_a_tile_of_no_keys():
    """The tile is a performance knob and zero is not a value of it."""
    q = torch.randn(2, 4, 1, 8)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=1, d=8, seed=24)
    with pytest.raises(ValueError, match="at least one key"):
        paged_attention_batched(
            q,
            k_pool,
            v_pool,
            _mapping([4, 4], 8, 16, 24),
            torch.tensor([4, 4]),
            n_rep=4,
            block=0,
        )


# --- the jitted body itself (needs a GPU; skips cleanly without one) ---------


@requires_triton_gpu
def test_triton_batched_kernel_matches_the_oracle_on_a_uniform_batch():
    """Every row the same length: the grid retires in one wave and nothing is ragged.

    The base case, and the one where the streamed read has nothing to skip. If this
    disagrees, the online softmax or the GQA head map is wrong and no amount of
    raggedness will show it more clearly.
    """
    lens = [16, 16, 16, 16]
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    k_pool, v_pool = _random_pools(64, n_kv, d, seed=30, device="cuda")
    q = torch.randn(len(lens), n_q, 1, d, device="cuda")
    mapping = _mapping(lens, width=16, num_slots=64, seed=30, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    out = paged_attention_batched_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep, block=8
    )
    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep)
    assert out.shape == ref.shape == (len(lens), n_q, 1, d)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_triton_batched_kernel_walks_each_row_its_own_length():
    """A ragged batch, which is the whole point: the loop bound is loaded per program.

    Row 0 walks one tile and row 3 walks a hundred, out of one launch, and the answer
    for row 0 must equal the answer a batch of only row 0 would have given. A bound
    computed from the program id instead of from `context_lens` would give every row
    the same walk and still be numerically fine on a uniform batch.
    """
    lens = [3, 40, 7, 61]
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    k_pool, v_pool = _random_pools(128, n_kv, d, seed=31, device="cuda")
    q = torch.randn(len(lens), n_q, 1, d, device="cuda")
    mapping = _mapping(lens, width=64, num_slots=128, seed=31, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    out = paged_attention_batched_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep, block=16
    )
    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_triton_batched_kernel_never_loads_the_padding_past_a_row():
    """Stronger than "masked off": the pad is not read, so it may be poison.

    The oracle indexes the pool with the whole rectangle and then kills the result,
    so its padding has to be a legal slot holding a finite number. The kernel's
    padding is -1 and the slot it would name holds NaN. A finite answer here is the
    proof that the ragged walk is ragged.
    """
    lens = [5, 29]
    n_q, n_kv, d, n_rep = 4, 1, 32, 4
    k_pool, v_pool = _random_pools(64, n_kv, d, seed=32, device="cuda")
    k_pool[-1] = float("nan")
    v_pool[-1] = float("nan")
    q = torch.randn(len(lens), n_q, 1, d, device="cuda")
    mapping = _mapping(lens, width=64, num_slots=63, seed=32, device="cuda", pad=-1)
    context_lens = torch.tensor(lens, device="cuda")

    out = paged_attention_batched_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep, block=8
    )
    assert torch.isfinite(out).all()


@requires_triton_gpu
def test_triton_batched_kernel_is_transparent_to_the_tile_size():
    """The tile is a SRAM knob: every legal value returns the same attention."""
    lens = [9, 29, 17]
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    k_pool, v_pool = _random_pools(64, n_kv, d, seed=33, device="cuda")
    q = torch.randn(len(lens), n_q, 1, d, device="cuda")
    mapping = _mapping(lens, width=32, num_slots=64, seed=33, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep)
    for block in (1, 2, 8, 13, 16, 32):
        out = paged_attention_batched_triton(
            q, k_pool, v_pool, mapping, context_lens, n_rep, block=block
        )
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), f"block={block}"


@requires_triton_gpu
def test_triton_batched_kernel_uses_the_gqa_head_mapping_from_the_config():
    """`h // n_rep` per program, with a head_dim that is not a power of two.

    A wrong head map passes every tile mechanic and corrupts half the heads, and the
    channel mask is what makes a head_dim of 48 legal at all.
    """
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=192,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=48,  # BLOCK_D rounds to 64 and masks the tail
    )
    lens = [11, 23]
    d = cfg.head_dim
    k_pool, v_pool = _random_pools(64, cfg.num_key_value_heads, d, seed=34, device="cuda")
    q = torch.randn(len(lens), cfg.num_attention_heads, 1, d, device="cuda")
    mapping = _mapping(lens, width=32, num_slots=64, seed=34, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    out = paged_attention_batched_triton(
        q, k_pool, v_pool, mapping, context_lens, cfg.num_kv_groups, block=16
    )
    ref = paged_attention_batched_reference(
        q, k_pool, v_pool, mapping, context_lens, cfg.num_kv_groups
    )
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_triton_batched_kernel_reads_genuinely_scattered_blocks():
    """The layout a live allocator produces: rows interleaved in one shared pool."""
    alloc = BlockAllocator(num_blocks=16, block_size=4)
    warm = BlockTable(alloc)
    warm.append(12)
    warm.free()  # hand the blocks back so the next tables reuse them LIFO
    tables = [BlockTable(alloc) for _ in range(3)]
    lens = [5, 11, 7]
    for table, n in zip(tables, lens):
        table.append(n)
    width = max(lens)
    mapping = torch.zeros(len(lens), width, dtype=torch.long)
    for i, (table, n) in enumerate(zip(tables, lens)):
        mapping[i, :n] = torch.tensor([table.slot(p) for p in range(n)])

    num_slots = alloc.num_blocks * alloc.block_size
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=35, device="cuda")
    q = torch.randn(len(lens), n_q, 1, d, device="cuda")
    mapping = mapping.to("cuda")
    context_lens = torch.tensor(lens, device="cuda")

    out = paged_attention_batched_triton(
        q, k_pool, v_pool, mapping, context_lens, n_rep, scale=0.3, block=4
    )
    ref = paged_attention_batched_reference(
        q, k_pool, v_pool, mapping, context_lens, n_rep, scale=0.3
    )
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_paged_attention_batched_dispatches_to_triton_on_cuda():
    """The entry point routes a CUDA tensor to the kernel and still matches the oracle."""
    lens = [4, 19]
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    k_pool, v_pool = _random_pools(64, n_kv, d, seed=36, device="cuda")
    q = torch.randn(len(lens), n_q, 1, d, device="cuda")
    mapping = _mapping(lens, width=32, num_slots=64, seed=36, device="cuda")
    context_lens = torch.tensor(lens, device="cuda")

    assert select_backend(q.device) == "triton"
    out = paged_attention_batched(q, k_pool, v_pool, mapping, context_lens, n_rep)
    ref = paged_attention_batched_reference(q, k_pool, v_pool, mapping, context_lens, n_rep)
    assert out.is_cuda
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)
