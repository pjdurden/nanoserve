"""Day 23 tests: the real `triton.jit` paged-attention kernel, and its dispatcher.

Day 22 wrote the fused paged attention as a grid of tlsim programs: one program per
query position, walking the history a tile of keys at a time, folding each tile into
a flash-attention online softmax. That loop is pinned to the Day-18 oracle and it is
the loop the GPU runs. Today it becomes an actual `@triton.jit` kernel, streaming
K/V tiles out of HBM through the block table, plus a `paged_attention` dispatcher
that picks the Triton path on CUDA and the Day-22 CPU model everywhere else.

A kernel needs a GPU, and this box has neither one nor Triton installed, so the
tests split the way the module does. The *host* half is ordinary Python and gets
ordinary tests: the geometry the kernel addresses with (`check_paged_inputs`), the
launch grid, the power-of-two tile rounding Triton's `tl.arange` demands, and the
backend choice. Those run everywhere and are where the addressing bugs actually
live. The *device* half, the jitted body itself, is gated behind
`requires_triton_gpu` and holds the kernel to `paged_attention_reference` on the
same shapes Day 22 used, skipping cleanly without a GPU the way the weights tests
skip without ./weights.

The split is the honest one. Nothing here claims the jitted body is verified on a
box that cannot run it; what is claimed is that everything the host computes and
hands the kernel is checked, and that the dispatcher never silently routes to a
backend that is not there.
"""

from __future__ import annotations

import pytest
import torch
from reference import requires_triton_gpu

from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import paged_attention_reference
from nanoserve.kernels.triton_paged_attention import (
    check_paged_inputs,
    has_triton,
    launch_grid,
    next_power_of_2,
    paged_attention,
    paged_attention_triton,
    select_backend,
)


def _random_pools(num_slots, n_kv, d, seed, device="cpu"):
    torch.manual_seed(seed)
    k_pool = torch.randn(num_slots, n_kv, d, device=device)
    v_pool = torch.randn(num_slots, n_kv, d, device=device)
    return k_pool, v_pool


# --- the tile size Triton demands -------------------------------------------


def test_next_power_of_2_leaves_powers_alone_and_rounds_the_rest_up():
    """`tl.arange` only takes power-of-two lengths, so every tile is rounded up.

    A head_dim of 64 or a tile of 32 is already legal and must pass through
    untouched; a head_dim of 96 or a tile of 13 has to become the next power of two
    and let the mask discard the lanes past the real extent. Rounding *up* is the
    only safe direction: rounding down would silently drop channels or keys.
    """
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        assert next_power_of_2(n) == n
    assert next_power_of_2(3) == 4
    assert next_power_of_2(13) == 16
    assert next_power_of_2(96) == 128
    assert next_power_of_2(129) == 256


def test_launch_grid_is_one_program_per_query_position_and_head():
    """The grid is the parallelism: every (position, head) pair is an independent softmax.

    In the Day-22 model one program owned one query position and looped its heads.
    On hardware the heads are free parallelism, so the grid gains an axis: program
    (i, h) owns query position i of query head h. Nothing is shared between them,
    which is why the grid can be exactly the product.
    """
    assert launch_grid(seq_q=1, n_q=8) == (1, 8)
    assert launch_grid(seq_q=7, n_q=32) == (7, 32)


# --- the geometry the kernel addresses with ---------------------------------


def test_check_paged_inputs_returns_the_geometry_the_kernel_addresses_with():
    """The host computes every integer the kernel turns into a pointer.

    `channels` (n_kv * head_dim) is the stride from one slot to the next in the flat
    pool, and `past` is how many tokens precede this step's queries, which is what
    makes query i's causal extent `past + i + 1`. Getting either wrong reads the
    wrong memory, so both are derived once, here, and tested here rather than
    recomputed inside a kernel nobody can single step.
    """
    q = torch.randn(1, 8, 3, 4)  # 8 query heads, 3 new queries, head_dim 4
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=2, d=4, seed=0)
    slot_mapping = torch.arange(9, dtype=torch.long)  # 9-token history

    geom = check_paged_inputs(q, k_pool, v_pool, slot_mapping, n_rep=4)
    assert geom.n_q == 8
    assert geom.seq_q == 3
    assert geom.head_dim == 4
    assert geom.n_kv == 2
    assert geom.num_slots == 16
    assert geom.channels == 8  # n_kv * head_dim: one slot's worth of elements
    assert geom.seq_total == 9
    assert geom.past == 6  # 9 total minus the 3 queries arriving this step


def test_check_paged_inputs_rejects_a_batch():
    """One sequence only, the same scope as the reference and `PagedKVCache`."""
    q = torch.randn(2, 8, 1, 4)
    k_pool, v_pool = _random_pools(num_slots=8, n_kv=2, d=4, seed=1)
    with pytest.raises(ValueError, match="batch"):
        check_paged_inputs(q, k_pool, v_pool, torch.arange(5), n_rep=4)


def test_check_paged_inputs_rejects_mismatched_k_and_v_pools():
    """K and V share a slot: a token's key and value live at the same flat index.

    The kernel forms one offset and loads both pools with it. Pools of different
    shapes would make that single offset mean two different things, so the shapes
    are checked once on the host instead of producing garbage on the device.
    """
    q = torch.randn(1, 8, 1, 4)
    k_pool = torch.randn(16, 2, 4)
    v_pool = torch.randn(16, 4, 4)  # different KV head count
    with pytest.raises(ValueError, match="same shape"):
        check_paged_inputs(q, k_pool, v_pool, torch.arange(5), n_rep=4)


def test_check_paged_inputs_rejects_a_history_shorter_than_the_queries():
    """`past` must not go negative: the new queries are part of the history.

    `slot_mapping` covers the whole sequence including the tokens generated this
    step, so `seq_total >= seq_q` always. A caller who passed only the *old* slots
    would compute a negative `past`, and every query's causal extent would be wrong
    by exactly the number of new tokens. Catch it at the boundary.
    """
    q = torch.randn(1, 8, 5, 4)  # 5 new queries
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=2, d=4, seed=2)
    slot_mapping = torch.arange(3, dtype=torch.long)  # only 3 slots total
    with pytest.raises(ValueError, match="shorter"):
        check_paged_inputs(q, k_pool, v_pool, slot_mapping, n_rep=4)


def test_check_paged_inputs_rejects_an_n_rep_that_does_not_tile_the_heads():
    """GQA's `head // n_rep` is only a valid map when n_q == n_kv * n_rep.

    The kernel sends query head h to KV head `h // n_rep` and never checks the
    result is in range. If `n_rep` is wrong the high heads address past the end of a
    slot and read the *next token's* K, which is a plausible-looking number and a
    silently wrong answer. The invariant is cheap to state, so state it.
    """
    q = torch.randn(1, 8, 1, 4)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=3, d=4, seed=3)  # 3 KV heads
    with pytest.raises(ValueError, match="n_rep"):
        check_paged_inputs(q, k_pool, v_pool, torch.arange(5), n_rep=4)  # 3*4 != 8


def test_check_paged_inputs_rejects_a_head_dim_that_disagrees_with_the_pool():
    """The query's head_dim is the pool's head_dim; they index the same channels."""
    q = torch.randn(1, 8, 1, 6)  # head_dim 6
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=2, d=4, seed=4)  # head_dim 4
    with pytest.raises(ValueError, match="head_dim"):
        check_paged_inputs(q, k_pool, v_pool, torch.arange(5), n_rep=4)


# --- the dispatcher ---------------------------------------------------------


def test_select_backend_falls_back_to_the_cpu_model_without_a_gpu():
    """No CUDA tensor, no Triton: a CPU device always routes to the Day-22 model.

    The dispatcher must never claim a backend it cannot run. On a CPU tensor the
    answer is "tlsim" regardless of whether Triton happens to be importable, because
    a jitted kernel cannot touch host memory.
    """
    assert select_backend(torch.device("cpu")) == "tlsim"


def test_select_backend_picks_triton_only_when_triton_is_importable():
    """On CUDA the choice is exactly `has_triton()`, so a missing package degrades.

    This box has no Triton, so the branch resolves to "tlsim" and the test pins the
    fallback. On a GPU box with Triton installed it resolves to "triton" and pins
    the fast path. Either way the assertion is the same expression the dispatcher
    uses, which is the point: the backend is a function of what is installed, not a
    hope.
    """
    expected = "triton" if has_triton() else "tlsim"
    assert select_backend(torch.device("cuda")) == expected


def test_paged_attention_on_cpu_matches_the_reference_on_a_decode_step():
    """The dispatcher is a real attention, not just a router: it agrees with the oracle.

    Routed to the CPU model, `paged_attention` must return what
    `paged_attention_reference` returns, to a few ulps (the online softmax
    reassociates the exponent sums). This is the end-to-end assertion that the
    public entry point, whatever backend it lands on, computes paged attention.
    """
    n_q, n_kv, d, n_rep = 8, 2, 4, 4
    num_slots, seq_total = 16, 7
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=5)
    q = torch.randn(1, n_q, 1, d)
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    out = paged_attention(q, k_pool, v_pool, slot_mapping, n_rep)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert out.shape == ref.shape == (1, n_q, 1, d)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_paged_attention_on_cpu_matches_the_reference_over_scattered_blocks():
    """A prefill over genuinely fragmented physical blocks, through the entry point.

    A warmed-then-freed allocator hands the sequence non-contiguous blocks, the
    layout the live cache produces. The dispatcher reads them through the block
    table and reproduces the reference, with an explicit `scale` to pin that the
    keyword survives the hop into the backend rather than being re-defaulted.
    """
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    warm = BlockTable(alloc)
    warm.append(8)
    warm.free()  # hand the blocks back so the next table reuses them LIFO
    table = BlockTable(alloc)
    seq_total = 10
    table.append(seq_total)
    assert table.block_ids != sorted(table.block_ids)  # genuinely scattered

    slot_mapping = torch.tensor([table.slot(p) for p in range(seq_total)], dtype=torch.long)
    num_slots = alloc.num_blocks * alloc.block_size
    n_q, n_kv, d, n_rep = 8, 2, 4, 4
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=6)
    q = torch.randn(1, n_q, seq_total, d)  # prefill: every query, causal
    scale = 0.3

    out = paged_attention(q, k_pool, v_pool, slot_mapping, n_rep, scale=scale)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep, scale=scale)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_paged_attention_triton_refuses_to_run_on_cpu_tensors():
    """A jitted kernel cannot address host memory, and saying so beats a segfault.

    The dispatcher never calls it this way, but the direct entry point is public and
    someone will. The failure has to be a clear Python error at the boundary, not a
    crash inside a launch.
    """
    q = torch.randn(1, 8, 1, 4)
    k_pool, v_pool = _random_pools(num_slots=16, n_kv=2, d=4, seed=7)
    slot_mapping = torch.arange(5, dtype=torch.long)
    with pytest.raises((RuntimeError, ValueError), match="cuda|CUDA|Triton|triton"):
        paged_attention_triton(q, k_pool, v_pool, slot_mapping, n_rep=4)


# --- the jitted body itself (needs a GPU; skips cleanly without one) ---------


@requires_triton_gpu
def test_triton_kernel_matches_the_reference_on_a_decode_step():
    """`seq_q == 1`: one query streams the whole history out of HBM in tiles.

    The gated twin of the Day-22 CPU test. The kernel's grid is (1, n_q): one
    program per head, each folding the history into its own online softmax. Loose
    tolerance because streaming reassociates the exponent sums, exactly as on the
    CPU model.
    """
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    num_slots, seq_total = 64, 37
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=8, device="cuda")
    q = torch.randn(1, n_q, 1, d, device="cuda")
    slot_mapping = torch.randperm(num_slots, device="cuda")[:seq_total]

    out = paged_attention_triton(q, k_pool, v_pool, slot_mapping, n_rep, block=16)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert out.shape == ref.shape == (1, n_q, 1, d)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_triton_kernel_matches_the_reference_on_a_partial_continuation():
    """`past > 0` and `seq_q > 1`: the rectangular causal band, and an explicit scale.

    Query i sees a strictly wider prefix than query i-1, the general case the decode
    step and the full prefill bracket. The kernel derives each program's extent from
    `past + i + 1`, so this is the test that a wrong `past` cannot hide behind.
    """
    n_q, n_kv, d, n_rep = 6, 3, 32, 2
    num_slots, seq_total, seq_q = 64, 21, 5
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=9, device="cuda")
    q = torch.randn(1, n_q, seq_q, d, device="cuda")
    slot_mapping = torch.randperm(num_slots, device="cuda")[:seq_total]
    scale = 0.3

    out = paged_attention_triton(q, k_pool, v_pool, slot_mapping, n_rep, scale=scale, block=8)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep, scale=scale)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_triton_kernel_is_transparent_to_the_tile_size():
    """The tile is a SRAM knob: every legal BLOCK_N returns the same attention.

    The GPU twin of Day 22's block-size sweep. If the answer moved with the tile,
    the online rescale would be wrong. Non-power-of-two tiles are rounded up by the
    host and their extra lanes masked off, so 13 must agree with 16.
    """
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    num_slots, seq_total = 64, 29  # prime, so most tiles leave a ragged tail
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=10, device="cuda")
    q = torch.randn(1, n_q, 1, d, device="cuda")
    slot_mapping = torch.randperm(num_slots, device="cuda")[:seq_total]

    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    for block in (1, 2, 8, 13, 16, 32, 64):
        out = paged_attention_triton(q, k_pool, v_pool, slot_mapping, n_rep, block=block)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), f"block={block}"


@requires_triton_gpu
def test_triton_kernel_uses_the_gqa_head_mapping_from_the_config():
    """Each program's `h // n_rep` must land on its own KV head, over scattered slots.

    Driven from a real `ModelConfig`, with a head_dim that is not a power of two so
    the `BLOCK_D` rounding and its channel mask are exercised on the device. A wrong
    head map passes every tile mechanic and still corrupts half the heads.
    """
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=192,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=48,  # not a power of two: BLOCK_D rounds to 64 and masks the tail
    )
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    num_slots, seq_total = 64, 23
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=11, device="cuda")
    q = torch.randn(1, n_q, 1, d, device="cuda")
    slot_mapping = torch.randperm(num_slots, device="cuda")[:seq_total]

    out = paged_attention_triton(q, k_pool, v_pool, slot_mapping, n_rep, block=16)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)


@requires_triton_gpu
def test_paged_attention_dispatches_to_triton_on_cuda():
    """The entry point routes a CUDA tensor to the kernel and still matches the oracle."""
    n_q, n_kv, d, n_rep = 8, 2, 64, 4
    num_slots, seq_total = 64, 19
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=12, device="cuda")
    q = torch.randn(1, n_q, 1, d, device="cuda")
    slot_mapping = torch.randperm(num_slots, device="cuda")[:seq_total]

    assert select_backend(q.device) == "triton"
    out = paged_attention(q, k_pool, v_pool, slot_mapping, n_rep)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert out.is_cuda
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3)
