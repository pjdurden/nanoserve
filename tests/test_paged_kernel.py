"""Day 22 tests: the fused paged-attention kernel, modeled on the CPU.

Day 18 built `paged_attention_reference`: the correctness oracle for paged
attention, a plain-torch read-then-score that gathers the whole history into a
contiguous `[seq_total, n_kv, d]` buffer and runs one softmax over it. Day 21
rewrote the *read* half of that as a grid of tlsim programs (`paged_gather`), but
it still assembled the full history before scoring, which is the one thing the
real kernel must never do: materializing the contiguous buffer throws away the
point of paging.

`paged_attention_kernel` is the fused version. It folds the Day-21 gather together
with the score and the softmax into a single streaming loop, expressed on the same
tlsim primitives (`launch`, `load`, `store`, `arange`, `cdiv`). One program owns a
query position; it walks the history a `block` of keys at a time, reading each tile
through the block table with a masked `load`, and updates a flash-attention style
online-softmax accumulator (running max, running denominator, running weighted-V
sum). The full `[kv_len, ...]` history is never held: peak state is one tile plus
the per-head accumulators. That is the exact loop next week's `triton.jit` kernel
runs, pinned here to the oracle on a box with no GPU.

The tests demand it agrees with `paged_attention_reference` on a decode step, on a
causal prefill, on a partial continuation (past history plus several new queries),
over physically scattered blocks, and independently of the tile size, since the
block is a performance knob and never a correctness one. The tolerance is loose
where the reference used `torch.equal`, because streaming the softmax reassociates
the exponent sums; equal-to-a-few-ulps, not bit-identical, is the honest bar for a
flash kernel.
"""

from __future__ import annotations

import torch

from nanoserve.cache import BlockAllocator, BlockTable
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import (
    paged_attention_kernel,
    paged_attention_reference,
)


def _random_pools(num_slots, n_kv, d, seed):
    torch.manual_seed(seed)
    k_pool = torch.randn(num_slots, n_kv, d)
    v_pool = torch.randn(num_slots, n_kv, d)
    return k_pool, v_pool


# --- the kernel matches the oracle across the shapes attention takes ---------


def test_kernel_matches_reference_on_a_decode_step():
    """One new query over the whole history: the streaming softmax equals the oracle.

    A decode step is `seq_q == 1`: a single query that attends to every past token.
    The fused kernel walks the history in tiles and folds each into an online
    softmax; the result must match the contiguous read-then-score reference to a
    few ulps (streaming reassociates the exponent sums, so not bit-identical).
    """
    n_q, n_kv, d, n_rep = 8, 2, 4, 4
    num_slots, seq_total = 16, 7
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=0)
    q = torch.randn(1, n_q, 1, d)
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    out = paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, block=4)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert out.shape == ref.shape == (1, n_q, 1, d)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_reference_on_a_causal_prefill():
    """A whole prompt at once: every query sees its own causal prefix, no further.

    Prefill is `seq_q == seq_total`, `past == 0`: query i may see keys 0..i. The
    kernel's per-query `kv_len = past + i + 1` reproduces the reference's
    rectangular causal band, so the two agree row for row.
    """
    n_q, n_kv, d, n_rep = 8, 2, 4, 4
    num_slots, seq_total = 16, 6
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=1)
    q = torch.randn(1, n_q, seq_total, d)
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    out = paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, block=4)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert out.shape == (1, n_q, seq_total, d)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_matches_reference_on_a_partial_continuation():
    """Several new queries over an existing history: the rectangular band, and scale.

    `past > 0` and `seq_q > 1` at once: three new queries appended to a six-token
    history. Query i (absolute position past+i) sees a strictly wider prefix than
    query i-1, the general causal band the two extremes above bracket. Also passes
    an explicit `scale` to pin that it is plumbed through, not hard-coded.
    """
    n_q, n_kv, d, n_rep = 6, 3, 4, 2
    num_slots, seq_total, seq_q = 24, 9, 3
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=2)
    q = torch.randn(1, n_q, seq_q, d)
    slot_mapping = torch.randperm(num_slots)[:seq_total]
    scale = 0.3

    out = paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, scale=scale, block=4)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep, scale=scale)
    assert out.shape == (1, n_q, seq_q, d)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_is_transparent_to_the_tile_size():
    """The block is a performance knob: every tile size gives the same attention.

    Tiles that divide the history, that leave a ragged tail, that are one token,
    and that are larger than the whole history must all return the same output (to
    tolerance) and all match the reference. A result that shifted with the block
    size would mean the online-softmax rescale was wrong, the streaming analogue of
    programs stepping on each other in the Day-21 gather.
    """
    n_q, n_kv, d, n_rep = 8, 2, 4, 4
    num_slots, seq_total = 32, 13  # prime, so most tile sizes leave a tail
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=3)
    q = torch.randn(1, n_q, 1, d)
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    for block in (1, 2, 3, 4, 8, 13, 32):
        out = paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, block=block)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), f"block={block}"


def test_kernel_reads_through_physically_scattered_blocks():
    """The paged point: fragmented physical blocks give the same attention output.

    A warmed-then-freed allocator hands the sequence non-contiguous physical
    blocks, exactly the layout the live cache produces. The kernel streams each
    tile through the block table's slots regardless, so it matches a reference read
    over the same scattered pool. Paging moved where the K/V live, not the answer.
    """
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    warm = BlockTable(alloc)
    warm.append(8)
    warm.free()  # hand two blocks back so the next table reuses them LIFO
    table = BlockTable(alloc)
    seq_total = 10
    table.append(seq_total)
    assert table.block_ids != sorted(table.block_ids)  # genuinely scattered

    slot_mapping = torch.tensor([table.slot(p) for p in range(seq_total)], dtype=torch.long)
    num_slots = alloc.num_blocks * alloc.block_size
    n_q, n_kv, d, n_rep = 8, 2, 4, 4
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=4)
    q = torch.randn(1, n_q, 1, d)

    out = paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, block=4)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_uses_the_gqa_head_mapping_from_the_config():
    """GQA is not decoration: each query head must read its own KV head's history.

    Driving the kernel from a real `ModelConfig` (8 query heads over 2 KV heads,
    repeat 4) and matching the reference proves the query-head to KV-head map
    (`head // n_rep`) is right. A single wrong head would pass the tile mechanics
    and still corrupt half the heads' softmax, so pinning it to the oracle is the
    check that catches it.
    """
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    num_slots, seq_total = 20, 11
    k_pool, v_pool = _random_pools(num_slots, n_kv, d, seed=5)
    q = torch.randn(1, n_q, 1, d)
    slot_mapping = torch.randperm(num_slots)[:seq_total]

    out = paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, block=4)
    ref = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_kernel_rejects_a_batch():
    """One sequence only, the same scope as the reference and `PagedKVCache`."""
    q = torch.randn(2, 8, 1, 4)  # batch of two
    k_pool = torch.randn(8, 2, 4)
    v_pool = torch.randn(8, 2, 4)
    slot_mapping = torch.arange(5, dtype=torch.long)
    try:
        paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep=4)
        raise AssertionError("expected a ValueError for batch > 1")
    except ValueError:
        pass
