"""Day 18 tests: the torch reference for paged attention (Week 6 opens).

Week 6 replaces the deliberately slow paged read (gather the whole history back
into a contiguous tensor every step, then run a normal SDPA over it) with a
kernel that attends *directly* over the scattered blocks, never materializing the
contiguous history. Before any kernel is trusted it needs a reference to be
checked against, and that reference is `paged_attention_reference`: a plain torch
function that takes the query, the layer's flat K/V pools, and the per-position
slot mapping the block table produces, and computes the attention output by
reading each past token's K/V through its slot.

The whole point of these tests is the same "paging changes nothing observable"
contract the cache tests pin, one level deeper: the output of the paged reference
must equal a straight contiguous SDPA over the identical K/V, even when the
physical blocks backing the history are scattered anywhere in the pool. Paging
moves *where* the K/V live, not *what* attention computes.

Everything here is pure torch on tiny tensors, so it runs on any box (no GPU, no
Triton). When the Triton kernel lands it gets held to this exact same output.
"""

from __future__ import annotations

import math

import torch

from nanoserve.cache import BlockAllocator, BlockTable, NaiveKVCache, PagedKVCache
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import paged_attention_reference
from nanoserve.layers import repeat_kv


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the cache tests use."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _contiguous_attention(
    q: torch.Tensor,
    k_hist: torch.Tensor,
    v_hist: torch.Tensor,
    n_rep: int,
    scale: float,
) -> torch.Tensor:
    """The ground truth: a plain SDPA over the contiguous history, exactly the
    math `gqa_attention` runs after the naive cache hands it the full K/V.

    q:              [1, n_q, seq_q, d] rotated queries for the new tokens.
    k_hist, v_hist: [1, n_kv, seq_total, d] compact (pre-repeat) history.
    Returns [1, n_q, seq_q, d], the attention output before o_proj.
    """
    k = repeat_kv(k_hist, n_rep)
    v = repeat_kv(v_hist, n_rep)
    seq_q = q.shape[2]
    kv_len = k.shape[2]
    past = kv_len - seq_q
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    causal = torch.full((seq_q, kv_len), float("-inf"), dtype=scores.dtype)
    scores = scores + torch.triu(causal, diagonal=past + 1)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(weights, v)


def _pool_from_history(k_hist, v_hist, slot_mapping, num_slots, n_kv, d):
    """Scatter a compact history [1, n_kv, seq, d] into flat pools at `slot_mapping`.

    Returns (k_pool, v_pool) each [num_slots, n_kv, d] with position p's K/V written
    at flat index `slot_mapping[p]`, the same layout `PagedKVCache` uses.
    """
    k_pool = torch.zeros(num_slots, n_kv, d)
    v_pool = torch.zeros(num_slots, n_kv, d)
    k_pool[slot_mapping] = k_hist[0].transpose(0, 1)  # [seq, n_kv, d]
    v_pool[slot_mapping] = v_hist[0].transpose(0, 1)
    return k_pool, v_pool


# --- the reference equals a contiguous SDPA ---------------------------------


def test_reference_matches_contiguous_prefill():
    """A whole-prompt prefill through the paged reference equals a plain SDPA.

    Store a compact history in a pool at the natural slots 0..seq-1, read it back
    through the reference, and demand byte-for-byte agreement with the contiguous
    truth: same gather, same repeat, same causal softmax, so the numbers are
    identical, not merely close.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    seq = 6

    torch.manual_seed(0)
    q = torch.randn(1, n_q, seq, d)
    k_hist = torch.randn(1, n_kv, seq, d)
    v_hist = torch.randn(1, n_kv, seq, d)

    slot_mapping = torch.arange(seq, dtype=torch.long)
    k_pool, v_pool = _pool_from_history(k_hist, v_hist, slot_mapping, seq, n_kv, d)

    out = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    truth = _contiguous_attention(q, k_hist, v_hist, n_rep, d**-0.5)
    assert out.shape == (1, n_q, seq, d)
    assert torch.equal(out, truth)


def test_reference_matches_single_decode_step():
    """One decode query over a stored history equals the contiguous SDPA.

    seq_q is 1 and past is the whole history, so the single query sees every past
    token (its mask row is all-visible), the common decode case the kernel exists
    to serve.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    seq_total = 7

    torch.manual_seed(1)
    q = torch.randn(1, n_q, 1, d)  # a single new token
    k_hist = torch.randn(1, n_kv, seq_total, d)
    v_hist = torch.randn(1, n_kv, seq_total, d)

    slot_mapping = torch.arange(seq_total, dtype=torch.long)
    k_pool, v_pool = _pool_from_history(k_hist, v_hist, slot_mapping, seq_total, n_kv, d)

    out = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    truth = _contiguous_attention(q, k_hist, v_hist, n_rep, d**-0.5)
    assert out.shape == (1, n_q, 1, d)
    assert torch.equal(out, truth)


def test_reference_reads_through_physically_scattered_blocks():
    """The headline: scattered physical blocks give the identical attention output.

    A real block table, fragmented by a free-and-reallocate, places the history's
    tokens on non-contiguous physical blocks. The reference reads each token
    through its slot regardless, so the output still equals the contiguous SDPA
    over the same K/V. Paging moved where the K/V live, not what attention did.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    seq_total = 10

    alloc = BlockAllocator(num_blocks=8, block_size=4)
    # Fragment the pool: take two blocks, hand them back, so the next table reuses
    # them LIFO and lands on non-contiguous physical blocks.
    warm = BlockTable(alloc)
    warm.append(8)
    warm.free()
    table = BlockTable(alloc)
    table.append(seq_total)
    assert table.block_ids != sorted(table.block_ids)  # genuinely scattered

    slot_mapping = torch.tensor(
        [table.slot(p) for p in range(seq_total)], dtype=torch.long
    )
    num_slots = alloc.num_blocks * alloc.block_size

    torch.manual_seed(2)
    q = torch.randn(1, n_q, seq_total, d)
    k_hist = torch.randn(1, n_kv, seq_total, d)
    v_hist = torch.randn(1, n_kv, seq_total, d)
    k_pool, v_pool = _pool_from_history(k_hist, v_hist, slot_mapping, num_slots, n_kv, d)

    out = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    truth = _contiguous_attention(q, k_hist, v_hist, n_rep, d**-0.5)
    assert torch.equal(out, truth)


def test_reference_honors_the_causal_mask():
    """A query may not see the future: mask a token's K/V and the past is unchanged.

    Over a multi-token prefill, the first query attends to position 0 alone and
    the last attends to the whole prefix. So corrupting the K/V of the final
    position must leave every earlier query's output identical (they never saw it)
    while changing the last query's output (it did).
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    seq = 5

    torch.manual_seed(3)
    q = torch.randn(1, n_q, seq, d)
    k_hist = torch.randn(1, n_kv, seq, d)
    v_hist = torch.randn(1, n_kv, seq, d)
    slot_mapping = torch.arange(seq, dtype=torch.long)
    k_pool, v_pool = _pool_from_history(k_hist, v_hist, slot_mapping, seq, n_kv, d)

    base = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)

    # Corrupt the last position's stored K/V; only the last query attends to it.
    k_pool[slot_mapping[-1]] += 100.0
    v_pool[slot_mapping[-1]] += 100.0
    perturbed = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)

    assert torch.equal(base[:, :, :-1], perturbed[:, :, :-1])  # earlier queries intact
    assert not torch.allclose(base[:, :, -1], perturbed[:, :, -1])  # last query moved


def test_reference_default_scale_is_inverse_sqrt_head_dim():
    """No scale argument means the standard 1/sqrt(head_dim), like gqa_attention."""
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups
    seq = 4

    torch.manual_seed(4)
    q = torch.randn(1, n_q, seq, d)
    k_hist = torch.randn(1, n_kv, seq, d)
    v_hist = torch.randn(1, n_kv, seq, d)
    slot_mapping = torch.arange(seq, dtype=torch.long)
    k_pool, v_pool = _pool_from_history(k_hist, v_hist, slot_mapping, seq, n_kv, d)

    default = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    explicit = paged_attention_reference(
        q, k_pool, v_pool, slot_mapping, n_rep, scale=1.0 / math.sqrt(d)
    )
    assert torch.equal(default, explicit)


def test_reference_shares_one_kv_head_across_a_query_group():
    """GQA: query heads in the same group attend against the same K/V head.

    With `n_rep` query heads per KV head, two query heads that fall in the same
    group and carry identical queries must produce identical outputs, because they
    score against the very same repeated key/value. Heads 0 and 1 share KV head 0
    (group size 4 here); giving them the same query pins that the repeat lines up
    the groups the way `repeat_kv` promises.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups  # 4 query heads per KV head
    seq = 4

    torch.manual_seed(5)
    q = torch.randn(1, n_q, seq, d)
    q[:, 1] = q[:, 0]  # heads 0 and 1 are both in group 0; make their queries equal
    k_hist = torch.randn(1, n_kv, seq, d)
    v_hist = torch.randn(1, n_kv, seq, d)
    slot_mapping = torch.arange(seq, dtype=torch.long)
    k_pool, v_pool = _pool_from_history(k_hist, v_hist, slot_mapping, seq, n_kv, d)

    out = paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep)
    assert torch.equal(out[:, 0], out[:, 1])


def test_reference_over_paged_cache_matches_naive_history():
    """End to end: read the real PagedKVCache pools and match the naive history.

    Drive a paged cache and a naive cache with identical K/V across layers and
    steps, then attend over the paged pools via the reference and over the naive
    contiguous history via the plain SDPA. Same output means the reference reads
    exactly what the cache stored, which is the tensor the Triton kernel will read.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups

    alloc = BlockAllocator(num_blocks=8, block_size=4)
    paged = PagedKVCache(cfg, alloc)
    naive = NaiveKVCache(cfg.num_hidden_layers)

    torch.manual_seed(6)
    for new_seq in (5, 1, 1):  # prefill 5, then two decode steps
        for layer in range(cfg.num_hidden_layers):
            k = torch.randn(1, n_kv, new_seq, d)
            v = torch.randn(1, n_kv, new_seq, d)
            paged.append(layer, k, v)
            naive.append(layer, k, v)

    seq_total = paged.seq_len
    slot_mapping = torch.tensor(
        [paged.table.slot(p) for p in range(seq_total)], dtype=torch.long
    )
    layer = 1  # check a non-zero layer to catch a per-layer pool mixup
    q = torch.randn(1, n_q, 1, d)  # a fresh decode query over the whole history

    out = paged_attention_reference(
        q, paged.k_pool[layer], paged.v_pool[layer], slot_mapping, n_rep
    )
    truth = _contiguous_attention(q, naive.k[layer], naive.v[layer], n_rep, d**-0.5)
    assert torch.equal(out, truth)


# --- Day 19: the fused read wired into PagedKVCache -------------------------
#
# Day 18 built `paged_attention_reference` standing alone: hand it a query, the
# flat pools, and a slot mapping and it computes attention over the scattered
# blocks. Day 19 hands the cache the wheel. `PagedKVCache.paged_attention` is the
# fused read: give it this step's rotated Q/K/V and it writes the K/V into its
# block pool (exactly as `append` does) and then attends over the whole history
# through the block table, returning the attention output directly. No contiguous
# history is ever rebuilt for attention to score against, which is the entire
# point of the kernel week: the gather-and-reassemble read of Day 16 is gone from
# the paged path.
#
# The contract is again "paging changes nothing observable": the fused read must
# return exactly what a plain SDPA over the naive contiguous history returns, and
# it must leave the cache in exactly the state a bare `append` would (same slots
# written, same seq_len, same pool bytes). Only what it *returns* differs: the
# attention output instead of the running K/V.


def test_fused_read_matches_naive_gather_across_prefill_and_decode():
    """The headline: the fused paged read equals a contiguous SDPA, every step.

    Drive a paged and a naive cache with identical K/V across a prefill and two
    decode steps, over every layer, the way the model does. At each (step, layer)
    a fresh query attends through the fused read, and the truth is a plain SDPA
    over the K/V the naive cache hands back. `torch.equal`, not `allclose`: the
    fused read scatters and gathers by exact index and runs the identical causal
    fp32 softmax, so the numbers are the same, not merely close.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups

    alloc = BlockAllocator(num_blocks=8, block_size=4)
    paged = PagedKVCache(cfg, alloc)
    naive = NaiveKVCache(cfg.num_hidden_layers)

    torch.manual_seed(7)
    for new_seq in (5, 1, 1):  # prefill 5, then two decode steps
        for layer in range(cfg.num_hidden_layers):
            k = torch.randn(1, n_kv, new_seq, d)
            v = torch.randn(1, n_kv, new_seq, d)
            q = torch.randn(1, n_q, new_seq, d)

            fused = paged.paged_attention(layer, k, v, q, n_rep)
            full_k, full_v = naive.append(layer, k, v)
            truth = _contiguous_attention(q, full_k, full_v, n_rep, d**-0.5)

            assert fused.shape == (1, n_q, new_seq, d)
            assert torch.equal(fused, truth)


def test_fused_read_stores_exactly_what_append_would():
    """The fused read and `append` differ only in what they return, not what they store.

    Run two identical caches side by side, one through `paged_attention`, one
    through `append`, with the same K/V. Their block tables and per-layer pools
    must end byte-for-byte identical: the fused read is `append`'s write plus a
    paged attention read, so the stored state cannot diverge. Only the return
    value differs (the attention output vs the running K/V).
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups

    fused_cache = PagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=4))
    append_cache = PagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=4))

    torch.manual_seed(8)
    for new_seq in (4, 1, 1):
        for layer in range(cfg.num_hidden_layers):
            k = torch.randn(1, n_kv, new_seq, d)
            v = torch.randn(1, n_kv, new_seq, d)
            q = torch.randn(1, n_q, new_seq, d)
            fused_cache.paged_attention(layer, k, v, q, n_rep)
            append_cache.append(layer, k, v)

    assert fused_cache.seq_len == append_cache.seq_len
    assert fused_cache.table.block_ids == append_cache.table.block_ids
    for layer in range(cfg.num_hidden_layers):
        assert torch.equal(fused_cache.k_pool[layer], append_cache.k_pool[layer])
        assert torch.equal(fused_cache.v_pool[layer], append_cache.v_pool[layer])


def test_fused_read_over_physically_scattered_blocks():
    """The fused read matches the contiguous truth even when blocks are scattered.

    Pre-fragment the pool so the sequence lands on non-contiguous physical blocks,
    then drive the fused read and a naive twin with identical K/V. The output still
    equals the contiguous SDPA: paging moved where the K/V live, not what attention
    computes, all the way through the cache method now, not just the bare reference.
    """
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    n_rep = cfg.num_kv_groups

    alloc = BlockAllocator(num_blocks=8, block_size=4)
    # Fragment: reserve and free two tables so the next allocation reuses blocks
    # out of order (LIFO), scattering the sequence across the pool.
    warm = BlockTable(alloc)
    warm.append(8)
    warm.free()
    paged = PagedKVCache(cfg, alloc)
    naive = NaiveKVCache(cfg.num_hidden_layers)

    torch.manual_seed(9)
    for new_seq in (6, 1, 1, 1):
        for layer in range(cfg.num_hidden_layers):
            k = torch.randn(1, n_kv, new_seq, d)
            v = torch.randn(1, n_kv, new_seq, d)
            q = torch.randn(1, n_q, new_seq, d)
            fused = paged.paged_attention(layer, k, v, q, n_rep)
            full_k, full_v = naive.append(layer, k, v)
            truth = _contiguous_attention(q, full_k, full_v, n_rep, d**-0.5)
            assert torch.equal(fused, truth)

    assert paged.table.block_ids != sorted(paged.table.block_ids)  # genuinely scattered


def test_fused_read_rejects_a_real_batch():
    """One sequence per cache until Phase 3, same guard the append write enforces."""
    cfg = _tiny_config()
    n_q, n_kv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    paged = PagedKVCache(cfg, BlockAllocator(num_blocks=4, block_size=4))
    k = torch.randn(2, n_kv, 1, d)
    v = torch.randn(2, n_kv, 1, d)
    q = torch.randn(2, n_q, 1, d)
    try:
        paged.paged_attention(0, k, v, q, cfg.num_kv_groups)
    except ValueError:
        return
    raise AssertionError("expected ValueError for batch > 1")
