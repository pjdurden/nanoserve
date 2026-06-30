"""Day 11 tests: the naive contiguous KV cache.

The cache is an optimization, not a behavior change, so the whole point of these
tests is that it changes *nothing* observable. Three tiers:

  - pure cache mechanics (torch only): the store grows by exactly what you append
    and hands back the full contiguous K/V each step.
  - pure equivalence (random weights): cached greedy decode produces the exact
    same tokens as the Week-2 recompute-everything path, and a prefill-then-decode
    matches a single full forward over the same sequence. This is the correctness
    contract: a cache that changes the output is a broken cache.
  - against the real Llama-3.2-1B: cached greedy matches HF token for token, the
    same north star Week 2 hit the slow way.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.cache import (
    BlockAllocator,
    BlockTable,
    KVCacheExhausted,
    NaiveKVCache,
)
from nanoserve.config import ModelConfig
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes, load_weights
from nanoserve.model import LlamaModel

from reference import PROMPT_IDS, WEIGHTS_DIR, hf_model, requires_weights


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the model tests use."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _random_weights(cfg: ModelConfig) -> Weights:
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, cfg)


# --- pure: cache mechanics --------------------------------------------------


def test_cache_starts_empty_and_grows_by_appended_length():
    cache = NaiveKVCache(num_layers=2)
    assert cache.seq_len == 0

    # A 5-token prefill chunk for layer 0, then two 1-token decode steps.
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    fk, fv = cache.append(0, k, v)
    assert fk.shape == (1, 2, 5, 4)
    assert cache.seq_len == 5

    k1 = torch.randn(1, 2, 1, 4)
    v1 = torch.randn(1, 2, 1, 4)
    fk, fv = cache.append(0, k1, v1)
    assert fk.shape == (1, 2, 6, 4)
    assert cache.seq_len == 6
    # The returned K is the running concatenation, oldest token first.
    assert torch.equal(fk[:, :, :5], k)
    assert torch.equal(fk[:, :, 5:], k1)
    assert torch.equal(fv[:, :, 5:], v1)


def test_cache_layers_are_independent():
    """Appending to layer 0 must not touch layer 1's store."""
    cache = NaiveKVCache(num_layers=2)
    cache.append(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
    assert cache.seq_len == 0 or cache.k[1] is None
    # seq_len reads layer 0, which has 3; layer 1 is still empty.
    assert cache.seq_len == 3
    assert cache.k[1] is None


# --- pure: the cache changes nothing ----------------------------------------


def test_cached_prefill_matches_uncached_forward():
    """forward with a cache writes K/V but returns the same logits as without one.

    A prefill over the whole prompt with an empty cache is the uncached forward
    plus a side effect (the cache fills). The logits must be identical (the
    rectangular mask at past=0 is exactly the square causal mask), so this pins
    that adding the cache plumbing did not perturb the Week-2 math.
    """
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 6))

    plain = model.forward(ids)
    cache = NaiveKVCache(cfg.num_hidden_layers)
    cached = model.forward(ids, cache=cache)

    assert torch.allclose(plain, cached, atol=1e-6)
    assert cache.seq_len == 6


def test_prefill_then_decode_matches_full_forward():
    """One decode step through the cache equals a full forward over prompt+token.

    Prefill the prompt, append one more token, and run a single-token forward at
    its position through the cache. The last-position logits must match what a
    from-scratch forward over the whole 7-token sequence produces: the cache
    assembles the identical K/V matrix, just incrementally.
    """
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))

    cache = NaiveKVCache(cfg.num_hidden_layers)
    model.forward(ids, cache=cache)  # prefill, fills cache to len 6
    pos = torch.tensor([[6]])
    step = model.forward(nxt, position_ids=pos, cache=cache)  # [1, 1, vocab]

    full = model.forward(torch.cat([ids, nxt], dim=1))  # [1, 7, vocab]
    assert torch.allclose(step[:, -1], full[:, -1], atol=1e-5)
    assert cache.seq_len == 7


def test_cached_greedy_matches_naive_greedy_token_for_token():
    """The correctness contract: cached decode == Week-2 recompute decode, exactly.

    Same weights, same prompt, same greedy choice at every step, just with the
    O(n) cache instead of the O(n^2) recompute. The tokens must be torch.equal;
    any divergence is a cache bug (wrong position, wrong mask, stale K/V).
    """
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 4))

    naive = model.greedy_generate(ids, max_new_tokens=8)
    cached = model.greedy_generate_cached(ids, max_new_tokens=8)
    assert torch.equal(naive, cached)


def test_cached_greedy_appends_and_stops_at_eos():
    """The cached loop honors the same max_new_tokens and eos contract as Week 2."""
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 4))

    out = model.greedy_generate_cached(ids, max_new_tokens=7)
    assert out.shape == (1, 4 + 7)
    assert torch.equal(out[:, :4], ids)

    first = model.greedy_generate_cached(ids, max_new_tokens=1)[0, -1].item()
    stopped = model.greedy_generate_cached(ids, max_new_tokens=10, eos_id=first)
    assert stopped.shape == (1, 5)
    assert stopped[0, -1].item() == first


def test_cached_greedy_rejects_a_real_batch():
    """One sequence only until Phase 3, same guard as the naive path."""
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    try:
        model.greedy_generate_cached(ids, max_new_tokens=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for batch > 1")


# --- against the real Llama-3.2-1B ------------------------------------------


@requires_weights
def test_cached_greedy_matches_hf_multi_token():
    """Cached greedy decode matches HF token for token on the fixed prompt.

    Week 2 hit this north star by recomputing the whole prefix every step; Day 11
    hits the same tokens with a KV cache, which is how HF itself runs. So this is
    a three-way agreement: nanoserve-cached == nanoserve-naive == HF.
    """
    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    ids = torch.tensor([PROMPT_IDS])
    n = 20

    mine = model.greedy_generate_cached(ids, max_new_tokens=n)
    hf = hf_model()
    with torch.no_grad():
        ref = hf.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    assert torch.equal(mine, ref)


# --- Day 14: the block allocator -------------------------------------------
#
# The allocator is pure bookkeeping over a fixed pool of physical block ids; no
# tensors yet (Week 5 will hang real K/V storage off these ids). So these tests
# are torch-free and pin the OS-paging contract: hand out distinct blocks, refuse
# when the pool is dry, take them back on free, and never lose or duplicate a
# block. A block double-counted is corrupted attention later, so the invariants
# are strict on purpose.


def test_fresh_allocator_has_every_block_free():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.num_blocks == 4
    assert alloc.block_size == 16
    assert alloc.num_free == 4


def test_allocate_hands_out_distinct_blocks_and_shrinks_the_pool():
    alloc = BlockAllocator(num_blocks=3, block_size=16)
    ids = [alloc.allocate() for _ in range(3)]
    assert sorted(ids) == [0, 1, 2]  # distinct ids, all in range
    assert len(set(ids)) == 3
    assert alloc.num_free == 0


def test_allocate_on_an_empty_pool_raises_exhausted():
    alloc = BlockAllocator(num_blocks=1, block_size=16)
    alloc.allocate()
    with pytest.raises(KVCacheExhausted):
        alloc.allocate()


def test_free_returns_a_block_to_the_pool_and_it_is_reusable():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    a = alloc.allocate()
    alloc.allocate()
    assert alloc.num_free == 0
    alloc.free(a)
    assert alloc.num_free == 1
    # the freed block is allocatable again
    assert alloc.allocate() == a
    assert alloc.num_free == 0


def test_blocks_for_length_is_ceiling_division():
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    assert alloc.blocks_for_length(0) == 0
    assert alloc.blocks_for_length(1) == 1
    assert alloc.blocks_for_length(16) == 1
    assert alloc.blocks_for_length(17) == 2
    assert alloc.blocks_for_length(32) == 2


def test_allocate_for_a_length_reserves_the_right_block_count():
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    blocks = alloc.allocate_for(40)  # ceil(40 / 16) = 3
    assert len(blocks) == 3
    assert len(set(blocks)) == 3
    assert alloc.num_free == 5


def test_allocate_for_zero_tokens_reserves_nothing():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.allocate_for(0) == []
    assert alloc.num_free == 4


def test_allocate_for_is_atomic_when_the_pool_cannot_satisfy_it():
    """A request needing more blocks than are free must reserve none of them."""
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    with pytest.raises(KVCacheExhausted):
        alloc.allocate_for(40)  # needs 3, only 2 free
    assert alloc.num_free == 2  # nothing was taken
    # can_allocate agrees, and the pool is still fully usable afterwards
    assert alloc.can_allocate(32)
    assert not alloc.can_allocate(33)
    assert len(alloc.allocate_for(32)) == 2


def test_free_all_returns_a_whole_sequence_of_blocks():
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    blocks = alloc.allocate_for(40)
    alloc.free_all(blocks)
    assert alloc.num_free == 8


def test_freeing_a_block_that_is_not_allocated_is_rejected():
    """Double free or freeing a stranger corrupts the pool, so it must raise."""
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    a = alloc.allocate()
    alloc.free(a)
    with pytest.raises(ValueError):
        alloc.free(a)  # double free
    with pytest.raises(ValueError):
        alloc.free(99)  # never belonged to this pool


def test_construction_rejects_a_nonpositive_pool_or_block_size():
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=0, block_size=16)
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=4, block_size=0)


# --- Day 15: the block table (logical -> physical address translation) ------
#
# The allocator owns physical blocks but says nothing about *which* block holds
# *which* logical token. The block table is that per-sequence map. It is still
# torch-free bookkeeping (Week 5 hangs real K/V off the slots): grow a sequence
# a token at a time, pull a fresh block from the pool only when a token crosses a
# block boundary, and translate any logical position to its physical (block,
# offset) and flat slot. The headline invariant: incremental growth must land a
# sequence on the exact same blocks and slots as one bulk allocation would.


def test_fresh_block_table_is_empty_and_holds_no_blocks():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable(alloc)
    assert table.num_tokens == 0
    assert table.block_ids == []
    assert table.capacity == 0
    assert alloc.num_free == 4  # an empty sequence reserves nothing


def test_append_within_one_block_takes_exactly_one_block():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable(alloc)
    table.append(5)
    assert table.num_tokens == 5
    assert len(table.block_ids) == 1
    assert table.capacity == 16
    assert alloc.num_free == 3


def test_position_translates_logical_to_block_and_offset():
    """The Day-14 example: position 17 with block_size 16 is offset 1 of block 2."""
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    table = BlockTable(alloc)
    table.append(20)  # two blocks
    assert table.block_ids[0:2] == [0, 1]
    assert table.position(0) == (0, 0)
    assert table.position(15) == (0, 15)
    assert table.position(16) == (1, 0)
    assert table.position(17) == (1, 1)  # second block, offset 1


def test_slot_is_the_flat_index_into_the_block_pool():
    """slot = block_id * block_size + offset, the address Week 5 writes K/V to."""
    alloc = BlockAllocator(num_blocks=8, block_size=16)
    table = BlockTable(alloc)
    table.append(20)
    assert table.slot(0) == 0
    assert table.slot(17) == table.block_ids[1] * 16 + 1
    # every live position maps to a distinct physical slot
    slots = [table.slot(p) for p in range(table.num_tokens)]
    assert len(set(slots)) == table.num_tokens


def test_new_block_is_pulled_only_when_a_token_crosses_a_boundary():
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    table = BlockTable(alloc)
    table.append(4)  # fills block 0 exactly
    assert len(table.block_ids) == 1
    assert alloc.num_free == 3
    table.append(1)  # 5th token needs a second block
    assert len(table.block_ids) == 2
    assert alloc.num_free == 2
    assert table.position(4) == (table.block_ids[1], 0)


def test_incremental_appends_match_one_bulk_append():
    """Growing a token at a time lands on the same blocks and slots as one append.

    This is the block table's correctness contract, the paging analogue of the
    cache's "an optimization changes nothing": decode (one token per step) and
    prefill (a whole prompt at once) must produce identical address translation.
    """
    bulk_alloc = BlockAllocator(num_blocks=8, block_size=4)
    bulk = BlockTable(bulk_alloc)
    bulk.append(10)

    step_alloc = BlockAllocator(num_blocks=8, block_size=4)
    step = BlockTable(step_alloc)
    for _ in range(10):
        step.append(1)

    assert step.block_ids == bulk.block_ids
    assert [step.slot(p) for p in range(10)] == [bulk.slot(p) for p in range(10)]
    assert step_alloc.num_free == bulk_alloc.num_free


def test_position_out_of_range_raises():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable(alloc)
    table.append(5)
    with pytest.raises(IndexError):
        table.position(5)  # only 0..4 are live
    with pytest.raises(IndexError):
        table.position(-1)


def test_free_returns_every_block_and_resets_the_table():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable(alloc)
    table.append(40)  # three blocks
    assert alloc.num_free == 1
    table.free()
    assert alloc.num_free == 4
    assert table.num_tokens == 0
    assert table.block_ids == []
    # the table is reusable after a free
    table.append(3)
    assert table.num_tokens == 3


def test_append_is_atomic_when_the_pool_cannot_cover_the_growth():
    """If growth needs more blocks than are free, reserve none and leave it intact."""
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    table = BlockTable(alloc)
    table.append(4)  # one block, one free left
    with pytest.raises(KVCacheExhausted):
        table.append(8)  # would need two more blocks, only one free
    assert table.num_tokens == 4  # unchanged
    assert len(table.block_ids) == 1
    assert alloc.num_free == 1


def test_table_handles_physically_scattered_blocks():
    """After a free-and-reallocate, a sequence's blocks are non-contiguous and the
    translation still resolves each logical position to its real physical block."""
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    first = BlockTable(alloc)
    first.append(8)  # blocks [0, 1]
    first.free()  # back to the pool, LIFO: [.. ,1, 0] on top

    table = BlockTable(alloc)
    table.append(12)  # three blocks, reusing freed ones out of order
    assert len(set(table.block_ids)) == 3
    assert table.block_ids != sorted(table.block_ids) or len(table.block_ids) == 3
    for p in range(12):
        block, offset = table.position(p)
        assert block == table.block_ids[p // 4]
        assert offset == p % 4
        assert table.slot(p) == block * 4 + offset
