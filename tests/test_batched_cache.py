"""Day 28 tests: one pool, one table per sequence, a batched decode read.

Day 27 batched the *prefill*: many ragged prompts padded into one rectangle, one
forward, row i identical to running prompt i alone. It stopped there, because the
KV cache was still single-sequence. `PagedKVCache` owns one block table, so a
padded batch reaching the fused read would have attended over whichever blocks
happened to be next in the pool, and `gqa_attention` refused rather than do that.

Today each row gets its own `BlockTable` over the *same* `BlockAllocator`, which is
the arrangement that makes a shared pool worth having: physical blocks are handed
out to whoever needs one next, rows interleave in the pool, and no row can name a
block it does not own. Two consequences the tests here pin:

  1. **The cache is ragged even though the batch is a rectangle.** The prefill
     write takes the batch's attention mask and stores only the real tokens, so a
     row of length 2 occupies 2 slots, not `max_len`. The padding is a property of
     the input tensor, not of the cache.
  2. **The pad mask becomes a context length.** Because no pad is ever written, the
     decode read never needs to be told which keys are fake: each row reads exactly
     `context_lens[i]` slots of its own. The rectangle comes back only for the slot
     mapping (`[batch, max_ctx]`), and the entries past a row's context are inert
     padding whose one job is to stay a legal index.

The oracles are the paths already verified: the single-sequence
`paged_attention_reference` per row, the single-sequence `PagedKVCache` per row,
and `greedy_generate_paged` per prompt. A batched decode that changes any token is
a bug, because batching is a throughput change and never a behaviour change.

Two tiers as usual: pure tests on tiny random weights, plus a `requires_weights`
run of the real Llama-3.2-1B where a leak across rows would be unmissable.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batch import pad_prompts
from nanoserve.cache import (
    BatchedPagedKVCache,
    BlockAllocator,
    KVCacheExhausted,
    PagedKVCache,
)
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import (
    paged_attention_batched_reference,
    paged_attention_reference,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel

from reference import PROMPT_IDS, WEIGHTS_DIR, requires_weights


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the other component tests use."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model(seed: int = 0) -> tuple[LlamaModel, ModelConfig]:
    """A tiny model on fixed random weights (seeded: greedy tokens must be stable)."""
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


def _kv(cfg: ModelConfig, batch: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch, cfg.num_key_value_heads, seq, cfg.head_dim)
    return torch.randn(*shape), torch.randn(*shape)


def _q(cfg: ModelConfig, batch: int, seq: int) -> torch.Tensor:
    return torch.randn(batch, cfg.num_attention_heads, seq, cfg.head_dim)


# --- pure: per-sequence tables over one shared pool ---------------------------


def test_each_row_gets_its_own_table_out_of_one_pool():
    cfg = _tiny_config()
    allocator = BlockAllocator(num_blocks=8, block_size=2)
    cache = BatchedPagedKVCache(cfg, allocator, batch_size=3)

    assert len(cache.tables) == 3
    assert all(table.allocator is allocator for table in cache.tables)
    assert allocator.num_free == 8  # nothing reserved until something is written


def test_ragged_write_stores_only_the_real_tokens():
    """The rectangle is the input's shape, not the cache's: pads are never written."""
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=3)
    batch = pad_prompts([[1, 2, 3], [4], [5, 6]], pad_id=0)  # lengths 3, 1, 2
    k, v = _kv(cfg, 3, batch.max_length)

    cache.write(0, k, v, batch.attention_mask)

    assert cache.seq_lens == [3, 1, 2]
    assert cache.cached_tokens == 6  # not 3 * 3


def test_the_real_tokens_land_in_the_pool_in_order():
    """Left padding puts row i's real K/V in its last columns; the cache re-orders."""
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    batch = pad_prompts([[1, 2, 3], [4]], pad_id=0)
    k, v = _kv(cfg, 2, 3)

    cache.write(0, k, v, batch.attention_mask)
    slots, _ = cache.slot_mapping(k.device)

    # Row 1's single real token is column 2 of the padded rectangle.
    torch.testing.assert_close(cache.k_pool[0][slots[1, 0]], k[1, :, 2, :])
    # Row 0 has no padding, so its three slots hold its three columns in order.
    for pos in range(3):
        torch.testing.assert_close(cache.k_pool[0][slots[0, pos]], k[0, :, pos, :])
        torch.testing.assert_close(cache.v_pool[0][slots[0, pos]], v[0, :, pos, :])


def test_rows_never_share_a_slot():
    """The invariant a shared pool lives or dies on: one slot, one token, one row."""
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=3)
    batch = pad_prompts([[1, 2, 3], [4], [5, 6]], pad_id=0)
    k, v = _kv(cfg, 3, batch.max_length)
    cache.write(0, k, v, batch.attention_mask)

    slots, lens = cache.slot_mapping(k.device)
    live = [int(s) for row, n in enumerate(lens.tolist()) for s in slots[row, :n]]
    assert len(live) == len(set(live)) == 6


def test_slot_mapping_is_a_rectangle_plus_a_context_length_per_row():
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=3)
    batch = pad_prompts([[1, 2, 3], [4], [5, 6]], pad_id=0)
    k, v = _kv(cfg, 3, batch.max_length)
    cache.write(0, k, v, batch.attention_mask)

    slots, lens = cache.slot_mapping(k.device)
    assert tuple(slots.shape) == (3, 3)  # [batch, max_ctx]
    assert lens.tolist() == [3, 1, 2]


def test_the_padding_in_the_slot_mapping_is_a_legal_index():
    """A -1 pad would silently wrap to the last slot; the gather runs before the mask."""
    cfg = _tiny_config()
    num_blocks, block_size = 8, 2
    cache = BatchedPagedKVCache(
        cfg, BlockAllocator(num_blocks=num_blocks, block_size=block_size), batch_size=2
    )
    batch = pad_prompts([[1, 2, 3], [4]], pad_id=0)
    k, v = _kv(cfg, 2, 3)
    cache.write(0, k, v, batch.attention_mask)

    slots, _ = cache.slot_mapping(k.device)
    assert int(slots.min()) >= 0
    assert int(slots.max()) < num_blocks * block_size


def test_a_decode_step_grows_every_row_by_one():
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    batch = pad_prompts([[1, 2, 3], [4]], pad_id=0)
    k, v = _kv(cfg, 2, 3)
    cache.write(0, k, v, batch.attention_mask)

    k1, v1 = _kv(cfg, 2, 1)
    cache.write(0, k1, v1)  # no mask: every column of a decode step is real

    assert cache.seq_lens == [4, 2]


def test_an_oversized_batch_leaves_every_table_untouched():
    """Atomicity across rows: the pool is checked before any row grows.

    `BlockTable.append` is already atomic for one sequence. With N sequences that is
    not enough: rows 0 and 1 can succeed and row 2 find the pool dry, leaving the
    batch half-written and the tables out of step with the tokens actually stored.
    """
    cfg = _tiny_config()
    allocator = BlockAllocator(num_blocks=2, block_size=2)  # 4 slots total
    cache = BatchedPagedKVCache(cfg, allocator, batch_size=3)
    batch = pad_prompts([[1, 2], [3, 4], [5, 6]], pad_id=0)  # needs 3 blocks, pool has 2
    k, v = _kv(cfg, 3, 2)

    with pytest.raises(KVCacheExhausted):
        cache.write(0, k, v, batch.attention_mask)

    assert cache.seq_lens == [0, 0, 0]
    assert allocator.num_free == 2


def test_write_rejects_a_mask_that_is_not_batch_by_seq():
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    k, v = _kv(cfg, 2, 3)
    with pytest.raises(ValueError, match="valid"):
        cache.write(0, k, v, torch.ones(2, 5, dtype=torch.bool))


def test_write_rejects_the_wrong_batch_size():
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    k, v = _kv(cfg, 3, 2)
    with pytest.raises(ValueError, match="batch"):
        cache.write(0, k, v)


def test_free_returns_every_rows_blocks_to_the_pool():
    cfg = _tiny_config()
    allocator = BlockAllocator(num_blocks=8, block_size=2)
    cache = BatchedPagedKVCache(cfg, allocator, batch_size=3)
    batch = pad_prompts([[1, 2, 3], [4], [5, 6]], pad_id=0)
    k, v = _kv(cfg, 3, 3)
    cache.write(0, k, v, batch.attention_mask)
    assert allocator.num_free < 8

    cache.free()
    assert allocator.num_free == 8
    assert cache.seq_lens == [0, 0, 0]


# --- pure: the batched read (the oracle is the single-sequence reference) ------


def test_batched_reference_matches_the_single_sequence_reference_per_row():
    cfg = _tiny_config()
    torch.manual_seed(7)
    n_rep = cfg.num_kv_groups
    num_slots = 12
    k_pool = torch.randn(num_slots, cfg.num_key_value_heads, cfg.head_dim)
    v_pool = torch.randn(num_slots, cfg.num_key_value_heads, cfg.head_dim)
    # Deliberately scattered and ragged: row 0 owns 4 slots, row 1 owns 2.
    rows = [[9, 2, 5, 11], [0, 7]]
    slot_mapping = torch.tensor([rows[0], rows[1] + [0, 0]], dtype=torch.long)
    context_lens = torch.tensor([4, 2], dtype=torch.long)
    q = _q(cfg, 2, 1)

    out = paged_attention_batched_reference(
        q, k_pool, v_pool, slot_mapping, context_lens, n_rep
    )

    assert tuple(out.shape) == (2, cfg.num_attention_heads, 1, cfg.head_dim)
    for row, slots in enumerate(rows):
        alone = paged_attention_reference(
            q[row : row + 1], k_pool, v_pool, torch.tensor(slots, dtype=torch.long), n_rep
        )
        torch.testing.assert_close(out[row : row + 1], alone, atol=1e-6, rtol=1e-5)


def test_the_read_ignores_everything_past_the_context_length():
    """The pad entries are gathered and then killed, so their contents cannot matter.

    The reference indexes the pool with the whole `[batch, max_ctx]` rectangle before
    it masks, so a padded entry really is read. Filling that slot with a value large
    enough to dominate any softmax proves the mask, not the arithmetic, is what keeps
    it out. A kernel skips the load instead; both must agree.
    """
    cfg = _tiny_config()
    torch.manual_seed(11)
    k_pool = torch.randn(6, cfg.num_key_value_heads, cfg.head_dim)
    v_pool = torch.randn(6, cfg.num_key_value_heads, cfg.head_dim)
    slot_mapping = torch.tensor([[1, 3, 4], [2, 5, 5]], dtype=torch.long)
    context_lens = torch.tensor([3, 1], dtype=torch.long)
    q = _q(cfg, 2, 1)

    before = paged_attention_batched_reference(
        q, k_pool, v_pool, slot_mapping, context_lens, cfg.num_kv_groups
    )
    k_pool[5] = 1e4  # slot 5 is only ever reached as row 1's padding
    v_pool[5] = 1e4
    after = paged_attention_batched_reference(
        q, k_pool, v_pool, slot_mapping, context_lens, cfg.num_kv_groups
    )

    assert torch.isfinite(after).all()
    torch.testing.assert_close(before, after)


def test_batched_reference_is_the_decode_read():
    """One new token per row. A ragged prefill is the dense masked path, not this."""
    cfg = _tiny_config()
    k_pool = torch.randn(6, cfg.num_key_value_heads, cfg.head_dim)
    with pytest.raises(ValueError, match="decode"):
        paged_attention_batched_reference(
            _q(cfg, 2, 2),
            k_pool,
            k_pool.clone(),
            torch.zeros(2, 3, dtype=torch.long),
            torch.tensor([3, 3]),
            cfg.num_kv_groups,
        )


def test_batched_reference_rejects_a_context_length_past_the_mapping():
    cfg = _tiny_config()
    k_pool = torch.randn(6, cfg.num_key_value_heads, cfg.head_dim)
    with pytest.raises(ValueError, match="context_lens"):
        paged_attention_batched_reference(
            _q(cfg, 2, 1),
            k_pool,
            k_pool.clone(),
            torch.zeros(2, 3, dtype=torch.long),
            torch.tensor([3, 4]),  # row 1 claims more history than it has slots
            cfg.num_kv_groups,
        )


def test_batched_reference_rejects_an_empty_context():
    """Zero visible keys is a 0/0 softmax; a decode query always has its own token."""
    cfg = _tiny_config()
    k_pool = torch.randn(6, cfg.num_key_value_heads, cfg.head_dim)
    with pytest.raises(ValueError, match="context_lens"):
        paged_attention_batched_reference(
            _q(cfg, 2, 1),
            k_pool,
            k_pool.clone(),
            torch.zeros(2, 3, dtype=torch.long),
            torch.tensor([2, 0]),
            cfg.num_kv_groups,
        )


# --- pure: the cache's own read ----------------------------------------------


def test_batched_read_matches_a_single_sequence_paged_cache():
    """Two rows in one pool must each get what their own `PagedKVCache` would give."""
    cfg = _tiny_config()
    torch.manual_seed(3)
    prompts = [[1, 2, 3], [4]]
    batch = pad_prompts(prompts, pad_id=0)
    k, v = _kv(cfg, 2, batch.max_length)
    kn, vn = _kv(cfg, 2, 1)
    q = _q(cfg, 2, 1)

    shared = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    shared.write(0, k, v, batch.attention_mask)
    out = shared.paged_attention(0, kn, vn, q, cfg.num_kv_groups)

    for row, prompt in enumerate(prompts):
        cols = batch.attention_mask[row].nonzero(as_tuple=True)[0]
        alone = PagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2))
        alone.append(0, k[row : row + 1][:, :, cols, :], v[row : row + 1][:, :, cols, :])
        expected = alone.paged_attention(
            0, kn[row : row + 1], vn[row : row + 1], q[row : row + 1], cfg.num_kv_groups
        )
        assert alone.seq_len == len(prompt) + 1
        torch.testing.assert_close(out[row : row + 1], expected, atol=1e-4, rtol=1e-3)


def test_a_neighbour_cannot_change_a_rows_output():
    """Isolation is the whole claim: same row, different batch-mate, same numbers."""
    cfg = _tiny_config()
    torch.manual_seed(5)
    k, v = _kv(cfg, 2, 2)
    kn, vn = _kv(cfg, 2, 1)
    q = _q(cfg, 2, 1)

    def run(neighbour_k: torch.Tensor) -> torch.Tensor:
        cache = BatchedPagedKVCache(
            cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2
        )
        k_in = k.clone()
        k_in[1] = neighbour_k
        cache.write(0, k_in, v)
        return cache.paged_attention(0, kn, vn, q, cfg.num_kv_groups)[0]

    torch.testing.assert_close(run(k[1]), run(k[1] * 100 + 7))


def test_the_paged_decode_read_refuses_a_key_mask():
    """Per-sequence tables replace the pad mask; accepting one would hide a bug."""
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    k, v = _kv(cfg, 2, 1)
    cache.write(0, k, v)
    with pytest.raises(ValueError, match="context"):
        cache.paged_attention(
            0, k, v, _q(cfg, 2, 1), cfg.num_kv_groups, attention_mask=torch.ones(2, 1)
        )


def test_the_batched_read_is_one_token_per_row():
    cfg = _tiny_config()
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=2), batch_size=2)
    k, v = _kv(cfg, 2, 2)
    with pytest.raises(ValueError, match="decode"):
        cache.paged_attention(0, k, v, _q(cfg, 2, 2), cfg.num_kv_groups)


def test_a_masked_prefill_onto_a_non_empty_cache_is_refused():
    """A masked forward attends over this step's K/V only, so it must be the first.

    The prefill branch is legal exactly because a prefill's history *is* the tokens it
    was handed. Run it a second time and the dense attention would quietly ignore
    everything already cached, which reads as a plausible answer to the wrong
    question. Chunked prefill is a real feature, and it is a different one.
    """
    model, cfg = _model()
    batch = pad_prompts([[1, 2, 3], [4]], pad_id=0)
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=4), batch_size=2)
    model.forward(
        batch.input_ids, batch.position_ids, cache=cache, attention_mask=batch.attention_mask
    )
    with pytest.raises(ValueError, match="already hold"):
        model.forward(
            batch.input_ids,
            batch.position_ids,
            cache=cache,
            attention_mask=batch.attention_mask,
        )


def test_a_prefill_mask_must_cover_this_steps_own_tokens():
    model, cfg = _model()
    batch = pad_prompts([[1, 2, 3], [4]], pad_id=0)
    cache = BatchedPagedKVCache(cfg, BlockAllocator(num_blocks=8, block_size=4), batch_size=2)
    with pytest.raises(ValueError, match="key mask"):
        model.forward(
            batch.input_ids,
            batch.position_ids,
            cache=cache,
            attention_mask=torch.ones(2, batch.max_length + 2, dtype=torch.bool),
        )


# --- pure: the batched decode loop -------------------------------------------


def test_batched_decode_matches_each_prompt_decoded_alone():
    """The headline: N prompts decoding together emit exactly their solo tokens."""
    model, _ = _model()
    prompts = [[3, 9, 14, 2], [7], [11, 5]]

    together = model.greedy_generate_batch(prompts, max_new_tokens=4, block_size=2)

    for row, prompt in enumerate(prompts):
        alone = model.greedy_generate_paged(
            torch.tensor([prompt]), max_new_tokens=4, block_size=2
        )
        assert together[row] == alone[0].tolist()


def test_batched_decode_returns_ragged_rows():
    model, _ = _model()
    prompts = [[3, 9, 14, 2], [7]]
    out = model.greedy_generate_batch(prompts, max_new_tokens=3, block_size=4)
    assert [len(row) for row in out] == [len(prompts[0]) + 3, len(prompts[1]) + 3]
    assert out[0][:4] == prompts[0]
    assert out[1][:1] == prompts[1]


def test_each_row_stops_at_its_own_eos():
    """Rows finish independently; the batch runs until the last one is done."""
    model, _ = _model()
    prompts = [[3, 9, 14, 2], [7]]
    first = model.greedy_generate_batch(prompts, max_new_tokens=1, block_size=4)
    eos = first[0][-1]  # whatever row 0 emits first, treat it as end-of-sequence

    out = model.greedy_generate_batch(prompts, max_new_tokens=5, eos_id=eos, block_size=4)
    assert out[0][-1] == eos and out[0].count(eos) == 1
    assert len(out[0]) == len(prompts[0]) + 1


def test_a_finished_row_keeps_paying_for_the_forward():
    """The static-batching bill, the decode-side twin of Day 27's `padding_waste`.

    A row that hit EOS is still a row of the rectangle, so it still gets a query, a
    slot, and a block until the whole batch drains. That is what continuous batching
    fixes, and asserting it here means the fix has something to be measured against.
    """
    model, _ = _model()
    prompts = [[3, 9, 14, 2], [7]]
    first = model.greedy_generate_batch(prompts, max_new_tokens=1, block_size=4)
    eos = first[0][-1]

    out, cache = model.greedy_generate_batch(
        prompts, max_new_tokens=5, eos_id=eos, block_size=4, return_cache=True
    )
    assert len(out[0]) == len(prompts[0]) + 1  # row 0 emitted one token
    assert cache.seq_lens[0] > len(prompts[0]) + 1  # and kept caching after it stopped


def test_batched_decode_rejects_a_pool_that_cannot_hold_the_batch():
    model, _ = _model()
    with pytest.raises(KVCacheExhausted):
        model.greedy_generate_batch(
            [[3, 9, 14, 2], [7]], max_new_tokens=4, block_size=2, num_blocks=1
        )


# --- real weights ------------------------------------------------------------


@requires_weights
def test_batched_decode_matches_single_decode_on_llama():
    """The same claim on the real 1B, where a leak across rows is unmissable."""
    from nanoserve.loader import load_weights

    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    prompts = [PROMPT_IDS, PROMPT_IDS[:3]]

    together = model.greedy_generate_batch(
        prompts, max_new_tokens=4, pad_id=cfg.vocab_size - 1
    )

    for row, prompt in enumerate(prompts):
        alone = model.greedy_generate_paged(torch.tensor([prompt]), max_new_tokens=4)
        assert together[row] == alone[0].tolist()
