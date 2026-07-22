"""Day 27 tests: many prompts, one padded forward.

Phase 3 opens on the plainest version of batching there is: take N prompts of
different lengths, stack them into one rectangular `[batch, max_len]` tensor, and
run a single forward. The rectangle is the whole problem. Prompts are ragged and
tensors are not, so the short rows get padded, and padding is a lie the model will
happily believe. Two things travel with the batch to stop it: an **attention
mask**, so a real query never scores against a pad key, and **position_ids that
skip the padding**, so each sequence's first real token is rotated to position 0.

Exactly one of those two turned out to be load-bearing, and finding out which is
what the control tests here are for. The mask is: under *left* padding the pads
sit at the low indices the causal mask happily permits, so dropping it moves the
logits by 7.5. The positions are not: a RoPE score depends only on the
*difference* between two positions, so shifting a whole row by a constant cancels
inside every dot product and the batched row comes out bit-identical either way.
The tests say so, including the limit of that invariance, since a *gap* in the
positions is not a uniform shift and is not free.

The oracle throughout is the already-verified single-sequence path: row i of the
batched logits must equal `forward(prompt_i)` run on its own.

Two tiers, the usual shape:
  - pure tests (torch only, tiny random weights): the padding helpers, the
    equality with the unbatched path, the controls, and the NaN corner.
  - a `requires_weights` test on the real Llama-3.2-1B, where a leak of one pad
    token would move the logits far more than the tolerance allows.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.batch import PaddedBatch, last_token_logits, pad_prompts
from nanoserve.cache import BlockAllocator, PagedKVCache
from nanoserve.config import ModelConfig
from nanoserve.layers import gqa_attention
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel

from reference import PROMPT_IDS, WEIGHTS_DIR, requires_weights


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the model/cache tests use."""
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _model() -> tuple[LlamaModel, ModelConfig]:
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg)), cfg


# --- pure: the padded batch itself ------------------------------------------


def test_pad_prompts_left_pads_every_row_to_the_longest():
    batch = pad_prompts([[5, 6, 7], [9], [1, 2]], pad_id=0)
    assert isinstance(batch, PaddedBatch)
    assert batch.batch_size == 3
    assert batch.max_length == 3
    assert batch.input_ids.tolist() == [[5, 6, 7], [0, 0, 9], [0, 1, 2]]
    assert batch.lengths.tolist() == [3, 1, 2]


def test_pad_prompts_marks_only_real_tokens_in_the_mask():
    batch = pad_prompts([[5, 6, 7], [9]], pad_id=0)
    assert batch.attention_mask.dtype is torch.bool
    assert batch.attention_mask.tolist() == [[True, True, True], [False, False, True]]


def test_pad_prompts_positions_restart_at_each_rows_first_real_token():
    """The RoPE half of the fix: a left-padded row is still positions 0..n-1.

    Row 1 holds one real token sitting at index 2 of the rectangle. Its RoPE
    position must be 0, not 2, or the model reads it as the third token of a
    sequence it never saw. Pad slots get 0 as well; nothing reads them.
    """
    batch = pad_prompts([[5, 6, 7], [9]], pad_id=0)
    assert batch.position_ids.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_pad_prompts_right_side_puts_the_padding_after_the_prompt():
    batch = pad_prompts([[5, 6, 7], [9]], pad_id=0, side="right")
    assert batch.input_ids.tolist() == [[5, 6, 7], [9, 0, 0]]
    assert batch.attention_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.position_ids[1, 0].item() == 0


def test_padding_waste_is_the_fraction_of_the_grid_that_is_padding():
    """The number that motivates everything after static batching.

    Three prompts of 3, 1 and 2 tokens cost a 3x3 grid = 9 slots, of which 3 are
    padding. A third of this forward is arithmetic on tokens that do not exist.
    """
    batch = pad_prompts([[5, 6, 7], [9], [1, 2]], pad_id=0)
    assert batch.num_pad_tokens == 3
    assert batch.padding_waste == pytest.approx(3 / 9)


def test_padding_waste_is_zero_when_every_prompt_is_the_same_length():
    batch = pad_prompts([[1, 2], [3, 4]], pad_id=0)
    assert batch.num_pad_tokens == 0
    assert batch.padding_waste == 0.0


def test_pad_prompts_rejects_an_empty_batch_or_an_empty_prompt():
    with pytest.raises(ValueError, match="at least one prompt"):
        pad_prompts([], pad_id=0)
    with pytest.raises(ValueError, match="empty"):
        pad_prompts([[1, 2], []], pad_id=0)


def test_pad_prompts_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side"):
        pad_prompts([[1, 2]], pad_id=0, side="middle")


def test_last_token_logits_picks_each_rows_last_real_position():
    """Left padding exists so this gather is `[:, -1]`; right padding needs it.

    The tensor here stands in for logits: value `10*row + column` makes the picked
    column readable. Left-padded rows all end on the last column; right-padded
    rows end at `length - 1`, which differs per row.
    """
    fake = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3, 1)
    left = pad_prompts([[5, 6, 7], [9]], pad_id=0)
    assert last_token_logits(fake, left).flatten().tolist() == [2.0, 5.0]
    right = pad_prompts([[5, 6, 7], [9]], pad_id=0, side="right")
    assert last_token_logits(fake, right).flatten().tolist() == [2.0, 3.0]


# --- pure: the batched forward against the single-prompt oracle --------------


def test_batched_forward_matches_each_prompt_run_alone():
    """The claim of the day: batching changes throughput, never the numbers."""
    model, cfg = _model()
    prompts = [[3, 9, 14, 2, 7], [11, 4], [8, 8, 1]]
    batch = pad_prompts(prompts, pad_id=0)
    batched = model.forward_batch(batch)
    last = last_token_logits(batched, batch)
    for row, prompt in enumerate(prompts):
        alone = model.forward(torch.tensor([prompt]))[0, -1]
        torch.testing.assert_close(last[row], alone, atol=1e-5, rtol=1e-4)


def test_greedy_token_batch_matches_each_prompt_greedy_alone():
    model, _ = _model()
    prompts = [[3, 9, 14, 2, 7], [11, 4], [8, 8, 1]]
    batch = pad_prompts(prompts, pad_id=0)
    batched = model.greedy_token_batch(batch)
    for row, prompt in enumerate(prompts):
        alone = model.greedy_token(torch.tensor([prompt]))
        assert batched[row].item() == alone.item()


def test_without_the_mask_the_left_padding_leaks_into_the_real_tokens():
    """Control: drop the mask and the batched row stops matching.

    Under left padding the pads sit at low indices, which the causal mask allows
    every real query to attend to, so the leak is total and silent. This test
    fails (i.e. the tensors match) only if the mask has stopped being applied.
    """
    model, _ = _model()
    prompts = [[3, 9, 14, 2, 7], [11, 4]]
    batch = pad_prompts(prompts, pad_id=0)
    unmasked = model.forward(batch.input_ids, batch.position_ids, attention_mask=None)
    alone = model.forward(torch.tensor([prompts[1]]))[0, -1]
    assert not torch.allclose(unmasked[1, -1], alone, atol=1e-3)


def test_a_uniform_position_offset_is_invisible_to_rope():
    """The half of the fix that turns out not to be a fix: RoPE is relative.

    Written expecting the opposite. Left padding pushes row 1's two real tokens to
    indices 3 and 4, so feeding the default `0..max_len-1` positions rotates them
    as positions 3 and 4 instead of 0 and 1, which sounds exactly like the bug the
    padding-aware positions exist to prevent. It is not: a RoPE score depends only
    on the *difference* of the two positions, so adding a constant to every
    position in a row cancels inside every dot product. Measured on this model the
    logits are bit-identical at a shift of 3, and 2e-6 off at a shift of 5000,
    which is float error in the cos/sin table, not a change of meaning.

    So the positions are kept padding-aware for reasons outside this forward pass
    (see `pad_prompts`), and the mask is the part that is load-bearing today.
    """
    model, _ = _model()
    prompts = [[3, 9, 14, 2, 7], [11, 4]]
    batch = pad_prompts(prompts, pad_id=0)
    alone = model.forward(torch.tensor([prompts[1]]))[0, -1]

    default_positions = model.forward(batch.input_ids, attention_mask=batch.attention_mask)
    torch.testing.assert_close(default_positions[1, -1], alone, atol=1e-5, rtol=1e-4)

    far = batch.position_ids.clone()
    far[1] = far[1] + 5000
    shifted = model.forward(batch.input_ids, far, attention_mask=batch.attention_mask)
    torch.testing.assert_close(shifted[1, -1], alone, atol=1e-4, rtol=1e-3)


def test_a_gap_in_the_positions_is_not_invisible():
    """The limit of that invariance: only a *uniform* shift cancels.

    Rotate the row's two real tokens to positions 0 and 7 rather than 0 and 1 and
    the distance between them changes, so the score changes. This is why the
    positions still have to be derived rather than improvised: uniform offsets are
    free, but a row that skips a position (which is what a wrong per-token
    position_ids computation produces) is a different sequence.
    """
    model, _ = _model()
    prompts = [[3, 9, 14, 2, 7], [11, 4]]
    batch = pad_prompts(prompts, pad_id=0)
    gapped = batch.position_ids.clone()
    gapped[1, -1] = 7
    out = model.forward(batch.input_ids, gapped, attention_mask=batch.attention_mask)
    alone = model.forward(torch.tensor([prompts[1]]))[0, -1]
    assert not torch.allclose(out[1, -1], alone, atol=1e-3)


def test_fully_padded_query_rows_stay_finite():
    """The NaN corner: a pad query whose only causal key is itself, and masked.

    Mask with `-inf` and that row's scores are all `-inf`, softmax divides zero by
    zero, and the NaN does not stay put: the next block reads the pad position as
    a *key* for the real tokens, so one NaN poisons the whole batch. The padding
    bias is `finfo.min`, not `-inf`, exactly so this row stays finite.
    """
    model, _ = _model()
    batch = pad_prompts([[3, 9, 14, 2, 7], [11]], pad_id=0)
    logits = model.forward_batch(batch)
    assert torch.isfinite(logits).all()


def test_right_padding_makes_the_mask_a_no_op_for_the_real_rows():
    """Worth knowing: with right padding the causal mask already hides the pads.

    Pads sit *after* every real token, so no real query is causally allowed to see
    them in the first place. Masked and unmasked agree on the real positions. The
    mask is not wasted work though: it is what makes left padding legal, and left
    padding is what puts every row's next-token logits in the same column.
    """
    model, _ = _model()
    prompts = [[3, 9, 14, 2, 7], [11, 4]]
    batch = pad_prompts(prompts, pad_id=0, side="right")
    masked = model.forward_batch(batch)
    unmasked = model.forward(batch.input_ids, batch.position_ids, attention_mask=None)
    for row, prompt in enumerate(prompts):
        n = len(prompt)
        torch.testing.assert_close(masked[row, :n], unmasked[row, :n], atol=1e-6, rtol=1e-5)


# --- pure: the mask contract at the attention layer --------------------------


def test_gqa_attention_rejects_a_mask_that_is_not_batch_by_kv_len():
    cfg = _tiny_config()
    b, seq = 2, 4
    x = torch.randn(b, seq, cfg.hidden_size)
    w = {
        "q": torch.randn(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
        "k": torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        "v": torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        "o": torch.randn(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
    }
    cos = torch.ones(b, seq, cfg.head_dim)
    sin = torch.zeros(b, seq, cfg.head_dim)
    with pytest.raises(ValueError, match="attention_mask"):
        gqa_attention(
            x, w["q"], w["k"], w["v"], w["o"], cos, sin, cfg,
            attention_mask=torch.ones(b, seq + 1, dtype=torch.bool),
        )


def test_paged_cache_rejects_a_padded_batch_for_now():
    """The honest boundary: the paged read is still one sequence, one table.

    A padded batch reaching the fused read would need a per-sequence block table
    and a per-sequence slot mapping, which is a later day this week. Until then it
    raises rather than quietly attending over another sequence's blocks.
    """
    cfg = _tiny_config()
    b, seq = 2, 4
    x = torch.randn(b, seq, cfg.hidden_size)
    cache = PagedKVCache(cfg, BlockAllocator(num_blocks=4, block_size=16))
    cos = torch.ones(b, seq, cfg.head_dim)
    sin = torch.zeros(b, seq, cfg.head_dim)
    with pytest.raises(ValueError, match="paged"):
        gqa_attention(
            x,
            torch.randn(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
            torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
            torch.randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
            torch.randn(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
            cos, sin, cfg,
            cache=cache,
            layer_idx=0,
            attention_mask=torch.ones(b, seq, dtype=torch.bool),
        )


# --- real weights ------------------------------------------------------------


@requires_weights
def test_batched_prefill_matches_single_prompts_on_llama():
    """The same claim on the real 1B: two ragged prompts, one forward, same logits.

    A single leaked pad token here would move the logits by far more than 1e-4, and
    a mis-rotated position would move them more still, so this is the end-to-end
    version of both controls above.
    """
    from nanoserve.loader import load_weights

    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    prompts = [PROMPT_IDS, PROMPT_IDS[:3]]
    batch = pad_prompts(prompts, pad_id=cfg.vocab_size - 1)
    last = last_token_logits(model.forward_batch(batch), batch)
    for row, prompt in enumerate(prompts):
        alone = model.forward(torch.tensor([prompt]))[0, -1]
        torch.testing.assert_close(last[row], alone, atol=1e-4, rtol=1e-3)
        assert last[row].argmax().item() == alone.argmax().item()
