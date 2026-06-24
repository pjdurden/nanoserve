"""Day 8 tests: the full forward pass, verified against the HF reference.

Two tiers, the same shape as the layer tests:
  - Pure tests run anywhere (torch only): the forward pass produces the right
    shape, and with both sublayers of every block zeroed the model collapses to
    `lm_head(norm(embed(ids)))`, which pins the embedding, the final norm, and
    the tied LM head independently of the block math.
  - `requires_weights` tests compare to the real Llama-3.2-1B: the full logits to
    ~1e-4, and the greedy next token exactly (the Week 2 token-for-token goal).
"""

from __future__ import annotations

import torch

from nanoserve.config import ModelConfig
from nanoserve.layers import rms_norm
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes, load_weights
from nanoserve.model import LlamaModel

from reference import PROMPT_IDS, WEIGHTS_DIR, hf_model, requires_weights


def _tiny_config() -> ModelConfig:
    """A small but structurally real config: GQA ratio 4, a couple of layers."""
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
    """A full, shape-correct random Weights bag with the LM head tied to embed.

    Reuses the loader's own `expected_shapes`, so this can never drift from the
    real tensor set, and aliases lm_head -> embed exactly as a tied model loads.
    """
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, cfg)


# --- pure: shape and wiring -------------------------------------------------


def test_forward_returns_logits_over_vocab():
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    b, seq = 2, 5
    ids = torch.randint(0, cfg.vocab_size, (b, seq))
    logits = model.forward(ids)
    assert logits.shape == (b, seq, cfg.vocab_size)


def test_forward_collapses_to_embed_norm_head_when_blocks_zeroed():
    """Zero every block's two output projections so each block is the identity.

    With `o_proj` and `down_proj` zeroed, both residual branches contribute 0, so
    every block returns its input unchanged and the whole stack is a no-op. The
    forward pass then reduces to exactly `lm_head(norm(embed(ids)))`. Reproducing
    that by hand pins three things the block tests never touch: the embedding
    lookup, the presence of the final norm, and the tied LM head. Forget the
    final norm and this fails; copy embed into a second lm_head tensor and the
    tie test below still passes but this still holds (same values), so this is
    the wiring check and the next test is the aliasing check.
    """
    cfg = _tiny_config()
    weights = _random_weights(cfg)
    for i in range(cfg.num_hidden_layers):
        weights[f"layers.{i}.attn.o_proj.weight"].zero_()
        weights[f"layers.{i}.mlp.down_proj.weight"].zero_()

    model = LlamaModel(cfg, weights)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    logits = model.forward(ids)

    embed = torch.nn.functional.embedding(ids, weights[EMBED])
    normed = rms_norm(embed, weights["norm.weight"], cfg.rms_norm_eps)
    expected = normed @ weights[LM_HEAD].T
    assert torch.allclose(logits, expected, atol=1e-5)


def test_lm_head_is_tied_to_embedding():
    """The output projection must be the input embedding matrix, not a copy."""
    cfg = _tiny_config()
    weights = _random_weights(cfg)
    assert weights[LM_HEAD].data_ptr() == weights[EMBED].data_ptr()


# --- pure: the greedy decode loop -------------------------------------------


def test_greedy_generate_appends_max_new_tokens():
    """With no eos, the loop grows the prompt by exactly `max_new_tokens`."""
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.greedy_generate(ids, max_new_tokens=7)
    assert out.shape == (1, 4 + 7)
    # The prompt is preserved as the prefix; only the tail is new.
    assert torch.equal(out[:, :4], ids)


def test_greedy_generate_is_deterministic_and_matches_step_by_step():
    """The loop is just `greedy_token` appended; run twice, get the same tokens.

    Also pins that each generated token is the argmax `greedy_token` would pick
    at that step, i.e. the loop adds nothing beyond append-and-recompute.
    """
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 4))

    a = model.greedy_generate(ids, max_new_tokens=5)
    b = model.greedy_generate(ids, max_new_tokens=5)
    assert torch.equal(a, b)

    manual = ids
    for _ in range(5):
        manual = torch.cat([manual, model.greedy_token(manual)[:, None]], dim=1)
    assert torch.equal(a, manual)


def test_greedy_generate_stops_at_eos_and_keeps_it():
    """Passing the first token it would emit as eos stops it after one token.

    The emitted eos is kept in the output (length grows by exactly 1), matching
    what HF `generate` returns, rather than being trimmed.
    """
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (1, 4))

    first = model.greedy_token(ids).item()
    out = model.greedy_generate(ids, max_new_tokens=10, eos_id=first)
    assert out.shape == (1, 5)
    assert out[0, -1].item() == first


def test_greedy_generate_rejects_a_real_batch():
    """One sequence only until Phase 3; a batch dim > 1 is an explicit error."""
    cfg = _tiny_config()
    model = LlamaModel(cfg, _random_weights(cfg))
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    try:
        model.greedy_generate(ids, max_new_tokens=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for batch > 1")


# --- against the real Llama-3.2-1B ------------------------------------------


@requires_weights
def test_forward_logits_match_hf():
    """Full-stack logits match HF `LlamaForCausalLM` on the fixed prompt to ~1e-4.

    Sixteen blocks of accumulated float error sit between this and the per-block
    1e-5 check, so the tolerance loosens to 1e-4; the greedy-token test below is
    the exact, tolerance-free correctness signal.
    """
    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    ids = torch.tensor([PROMPT_IDS])

    mine = model.forward(ids)
    hf = hf_model()
    with torch.no_grad():
        ref = hf(ids).logits
    assert torch.allclose(mine, ref, atol=1e-4)


@requires_weights
def test_greedy_next_token_matches_hf():
    """The argmax next token equals HF's, exactly. The Week 2 north star, seeded.

    Logit values can drift at 1e-4; the *choice* of token must not. This is the
    first token of greedy decode agreeing with the reference, which is what Week 2
    extends to a full token-for-token sequence once a cache makes it affordable.
    """
    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    ids = torch.tensor([PROMPT_IDS])

    mine = model.greedy_token(ids)
    hf = hf_model()
    with torch.no_grad():
        ref = hf(ids).logits[:, -1].argmax(dim=-1)
    assert torch.equal(mine, ref)


@requires_weights
def test_greedy_generate_matches_hf_multi_token():
    """The whole greedy continuation matches HF token for token. Week 2 north star.

    Day 8 proved one step; this is the loop. nanoserve's no-cache greedy decode
    must produce the exact same multi-token continuation as HF `generate` with
    sampling off. HF uses its own KV cache internally and nanoserve recomputes
    the prefix every step, so this also confirms the two paths are numerically
    close enough that the argmax never diverges across the whole run.

    Note: this loads two fp32 copies of the 1B model. On a memory-tight box run it
    alone, or verify via the save-free-reload path in the day-09 log (run HF,
    save its ids, free it, then load nanoserve), which peaks at one model.
    """
    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(cfg, load_weights(WEIGHTS_DIR))
    ids = torch.tensor([PROMPT_IDS])
    n = 20

    mine = model.greedy_generate(ids, max_new_tokens=n)
    hf = hf_model()
    with torch.no_grad():
        ref = hf.generate(ids, max_new_tokens=n, do_sample=False, use_cache=True)
    assert torch.equal(mine, ref)
