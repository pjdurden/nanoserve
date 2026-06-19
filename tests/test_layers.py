"""Day 4 tests: RMSNorm and RoPE, verified against the HF reference.

Two tiers, same as the loader tests:
  - Pure math tests run anywhere (torch only): the RMSNorm formula, the
    rotate-half convention, and that the llama3 rescaling touches the right
    frequencies. The RoPE-apply test pins our convention to transformers' own
    `apply_rotary_pos_emb` on random inputs, so a flipped sign or interleaved
    pairing fails loudly without needing the gated weights.
  - `requires_weights` tests compare to the real Llama-3.2-1B: RMSNorm against
    the `input_layernorm` hook, and the cos/sin table + inv_freq against the
    model's own `rotary_emb`, all to ~1e-5 or tighter.
"""

from __future__ import annotations

import torch

from nanoserve.config import ModelConfig
from nanoserve.layers import (
    RotaryEmbedding,
    apply_rotary,
    rms_norm,
    rotate_half,
)

from reference import PROMPT_IDS, WEIGHTS_DIR, hf_model, requires_weights

CONFIG = ModelConfig()


# --- RMSNorm: pure math -----------------------------------------------------


def test_rms_norm_matches_manual():
    # x = [3, 4], rms = sqrt((9+16)/2) = sqrt(12.5) = 3.5355..., eps tiny.
    x = torch.tensor([[3.0, 4.0]])
    w = torch.ones(2)
    out = rms_norm(x, w, eps=0.0)
    rms = (12.5) ** 0.5
    expected = x / rms
    assert torch.allclose(out, expected, atol=1e-6)


def test_rms_norm_unit_weight_gives_unit_rms():
    # With weight=1 and eps->0, the output's own RMS is 1 by construction.
    torch.manual_seed(0)
    x = torch.randn(4, 2048) * 7.0 + 2.0  # arbitrary scale/shift
    out = rms_norm(x, torch.ones(2048), eps=1e-6)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-3)


def test_rms_norm_applies_per_channel_gain():
    x = torch.ones(1, 4)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # normalized x is all-ones (rms of ones is 1), so output == weight.
    out = rms_norm(x, w, eps=0.0)
    assert torch.allclose(out, w.unsqueeze(0), atol=1e-6)


# --- RoPE: pure math --------------------------------------------------------


def test_rotate_half_swaps_and_negates():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # halves [1,2] and [3,4]
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_matches_hf_reference():
    """Pin our rotate-half apply to transformers' own implementation.

    Same random q/k/cos/sin into both; any disagreement is a convention bug
    (sign, half-vs-interleaved, broadcast axis), independent of the weights.
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb as hf_apply,
    )

    torch.manual_seed(0)
    b, n_q, n_kv, seq, d = 2, 32, 8, 5, 64
    q = torch.randn(b, n_q, seq, d)
    k = torch.randn(b, n_kv, seq, d)
    cos = torch.randn(b, seq, d)
    sin = torch.randn(b, seq, d)

    q_mine, k_mine = apply_rotary(q, k, cos, sin)
    q_hf, k_hf = hf_apply(q, k, cos, sin)
    assert torch.allclose(q_mine, q_hf, atol=1e-6)
    assert torch.allclose(k_mine, k_hf, atol=1e-6)


def test_llama3_rescaling_touches_only_low_frequencies():
    """The llama3 rescaling stretches long wavelengths, leaves short ones alone.

    Build inv_freq with and without the scaling and check: high-frequency pairs
    (short wavelength) are untouched, and the lowest-frequency pair is divided by
    exactly `factor`.
    """
    from nanoserve.layers import _default_inv_freq, _llama3_inv_freq

    plain = _default_inv_freq(CONFIG.head_dim, CONFIG.rope_theta)
    scaled = _llama3_inv_freq(plain, CONFIG.rope_scaling)

    # Pair 0 is the fastest rotation (shortest wavelength): untouched.
    assert torch.allclose(scaled[0], plain[0])
    # The slowest pair has the longest wavelength: divided by `factor`.
    assert torch.allclose(scaled[-1], plain[-1] / CONFIG.rope_scaling.factor)
    # Scaling never speeds a frequency up.
    assert torch.all(scaled <= plain + 1e-12)


# --- against the real Llama-3.2-1B ------------------------------------------


@requires_weights
def test_inv_freq_matches_hf_buffer():
    """Our llama3-rescaled inv_freq equals the model's own rotary buffer."""
    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    rope = RotaryEmbedding(cfg)
    hf = hf_model()
    assert torch.allclose(rope.inv_freq, hf.model.rotary_emb.inv_freq, atol=1e-9)


@requires_weights
def test_cos_sin_matches_hf_rotary():
    """cos/sin table matches `model.rotary_emb` for the fixed prompt positions."""
    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    rope = RotaryEmbedding(cfg)
    position_ids = torch.arange(len(PROMPT_IDS))[None]  # [1, seq]

    cos, sin = rope.cos_sin(position_ids)
    hf = hf_model()
    x = torch.zeros(1, len(PROMPT_IDS), cfg.hidden_size)  # only x.dtype/device used
    cos_hf, sin_hf = hf.model.rotary_emb(x, position_ids)
    assert torch.allclose(cos, cos_hf, atol=1e-6)
    assert torch.allclose(sin, sin_hf, atol=1e-6)


@requires_weights
def test_rmsnorm_matches_hf_input_layernorm():
    """RMSNorm on the embedding output matches layer 0's `input_layernorm` hook.

    Capture both the embedding output (the layernorm's input) and the layernorm
    output, then reproduce the second from the first with our `rms_norm` and the
    loader's weight, to 1e-5.
    """
    from nanoserve.loader import load_weights

    hf = hf_model()
    acts = {}

    def grab(name):
        def hook(_m, _inp, out):
            acts[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    h1 = hf.model.embed_tokens.register_forward_hook(grab("embed"))
    h2 = hf.model.layers[0].input_layernorm.register_forward_hook(grab("ln0"))
    try:
        with torch.no_grad():
            hf(torch.tensor([PROMPT_IDS]))
    finally:
        h1.remove()
        h2.remove()

    weights = load_weights(WEIGHTS_DIR)
    w = weights["layers.0.attn_norm.weight"]
    mine = rms_norm(acts["embed"], w, eps=weights.config.rms_norm_eps)
    assert torch.allclose(mine, acts["ln0"], atol=1e-5)
