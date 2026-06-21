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
    gqa_attention,
    repeat_kv,
    rms_norm,
    rotate_half,
    swiglu,
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


# --- SwiGLU: pure math ------------------------------------------------------


def test_swiglu_matches_manual_definition():
    """swiglu == down(silu(gate(x)) * up(x)), with bias-free [out, in] weights."""
    torch.manual_seed(0)
    hidden, inter = 8, 20
    x = torch.randn(3, hidden)
    gate_w = torch.randn(inter, hidden)
    up_w = torch.randn(inter, hidden)
    down_w = torch.randn(hidden, inter)

    silu = lambda t: t * torch.sigmoid(t)  # noqa: E731
    expected = (silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T

    assert torch.allclose(swiglu(x, gate_w, up_w, down_w), expected, atol=1e-6)


def test_swiglu_gate_branch_is_not_symmetric():
    """Guard against swapping gate/up: SiLU is on the gate branch only.

    SiLU is nonlinear, so silu(gate)*up != silu(up)*gate in general. If a refactor
    ever swaps the two branches, this random case will disagree and fail.
    """
    torch.manual_seed(1)
    hidden, inter = 8, 20
    x = torch.randn(3, hidden)
    gate_w = torch.randn(inter, hidden)
    up_w = torch.randn(inter, hidden)
    down_w = torch.randn(hidden, inter)

    straight = swiglu(x, gate_w, up_w, down_w)
    swapped = swiglu(x, up_w, gate_w, down_w)  # gate/up exchanged
    assert not torch.allclose(straight, swapped, atol=1e-4)


# --- GQA attention: pure math -----------------------------------------------


def _tiny_attn_config() -> ModelConfig:
    """A small but structurally real GQA config: 8 query heads, 2 KV heads."""
    return ModelConfig(
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=2,  # repeat factor 4, same ratio as the 1B model
        head_dim=4,
    )


def _identity_rope(batch: int, seq: int, head_dim: int):
    """cos=1, sin=0 makes `apply_rotary` the identity, isolating the attn math."""
    return torch.ones(batch, seq, head_dim), torch.zeros(batch, seq, head_dim)


def test_repeat_kv_repeats_each_head_into_a_contiguous_group():
    # Two KV heads with distinguishable constant values; repeat 3x.
    x = torch.stack(
        [torch.full((1, 2, 4), 1.0), torch.full((1, 2, 4), 2.0)], dim=1
    )  # [batch=1, n_kv=2, seq=2, d=4]
    out = repeat_kv(x, 3)
    assert out.shape == (1, 6, 2, 4)
    # KV head 0 -> output heads 0,1,2 ; KV head 1 -> output heads 3,4,5.
    assert torch.all(out[:, 0:3] == 1.0)
    assert torch.all(out[:, 3:6] == 2.0)


def test_repeat_kv_noop_when_factor_is_one():
    x = torch.randn(2, 8, 5, 4)
    assert repeat_kv(x, 1) is x


def test_gqa_attention_matches_manual_reference():
    """Full attention recompute via an independent path (repeat_interleave).

    Identity RoPE so this isolates projections, the GQA repeat, the causal mask,
    softmax/scaling, and o_proj from the rotary convention (which has its own
    test). The reference uses `repeat_interleave`, a different implementation of
    the KV repeat than the production `expand`+`reshape`, so they cross-check.
    """
    torch.manual_seed(0)
    cfg = _tiny_attn_config()
    b, seq = 1, 5
    n_q, n_kv, d, h = (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.hidden_size,
    )
    x = torch.randn(b, seq, h)
    q_w = torch.randn(n_q * d, h)
    k_w = torch.randn(n_kv * d, h)
    v_w = torch.randn(n_kv * d, h)
    o_w = torch.randn(h, n_q * d)
    cos, sin = _identity_rope(b, seq, d)

    mine = gqa_attention(x, q_w, k_w, v_w, o_w, cos, sin, cfg)

    # Independent reference.
    q = (x @ q_w.T).view(b, seq, n_q, d).transpose(1, 2)
    k = (x @ k_w.T).view(b, seq, n_kv, d).transpose(1, 2)
    v = (x @ v_w.T).view(b, seq, n_kv, d).transpose(1, 2)
    k = k.repeat_interleave(cfg.num_kv_groups, dim=1)
    v = v.repeat_interleave(cfg.num_kv_groups, dim=1)
    scores = (q @ k.transpose(2, 3)) / (d**0.5)
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
    weights = torch.softmax(scores + mask, dim=-1)
    out = (weights @ v).transpose(1, 2).reshape(b, seq, h) @ o_w.T

    assert torch.allclose(mine, out, atol=1e-6)


def test_gqa_attention_is_causal():
    """Output at position i must not depend on any token after i.

    Perturb the last token's input and assert every earlier output row is
    unchanged. A flipped mask triangle (or no mask) fails this loudly.
    """
    torch.manual_seed(1)
    cfg = _tiny_attn_config()
    b, seq, h, d = 1, 5, cfg.hidden_size, cfg.head_dim
    x = torch.randn(b, seq, h)
    weights = [
        torch.randn(cfg.num_attention_heads * d, h),
        torch.randn(cfg.num_key_value_heads * d, h),
        torch.randn(cfg.num_key_value_heads * d, h),
        torch.randn(h, cfg.num_attention_heads * d),
    ]
    cos, sin = _identity_rope(b, seq, d)

    out = gqa_attention(x, *weights, cos, sin, cfg)
    x2 = x.clone()
    x2[:, -1] += 10.0  # clobber the final token only
    out2 = gqa_attention(x2, *weights, cos, sin, cfg)

    # Rows 0..seq-2 see only the past, so they are untouched; the last row moves.
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out[:, -1], out2[:, -1], atol=1e-4)


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


@requires_weights
def test_swiglu_matches_hf_layer0_mlp():
    """SwiGLU matches layer 0's `mlp` hook on its real input, to 1e-5.

    Capture the `post_attention_layernorm` output (the MLP's true input) and the
    `mlp` output, then reproduce the second from the first with our `swiglu` and
    the loader's gate/up/down weights.
    """
    from nanoserve.loader import load_weights

    hf = hf_model()
    acts = {}

    def grab(name):
        def hook(_m, _inp, out):
            acts[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    h1 = hf.model.layers[0].post_attention_layernorm.register_forward_hook(grab("mlp_in"))
    h2 = hf.model.layers[0].mlp.register_forward_hook(grab("mlp_out"))
    try:
        with torch.no_grad():
            hf(torch.tensor([PROMPT_IDS]))
    finally:
        h1.remove()
        h2.remove()

    w = load_weights(WEIGHTS_DIR)
    mine = swiglu(
        acts["mlp_in"],
        w["layers.0.mlp.gate_proj.weight"],
        w["layers.0.mlp.up_proj.weight"],
        w["layers.0.mlp.down_proj.weight"],
    )
    assert torch.allclose(mine, acts["mlp_out"], atol=1e-5)


@requires_weights
def test_gqa_attention_matches_hf_layer0_self_attn():
    """GQA attention matches layer 0's `self_attn` hook on its real input, 1e-5.

    The attention sublayer's true input is the `input_layernorm` (attn_norm)
    output, and its output (out[0]) is post o_proj, before the residual add. We
    capture both, build the RoPE tables for the prompt's positions with our own
    RotaryEmbedding (already pinned to HF's rotary), and reproduce the second
    tensor from the first with `gqa_attention` and the loader's q/k/v/o weights.
    """
    from nanoserve.loader import load_weights

    cfg = ModelConfig.from_json(WEIGHTS_DIR)
    hf = hf_model()
    acts = {}

    def grab(name):
        def hook(_m, _inp, out):
            acts[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    h1 = hf.model.layers[0].input_layernorm.register_forward_hook(grab("attn_in"))
    h2 = hf.model.layers[0].self_attn.register_forward_hook(grab("attn_out"))
    try:
        with torch.no_grad():
            hf(torch.tensor([PROMPT_IDS]))
    finally:
        h1.remove()
        h2.remove()

    rope = RotaryEmbedding(cfg)
    position_ids = torch.arange(len(PROMPT_IDS))[None]  # [1, seq]
    cos, sin = rope.cos_sin(position_ids)

    w = load_weights(WEIGHTS_DIR)
    mine = gqa_attention(
        acts["attn_in"],
        w["layers.0.attn.q_proj.weight"],
        w["layers.0.attn.k_proj.weight"],
        w["layers.0.attn.v_proj.weight"],
        w["layers.0.attn.o_proj.weight"],
        cos,
        sin,
        cfg,
    )
    assert torch.allclose(mine, acts["attn_out"], atol=1e-5)
