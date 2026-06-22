"""The building blocks: RMSNorm, RoPE, GQA attention, SwiGLU MLP. Weeks 1-2.

Verify each piece against the HF reference to about 1e-5 before moving on. The
common mismatch sources are RoPE application, the causal mask, and dtype.

Day 4 lands the two position/normalization pieces:

- RMSNorm: the cheap, bias-free normalizer every Llama block uses twice.
- RoPE: rotary position embedding, with the Llama-3.2 "llama3" frequency
  rescaling. This is the piece people get subtly wrong, so it is built to mirror
  HuggingFace term-for-term and verified against `model.rotary_emb` to ~1e-6.

Day 5 adds the block's other sublayer:

- SwiGLU MLP: the gated feed-forward (`down(silu(gate(x)) * up(x))`), verified
  against the HF `LlamaMLP` activation to ~1e-5.

Day 6 adds the block's first sublayer:

- GQA attention: the plain prefill math (Q/K/V projections, RoPE on q/k, the
  KV-head repeat, a causal mask, softmax, output projection), verified against
  the HF `self_attn` activation to ~1e-5. No KV cache yet; that is Week 2.

Everything here is a plain function or a small helper class, not an `nn.Module`.
nanoserve never trains, so there is nothing to register; the layer modules in
Weeks 1-2 just call these with weights pulled from the loader by name.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import ModelConfig

# --- RMSNorm ----------------------------------------------------------------
#
# Root-mean-square layernorm: scale each token vector by 1/rms, then a learned
# per-channel gain. No mean-subtraction and no bias, which is the whole point.
# HF computes the statistic in fp32 even when the model runs in bf16, then casts
# back before applying the gain; we mirror that exactly so the bf16 path will
# match later without surprises. In the fp32 Week-1 pipeline the cast is a no-op.


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm to the last dimension of `x`.

    x:      [..., hidden] activations.
    weight: [hidden] learned gain (the `*_norm.weight` tensors from the loader).
    eps:    `config.rms_norm_eps`, added under the sqrt for numerical safety.

    Returns a tensor of the same shape and dtype as `x`.
    """
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


# --- RoPE: rotary position embedding ----------------------------------------
#
# RoPE encodes a token's position by *rotating* its query/key vectors, in
# (x, y) pairs, by an angle proportional to the position. Because a dot product
# is rotation-invariant in a way that depends only on the angle *difference*,
# attention between positions m and n ends up depending on (m - n): relative
# position falls out of absolute rotations for free. That is why position is a
# rotation here, not a vector you add.
#
# Two details make this Llama-3.2 and not textbook RoPE:
#   1. The "llama3" frequency rescaling (`_llama3_inv_freq` below). Low
#      frequencies (long wavelengths) are stretched by `factor` so the 8k
#      pretraining context generalizes to 131k, with a smooth interpolation band
#      in between. Skip this and short prompts still match but long-context
#      positions drift from HF.
#   2. The rotate-half layout. The cos/sin table is `cat((freqs, freqs))`, so
#      dim i and dim i+half share an angle, and `rotate_half` pairs them. This
#      is the GPT-NeoX convention HF uses, not the interleaved (x0,x1) pairing.


def _default_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """Plain RoPE inverse frequencies: theta^(-2i/d) for i in [0, d/2).

    One frequency per rotated pair (head_dim // 2 of them). Index 0 is the
    fastest-rotating pair, the last is the slowest.
    """
    idx = torch.arange(0, head_dim, 2, dtype=torch.int64).float()
    return 1.0 / (theta ** (idx / head_dim))


def _llama3_inv_freq(inv_freq: torch.Tensor, scaling) -> torch.Tensor:
    """Apply the Llama-3.2 'llama3' rescaling to plain inverse frequencies.

    Mirrors transformers' `_compute_llama3_parameters` term-for-term:
      - wavelengths shorter than `high_freq_wavelen`: untouched (local detail).
      - wavelengths longer than `low_freq_wavelen`: divided by `factor` (so the
        slowest rotations cover a `factor`x longer context).
      - in between: a smooth blend between the two regimes.
    """
    factor = scaling.factor
    low_freq_factor = scaling.low_freq_factor
    high_freq_factor = scaling.high_freq_factor
    old_context_len = scaling.original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # Long wavelengths get stretched; short ones pass through unchanged.
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # Smooth interpolation for the medium band.
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_llama)


class RotaryEmbedding:
    """Precomputes inverse frequencies once; builds the cos/sin table per call.

    Construct from a `ModelConfig` so the head_dim, theta, and llama3 scaling all
    come from the same place as the weights. `cos_sin(position_ids)` returns the
    tables `apply_rotary` needs; it is cheap (one outer product) and position-ids
    driven, so it works for both a contiguous prefill and the scattered
    positions of a continuous-batching decode step later.
    """

    def __init__(self, config: ModelConfig):
        inv_freq = _default_inv_freq(config.head_dim, config.rope_theta)
        if config.rope_scaling is not None and config.rope_scaling.rope_type == "llama3":
            inv_freq = _llama3_inv_freq(inv_freq, config.rope_scaling)
        self.inv_freq = inv_freq  # [head_dim // 2]
        # llama3 leaves the cos/sin amplitude alone; other rope types may scale.
        self.attention_scaling = 1.0
        self.head_dim = config.head_dim

    def cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (cos, sin) for the given positions.

        position_ids: [batch, seq] integer positions.
        Returns cos, sin each [batch, seq, head_dim], computed in fp32 (HF forces
        fp32 here regardless of model dtype, because small angle errors compound).
        """
        # outer product positions x frequencies -> [batch, seq, head_dim/2]
        freqs = position_ids[..., None].float() * self.inv_freq.to(position_ids.device)
        # duplicate so dims i and i+half share an angle (rotate-half layout)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair dim i with dim i+half: return [-x_second_half, x_first_half].

    This is the GPT-NeoX / HF Llama convention, paired with the `cat((f, f))`
    cos/sin table above. Together they implement, per pair, the 2D rotation
    (x, y) -> (x cos - y sin, x sin + y cos).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query and key by the RoPE angles in `cos`/`sin`.

    q, k:        [batch, heads, seq, head_dim] (q has 32 heads, k has 8 for GQA).
    cos, sin:    [batch, seq, head_dim] from `RotaryEmbedding.cos_sin`.
    unsqueeze_dim: the head axis to broadcast cos/sin over (1 for [b, h, s, d]).

    Returns the rotated (q, k), same shapes. Matches HF `apply_rotary_pos_emb`.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# --- SwiGLU MLP -------------------------------------------------------------
#
# The block's second sublayer. A plain MLP would be `down(act(up(x)))` with one
# projection up to the wide intermediate dim and one back. SwiGLU splits the
# up-projection in two: a `gate` branch that is passed through SiLU, and an `up`
# branch that is not, and it multiplies them elementwise before projecting down:
#
#     down( silu(gate(x)) * up(x) )
#
# The gate is a learned, input-dependent valve on the up branch (this is the
# "gated linear unit" idea); SiLU is the smooth gate nonlinearity. The cost is a
# third matrix, which is why Llama sizes the intermediate dim at ~2.7x hidden
# (8192 for this 2048-hidden model) rather than the classic 4x.
#
# The one thing to get right: SiLU goes on the *gate* branch only, and the two
# branches are multiplied, not added. Swap the branches or move the nonlinearity
# and short prompts still look plausible while the logits quietly drift off HF.
#
# Llama's projections are bias-free, so each branch is a single matmul. Weights
# arrive [out, in] (the torch/HF convention), exactly what `F.linear` wants, so
# there is no transpose to fumble here.


def swiglu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the SwiGLU MLP to the last dimension of `x`.

    x:           [..., hidden] activations (the `mlp_norm` output in a block).
    gate_weight: [intermediate, hidden] `mlp.gate_proj.weight` from the loader.
    up_weight:   [intermediate, hidden] `mlp.up_proj.weight`.
    down_weight: [hidden, intermediate] `mlp.down_proj.weight`.

    Returns a tensor of the same shape as `x`. Matches HF `LlamaMLP.forward`.
    """
    gate = F.silu(F.linear(x, gate_weight))
    up = F.linear(x, up_weight)
    return F.linear(gate * up, down_weight)


# --- GQA attention ----------------------------------------------------------
#
# The block's first sublayer, and the only one that mixes information *across*
# tokens. Grouped-query attention (GQA) is the memory trick that makes the KV
# cache affordable: instead of one key/value head per query head, Llama-3.2-1B
# has 32 query heads but only 8 KV heads, and each KV head is shared by a group
# of 4 query heads. Fewer KV heads means a 4x smaller KV cache to store and
# stream per token, which is the whole reason this arc ends in a paged cache.
#
# This is the plain prefill math: the whole prompt at once, no cache yet. The
# steps mirror HF `LlamaAttention.forward` term-for-term so the 1e-5 check is
# easy to reason about:
#   1. Project x to q (32 heads), k, v (8 heads each), and split into heads.
#   2. Rotate q and k by RoPE (Day 4 plugs in right here).
#   3. Repeat the 8 KV heads 4x so every query head has a key/value to attend to.
#   4. scores = q . k^T / sqrt(head_dim), add a causal mask, softmax (in fp32).
#   5. Weighted sum of v, concat the heads back, project out with o_proj.
#
# The three places this goes subtly wrong, in order of how often: the causal
# mask (off-by-one or wrong triangle), the head reshape (view-then-transpose,
# not a bare reshape that interleaves heads and positions), and the GQA repeat
# (each KV head must map to a *contiguous* group of query heads, which is what
# the expand-then-reshape below guarantees).


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat each KV head `n_rep` times to line up with the query heads.

    x:     [batch, num_kv_heads, seq, head_dim].
    n_rep: query heads per KV head (`config.num_kv_groups`, 4 here).

    Returns [batch, num_kv_heads * n_rep, seq, head_dim], where KV head h becomes
    output heads [h*n_rep : (h+1)*n_rep]. That contiguous grouping is the GQA
    contract: query heads 0-3 share KV head 0, 4-7 share KV head 1, and so on.
    Matches HF `repeat_kv` (expand a new axis, then fold it into the head axis),
    which is a view, not a copy, so it costs no extra memory.
    """
    batch, num_kv_heads, seq, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq, head_dim)


def gqa_attention(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
) -> torch.Tensor:
    """Single-block grouped-query attention over a full prompt (prefill, no cache).

    x:           [batch, seq, hidden] activations (the `attn_norm` output).
    q_weight:    [n_heads*head_dim, hidden] `attn.q_proj.weight` from the loader.
    k_weight:    [n_kv*head_dim, hidden]    `attn.k_proj.weight`.
    v_weight:    [n_kv*head_dim, hidden]    `attn.v_proj.weight`.
    o_weight:    [hidden, n_heads*head_dim] `attn.o_proj.weight`.
    cos, sin:    [batch, seq, head_dim] RoPE tables from `RotaryEmbedding.cos_sin`.
    config:      supplies the head counts, head_dim, and GQA repeat factor.

    Returns [batch, seq, hidden]. Matches HF `LlamaAttention.forward` output
    (post o_proj, before the residual add) to ~1e-5.
    """
    batch, seq, _ = x.shape
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    d = config.head_dim

    # Project, then split the flat projection into heads. The view-then-transpose
    # order matters: reshape to [b, seq, heads, d] first so each head owns a
    # contiguous slice, *then* move the head axis up to [b, heads, seq, d].
    q = F.linear(x, q_weight).view(batch, seq, n_q, d).transpose(1, 2)
    k = F.linear(x, k_weight).view(batch, seq, n_kv, d).transpose(1, 2)
    v = F.linear(x, v_weight).view(batch, seq, n_kv, d).transpose(1, 2)

    # RoPE rotates q and k by position before they ever meet (Day 4).
    q, k = apply_rotary(q, k, cos, sin)

    # Grow 8 KV heads to 32 so every query head has a partner.
    k = repeat_kv(k, config.num_kv_groups)
    v = repeat_kv(v, config.num_kv_groups)

    # Scaled dot-product scores, then a causal mask so position i cannot see j>i.
    # The mask is added before softmax; -inf entries become exactly zero weight.
    # The diagonal is always unmasked, so no row is ever fully -inf (no NaN).
    scaling = d**-0.5
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    causal = torch.full((seq, seq), float("-inf"), dtype=scores.dtype, device=scores.device)
    causal = torch.triu(causal, diagonal=1)
    scores = scores + causal

    # HF computes the softmax in fp32 even in a lower-precision model, then casts
    # back; we mirror it so the bf16 path matches later. In fp32 it is a no-op.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # Weighted sum of values, then undo the head split: [b, h, seq, d] ->
    # [b, seq, h, d] -> [b, seq, hidden], the reverse of the projection reshape.
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2).reshape(batch, seq, n_q * d)
    return F.linear(out, o_weight)


# --- the transformer block --------------------------------------------------
#
# Day 7, the Week 1 done line. The four computed pieces above are all that a
# Llama block contains; this is the wiring that turns them into one. A block has
# two sublayers (attention, then the MLP), and the only structural decisions are
# *where the norm goes* and *what the residual skips*.
#
# Llama is pre-norm: each sublayer normalizes its input, and the residual adds
# the sublayer's output back to the *un-normalized* input. So the norm sits
# inside the residual branch, not on the highway:
#
#     x = x + attention( attn_norm(x) )
#     x = x + swiglu(    mlp_norm(x)  )
#
# Two things are easy to get subtly wrong and both still "run":
#   1. The residual must add back the input to the sublayer, i.e. the value of
#      `x` *before* the norm, never the normalized tensor. Add back the norm
#      output instead and the residual highway no longer carries the raw signal.
#   2. Each sublayer has its own norm with its own weights (attn_norm vs
#      mlp_norm). Reusing one for both passes shapes fine and quietly drifts.
#
# There is no new math here, so there is no new 1e-5 risk from a formula; the
# whole risk is in the dataflow. That is why the test feeds the real layer-0
# input through this and checks the block output against HF `model.layers[0]`.


def transformer_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    config: ModelConfig,
) -> torch.Tensor:
    """One pre-norm Llama decoder block: attention sublayer, then MLP sublayer.

    x:       [batch, seq, hidden] activations (the block's input; for block 0
             this is the token embeddings).
    weights: the block's tensors keyed by in-block name, exactly what
             `Weights.layer(i)` returns: `attn_norm.weight`, `attn.q_proj.weight`
             and the rest of `attn.*`, `mlp_norm.weight`, and `mlp.*`.
    cos, sin: [batch, seq, head_dim] RoPE tables from `RotaryEmbedding.cos_sin`.
    config:  supplies rms_norm_eps and the head geometry for attention.

    Returns [batch, seq, hidden]. Matches HF `LlamaDecoderLayer.forward` output
    (out[0], the hidden state) to ~1e-5.
    """
    eps = config.rms_norm_eps

    # Attention sublayer: norm inside the branch, residual over the raw input.
    residual = x
    h = rms_norm(x, weights["attn_norm.weight"], eps)
    h = gqa_attention(
        h,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.o_proj.weight"],
        cos,
        sin,
        config,
    )
    x = residual + h

    # MLP sublayer: its own norm (mlp_norm, not attn_norm), again inside the branch.
    residual = x
    h = rms_norm(x, weights["mlp_norm.weight"], eps)
    h = swiglu(
        h,
        weights["mlp.gate_proj.weight"],
        weights["mlp.up_proj.weight"],
        weights["mlp.down_proj.weight"],
    )
    return residual + h


# TODO(week2): stack blocks in model.py (embed -> blocks -> norm -> lm_head)
