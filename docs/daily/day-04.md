---
title: "Day 04: RMSNorm and RoPE, position as a rotation"
parent: Daily log
nav_order: 4
---

# Day 04: RMSNorm and RoPE, position as a rotation

Date: 2026-06-19 · Week 1 · Phase 0 Foundations

## What I added today
`src/nanoserve/layers.py`: `rms_norm` (the bias-free normalizer every block uses
twice) and the RoPE stack: `RotaryEmbedding` (precomputes inverse frequencies
with the Llama-3.2 "llama3" rescaling, builds the cos/sin table per call),
`rotate_half`, and `apply_rotary`. Plus `tests/test_layers.py` (9 tests): pure
math checks that run anywhere, and four gated against the real Llama-3.2-1B.

## Why it matters
These are the first two pieces that turn loaded weights into computed
activations. RMSNorm is trivial. RoPE is where attention learns *where* a token
is, and it is the classic place to be quietly, untestably wrong, so it gets
verified hard before any attention math sits on top of it.

## What I learned
Position is a rotation, not a vector you add. RoPE rotates each (x, y) pair of a
query/key by an angle proportional to the position. A dot product only cares
about the angle *difference* between two rotated vectors, so attention between
positions m and n comes out depending on (m - n): relative position falls out of
absolute rotations for free, with no learned position table.

The Llama-3.2 wrinkle is the "llama3" frequency rescaling. Plain RoPE picks one
frequency per pair, geometrically spaced. Llama leaves the fast (short
wavelength, local) frequencies alone, divides the slow (long wavelength, global)
ones by 32, and smoothly interpolates the band between, so an 8k-pretrained model
addresses a 131k context without retraining. My test pins exactly that: pair 0
(fastest) is untouched, the slowest pair is divided by precisely `factor`.

The convention is the other trap. The cos/sin table is `cat((freqs, freqs))` and
`rotate_half` pairs dim i with dim i+32 (the GPT-NeoX layout HF uses), not the
interleaved (x0, x1) pairing you would write from the paper. Get that backwards
and short prompts still look plausible while every number is wrong. So one test
feeds identical random q/k/cos/sin into my `apply_rotary` and transformers' own
`apply_rotary_pos_emb` and demands they agree: that isolates the convention from
the weights entirely.

Best part: the match is not "within 1e-5", it is bit-exact, 0.00e+00 on RMSNorm
output, on the cos/sin table, and on inv_freq. That is the payoff of mirroring HF
op-for-op in fp32 instead of writing my own clever version: there is no rounding
to argue about, the bytes are identical.

## Diagram
[rope-frequency-bands.svg](../diagrams/rope-frequency-bands.svg) — the head_dim/2
RoPE frequencies, the three llama3 regimes (high freq untouched, medium smoothly
interpolated, low freq divided by 32), and why that buys an 8k to 131k context
stretch.

## Tomorrow
Day 5: SwiGLU MLP in `layers.py` (gate, up, down; silu(gate) * up then down),
verified within 1e-5 of the HF layer's `mlp` hook. Short one, per the buffer
rule, so I may start Day 6's GQA attention if there is time.

---
Post angle: Day 4 of building an inference engine from scratch. Today: RoPE, the
trick that tells attention where each token is. The idea that finally clicked is
that position is a rotation, not a number you add. You spin each query and key by
an angle set by its position, and because a dot product only sees the angle
between two vectors, attention ends up depending on the distance between tokens,
not their absolute spots. Relative position for free, no lookup table. Llama 3.2
adds one move on top: it leaves the fast local frequencies alone and stretches
the slow global ones by 32x, which is the whole reason a model trained on 8k
tokens can read 131k. I rebuilt it in about 40 lines and it matches Hugging Face
to the bit, 0.00 difference, because I mirrored their math step for step instead
of inventing my own. The unglamorous lesson under the cool one: most of "getting
RoPE right" is getting the pairing convention right, and the only way you know is
to put your output next to theirs and diff it.
