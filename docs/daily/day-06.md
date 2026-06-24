---
title: "Day 06: GQA attention, the one sublayer that mixes tokens"
parent: Daily log
nav_order: 6
---

# Day 06: GQA attention, the one sublayer that mixes tokens

Date: 2026-06-21 · Week 1 · Phase 0 Foundations

## What I added today
`src/nanoserve/layers.py`: `gqa_attention`, the single-block prefill attention
(Q/K/V projections, RoPE on q and k, the KV-head repeat, a causal mask, softmax,
and o_proj), plus the `repeat_kv` helper. Five new tests in `tests/test_layers.py`:
three pure-math (the contiguous KV repeat, a full attention recompute via an
independent path, and a causality trap) and one gated against the real
Llama-3.2-1B `self_attn` hook to 1e-5. Full suite is 41 green.

## Why it matters
A transformer block has two sublayers. The MLP (Day 5) transforms each token on
its own; attention is the only place where tokens look at each other. With
attention done, all four computed pieces of a block exist (RMSNorm, RoPE, SwiGLU,
attention), and tomorrow they assemble into one verified block. This is also the
sublayer the rest of the project is really about: the KV cache, continuous
batching, and the paged-attention kernel all exist to make this one operation
cheap at serving time.

## What I learned
Grouped-query attention is a memory decision wearing a math costume. Plain
multi-head attention gives every query head its own key and value head; the
problem is that at serving time you cache K and V for every past token, so the
cache grows with the number of KV heads. Llama-3.2-1B keeps 32 query heads but
only 8 KV heads, and shares each KV head across a group of 4 query heads. The
attention math is unchanged; you just store and stream 4x fewer keys and values.
That single ratio is why the rest of this engine (the paged cache, the kernel) is
even tractable, so it was worth getting the grouping exactly right.

The way GQA actually runs is almost an anticlimax: you project k and v as 8 heads,
then `repeat_kv` expands them back to 32 right before the scores, so the score
matmul is plain 32-head attention again. The only thing that matters is *which*
query heads share a KV head. They must be contiguous: query heads 0-3 share KV
head 0, 4-7 share KV head 1, and so on. The expand-then-reshape in `repeat_kv`
(insert a length-4 axis next to the head axis, then fold it in) produces exactly
that grouping; a stray `tile`/`repeat` would interleave the heads and the model
would still run while quietly attending with the wrong keys. So the first test
builds two KV heads with distinguishable values and asserts head 0 lands in output
slots 0-2 and head 1 in 3-5.

The three classic mismatch suspects all showed up as things to be careful about:

- The causal mask. Add a `-inf` upper triangle (diagonal=1) to the scores before
  softmax so position i never sees a later token. The diagonal stays unmasked, so
  no row is ever entirely `-inf` and softmax never returns NaN. I wrote a dedicated
  trap for this: perturb only the last token's input, and every earlier output row
  must be bit-identical. A flipped triangle or a missing mask fails it loudly.
- The head reshape. It has to be view-to-`[b, seq, heads, d]` then transpose to
  `[b, heads, seq, d]`. A bare reshape straight to `[b, heads, seq, d]` interleaves
  positions and heads and silently corrupts everything downstream.
- The GQA repeat itself, above.

For the pure-math equivalence test I recomputed attention by an independent path
(`repeat_interleave` for the KV repeat instead of the production `expand`+`reshape`,
identity RoPE so the rotary convention stays out of it) and matched to 1e-6. Then
the real test: feed the layer's true input (the `input_layernorm` output) through
`gqa_attention` with the loader's q/k/v/o weights and our own RoPE table, and
compare to the HF `self_attn` output. 1e-5 on the first run. The softmax-in-fp32
detail (HF upcasts even in a low-precision model, then casts back) is a no-op in
this fp32 pipeline but I mirrored it so the bf16 path matches later without
surprises.

## Diagram
[gqa-attention.svg](../diagrams/gqa-attention.svg) — one group of 4 query heads
feeding a single shared KV head, the 32-vs-8 KV-cache size comparison (the 4x
saving), and the five-line prefill math, with the three traps called out.

## Tomorrow
Day 7, the Week 1 done line: assemble the full transformer block in `layers.py` /
`model.py` (attn_norm, attention, residual, mlp_norm, SwiGLU, residual) on the
real loaded weights, and verify the whole block output within 1e-5 of HF
`model.layers[0]`. Then a Week 1 recap: the map, the weights, the four pieces, the
block.

---
Post angle: Day 6 of building an inference engine from scratch. Today: attention,
the one part of a transformer block where tokens actually look at each other.
Modern models use a memory trick called grouped-query attention. Plain attention
gives every one of the 32 query heads its own key and value, and at serving time
you have to cache a key and value for every token you have seen, so that cache
balloons. Llama-3.2-1B keeps the 32 query heads but only 8 key/value heads, and
lets each key/value head be shared by a group of 4 query heads. The math you run is
identical (you just copy each shared head back up to 32 right before the scores),
but you store and stream 4x fewer keys and values. That one ratio is the reason a
KV cache is affordable at all, which is the whole rest of this project. The part to
get right is not the math, it is the bookkeeping: which query heads share a head,
that the causal mask actually stops a token from seeing the future, and that
splitting the projection into heads does not scramble positions and heads together.
It matches Hugging Face to 1e-5, and I wrote a test that pokes a future token and
demands every earlier output stay frozen, to prove the mask really is causal.
