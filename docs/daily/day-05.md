---
title: "Day 05: SwiGLU, the gated MLP every modern model uses"
parent: Daily log
nav_order: 5
---

# Day 05: SwiGLU, the gated MLP every modern model uses

Date: 2026-06-20 · Week 1 · Phase 0 Foundations

## What I added today
`src/nanoserve/layers.py`: `swiglu`, the block's feed-forward sublayer
(`down(silu(gate(x)) * up(x))`), three bias-free matmuls. Plus three tests in
`tests/test_layers.py`: two pure-math (the definition, and a guard that the gate
and up branches are not interchangeable) and one gated against the real
Llama-3.2-1B `mlp` hook to 1e-5. Full suite is 36 green.

## Why it matters
A transformer block is two sublayers: attention, which mixes information across
tokens, and the MLP, which transforms each token on its own. The MLP is where
most of a model's parameters and most of its raw compute live. Day 4 did the
normalizer; this is the second computed piece. After tomorrow's attention, all
four ingredients of a block exist.

## What I learned
A plain MLP is `down(act(up(x)))`: project up to a wide hidden dim, apply a
nonlinearity, project back. SwiGLU splits the up-projection into two parallel
matrices. One, the `gate`, goes through SiLU; the other, `up`, does not. They are
multiplied elementwise, then projected down:

    down( silu(gate(x)) * up(x) )

The intuition that made it click: the gate is a learned, input-dependent valve.
For each of the 8192 intermediate channels, `silu(gate(x))` produces a soft 0-to-1
multiplier on the matching `up(x)` channel, so the network can open and close
parts of the feed-forward per token, instead of pushing everything through one
fixed nonlinearity. That is the "gated linear unit" idea; SiLU is just the smooth
gate function.

It costs a third matrix, so Llama sizes the intermediate dim at about 2.7x hidden
(8192 for this 2048-hidden model), not the classic 4x. Same parameter budget,
spent on a gate instead of a wider single projection.

The one thing to get exactly right is which branch the nonlinearity sits on. SiLU
goes on the gate only, and the branches multiply, not add. Swap them, or move the
SiLU, and because SiLU is nonlinear the output genuinely changes, but short
prompts still look plausible while the logits drift off HF. So one test does the
real comparison (1e-5 against the HF `mlp` hook on the layer's true input), and a
second is a pure trap test: run `swiglu` with gate and up exchanged and assert the
result disagrees, so a future refactor that flips them fails loudly.

Easiest 1e-5 of the week. No conventions to argue about like RoPE had, just three
matmuls in the right order. The HF `LlamaMLP.forward` is the same three lines, so
mirroring it is trivial and the match is immediate.

## Diagram
[swiglu-gated-mlp.svg](../diagrams/swiglu-gated-mlp.svg) — x fans out to the gate
and up projections, SiLU on the gate branch, elementwise multiply, then down. The
shapes (2048 to 8192 and back) and the one trap (SiLU on the gate only) called out.

## Tomorrow
Day 6: GQA attention in `layers.py`. Q/K/V projections, RoPE on q/k (Day 4 plugs
in here), repeat the 8 KV heads to match 32 query heads, causal mask, softmax,
output projection. No KV cache yet, just the plain prefill math, verified within
1e-5 of the HF `self_attn` hook. The usual mismatch suspects: the mask, the head
reshape, and the GQA repeat.

---
Post angle: Day 5 of building an inference engine from scratch. Today: SwiGLU, the
feed-forward block almost every modern model uses. A normal MLP widens each token
vector, bends it through one nonlinearity, and narrows it back. SwiGLU adds a
second parallel projection and uses it as a gate: one branch goes through SiLU to
make a soft per-channel valve, the other branch passes straight through, and they
get multiplied. So the network can open and shut parts of the feed-forward per
token instead of forcing everything through one fixed curve. It pays for that with
a third matrix, which is why Llama makes the inner dimension about 2.7x the hidden
size instead of the textbook 4x. The whole thing is three matmuls in the right
order, and it matches Hugging Face to 1e-5 on the first try. The only way to get
it wrong is to put the SiLU on the wrong branch, so I wrote a test that swaps the
two branches and demands the answer change, to catch exactly that.
