# Day 07: one full transformer block, end to end vs HuggingFace

Date: 2026-06-22 · Week 1 · Phase 0 Foundations

## What I added today
`src/nanoserve/layers.py`: `transformer_block`, the pre-norm Llama decoder block
that wires the four pieces from Days 4-6 into one (attn_norm, attention, residual,
mlp_norm, SwiGLU, residual). Three new tests in `tests/test_layers.py`: two
pure-math (a residual-wiring trap and a per-sublayer-norm trap) and one gated
against the real Llama-3.2-1B `model.layers[0]` hook to 1e-5. Full suite is 44 green.

## Why it matters
This is the Week 1 done line. The whole model is this block stacked sixteen times
between an embedding and a final norm, so a block verified end to end against HF
means the forward pass in Week 2 is mostly a `for` loop. It is also the first time
all four computed pieces run together on the real weights, which is the moment a
wiring mistake (rather than a per-piece math mistake) would show up.

## What I learned
There is no new math in a transformer block. Every formula was already built and
checked to 1e-5 on its own. The entire risk today was *dataflow*, and a block has
exactly two structural decisions: where the norm sits, and what the residual skips.

Llama is pre-norm. The norm lives *inside* each sublayer's branch, and the residual
adds the sublayer output back to the input the sublayer saw *before* the norm:

    x = x + attention( attn_norm(x) )
    x = x + swiglu(    mlp_norm(x)  )

The trap is that the normalized tensor is right there in your hand when you go to
write the residual add, and adding *that* back instead of the raw `x` typechecks,
runs, and produces plausible-looking activations. The highway just quietly stops
carrying the un-normalized signal. The other easy slip is reusing one norm for both
sublayers: `attn_norm` and `mlp_norm` are different learned weights, but they are
the same shape, so a copy-paste that points the MLP at `attn_norm` also just runs.

Both bugs are invisible to a smoke test and invisible to the eye, so I wrote a trap
for each instead of trusting the HF compare alone:

- Residual wiring: zero out `o_proj` and `down_proj` (the last matmul of each
  sublayer), which forces both sublayer contributions to exactly 0. A correctly
  wired block then returns its input untouched (`x + 0 + 0`). If a residual added
  back the normalized input, or skipped, the output drifts off `x` and the test
  fails. This pins the skip connections without needing the real weights.
- Per-sublayer norm: pin the attention residual to `x` (zero `o_proj` again), then
  perturb only `mlp_norm.weight`. The output must move, which proves the MLP branch
  actually reads `mlp_norm` and not some shared norm.

Then the real one: feed layer 0 its true input (the token embeddings, captured off
the `embed_tokens` hook), run `transformer_block` with the loader's full layer-0
weight set and our own RoPE table, and compare to what HF's `model.layers[0]` hook
emits. 1e-5 on the first run. The `Weights.layer(0)` helper from the Day 3 loader
returns exactly the in-block-named dict the function wants, so the call site is one
line, which is the payoff for getting the names right back on Day 3.

That closes Week 1: the map, the weights, the four pieces, and now the block they
form, all checked against the real model. Week 2 stacks sixteen of these and chases
HF token for token.

## Diagram
[transformer-block.svg](../diagrams/transformer-block.svg) — the block dataflow
(input, attn_norm, attention, residual, mlp_norm, SwiGLU, residual, output) with
the residual highway drawn as the skip around each sublayer, the whole block in
four lines, and the two wiring traps (adding back the normalized tensor instead of
raw x; pointing both sublayers at one norm).

## Tomorrow
Week 2, Day 8: start the full forward pass in `model.py`, the embedding lookup and
the loop that stacks the block sixteen times into a final hidden state, on the way
to greedy decode matching HuggingFace token for token.

---
Post angle: Day 7 of building an inference engine from scratch, and the end of week
one. Today I assembled a full transformer block from the four pieces I built this
week, and it matches Hugging Face to 1e-5 on the first run. The lesson of the day is
that a block has no new math in it at all. Every formula was already proven. The
only thing that can go wrong is the plumbing: a transformer is pre-norm, so each
sublayer normalizes its own input and then the residual adds the result back to the
input as it was *before* that norm, not after. The normalized tensor is sitting
right there when you write the add, and grabbing it by mistake compiles, runs, and
produces activations that look completely reasonable while quietly breaking the
residual highway. The other trap is that the two norms in a block are different
learned weights that happen to be the same shape, so pointing both sublayers at one
of them also just runs. Neither shows up in a smoke test, so I wrote two trap tests:
one zeros the last matmul of each sublayer so a correct block becomes the identity
(proving the skips are wired), and one perturbs only the second norm and demands the
output move (proving each sublayer reads its own). That is week one done: the
weights load, the four pieces are each verified against the real Llama-3.2-1B, and
now the block they form is too. Week two stacks sixteen of these and goes for
matching the reference token for token.
