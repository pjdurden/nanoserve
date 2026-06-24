---
title: "Day 02: weights, config, and a verification harness"
parent: Daily log
nav_order: 2
---

# Day 02: weights, config, and a verification harness

Date: 2026-06-17 · Week 1 · Phase 0 Foundations

## What I added today
A real `scripts/download_weights.py` (gated snapshot_download for Llama-3.2-1B
plus a config and safetensors inventory dump), a fully populated
`config.py:ModelConfig` with `from_json` loading and a `RopeScaling` block, and a
forward-hook verification harness in `tests/reference.py` that every Week 1
component test will compare against. Day-2 tests in `tests/test_config.py` pass,
ruff clean.

## Why it matters
The harness is the spine of the whole correctness story: from here on, every
piece I write (RMSNorm, RoPE, SwiGLU, attention, a full block) gets asserted to
within 1e-5 of the HuggingFace activation for the same module. Building that on
Day 2, before any real math, means I never write a layer I cannot immediately
prove right.

## What I learned
Llama-3.2-1B is not plain RoPE. Its config carries a `rope_scaling` block with
`rope_type: llama3`, factor 32, stretching the 8k pretraining context to 131k by
rescaling low frequencies. If Day-4 RoPE applies only `rope_theta` it will match
HF at short positions and silently drift at long ones. I put `RopeScaling` in the
config now so that gotcha is impossible to forget later. Also confirmed the shape
math: 32 query heads x 64 head_dim = 2048 hidden, 8 KV heads, so GQA repeats each
KV head 4 times.

## Diagram
[verification-harness.svg](../diagrams/verification-harness.svg) — one input, two
engines (HuggingFace reference vs my hand-written layer), diff every internal
tensor to 1e-5. This is the loop every Week 1 day runs against.

## Tomorrow
Day 3: implement `loader.py`, the safetensors-key to nanoserve-tensor name
mapping. Watch for tied embeddings (no separate lm_head.weight) and per-layer
naming.

---
Post angle: Day 2 of building an inference engine from scratch. Before writing a
single layer I wired up the thing that tells me whether a layer is correct: load
the model in HuggingFace, hook every internal module, and diff my tensors against
theirs to 1e-5. The surprise of the day was that Llama 3.2 quietly rescales its
RoPE frequencies (rope_type llama3, factor 32) to reach 131k context. Miss that
and your engine looks right on short prompts and rots on long ones. Correctness
infra first, math second.
