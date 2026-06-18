# Day 03: the loader, and the name-mapping rabbit hole

Date: 2026-06-18 · Week 1 · Phase 0 Foundations

## What I added today
`src/nanoserve/loader.py`: reads the Llama-3.2-1B safetensors and files every
tensor under nanoserve's own names, shape-checked against `ModelConfig` at load
time. The deliverable is a `Weights` container the Week 1-2 layers will pull
from. Plus `tests/test_loader.py` (24 tests, the heavy ones gated on ./weights)
and a `pythonpath = ["src"]` pytest setting so the src-layout package imports
without an editable install.

## Why it matters
Nothing downstream should ever touch a HuggingFace key string. The loader is the
one place that knows `model.layers.0.self_attn.q_proj.weight` is what nanoserve
calls `layers.0.attn.q_proj.weight`. Get the mapping right once, here, and every
layer I write next gets its weights by asking for a clean canonical name.

## What I learned
Half of "loading a model" is just getting the names right, and the file fights
you in three small ways. (1) The on-disk Llama-3.2-1B has **146** tensors but a
correct nanoserve model has **147**: there is no `lm_head.weight` on disk at all.
The output projection is tied to the input embedding, so I synthesize lm_head as
an *alias* of `embed_tokens` (shared storage, asserted by equal `data_ptr()`),
not a second 128256x2048 matrix. (2) The shapes encode GQA before any attention
math runs: q_proj is [2048, 2048] but k_proj/v_proj are [512, 2048], because 8 KV
heads x 64 = 512. Checking that at load time turns a head-count bug into a load
error instead of a silent wrong answer three days later. (3) HF keeps q/k/v as
three separate matrices; some engines fuse them into one. I kept them split for
Week 1 because it mirrors HF exactly and makes the upcoming 1e-5 verification
trivial to reason about. Fusing is a deliberate later optimization, not a
loading detail.

The strongest test is not "did the names line up" but "did the right *bytes* land
under the right name": I read a handful of tensors straight from the file and
assert the loaded q/k/gate/norm match. That is what actually catches a swapped
q<->k mapping, which shapes alone would not.

## Diagram
[weight-name-mapping.svg](../diagrams/weight-name-mapping.svg) — the HF key on the
left, the nanoserve name on the right, the GQA shape asymmetry (q to 2048, k/v to
512), and the tie: lm_head has no tensor on disk and aliases embed_tokens.

## Tomorrow
Day 4: RMSNorm + RoPE in `layers.py`, verified to 1e-5 against the HF hook. RoPE
is the one people get wrong (inv_freq, the cos/sin table, and the llama3
frequency rescaling already captured in config).

---
Post angle: Day 3 of building an inference engine from scratch. Today was the
unglamorous half of "load a model": getting the names right. Llama-3.2-1B ships
146 tensors on disk, but a correct model has 147. The missing one is lm_head: the
output projection is tied to the input embedding, so it is an alias, not a copy,
and storing it twice would waste 250MB for nothing. The shapes also quietly
encode GQA before you write a line of attention: query projects to 2048, but key
and value only to 512, because 8 KV heads are shared across 32 query heads. I
check all of that at load time, so a head-count mistake becomes a clear load
error instead of a silently wrong answer days later. And the test that matters is
not "do the names match" but "did the right bytes land under the right name",
which is the only thing that catches a swapped q and k.
