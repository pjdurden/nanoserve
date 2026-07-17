---
title: "Day 25: the fused read dispatches to the kernel backend"
parent: Daily log
nav_order: 25
---

# Day 25: the fused read dispatches to the kernel backend

Date: 2026-07-16 · Week 6 · Phase 2 Paged memory

## What I added today
One line moved in `src/nanoserve/cache.py`, and it is the line the whole of Week 6
was building toward. `PagedKVCache.paged_attention`, the cache's fused read, called
`paged_attention_reference` directly through Day 24: the byte-exact torch oracle,
the same SDPA over the assembled slots the tests compare against. Day 25 routes it
through `paged_attention_dispatch`, the Day-23 entry point that asks
`select_backend(q.device)` which read this device can actually run. On a CUDA tensor
with Triton installed that is the `@triton.jit` kernel; on a CPU it is the Day-22
tlsim model, the same streaming loop written in torch. The engine gets the kernel on
a card and the correct-and-slow model on a laptop, and the cache never learns which.

Two new tests in `tests/test_cache.py` pin the wiring. One spies on the dispatcher
symbol the cache imports and asserts the read calls it exactly once, handed this
layer's own K/V pools and the slot mapping for the whole history, then hands its
result straight back. The other feeds the cache's pool through both the dispatched
read and `paged_attention_reference` and pins them to `atol=1e-5`, the same tolerance
the kernel itself is held to. The consequence rippled into three older equality tests
that had pinned the cache read `torch.equal` against a contiguous SDPA: they now carry
the flash tolerance, because the model no longer runs the byte-exact reference. Suite
**186 green** (5 GPU-gated skips), ruff clean.

## Why it matters
Until today Week 6 was a kernel in a file. Day 22 wrote the fused CPU model, Day 23
wrote the real Triton kernel, Day 24 built the instrument to read its scaling, and
every one of them was reachable only from a test. The model's actual attention still
ran `paged_attention_reference`, the oracle. This is the wiring step that turns the
kernel into the engine's real attention: the paged forward now dispatches, so the
same code path picks up the GPU kernel when a card is present and falls back to the
model that runs anywhere when it is not. The reference did not disappear, it changed
jobs. It is the oracle both backends are graded against, not the path either one runs.

The honest cost is that paged attention is no longer bit-identical to the naive
contiguous path. Both backends stream the online softmax, which reassociates the
exponent sums, so the output lands a few ulps off a plain SDPA (about 7e-6 on the
two-layer test model) rather than exactly on it. That is not a regression, it is the
accuracy trade every flash-attention kernel makes, and it belongs on the model's path
precisely because that is where the kernel now lives. The token-level tests still hold
because argmax over these logits does not flip at 1e-5; the logit-level test that
asked for 1e-6 now asks for 1e-5, the tolerance the streaming softmax actually
delivers.

## What I learned
1. **Byte-identity was a property of the oracle, not of paging.** Every "paging
   changes nothing" test through Day 24 passed `torch.equal` because the cache read
   *was* the reference, running the identical causal fp32 softmax by exact index. The
   moment the model runs the streaming kernel instead, the scatter and gather stay
   exact (same K/V, same slots) but the softmax summation order moves, and equality
   becomes closeness. The tests that flipped from `equal` to `allclose` were not
   loosened to hide a bug; they caught exactly the numerics change the dispatch
   introduced, which is what a good test does.
2. **The right tolerance is the one the backend is already held to.** The kernel
   tests pin tlsim and Triton to `paged_attention_reference` at `atol=1e-5, rtol=1e-4`.
   When the cache read started running that same backend, the correct new tolerance
   for the cache tests was not a number I picked to make green, it was that same
   1e-5, propagated up one level. A tolerance that means something is a tolerance
   borrowed from where the approximation is defined.
3. **A dispatcher makes the fallback the default and the kernel the exception.** The
   branch reads `"triton" if device.type == "cuda" and has_triton() else "tlsim"`, so
   the CPU model is what runs unless both a card and the package are present. That is
   the right default for a repo that must run on a laptop: the slow-and-correct path
   is the one you get for free, and the fast path is the one that has to prove its
   preconditions. The engine is portable first and fast where it can be, not the
   other way around.

## Diagram
[paged-read-dispatch.png](../diagrams/paged-read-dispatch.png). The cache read at the
top, the `select_backend` pill below it, and the two branches: the Triton kernel on
`cuda & has_triton()`, the tlsim model on `else`. The reference sits dashed in the
middle, with both backends held to it at `atol=1e-5`, no longer on the path. Both
converge to the one attention output that flows back to `gqa_attention`. The banner
carries the trade: Day 24 ran the byte-exact reference, Day 25 runs the streaming
softmax, so paged is flash-close to naive, not bit-identical.

## Tomorrow
Measure the dispatch where it matters. The read benchmark and the scaling fit
(Days 20 and 24) time `gather` and `fused` as standalone functions; wire the runner
to time the cache's *dispatched* read end to end, so the sweep reports the path the
engine actually takes rather than a function called in isolation. On CPU that pins the
tlsim constant against the gather constant under the real slot mapping; on a card it is
the first number that shows the kernel bending the constant down, which is the whole
claim Week 6 has been setting up to make. Keep the reference as the correctness gate
the timed path is checked against before any number is trusted.

## Post angle
Day 25 of building an LLM inference engine from scratch. Today was one moved line and
it is the one the whole kernel week was for. My paged cache's attention read was still
calling the byte-exact torch reference, the oracle. The real Triton kernel and its CPU
model existed but only the tests ever touched them. Today the cache read dispatches:
`select_backend(q.device)` picks the Triton kernel on a card and the streaming CPU
model on a laptop, and the model's actual attention runs whichever the box can run.
The reference did not go away, it changed jobs, from the path the model runs to the
oracle both backends are graded against. The honest part: paged attention is no longer
bit-identical to the naive path. Both backends stream the online softmax, which
reassociates the exponent sums, so the output lands about 7e-6 off a plain SDPA rather
than exactly on it. That is not a bug, it is the accuracy trade every flash-attention
kernel makes, and three of my "paging changes nothing" tests flipped from `torch.equal`
to `allclose` to carry it. The tolerance I gave them was not invented, it is the same
1e-5 the kernel is already held to against the reference, propagated up one level. This
is the shape vLLM and SGLang ship: a dispatcher that runs the fast kernel where the
hardware allows and a correct fallback everywhere else, with the reference kept as the
gate, not the path. 186 green.
