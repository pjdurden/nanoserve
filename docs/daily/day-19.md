---
title: "Day 19: the fused paged read wired into attention"
parent: Daily log
nav_order: 19
---

# Day 19: the fused paged read wired into attention

Date: 2026-07-03 · Week 6 · Phase 2 Paged memory

## What I added today
`PagedKVCache.paged_attention(layer, k, v, q, n_rep, scale=None)` in
`src/nanoserve/cache.py`, and the wiring that puts `gqa_attention` on it. The new
method is the *fused read*: hand it this step's rotated Q/K/V and it writes the
K/V into its block pool (the same write `append` does, now factored into a shared
`_write`) and then attends over the whole history through the block table via the
Day-18 `paged_attention_reference`, returning the attention output before o_proj.
No contiguous history is rebuilt. `gqa_attention` gained one duck-typed branch: if
the cache exposes `paged_attention`, it calls it and takes the output straight
back, skipping the local repeat/mask/softmax; a naive cache has none, so it still
takes the `append` gather path unchanged. Five new tests (four in
`tests/test_paged_attention.py` for the method, one spy test in
`tests/test_cache.py` for the plumbing), pure suite **124 green**, ruff clean.

## Why it matters
Day 16 built the paged cache but read it the slow way: every step it gathered the
scattered blocks back into one contiguous `[1, n_kv, seq, d]` tensor and ran a
normal SDPA over it, which re-materializes the very buffer paging exists to avoid.
Day 18 wrote the torch reference that reads through the block table directly and
froze it. Today those two meet: the model no longer rebuilds the history to
attend. The paged path writes each token's K/V to its slot and scores the query
against the scattered blocks in place, exactly the shape a Triton kernel will take
when it lands. This is the wiring that makes the reference load-bearing instead of
a test-only oracle, and it is the last CPU-provable step before the kernel: the
fused read is byte-identical to the gather (every paged equality test still
passes, now running through the new path), so the interface and the math are
nailed down and only the implementation underneath is left to swap.

## What I learned
1. **A behaviour-preserving change still needs a failing test, just aimed one level
   down.** The fused read produces the exact same tokens as the gather, so no
   output test could fail before I wired it: the change is invisible at the model's
   surface. The honest TDD target was the new *method* (it did not exist, so its
   tests failed with `AttributeError`) plus a spy on the plumbing: forward over a
   paged cache must call `paged_attention` once per layer and `append` zero times.
   That spy is what actually pins the wiring; the equality tests only confirm it
   did not change what comes out.
2. **Factor the write before you fork the read.** `append` and `paged_attention` are
   the same write followed by two different reads. Pulling the write into `_write`
   (grow the shared table on layer 0, lazily allocate the pools, scatter the K/V to
   this step's slots) means both reads leave the cache in provably identical state,
   and one test asserts exactly that: run two caches side by side, one through each
   read, and their tables and pools end byte-for-byte equal. Only the return value
   differs. Duplicating the write instead would have been two places for a slot bug
   to hide.
3. **Duck-typing keeps the layer decoupled from the cache.** `gqa_attention` never
   imports the cache classes; it already called `cache.append` blind. The fused
   path is the same trick: `getattr(cache, "paged_attention", None)`. Present means
   paged, so read through the blocks and return; absent means naive, so gather and
   score. No `isinstance`, no import cycle (`cache` imports the kernel reference,
   `layers` imports neither), and the split between the two backends is one line.
4. **The compact-then-repeat discipline survives the fusion.** The cache still
   stores the 8-head K/V, and the GQA repeat to 32 heads still happens at read time
   inside the reference, as a view. Fusing the read did not tempt the cache into
   storing the blown-up K/V; the whole reason GQA shrinks the cache 4x is preserved
   because the repeat lives on the read side, wherever that read now happens.

## Diagram
[paged-fused-read.png](../diagrams/paged-fused-read.png). Two reads off the same
block pool. The top path is Day 16's gather: scattered blocks reassembled into a
contiguous history, then a normal SDPA, the buffer paging meant to avoid. The
bottom path is today's fused read: `gqa_attention` hands the cache Q/K/V, the K/V
is scattered to its slots, and attention scores the query over the blocks in place
through `slot_mapping`, returning the output with no contiguous buffer ever built.
The dashed box marks the history the top path materializes and the bottom path
never does.

## Tomorrow
Week 6's kernel. With the fused read wired and pinned to the reference, the next
step is the first cut of the `triton.jit` paged-attention kernel behind the same
`paged_attention` signature, held byte-close to `paged_attention_reference`. It
needs a GPU, so it may run gated like the weights tests; the reference and the
fused shape proven on CPU today are the fixed target it gets checked against.
Alternatively, microbenchmark the fused read against the gather to put a number on
what the kernel has to beat.

## Post angle
Day 19 of building an LLM inference engine from scratch. Day 16 stored the KV cache
in scattered blocks but read it the slow way: gather every block back into one
contiguous tensor each step, then run normal attention over it, rebuilding the
exact buffer paging exists to avoid. Day 18 wrote the torch reference that reads
through the block table directly and froze it. Today they meet: attention no longer
rebuilds the history. The cache gained a fused read, write this step's K/V to its
slot, then score the query over the scattered blocks in place, and `gqa_attention`
takes the output straight back. The interesting part was testing a change that
changes nothing you can see: the fused read is byte-identical to the gather, so no
output test could fail first. So the real target was the new method (it did not
exist) plus a spy on the plumbing, forward over a paged cache must call the fused
read once per layer and the gather zero times. That spy is what pins the wiring.
This is the shape a Triton kernel will take when it lands; today nails the
interface and the math on CPU so only the implementation underneath is left to
swap, the same reference-behind-every-kernel discipline vLLM and SGLang lean on.
124 tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
