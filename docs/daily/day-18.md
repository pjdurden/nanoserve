---
title: "Day 18: the torch reference for paged attention, Week 6 opens"
parent: Daily log
nav_order: 18
---

# Day 18: the torch reference for paged attention, Week 6 opens

Date: 2026-07-02 · Week 6 · Phase 2 Paged memory

## What I added today
`paged_attention_reference` in `src/nanoserve/kernels/paged_attention.py`: a plain
torch function that computes one sequence's attention by reading K/V *through the
block table* instead of over a pre-assembled contiguous history. It takes the
rotated query `[1, n_q, seq_q, d]`, the layer's flat physical pools
`[num_slots, n_kv, d]` (exactly what `PagedKVCache.k_pool[layer]` holds), and the
per-position `slot_mapping` the table produces, then gathers each past token
through its slot, repeats the KV heads, applies the same rectangular causal mask
and fp32 softmax as `gqa_attention`, and returns the attention output before
o_proj. Seven new tests in `tests/test_paged_attention.py`, pure suite **119
green**, ruff clean. Nothing is wired into the model yet: this is the component,
built and pinned on its own, the way Days 14 and 15 built the allocator and the
table before anything used them.

## Why it matters
Week 6 is the headline of Phase 2: replace the deliberately slow paged read (Day
16 gathers the whole history back into a contiguous tensor every step, then runs a
normal SDPA) with a hand-written Triton kernel that attends over the scattered
blocks directly and never materializes that history. A kernel is only trustworthy
against a reference, and this is that reference. It fixes the exact contract the
kernel must meet: same inputs (query, the flat pools, the slot mapping), same
output to about 1e-3. Writing it in plain torch first means the day the kernel
lands there is already a test that says whether it is right, on any box, with no
GPU in the loop. The reference is honest about what it is: it still gathers with a
torch index rather than streaming blocks, so it is not the fast path. What it nails
down is the *interface and the math*, so the kernel has a fixed target instead of a
moving one.

## What I learned
1. **A reference is a frozen target, not a fast path.** The temptation on the first
   day of a kernel week is to start writing the kernel. But a kernel verified
   against nothing is a kernel you cannot trust, and a reference that keeps changing
   while you optimize is no reference at all. So the reference gets built and pinned
   first, deliberately slow (a torch gather), and it does not move again. Every
   later kernel test compares against this one function's output. This is the same
   discipline as Day 16's "prove paged output equals naive output before trusting a
   kernel", pushed one layer down to the attention itself.
2. **The slot mapping is the whole interface between paging and attention.** All the
   block table, allocator, and scattered-pool machinery collapses, at the attention
   boundary, into a single `[seq_total]` integer array: position p's K/V lives at
   flat slot `slot_mapping[p]`. Attention needs nothing else about paging. Making
   that the argument (rather than passing the cache or the table object) is what
   lets the future Triton kernel take the identical signature, and it is exactly the
   `slot_mapping` real paged-attention kernels are handed.
3. **Reusing `gqa_attention`'s exact mask and fp32 softmax is what makes equality
   testable.** The reference is not "an" attention, it is *the* attention the model
   already runs, expressed over a paged read. Same `triu(diagonal=past+1)` band,
   same `softmax(..., dtype=float32).to(q.dtype)`. Because the gather is an exact
   index copy and the math is identical op for op, the tests assert `torch.equal`,
   not just `allclose`: byte-for-byte, even when the physical blocks are scattered.
   A looser bound would have hidden a real slot bug behind rounding.
4. **A causal test needs a perturbation, not just a shape check.** The strongest
   test corrupts the last position's stored K/V and asserts every *earlier* query's
   output is unchanged while the last query's output moves. A future token that
   leaks into an earlier query's softmax is exactly the bug an off-by-one in the
   mask band causes, and only a perturbation catches it; a shape or a single
   all-visible decode row would pass right over it.

## Diagram
[paged-attention-read.png](../diagrams/paged-attention-read.png). The query on the
left; the scattered physical pool on the right with a sequence's tokens on
out-of-order blocks. The `slot_mapping` in the middle is the one array attention
needs: it gathers each logical position's K/V through its slot, and the reference
scores the query against that history under the causal mask. The dashed box marks
the contiguous history the reference gathers but the Triton kernel will not: same
output, without ever assembling it.

## Tomorrow
Two ways forward for Week 6. Either the first cut of the Triton `triton.jit`
kernel, held to this reference (needs a GPU, so it may run gated like the weights
tests), or wire the reference into the paged read path so `PagedKVCache` can hand
attention the output directly instead of a re-gathered contiguous tensor, proving
the fused shape end to end on CPU first. The reference pinned today is the oracle
either way.

## Post angle
Day 18 of building an LLM inference engine from scratch. Week 6 is the kernel week:
replace the paged KV read with attention that reads scattered blocks directly,
never rebuilding the contiguous history. But you do not start a kernel week by
writing the kernel. You start it by writing the reference the kernel gets checked
against, and freezing it. So today is a plain torch `paged_attention_reference`:
give it the query, the layer's flat KV pools, and the one array that is the entire
interface between paging and attention, `slot_mapping`, position p's K/V lives at
flat slot `slot_mapping[p]`. It gathers each past token through its slot, then runs
the exact same causal, GQA-repeated, fp32 softmax the model already runs. Because
the gather is an exact copy and the math is identical op for op, the tests assert
`torch.equal`, not just close, even when the physical blocks are scattered out of
order across the pool. That is the point: paging moves where the K/V live, not what
attention computes. The reference is honest about being the slow path, it still
gathers with a torch index instead of streaming blocks the way the kernel will. But
now the kernel has a fixed target: same inputs, same output to 1e-3, testable on
any box with no GPU in the loop. This is the same discipline vLLM and SGLang lean
on, a torch reference behind every fused kernel. 119 tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
