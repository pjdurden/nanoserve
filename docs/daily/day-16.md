---
title: "Day 16: the paged KV cache, real K/V through scattered blocks"
parent: Daily log
nav_order: 16
---

# Day 16: the paged KV cache, real K/V through scattered blocks

Date: 2026-07-01 · Week 5 · Phase 2 Paged memory

## What I added today
`PagedKVCache` in `src/nanoserve/cache.py`: the piece that finally hangs real K/V
tensors off the Day-14 pool and the Day-15 block table. It is a drop-in for
`NaiveKVCache` at the one interface attention uses, the same
`append(layer, k, v) -> (full_k, full_v)` and the same `seq_len`, so `layers.py`
and `model.py` do not change at all. What changes is purely where the K/V live:
instead of one growing contiguous tensor per layer, each layer gets a *fixed* flat
pool `[num_blocks * block_size, num_kv_heads, head_dim]`, and every token's K/V is
written at the flat `slot` the block table assigns. `model.greedy_generate_paged`
drives the same prefill-then-decode loop as the cached path (I pulled that loop out
into a shared `_greedy_decode` so naive and paged run identical code over different
storage). Eight new tests in `tests/test_cache.py`, pure suite **107 green**, ruff
clean, and the gated real-model test confirms paged greedy matches HF token for
token on Llama-3.2-1B.

## Why it matters
This is the whole point of Weeks 4-5 arriving at once. The allocator owns the pool,
the block table maps logical positions to physical slots, and today the K/V
actually flow through that map. A sequence's blocks can sit scattered anywhere in
the pool, yet attention still receives one clean contiguous
`[1, num_kv_heads, seq, head_dim]` tensor, oldest token first, byte-for-byte what
the naive cache would hand it. That gap, scattered underneath and contiguous on
top, is exactly what lets many sequences pack into one shared pool later (Weeks
8-9) instead of each reserving a worst-case buffer. And it is verifiable *now*, in
plain torch, before any kernel exists: if the gathered history is identical, the
logits cannot differ.

## What I learned
1. **One block table, shared across all layers.** My first instinct was a table
   per layer, but that is wrong and expensive: every layer stores the same tokens
   at the same logical positions, so the logical->physical map is identical for all
   of them. A physical block id therefore names a slot in *every* layer's pool at
   once, which is exactly how vLLM works. One table, grown once per step, and each
   layer owns only its own K/V pool. A table per layer would have pulled 16x the
   blocks for nothing.
2. **The step boundary is layer 0.** The table has to grow exactly once per decode
   step, not once per layer's `append`. Since `model.forward` always walks layers
   `0..15` in order, layer 0's append is the clean step boundary: it grows the
   table and records the slots for the new tokens, and the other 15 layers reuse
   those same slots. Encoding that as an explicit, documented invariant kept the
   code a few lines instead of a stateful mess.
3. **Write is a scatter, read is a gather, and the read is deliberately naive.**
   Write: `pool[slots] = k[0].transpose(0, 1)`, scattering `new_seq` vectors to
   their physical slots. Read: gather `[slot(0), slot(1), ..., slot(seq-1)]` back
   and transpose to `[1, n_kv, seq, d]`. Rebuilding the full contiguous history
   every step is not how the fast path will work, Week 6's Triton kernel attends
   over the scattered blocks directly, but doing it the slow, obvious way first is
   what makes "paged output == naive output" a provable claim before I trust a
   kernel.
4. **Lazy pool allocation dodges a dtype/device guess.** The pools are `None` until
   the first `append`, then allocated with `k.dtype` and `k.device`. No config flag
   for precision, no CPU/GPU assumption baked into the constructor; the storage
   simply matches whatever attention actually produced.

## Diagram
[paged-write-read.png](../diagrams/paged-write-read.png). Left: the write path, a
new token's K/V routed through `table.slot(pos)` into one physical cell. Right: the
read path, gathering the scattered cells back into the contiguous tensor attention
consumes. Middle: `block_ids = [2, 0]`, a two-block sequence whose blocks are out
of order because block 0 was reused after an earlier sequence freed it. The logical
view stays `0..6` in order; the physical placement is scattered; the gather makes
them agree.

## Tomorrow
Wire the paged cache into the sampling path too (`generate`/`generate_stream`
currently build a `NaiveKVCache` directly), and add a `free()`-and-reuse stress
run so a finished sequence's blocks provably return to the pool for the next one.
That closes Week 5 and sets up Week 6: replace today's gather-and-reassemble read
with a hand-written Triton paged-attention kernel, verified against this exact
torch reference.

---
Post angle: Day 16 of building an LLM inference engine from scratch. The last two
days built the allocator (a pool of fixed KV blocks) and the block table (which
block holds which token). Today the K/V actually flow through them: a paged KV
cache. Here is the trick that makes it verifiable. It is a drop-in for the naive
contiguous cache at the one interface attention uses, so nothing in the model
changes. Underneath, each layer's K/V no longer live in one growing tensor; they
live in a fixed pool of physical blocks, and every token is written at the flat
slot the block table hands back. Those blocks can be scattered anywhere in the
pool, reused out of order from sequences that already finished, yet attention still
receives one clean contiguous tensor, oldest token first, byte for byte what the
naive cache would give it. Two design calls earned their keep. One block table
shared across all 16 layers, not one per layer, because every layer stores the same
tokens at the same positions, so a block id names a slot in every layer's pool at
once, exactly how vLLM does it; a table per layer would have grabbed 16x the memory
for nothing. And the read is deliberately the slow, obvious gather, rebuilding the
full history every step, because that is what lets me prove paged output equals
naive output token for token before I trust next week's Triton kernel. It matches
HF on Llama-3.2-1B exactly. 107 tests green.
#AI #LLM #vLLM #BuildInPublic #PagedAttention #Claude #OpenAI #MachineLearning
