---
title: "Day 11: the KV cache, 5.7x faster and token-for-token identical"
parent: Daily log
nav_order: 11
---

# Day 11: the KV cache, 5.7x faster and token-for-token identical

Date: 2026-06-26 · Week 3 · Phase 1 Correct generation

## What I added today
`src/nanoserve/cache.py`: `NaiveKVCache`, a contiguous per-layer K/V store that
grows by `torch.cat` each step and hands back the running history. Attention now
takes an optional `cache`/`layer_idx`: it appends this step's rotated K/V and
scores the query against the whole history, with the causal mask generalized from
a square triangle to a rectangle (`triu(..., diagonal=past+1)`) so the same code
path is prefill (`past=0`) and one decode step (`seq=1`). `LlamaModel` threads the
cache through `forward` by layer index and adds `greedy_generate_cached`: prefill
the prompt once, then forward a single token per step. Eight new tests in
`tests/test_cache.py`; full suite 80 green. On the real Llama-3.2-1B, 40 tokens
went from 87.3s to 15.2s, a 5.74x speedup, output bit-identical to the naive path.

## Why it matters
This is the one optimization the whole arc is named after. Week 2's greedy decode
re-ran attention over the entire growing prefix every step, which is O(n^2): the
fortieth token redoes the work of the first thirty-nine. But a past token's key
and value never change once it is fixed, so that recompute is pure waste. Cache
them, append one column per step, and decode is O(n). Every paged-cache and
scheduler day after this is about managing this cache under memory pressure and
across many sequences; today is the cache existing at all.

## What I learned
The cache is an optimization, so the entire job is for it to change *nothing*
observable, and the design pressure all came from keeping the verified Week-2 math
untouched while making it incremental:

1. **One rectangular mask covers prefill and decode.** Week 2 used a square
   `[seq, seq]` causal triangle. With a cache the `seq` new queries score against
   `kv_len` cached keys, so the mask is `[seq, kv_len]`: query i sits at absolute
   position `past + i` (where `past = kv_len - seq`) and may see keys `0..past+i`.
   `triu(full(-inf), diagonal=past+1)` is exactly that band. The pretty part is
   that `past=0` collapses it back to the Week-2 square triangle, so the prefill
   path is numerically unchanged, and a single decode query (`seq=1`, `past=kv_len
   -1`) gets an all-zero row: it sees the whole history, which is the point.
2. **Cache the compact GQA K/V, not the repeated version.** The 8 KV heads get
   expanded to 32 for the score matmul, but that expansion is a read-time view.
   Append *before* `repeat_kv` and the cache stores 8 heads; append after and it
   stores 32 and you have quietly thrown away the entire reason GQA exists. The
   4x memory saving GQA buys is only real if the cache respects it.
3. **Position is now explicit, because the prefix is gone.** In Week 2 the whole
   sequence was passed every step, so `position_ids` defaulting to `0..len-1` was
   always right. A decode step forwards one token with no prefix attached, so its
   absolute position has to be handed in (it is exactly `cache.seq_len` just
   before the append). Get this wrong and RoPE rotates the new K for the wrong
   slot, which does not throw, it just slowly drifts the tokens off the reference.

The test that matters is not the speed, it is `torch.equal`: cached greedy decode
produces the same tokens as the Week-2 recompute path, and both equal HF. A cache
that is faster but changes one token in forty is not faster, it is broken. The
5.74x is on a 5-token prompt generating 40 on CPU; the gap widens with sequence
length, because that is the whole O(n^2)-versus-O(n) story made visible.

## Diagram
n/a today. The paged-cache diagram is already drawn for Weeks 4-5; this naive
contiguous cache is the thing that diagram replaces, so I will draw the
contiguous-versus-paged before/after when the paging lands.

## Tomorrow
Week 3's remaining thread: wire `sample` (Day 10) into a cached `generate` so a
`do_sample` request runs the same fast decode loop, and start measuring the cache
properly (tokens/sec versus prompt length) to set up the Week-4 motivation, which
is that this contiguous buffer wastes most of its VRAM.

---
Post angle: Day 11 of building an LLM inference engine from scratch. Today is the
KV cache, the one idea that makes inference fast. Yesterday's greedy decode re-ran
attention over the entire prefix every single step, which is O(n^2): the 40th
token redoes the work of the first 39. But a past token's key and value never
change once the token is fixed, so all that recompute is waste. So I store them:
compute each token's K and V once, append one column per step, and decode drops to
O(n). On the real Llama-3.2-1B, 40 tokens went from 87 seconds to 15, a 5.7x
speedup, and the tokens come out bit-for-bit identical to the slow path. That
identity is the actual test, not the clock. A cache that is faster but changes one
token in forty is not faster, it is broken. Two things that bite: cache the
compact 8-head GQA K/V, not the 32-head expanded version, or you throw away the
entire reason GQA exists; and a decode step now has to be told its token's
absolute position explicitly, because the prefix it used to be inferred from is
gone. The mask is the neat part: one rectangular causal mask is both the prefill
triangle and the single-query decode row, depending only on how much history is
already cached. #AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
