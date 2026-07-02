---
title: "Day 17: paging under the sampling path, free-and-reuse closes Week 5"
parent: Daily log
nav_order: 17
---

# Day 17: paging under the sampling path, free-and-reuse closes Week 5

Date: 2026-07-02 · Week 5 · Phase 2 Paged memory

## What I added today
The sampling path can now run over the paged pool, and a finished sequence frees
its blocks back so the next one reuses them. `generate` and `generate_stream`
grow three arguments: `paged=True` (a self-managed pool sized to this one run,
the same sizing `greedy_generate_paged` uses), `block_size`, and an optional
`allocator` to share one pool across sequences. A small `_sampling_cache` helper
picks the cache: `NaiveKVCache` by default, `PagedKVCache` when paging is asked
for. The decode loop itself did not change; it already spoke only the
`append`/`seq_len` interface both caches share, so choosing the cache is the whole
edit. The paged cache is freed in a `finally`, so a run returns every block the
moment it ends (normal stop, eos, or a caller that abandons the stream). Five new
tests in `tests/test_generate.py`, including the free-and-reuse stress run: one
pool sized for a single sequence serves two back to back, and the second only
succeeds because the first freed. I also de-flaked one Day-11 tolerance test
(details below). Pure suite **112 green**, ruff clean, and the gated real-model
tests still match HF token for token.

## Why it matters
This closes Week 5. Days 14-16 built the allocator, the block table, and the
paged cache, and wired paging into the *greedy* decode path. But the real decode
path is the sampling one (`generate`), and it was still building a `NaiveKVCache`
directly. Today paging reaches the path the server actually calls. More
importantly, free-and-reuse is the first time a finished sequence's memory
provably returns to the pool for another sequence to take. That is the entire
economic argument for paging: a pool sized for one sequence can serve a stream of
them, one after another, because each hands its blocks back on the way out. The
shared pool under *many concurrent* sequences is Weeks 8-9, but the reuse
primitive it stands on is here and tested now.

## What I learned
1. **Free-on-finish belongs in a `finally`, not after the loop.** `generate_stream`
   is a generator with several early returns (eos, the last-token short circuit)
   and it can also be abandoned mid-drain, which raises `GeneratorExit` inside it.
   Only a `finally` covers all of those. Put the `free()` after the loop body and a
   caller who reads three tokens and walks away leaks the whole sequence's blocks
   forever. The `finally` is what makes "a finished sequence never leaks a block"
   true regardless of how the stream ends.
2. **An `allocator` argument is what makes reuse observable.** `paged=True` alone
   builds a fresh pool per call, so two runs never touch the same blocks and reuse
   is invisible. Passing one shared `BlockAllocator` into both runs, sized to
   exactly one sequence, turns the claim into a test: the second `generate` can
   only allocate because the first freed, and if free-on-finish regressed it would
   raise `KVCacheExhausted` instead of silently passing.
3. **Choosing the cache, not forking the loop, is the point of Day 16's interface.**
   The reason wiring paging into sampling was a helper and three keyword arguments,
   not a second decode loop, is that `NaiveKVCache` and `PagedKVCache` expose the
   identical `append`/`seq_len`. The loop never learns which one it holds. Every
   day I resist duplicating that loop is a day the two decode modes cannot drift.
4. **A tolerance can be flaky without any code being wrong.** A Day-11 test compared
   the incrementally-cached forward against a one-shot full forward at `atol=1e-5`
   on unseeded random weights. Both assemble the same K/V, but they accumulate the
   attention sum in a different order (incremental concat vs one contiguous matmul),
   so fp32 rounding alone drifts by ~1e-5 and tripped the bound maybe two runs in
   five. Loosening it to `1e-4`, the model's documented HF-agreement level, fixes
   the flake without hiding anything: a real cache bug moves logits far more than
   1e-4.

## Diagram
[paged-free-reuse.png](../diagrams/paged-free-reuse.png). One shared pool sized for
a single sequence, three snapshots in time. Top: the sampling path builds a
`PagedKVCache` and frees it in a `finally`. (1) Sequence A decodes and fills all
three blocks, `num_free` = 0. (2) A finishes and `free()` returns every block,
`num_free` back to 3. (3) Sequence B decodes over the *same* pool, reusing the
blocks A gave back. Same seed draws the same tokens whichever cache is underneath,
because paging moves where the K/V live, not what attention computes.

## Tomorrow
Week 5 is done: the allocator, block table, paged cache, and now paging under both
decode paths with real free-and-reuse. Week 6 replaces the deliberately slow read,
the gather-and-reassemble that rebuilds the full contiguous history every step,
with a hand-written Triton paged-attention kernel that attends over the scattered
blocks directly. The torch reference built across Weeks 4-5 is exactly what that
kernel gets verified against: same tokens, same logits, now without materializing
the history.

## Post angle
Day 17 of building an LLM inference engine from scratch. Days 14-16 built paged
memory: a pool of fixed KV blocks, a block table mapping tokens to blocks, and a
paged cache that stores real K/V through it, verified byte-for-byte against the
naive contiguous cache. But all of that was wired only into greedy decode. Today
it reaches the sampling path, the one the server actually calls, and closes the
loop with free-and-reuse. The edit was small on purpose: `generate` and
`generate_stream` gain a `paged` switch and an optional shared allocator, and a
one-line helper picks the cache. The decode loop does not change at all, because
the naive and paged caches expose the exact same `append`/`seq_len` interface, so
the loop never learns which one it holds. Two things earned their keep. The
`free()` lives in a `finally`, not after the loop, because a streaming generator
can stop at eos, at the token cap, or be abandoned half-drained, and only a
`finally` catches every exit; anywhere else and a caller who reads three tokens
and walks away leaks the whole sequence's blocks. And a shared allocator argument
is what makes reuse a test instead of a claim: one pool sized for a single
sequence serves two in a row, and the second only allocates because the first
freed. If free-on-finish regressed, it would raise instead of quietly passing.
This is the reuse primitive the concurrent shared pool stands on later. It still
matches HF on Llama-3.2-1B token for token. 112 tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
</content>
</invoke>
