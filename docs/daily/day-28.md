---
title: "Day 28: one pool, one block table per sequence"
parent: Daily log
nav_order: 28
---

# Day 28: one pool, one block table per sequence

Date: 2026-07-23 · Week 7 · Phase 3 Batching and scheduler

## What I added today
`BatchedPagedKVCache` in `src/nanoserve/cache.py`: N sequences, N `BlockTable`s, and
one `BlockAllocator` and per-layer pool underneath all of them. `write(layer, k, v,
valid)` takes the padded batch's key mask and stores only the real columns, so a row
of length 2 occupies 2 slots and the rectangle never reaches the pool. `slot_mapping()`
hands the read a `[batch, max_ctx]` table of slots plus a `context_lens` vector, and
`paged_attention` is the batched decode read over it. `seq_lens` and `cached_tokens`
report what the batch is actually holding.

The read itself is `paged_attention_batched_reference` in `kernels/paged_attention.py`,
the same shape of thing Week 6 started from: one gather per row through its own slots,
GQA repeat, softmax, with everything past `context_lens[i]` biased to `finfo.min`. No
causal mask, because a decode query sits at the end of its own history and may see all
of it.

`gqa_attention` grew the branch that ties the two together. A cache that exposes a
ragged `write` and a forward that carries a mask is a *prefill*: store each row's real
K/V in that row's blocks, then attend densely over this step's own tokens, since on a
prefill the history is exactly what was handed in. No mask means *decode*, which falls
through to the fused batched read. `LlamaModel.greedy_generate_batch` is the loop that
uses both: prefill the padded rectangle once, then one token per row per step, each row
at its own absolute position, until every row has hit EOS or the cap.

Twenty-eight new tests in `tests/test_batched_cache.py`, all against oracles that
already exist: the batched read equals `paged_attention_reference` on each row's slots
alone, the batched cache equals a per-row `PagedKVCache`, and `greedy_generate_batch`
emits exactly the tokens `greedy_generate_paged` emits for each prompt run by itself,
on the tiny model and on the real Llama-3.2-1B. Suite **239 green** (5 GPU-gated
skips), ruff clean.

## Why it matters
Day 27 batched a prefill and had to stop there. One forward over N prompts is a real
win and a useless one on its own, because a served request is one prefill and then
hundreds of decode steps, and the decode steps are where a single-sequence engine
wastes the card. The blocker was structural: `PagedKVCache` owns one block table, so
there was no way to say which physical blocks belong to which row, and `gqa_attention`
refused a padded batch on the paged path rather than attend over a neighbour's blocks.
Per-sequence tables are the missing piece, and they are cheap: the tables are the only
per-row state, and the blocks stay common property.

The shared pool is the whole point, not an implementation detail. N contiguous caches
side by side would also let a batch decode, and it would reserve each row a worst-case
buffer and cap how many rows fit. One pool with per-row tables means a short sequence
occupies exactly what it uses, blocks are handed to whoever needs one next, and rows
interleave physically: in today's diagram sequence A's fourth block is number 6, not 3,
because B and C took the next free ones first. That indirection is what the Weeks 8-9
scheduler will allocate out of, and it is why paging was worth building before batching
rather than after.

It also finishes an argument Day 27 started. Padding is a property of the input tensor.
Now that the cache is ragged, the only padding that survives is in the prefill
rectangle and in the read's addressing rectangle, and neither of those stores anything.
What is left of the static-batching bill is the part a scheduler has to fix rather than
a data structure: a row that has emitted EOS keeps getting a query, a slot and a block
every step until the slowest row in the batch finishes. There is a test asserting that
waste, so Week 8 has something concrete to remove.

## What I learned
1. **The key mask does not follow the tokens into the cache; it turns into a count.**
   I expected to thread `attention_mask` through the paged read the way Day 27 threaded
   it through the dense one, and there is nothing for it to do there. A mask exists to
   silence pad keys, and per-sequence tables mean no pad key was ever written: the write
   consumes the mask, the pool holds only real tokens, and the read only needs to know
   *how many* of a row's slots are real, not *which*. `[batch, kv_len]` of bools became
   `[batch]` of ints. So the batched read now refuses a mask outright, because at that
   point one is either redundant or is hiding a bug. The mask is a prefill-time object
   and the context length is its decode-time shadow, and I had been treating them as the
   same tensor.
2. **Atomic per sequence is not atomic per batch, and the check has to happen on layer
   0.** `BlockTable.append` already reserves all-or-nothing for one sequence, which
   turns out to mean nothing here: rows 0 and 1 can grow and row 2 find the pool dry,
   leaving the tables disagreeing with the tokens actually stored. So the batch's whole
   block demand is summed and checked before any row moves. The second half cost me a
   test run. I also guard that a masked prefill only lands on an empty cache, and the
   first version of that guard fired on layer 1 of a perfectly good prefill: layer 0 is
   the layer that grows the tables, so by layer 1 the rows already hold this step's
   tokens. Any "is this the first write?" question in a cache where one layer owns the
   bookkeeping can only be asked on that layer.
3. **The padding inside `slot_mapping` is a real memory access, not a placeholder.** The
   reference indexes the pool with the entire `[batch, max_ctx]` rectangle and masks
   afterwards, so every padded entry is genuinely dereferenced; there is a test that
   writes 1e4 into the slot a short row pads with and asserts the output does not move.
   Which means the pad value has to be an in-range index, and specifically not the -1 I
   reached for first: torch wraps a negative index onto the end of the pool, silently
   reads whatever token lives in the last slot, and the mask then hides the evidence.
   Zero is always legal. A kernel makes this vanish by never issuing the load, which is
   exactly why the reference has to be the thing held to the oracle.

## Diagram
[per-sequence-block-tables.png](../diagrams/per-sequence-block-tables.png). Three
sequences with their own block tables on the left, the read's `[3, 7]` slot mapping and
`context_lens` column on the right with the inert padding dashed out, and the shared
20-slot pool across the middle, coloured by owner so the interleaving is visible along
with the two blocks that are reserved but not yet written. The three boxes below are the
day's lessons: the mask becoming a context length, the padding that still has to be a
legal index, and the all-or-nothing reservation. The banner is the bill still owed: a
finished row keeps its query, slot and block until the batch's slowest row is done.

## Tomorrow
Measure it. Static batching now runs end to end, so the week's remaining question is
what it actually buys and where it stops paying: tokens per second against batch size,
and the head-of-line blocking number, which is what it costs the seven short rows in a
batch to sit behind one long one. The existing benchmark harness already knows how to
time a decode loop. That measurement is the honest setup for Week 8's continuous
batching, the same way `padding_waste` was for today. One debt noted while writing it:
the batched read is still a reference that gathers a `[batch, max_ctx]` rectangle, so a
short row pays for the longest row's history. Lowering it to a kernel with one program
per row and its own context length is the Week 6 treatment applied to the batch axis.

## Post angle
Day 28 of building an LLM inference engine from scratch. Yesterday I batched the
prefill: N ragged prompts padded into one rectangle, one forward, each row identical to
running it alone. It stopped there, because the paged KV cache owned exactly one block
table, so a batch had no way to say whose physical blocks were whose. Today each row
gets its own table over the same shared pool, which is what makes the decode steps batch
too. The thing I did not expect is what happened to the attention mask. I assumed I
would thread it into the paged read the way I threaded it into the dense one. There is
nothing for it to do there. A mask exists to silence pad keys, and once each sequence
owns its own table, no pad is ever written: the write consumes the mask, the pool holds
only real tokens, and the read only needs to know how many of a row's slots are real,
not which. A `[batch, kv_len]` tensor of bools became a `[batch]` vector of ints. The
mask is a prefill object and the context length is its decode shadow. The read now
refuses a mask outright, because at that point one would either be redundant or be
hiding a bug. Two things bit on the way. Blocks are handed out all-or-nothing per
sequence, which means nothing for a batch: rows 0 and 1 can grow and row 2 find the pool
dry, so the whole batch's block demand has to be checked before any row moves. And the
padding inside the slot mapping is a real memory access, not a placeholder, because the
reference gathers the whole rectangle before it masks. I reached for -1 as the pad
index. Torch wraps that onto the last slot of the pool, reads someone else's token, and
the mask hides the evidence. Zero is always legal. What is left is the bill only a
scheduler can pay: a row that has hit EOS keeps its query, its slot and its block until
the slowest row in the batch finishes, and there is now a test that asserts that waste
so continuous batching, the thing vLLM and SGLang are actually famous for, has something
concrete to remove. 239 green.
