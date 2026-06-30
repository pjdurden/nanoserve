---
title: "Day 15: the block table, address translation for a scattered sequence"
parent: Daily log
nav_order: 15
---

# Day 15: the block table, address translation for a scattered sequence

Date: 2026-06-30 · Week 4 · Phase 2 Paged memory

## What I added today
`BlockTable` in `src/nanoserve/cache.py`: the per-sequence map from a logical
token position to the physical block that holds it. It sits on top of yesterday's
allocator and adds `append(num_new_tokens)`, which grows the sequence and pulls a
fresh block from the pool only when growth crosses a block boundary; `position(p)`,
which translates a logical position to `(block_id, offset)`; `slot(p)`, the flat
index `block_id * block_size + offset` that Week 5's paged read/write will address;
and `free()`, which hands every block back and resets. Ten new tests in
`tests/test_cache.py` pin it: address translation (the Day-14 example, position 17
lands at offset 1 of the second block), lazy allocation across a boundary, atomic
growth under exhaustion, and the headline equivalence. Pure suite **100 green**,
ruff clean.

## Why it matters
The allocator owns physical blocks but is blind to meaning: it hands out ids and
takes them back, nothing more. Nothing yet says *which* block holds *which* token.
The block table is that missing layer, and it is the exact counterpart of an OS
page table: a sequence is an ordered list of block ids, and logical position `p`
lives in `block_ids[p // block_size]` at offset `p % block_size`. That one line is
the whole paging trick. It buys the freedom the Week-3 contiguous cache never had:
the physical blocks can sit anywhere in the pool, scattered, and the model still
sees one clean contiguous sequence, because translation does not care whether the
blocks are next to each other.

## What I learned
1. **Lazy allocation is what makes paging cheap on the hot path.** A new block is
   pulled at exactly one moment: when a token's position equals the current
   capacity, so it has no block yet. The common decode step appends one token in
   the middle of the last block and allocates *nothing*. Memory is grabbed one
   block at a time, only at the instant it is first written, never reserved up
   front. So `append(1)` on a half-full block is pure arithmetic, no pool touch.
2. **The equivalence test is the real contract, and it is the same shape as the
   cache's.** The KV cache had to prove an optimization changes nothing; the block
   table has to prove that growing a token at a time (decode) lands a sequence on
   the *exact same blocks and slots* as one bulk allocation (prefill). If those
   diverged, prefill and decode would write the same token's K/V to different
   physical addresses and attention would read garbage. So the test grows one
   table by `append(1)` ten times, another by a single `append(10)`, and asserts
   identical `block_ids` and identical `slot(p)` for every position.
3. **Growth has to be atomic, inherited from the allocator.** If `append(n)` needs
   two new blocks and only one is free, it must take *none* and raise, leaving
   `num_tokens` and `block_ids` exactly as they were. A half-grown table is the
   same disease as a half-filled allocation: blocks stranded, and `num_tokens` out
   of sync with the blocks actually held. `append` leans on `allocate_for`, which
   is already all-or-nothing, so the table inherits the guarantee for free.
4. **Concrete trace, block_size 4, after one sequence freed (so reuse scatters):**

   ```
   seq A (8 tok) -> blocks [0, 1], then free()    # pool LIFO now tops with 1, 0
   seq B append(12):
     index 0 -> block 1   holds positions 0..3
     index 1 -> block 0   holds positions 4..7
     index 2 -> block 2   holds positions 8..11
   position 5  -> block_ids[5 // 4] = block_ids[1] = 0, offset 5 % 4 = 1
   slot(5)     -> 0 * 4 + 1 = 1
   ```

   B's blocks are `[1, 0, 2]`, not a contiguous run, because reuse is last-in
   first-out and B is running on the memory A just vacated. The logical view is
   still `0..11` in order; the physical placement is scattered; translation
   resolves each one correctly anyway. That gap between the two views *is* paging.

## Diagram
[block-table.png](../diagrams/block-table.png). Left: the logical positions of a
20-token sequence. Middle: the block table mapping index 0 to block 5 and index 1
to block 2. Right: those two blocks scattered in the physical pool among free ones.
Bottom: position 17 translated all the way to flat slot 33.

## Tomorrow
The table computes *where* every token's K/V goes but stores no K/V yet. Next is
the physical side of Week 5: a real `[num_blocks, block_size, num_kv_heads,
head_dim]` pool tensor, writing each token's K/V to the slot this table returns,
and gathering a sequence's K/V back out of scattered blocks for attention. Once
that read-gather matches the naive contiguous cache token for token, paged
attention works in plain torch, before any kernel.

---
Post angle: Day 15 of building an LLM inference engine from scratch. Yesterday I
built the allocator that owns a pool of fixed KV blocks. Today is the block table,
the piece that remembers which block holds which token, and it is just an operating
system page table aimed at the KV cache. A sequence is an ordered list of block
ids, and logical position p lives in block_ids[p // block_size] at offset p modulo
block_size. That single line is the whole trick, because it lets a sequence's
blocks sit scattered anywhere in the pool while the model still sees one clean
contiguous run. Two things earned their tests. First, allocation is lazy: a new
block is pulled only when a token crosses a block boundary, so a normal decode step
appends one token mid-block and allocates nothing, which is what keeps paging cheap
on the hot path. Second, and this is the real contract, growing a sequence one
token at a time (decode) has to land it on the exact same blocks and slots as one
bulk allocation (prefill); if those ever disagreed, the two paths would write the
same token's K/V to different physical addresses and attention would read garbage.
So the test grows one table by single appends, another in one shot, and asserts the
block ids and every flat slot match. The flat slot is the punchline: block_id times
block_size plus offset is the one integer next week's paged read and write will
address, scattered blocks underneath, contiguous view on top. 100 tests green.
#AI #LLM #vLLM #BuildInPublic #PagedAttention #Claude #OpenAI #MachineLearning
