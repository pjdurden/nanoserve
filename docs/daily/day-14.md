---
title: "Day 14: the block allocator, paging for attention starts as pure bookkeeping"
parent: Daily log
nav_order: 14
---

# Day 14: the block allocator, paging for attention starts as pure bookkeeping

Date: 2026-06-29 · Week 4 · Phase 2 Paged memory

## What I added today
`BlockAllocator` in `src/nanoserve/cache.py`: a fixed pool of physical KV blocks
with `allocate`, `allocate_for(num_tokens)`, `free`, and `free_all`, plus a
`KVCacheExhausted` exception for when the pool runs dry. It is deliberately
torch-free: pure integer bookkeeping over block ids, a free list paired with an
allocated set. Eleven new tests in `tests/test_cache.py` pin the contract
(distinct blocks out, atomic multi-block reserve, reuse after free, double-free
rejected, exhaustion raises). Pure suite **90 green**, ruff clean. This is the
first brick of Week 4: the contiguous Week-3 cache stays, but the machinery that
will replace it starts here.

## Why it matters
The Day-13 sweeps showed decode speed was already fine; the next wall is memory.
A contiguous per-sequence cache has to reserve one growing buffer sized to each
request's worst case, which fragments VRAM and caps how many sequences fit. Paged
attention borrows the operating-system trick: carve memory into fixed blocks and
let a sequence's logical token positions map to physical blocks that can sit
anywhere. Today is only the allocator, the part that owns the pool and never loses
or double-hands-out a block. Getting that invariant exactly right now is what lets
the block table (next) and the paged read/write (Week 5) be the easy part.

## What I learned
1. **The whole component is one invariant: every block is in exactly one of two
   places, the free list or the allocated set.** I store the free ids as a stack
   and the handed-out ids as a set. `allocate` pops the stack and adds to the set;
   `free` checks membership in the set and pushes back. That pairing is what makes
   a double free (or freeing a block that never belonged to this pool) a loud
   `ValueError` instead of silent corruption, because a block handed to two
   sequences is two sequences writing K/V over each other. The set is not an
   optimization, it is the safety rail.
2. **`allocate_for` has to be atomic.** A request needs `ceil(num_tokens /
   block_size)` blocks. If the pool cannot cover all of them, it must reserve
   *none* and raise, because a half-filled allocation strands blocks that no
   sequence owns and nothing ever frees. So the count is checked up front, before
   any block moves. A leak in an allocator is not a slow drip; it is a block that
   is gone for the lifetime of the server.
3. **Concrete trace, a 6-block pool of 16-token blocks:**

   ```
   fresh pool: num_free = 6
   seq A (40 tok): [0, 1, 2] | free = 3      # ceil(40/16) = 3
   seq B (20 tok): [3, 4]    | free = 1      # ceil(20/16) = 2
   seq C (16 tok): [5]       | free = 0
   next request -> KVCacheExhausted: all 6 blocks allocated
   seq A finishes, free_all([0,1,2]) | free = 3
   seq D (30 tok) reuses freed blocks: [2, 1] | free = 1
   ```

   Two things in that last line. The pool refilled from A's exact blocks, so D is
   running on physical memory A just vacated, which is the entire point of a pool:
   one sequence's death is another's allocation. And D got `[2, 1]`, not `[0, 1]`,
   because the free list is a stack and reuse is last-in-first-out. Order does not
   matter for correctness (a block is a block), but it is a reminder that the
   allocator hands out *whatever is free*, not contiguous runs. A sequence is a
   list of block ids, and they can be scattered.
4. **`KVCacheExhausted` is named now on purpose.** On one sequence it just means
   "context too long for the pool". But it is the same event that, once many
   sequences share the pool in Weeks 8-9, becomes a scheduling decision: stop
   admitting new requests, or preempt a running one and recompute it later to
   reclaim its blocks. Giving the out-of-memory condition its own type means the
   future scheduler catches one specific thing, not a bare `RuntimeError` it has
   to guess about.

## Diagram
[block-allocator.svg](../diagrams/block-allocator.png). Left: a pool of eight
blocks, two sequences scattered across it, the rest free. Right: the free list
and allocated set, the four operations, and the exhaustion signal.

## Tomorrow
The allocator owns blocks but nothing yet says *which* block holds *which* logical
token. Next is the block table: a per-sequence map from logical position to
physical block, so position 17 with block_size 16 lands at offset 1 of the
sequence's second block. That is the address translation, and once it exists Week
5 can finally store and read real K/V through it.

---
Post angle: Day 14 of building an LLM inference engine from scratch. Week 3 proved
decode speed is fine; the next wall is memory. So today I started paged attention,
which is just the operating-system paging trick aimed at the KV cache: carve
memory into fixed-size blocks and let a sequence scatter across whatever is free,
instead of reserving one giant contiguous buffer sized to its worst case. Day 14
is only the allocator, and I kept it deliberately boring: no tensors, pure integer
bookkeeping. A free list of block ids plus a set of the ones handed out, and the
whole thing rests on one invariant, every block is in exactly one of those two,
always. That is what turns a double free into a loud error instead of two
sequences silently writing K/V over each other. The one subtlety worth the time is
that multi-block allocation has to be atomic: a request needs ceil(len/block_size)
blocks, and if the pool cannot cover all of them it reserves none, because a
half-done allocation strands blocks nothing ever frees, and a leak in an allocator
is a block gone for the life of the server. A 6-block pool fills with three
sequences, the fourth request raises KVCacheExhausted, then the first sequence
finishes and the fourth runs on the exact blocks it vacated. One sequence's death
is another's allocation. That exhaustion signal is named on purpose: in a few
weeks it stops meaning "context too long" and starts meaning "time to preempt
something". 90 tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
