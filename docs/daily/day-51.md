---
title: "Day 51: the read rectangle stopped being built, and the length that could not be shared"
parent: Daily log
nav_order: 51
---

# Day 51: the read rectangle stopped being built, and the length that could not be shared

Date: 2026-09-05 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.slots`, the persistent addressing the decode step reads through.
`SlotTable` is one `[max_batch_size, max_model_len]` int64 buffer allocated at
construction and never reallocated. `append(rows, slots)` writes one cell per row at
that row's own next column; `resync(table, row)` copies a whole row in from its
`BlockTable`; `sync(tables, rows)` does that for any row whose length disagrees;
`reset(row)` sets a length to zero and clears nothing. `read(rows, width)` hands
back the rectangle and the lengths, and the rectangle is `slots[:n, :width]` when
`rows` is a prefix of the table's, which is basic slicing and therefore a *view*:
the buffer's storage, the buffer's address, the buffer's row stride. Otherwise it is
`slots[:, :width].index_select(0, rows)`, narrowed before the gather so the copy is
the rectangle and not the whole buffer. `is_window`, `window_share`, `appended_cells`,
`resynced_cells`, `resyncs`, `windows`, `gathers` and `render` are what it says about
itself.

`rebuild_mapping(tables, rows)` is Day 50's construction, extracted rather than
deleted: it is the resync path, the oracle the table is graded against in the tests,
and the arm the benchmark times.

The wiring is five changes. `BatchedPagedKVCache.__init__` takes `max_model_len`,
clamps it to the pool (a row can never hold more tokens than the pool has slots) and
builds the table. `plan_decode` syncs the mirror *before* the block tables grow,
appends the step's write slots, and reads the plan's `slot_mapping` and
`context_lens` out of the table instead of building a grid. `adopt_row`, `reset_row`
and `free` keep the mirror honest. `Engine.build` takes `max_model_len` and
`build_engine` passes `plan.max_model_len`, because `[max_batch_size,
num_blocks * block_size]` int64 on a real card is hundreds of megabytes of pure
addressing.

The arithmetic is `table_cells`, `table_bytes`, `block_table_cells`,
`block_table_bytes`, `incremental_cells` and `resident_share`; the gates are
`check_slots_agree`, `check_mapping_is_window`, `check_window_intact`,
`check_appends_incremental`, `check_resyncs_bounded` and `check_table_fits`.
`tests/test_slots.py` is 62 tests. Suite **1375 green** (5 GPU-gated skips), ruff
clean.

`slotbench.py`, host only, no weights, 4 rows from a 16-token prompt, CPU
([day-51-slotbench.csv](data/day-51-slotbench.csv)):

    steps      rebuild      window      gather   speedup       cells written
        8      0.461ms     0.485ms     0.621ms     0.95x         656 ->    96
       64      5.697ms     2.488ms     3.158ms     2.29x      12,416 ->   320
      128     17.354ms     5.308ms     6.422ms     3.27x      41,216 ->   576
      512    224.343ms    19.617ms    25.046ms    11.44x     558,080 -> 2,112

## Why it matters
**The rectangle stopped being an allocation.** Day 50's addressing was correct and
did O(n^2) host work to append n tokens: `rows * max_ctx` Python-level `slot()` calls
plus one host-to-device build, every step, with `max_ctx` growing by one each time.
The persistent table writes `rows` cells a step whatever the histories are worth, so
`check_rebuild_bounded`'s 272x at 512 steps becomes 264x *fewer* cells, and the
forward's rectangle is not built at all, it is pointed at.

**264x fewer cells is 11x less time, and the gap is the honest part.** What is left
per step is a Python loop over rows calling `table.slot`, plus three small
`torch.tensor` builds in `append` and one in `read`. That is O(rows) with a fat
constant, not O(rows) and free, and at 8 steps the persistent table is 5% *slower*
than the rebuild because the prompt is resynced once and the run is too short to pay
it back. The crossover is somewhere under 64 steps. A day that reports the cell ratio
and not the clock would be claiming 264x for an 11x change.

**A window only exists when the rows are a prefix.** `slots[:3, :w]` is basic slicing
and gives a view. `slots[[0, 2], :w]` is advanced indexing and gives a copy, and a
scheduler hands out whichever row slots are free, so a real batch is often not a
prefix. The gather arm is 25.0ms against the window's 19.6ms at 512 steps: still far
cheaper than the rebuild, still no host loop and no transfer, and still a fresh
allocation. That matters for exactly one thing, which is why this day happened at
all: a CUDA graph replays kernels bound to the addresses they were recorded with. So
`check_mapping_is_window` is a capture gate and not a correctness gate, and its
message names the row set, because the fix is on the scheduler's side.

**A window is live storage, and that is why only one of the two tensors could be
one.** An append writes at column `length`. For the longest row in a batch that is
past the right edge of every rectangle handed out before it, so a held window keeps
saying what it said. For a *short* row it is inside one: a row of length 2 in a
4-wide rectangle has its next token written over column 2 of a rectangle somebody may
still be holding. Nothing keeps that cell out of the read except a `context_lens`
that did not move. So `slot_mapping` is a view and `context_lens` is a copy, and the
copy is not a concession: it is `[rows]`, so it costs `rows` int64, and it is what
keeps Day 50's staleness gate armed.

**The price moved from time to bytes.** The rebuild allocated nothing and was
quadratic in the run length; the table is constant in the run length and reserves
`max_batch_size * max_model_len * 8` bytes for the process. 256 rows at 131,072
tokens is 268 MB of addressing before a single token lands, which is why
`Engine.build` and `build_engine` learned to pass the length the pool was planned
around. vLLM does not keep this table: it keeps *block ids*,
`[max_seqs, max_blocks]`, and computes `block_id * block_size + offset` in the
kernel, which is the same information 16x smaller. `block_table_bytes` is that
arithmetic and `check_table_fits` prints it in the failure. nanoserve stores slots
because its reference read gathers per token and has nowhere to do the multiply,
which makes the 16x a property of the kernel and not of the table.

## What I learned
1. **A gate that can never fail looks exactly like a gate that always passes, and I
   wrote one today.** My first `check_window_intact` compared the plan's rectangle
   against the block tables it addresses. The test for it, a row reset and handed to
   a new request under a plan that still held its window, refused to fail. Of course
   it did: the window is a view of the table, so when the resync rewrote the row both
   sides of the comparison moved together. The gate that works compares the window
   against the plan's *own* copies, `write_slots` and `context_lens`, because those
   are the only things in the room that cannot move. Checking live state against live
   state is not a check.
2. **Sharing storage turns construction checks into liveness checks, and they are the
   same two lines.** Day 50's `check_plan_addressing` asserts "the write slot is the
   row's last real entry in the rectangle", a statement about how a rectangle had
   been built out of a list. Over a window, the identical comparison is a statement
   about whether anything has rewritten it since. Same code, different question, and
   the second question only exists because the storage is shared. I would not have
   found the second one by reading the first.
3. **The mirror's staleness test is length alone, and that is only enough because of
   an invariant somewhere else.** A row's slots change by being appended to (which the
   table does itself) or by the row changing tenant, and tenant changes go through
   `reset_row`, which drops the length to zero. So a length comparison catches
   everything. That is a real dependency on `BlockTable.adopt` refusing a non-empty
   table, written for an unrelated reason on Day 31, and if it ever relaxes this
   sync goes silently wrong. `check_slots_agree` compares every slot and not just the
   lengths for exactly that reason.
4. **Padding stopped being zero and nothing cared, which says the original promise
   was never the real one.** Day 50 has a test that a plan's padding is slot 0. It
   still passes, because the buffer starts zeroed, but it is no longer a promise: a
   row that is reset and re-tenanted shorter has its old tenant's slots sitting past
   its length. That was always fine. The requirement was only ever "a legal index",
   because the reference gathers before it masks, and every value ever written into
   this buffer is a real slot. Writing the test that documents the new behaviour was
   the moment I noticed the old test was pinning an implementation detail.
5. **Nothing is cleared on reset, and refusing to clear is the right answer twice
   over.** Zeroing a row would be `max_model_len` device writes to hide values that
   `context_lens` already masks, on the path that runs when a request finishes and
   the next one is admitted. It is the same reasoning as not compacting the pool: the
   cost of tidiness is paid every time and the benefit is never collected.
6. **`index_select` after narrowing, not before.** `slots[rows]` copies
   `[len(rows), max_model_len]`, which at a real `max_model_len` is the entire buffer
   for the sake of a 40-column rectangle. `slots[:, :width].index_select(0, rows)`
   copies `[len(rows), width)`. Advanced indexing does not know what you were going
   to slice off next, and the difference here is three orders of magnitude.
7. **Sizing a persistent buffer is a serving decision, and it had never been passed
   down.** `max_model_len` existed in `launch.py` and reached the scheduler and the
   pool plan, but the cache had no use for it and was constructed without it. The
   moment something in the cache is `O(max_model_len)` the number has to arrive, and
   the default I fell back to (the whole pool) is a bound that is always true and
   never sensible. A default that cannot be wrong is not the same as a default that
   is right.
8. **The benchmark needed a third arm to say anything true.** With only "rebuild" and
   "window" the CSV reads as a clean win and hides that half of real batches will not
   get a window at all. Adding the scattered-rows arm cost fifteen lines and is the
   column I would keep, because it is the one that says what the scheduler owes the
   capture.

## Diagram
[persistent-slot-table.png](../diagrams/persistent-slot-table.png). Left top is what
one decode step writes, before and after, with the buffer drawn and this step's two
cells picked out. Right top is the prefix condition, the same read as a view and as a
copy. Left bottom is the measured sweep. Right bottom is the hazard: an append at
column `length` lands outside a long row's held window and inside a short row's, and
the gate that a live `context_lens` would have disarmed.

## Tomorrow
The address is fixed and the shape is not. The rectangle is still `[rows, max_ctx]`
with both axes moving: `max_ctx` grows by one per step and the row count is whatever
the scheduler admitted, so the shape set is still one per step and `static` mode
still builds until it hits the recompile limit. Day 52 closes it with Day 49's
bucketing arithmetic on the axis it was written for: pad the batch up to a bucketed
row count and read at a bucketed width, so the forward sees one of a small set of
shapes at one address. `context_lens` is what makes the padding free, since a padded
row with length 0 attends over nothing and a padded column is masked already. Then
`compilebench.py` is finally measuring a compiler that gets the same graph twice on
purpose, and `write_slots` and `positions`, still built out of Python lists every
step, are the last two allocations between here and a capture.

## Post angle
Day 51 of building an LLM inference engine from scratch. Yesterday's decode plan
rebuilt a `[rows, ctx]` int64 rectangle on the host every step to say where each
token's K/V lives, and `ctx` grows by one per step, so 512 steps over 4 rows wrote
558,080 cells to append 2,048. Today it is one `[max_batch_size, max_model_len]`
buffer allocated once, one cell written per row per step, and the forward gets
`slots[:rows, :width]`: a view, the buffer's own storage at the buffer's own address.
2,112 cells instead of 558,080, and 19.6ms instead of 224.3ms. The gap between 264x
fewer cells and 11x less time is the honest bit: what is left is a Python loop over
rows and a few small tensor builds, so it is O(rows) with a fat constant, and at 8
steps the persistent table is actually 5% slower because the prompt gets resynced
once and the run is too short to pay it back. Two things I did not expect. A window
only exists when the batch's rows are a *prefix* of the table's: `slots[:3, :w]` is
basic slicing and gives a view, `slots[[0,2], :w]` is advanced indexing and gives a
copy, and a scheduler hands out whichever row slots are free. That is a scheduler
knob, not a cache one, and it decides whether a CUDA graph can be captured over the
step at all. And a window is live storage, which broke a gate I wrote yesterday. An
append writes at column `length`: past the right edge of a long row's held rectangle,
but *inside* a short row's, over its padding. So the rectangle can be shared and the
lengths cannot: a `context_lens` that read live out of the table would always agree
with the cache, and yesterday's staleness gate would stop being able to fail rather
than start failing. Share the buffer a later step writes past, copy the one it writes
over. vLLM keeps block ids rather than slots, which is the same table 16x smaller
with the multiply moved into the kernel, and that is the version this one is standing
in for. 1375 green.
