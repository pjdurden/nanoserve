---
title: "Day 52: the shape stopped moving, and the closed set that is too big to capture"
parent: Daily log
nav_order: 52
---

# Day 52: the shape stopped moving, and the closed set that is too big to capture

Date: 2026-09-06 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.buckets`, the policy that decides which shapes a decode step is allowed
to present. `DecodeBuckets(max_batch_size, max_model_len)` is built from the cache's
own two limits, because a row bucket the cache has no row for and a width wider than
the slot table are both a crash rather than a rounding decision. `rows` is the power
of two ladder filtered to what fits, plus `max_batch_size` itself (a server picks 6
and without that a full batch would have no bucket at all); `widths` is the multiples
of `width_multiple` up to `max_model_len`, with the last one being `max_model_len`
even when that is not a multiple. `row_bucket`, `width_bucket` and `shape_for` do the
rounding, `shapes` is the whole capture list and `count` is its size.

`bucket_run`, `padded_cells` and `waste` are what a run costs through them. The gates
are `check_shape_in_set`, `check_run_closed`, `check_capture_budget` and
`check_waste_bounded`, plus `check_pad_inert`, which is the correctness one.

The wiring is four changes. `SlotTable.read` takes `pad_rows` and widens the
rectangle to `len(rows) + pad_rows`, which over a prefix is the same basic slice one
row taller and therefore still a window; the padded rows' lengths are 0. `DecodePlan`
grows `pad_rows` and `sink_slot`, and with them `graph_rows`, `graph_width`,
`is_bucketed`, `graph_cells`, `pad_cells` and `pad_share`; `batch_size` still means
cache rows and `graph_rows` means what the forward runs, and the whole point of the
day is that the two are allowed to disagree. `plan_decode` takes `buckets`, pads the
write slots with the sink and the positions with 0. `BatchedPagedKVCache` gains
`bucket_decode`, `decode_buckets` and `sink_slot`, and `Engine.build` passes it
through: the engine pads `input_ids` and drops the padded rows off `logits` after the
forward.

`decode_shape` now reports `plan.graph_rows` and `plan.graph_width`, so
`CompiledDecode.distinct` is the number of graphs the run really asks for.
`tests/test_buckets.py` is 56 tests. Suite **1431 green** (5 GPU-gated skips), ruff
clean.

`bucketbench.py`, host only, no weights, no compiler, 3 rows of an 8-row cache from
a 16-token prompt, width multiple 128
([day-52-bucketbench.csv](data/day-52-bucketbench.csv)):

    steps   shapes open   bucketed    waste       cells open -> bucketed    plan ms
        8             8          1    35.9%          492 ->        768   0.59 -> 0.61
       64            64          1    54.5%        9,312 ->     20,480   4.02 -> 4.13
      128           128          2    53.6%       30,912 ->     66,560   8.13 -> 8.75
      512           512          5    38.5%      418,560 ->    680,960  32.56 -> 34.91

and the second table, which is the one that decides whether any of this is a capture
list, at 256 rows and 8192 tokens:

    width multiple    128:  9 rows x 64 widths =  576 shapes
    width multiple    512:  9 rows x 16 widths =  144 shapes
    width multiple   2048:  9 rows x  4 widths =   36 shapes
    width multiple   8192:  9 rows x  1 width  =    9 shapes

## Why it matters
**Three days of work were pointing at a dimension that still moved.** Day 49 removed
the graph breaks and found the guard that mattered was on `table.num_tokens`. Day 50
moved the addressing out of the forward so the only thing left to guard on is a
tensor shape. Day 51 made that tensor a window on a persistent buffer, so the address
stopped moving. The shape was the last one, and it moves on both axes: `rows` is
whatever the scheduler admitted and `max_ctx` grows by one every step for the whole
run. 512 steps, 512 shapes, and dynamo abandons the frame after eight. Rounding both
axes up turns 512 into 5, and the 5 is not the run length, it is the number of width
buckets the run crossed. A 512-step run that starts and finishes inside one bucket
presents one.

**The padding has to be inert in two directions and only one of them is obvious.**
Reading is easy: a padded row's `context_lens` entry is 0, the mask covers its whole
row, and it softmaxes over `finfo.min` everywhere, which is finite. Writing is the
half that corrupts. A planned decode write is one `index_put` over the whole batch,
so a padded row writes K/V whether anybody wants it to or not, and "somewhere" must
not be a slot a sequence owns. So the cache keeps a **sink slot**, one row past the
end of the pool the allocator hands out: a legal index into `k_pool` that no block
maps to and no `BlockTable` can name.

**Slicing the write instead would have been the same bug three days running.** The
obvious alternative is `k[:real_rows]`, and it is `k[:3]` where 3 is a Python integer
that changes when the scheduler admits or reaps. Inside the traced region that is a
guard on a value, which is exactly what Day 49 measured a forty-six-fold regression
on and exactly what Day 50 existed to remove. The whole point of padding is to stop
the traced region from knowing how many rows are real, so the one thing it must not
do is tell it.

**Closed is not the same as small, and the arithmetic is a product.** `count` is
`len(rows) * len(widths)`, and the width axis is unbounded in a way the row axis is
not. 256 rows gives 9 row buckets, which is fine. 8192 tokens at a 128-token multiple
gives 64 widths, so the closed set is 576 shapes, and every member is a real compile
or a real capture with its own memory. That is worse than the open set, not better.
The width multiple buys it back in padding: 2048-token widths over the same model is
4 widths and 36 shapes. `check_capture_budget` is the line where "closed" and
"capturable" stop being the same word, and it is the gate I expected to write last
and ended up leaning on most.

**vLLM does not pay this, and the reason is a property of its kernel.** Its cudagraph
capture list is over batch sizes only: the kernel walks a block table and reads each
row's length out of a tensor, so the context axis never reaches a shape guard at all.
nanoserve buckets both axes because its reference read gathers a `[rows, width]`
rectangle of K/V *before* it masks, which makes the width a real tensor dimension.
The second column of that table is the price of the reference read, stated as
compiles rather than as microseconds.

**And the padding is real work.** 38.5% of the cells the 512-step run computes are
there to hold the shape still: 680,960 against 418,560. That is the fourth time this
engine has bought a fixed rectangle and paid for the corners, after Day 29's
`waste_fraction`, Day 34's `prefill_padding_waste` and Day 50's `padding_share`. The
trade is only worth taking if something downstream actually reuses the graph, which
is why this day ends with a flag that defaults to off.

## What I learned
1. **A property named `batch_size` had two meanings and I had not noticed until one
   of them had to change.** `DecodePlan.batch_size` was "rows in this step", and that
   was the row count the scheduler picked *and* the first dimension of every tensor,
   because they were the same number. Padding splits them, and every call site had
   quietly picked one meaning. `_planned_write` wanted the tensor dimension,
   `check_plan_current` wanted the cache rows, and the two lines looked identical. I
   added `graph_rows` for the tensor side and left `batch_size` meaning cache rows,
   and the useful part was not the new property, it was being forced to say which one
   each existing line had meant all along.
2. **The sink cannot come out of the allocator, and the audit is what says so.** My
   first version called `allocator.allocate()` once and kept the block. Day 35's
   `audit_blocks` fails that immediately and correctly: a block that is allocated and
   held by no request is stranded, which is the exact shape of a leak. The sink is not
   a block. It is one row past the end of the pool tensor, so it is an address that
   exists physically and cannot exist in the ledger, and the pool is allocated one row
   taller unconditionally so there is one pool shape rather than two.
3. **Day 27's `finfo.min` paid off in a place it was not written for.** A padded row
   has `context_lens = 0`, so every column of its score row is masked and the softmax
   runs over a row of identical values. With `-inf` that is 0/0 and a NaN; with
   `finfo.min` it is a uniform distribution over garbage that the caller throws away.
   The choice was made on Day 27 for padded *prefill* columns and it is what makes a
   zero-length row legal here at all. I did not plan that and would not have found it
   before it broke.
4. **The gate I wrote for completeness is the one with the finding in it.**
   `check_capture_budget` was going to be a two-line sanity check. Then I put real
   serving numbers through it and the default configuration produces 576 shapes,
   which is not a capture list, it is a compile bill. The whole day reads as a win
   until you multiply the two axes, and the arithmetic that says so is four lines
   long. Writing the price next to the feature is the only reason I found out on the
   day I built it rather than in a profile next week.
5. **The row bucket ladder needs `max_batch_size` in it and powers of two do not
   supply it.** `(1, 2, 4, 8, ...)` filtered to a cache of 6 rows gives `(1, 2, 4)`,
   and then a full batch of 6 has no bucket and `bucket_for` raises. A full batch is
   the case a serving system is *most* likely to hit. Adding the cache's own row count
   to the set is one line and it is the difference between the feature working and the
   feature working until the machine is busy.
6. **A width bucket has a ceiling that a row bucket does not, and it is the slot
   table.** `round_up(8100, 128)` is 8192, which is fine if `max_model_len` is 8192
   and a `ValueError` out of `SlotTable.read` if it is 8000. So `width_bucket` clamps
   to `max_model_len` and the top bucket is `max_model_len` whether or not it is a
   multiple. The rounding is bounded above by a real allocation, and the moment a
   bucket set is constructed from anything other than the cache it is describing, that
   invariant is gone.
7. **The padding cost the host almost nothing and I checked because I expected it
   to.** A padded read is `slots[:n + pad, :width]` instead of `slots[:n, :width]`,
   which is the same basic slice, and the extra write slots are a list append. The
   measurement is 34.9ms against 32.6ms over 512 steps, about 7%, and all of it is
   the wider `write_slots` and `positions` tensors. The cost of this day is device
   work (38.5% more cells) and not host work, which is the opposite of every other
   day this week.
8. **A benchmark whose two knobs are the same number hides half the feature.** The
   first version of `bucketbench.py` used `rows` as both the batch and the cache's
   row count, so `row_bucket(3)` was 3, `pad_rows` was always 0, and the sweep only
   ever exercised the width axis. Splitting them into `--rows 3 --max-batch 8` is
   what made the row padding appear, and it moved the reported waste from 18% to
   38.5%. The number the benchmark was reporting before was true and was measuring a
   configuration that does not happen.

## Diagram
[bucketed-decode-shape.png](../diagrams/bucketed-decode-shape.png). Left top is the
open shape set: one rectangle per step, growing. Right top is the same run bucketed,
with the padded row and the padded columns drawn in and the row's two destinations
marked (reads nothing, writes to the sink). Left bottom is the sink slot's place in
the pool, one past the last block the allocator can hand out. Right bottom is the
product: rows times widths at four width multiples, and the line where a capture list
becomes a compile bill.

## Tomorrow
The shape is closed, the address is fixed, and the plan is replayable, which is the
whole precondition list for a capture. What is left between here and
`torch.cuda.graph` is on the input side: `write_slots` and `positions` are still
built out of Python lists into fresh tensors every step, and the engine's padded
`input_ids` is a `torch.cat` every step. A replay does not accept a fresh tensor; it
reads the buffer it was recorded against, so Day 53 is the persistent input buffers,
written in place, and the arithmetic for how many of them a bucket set needs. Then
`check_capture_budget` stops being a projection and starts being a memory number,
because a capture list of 36 is 36 recorded graphs each holding its own workspace.

## Post angle
Day 52 of building an LLM inference engine from scratch. A decode step's rectangle is
`[rows, max_ctx]`, and `max_ctx` grows by one every step, so a 512-step run presents
512 distinct shapes to the compiler and dynamo gives up after eight. Today both axes
get rounded up: the batch is padded to a power-of-two row bucket and the rectangle is
read at a multiple of 128, so 512 shapes become 5, and the 5 is not the run length,
it is how many width buckets the run crossed. The interesting part is the padded row.
It has to be inert in two directions and only one of them is obvious. Reading is
easy: `context_lens = 0` masks the whole row, and it softmaxes over `finfo.min`,
which is finite, which is a choice I made on Day 27 for a completely different reason
and did not expect to collect on. *Writing* is the half that corrupts. A planned
decode write is one `index_put` over the whole batch, so a padded row writes K/V
whether you want it to or not, and it must not land in a slot some sequence owns. So
the cache keeps a sink slot: one row past the end of the pool the allocator hands
out, a legal address that no block maps to. The obvious alternative, slicing the
write to `k[:real_rows]`, is a Python integer that changes every step sitting inside
the traced region, which is precisely the guard Day 49 measured a 46x regression on.
And the thing I did not expect: closed is not the same as small. The set size is
`len(row_buckets) * len(widths)`, a product, and at 256 rows and 8192 tokens with a
128-token multiple that is 576 shapes, each one a real compile. Worse than leaving it
open. Coarsen the width multiple to 2048 and it is 36, paid for in padding. vLLM
never pays this because its capture list is over batch sizes only: its kernel reads
each row's length from a tensor, so the context axis never reaches a guard. Mine
gathers the rectangle before it masks, so the width is a real dimension. 38.5% of the
cells the bucketed run computes are padding. 1431 green.
