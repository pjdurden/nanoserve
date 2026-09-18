---
title: "Day 61: the capture list loses its width axis, and the price changes currency"
parent: Daily log
nav_order: 61
---

# Day 61: the capture list loses its width axis, and the price changes currency

Date: 2026-09-17 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
`DecodeBuckets(streamed=True, block=...)`, and the three things downstream that had
to move with it.

Day 52 bucketed both axes of the decode shape because both of them moved, and the
price was a product: `len(rows) * len(widths)`, which at 256 slots and 8192 tokens on
a 128-token multiple is **576 graphs**. Day 59 wrote a read that never builds the
score rectangle and Day 60 wired it to a flag, and today is the day that spends it.
The width axis was in the set for exactly one reason and it is a property of the
*read*, not of the model: the Day-28 rectangle gathers a `[rows, width]` mapping and
scores it into a `[rows, heads, 1, width]` tensor, so a wider mapping costs real
memory. A read that walks `cdiv(context_lens[row], block)` tiles of its own row does
not care. So `streamed=True` collapses the widths to `(max_model_len,)`, `count` stops
being a product, and the same cache's list is **9 graphs**.

`cells_for` is the second half and it is not optional. `waste` measured against
`DecodeShape.cells` would read 99% on a streamed set, because the set rounds every
context up to `max_model_len` and that rectangle is never built. So the bucket set
answers "what does the read really touch": `row_bucket * width_bucket` on the default,
`rows * round_up(ctx, block)` on the streamed one, and `padded_cells` and `waste` go
through it.

`check_read_matches(buckets, read)` is the gate, and it runs inside
`check_capture_ready` so a boot refuses rather than a dashboard disagreeing at 3am.
`shared_pool_bytes`, `private_pool_bytes` and `pool_sharing_ratio` take a `block`.
`CapturePlan` carries one, prices its arena in it, and publishes it under `read_block`
in `/health`. `plan_capture` grows a fifth `width_bound_by`, `"read"`, and loses three
candidates on that arm.

`tests/test_width_axis.py` is 54 tests. Suite **1952 green** (5 GPU-gated skips), ruff
clean.

`widthbench.py`, host only, no weights
([day-61-widthbench.csv](data/day-61-widthbench.csv)):

    the capture list for one cache, both reads, 32 heads, fp32 scores, 32-key tiles:
      slots  context         read   rows x widths   shapes   startup      arena   waste
          8     2048    rectangle       4 x 16        64     3.20s      2.00 MiB  33.0%
          8     2048     streamed       4 x 1          4     0.58s      0.03 MiB   2.9%
          8     4096    rectangle       4 x 32       128     6.40s      4.00 MiB  33.0%
          8     4096     streamed       4 x 1          4     0.58s      0.03 MiB   2.9%
         16     8192    rectangle       5 x 64       320    16.00s     16.00 MiB  33.0%
         16     8192     streamed       5 x 1          5     0.72s      0.06 MiB   2.9%
         32     4096    rectangle       6 x 32       192     9.60s     16.00 MiB  33.0%
         32     4096     streamed       6 x 1          6     0.87s      0.12 MiB   2.9%
         64     8192    rectangle       7 x 64       448    22.40s     64.00 MiB  33.0%
         64     8192     streamed       7 x 1          7     1.01s      0.25 MiB   2.9%
        256     8192    rectangle       9 x 64       576    28.80s    256.00 MiB  33.0%
        256     8192     streamed       9 x 1          9     1.30s      1.00 MiB   2.9%

    what the axis was costing, per deployment:
      slots  context   shapes   startup    arena
          8     2048      16x        6x      67x
         16     8192      64x       22x     267x
         64     8192      64x       22x     256x
        256     8192      64x       22x     256x

## Why it matters
**One of those three ratios is the length of the list and the other two are not, and
keeping them apart is most of the day.** `shapes` is the list: 576 to 9, and that is
arithmetic about a product whose second factor became 1. `arena` is not about the list
at all. It is `width / block` at the widest shape, which is Day 59's
`workspace_saving`, and it would be 256x if the list had one member: a shared arena is
sized by its largest graph, so shortening the list never touched it and changing what
a graph *holds* is the only thing that ever could. And `startup` is the one I had to
walk back, which is the next section.

**The startup saving is 22x, not 64x, and the boot table is what said so.** Nine
graphs instead of 576 looks like it should divide the warm-up by 64, and the measured
boot of the toy model says 6.6 ms a graph on the rectangle arm and 19.2 ms on the
streamed one. Every capture runs the read once, and the streamed read is `tlsim` in
Python, so recording a shape costs about 2.9x what it costs on the default. The first
version of `widthbench.py` priced both arms at the same 50 ms and printed a 64x, which
is the kind of number that is right in the arithmetic and wrong in the world. The
column now carries `STREAMED_CAPTURE_PENALTY` and the honest reading is that half of
this saving is a property of this box and goes away with Triton, and the other half,
the memory, does not.

**The combination that corrupts is a streamed bucket set under a rectangle read, and
nothing else in the repo would have caught it.** The set rounds every context to
`max_model_len`, so `plan_decode` hands the read `slots[:rows, :8192]` on the first
decode step of every request. The rectangle read gathers all of it and scores a
`[256, 32, 1, 8192]` tensor for a batch whose longest history is nine tokens: 268 MB,
Day 54's number, materialised at the *start* of a run instead of at the end of a long
one. Nothing raises. The attention is right, because the padding is masked; the tokens
are right; the pool audit is right. The only symptom is that the memory bound this
module exists to impose is gone. The two halves are built from one flag in
`BatchedPagedKVCache` so they cannot be wired apart, and `check_read_matches` is for
every caller who assembles them by hand.

**Three of `plan_capture`'s four ceilings stopped being width ceilings, and the graph
limit turned into a statement about rows.** Day 55 ended on "a budget is not a
statement about the length of the capture list, it is a statement about one width",
and Day 56 found the graph limit reduces to the same knob. Both of those are
consequences of a workspace that is `rows * heads * width`. Under a tile it is
`rows * heads * block`, so a budget is not a ceiling on anything, it is a yes or no
about one tile, and the refusal has to say so rather than divide. The limit is worse:
with a single width the list *is* the row axis, so `width_from_limit`'s division has
no axis left to truncate and a limit under the row count has to name `max_rows` as the
only knob. A width flag is ignored on that arm, because honouring it would record a
list at a width no step will ever present.

**And Day 60's `/health` saving became a tautology, which I would rather write down
than quietly leave in a table.** `PagedRead` charges `score_cells` off
`slot_mapping.shape[1]`, which under a streamed bucket set is `max_model_len` on every
call, and holds `min(block, width)`, which is `block` on every call. So the quotient is
exactly `max_model_len / block` however long the run was: a server that answered one
twelve-token request reports the same saving as one that streamed for an hour. Day
60's 4.0x to 5.0x came off a cache that still bucketed the width, where the quotient
really was integrated over a growing rectangle. It cannot be fixed, and the reason is
the same reason the counter was priced in shapes in the first place: the honest
numerator is `int(context_lens.max())`, which is Day 48's synchronisation once per call
per layer to make a log read better.

## What I learned
1. **A bucket set is a claim about a read, and I had been treating it as a claim about
   a cache.** `DecodeBuckets` has taken `max_batch_size` and `max_model_len` since Day
   52 and both come off the cache, which made it feel like a property of storage. It is
   not. The row axis is a property of the scheduler and the width axis is a property of
   the kernel, and the only reason they were both in one object is that both of them
   moved. Take away the read that gathers and one of them was never the cache's
   business at all.
2. **A price is only comparable inside one currency, and switching arms switches the
   currency.** `waste` has meant "share of computed cells nobody asked for" since Day
   29, through `prefill_padding_waste` and `padding_share`, and it survived four
   redefinitions of what a cell is. This is the first time the *denominator* stopped
   existing. A 99% waste against a rectangle that is never built is not a pessimistic
   reading, it is a reading of nothing, and `cells_for` exists so the number keeps
   meaning what its name says.
3. **`round_up` and `min` are two different questions about the same tile and I wrote
   the wrong one first.** `streamed_score_cells` uses `min(block, width)`, which is
   what a program *holds*: one tile, and never more than the context. `cells_for` needs
   `round_up(width, block)`, which is what a program *walks*: whole tiles, the last one
   masked. My first draft reused `streamed_score_cells` and reported a four-token row
   under a 32-key tile as four cells of work, which is not what the loop does.
4. **The graph limit was never a width cap, it was a cap on a product that happened to
   have a width in it.** `width_from_limit` is one of the cleverest lines in the repo
   and it is clever in the way that hides an assumption: `limit // row_count` is only a
   width because the list is `rows x widths`. With one width that division silently
   asks the wrong question, and the tell was that its refusal message talks about width
   buckets in a set that has one.
5. **The dangerous half of a flag is not the arm it turns on.** Day 60's whole subject
   was that a streamed server and a rectangle server are indistinguishable from
   outside. Today's is that a streamed *bucket set* and a rectangle *read* are also
   indistinguishable from outside, and that pair is not harmless: it is the one
   combination in two days of work where the process quietly holds two orders of
   magnitude more memory than anybody planned. Two switches that are each safe do not
   make four safe states.
6. **A measured boot is worth more than a measured kernel when the claim is about
   startup.** I had the shapes, the arena and the arithmetic, and the arithmetic said
   64x off the warm-up. One `build_engine` through `plan_capture` and `warm_engine` on
   both arms, which is the path `serve.py` takes, said 22x and said why. The tables
   that have been wrong in this project have almost all been the ones where a
   multiplication stood in for a second.
7. **Publishing the degeneration is cheaper than removing the number.** The obvious
   move on finding that `saving` is now a constant was to stop reporting it under a
   streamed set. That would make `read_from_health` and `check_arm_read` refuse on the
   arm they were written for, which is Day 60's lesson about a read being not optional,
   in reverse. The number is still true, it is just no longer about the traffic, and a
   paragraph on the property is the right size of fix.

## Diagram
[capture-list-loses-a-width.png](../diagrams/capture-list-loses-a-width.png). Left top
is the collapse: one cache, two sets, 576 graphs against 9, and what `width_bucket`
does instead. Right top is the pair that corrupts and the pair that only wastes, with
the reason nothing downstream notices either. Left bottom is the currency change, both
charges side by side, and why a padded row and a padded column both cost zero. Right
bottom is the measured table with the three ratios kept apart.

## Tomorrow
The set is the row axis and the arena is a tile, and both of those are statements this
box cannot cash. Everything above is memory and startup, measured on a CPU, and the
read underneath is still `tlsim` in Python at 8x to 67x per call. Day 62 is the Triton
version of `paged_attention_batched_kernel`: the same grid of `(row, query head)`, the
same online softmax, the same `block` as the SRAM tile, graded against Day 59's
reference on the toy pools that already exist and against the rectangle on a real
forward. That is the call that turns every ratio in this file into latency instead of
a promise, and it is the one thing left before the CUDA run Day 58 pointed at is worth
booking hardware for. The honest caveat is unchanged in shape and smaller in scope:
nothing here made a single token arrive faster, and the next day is the first one in
the whole streak whose whole job is that it should.

## Post angle
Day 61 of building an LLM inference engine from scratch. The CUDA graph capture list
for a 256-slot, 8192-token server was 576 shapes, and today it is 9. Not because I
tuned anything: because the width axis was never a property of the model, it was a
property of the read. The old decode read gathers a `[rows, width]` slot mapping and
scores it into a `[rows, heads, 1, width]` rectangle, so a wider mapping is real
memory, so the width has to be bucketed, so the capture list is a product and the
product is the bill. The read I wrote on Day 59 walks `cdiv(context_lens[row], block)`
tiles of its own row and holds one tile. Hand it the whole table on every step and it
touches exactly the same cells. So the width stops being a thing to round, it becomes
a constant, and the list is the row axis, which is what vLLM's capture list has always
been and now I know the actual reason. Two things I got wrong on the way. The first:
`waste` reported 99% on the new set, and `waste` was right about the arithmetic and
measuring a rectangle that is never built. The padding price had to change currency,
so a bucket set now says what its own read really touches, which is
`rows * round_up(ctx, block)` and not `rows * width`, and `round_up` rather than `min`
because that is tiles *walked* and not the one tile held. The second: 9 graphs instead
of 576 is not 64x off the warm-up. A measured boot says 6.6 ms a graph on the old arm
and 19.2 on the new one, because every capture runs the read once and the read is
still a Python loop, so the real startup saving is 22x. The memory half is honest
either way: the shared arena goes from 256 MiB to 1 MiB, and that ratio is
`width / block` at the widest shape, which would hold if the list had a single member.
And the dangerous discovery: a streamed bucket set under the *old* read hands the
rectangle a full-width mapping on the first token of every request. 268 MB of one
intermediate, correct tokens, nothing raises. Two switches that are each safe do not
make four safe states, so the boot refuses that pair now. 1952 green.
