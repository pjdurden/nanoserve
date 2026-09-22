---
title: "Day 65: the third read arrived before its memory did"
parent: Daily log
nav_order: 65
---

# Day 65: the third read arrived before its memory did

Date: 2026-09-21 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
`SPLIT` on `PagedRead`, a third mode beside `RECTANGLE` and `STREAMED`, and the three
places the wiring turned out to touch.

**`reads.py`.** `PagedRead(mode=SPLIT, block=...)` holds a `SplitWorkspace` it did not
allocate. `attach` takes it, refuses it on a read that would never address it, refuses
an arena partitioned in a different tile, and refuses a *second* arena while
tolerating the same one twice. `splits` reads the chunk count off the workspace and is
0 without one. A call between construction and `attach` raises `SplitUnsound` rather
than allocating, which is the whole reason the method exists. `streamed` goes back to
meaning "is exactly Day 59's read" and `tiled` is the new question, `mode in (STREAMED,
SPLIT)`: the width axis turns on whether a read holds a tile, and the split holds
tiles too.

**`ReadStats` grows `splits`**, in `as_dict`, `from_dict`, `render` and `since`, and
`since` refuses a change of it on the mode's grounds and not the backend's: a process
does not acquire chunks the way it discovers a backend. `_charge` gains a third arm,
`split_score_cells(shape, heads, block, splits)`, so the published saving is the
streamed read's divided by the chunks.

**`buckets.py`.** `DecodeBuckets(..., splits=)`, refused without `streamed=True` for
`read_workspace_bytes`'s reason: a split is a partition of the streamed read's tile,
and a set that still buckets the width has no tile to partition. `check_read_matches`
keys its width clauses on `tiled` instead of `streamed` and grows four more: a split
set under an unsplit read, a split read under an unsplit set, a split read holding no
arena at all, and two chunk counts that disagree. `row_axis` comes out of the
constructor as a function, because the split count is planned *over* the row axis and
has to exist before the set does.

**`cache.py`.** `split_read=True`, refused alongside `streamed_read` and refused
without `bucket_decode`. `read_splits` is `plan_splits` over this cache's own row axis
at its own table width, computed at construction. `allocate_split_workspace(device)`
is the second half: every number an arena needs is already a property of the cache or
its model, so `device` is the only argument, and a caller cannot get the arena wrong
without getting the cache wrong first.

`tests/test_reads.py` grows 30 and `tests/test_width_axis.py` 9. Suite **2154 green**
(21 GPU-gated skips), ruff clean.

`modebench.py`, host only, no weights
([day-65-modebench.csv](data/day-65-modebench.csv)):

    one planned decode step, 4 rows over a 2048-wide table, 4-key tiles:
          mode  backend  splits  charged   held  saving  vs oracle
     rectangle    torch       0     4096   4096    1.0x   0.00e+00
      streamed    tlsim       0    65536    128  512.0x   2.38e-07
         split    tlsim       4    65536    512  128.0x   2.38e-07

    one decode step at 256 rows, 32 heads, 8192 tokens, 32-key tiles:
          read  splits  held cells  published     partials
     rectangle       0    67108864       1.0x      0.00 MB
      streamed       0      262144     256.0x      0.00 MB
         split      16     4194304      16.0x     76.55 MB

    check_read_matches over every pair this repo can build:
                         set             read  verdict
             bucketed widths        rectangle  ok
             bucketed widths         streamed  refused: width buckets
             bucketed widths  split, no arena  refused: width buckets
             bucketed widths            split  refused: width buckets
             bucketed widths  split, 2 chunks  refused: width buckets
                   one width        rectangle  refused: no width axis
                   one width         streamed  ok
                   one width  split, no arena  refused: nobody sized
                   one width            split  refused: nobody sized
                   one width  split, 2 chunks  refused: nobody sized
        one width, 16 chunks        rectangle  refused: no width axis
        one width, 16 chunks         streamed  refused: nobody addresses
        one width, 16 chunks  split, no arena  refused: no workspace
        one width, 16 chunks            split  ok
        one width, 16 chunks  split, 2 chunks  refused: 2 against 16

## Why it matters
**This is the first read in the engine that cannot run on its own, and the shape of
the day is what that costs.** The rectangle needs a pool and a mapping. The streamed
read needs those and a tile, which it carries itself because a tile is an integer. The
split needs an arena somebody else allocated, at a width somebody else fixed, and a
`PagedRead` is constructed when the cache is, which on the boot path is before
anything has been moved to a device. So there is a window where the mode is set and
the workspace is not, and it is a real window rather than a theoretical one: it is
every line of boot between `BatchedPagedKVCache(...)` and the call that reserves
memory. Only three things can happen in it. The read can allocate its own, which is
the unpriced `torch.empty` the entire previous day existed to remove. It can silently
do something else, which is a flag that means one thing in the log and another in the
process. Or it can refuse, which is the one that leaves the operator holding a
sentence instead of a mystery. Nothing else this week has had two legitimate states
before its first step.

**`streamed` was the right question with two reads and the wrong one with three, and
that is a specific kind of mistake worth naming.** Day 61 collapsed the bucket set for
a read that holds a tile instead of a rectangle and called the predicate `streamed`,
because at the time "holds a tile" and "is Day 59's read" picked out the same object.
They are not the same question, they were the same *answer*, and a third read is
exactly the thing that separates them. The split gathers no more than the stream does,
so a width bucket buys it nothing either, and a gate still asking the narrower
question would have refused every legitimate split server with a message about a
rectangle it never builds. Two predicates now, and the one the gate reads is the one
about memory.

**A gate that had one clause has four, and the four name four different mornings.**
The width clauses say a set and a read disagree about whether the context is an axis.
The new ones say they disagree about an arena, and the arena is the largest thing a
split server reserves: 76.55 MB of partials at serving size, next to 8.39 MB of score
tiles. A set priced for sixteen chunks under a read that runs one has bought fifteen
sixteenths of a workspace nobody addresses; a split read under a set priced in whole
tiles is a launch nobody sized; a split read holding no arena is a boot path that
stopped one call short. Same class of failure, three different fixes, and the only
reason to split them into three messages is that a single "these disagree" would send
somebody to the wrong one. Twelve of the fifteen pairs in the matrix refuse, and not
one of them would have raised in a running server: every one of them answers
*correctly*, against a memory number nobody in the process can see is wrong.

**And the published saving means a third thing now, which makes it more useful rather
than less.** `saving` is `score_cells / held_cells` and has been since Day 60. On the
rectangle it is 1.0x because a read that materialises the row holds every cell it was
charged. On the stream it is `width / block`. On the split it is that divided by the
chunks, because a split does not make the tile smaller, it makes more of them alive at
once: the split count is a grid axis and every program on it is scoring. So a split
server reports 16x where its streamed neighbour reports 256x, and a gate that only
knew two reads would file that as a broken streamed server. What the quotient does not
contain is the arena, and that is deliberate: the partials are not scores, they are
constant for the life of the process, and they are priced once at boot. A per-call
counter that folded them in would restate a boot number once per layer per step and
make the ratio traceable to neither.

## What I learned
1. **The split read agrees with the streamed one on the padded row, including the
   NaN, and I expected to have to make that true.** A bucketed decode step pads rows
   up to a bucket and gives the padding `context_lens == 0`, which is
   `check_pad_inert`'s contract. Day 63's `reduce_partials` refuses that row on the
   host with a good reason: every partial max is `-inf`, the joint max is `-inf`, the
   rescale is `(-inf) - (-inf)`, and the output is NaN rather than an error. I went
   looking for how to soften it and found that the *kernels* never call it. Both
   passes are written as programs, both do `exp(m_s - joint)` inline, and both produce
   exactly the NaN the streamed read has produced on a padded row since Day 59.
   `reduce_partials` is the plain-torch oracle the programs are graded against, not
   the thing on the read path, and the strictness that is right in an oracle would
   have been wrong in the read. A third read that raised where the second one shrugged
   would have made the flag change what a server *accepts*, which is precisely the
   failure the "both branches refuse what the oracle refuses" rule exists to stop.
2. **The split read needs `bucket_decode` and the other two do not, and I put that in
   the constructor rather than at the first step.** The arena is partitioned once,
   from one mapping width, and reserved for the life of the process; an unbucketed
   cache presents whatever width its longest row happens to have this step. The
   streamed read survives that because its loop bound is each row's own length, so a
   width it never reads costs it nothing. A split *partitions* the width, so a
   different width is a different partition of the same history, and Day 64's
   `check_workspace_covers` would refuse it correctly at the first decode step, in the
   middle of a warm-up, three files away from the flag that caused it. The reason is a
   property of the configuration and not of any batch, so the refusal belongs where
   the configuration is assembled.
3. **`attach` is idempotent on the same workspace and refuses a different one, and the
   asymmetry is not fussiness.** A recorded graph is a list of launches bound to the
   pointers it recorded. Swapping the arena under a read that has already been
   captured leaves the replay writing into storage nothing reads and reading storage
   nothing writes, and what comes out is finite, plausible and one step stale. So the
   check is on identity rather than on shape: two arenas of exactly the same shape,
   dtype and device are not interchangeable here, and that is the only gate in this
   repo where "equal" and "the same" come apart in a way that matters.
4. **The chunk count had to be planned before the bucket set existed, and pulling one
   line out of a constructor is what fixed it.** `plan_splits` is a max over
   `choose_splits` of every row bucket in the list, and `DecodeBuckets` now needs the
   count to be constructed. Building a throwaway set to read its rows off would work
   and would also be a second construction whose arguments can drift from the real
   one's. `row_axis(max_batch_size)` is the four lines that were inline since Day 52,
   named, and both the set and the split count now come off the same call.
5. **The counter was the only witness on Day 60 and the split makes that argument
   sharper, because now two servers can differ in a field that is not the mode.** Day
   62 added `backend` because `mode` is what the operator asked for and the backend is
   what the box could give them. `splits` is the third question in that family and it
   is the one a memory budget turns on: two servers both reporting `split` on `triton`
   can be holding arenas a factor of sixteen apart, and `rows * heads * splits *
   (head_dim + 2)` is why. Nothing else in the payload would say so.
6. **A read with no arena reports `splits: 0`, which is also what a rectangle reports,
   and I left the ambiguity in on purpose.** The alternative is a fourth field or a
   null, and both of those cost a reader something on every payload to disambiguate a
   state that lasts a few milliseconds at boot. A process in that state has completed
   no decode step, so `calls` is 0 next to it and the pair already says which. That is
   the cheapest place to put the answer, and the gate that actually cares
   (`check_read_matches`) has the mode in hand and does not have to infer anything.
7. **The wiring table's charges are not comparable across its own rows, and that is
   Day 61's note arriving as an inconvenience rather than as a paragraph.** The
   rectangle arm still buckets the width and rounds a 9-token history up to 128; the
   other two read at the table's full 2048. So the rectangle is charged 4096 cells and
   the others 65536, and the savings in that column each describe their own arm and
   nothing across it. It is the honest way round, because the numerator really is the
   width that arm's read would have gathered, but it means the only table where the
   three reads are directly comparable is the one that holds the width fixed by hand.

## Diagram
[read-becomes-three-modes.png](../diagrams/read-becomes-three-modes.png). Left top is
the two-step arrival: the flag at construction, the arena at boot, the step, and the
refusal that lives in the gap. Right top is what each read holds and which of them
needs an arena, with `tiled` and `streamed` side by side. Left bottom is the whole
gate matrix, three passes against twelve refusals. Right bottom is the serving-size
table in both currencies, the published saving and the partials that are not in it.

## Tomorrow
The flag stops at the cache. Day 66 is threading it the rest of the way:
`Engine.build(split_read=True)`, `build_engine` and `build_app`, a `--split-read`
argument next to `--streamed-read`, and the one call on the boot path that turns
`plan_capture(split_read=True)` into an allocated arena, which is Day 64's
`allocate_for` finally having a caller. The gate that comes with it is the one this
day did not need: `check_capture_ready` refuses a split cache whose arena was never
attached, and the boot line should print `SplitWorkspace.render` next to
`CapturePlan.pool_bytes` so the two numbers a split server reserves appear together
rather than one of them appearing at all.

The acceptance run wants a third arm after that. `test_the_same_engine_answers_the_same_bytes_on_either_read`
is two live servers over a socket and the split makes it three, which is the only
comparison that settles whether a read that agrees to 1e-5 puts the same token on the
wire. It is slow on this box because every arm is a tlsim loop in Python, so it is a
day's decision and not a footnote.

And the caveat is unchanged and I am not going to dress it up. The split is wired, the
counters publish it, fifteen (set, read) pairs are enumerated and twelve of them
refuse, and not one of those is a second. `graphbench.py --weights ./weights --device
cuda --rates 1,2,4,8` on all three arms is still the first run to book hardware for,
and it has been the first thing on that list since Day 58.

## Post angle
Day 65 of building an LLM inference engine from scratch. Today the flash-decoding read
finally got wired into the engine as a third mode beside the rectangle and the stream,
and the interesting part was not the dispatch. It was that this is the first read in
the whole repo that cannot run on its own. The rectangle needs a pool and a mapping.
The streamed read needs those plus a tile width, which it carries itself, because a
tile is just an integer. The split needs an arena somebody else allocated, at a width
somebody else fixed, and yesterday is why: three `torch.empty` calls inside a captured
read are perfectly legal and completely unpriced, so the workspace became something
the plan owns. But a read object is constructed when the cache is, and on the boot
path that is before anything has been moved to a device. So there is a window where
the mode is set and the memory is not, and only three things can happen in it. The
read allocates its own, which undoes the entire previous day. It quietly does
something else, which is a flag that means one thing in the log and another in the
process. Or it refuses, and the operator gets a sentence instead of a mystery. The
second thing I did not see coming: my width-axis gate had been asking the wrong
question for four days without being wrong. Day 61 collapses the bucket set for a read
that holds a tile instead of a rectangle, and I called that predicate `streamed`,
because with two reads "holds a tile" and "is the streamed read" pick out the same
object. They are not the same question, they were the same answer, and a third read is
exactly what separates them. And the saving a server publishes means a third thing
now. A split does not make the tile smaller, it makes more of them alive at once,
because the split count is a grid axis. So a split server reports 16x where its
streamed neighbour reports 256x, and that is not a broken streamed server, it is the
trade: 4,194,304 score cells held instead of 262,144, plus 76.55 MB of partials the
quotient does not contain at all. vLLM and SGLang both ship the split and the unsplit
kernel side by side rather than replacing one, and building the wiring is what made me
understand why that is a scheduling decision and not a leftover. 2154 green, fifteen
(set, read) pairs enumerated and twelve of them refused, and still not one measured
second.
