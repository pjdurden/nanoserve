---
title: "Day 64: the allocation was legal, it was just unpriced"
parent: Daily log
nav_order: 64
---

# Day 64: the allocation was legal, it was just unpriced

Date: 2026-09-20 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
`src/nanoserve/partials.py`: the split read's workspace, allocated once by the plan
and handed to the kernel as a window, the way Day 51's slot table already hands it a
mapping.

Day 63 ended on a caveat I wrote too confidently. It said a captured region cannot
allocate, so the partials have to become a planned buffer. The first half of that is
wrong. A `torch.empty` made while a CUDA graph is recording is served from the
graph's memory pool, its address is baked into the replay exactly like every other
intermediate's, and the launch is perfectly legal. What is wrong with it is quieter,
and it is the real reason for the day: **nothing priced it.** `CapturePlan.pool_bytes`
was `shared_pool_bytes` over the score term alone, so a process running the split read
reserved an arena it had under-reported by the whole workspace, once per graph, at the
largest shape in the list. At serving size that is 1.05 MB reported against 84.93 MB
spent, and the allocator was the only thing in the process that knew.

`SplitWorkspace` is the three buffers, `[max_rows, n_q, splits]` twice and
`[max_rows, n_q, splits, head_dim]` once, in fp32. `allocate_partials` reserves them
keyword-only, and `allocate_for` builds one off a `CapturePlan` by duck typing so the
dependency runs down the boot path and not up it. `SplitWorkspace.rows` is the window,
`is_window` and `addresses` are the witnesses, and `check_workspace_covers` is the
plan-shaped gate: a mapping of a width the partition was not computed from, a model
with the wrong head count, a batch wider than the row ceiling.

`plan_splits` in `kernels/flash_decoding.py` is one split count for a whole capture
list, and `check_partials` is the buffer-shaped gate inside both kernels.
`paged_attention_split_kernel`, `paged_attention_split_triton` and
`paged_attention_split` all take `partials=`, and when nobody names a `splits` the
handed-in arena answers the question.

`captured.py` grows `partial_cells`, `split_score_cells`, `split_workspace_bytes` and
`PARTIAL_ITEMSIZE`, and `read_workspace_bytes` becomes a three-way dispatch.
`launch.py` grows `CapturePlan.splits` and `CapturePlan.head_dim`, and
`plan_capture(split_read=True)` calls `plan_splits` once, at plan time, on the width
the bucket set fixed.

`tests/test_partials.py` is 26 tests; `tests/test_flash_decoding.py`,
`tests/test_captured.py` and `tests/test_width_axis.py` grow the rest. Suite
**2115 green** (21 GPU-gated skips), ruff clean.

`arenabench.py`, host only, no weights
([day-64-arenabench.csv](data/day-64-arenabench.csv)):

    a planned arena against the oracle (7-wide mapping, 2-key tiles):
      1 splits of 8 keys, 0.0024 MiB: worst |planned - oracle| 3.58e-07 over 64 reads, addresses moved: False
      2 splits of 4 keys, 0.0049 MiB: worst |planned - oracle| 2.38e-07 over 64 reads, addresses moved: False
      4 splits of 2 keys, 0.0098 MiB: worst |planned - oracle| 3.58e-07 over 64 reads, addresses moved: False

    the shared capture arena at 256 rows, 32 heads, 8192 tokens, 32-key tiles:
                  read  shapes          arena  vs rectangle
             rectangle      36      268.44 MB          1.0x
              streamed       9        1.05 MB        256.0x
        split (16-way)       9       84.93 MB          3.2x

    one arena over a 9-bucket list, 16 splits of 512 keys:
     rows  wants  programs   this shape    charged
        1     16       512      0.33 MB    0.33 MB
        2     16      1024      0.66 MB    0.66 MB
        4     16      2048      1.33 MB    1.33 MB
        8      8      2048      1.33 MB    2.65 MB
       16      4      2048      1.33 MB    5.31 MB
       32      2      2048      1.33 MB   10.62 MB
       64      1      2048      1.33 MB   21.23 MB
      128      1      4096      2.65 MB   42.47 MB
      256      1      8192      5.31 MB   84.93 MB

    allocator calls for the partials, 16 layers, 1000 decode steps:
                 owner  per step    per run
              the read        48      48000
              the plan         3          3

## Why it matters
**The bug was not in the allocation, it was in the accounting, and those fail
differently.** An allocation that is illegal under capture fails loudly on the first
recording, in front of whoever is warming the server, with a traceback naming the
graph. An allocation that is legal and unpriced does not fail at all until the arena
is 81 times the size the plan printed at boot, and then it fails as an out-of-memory
error somewhere else, on a shape the plan never mentioned. Day 54 built `pool_bytes`
so a server that cannot start can explain itself in one line, and a read that spends
memory outside that line makes the line into a lie rather than an underestimate. The
fix is not the buffer. The fix is `split_workspace_bytes` existing, and the buffer is
what makes it checkable.

**Only the row axis narrows, and the alternative does not raise.** Both passes address
the workspace flat, `(row * n_q + head) * splits + split`, because that is the form
`tl.store` takes. The arena is allocated at the list's widest split count, so the
obvious way to hand a smaller launch its slice is `buffer[:, :, :splits]`, which is a
tensor of exactly the right shape, the right dtype and the right device whose stride
is the *allocated* count. The flat arithmetic does not know that. Every program past
the first would read and write another `(row, head)`'s slot, the reduction would fold
whatever it found, and the read would return a finite, plausible vector that is not
the attention over that row. A window on the outermost axis of a contiguous buffer is
still contiguous and still starts at the base address, which is the one narrowing that
survives, and it is exactly why `check_partials` refuses on contiguity rather than on
shape.

**The split count has to be one number for the list, and the max is the only direction
that keeps the split.** `choose_splits` answers per launch. A capture list is not one
launch: it is a graph per shape, all replaying against one arena, so the per-bucket
answers collapse. Rounding a bucket *up* costs it chunks it does not fill, and Day 63
made that free, because an empty chunk walks nothing, stores `-inf, 0, 0` and drops
out of the reduction with no branch. Rounding *down* takes the splits off the one-row
batch, which is the only batch a split was ever for. So `plan_splits` is a max, and
the whole design of the empty chunk is what pays for it.

**And the two maxima are at opposite ends of the same list.** The split axis peaks at
the narrowest bucket, because a small grid is what needs filling: `choose_splits(1)` is
16 and `choose_splits(256)` is 1. The arena is `rows * heads * splits * (head_dim + 2)`
and peaks at the widest, because a ceiling division of a program target makes
`rows * splits(rows)` climb to that target and then flatten rather than fall. So one
rectangular arena is the product of both maxima, 84.93 MB, while the largest single
shape in the list needs 5.31 MB. **16x, and it is not a bug.** The alternative is a
split count per row bucket, which is a different `tl.constexpr` per bucket, which is a
compiled kernel per bucket: the specialisation Day 61 spent a whole day collapsing,
bought back on a different axis. One compiled body against a fatter arena is a trade,
and the only thing I can do about it today is measure it and say which way I went.

## What I learned
1. **I wrote a caveat yesterday that was true in its conclusion and wrong in its
   reason, and the wrong reason would have sent somebody to the wrong fix.** "A
   captured region replays the addresses it was captured with, so it cannot allocate"
   is a sentence I believed hard enough to put in a module docstring. It is not how
   graph capture works: allocations under capture go to the graph's pool and replay at
   the same address, which is the entire purpose of a pool handle. Had the reason been
   the real one, the fix would have been "make the read not allocate" and it would have
   stopped there. The real reason is "nothing prices it", and that fix is a pricing
   function, a plan field and a budget check, which is most of today.
2. **A gate on contiguity is a gate on arithmetic, and it took writing the wrong slice
   to see it.** I reached for `buffer[:, :, :splits]` first, because that is the shape
   the launch wants and PyTorch hands it over without a word. The refusal is not about
   memory layout as a tidiness matter: the kernel does not *take* a tensor, it takes a
   pointer and an integer it multiplies, and a slice is a promise about strides the
   pointer never sees. Every gate in this repo that reads a `.data_ptr()` is the same
   shape of thing, and this is the first one that reads `.is_contiguous()`.
3. **Day 63's empty chunk turned out to be load-bearing twice, and the second time was
   a day later in a different file.** It was built so pass two could read a slot
   unconditionally. It is *also* what makes a reused buffer safe, because a program
   that always stores means no step can see the step before it, and it is what makes
   `plan_splits`'s max cheap, because a bucket given surplus chunks pays nothing for
   them. I took the `-inf` fill out of the tlsim path expecting to have to justify it
   and found there was nothing to justify: the fill had never been read.
4. **`torch.empty` and not `zeros` is a claim about the kernel, not a saving.** The
   saving is nothing; the arena is written before it is read either way. What the
   choice says is "no slot's prior contents matter", and a `zeros` there would be a
   quiet assertion of the opposite that nobody would ever test. Writing the honest one
   is what made the garbage-filled-workspace test worth having.
5. **The head_dim is the first axis in this engine that a workspace has and a score
   rectangle never did.** Every memory number from Day 54 to Day 61 is
   `rows * heads * query * something`, because scoring sums the channels away before
   the intermediate exists. A partial accumulator has not been divided yet, so it is a
   head wide, and `read_workspace_bytes` refuses to price a split without one rather
   than inferring it. Small, and it is the clearest statement I have of what the two
   reads actually hold.
6. **Two itemsizes in one workspace, and it is the only one in the file.** The score
   tiles follow the pool and halve on a bf16 card. The partials do not, because a
   different program rescales them by `exp(m - M)` after a round trip through memory,
   and that trip is the one place in this read where the range is load-bearing.
   Pricing the arena at one itemsize would be wrong in whichever direction the
   deployment chose, so `split_workspace_bytes` adds two terms in two currencies and
   `PARTIAL_ITEMSIZE` is read off the dtype the kernel really allocates.
7. **48 allocator calls a step is the number that made the eager case concrete.** The
   capture argument is the interesting one and it is also the one that only matters on
   a card that is running graphs. Outside a graph the read allocates three tensors per
   layer per step, which on a 16-layer model over a thousand steps is 48,000 calls to
   hold bytes that never change size. That is not the headline and it did not need to
   be; it is the version of the claim that is true on this box.

## Diagram
[arena-becomes-planned.png](../diagrams/arena-becomes-planned.png). Left top is who
owns the three buffers, 48 allocator calls a step against 3 at boot, with the note
that nothing is zeroed. Right top is the flat addressing and the two slices, the row
window in green and the split slice in red with its unchanged stride. Left bottom is
the bucket table with `wants` against `charged` and the 16x. Right bottom is the arena
in all three reads' currencies and the 81x under-report.

## Tomorrow
The split is now wireable. Day 65 is `PagedRead` growing a third mode: `SPLIT`
alongside `RECTANGLE` and `STREAMED`, holding a `SplitWorkspace` the cache hands it,
charging `split_score_cells` into the same counters Day 60 built, and reporting the
split count next to Day 62's backend in the health payload. The gate is
`check_read_matches`'s: a split bucket set under a streamed read is a plan whose arena
nobody addresses, and a streamed set under a split read is a launch nobody sized.

The 16x arena is the thing I would fix next if the wiring did not come first. There is
a design that gets it back: allocate the arena *flat* rather than as a rectangle, size
it by `max(rows * heads * splits(rows))` over the list instead of by the product of
the two maxima, and hand each shape a contiguous prefix reshaped to its own grid. That
costs a compiled body per row bucket, so it is a real trade and not an oversight, and
it wants a measurement on a card to settle rather than an opinion here.

The caveat has not moved and I will not dress it up. Six kernels' worth of arithmetic
is tested, the arena is priced to the byte, the addresses hold across 64 reads, and
not one of those is a second. `graphbench.py --weights ./weights --device cuda --rates
1,2,4,8` on both arms is still the first run to book hardware for, and it has been the
first thing on that list since Day 58.

## Post angle
Day 64 of building an LLM inference engine from scratch. Yesterday I wrote a caveat I
was sure about: a CUDA graph replays kernels bound to the addresses they were recorded
with, so the flash-decoding workspace cannot be a `torch.empty` inside the read. That
is wrong, and being wrong about it is the most useful thing that happened this week.
An allocation made while a graph is capturing is served from the graph's own memory
pool and replayed at that address, like every other intermediate. It is legal. What is
actually wrong with it is much quieter: nothing *prices* it. My capture plan printed
the arena at boot from the score term alone, so a process running the split read
reserved 1.05 MB on paper and spent 84.93 MB in fact, once per graph, at the largest
shape in the list. An illegal allocation fails in front of you during warm-up. An
unpriced one fails as an out-of-memory error later, somewhere else, on a shape your
boot line never mentioned. So today the partials become a buffer the plan owns and
hands to the kernel as a window, which is the same move I made on the slot mapping on
Day 51, and the point of it is the pricing function rather than the buffer. Two things
I did not expect. First, only the *row* axis of that arena can narrow. Both passes
address it flat as `(row * n_q + head) * splits + split`, because that is the form
`tl.store` takes, so slicing the split axis gives you a tensor of exactly the right
shape whose stride is still the allocated count: every program past the first writes
its neighbour's slot, nothing faults, and the read returns a finite plausible vector
that is not the answer. The gate is on contiguity, not on shape. Second, the split
count and the arena size peak at opposite ends of the same capture list. A one-row
batch asks for 16 splits because a small grid is what a split is for; a 256-row batch
asks for 1 because the card is already a queue. But the arena is rows times splits, so
it peaks at 256 rows. One rectangular arena over the list is 84.93 MB while the largest
single shape in it needs 5.31 MB. 16x, and it is not a bug: the alternative is a
`constexpr` per row bucket, which is a compiled kernel per bucket, which is exactly the
specialisation Day 61 spent a day collapsing. vLLM and SGLang both ship a fixed
partition for the same reason. 2115 green, the arena priced to the byte, the addresses
unmoved across 64 reads, and still not one measured second.
