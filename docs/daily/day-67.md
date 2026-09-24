---
title: "Day 67: the partials got one owner, and the owner had the right rows"
parent: Daily log
nav_order: 67
---

# Day 67: the partials got one owner, and the owner had the right rows

Date: 2026-09-23 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" had two items. I did the small one: deciding which boot line
owns the split read's partials. It turned out not to be a matter of taste. Answering
it turned up a second, worse bug in the budget probe.

**`captured.py`.** `split_workspace_bytes` is now the sum of two new functions, and
its value hasn't changed. `split_tile_bytes` is the score tiles, one per chunk, which
is what a CUDA graph's pool holds. `split_partial_bytes` is the max, denominator and
accumulator per (row, head, chunk), which is what `SplitWorkspace.bytes` reports.
Each half keeps the refusal the total had for its own itemsize.

**`launch.py`.** `CapturePlan` gets one field and two properties:
- `arena_rows` is the row count the partials arena is allocated for: the scheduler's
  slot count under a split, 0 otherwise.
- `pool_bytes` under a split is now the tiles only.
- `partial_bytes` is the arena priced at `arena_rows`.
- `reserved_bytes` is the sum, and it's the number the budget probe is asked about.

`plan_capture` sets `arena_rows = plan.max_batch_size` and checks tiles at the
recorded rows plus partials at the arena rows. Its refusal now says that lowering
`max_rows` doesn't shrink the partials. `describe` ends a split's capture line with
"partials on the split workspace line". `check_arena_matches_capture` gains an
equality: `capture.partial_bytes == workspace.bytes`, or `BootUnsound`.

`tests/test_partial_owner.py` has 15 new tests. One Day 64 test in
`tests/test_width_axis.py` now asserts its total against `reserved_bytes` instead of
`pool_bytes` (explained below). Suite **2195 green** (21 GPU-gated skips), ruff clean.

`ownerbench.py`, host only, arithmetic only
([day-67-ownerbench.csv](data/day-67-ownerbench.csv)):

    a 16-way split server, 256 slots, 8192 tokens, 32 heads x 128, 32-key tiles:
    warm-rows          capture line   arena line  lines summed   probe asked        held
          256   day66      84.93 MB     68.16 MB     153.09 MB      84.93 MB    84.93 MB
          256   day67      16.78 MB     68.16 MB      84.93 MB      84.93 MB    84.93 MB
           64   day66      21.23 MB     68.16 MB      89.39 MB      21.23 MB    72.35 MB
           64   day67       4.19 MB     68.16 MB      72.35 MB      72.35 MB    72.35 MB
            8   day66       2.65 MB     68.16 MB      70.81 MB       2.65 MB    68.68 MB
            8   day67       0.52 MB     68.16 MB      68.68 MB      68.68 MB    68.68 MB
            1   day66       0.33 MB     68.16 MB      68.49 MB       0.33 MB    68.22 MB
            1   day67       0.07 MB     68.16 MB      68.22 MB      68.22 MB    68.22 MB

## Why it matters
**The graph pool never holds a partial, so the capture line had no claim to them.**
Day 64 priced the partials into `pool_bytes` because an allocation made while a
graph records is served from the graph's pool. That argument was true for about a
day. The same day moved the partials into an arena allocated once. Day 65 made
`PagedRead` refuse a split launch that wasn't handed that arena, so no recording can
allocate a partial. Day 66 printed the arena on its own line, which meant the same
bytes appeared twice: at serving size, 153.09 MB across two boot lines for 84.93 MB
actually held. The choice of owner follows from where the bytes live. The pool is
the tiles, the arena is the partials, and the capture line says where the partials
went, so a reader who knows a split holds them doesn't go looking in the wrong number.

**The more serious bug went the other way, and only showed up once the two numbers
were separated.** The capture list is sized by the *recorded* rows, which
`--warm-rows` can trim. The arena is sized by the *served* rows, the slot count,
because a shape the list skipped still runs eagerly against the same buffers (Day 66
learned this, and the arena's rows became an inequality). But Day 64's budget check
priced tiles and partials together at the widest *recorded* shape. So a
`--warm-rows 1` server asked the probe about 0.33 MB and then reserved 68.22 MB, about 200x more
than it checked. The boot log over-reported with the full list and the probe
under-asked with a trimmed one. Two opposite errors, one cause: the partials had no
owner and no row count of their own.

**The fix is a field, not a formula.** `arena_rows` is the one number the plan was
missing, and every other change follows from it. `partial_bytes` is priced at it.
`reserved_bytes` is what the probe checks. `check_arena_matches_capture` holds the
priced bytes equal to the allocated bytes. Day 66's rows clause was a bound, about
what's recorded. This one is an equality, about what was priced. They're different
questions, and a `--warm-rows` server needs the answer to both.

## What I learned
1. **Day 66 guessed the double count at 76.55 MB. It's 68.16 MB.** Since the count
   is in `ownerbench.py` now, it's a number rather than a guess: `256 x 32 x 16 x
   (128 + 2) x 4` bytes. The guess was in the right range, and this whole week has
   been about ranges not being numbers.
2. **The Day 64 test changed what it asserts, and I left it in place instead of
   deleting it.** `test_the_arena_of_a_split_plan_is_the_tiles_plus_the_partials`
   still asserts the Day 64 total, because that total is still the right answer to
   "what does this plan commit the card to". What changed is which property holds
   it: `reserved_bytes` now, not `pool_bytes`. The test's docstring says so and
   points here.
3. **Without the partials, a split's pool is exactly `splits x` the streamed pool.**
   That's a check you can do in your head, and it's now a test. Before today the
   ratio was 81x at serving size and meant nothing. Now it's 16x, the split count,
   which says the pool holds one tile per chunk and nothing else.
4. **Lowering `--warm-rows` had never been a way to fit a split on a smaller card.**
   The old refusal told the operator to lower `max_rows`, which shrank the number
   being checked and not the memory being reserved. The new message says the
   partials are sized by the slot count and names the two knobs that actually move
   them: batch size, or no split.
5. **I put `partial_bytes` in neither `as_dict` nor `describe`, and that was
   deliberate.** The arena reports the same number under `split_workspace` once it
   exists. A second key under `cuda_graphs` would recreate the double count in
   `/health`, one day after removing it from the log. The plan keeps its own copy
   for the one caller that needs it before the arena exists, which is the probe.

## Diagram
[partials-one-owner.png](../diagrams/partials-one-owner.png). Top left: a split
launch's two halves and where each one lives. Top right: the row count each number
is sized by, with Day 66's probe in red. Bottom left: `ownerbench.py` at 256 and 1
warm rows, before and after. Bottom right: the boot lines with each byte printed
once, and the new equality.

## Tomorrow
The other half of yesterday's "Tomorrow" is still open: the acceptance run's third
arm. `test_the_same_engine_answers_the_same_bytes_on_either_read` compares two live
servers over a socket, and a `--split-read` server can now boot into it. On this box
every arm is a Python tlsim loop, so Day 68 is either that run with `PLANS` cut down
to stay under a minute, or a written decision that it waits for the card.

The hardware caveat hasn't changed: `graphbench.py --weights ./weights --device cuda
--rates 1,2,4,8` on all three arms is the first run to book, and the flag it was
waiting on has existed since yesterday.

## Post angle
Day 67 of building an LLM inference engine from scratch. Yesterday's note said my
flash-decoding server printed its partials twice on the boot log: once inside the
CUDA graph workspace, once on the arena's own line. Deciding which line owns them
looked like a style choice. It wasn't. A split launch never allocates a partial
under capture, because the read refuses to run without the arena, so the graph pool
only ever holds the score tiles. Separating the two numbers exposed the real bug.
The capture list is sized by the rows you record, and `--warm-rows` can trim that.
The arena is sized by the rows you serve, because a skipped shape still runs
eagerly against it. The budget probe priced the partials at the recorded rows. At
`--warm-rows 1` on a 256-slot server, the probe checked 0.33 MB and the process
reserved 68.22 MB. The fix is one field, `arena_rows`, plus a boot gate that holds
the priced bytes exactly equal to the allocated bytes. vLLM and SGLang both size
these workspaces up front at startup. Building it myself showed me that "up front"
only helps when every number is priced at the rows it's actually allocated for.
2195 green, and still not one measured second.
