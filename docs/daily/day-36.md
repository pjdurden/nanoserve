---
title: "Day 36: the edge cases, and the watermark finally measured"
parent: Daily log
nav_order: 36
---

# Day 36: the edge cases, and the watermark finally measured

Date: 2026-08-10 · Week 9 · Phase 3 Batching and scheduler

## What I added today
The three things the daily log has been putting on the "tomorrow" line since
preemption landed, now that Day 35 built the harness that makes them worth writing
down. `tests/test_edges.py`, thirty-four tests: aborting a request that is currently
preempted, a prompt whose length is an exact multiple of the block size, and the
watermark, a 1% default since Day 30 that nobody had ever measured.

The first two held, and they are in the file as regressions rather than as fixes. A
preempted request holds nothing, so an abort of one has nothing to release and the
release path walks both queues already; a prompt that exactly fills its blocks needs
`blocks_for_length(n)` blocks and the ceiling is right on the boundary, so its first
decode buys a block on iteration one instead of on iteration five and every ordering
question the grow-then-admit loop answers gets asked immediately. An edge that works
by accident and an edge that works by design look identical until something asserts
on it, which is the only reason those tests exist.

The third did not hold. `Scheduler` gains `watermark_blocks` as a constructor
argument (the reserve named in the unit it is spent in, not as a share), an
`admissible_blocks` property, and a one-line change in `_admit` that is the day's
bug fix. `batchbench.py` gains `WatermarkPoint` and `sweep_watermarks`, which runs
one workload through several reserve sizes and counts what each changed, with no
model in it: what a watermark moves is preemptions and iterations, and both are
integers the scheduler computes without a tensor in sight. Suite **529 green** (5
GPU-gated skips), ruff clean.

## Why it matters
**The watermark could be a wall.** It is subtracted at admission and ignored during
growth, and both halves of that are deliberate: a newcomer should not take the
pool's last block, and a running row must never be starved by an accounting rule.
The gap between them is that a preempted request has to be *re-admitted*, and
admission is where the subtraction happens.

A pool of 6 blocks of 4, a reserve of 3, two requests, and nothing here is misuse:

    b is admitted holding 1 block        well inside what the reserve leaves
    b grows to 4 blocks                  growth ignores the reserve, on purpose
    b is preempted                       blocks back, 13 tokens kept, K/V gone
    a finishes, the pool is empty        b needs 4. the reserve leaves 3. refused.

Six blocks free, nobody running, one request waiting, and no future iteration
changes any of those three numbers. Both queues stop and the engine spins forever. A
fresh request whose prompt alone is larger than the reserve leaves hangs the same
way, having been told yes at the door: `add_request` refuses anything too large for
the whole pool, so passing the door is a promise to run it.

The fix is one line, and the sentence it enforces is the invariant that was missing:

    reserve = self.watermark_blocks if need <= self.admissible_blocks else 0

The watermark may delay a request, it may never exclude one. It drops the reserve
and nothing else, so the request still waits for blocks that really exist, and that
is what makes it terminate: the oldest running request can never be preempted, so it
finishes, so the pool drains, and the door already proved this request fits an empty
one.

**And the measurement, which is the other half.** First, what the default actually
reserves: `int(0.01 * num_blocks)` is 0 for every pool under a hundred blocks, which
is every pool in this repo's tests and every pool in its benchmarks. The default is
not wrong, it is inert. A knob that has never moved has never been measured either.

Given real blocks it does what it claims. Eight requests, three slots, a pool of
eight blocks of four, budget 10:

| reserve | evictions | of those, thrash | iterations | recompute share |
|---|---|---|---|---|
| 0 blocks | 4 | 4 | 38 | 23.7% |
| 1 block | 3 | 3 | 41 | 18.8% |
| 2 blocks | 1 | 1 | 41 | 7.0% |
| 3 blocks | 0 | 0 | 41 | 0.0% |

Three blocks of eight held back removes the whole recompute bill for three extra
iterations. The thrash became waiting, which is still time somebody pays, and the
counter-case is in the same sweep: on a pool of 16 blocks and 6 slots the same
reserve removed nothing at all and still cost four iterations, and on a roomy pool it
does nothing in either direction. It is a brake. A brake is not always what you want.

## What I learned
1. **A resource policy that two code paths disagree about is a deadlock waiting for
   a workload.** Admission subtracts the watermark and growth does not, and each of
   those decisions is right on its own. What is wrong is that nothing owned the
   relationship between them, so a request could legally grow into a size that
   admission would never again accept. The bug is not in either rule, it is in the
   space between them, and it was invisible for six days because the default reserve
   is zero and zero makes the two rules identical.
2. **The mechanism in the comment was not the mechanism in the data.** Day 30 wrote
   that a newcomer takes the pool's last block and is evicted a step later. What the
   sweep shows is that every victim was evicted exactly three iterations after being
   admitted, holding exactly three generated tokens: a 6-token prompt in blocks of 4
   is admitted with two free slots in the tail of the block it bought, spends them
   one token per iteration, and runs the pool dry when it reaches the boundary. The
   delay is the tail, not a step. So `thrashed_admissions` counts evictions of a
   request that had not yet filled the block its admission bought, and with the
   1-iteration window I wrote first it counted 1 where the block-wide window counts
   4. The metric that matched the story I believed would have reported that the
   watermark fixes almost nothing.
3. **The unit a knob is configured in should be the unit it is spent in.** A share
   of the pool sounds portable and is, at four thousand blocks. At eight blocks it is
   a rounding error with an `int()` in front of it, and the truncation does
   arithmetic the caller never asked for. Keeping the share as the default is still
   right, because a real deployment sizes its pool in thousands of blocks and
   "reserve a little" is what it means; the fix is that the block count is now
   sayable, which is also the only way a sweep can state what it swept.
4. **Two of the three edges held, and finding that out cost the same as finding a
   bug.** Writing an abort-while-preempted test is the same work whether it passes
   or fails, and the passing version is worth what the failing one would have been:
   next time somebody changes the release path, that edge is no longer relying on
   nobody having touched it.

## Diagram
[watermark-measured.png](../diagrams/watermark-measured.png). Left is the deadlock as
a four-step trace with the six-block pool drawn under each step, b's holding growing
from one block to four and coming back as nothing, ending on the empty pool that
still refuses it, then the one-line fix and the sentence it enforces. Right is the
default measured (every pool under a hundred blocks reserves zero) over the sweep
table, with the counter-case underneath. The banner is the mechanism correction: the
victims were evicted three iterations after admission holding three tokens, so the
thrash window is a block wide and not a step.

## Tomorrow
Week 9 closes here and Phase 4 opens: the engine gets an HTTP server in front of it.
`server.py` has been a docstring and three TODOs since Week 1. Day 37 is the first
of them, the FastAPI app and the request bridge, and the interesting problem is the
one the file's own docstring names: `Engine.step()` is synchronous and owns the GPU,
so the server cannot call it per request. One loop pulls from a queue, one queue is
fed by many coroutines, and each of them has to be woken with its own tokens when
its own request finishes rather than when the batch does, which is exactly the
property Week 8 built and nothing outside the engine has consumed yet.

## Post angle
Day 36 of building an LLM inference engine from scratch. Today I measured a knob I
wrote six weeks ago and had never once switched on. The watermark: admission refuses
to spend the last few percent of the KV block pool, so a newcomer cannot take the
last block and get evicted for it a step later. Default 1%. Three things fell out.
One: `int(0.01 * num_blocks)` is 0 for any pool under a hundred blocks, which is
every pool this repo has ever run, so the feature has been switched off since the day
I wrote it. Two: given actual blocks it works. Eight requests through three slots and
a pool of eight blocks, holding three of them back took evictions from 4 to 0 and the
recompute bill from 23.7% of the forward to zero, for three extra iterations. The
thrash became waiting, which somebody still pays for, and on a bigger pool in the
same sweep the identical reserve removed nothing and still cost four iterations. It
is a brake, not a free win. Three, and this is the one worth the day: the watermark
could deadlock the engine. Admission subtracts the reserve and growth ignores it, and
both of those are correct in isolation. But a request that gets admitted holding one
block, grows to four, and is then preempted has to be admitted *again*, needing four
at once, into a rule that leaves three. Six blocks free, nothing running, one request
waiting, and no iteration can change any of those numbers. It spins forever. The fix
is one line and the invariant it enforces is a sentence: the watermark may delay a
request, it may never exclude one. Also: the comment explaining why the watermark
exists described the wrong mechanism. Victims are not evicted a step after admission,
they are evicted when they run out of the free tail of the block admission bought
them, which here was exactly three iterations and three tokens later. My first thrash
metric used a one-iteration window and counted 1 where the block-wide window counts 4.
A metric that matches the story you already believe is not a measurement. vLLM has
the same watermark for the same reason. 529 green.
