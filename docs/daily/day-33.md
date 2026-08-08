---
title: "Day 33: blocks a step at a time, and the eviction that pays for it"
parent: Daily log
nav_order: 33
---

# Day 33: blocks a step at a time, and the eviction that pays for it

Date: 2026-08-07 · Week 9 · Phase 3 Batching and scheduler

## What I added today
Incremental allocation, and the preemption that has to come with it. Week 8 admitted
a request by reserving `worst_case_tokens`, prompt plus every token it was still
allowed to emit, which bought one very useful property: a running request already
owned every block it could ever need, so no forward could fail and no rollback path
had to exist. It also booked the pool for text nobody wrote. Today admission buys
`blocks_for_length(num_tokens)` instead, and `Scheduler.blocks_needed_for` is the
only allocation rule left in the file. It reads the same for every state a request
can be in: a fresh request needs its prompt's blocks, a running one needs 0 on most
steps and 1 on the step that crosses a boundary, and a preempted one coming back
needs blocks for its prompt plus everything it generated before it lost them.

`schedule()` is now release, then grow, then admit. Growth runs oldest first and
takes its blocks from the youngest running request, which is `_preempt`: blocks back
to the pool, slot back to the free list, `output_token_ids` untouched, and the
request pushed onto the *head* of the waiting queue. That is preemption by
recompute, the RUNNING to WAITING edge Day 30 wrote into the transition table as
illegal because nothing existed to make it safe. A request with nobody younger to
evict is its own victim. The engine side is three things: `Scheduler.on_release`, a
hook the engine fills with its cache-row reset so a row is emptied inside `schedule`
rather than around it, `BlockTable.extend` so a row learns about the block the
scheduler just handed its request, and one word in the prefill, `token_ids` instead
of `prompt_token_ids`, which is the whole of the resume. Thirty-eight net new tests,
mostly `tests/test_preemption.py`, in the two halves the last few days used: the
scheduler over plain integers for allocation, victim choice and queue order, then a
tiny `LlamaModel` for the claim that actually matters. Suite **428 green** (5
GPU-gated skips), ruff clean.

## Why it matters
Four requests, a 5-token prompt each, 24 tokens of budget each, four slots,
`block_size` 4, and the same 96 tokens collected in every row of this table:

| pool | policy | iterations | occupancy | preemptions | tokens recomputed |
|---|---|---|---|---|---|
| 48 tokens | Week 8 | 96 | 0.250 | n/a | 0 |
| 48 tokens | Day 33 | 44 | 0.545 | 3 | 55 |
| 96 tokens | Week 8 | 48 | 0.500 | n/a | 0 |
| 96 tokens | Day 33 | 28 | 0.857 | 1 | 25 |
| 128 tokens | either | 24 | 1.000 | 0 | 0 |

A request that will end up 29 tokens long reserved 8 blocks on Week 8's rule, so a
96-token pool held exactly two of them and the engine ran two rows in a four-row
batch for the entire run. On today's rule the same pool starts all four (two blocks
each), and the engine averages 3.4 rows per forward instead of 2. The last line is
the honest bound: when every request really does use its whole budget, the worst
case was right all along, and at 128 tokens the two policies are the same engine.
Everything above that line is the gap between what a request might do and what it
does, which is where the concurrency was hiding.

On the Day-29 shape (eight requests, 5-token prompts, seven wanting 4 tokens and one
wanting 32, eight slots, `block_size` 4) the same change shows up as a pool size
rather than an occupancy: running all eight at once took **31 blocks** on Week 8's
rule and takes **16** now, because seven of those requests were holding three blocks
each for a 9-token life. That workload never preempts, which is the point of
measuring both: preemption is what happens when the pool is genuinely too small, not
a cost of the allocation change.

`reservation_waste` is 0.0 now and stays 0.0, by construction rather than by luck,
because held and needed are computed from the same number. What replaces it is
`fragmentation_waste`, the unfilled tail of each sequence's last block, bounded by
`block_size - 1` tokens per sequence however long the sequence gets. That is the
paging bound vLLM quotes a few percent for, and it is the reason `block_size` is a
knob and not a constant: bigger blocks mean fewer tables and more dead tail.

## What I learned
1. **Incremental allocation and preemption are one feature, not two.** I scoped this
   day as "allocate a block at a time" and tried to leave the eviction path for
   tomorrow, and there is no such day. Without the worst-case reservation the pool
   can be drained to a state where a running request has no block for its next token,
   and no admission-side watermark fixes it: I worked out a one-step lookahead
   reserve, one block per running row, and it holds for exactly one iteration, then
   the next boundary crossing arrives with an empty pool anyway. The choice is not
   "allocate incrementally, and preempt if you feel like it". It is "reserve the
   worst case, or have a recovery path", and everything else is a way of postponing
   that.
2. **The victim policy is a liveness proof, not a heuristic.** Preempting the
   youngest running request looks arbitrary next to preempting the largest, which
   would free more blocks per eviction. Seniority is what makes the loop terminate.
   The oldest running request is never a victim, so it always reaches its own last
   token, so it always frees its blocks, so the pool always drains: somebody
   finishing is the only reason this is not two requests taking turns evicting each
   other forever. And the door check from Day 30, which refuses a request whose worst
   case exceeds an empty pool, quietly became the second half of that proof. It was
   about a blocked FIFO head; it is now what guarantees that a request which has
   evicted everybody else fits in the pool it is alone with. The worst case did not
   go away today. It moved from the allocator to the doorman.
3. **Recompute has to resume, not restart, and the type system will not tell you
   which one you wrote.** The obvious implementation of "put it back on the queue"
   throws away the generated tokens with the K/V, and the engine still runs, still
   drains, still returns the right number of tokens per request, and returns
   different text. The fix is that a preempted request keeps `output_token_ids` and
   its next prefill runs over `token_ids`, so it is admitted as a longer version of
   itself and the token it samples is the one it would have sampled anyway. The bug I
   actually hit was one step subtler: the first version re-prefilled the prompt only,
   and the real 1B happily emitted a duplicate token in the middle of the sequence
   (`... 315, 279, 279, 502 ...` where it should have been `... 315, 279, 502, 220
   ...`). The test that catches this compares a cramped engine
   against a roomy one on identical prompts, and it is the only test in the file I
   would refuse to ship without.
4. **Two lists that must agree will drift, and the drift is silent.** The request's
   `block_ids` is the scheduler's record and the cache row's `BlockTable.block_ids` is
   the tensor addressing, and Week 8 got away with copying one into the other once at
   adoption because reservations never changed size. Now they change every few
   iterations. Forgetting to sync does not raise: `BlockTable.append` sees a table
   short of capacity and helpfully allocates a block of its own, the pool is booked
   twice for one sequence, and the damage surfaces later, somewhere else, as a
   stranger's K/V in somebody's context. The same shape of bug lives in the slot: a
   preempted slot can be handed to a new request inside a single `schedule` call, so
   the row reset cannot happen before or after that call, which is why the scheduler
   now calls back into the engine at the moment the slot moves.

## Diagram
[incremental-allocation.png](../diagrams/incremental-allocation.png). Left is one
request's blocks under the two policies: eight held at admission with six hatched
empty, against two held with the rest drawn as outlines it will buy at tokens 9, 13
and 17, and under them the one allocation rule and the state machine with the
preempt edge drawn in red instead of ghosted. Right is the cascade on a pool of
three: a grows into the block c is evicted from, b finds nobody younger and is its
own victim, with the two boxes for why the victim is the youngest and why recompute
rather than swap. The table underneath is the measured pair of pools, and the banner
is the trade: same 96 tokens, reservation waste zero, 25 tokens of K/V computed twice
and one request whose answer was late so an older one could be early.

## Tomorrow
The preemption path exists and nothing yet reports it where it hurts. `batchbench`
measures occupancy and latency but knows nothing about victims, so a run that
recomputes half its tokens looks like a run that scheduled well, and the recompute
surcharge lands invisibly inside `prefill_tokens`. Next is putting preemption into
the measurement: preemptions per request, the recompute bill as a share of total
forward work, and the latency of a preempted request against one that ran straight
through, which is the number a caller actually feels. That is also the first honest
comparison of recompute against swap, since swap trades the same latency for PCIe
bandwidth instead of FLOPs, and I want the recompute numbers before I argue about
which is better on a single GPU. After that, Week 9 owes a stress test: many
concurrent sequences through a pool small enough to preempt constantly, checking that
the text never changes and no block goes missing.

## Post angle
Day 33 of building an LLM inference engine from scratch. Week 8's scheduler reserved
prompt plus the whole token budget when it admitted a request, which made a running
row unkillable and booked the pool for text nobody ever wrote. Today it buys blocks a
step at a time. Four requests, 24 tokens of budget each, four slots, a 96-token pool:
the old rule fit two of them, ran a half empty batch for the whole run and took 48
iterations. The new rule starts all four, averages 3.4 rows per forward and finishes
in 28. Same 96 tokens of output. The thing I did not expect is that this is not a
feature you can ship on its own. Take away the worst-case reservation and a running
request can find the pool dry mid-generation, and no admission-side watermark saves
you: I worked out a one-step lookahead reserve and it holds for exactly one
iteration. So today also had to build the RUNNING to WAITING edge I deliberately left
out on Day 30: preemption by recompute. Blocks and slot back to the pool, tokens
kept, and the request goes to the head of the queue to be prefilled again later over
prompt plus what it already generated. Two things I would have got wrong. First, the
victim policy is a liveness proof, not a heuristic: evicting the youngest means the
oldest can never be evicted, so it always finishes, so the pool always drains.
Evicting the largest frees more blocks and lets two requests take turns evicting each
other forever. Second, recompute has to resume, not restart. My first version
re-prefilled the prompt only, and the engine kept running, kept draining, returned
the right number of tokens, and returned different text: a duplicated token in the
middle of the sequence. The test is a cramped engine against a roomy one on the same
prompts, and it is the only one here I would refuse to ship without. Reservation
waste is now 0.0. What is left is the unfilled tail of each last block, which is the
paging bound vLLM quotes a few percent for, and 25 tokens of K/V bought twice. 428
green.
