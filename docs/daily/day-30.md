---
title: "Day 30: a request is a state, not a row index"
parent: Daily log
nav_order: 30
---

# Day 30: a request is a state, not a row index

Date: 2026-07-27 · Week 8 · Phase 3 Batching and scheduler

## What I added today
`src/nanoserve/scheduler.py`, which had been a stub since Week 1 and is now Week 8's
first half: the request state machine and the two queues over it. `RequestState` is
three values (waiting, running, finished) and `_LEGAL_TRANSITIONS` is the machine
written as data, so the edges that exist are the edges the engine can take and
everything else raises `IllegalTransition` rather than corrupting something quietly.
`Request` is the object that replaces "row index into a fixed batch": it owns its
prompt, its generated tokens, the slot it occupies and the blocks it reserved, and
`append_token` applies the two stopping rules (EOS first, then the budget, so a
request that emits EOS as its last budgeted token is recorded as `stop` rather than
`length`).

`Scheduler` holds a `waiting` deque and a `running` list over the Day-14
`BlockAllocator`, and all of it happens in `schedule()`, in one order: release the
finished, then admit what fits, then hand back a `SchedulerOutput`. Reversed, a slot
freed this iteration could not be refilled until the next one, which is the exact
bug the week exists to remove. Admission needs two separate resources, a **slot**
(a row index in the batched cache, lowest free one reused) and **KV blocks**
(reserved atomically for `worst_case_tokens`, the prompt plus every token the
request may still emit). It is FIFO and does not skip a blocked head. The output
splits `admitted` (needs a prefill) from `decode` and carries `finished`, which is
how the engine learns an answer is ready.

Thirty-five new tests in `tests/test_scheduler.py`, pure: a real allocator, integer
token ids, no model, because every decision here is bookkeeping and bookkeeping is
what is subtly wrong when it is wrong. Suite **314 green** (5 GPU-gated skips), ruff
clean.

## Why it matters
Day 29 measured the two bills a static batch pays and this is the structural change
that stops paying them. Same ragged shape as that measurement, seven rows of 4
tokens and one of 32, counted through the new scheduler: the loop issues 60 decode
tokens and collects 60, against the static batch's 264 issued for the same 60
useful. That is **77.3% of the forward that no longer happens**, and it is 0.0%
waste rather than a smaller number, because a row is in the batch only while it
still wants a token. The latency half moves too: r0's answer is returned at
iteration 4 instead of iteration 32, its own last token instead of the straggler's.

The queue case is the one that shows what the slot recycling is for. Twenty-four
requests of that shape through 8 slots takes 45 iterations here against 96 for three
static waves of eight, a 2.1x cut in makespan and 4.3x less issued work, because a
slot is refilled the moment it comes back instead of at the end of a wave. Those are
counts of steps and rows rather than wall clock, which is the honest framing: this
module has no forward pass in it, and Day 31 is where the same shape gets timed
against the real model.

The cost is recorded rather than hidden. Admission reserves the worst case, so a
running request can never fail mid-flight, which is what makes a scheduler safe to
build before preemption exists. `reservation_waste` says what that buys and what it
costs: four requests with an 8-token prompt and a 200-token budget reserve 52 blocks
and occupy 4, a waste of **0.92**, and a request that stops at its first token held
12 of its 13 blocks for nothing. That is concurrency the pool could have sold, and
it is precisely the debt Week 9's incremental allocation and preemption pay off.

## What I learned
1. **The order inside one iteration is load-bearing, and only utilisation notices.**
   Admit-then-reap reads exactly as well as reap-then-admit and is wrong by exactly
   one step: the slot a request gives up at iteration N cannot be filled until N+1,
   so under a full queue every completion costs an idle row for one forward. No test
   about correctness catches it, because nothing is incorrect, and that is the shape
   most scheduler bugs have. It is the first thing the class docstring says now,
   since the whole week is the claim that a slot is refilled *immediately*.
2. **Two resources, not one, and they run out for different reasons.** Slots and
   blocks are both caps on admission and it is tempting to collapse them into one
   number, but a slot is a shape constraint fixed when the cache is built and a
   block is a memory constraint that moves every step. Checking them separately is
   what makes the two exhaustion cases distinguishable in a trace: no free slot
   means the batch is full and the pool is fine, no free blocks means the opposite,
   and only the second one is Week 9's problem.
3. **FIFO with no skip-ahead is a choice, and the free-looking alternative starves
   the big request.** When the head of the queue does not fit, looking past it for
   something smaller raises utilisation immediately and costs nothing visible. What
   it actually does is convert a visible queue into an invisible one: a large prompt
   behind a steady stream of small ones is never the request that fits, so it waits
   forever while the dashboard reports a busy, healthy server. Stopping at the
   blocked head makes a few small requests wait one iteration and gives the large
   one a guaranteed turn, and that is the trade production schedulers make too.
4. **Writing the state machine as data is what made the missing edge obvious.** Three
   states feels too small to need a transition table, and the table is what forced
   two decisions I would otherwise have left implicit: that a self-transition is a
   caller bug rather than a no-op, and that running to waiting must *not* exist yet.
   That edge is preemption, and without Week 9's recompute path a request that took
   it would keep its slot and strand its blocks silently. An edge you have not built
   the other half of is better absent than permissive.

## Diagram
[request-state-machine.png](../diagrams/request-state-machine.png). Left is the
machine itself: waiting to running on admit (with the two resources that gate it,
a slot and the block reservation), running to finished on stop or length, waiting to
finished on abort, and the greyed running-to-waiting edge marked Week 9. Right is
what it buys, eight rows on four slots as an iteration timeline: a static batch as
one rectangle with the finished rows hatched to the end, and the scheduled version
underneath where each row leaves at its own last token and a waiting request drops
into the slot on the very next iteration. The three boxes are the day's decisions
(an object can leave a batch and an index cannot, release before admit, FIFO with no
skip-ahead) and the banner is the arithmetic on the Day-29 shape: 264 issued for 60
useful against 60 for 60, and row 0 back at iteration 4 instead of 32.

## Tomorrow
Wire the scheduler to the model: `Engine.step()` takes a `SchedulerOutput`, runs the
newly admitted requests through the Day-27 padded prefill and the running ones
through the Day-28 batched decode step, and feeds the sampled tokens back with
`append_token`. The interesting part is that the batch changes shape every
iteration, so the `BatchedPagedKVCache` row a request owns has to be its scheduler
slot and nothing else, and a finished row's cache row has to be reset before the
next request lands on it. Then rerun Day 29's blocking measurement against the
scheduled loop on the real model, which is the acceptance test the week was given:
the waste fraction and the head-of-line inflation both have to fall on the same box
from the same harness. Two debts carried: the worst-case reservation above (Week 9),
and the Day-28 batched read still gathering a `[batch, max_ctx]` rectangle so a short
row pays for the longest row's history.

## Post angle
Day 30 of building an LLM inference engine from scratch. Yesterday I measured what
static batching costs: 79% of the forward computed for rows that had already
finished, and a short request handed back at 6.6x its own latency. Today is the
structural fix, and it is smaller than it sounds. A request stops being an index
into a fixed rectangle and becomes an object with a state: waiting, running,
finished. Three states, one dict of legal transitions, two queues. That is it, and
it is enough, because now a finished request can leave the batch mid-flight and a
waiting one can take the slot it gave up on the very next iteration. Counted on
yesterday's ragged shape, seven rows of 4 tokens behind one row of 32: the static
batch issues 264 tokens to collect 60, the scheduled loop issues 60 to collect 60,
and the short rows come back at iteration 4 instead of 32. Twenty-four requests
through eight slots is 45 iterations instead of 96. Two details that look like
nothing and are not. First, the order inside one iteration: admit-then-reap is
perfectly correct and idles the freed slot for exactly one forward, so release
comes first and admit second. Second, when the request at the head of the queue
does not fit, it is very tempting to look past it for a smaller one that does.
That is free utilisation and it starves large prompts forever while the server
looks busy and healthy, so the queue stops at the blocked head on purpose. The
honest part:
admission reserves the worst case, prompt plus the whole token budget, which is why
nothing can fail mid-flight and also why four requests with a 200-token budget hold
52 blocks to occupy 4. The module reports that as `reservation_waste` (0.92) rather
than hiding it, because it is exactly what vLLM's incremental allocation plus
preemption exists to fix, and that is Week 9. 314 green.
