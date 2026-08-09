---
title: "Day 35: the invariants, checked between every iteration"
parent: Daily log
nav_order: 35
---

# Day 35: the invariants, checked between every iteration

Date: 2026-08-08 · Week 9 · Phase 3 Batching and scheduler

## What I added today
The stress test Week 9 has owed since preemption landed, and the thing that makes
one worth writing: `nanoserve/audit.py`, the engine's invariants written down as
code. `audit_pool` says every block id is free or held by exactly one request and
never both and never neither, which is four separate ways to be wrong and four
different messages. `audit_slots` says the same about slots, which are cache rows.
`audit_requests` says a waiting request holds nothing, a running one holds a slot
and exactly the blocks its tokens need, and the waiting queue is in arrival order
with nothing waiting older than anything running. `audit_rows` is the tensor half:
a free slot's row is empty, a held row's block list is a prefix of its tenant's
reservation, and the K/V it holds is behind its tenant's text and never ahead.
`soak` is those checks in a loop around `Engine.step`: submit far more requests than
there are slots, drain through a pool far too small for them, audit between every
iteration, and hand back a `SoakReport` with what the pressure cost. Four public
accessors exist now because the auditor needs to see the ledgers rather than their
sizes: `BlockAllocator.free_blocks` and `allocated_blocks`, `Scheduler.free_slots`
and `known_ids`. Thirty-eight new tests in `tests/test_soak.py`, and every check is
proved to fire by corrupting exactly one thing, since an assertion nobody has ever
seen fail is a comment. Suite **495 green** (5 GPU-gated skips), ruff clean.

I also seeded the model helper in `tests/test_batch.py`, which was the one unseeded
one in the suite. Torch draws its default generator's seed from the OS at process
start, so that file was a different experiment every run, and about one draw in a
hundred and fifty produces weights where a position gap moves the logits by less
than the test's tolerance. It failed on me today for the first time in nine days and
had nothing to do with anything I wrote.

## Why it matters
Here is the experiment the whole day is for. Twelve requests, three slots, ten
blocks, and one block lost exactly once, on the third release, by a `free_all` that
drops its first id.

With the audit running between iterations:

    InvariantViolation at iteration 9:
    block 2 is allocated but no request holds it

Without it, the same run, same seed, same everything: all twelve requests finish,
in thirty-two iterations, with zero preemptions, and every answer is identical to
the run with no leak in it. The pool ends the run holding nine blocks of ten and
says nothing at all. No test of the output can see that, because the output is
correct. The damage is that this pool is now ten percent smaller forever, and the
run that eventually notices is some later workload that preempts more than it should
for a reason nobody can name.

That is the shape of every failure preemption adds. They are not wrong answers, they
are accumulative, so the test has to look at the state rather than the answer.

The soak's own numbers, six requests through three slots, nine tokens each, only the
pool moving:

| pool | evictions | worst request | recompute share | text |
|---|---|---|---|---|
| 64 blocks | 0 | never evicted | 0.0% | baseline |
| 16 blocks | 0 | never evicted | 0.0% | same |
| 8 blocks | 2 | evicted once | 23.5% | same |
| 5 blocks | 5 | evicted twice | 38.6% | same |

And the hard one, ten requests through four slots and six blocks: 52 iterations, 16
evictions, seven of the ten evicted twice, the oldest never once, and **51.3% of the
forward spent on K/V bought a second time**. Same text as a pool that never
preempts. Every block back, every slot back, every row empty.

The honest note: the audit did not find a bug in the engine. Days 30 to 34 tested
each of these edges individually and they held. What it found is the ones I injected
on purpose, which is the only evidence that the harness is a test and not a
decoration, and it is why four of the thirty-eight tests break the engine deliberately
and assert on the message.

## What I learned
1. **The observation point is part of the invariant.** These checks are true of a
   settled engine, between iterations, and deliberately false in the middle of a
   `schedule`, where a slot is taken from one request and given to another. Two of
   them carry slack for that: a running request's blocks must cover `num_tokens - 1`,
   because between a sample and the next schedule it holds one token more than its
   blocks do, and a cache row may be one block behind its request, because the
   scheduler grows the request at the top of an iteration and the engine syncs the
   row just before the write. Both bits of slack are real work in flight rather than
   sloppiness, and the price is that a check with an off-by-one bug of exactly that
   size would pass. The pool ledger has no slack at all, which is exactly why it is
   the check that finds things.
2. **A conservation bug is invisible to a test that asserts on output.** I have
   written ninety-odd tests this phase that generate tokens and compare them, and
   not one of them could have caught the leak above, because the leak does not change
   a token. This is a general shape and not a nanoserve one: any resource a system
   recycles needs a test that counts the resource, and the only reliable place to
   count is between operations, not at the end, because at the end everything has
   been released and a leak and a clean run look the same until the very last
   comparison.
3. **Progress is a weaker property than throughput, and the loop needs the weak
   one.** The soak's livelock detector accepts a preemption as progress, which feels
   wrong: an engine thrashing its pool is going nowhere. But an engine that preempts
   is still moving state, and a detector that called thrash a deadlock would fire on
   the healthy 5-block run, whose whole point is that it thrashes and still finishes.
   So the exception is for the shape that cannot recover (no token collected, nothing
   finished, nothing preempted, work still queued: a waiting head that cannot fit and
   nobody running to free anything) and the thrash is reported as a number,
   `max_preemptions`, for a human to judge.
4. **The invariant nobody had written down was the queue's order.** `_admit` only
   ever looks at `waiting[0]`, so a waiting queue out of arrival order does not raise
   anything, it just quietly starves whoever got pushed behind. That was untestable
   before Day 33, because nothing ever put a request back on the queue; now
   preemption does, at the *head*, and the head is precisely what admission trusts.
   Writing the checker is what made me notice that the ordering claim in the Day-33
   docstring ("anything still waiting arrived after it") was load-bearing and
   unchecked. It holds, and now something says so every iteration.

## Diagram
[soak-invariants.png](../diagrams/soak-invariants.png). Left is the ledger drawn:
ten blocks coloured by owner, the three set identities underneath, the four slots
with the rows that have to agree with them, and the two kinds of slack with the note
that the pool has none. Right is the leak experiment, the audited run stopping at
iteration 9 with the block id in the message against the unaudited run finishing all
twelve requests correctly and quietly ending nine blocks short, then the pool sweep
with the text column that never changes. The banner is the hard soak: 52 iterations,
16 evictions, half the forward spent on K/V bought twice, and everything handed back.

## Tomorrow
The scheduler edge cases that have been piling up all week, now that there is a
harness to run them under: aborting a request that is currently preempted (its
tokens are alive, its K/V is not, and it is sitting in the middle of the waiting
queue), a prompt whose length exactly fills a block so the first decode crosses a
boundary immediately, and the watermark, which has been a 1% default since Day 30
that nobody has measured. The watermark is the interesting one: it exists to stop a
newcomer taking the pool's last block and being preempted for it a step later, and
`soak` plus Day 34's sweep is finally enough machinery to say whether it does. After
that Week 9 closes and the engine gets an HTTP server in front of it.

## Post angle
Day 35 of building an LLM inference engine from scratch. Today's test: take the
engine, lose one KV block on purpose, once, and see who notices. Twelve requests,
three slots, ten blocks, and a `free_all` that drops one id on the third release.
Without a check, the run finishes all twelve requests in thirty-two iterations, zero
preemptions, and every single answer is identical to the run with no leak in it. It
just ends holding nine blocks of ten and never mentions it. No test I have written
this phase could catch that, because none of them look at anything but tokens, and
the tokens are right. The pool is simply ten percent smaller from now on, and the
run that suffers is some later one that preempts more than it should for no reason
anybody can name. So today is the invariants written down as code, four of them:
every block is free or held by exactly one request, never both and never neither;
same for slots, which are cache rows; a waiting request holds nothing and a running
one holds exactly what its tokens need; a free slot's cache row is empty and a held
one addresses its own tenant's blocks. Then a soak that runs those between every
iteration while far more requests than slots grind through a pool far too small.
With the check in, that leak is caught at iteration 9, by name: block 2 is allocated
and no request holds it. Two things I got wrong. The invariants are not true at every
instant, only between iterations, and two of them need one unit of slack (a request
holds one token more than its blocks until the next schedule, and a cache row can be
one block behind its request) because that slack is real work in flight. The pool
ledger needs none, which is why it is the check that finds things. And the livelock
detector has to count a preemption as progress, even though thrash goes nowhere,
because the run whose whole point is that it thrashes and still finishes is the run
I most want to pass. Hard soak: ten requests, four slots, six blocks, 52 iterations,
16 evictions, 51.3% of the forward spent on K/V bought a second time, same text as a
pool that never preempts, every block back, every slot back. vLLM and SGLang keep
this kind of checking behind a flag for the same reason. 495 green.
