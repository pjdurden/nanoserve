---
title: "Day 37: an async server over a synchronous GPU loop"
parent: Daily log
nav_order: 37
---

# Day 37: an async server over a synchronous GPU loop

Date: 2026-08-17 · Week 10 · Phase 4 Serving layer

## What I added today
`server.py` has been a docstring and three TODOs since Week 1. Two of them are
closed. `src/nanoserve/serving.py` is the bridge: one background task that steps
the engine forever, an inbox that HTTP handlers push arrivals onto, and one future
per request resolved when *that* request finishes. `src/nanoserve/server.py` is the
FastAPI app over it: `POST /v1/completions` in the OpenAI shape, `GET /health`, and
a set of refusals. `tests/test_serving.py` (20 tests) and `tests/test_server.py`
(16) run the whole thing in-process over the tiny random model, so nothing here
needs `./weights` or a socket. Suite **565 green** (5 GPU-gated skips), ruff clean.

Every day until now had exactly one caller. `Engine.generate` takes a list of
prompts, drives `step()` in a `while` loop, and owns the process until it drains.
That is right for a benchmark and useless for a server, where requests arrive from
sockets that already exist, at moments nobody chose, and the entire point of Week 8
is that a request admitted now joins a batch already in flight.

## Why it matters
**`Engine.step()` blocks and owns the GPU, and an event loop is one thread.** Call
the step from a coroutine and the loop stops for its whole duration: no socket is
read, no arrival accepted, no finished answer written back. The server is still
correct, and every request in the batch still gets the right tokens, which is
exactly what makes this a bad bug to have. So the step runs in a worker thread and
the loop is free while it does. Measured, on 8 requests through 4 slots, with a
ticker coroutine standing in for handlers:

| where the step runs | turns the loop got | median stall | max stall |
|---|---|---|---|
| worker thread | 54,362 | 5 us | 0.61 ms |
| the event loop | 22 | 1.35 ms | 3.05 ms |

Twenty two chances to do anything else in the whole run, one per step, against
fifty four thousand. And that is a 1.5 ms step on a tiny CPU model; a real forward
is tens of milliseconds, and the stall column is what a `/health` probe, a new
connection, and every already-finished answer wait behind.

**A thread means two writers, so arrivals need a queue.** The moment the step
leaves the loop, the scheduler has two threads reaching for it: the worker inside
`schedule()`, and the loop inside a handler that wants to add a request. There is
no lock in `Scheduler`, on purpose, because everything it does is meant to happen
at one point in the iteration. So a handler never touches the engine. It appends to
an inbox, and the loop drains the inbox at the top of a turn, which is the one
instant when nothing is mid-forward. Aborts take the same road. The order inside
the loop is the whole invariant:

    drain the inbox        \
    apply queued aborts     >  no await between these: one writer, no lock
    settle finished futures /
    await to_thread(step)      the only line that leaves this thread

**Deliver on finish, not on the reap.** `SchedulerOutput.finished` reports a
request one iteration *after* its last token, because releasing its slot and blocks
is the next schedule's first job. It is a release event and I had been reading it as
a completion event. Waiting for it charges every caller an extra iteration for
something the engine already knew, and the bill is regressive:

| budget | last token emitted | reported finished | latency if I waited |
|---|---|---|---|
| 4 | step 4 | step 5 | +25% |
| 6 | step 6 | step 7 | +17% |
| 12 | step 12 | step 13 | +8% |
| 9 | step 18 | step 19 | +6% |

The shortest request pays the most, which is the same shape as the head-of-line
bill Week 8 removed, one iteration tall. So the loop settles on
`request.is_finished` and lets the reap happen behind the answer. The test that
pins it holds the reap step shut inside a worker thread: an answer that arrives
anyway is an answer that did not wait for it.

**Compatible means the fields that change the answer.** The engine samples with
`argmax`. Accepting `temperature=0.7` and returning greedy tokens with a 200 on
them is a wrong answer nothing downstream can detect, so `temperature`, `stream`
and `n` are declared in the schema in order to be refused by name, with a 400 that
says when they land. Status codes are claims about whose fault it is: a token id
outside the vocab is a 400 here and an `IndexError` in the embedding lookup (a 500)
without the check, which I verified rather than assumed. A budget too large for the
pool is a 400 whose refusal is made by `Scheduler.add_request` inside a loop shared
with every other request, comes back on one future, and becomes one caller's status
code while the others keep generating.

Eight concurrent requests through one loop: **18 iterations against 54 run one at a
time**, 3.0x, with each answer identical to its solo run.

## What I learned
1. **A shared loop turns every per-request error into a broadcast unless you route
   it.** Offline, `Engine.generate` has one caller and may simply raise at it. Here
   a rejected prompt is refused inside the drain, on behalf of somebody who is not
   on the stack, and letting that propagate kills the loop and hangs every other
   open connection. Three of the four error paths in the bridge exist only because
   the loop is shared: the rejected arrival goes to its own future, a step that
   raises goes to all of them, a schedule that stops progressing goes to all of them
   too. A loop that dies quietly is a server where nothing ever answers again, and
   that is strictly worse than a server that returns 500s.
2. **The offline hang guard and the serving one are not the same guard.**
   `run_to_completion` raises after N iterations, which is a fine way to learn about
   a deadlock from a terminal and an unacceptable way to learn about one from a
   request that never returns. Day 36's watermark deadlock was exactly this shape.
   Same detection, different obligation: a server has to fail sockets.
3. **The first version of the inline comparison deadlocked, which was the honest
   result.** My inline loop had no `await` in the busy path, so once it had work it
   spun forever and never yielded: the second request was never even submitted and
   the first request's waiter never ran. I added `await asyncio.sleep(0)` after the
   step to make the comparison fair. The unfair version is what an inline server
   actually looks like when somebody writes it in a hurry, and it does not degrade,
   it stops.
4. **A disconnect has to reach the scheduler.** A cancelled handler that leaves its
   request running is a slot and a block reservation held for an answer nobody will
   read, on a pool that Day 33 deliberately stopped over-reserving. So `generate`
   catches `CancelledError` on its way out and aborts, and the settle pass aborts on
   a cancelled future too, for callers who used `submit` directly. The abort still
   has to wait for an iteration boundary like everything else.
5. **Ids are the bridge's map key, so a duplicate is a correctness bug, not a
   nuisance.** Two in-flight requests with one id means one caller gets the other's
   tokens. It is refused synchronously in `submit`, along with an empty prompt and a
   budget below one, because a check that needs nothing from the engine belongs in
   the handler where it can be a 400 before the loop has spent an iteration on it.

## Diagram
[serving-bridge.png](../diagrams/serving-bridge.png). Left is the loop's turn drawn
as four bands with the thread boundary marked: drain, aborts, settle, then the one
`await` that leaves the event loop, with the note that nothing between the drain and
the step yields, which is why one writer is enough and no lock is needed. Right is
the two measurements: the stall table for step-on-loop against step-in-thread, and
the reap tax with the short request paying 25% and the long one 6%. The banner is
the rule the refusals follow: a parameter that would change the answer is honoured
or refused, never accepted and ignored.

## Tomorrow
Day 38 points this at the real model. `serving.py` and `server.py` both take their
engine and their tokenizer by injection and have never met Llama-3.2-1B, so what is
missing is the launcher: load the weights, size the block pool against actual VRAM
instead of a test constant, build the HF tokenizer, hand the three of them to
`create_app`, and run it under uvicorn so the thing can genuinely be curled. The
interesting part is the pool sizing, which is the first time this project has had to
turn a number of gigabytes into a number of blocks. Streaming is Week 11 and the
bridge is already the right shape for it: one future becomes one queue.

## Post angle
Day 37 of building an LLM inference engine from scratch. Today the engine got an
HTTP server, and the whole problem is one sentence: `Engine.step()` is a blocking
call that owns the GPU, and an asyncio event loop is one thread. Call the step from
a request handler and the loop stops for its entire duration. Nothing else runs: no
new connection is accepted, no arrival is queued, no already-finished answer is
written back to the socket it belongs to. The server is still *correct*, every
request still gets the right tokens, which is what makes it a nasty bug. I measured
it with a ticker coroutine standing in for handlers, 8 requests through 4 slots: with
the step on the event loop the loop got 22 turns in the whole run, one per step, and
stalled 1.35 ms at a time. With the step in a worker thread it got 54,362 turns and
stalled 5 us. That is a 1.5 ms step on a tiny CPU model. A real forward is tens of
milliseconds. But the moment the step goes to a thread, the scheduler has two
writers: the worker inside `schedule()` and the loop inside a handler calling
`add_request`. There is no lock in the scheduler and there should not be, because
everything it does is meant to happen at one point in the iteration. So handlers
never touch the engine. They append to an inbox, and the loop drains it at the top
of a turn, with no `await` between the drain and the step. One writer, no lock, one
answer to "when is it safe to add a request". The thing I got wrong first: I was
delivering answers off `SchedulerOutput.finished`, which reports a request one
iteration *after* its last token, because releasing its slot and blocks is the next
schedule's job. That is a release event, not a completion event, and waiting for it
taxes every caller an iteration for a fact the engine already had. It is regressive
too: +25% latency on a 4 token completion, +6% on an 18 step one. Delivering on
`request.is_finished` instead makes it free. Also, my first attempt at the inline
comparison deadlocked, because a loop with no await in its busy path never yields,
so the second request was never even submitted. That is not a straw man, that is
what an inline server does. vLLM's AsyncLLMEngine is this same shape. 565 green.
