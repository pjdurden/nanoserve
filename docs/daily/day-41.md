---
title: "Day 41: the acceptance test, and the caller nobody heard leave"
parent: Daily log
nav_order: 41
---

# Day 41: the acceptance test, and the caller nobody heard leave

Date: 2026-08-22 · Week 11 · Phase 4 Serving layer

## What I added today
`acceptance.py`, which is the Week 11 acceptance test as a harness: `live_server`
runs the real app under uvicorn on a real port, `ClientPlan` and `mixed_crowd`
describe a crowd that disagrees with itself about everything, `run_solo` and
`run_crowd` execute it, and `check_answers`, `check_survivors` and
`check_pool_returned` are the phase's four claims written as three functions.
`tests/test_acceptance.py` is 30 tests. Then the thing it found, which is a fix in
`server.py`: `ClientGone`, `await_client_disconnect`, `run_until_client_leaves`,
and 8 more tests in `test_server.py`. Suite **731 green** (5 GPU-gated skips), ruff
clean.

The acceptance run, tiny model, 12 clients on one socket, 3 slots, 48 tokens of KV:

    6 streaming, 6 unary · 8 sampled, 4 greedy · 3 hang up halfway
    peak_active 12, peak_running 3, 37 positions recomputed under preemption
    9 completed answers, 9 byte-identical to the same request run alone

And the real Llama-3.2-1B, CPU fp32, 4 clients over one socket:

    solo pass 19.0s   crowd pass 13.6s   peak_running 4

    greedy   " Paris. It is the most populous city in France and the"
    seed 0   " Paris. It is a very beautiful city. It is the"
    seed 1   " one of the most beautiful cities in the world. Paris is"
    top_k=40 " Paris and the official language of France is the French language."

all four byte-identical alone and in the crowd. That table is the good news and it
is not the interesting part of the day.

## Why it matters
**The transport every earlier test used cannot express a client that leaves.**
Days 37 through 40 drove the app through `httpx.ASGITransport`, which calls the
FastAPI app as a Python coroutine: no TCP, no HTTP parser, no chunked encoding,
and no socket. Day 37 wrote the disconnect path and tested it there, and the test
was real: cancel the caller's coroutine, and `AsyncEngine.generate` aborts the
request inside its own `except CancelledError`. What is not real is the premise.
**Nothing cancels an ASGI handler when a connection drops.** uvicorn turns a closed
socket into an `http.disconnect` *message*, delivered on `receive`, and a handler
parked on `await serving.generate(...)` is not reading messages. It never finds
out.

So the first thing the acceptance run measured, on the tiny model, was this:

| unary client asks for 800 tokens, hangs up at 0.2s | tokens the engine spent | engine idle at |
|---|---|---|
| before today | 800 of 800 | 1.36s |
| after today | 146 of 800 | 0.21s |

The pool came back either way, which is why every existing test was green: the
request finished normally, freed its blocks and freed its slot, and the ledger
balanced. What it did in between was hold a slot and a row of every forward for a
caller who was gone. On the real 1B that bill is legible: a client that asks for
128 tokens and stays pays 37.9 seconds of a 12-core CPU for them. Before today a
client that asked for 128 and left after one second paid the same 37.9 seconds,
and so did everybody queued behind it. After today the same request spends **one**
token and the loop is idle 0.6s later.

**Streaming was never affected, and that is exactly why this survived.** Starlette
watches for `http.disconnect` itself while it is producing a response body, so a
stream has always been cancelled promptly: the same measurement on the streaming
endpoint reads 3 tokens of a 64-token budget. Two endpoints, one of them correct,
and the correct one is the one with the visible feature and the loud tests. A
suite that proves "disconnects are handled" while meaning "disconnects are handled
on one of the two paths" is worse than one that proves nothing, because it is
believed.

The fix is a race, in `run_until_client_leaves`: the generation and a watcher on
`receive` are two tasks, and the first to finish decides. If it is the work, its
result comes back, and its exception comes back too, which is how a
`KVCacheExhausted` from `Scheduler.add_request` stays a 400 instead of becoming a
disconnect. If it is the watcher, the generation is cancelled, **awaited** (the
abort lives inside `generate`'s `except CancelledError`, and that only runs when
the cancellation is delivered), and the handler returns 499, nginx's "client
closed request", written to a socket nobody is reading. It exists for the access
log, which is the only place it can ever appear, and it is a real status rather
than a 200 because a log line claiming this request succeeded would be false.

**An acceptance test is a comparison, not a checklist.** The strong form of "no
request receives another request's tokens" is not a set of assertions about
crosstalk, mis-indexed rows, shared generators and cache rows addressing the wrong
block table. It is one comparison: run every plan alone on this server, run the
same plans together on the same server, and require the text to be identical. That
covers all four failure modes and every one nobody has thought of, and it is why
the baseline is deliberately the *same server one client at a time* rather than an
offline `generate`. The only variable allowed to differ between the two passes is
concurrency. A baseline computed offline would also be re-testing that the HTTP
layer agrees with the engine, which is Day 37's job.

**The pool coming back is two different questions.** `check_pool_returned` asks the
ledger first, through Day 35's `audit_engine`: is the pool *consistent*, no block
held twice, none allocated and owned by nobody. Then the counters: is the pool
*empty*. They are not the same check and neither implies the other. A request still
running after the crowd has left passes the audit, because holding blocks is what a
running request is entitled to do, and fails the count. That is precisely the shape
of the leak the disconnect bug would have had if the request's budget were
unbounded, and the reason the audit alone did not catch it.

## What I learned
1. **A test that cannot prove it was hard will get easier without telling you.**
   `CrowdReport.peak_running` is sampled from `/health` while the clients are in
   flight, and every check in the file passes trivially against a crowd of one. It
   is the same instrument as Day 35's `max_preemptions` and it earned itself
   immediately: the first crowd I ran, at 32 blocks, recomputed **zero** positions.
   Nothing had been preempted, so "your tokens do not depend on who else was in the
   batch" was being asserted over a batch nobody was ever evicted from. At 16
   blocks, still zero. At 12 it recomputes 37, and the answers still match. The
   pool size in the fixture is a measured number, not a guess.
2. **Ephemeral ports, bound by the harness and not by uvicorn.** `port=0` asks the
   kernel to choose and the caller needs the number before anything can connect, so
   the socket is created and bound here and uvicorn is handed it. The alternative,
   a fixed port, has a failure mode worth naming: one leaked server makes every
   *later* test in the run fail with `address already in use`, which is the hardest
   kind of failure to read because the test that broke is not the test that failed.
3. **Wait for `server.started`, not for the port to accept.** Uvicorn sets that
   flag after the lifespan has run, and the lifespan is where `AsyncEngine.start`
   happens. Polling the port instead would race the loop's startup and produce a
   first request that times out once in a hundred runs, which is the worst
   frequency: often enough to be real, rare enough to be blamed on the box.
4. **A disconnect is queued, so "is the pool free yet" has to be asked of the loop
   thread.** The abort marks a request and the loop applies it at the top of its
   next turn, so the moment `run_client` returns, the departed client may still be a
   running row holding blocks. Reading the allocator's counters at that instant is a
   race that fails one run in twenty and looks exactly like a leak. `wait_until_idle`
   asks `/health` instead, which is answered by the same event loop that owns the
   scheduler, so a reading of zero is a statement made by the thread that would
   know.
5. **The watcher has to ignore every message that is not a disconnect.** uvicorn's
   `receive` returns `http.request` for anything arriving on the socket, including
   a pipelined follow-up on a keep-alive connection, and it blocks on an event set
   by new data *or* by `connection_lost`. A watcher that resolved on the first
   message it saw would declare every ordinary client gone and abort their request
   mid-generation, which is a much more expensive bug than the one it fixes.
6. **In-thread, not in-process, and that is a choice about claim 3.** A subprocess
   would be one notch more faithful and would cost the pool check entirely: across a
   process boundary the only thing that can be asked about the allocator is whatever
   `/health` prints. In a thread the socket, the parser, the chunked encoding and
   the disconnects are all real, and the test still holds the `Engine` it audits.
   What is given up is process isolation, which none of the four claims are about.

## Diagram
[acceptance-crowd.png](../diagrams/acceptance-crowd.png). Left is the crowd: twelve
clients, the mix, the three slots they queue for, and the four claims each with the
number that proves it. Right top is the bug, as the two transports side by side:
what `ASGITransport` delivers when a client cancels, and what a socket delivers,
which is a message nobody was reading. Right middle is the fix as a race, with the
two exits. Right bottom is the bill, tiny model and real 1B.

## Tomorrow
Phase 4 is done. Day 42 opens Week 12 and Phase 5 with the benchmark harness: TTFT
and inter-token latency per request, throughput across the whole run, and a
baseline against HF `generate` on the same prompts and the same box. Today already
put two of those numbers on the table by accident (37.9 seconds for 128 tokens on
CPU fp32, and a crowd pass that finished faster than the solo pass) and neither is
measured in a way that survives being quoted. The first job is a measurement
harness that separates queueing from prefill from decode, because "it is slow" is
not a finding and "the second token arrives 290ms after the first" is.

## Post angle
Day 41 of building an LLM inference engine from scratch. The Week 11 acceptance
test: a real uvicorn on a real port, twelve concurrent clients mixing streaming and
unary, greedy and seeded, some of them hanging up halfway, against a pool small
enough to force preemption while they are all connected. The claim I care about is
one comparison, not a checklist: run every request alone, run them all together,
require byte-identical text. That catches crosstalk, mis-indexed rows, a shared
sampler and a cache row addressing the wrong blocks, all with one assertion. It
passed. Then the part I did not plan. Days 37 to 40 all ran through
`httpx.ASGITransport`, which calls the app as a coroutine, so "the client hung up"
there means "somebody cancelled your coroutine". Over a real socket nothing does
that. uvicorn delivers a disconnect as a *message* on `receive`, and a handler
parked on `await generate(...)` is not reading messages, so it never finds out. My
unary endpoint had a disconnect abort that was correct, tested, and had never once
been triggered on a socket. Measured: a client asks for 800 tokens, hangs up at
0.2s, the engine generates all 800. On the real Llama-3.2-1B that is 37.9 seconds
of CPU held for a caller who left after one second, plus everybody queued behind
them. Streaming was fine the whole time, because Starlette watches for the
disconnect itself while a response body is being produced, and that asymmetry is
exactly why this lived under a green suite: the loud path was correct and the quiet
one was not. Fix is a race between the generation and a watcher on `receive`, with
the cancelled task awaited so the abort actually lands, and a 499 for the access
log. Same client now spends 1 token instead of 128. 731 green.
