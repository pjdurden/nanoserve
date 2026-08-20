---
title: "Day 39: the stream, and the token that is half a character"
parent: Daily log
nav_order: 39
---

# Day 39: the stream, and the token that is half a character

Date: 2026-08-19 · Week 11 · Phase 4 Serving layer

## What I added today
`src/nanoserve/detokenizer.py`, `AsyncEngine.stream` in `serving.py`, and the SSE
half of `POST /v1/completions` in `server.py`. `stream=true` was a 400 that said
"Week 11"; this is Week 11. `tests/test_detokenizer.py` (25 tests) and
`tests/test_stream.py` (31) are the new files, and Day 37's
`test_streaming_is_refused_rather_than_silently_batched` became
`test_streaming_is_no_longer_refused`. Suite **654 green** (5 GPU-gated skips),
ruff clean.

It streams. Against the real Llama-3.2-1B on CPU, fp32, a 128-block pool:

    [ 1.568s] ' Paris'      [ 2.132s] ' It'      [ 3.259s] ' populous'
    [ 1.848s] '.'           [ 2.413s] ' is'      [ 5.792s] '' finish=length
    ...                                          [ 5.793s] data: [DONE]

Same sixteen tokens the unary endpoint returns for the same prompt, and the first
one is readable at 1.568s instead of at 5.793s. Nothing about the engine changed
to get that: **streaming is a delivery schedule, not a different model.** Which is
the property the whole test file exists to hold down, because the moment streaming
changes a token it has stopped being a delivery decision.

The day has two halves and only one of them is the queue.

## Why it matters
**One future becomes one queue, and it must not have a bound.** The loop already
had the right shape: `_settle` walks the live requests at every iteration
boundary and hands each finished one to its own caller. Streaming adds one line
to it, `waiter.deliver()`, which pushes whatever output tokens this request has
emitted since the last look. Everything hard about the bridge (the inbox, the
single-writer rule, the per-request error routing) is unchanged and paid for.

The one new design call is the queue's depth, and it is unbounded on purpose. A
bounded queue is backpressure, and backpressure on a stream is backpressure on
*everybody*: there is no way to pause one row of a batch, so a slow reader would
become the pace of a forward pass that a dozen other requests are rows of. The
tokens wait in memory instead, which is a few bytes per token for a request that
is already holding blocks. The test for this reads nothing until the engine has
parked, then drains, and asserts nothing was lost.

**A token is a run of bytes, not a run of characters, and this is where a naive
stream lies.** Llama-3's tokenizer is byte-level BPE. The vocabulary is built over
the 256 byte values and a merge never checks whether the run it just made ends on
a UTF-8 boundary, so a 4-byte emoji is routinely two tokens:

    "\U0001f600"  ->  [76460, 222]
    decode([76460]) = "�"      decode([222]) = "�"
    decode([76460, 222]) = "\U0001f600"

Decode per token and the stream prints two black diamonds where the model wrote
one emoji, while `/v1/completions` prints the emoji, from the same ids. The bug
lives entirely in the incremental path, so no amount of unary testing finds it,
and it is invisible until it isn't:

| 600 random strings over | naive per-token decode wrong | incremental wrong |
|---|---|---|
| plain English | 0 | 0 |
| the same plus emoji, CJK, accents | **517** | 0 |

Zero and 517 out of the same generator with a different alphabet. That is the
shape of a bug that ships.

**The fix is to never decode a token alone.** Keep the ids, decode a small window
of them twice, and emit the difference:

    pre   = decode(ids[prefix_offset:read_offset])
    whole = decode(ids[prefix_offset:])
    delta = whole[len(pre):]

Two decodes of a handful of ids per token, against a forward pass measured in tens
of milliseconds. It is free. What it buys is that the decoder always sees the
bytes on both sides of every boundary it is asked about, which is the only way it
can tell a character from the front of one. And then the hold-back: a window that
ends mid-character still decodes to a trailing U+FFFD, so a delta ending in one is
not emitted at all and the offsets do not move. **The cost of that is bounded by
3 tokens**, because UTF-8 is at most 4 bytes and a token carries at least one.
Late, never wrong, and never more than three tokens late. vLLM's
`detokenize_incrementally` is this, with the same two offsets and the same U+FFFD
test.

**A stream chooses its status code before it has an answer.** This is the one
thing about SSE that changes the shape of a handler. A response's status line goes
out with its first byte, so the instant the first frame is written the server has
committed to a 200 and every later failure has to be reported inside a body that
already claims success. So the handler pulls the *first* update from the bridge
before it constructs the `StreamingResponse`. That costs nothing (the first update
is the first token, so the headers leave when the caller had something to read
anyway) and it keeps "this prompt can never fit the pool" a 400 instead of an
error frame buried in a 200. What is left after that becomes
`data: {"error": ...}` and then `[DONE]`, because the alternative is a body that
simply stops, which is byte-for-byte indistinguishable from a complete one.

**A stopped reader is not a stopped request.** Day 37 gets the disconnect from
`CancelledError` at the await inside `generate`. A stream never awaits there. It
gets it from the generator being closed, by `break`, by `aclose`, or by Starlette
cancelling the response body when `http.disconnect` arrives, and the `finally` in
`AsyncEngine.stream` is what turns any of those into an abort. Measured against
the live server with a raw socket, asking for 200 tokens and hanging up after two
frames:

    iterations before the request:   16
    iterations 3s after hanging up:  19
    iterations 7s after hanging up:  19   (a 200-token budget would have been ~216)

Without that `finally` the row keeps generating into a queue nobody will ever
read, holding a slot and blocks that nothing collects, and a server under flaky
clients leaks its whole pool one hang-up at a time.

## What I learned
1. **Three of my disconnect tests passed before the abort existed.** They asserted
   "no active requests, every block back" after hanging up, and a request that
   simply ran out of budget also ends with no active requests and every block
   back, forty forwards later. The assertion with teeth is the step count:
   `stats["steps"] < 10` fails at 41 when the abort is deleted and passes at 3
   when it is not. I only found this because I deleted the `finally` to check the
   tests could see it, which is now a habit worth keeping for anything whose
   failure mode is "the right end state, expensively".
2. **`httpx`'s ASGI transport cannot express a client that stops reading.** It
   runs the whole app to completion, collects the body parts into a list, and
   *then* returns a response, so `break`ing out of `aiter_lines` breaks out of a
   response that already finished. Every HTTP-level disconnect test written
   through it passes whether or not the disconnect path exists. Driving the ASGI
   callable by hand fixes it, and a hang-up turns out not to be a socket mystery:
   it is one message, `{"type": "http.disconnect"}` on `receive`, and Starlette
   cancels the body generator when it arrives.
3. **`StreamUpdate.request` is a live object and only its final update is safe to
   read.** I wrote `assert all(not u.request.is_finished for u in updates[:2])`
   and it failed, correctly: by the time the consumer looks, the loop has moved
   on. So the token id is copied into the update and everything terminal
   (`finish_reason`, the counts) is read off the final one. A consumer on a slow
   socket must see the token that was current when the update was made, not the
   one that is current when it gets around to looking.
4. **Proving "the first token arrives early" with a stopwatch is how you get a
   flaky suite.** Counting `serving.steps` from the consumer races the loop, which
   keeps stepping inside `to_thread` while the consumer is being scheduled. A gate
   settles it: an engine that runs one step per ticket, held after the prefill, so
   `(token is not None, is_finished, num_output_tokens, active, steps) ==
   (True, False, 1, 1, 1)` is a statement about the bridge's ordering rather than
   about how fast this box is.
5. **A frame with no text is not a frame.** The hold-back means `append` returns
   `""` for the first three bytes of an emoji, and emitting a chunk for each of
   those would send three `data:` frames whose `text` is empty, which a client is
   entitled to render as nothing and count as three tokens. Skipping them is not
   an optimization, it is the difference between a delta protocol and a protocol
   that reports the tokenizer's internals.

## Diagram
[sse-streaming.png](../diagrams/sse-streaming.png). Left is the detokenizer: the
emoji split across two tokens with each half decoding to a diamond and the pair
decoding correctly, then the two-decode window and the U+FFFD hold-back, then the
0 against 517 table. Right top is the same request on two delivery schedules over
one unchanged row of `Engine.step()`, with the note on why the queue has no bound.
Right bottom is the SSE frame sequence with the red line across it: everything
above it can still be a status code, everything below it is inside a 200 that has
already gone out. The banner is the measurement from the live server.

## Tomorrow
Day 40 stays in Week 11 and takes the other half of `_check`: sampling.
`temperature` and `n` are still 400s that name themselves, and `sampling.py` has
had top-k and top-p since Week 6 without ever being wired to a request. The work
is per-request sampling parameters through `Request` and into the batched sampler,
which is more interesting than it sounds because the batch is now heterogeneous:
one row wants greedy, another wants `top_p=0.9`, and they are columns of the same
logits tensor. The gotchas are seeding (a server cannot use a global generator
without one request's draw depending on who it shared a step with) and the fact
that greedy is not "temperature 0" in floating point.

## Post angle
Day 39 of building an LLM inference engine from scratch. `stream=true` works: the
first token of a completion now reaches the client at 1.568s instead of the 5.793s
the whole answer takes, same sixteen tokens, nothing about the engine changed.
Half the day was the queue, and the queue was easy, because Day 37's loop already
settles each request at its own last token; streaming just pushes at every token
instead of resolving at the last. The one real decision there is that the queue is
unbounded, and it has to be: bounding it is backpressure, and there is no way to
pause one row of a GPU batch, so a slow reader would become the pace of a forward
pass a dozen other requests are sharing. The other half is the part I did not see
coming. Llama-3's tokenizer is byte-level BPE, so a token is a run of bytes and
nothing makes those runs land on UTF-8 boundaries. The emoji is literally two
tokens, and decoding either alone gives you the replacement character. Decode
per token, as every naive streaming implementation does, and your stream prints
black diamonds where your non-streaming endpoint prints an emoji, from identical
ids. I measured it: over 600 random strings of plain English, per-token decoding
is wrong 0 times. Over 600 of the same with emoji and CJK mixed in, it is wrong
517 times. The fix is to never decode a token alone: keep the ids, decode a small
window twice, once without the new token and once with, and emit the difference,
and hold back anything that ends in U+FFFD until the next token completes it. That
hold-back is bounded at 3 tokens because UTF-8 is at most 4 bytes. Late, never
wrong. And one thing about SSE that changes the shape of the handler: the status
code goes out with the first byte, so once a frame is written you cannot return a
400 any more. The handler pulls the first update from the engine *before* it
builds the response, which costs nothing and keeps "this prompt can never fit the
KV pool" an honest 400 instead of an error buried inside a 200. vLLM's
`detokenize_incrementally` is where the window trick comes from. 654 green.
