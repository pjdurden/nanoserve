---
title: "Day 12: sampling meets the cache, one decode loop for greedy and sampling"
parent: Daily log
nav_order: 12
---

# Day 12: sampling meets the cache, one decode loop for greedy and sampling

Date: 2026-06-27 · Week 3 · Phase 1 Correct generation

## What I added today
`LlamaModel.generate_stream` and `LlamaModel.generate` in `src/nanoserve/model.py`:
the Day-11 KV cache and Day-10's `sample` composed into one decode loop. Prefill
the prompt into the cache once, then each step forwards a single token, takes the
last position's logits, and *draws* the next token via `sample` instead of taking
the argmax. `temperature == 0` short-circuits to greedy, so the same loop is both
modes. `generate_stream` yields token ids as they land (for streaming); `generate`
drains it into a tensor. Rewired `chat.py` to stream through it and added
`generate.py`, a one-shot CLI (`generate.py "prompt" --temperature 0.8 --top-p
0.95 --seed 0`). Seven new tests in `tests/test_generate.py`; full suite 87 green.

## Why it matters
This is the last piece of Week 3 and the first time the engine looks like a real
serving primitive: a prompt in, a stream of tokens out, with the `temperature`,
`top_k`, and `top_p` knobs an OpenAI-style request actually sets, running over the
fast cached path rather than the O(n^2) recompute. Everything from here (paged
cache, batching, the HTTP server) is about doing *this* for many requests at once;
today is one request, done the way a request is really done.

## What I learned
The whole day was an exercise in *not* writing a second decode loop. The
temptation is a `greedy_decode` and a `sample_decode` side by side; the discipline
is that greedy is `sample` at `temperature == 0`, so there is exactly one loop and
greedy is a corner of it.

1. **The cache and the sampler are orthogonal, and that is the point.** The cache
   changes how the logits for the next position are *computed* (remember the past
   instead of recomputing it); the sampler changes what is *done* with those
   logits (draw instead of argmax). Neither touches the other, so wiring them
   together was a three-line change: same cached prefill, same one-token decode
   step, just `sample(logits[0, -1], ...)` where `argmax` used to be.
2. **Two independent routes to greedy, and both have to agree.** `temperature == 0`
   reaches greedy by short-circuiting `sample` to the argmax (no RNG touched).
   `top_k == 1` reaches it by leaving a one-hot distribution that `multinomial`
   can only resolve one way. The tests pin both against `greedy_generate_cached`
   with `torch.equal`, which is what keeps greedy from quietly drifting into a
   separate implementation.
3. **Determinism has to be explicit now, and testable.** A seed builds a
   `torch.Generator` threaded through every draw, so a run reproduces exactly.
   The subtle test is the opposite one: a determinism test alone would still pass
   if the loop ignored the RNG and always took the argmax, so there is a second
   test that flattens the distribution with `temperature = 2.0` and asserts that
   different seeds produce different continuations. Reproducible *and* actually
   random, pinned separately.

The satisfying check is `test_stream_matches_collected_under_same_seed`: the
tokens `generate_stream` yields, appended to the prompt, equal the tensor
`generate` returns, under the same seed. Streaming and batch generation are the
same draws in the same order, which is the property a server leans on when it
streams a response it could also have returned whole.

## Diagram
n/a. The Day-10 sampling funnel and the cache description already cover the two
halves; the interesting thing today is that they compose without a diagram's worth
of new machinery.

## Tomorrow
Week 3 is done (sampling, cache, and now both together). Start measuring properly:
tokens/sec versus prompt length and versus generated length, to turn the Day-11
"5.74x" point measurement into a curve, and use it to set up Week 4's motivation,
that this naive contiguous cache wastes most of its VRAM once you want many
sequences at once.

---
Post angle: Day 12 of building an LLM inference engine from scratch. Yesterday was
the KV cache (fast decode), the day before was sampling (temperature, top-k,
top-p). Today I wired them together, and the lesson was about not writing a second
decode loop. The cache and the sampler are orthogonal: the cache changes how you
compute the next token's logits (remember the past instead of recomputing it), the
sampler changes what you do with those logits (draw from them instead of taking
the max). So composing them was a three-line change, and greedy decode did not get
its own loop, it is just sampling at temperature 0, the corner where the draw
collapses to the argmax. I pinned that two different ways: temperature 0 and top-k
1 must both reproduce the greedy tokens exactly. The test I like most is the
inverse of determinism: a seed makes a run reproducible, but a determinism test
alone would still pass if the loop secretly ignored the random number generator
and always took the max, so there is a second test that flattens the distribution
and checks that different seeds actually explore different continuations.
Reproducible and actually random, proven separately. The engine now takes a prompt
and streams tokens with the same temperature, top_k, and top_p knobs a real
request sets. 87 tests green. #AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
