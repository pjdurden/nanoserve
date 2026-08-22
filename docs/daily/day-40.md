---
title: "Day 40: one logits tensor, four callers who disagree"
parent: Daily log
nav_order: 40
---

# Day 40: one logits tensor, four callers who disagree

Date: 2026-08-21 · Week 11 · Phase 4 Serving layer

## What I added today
`SamplingParams` and `BatchedSampler` in `sampling.py`, a `sampling` field on
`Request`, `Engine.sampler` in place of the two `.argmax(dim=-1)` calls, and
`temperature` / `top_k` / `top_p` / `seed` on `POST /v1/completions`. Day 37 made
`temperature` a 400 that named itself and said it would stay one until the sampler
existed. The sampler exists. `tests/test_batched_sampling.py` is the new file (21
tests) and the rest went where the thing they test lives: 7 in `test_engine.py`, 3
in `test_serving.py`, 7 in `test_server.py`, and Day 39's
`test_temperature_is_still_refused_when_streaming` became
`test_a_bad_sampling_parameter_is_still_a_400_when_streaming`. Suite **693 green**
(5 GPU-gated skips), ruff clean.

The real 1B, on CPU, fp32, at `temperature=0.8, top_p=0.9`:

    prompt   "The capital of France is"
    greedy   " Paris. It is the most populous city in France and the"
    seed 0   " Paris. It is a very beautiful city. It is the"
    seed 1   " one of the most beautiful cities in the world. Paris is"
    seed 2   " Paris and the official language of France is French. French is"

Three seeds, three different continuations, all of them fluent. The interesting
line is the fourth one, which is not in that table.

## Why it matters
**The tensor is homogeneous and the parameters are not.** Since Day 31 the rows of
one forward have belonged to different callers, and until today that did not
matter to the sampler, because there was no sampler: `logits[:, -1].argmax(dim=-1)`
is one call and it means the same thing for every row. Now row 0 wants greedy, rows
1 and 2 want `top_p=0.9` at two different temperatures, row 3 wants `top_k=40`, and
they are slices of one `[rows, vocab]` tensor. The batch is heterogeneous in what
it asked for and cannot be split up, because splitting it up is undoing the whole
of Week 8.

So the sampler splits it three ways instead. **Greedy rows leave first**, through
one `argmax` over their sub-block. **Sampled rows are grouped by `filter_key`**,
which is `(top_k, top_p)` and deliberately not temperature: temperature is a divide
and vectorises against a `[m, 1]` column whatever the values are, while top-k and
top-p compute a threshold inside the filter from a scalar, so rows that disagree
about k or p have to be different calls. A real batch has a handful of distinct
filter settings, not one per request, so that is a few kernel launches per step
rather than one per row. vLLM vectorises the whole thing with per-row k and p
tensors; this is the same idea with less machinery and the same answers.

**The draw itself is per row, and that is the one place batching had to be given
up.** One `torch.multinomial` over `[m, vocab]` advances one generator once for the
whole block, which makes every row's token a function of who else was in the block.
That is exactly the coupling a seed exists to remove. I expected to pay for
un-batching it, and measured what it costs, 32 rows over Llama-3's 128256-entry
vocabulary, CPU, fp32:

| per step | ms |
|---|---|
| `argmax` over the block (all greedy) | 4.5 |
| `top_p=0.9` filter, which is a full sort | 91.4 |
| one batched multinomial (the version rejected) | 91.7 |
| 32 separate per-row draws (the version shipped) | 96.1 |

Un-batching the draw costs 4.5 ms out of 96, about 5%, because a multinomial over
128k probabilities is a cumulative sum over the vocabulary that you pay per row
either way; batching it saves the loop overhead and nothing else. The line above it
is the one worth staring at. **The expensive part of sampling is not the randomness,
it is the sort**: `top_p` is a full `torch.sort` of every row's whole vocabulary,
and it costs twenty times what a greedy step costs. That is the number that makes
vLLM's fused sampling kernels look less like premature optimisation than they did
yesterday.

**A request's tokens must not depend on who it shared a step with.** This is the
claim Day 31 made for greedy and it is much harder to keep once there is an RNG in
the loop. Measured: one request run 40 times, batched with zero to three other
sampled requests, everything else fixed.

    one shared generator     ->  4 distinct answers, one per batch composition
    a generator per request  ->  1 distinct answer

Four, not because the sampler is flaky, but because it is deterministic in the
wrong variable: with one generator the answer is a function of how many rows drew
before this one in the same step, which is a function of who else happened to post
at that moment. On the real 1B the same test reads:

    seed 0, alone               " Paris. It is a very beautiful city. It is the"
    seed 0, sharing with 3 more " Paris. It is a very beautiful city. It is the"

byte for byte, which is what a seed is supposed to mean and what a shared generator
cannot deliver at any batch size above one.

**Greedy is not `temperature=0`, in floating point.** It is tempting to make greedy
the limit of the sampled path and delete the branch, and `logits / 0` is `inf`,
whose softmax is `nan`, and a draw over `nan` either raises or lies. Even a small
positive temperature is not it: at 0.01 two logits a thousandth apart are still a
coin flip. So greedy is its own path, and the second property of that path matters
as much as the first: it consumes no randomness, so **adding a greedy row to a batch
cannot move anybody else's token**. If greedy went through the softmax it would eat
a draw and shift every seeded request that shared the step with it.

**The RNG state does not belong to the request.** `SamplingParams` is four numbers
and it lives on `Request`; the `torch.Generator` lives in `BatchedSampler`, keyed by
request id. Two reasons. `scheduler.py` is the one module here that owns no tensors
and can be tested over plain integers, and hanging a generator off `Request` would
end that. And a generator needs a lifetime: it must survive from one step to the
next, and it must be dropped when the request finishes, which is a thing the
sampler can do at the point it learns a request is gone (`out.finished`, which
covers aborts too) and a dataclass cannot do at all. Preempted requests are
deliberately not in that list: they come back, and their generator has to be where
they left it.

## What I learned
1. **Preemption already replays the RNG correctly, and it is not obvious why.** A
   preempted request loses its K/V and re-prefills over prompt-plus-generated,
   which emits exactly one token, the same as the decode step it replaced. Same
   number of draws, same order, same generator state, so the sampled text survives
   a memory decision the caller never sees. It works because Day 33 chose to keep
   the *tokens* and throw away the K/V. Had recompute been a re-run from the prompt,
   every preemption would have silently rewritten the answer, and the test that
   catches that (`test_preemption_does_not_change_a_sampled_requests_tokens`, a
   64-block pool against a 3-block pool on identical requests) would have been the
   only thing standing between me and a bug that appears only under memory pressure.
2. **Re-seeding per step is the version of this bug that looks correct.**
   `Generator().manual_seed(params.seed)` inside the sampling call is one line
   shorter, reproducible in every test that samples once, and emits the same token
   forever on the second token onward, because every draw comes from the same state.
   The test that sees it is not a correctness assertion, it is
   `len(set(draws)) > 1` over 20 steps of a flat distribution. Worth writing that
   kind of assertion any time state is supposed to advance.
3. **`SamplingParams(temperature=0, seed=8)` must not allocate a generator.** It is
   a perfectly ordinary request body (a client that always sends `seed` and
   sometimes sends `temperature=0`), and creating the generator at admission
   instead of at the first draw would have leaked one per greedy request. Creating
   it in `_generator_for`, at the point a draw is actually about to happen, makes
   that case free without a special case.
4. **Serving a parameter and policing its value are the same day's work.**
   `top_p=1.5` is not a nucleus and `temperature=-1` is a distribution turned inside
   out. Building `SamplingParams` in the handler is what turns those into a 400
   that names the field, before the engine has spent an iteration; constructed on
   the loop thread instead, the `ValueError` would surface as somebody's 500. It is
   the same argument Day 37 made about `KVCacheExhausted`, run the other way: check
   in the handler what the handler can check.
5. **`top_k=1` is a free end-to-end test of the whole path.** With one candidate
   unmasked the multinomial has exactly one outcome, whatever the temperature and
   whatever the seed, so `top_k=1` at `temperature=1.7` must return the greedy text
   over HTTP. If it does not, the filters are being applied to the wrong rows, and
   that is the failure mode the row-major/group-major indexing in `sample_batch`
   invites.

## Diagram
[heterogeneous-sampling.png](../diagrams/heterogeneous-sampling.png). Left top is
the split: four rows with four parameter sets, the greedy door, the two filter
groups, the per-row draw underneath. Left bottom is the cost table with the sort
called out. Right top is the coupling measurement, 4 distinct answers against 1.
Right bottom is the two versions that look right and are not, the temperature-zero
divide and the per-step re-seed. The banner is the real 1B, same seed, alone and in
a crowd.

## Tomorrow
Day 41 is the Week 11 acceptance test, which is the last thing Phase 4 owes: a real
uvicorn process on a socket, a crowd of concurrent clients mixing streaming and
unary, greedy and seeded sampling, short prompts and long ones, some of them hanging
up halfway. The claims to hold down are the ones every day of this phase asserted
separately and none of them asserted together: no request ever receives another
request's tokens, every seeded request matches its solo run, the pool comes back to
completely free when the crowd leaves, and the process survives clients that
disappear. It is also the first day the thing is exercised through curl rather than
through an ASGI transport, which has already been shown (Day 39, learning 2) to
hide an entire class of disconnect bug.

## Post angle
Day 40 of building an LLM inference engine from scratch. `temperature`, `top_k`,
`top_p` and `seed` are served now, which sounds like plumbing and is not, because
under continuous batching the rows of one forward belong to different callers and
they no longer agree on how to sample. One row wants greedy, two want `top_p=0.9` at
different temperatures, one wants `top_k=40`, and they are slices of the same
`[rows, vocab]` tensor. Greedy rows go out through one `argmax`. Rows that share a
`(top_k, top_p)` go through one filter call, with temperature as a per-row column
divide, because a batch has a handful of distinct settings and not one per request.
And then the part I thought would be the compromise: the draw itself has to be per
row. One batched multinomial advances one generator once for the whole block, so
every row's token becomes a function of who else was in the block, which is exactly
what a seed exists to prevent. I measured what un-batching it costs at 32 rows over
Llama-3's 128k vocab: 96.1 ms against 91.7 ms, about 5%, because a multinomial over
128k probabilities is a cumulative sum you pay per row either way. The line above it
in the same table is `top_p`, at 91.4 ms, which is a full sort of every row's whole
vocabulary and twenty times what a greedy step costs. The expensive part of sampling
is not the randomness, it is the sort, and that is why vLLM has fused kernels for
this. Two things that look right and are not. Greedy is not `temperature=0`: in
floating point that is a divide by zero, `inf`, then `nan`, and it also has to
consume no randomness, or adding one greedy row to a batch shifts every seeded
request sharing the step. And a generator re-seeded per step emits the same token
forever, so the seed goes in once at the request and the state lives in the sampler,
keyed by request id, dropped when the request finishes. Measured on the real
Llama-3.2-1B: same prompt, same seed, alone and sharing a batch with three other
sampled requests, byte-identical output. 693 green.
