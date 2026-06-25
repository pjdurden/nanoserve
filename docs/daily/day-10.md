---
title: "Day 10: sampling, pinned bit-for-bit to the HuggingFace warpers"
parent: Daily log
nav_order: 10
---

# Day 10: sampling, pinned bit-for-bit to the HuggingFace warpers

Date: 2026-06-25 · Week 3 · Phase 0 Foundations

## What I added today
`src/nanoserve/sampling.py`: the four pure transforms behind sampling.
`apply_temperature` (a plain logit divide), `top_k_filter` (keep the k largest,
mask the rest to `-inf`), `top_p_filter` (keep the smallest nucleus whose mass
reaches `p`), and `sample`, which composes them in HuggingFace's order
(temperature, top-k, top-p, softmax, one `multinomial` draw) and short-circuits
to the argmax at `temperature == 0`, so greedy is just the zero-temperature
corner of the same function. Eighteen new tests in `tests/test_sampling.py`:
fourteen pure (the math on hand-checkable logits, plus the composition edges) and
four that pin each mask and the full pipeline bit-for-bit to transformers' own
`TemperatureLogitsWarper` / `TopKLogitsWarper` / `TopPLogitsWarper`. Full suite
72 green.

## Why it matters
Greedy decode only ever needs the single largest logit, so Week 2 could be sloppy
about exact logit values and still match the reference. Sampling cannot: the next
token is *drawn* from the distribution, so the shape of that distribution has to
be right, not just its argmax. This is the first knob a real serving request sets
(`temperature`, `top_k`, `top_p` in an OpenAI-style body), so it has to match what
people already expect from those numbers.

## What I learned
The reason to pin the masks to transformers rather than "implement nucleus
sampling" is that every detail of these filters is a convention with an
off-by-one waiting in it, and the reference already made every choice:

1. **top-p sorts ascending, not descending.** HF's `TopPLogitsWarper` sorts the
   logits low-to-high, takes the cumulative softmax, and removes everything whose
   cumulative mass is `<= 1 - p`. So for probs `[0.5, 0.3, 0.15, 0.05]` and
   `p = 0.8`, the ascending cumsum is `[0.05, 0.20, 0.50, 1.00]` and the cut is at
   `1 - 0.8 = 0.2`. The `0.20 <= 0.20` boundary token is removed, leaving the
   `{0.5, 0.3}` nucleus. Reason about it from the wrong end and the nucleus is
   off by a token, which never throws, it just samples from a subtly wrong set.
2. **the nucleus is never empty.** `min_tokens_to_keep` forces the most likely
   token to survive even when `p` is tiny, so a confident step that would
   otherwise mask everything still has something to draw. Without it a peaked
   distribution plus a small `p` is an all-`-inf` softmax, which is `nan`.
3. **masking is just `-inf`.** There is no separate "candidate set" data
   structure: a filtered logit becomes `-inf`, its softmax probability is exactly
   `0`, and `multinomial` can never select it. The three filters compose by
   stacking `-inf`s, which is why their order only matters for top-p (it measures
   a nucleus over whatever top-k already kept).

The satisfying test is the last one: temperature then top-k then top-p, composed
by hand, equals the three warpers stacked in that order, `torch.equal`, no
tolerance. Same trick as Day-4 RoPE, where I pinned my rotation to HF's
`apply_rotary_pos_emb` instead of trusting my own sign convention. Match the
reference, do not reinvent it.

One thing greedy let me skip that sampling forces: determinism now has to be
*explicit*. `sample` threads a `torch.Generator` through `multinomial`, so a
seed reproduces a run exactly; the tests lean on that to assert a peaked
distribution lands on its mode the large majority of 300 draws without being
flaky.

## Diagram
[sampling-funnel.svg](../diagrams/sampling-funnel.svg) — logits in, the three
knobs in order (temperature reshapes, top-k and top-p mask the tail to `-inf`),
softmax then `multinomial` out, the dashed greedy bypass at `temperature == 0`,
and the note that the masks are pinned to the transformers warpers.

## Tomorrow
Wire `sample` into a `generate` path alongside `greedy_generate` (a `do_sample`
switch that calls `sample` per step instead of the argmax), or start Week 3's
real subject, the naive contiguous KV cache, now that the sampling math it will
feed is settled. Lean toward the cache: the sampler is done and tested, and the
O(n^2) decode is the thing actually worth fixing.

---
Post angle: Day 10 of building an LLM inference engine from scratch. Greedy decode
was the easy case: you only ever need the single largest logit, so a tiny float
error is invisible. Today is sampling, where the actual probabilities matter
because the token is drawn from them, and every detail is a convention with an
off-by-one waiting in it. So I did not "implement nucleus sampling", I pinned my
top-k and top-p masks bit-for-bit to the exact functions HuggingFace uses, the
same way I pinned my RoPE rotation to theirs on Day 4. The one that bites: top-p
sorts logits ascending and cuts at 1 minus p, so for probs 0.5, 0.3, 0.15, 0.05
and p = 0.8 the nucleus is exactly the first two tokens, and getting the direction
wrong never throws, it just quietly samples from the wrong set. Masking is just
setting a logit to negative infinity so its softmax probability is zero, and the
filters always keep at least one token so the distribution is never empty. Greedy
is now just the temperature equals zero corner of the same function. 72 tests
green, four of them checking my masks equal the reference with no tolerance at all.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
