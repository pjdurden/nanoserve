# Day 09: greedy decode, the whole continuation matches HuggingFace

Date: 2026-06-24 · Week 2 · Phase 0 Foundations

## What I added today
`src/nanoserve/model.py`: `greedy_generate`, the decode loop. It is the Day-8
single step (`greedy_token`, an argmax over a full forward) wrapped in a loop:
take the argmax, append it to the sequence, recompute, repeat until
`max_new_tokens` or an optional `eos_id`. One sequence only (it rejects a batch
dim greater than 1; ragged multi-sequence batching is Phase 3). Five new tests in
`tests/test_model.py`: four pure (exact output length, determinism plus
equivalence to a hand-rolled step-by-step loop, eos stops-and-keeps-the-token,
and the batch guard) and one gated against the real Llama-3.2-1B (the full greedy
continuation matches HF `generate(do_sample=False)` token for token). Also
committed `chat.py`, the interactive REPL that drives this loop. Pure suite is 41
green; the gated multi-token match verified directly (see below).

## Why it matters
Day 8 proved one decode step agreed with the reference. This is the loop around
it, and the loop is the actual product of Week 2: not "the first token matches"
but "the whole continuation matches". On the prompt "The test of a", twenty greedy
tokens come out **bit-identical** to HuggingFace, ids and all:

    The test of a good leader is how well they can handle the pressure of a
    crisis. The test of a good leader

`torch.equal` on all twenty ids is `True`. That is the Week 2 north star reached
in miniature, before any cache exists to make it fast.

## What I learned
The quietly reassuring part is *how* the two agree. HF runs greedy decode through
its own KV cache; nanoserve has no cache yet and recomputes the entire prefix on
every step. Two completely different execution paths, and yet the argmax never
diverges across twenty steps. That is the whole reason the Day-8 logit tolerance
(~1e-4) is good enough: a greedy decode only cares about which logit is largest,
not its exact value, so as long as the gap to the runner-up never closes, small
float drift is invisible to the token stream. Greedy is the forgiving case;
sampling in Week 3 will be less so.

Three smaller things the loop made concrete:

1. The cost is in the shape of the loop, not the math. With no cache, step `k`
   re-runs a forward over a length-`k` prefix, so generating `n` tokens is
   O(n^2) work. You can feel it: each token is slower than the last. This is
   exactly the waste Week 3's contiguous cache removes by turning the loop body
   into a single new-token forward, and Week 5's paged cache makes affordable at
   scale. The slow version first, on purpose, so the speedup later is measurable.
2. `position_ids` needs no special handling here. `forward` defaults them to
   `0..len-1`, and because the whole grown prefix is passed every step, that stays
   correct as the sequence lengthens. The scattered-position case only matters
   once a cache means you feed just the new token, which is a Week 3 problem.
3. Stop-on-eos keeps the eos token. HF's `generate` returns the eos in the output
   rather than trimming it, so the test pins that: pass the first token the model
   would emit as `eos_id` and the sequence grows by exactly one, ending on that
   token. A decode loop that silently dropped the stop token would pass a length
   check and still disagree with the reference.

Honesty note, same constraint as Day 8: the gated test loads two fp32 copies of
the 1B model and my 14GB box OOM-kills that. So the pure suite (41 tests) is the
green run, and I verified the multi-token match with the save-free-reload dance:
generate the HF continuation, save its ids, free HF, then load nanoserve and
compare, which only ever holds one model in memory. The gated test
`test_greedy_generate_matches_hf_multi_token` is the spec; it passes on a box with
the RAM to hold both, and on a rented GPU.

## Diagram
[greedy-decode-loop.svg](../diagrams/greedy-decode-loop.svg) — the loop
(prompt, forward over the whole prefix, argmax of the last position, append, back
to forward), the two stop conditions, and the verified 20-token continuation with
`torch.equal == True`.

## Tomorrow
Week 2 is essentially done a few days early (full forward Day 8, greedy loop
Day 9). Buffer day: either start Week 3 early by writing the temperature/top-k
sampling functions so a bad day still has something to push, or add a short
generation-throughput note showing the O(n^2) slowdown per token as the
motivation graph for the KV cache. Lean toward starting sampling.

---
Post angle: Day 9 of building an LLM inference engine from scratch. Yesterday one
decode step matched HuggingFace; today the whole loop does. greedy_generate is the
simplest thing that could work: take the argmax, stick it on the end, run the
model again, repeat. On the prompt "The test of a" my engine generates twenty
tokens that are bit-for-bit identical to what HuggingFace generates, ids and all:
"good leader is how well they can handle the pressure of a crisis." The part I like
is that HuggingFace does this with a KV cache and mine recomputes the entire prefix
every single step, two totally different execution paths, and the chosen tokens
never diverge once across twenty steps. That is why a tiny logit difference at the
fifth decimal does not matter for greedy decode: you only need the largest logit to
stay the largest, not to be exactly equal. It also means my loop is O(n squared),
each token slower than the last, which is the whole motivation for the KV cache I
build next week. Slow and correct first, fast second.
