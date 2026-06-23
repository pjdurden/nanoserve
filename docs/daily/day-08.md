# Day 08: the full forward pass, matching HuggingFace to 2.4e-5

Date: 2026-06-23 · Week 2 · Phase 0 Foundations

## What I added today
`src/nanoserve/model.py`: `LlamaModel`, the full forward pass. Embedding lookup,
the loop that stacks the Day-7 decoder block sixteen times, the final RMSNorm, and
the tied LM head, producing logits over the whole vocabulary. Plus a `greedy_token`
helper (argmax of the last position). Five new tests in `tests/test_model.py`:
three pure (output shape, a blocks-zeroed collapse to `lm_head(norm(embed(ids)))`,
and the embed/lm_head tie), and two gated against the real Llama-3.2-1B (full logits
to 1e-4, greedy token exact). Suite is 47 pure-green; the two gated tests verified
on the model directly.

## Why it matters
This is the first time the whole network runs, not just a piece of it. Every formula
was already proven to 1e-5 on its own through Week 1, so today was the integration
moment: does the model, end to end, agree with the reference? On the prompt
"The test of a" the max absolute logit difference across all 16 layers and 128,256
vocabulary entries is **2.4e-5**, and the greedy next token is " good", which is
exactly what HuggingFace picks (token id 1695). That is the Week 2 north star, seeded:
a single decode step that matches the reference token for token.

## What I learned
A forward pass is mostly bookkeeping once the block is right, but it has three places
that run clean and are still wrong, and none of them are math:

1. The RoPE table is built once, not per layer. The cos/sin angles depend only on
   position, never on which layer you are in, so you build them from `position_ids`
   one time and hand the same tensors to all sixteen blocks. Rebuilding per layer is
   just wasted work, but using a different convention than the block test used would
   be a silent bug.
2. The final norm is real and easy to skip. There is an RMSNorm after the last block
   and before the head. Drop it and you are off by a whole normalization, which
   presents as "close to HF but never quite matching" rather than an obvious break.
3. The LM head is the embedding matrix, reused. `lm_head.weight` is literally the same
   tensor as `embed_tokens.weight` (the loader aliased them on Day 3), so the output
   projection is `norm(x) @ embed.T`. The only way to get this wrong is to accidentally
   make a second copy and let the two drift.

The trap test mirrors the Day-7 trick: zero every block's `o_proj` and `down_proj` so
both residual branches contribute exactly 0 and each block becomes the identity. The
whole stack then collapses to `lm_head(norm(embed(ids)))`, which I reproduce by hand.
That pins the three things above without touching the block math, which has its own
tests.

One environment note worth recording for the build-in-public honesty: the gated logit
comparison loads two fp32 copies of the model (mine and HF) and my local box has 14GB
of RAM, so running both at once gets OOM-killed. I verified the 2.4e-5 by running the
reference first, saving its logits, freeing it, then loading nanoserve and comparing,
which peaks at one model in memory. The real multi-token runs move to a rented GPU.

## Diagram
[full-forward.svg](../diagrams/full-forward.svg) — the whole pass top to bottom
(input_ids, embed, the 16-deep block stack, final norm, tied lm_head, logits), with
the RoPE table drawn as a single artifact feeding every block, the lm_head-to-embed
tie as a dashed back-edge, and the Day-8 verification numbers (2.4e-5, " good").

## Tomorrow
Week 2, Day 9: turn the single greedy step into a real greedy decode loop (append the
argmax, recompute, repeat) and check that nanoserve and HuggingFace produce the same
multi-token continuation, still on the slow no-cache path before Week 3 adds the
contiguous KV cache that makes it affordable.

---
Post angle: Day 8 of building an LLM inference engine from scratch, week two. Today I
stacked the transformer block I finished last week sixteen times into the full model:
embedding, the stack, a final norm, and the output head, all the way to logits. The
test is whether the whole thing agrees with Hugging Face, and on the prompt "The test
of a" my logits match theirs to 2.4e-5 across all sixteen layers and 128k vocab
entries, and the greedy next token is the same one they pick, " good". There is almost
no new math in a forward pass once the block is right. The bugs that survive are all
plumbing: the rotary position table has to be built once and shared across every layer
rather than rebuilt per layer, the final norm before the head is a real step that is
easy to forget and leaves you off by a whole normalization, and the output head is not
a new matrix at all, it is the input embedding matrix reused, so the only way to break
it is to accidentally copy it. I pinned all three with one trap test that zeroes each
block down to the identity so the model collapses to embed, norm, head, which I can
check by hand. A logit value can drift at the fifth decimal, but the token it chooses
must not, and that exact-token agreement is the thing week two is really chasing.
