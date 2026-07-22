---
title: "Day 27: many ragged prompts, one padded forward"
parent: Daily log
nav_order: 27
---

# Day 27: many ragged prompts, one padded forward

Date: 2026-07-21 · Week 7 · Phase 3 Batching and scheduler

## What I added today
Phase 3 opens, and it opens on the plainest batching there is. `src/nanoserve/batch.py`
is new: `pad_prompts` takes N prompts of different lengths and returns a `PaddedBatch`,
the rectangle plus the three tensors that make the rectangle honest. `input_ids`
`[batch, max_len]` with a pad id in the empty slots, `attention_mask` `[batch, max_len]`
marking which columns are real tokens, and `position_ids` restarting at 0 at each row's
first real token. It carries `lengths`, a `last_index()` that knows where each row ends,
and `padding_waste`, the fraction of the forward spent on tokens that do not exist.
`last_token_logits` gathers each row's next-token distribution out of the logit grid.

The other half is the mask reaching attention. `gqa_attention` takes an optional
`attention_mask` and adds a per-key bias on top of the causal band; `transformer_block`
and `LlamaModel.forward` thread it through unchanged. Two new model entry points,
`forward_batch(batch)` and `greedy_token_batch(batch)`, run a whole padded batch and
take each row's argmax. Left padding is the default, because with it `logits[:, -1]` is
every row's last real token, so a batch stays one tensor with no per-row gather.

Nineteen new tests in `tests/test_batch.py`, held against the oracle that already
exists: row i of a batched forward must equal `forward(prompt_i)` run alone. That is
pinned on the tiny random model and on the real Llama-3.2-1B, along with the control
that proves the mask is doing the work, the NaN corner, and the boundary where a padded
batch is refused by the paged read. Suite **211 green** (5 GPU-gated skips), ruff clean.

## Why it matters
Everything through Day 26 ran exactly one sequence. That is the right shape for
learning the math and the wrong shape for a server, because a 1B model decoding a single
sequence leaves the card almost entirely idle. The matmuls are memory-bound: the weights
have to be dragged out of HBM every step no matter what, so pushing 8 sequences through
that same read costs barely more wall-clock than pushing 1. Batching is not a tuning
knob, it is the difference between using the GPU and paying for it.

The obstacle is that prompts are ragged and tensors are rectangles. Padding is how you
square that, and padding is a lie the model believes unless something stops it: a pad
slot still produces a real row of K and V, computed from a token id that means nothing,
and a query that scores against it spends softmax mass on a token that was never sent.
So the day is really about which parts of that lie are load-bearing, which is exactly
what the controls test rather than assert.

`padding_waste` is in the API for the same reason. Three prompts of 5, 2 and 3 tokens
cost a 3x5 grid, a third of which is padding, and one 500-token prompt batched with
seven 20-token prompts is 86% padding. Static batching gets the throughput win and hands
back a bill sized by the longest member of the batch. Printing that number now is what
makes the case for the ragged packing and the iteration-level scheduler of Week 8
honest, rather than something I assert because vLLM does it.

## What I learned
1. **Half of what "everyone knows" about left padding is not true for RoPE.** I wrote
   two control tests, one dropping the mask and one dropping the padding-aware positions,
   expecting both to break the match against the single-prompt run. The mask control
   passed: without it the logits move by 7.5, because left padding parks the pads at the
   low indices the causal mask already lets every query read. The position control
   failed, and it failed because the code was right and the test was wrong. A RoPE score
   depends only on the *difference* between the query and key positions, so adding a
   constant to every position in a row cancels inside every dot product. Measured: 0.0
   difference at a shift of 3, and 2e-6 at a shift of 5000, which is float error in the
   cos/sin table, not a change in meaning. I kept the padding-aware positions (a decode
   step addresses a block table by absolute position, and a long padded batch should not
   push short rows toward the context edge) but I rewrote the test to state the property
   it actually has, and added the case that shows the limit: a *gap* in the positions is
   not a uniform shift, and is not free.
2. **The padding bias has to be `finfo.min`, not `-inf`, and the reason is not style.**
   Under left padding a pad slot is also a *query* row, and the only key it is causally
   allowed to read is itself, which the mask then silences. With `-inf` that row is all
   `-inf`, softmax computes 0/0, and the NaN does not politely stay in the padding: the
   next block reads that same position as a *key* for the real tokens, so one NaN in one
   pad slot poisons the entire batch. A large finite bias makes the row a harmless
   one-hot instead. This is why HF uses `finfo.min` everywhere, and it is invisible until
   you batch, because a single sequence never has a fully-masked query row.
3. **Right padding makes the mask look unnecessary, which is a trap.** With pads after
   every real token, the causal mask already forbids looking at them, so masked and
   unmasked agree exactly on the real positions; there is a test that says so. It would
   be easy to conclude the mask is optional. It is not, it is what makes *left* padding
   legal, and left padding is what puts every row's next-token logits in the same column.
   The choice of padding side and the need for a mask are one decision, not two.

## Diagram
[padded-batch.png](../diagrams/padded-batch.png). Three ragged prompts on the left, the
`[3, 5]` left-padded rectangle in the middle with the pad slots dashed out, and the last
column boxed as the one every row reads its next token from. On the right the two
companion tensors, labelled with what today's measurements showed: the mask load-bearing
at 7.5 logits, the positions free at 0.0 under a uniform shift. The `finfo.min` corner
sits below them. The bottom bar is the bill: 5 pad slots out of 15, and the banner
carries the 86% case that motivates Week 8.

## Tomorrow
Batch the decode step, not just the prefill. Today's batch runs one forward and stops,
because the KV cache is still single-sequence: `PagedKVCache` owns one block table, and
`gqa_attention` refuses a padded batch on the paged path rather than attend over a
neighbour's blocks. The next step is per-sequence block tables in one batch, so each row
has its own logical-to-physical map into the shared pool and the fused read takes a slot
mapping per row. That is what turns static batching from a prefill trick into a decode
loop, and it is the last structural piece before the scheduler has something worth
scheduling.

## Post angle
Day 27 of building an LLM inference engine from scratch. Phase 3 starts: many prompts,
one forward. Prompts are ragged, tensors are rectangles, so short rows get padded, and
padding is a lie the model believes unless you stop it. Everyone knows the two fixes:
mask the pad keys, and shift the positions so each row starts at 0. I wrote a control
test for each, expecting both to break the match against running each prompt alone. Only
one did. Dropping the mask moved the logits by 7.5, because left padding parks the pads
at exactly the low indices the causal mask already lets every query read. Dropping the
position shift changed nothing: 0.0 difference, bit-identical. RoPE scores depend only
on the difference between two positions, so adding a constant to a whole row cancels
inside every dot product. Even a shift of 5000 only moved it 2e-6, which is float error
in the cos/sin table. My test was wrong, not my code, so I rewrote it as the property it
actually has, plus the case that shows the limit: a *gap* in the positions is not a
uniform shift, and that one is not free. The other thing batching teaches you fast: the
pad bias has to be a large finite number, not `-inf`. A left-padded pad slot is also a
query row whose only legal key is itself, which the mask silences, so `-inf` makes that
row softmax 0/0, and the NaN does not stay in the padding. The next block reads that
position as a key for the real tokens and one NaN eats the whole batch. That is why HF
uses `finfo.min`, and you never see it with a single sequence. The bill for all this is
in the API: `padding_waste` says a third of my toy batch is padding, and one 500-token
prompt batched with seven 20-token ones is 86%. A batch is sized by its longest member,
which is the honest argument for the continuous batching vLLM and SGLang ship, and what
Week 8 is for. 211 green.
