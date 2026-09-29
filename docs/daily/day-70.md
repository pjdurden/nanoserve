---
title: "Day 70: pass one got its own door too"
parent: Daily log
nav_order: 70
---

# Day 70: pass one got its own door too

Date: 2026-09-28 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was the Day 69 move applied to pass one.
`_paged_attention_split_fwd` could only be reached through the read, and the read
only ever shows a test the *reduced* answer: whatever pass one stored in a chunk's
slot has been folded with every other chunk before anyone looks at it. So pass one
now has its own entry point on both backends, and both reads call it.

**`flash_decoding.py`.** Three functions:
- `split_partials_kernel` is the tlsim pass one, moved out of
  `paged_attention_split_kernel` without changes. It takes the read's arguments and
  returns the workspace, `(part_max, part_denom, part_acc)`. A handed-in workspace
  is written in place and comes back as itself, so the caller's buffers are the
  answer and there is no copy to disagree with them.
- `split_partials_triton` launches `_paged_attention_split_fwd` by itself, with the
  same refusals as every other jitted entry point (`ValueError` on host tensors,
  `RuntimeError` with no Triton). `paged_attention_split_triton` is now two lines:
  this, then `split_reduce_triton`.
- `split_partials` dispatches on `q`, like the read.

`paged_attention_split_kernel` is now the same two calls on the tlsim side. The
`split_reduce*` functions from Day 69 also went into `__all__`, which I forgot
yesterday.

**`tests/test_split_partials.py`.** `_chunk_partials` is the oracle: for every
`(row, head, split)` it gathers exactly the keys chunk `s` owns,
`[s * chunk, min(s * chunk + chunk, ctx))`, and computes their max, the
unnormalised denominator and the weighted-V sum in plain torch. With `overrun=True`
it becomes the mutant, a pass one whose chunk end is dropped (`hi = ctx`), so each
chunk walks from its own start to the row's end. The control and the mutant are
the same function with one flag, so they can't drift apart. As in Day 69, the
controls are checked first: a batch of rows that each fit in one chunk is asserted
to be one the overrun survives, and a row that crosses a chunk is asserted to be
one it doesn't. One test puts the overrun into the module and shows that the read
breaks on a 40-key row split three ways and stays exact on a 3-key row.

14 new tests run here and 4 more are GPU-gated. Suite **2247 green** (31 GPU-gated
skips), ruff clean.

The overrun, graded two ways (one row, width 24, chunk 8, seed 3, max abs error):

    keys   live chunks   slot error (acc)   read error (mutant)   read error (kernel)
       3             1                  0               6.0e-08               6.0e-08
       8             1            3.0e-08               1.2e-07               1.2e-07
       9             2              0.455                 0.112               1.2e-07
      16             2               2.30                 0.213               1.8e-07
      24             3               3.04                 0.182               1.2e-07

With the overrun put into the tlsim pass one, 7 of the 14 new host tests fail, and so
do 8 in `test_flash_decoding.py`.

## Why it matters
**The read hides pass one's mistakes behind pass two.** A chunk that walks keys it
doesn't own double-counts them, but by the time the read returns, those keys have
been rescaled by `exp(m_s - M)` and averaged with the honest chunks. The table
shows what that does: at 24 keys the mutant's accumulator in chunk 0 is off by 3.04,
and the attention the read returns is off by 0.18. The bug is still visible, just
4 to 17 times quieter. On a card with bf16 pools and a 1e-3 tolerance a smaller bug
of the same kind could fit inside the noise, and the per-chunk check won't let that
happen.

**The bug has the shape Day 68's did.** On a row that fits in one chunk, chunk 0
stops at the row's end whether or not it knows its own end, and every later chunk
owns nothing. The overrun is exact there (rows 3 and 8 in the table), which is
every request shorter than the 512-key partition. Grading a slot against "the keys
this chunk owns" is exact about which keys those are. A test of the reduced output
is only exact about the row.

**The jitted pass one now has a named target.** On this box it never launches, same
as the reduce did yesterday, so the four gated tests are the list that has to go
green on the first card: each one asserts first that the overrun would fail its case,
then compares `split_partials_triton` against the oracle chunk by chunk.

## What I learned
1. **The error in a slot is louder than the error in the output, and the reduce is
   why.** The overrun's extra keys land in the early chunks, and pass two doesn't add
   those chunks raw: it rescales each by `exp(m_s - M)` and divides by the joint
   denominator, so a corrupted chunk ends up as one weighted term among several.
   The same correction that makes the split exact also dilutes pass one's mistakes.
   Checking the slot sees the damage before that happens.
2. **The first key past a boundary is where it starts.** At 8 keys, the overrun is
   exact to 3e-8. At 9, chunk 0 folds in key 8, which chunk 1 also folds, and the slot
   error jumps to 0.455. That's the smallest case that tells the two apart, and it is
   one key long, not one chunk long.
3. **Returning the handed-in tensors themselves is part of the contract.** An
   owned workspace comes back as views of the flat buffers the programs wrote.
   A handed-in one comes back as the caller's own objects, not new views of the
   same memory, so `a is b` holds. A test asserts that, because a caller that keys
   a capture on buffer identity would otherwise see a new tensor every step.
4. **After today, a split read is just composition.** Each backend's read is the
   input gate, pass one, then pass two. Nothing lives only inside the read anymore,
   so no part of the split path can be reached only by the tests that happen to
   feed the read long enough rows.

## Diagram
[split-partials-door.png](../diagrams/split-partials-door.png). Top left: the split
read as two named calls, each with its own test file. Top right: the keys each chunk
owns on a 20-key row, honest and overrun. Bottom left: the overrun's error in the
slot and in the output as the row grows. Bottom right: which overrun is caught where,
today and on a card.

## Tomorrow
Both passes are now callable on their own, but the jitted ones still only get
checked if someone runs the gated tests on a card. A `--split-read` server picks its
read at construction and gets its arena before the first step (Days 65 and 66),
which leaves a natural place for a one-shot boot probe: run `split_partials` and
`split_reduce` on a tiny workspace that has a row crossing a chunk, compare against
`_chunk_partials`' plain-torch arithmetic moved into the module as a reference, and
refuse to serve if they disagree. Then the first boot on a card grades the jitted
passes by name, even if nobody remembers to run pytest there.

The hardware caveat hasn't changed: `graphbench.py --weights ./weights --device cuda
--rates 1,2,4,8` on all three arms, with prompts past 514 tokens, is still the first
run to book. Today adds `pytest tests/test_split_partials.py` to the list that has to
go green on the first card, next to `test_split_reduce.py`.

## Post angle
Day 70 of building an LLM inference engine from scratch. Flash-decoding cuts a row's
KV history into chunks, and each chunk stores a partial softmax (max, denominator,
weighted V) for a second pass to merge. I planted a bug in pass one: the chunk
forgets where it ends and walks to the row's end, so later keys get counted two or
three times. Through the full read on a 24-key row, the error is 0.18. In the
chunk's own slot, it's 3.04. On any row shorter than one chunk, it's zero, and that
covers every prompt under 512 tokens. So pass one now has its own entry point, and
each chunk is checked against a softmax over exactly the keys it owns. vLLM and
SGLang ship this two-pass split. What I'm learning by building it is how to test
each pass on its own. 2247 green.
