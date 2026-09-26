---
title: "Day 69: pass two got its own door"
parent: Daily log
nav_order: 69
---

# Day 69: pass two got its own door

Date: 2026-09-25 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was a GPU-gated test that holds the jitted reduce,
`_split_reduce_fwd`, equal to `reduce_partials` on rows that cross a chunk. Writing
it hit a problem first: there was nothing to call. The reduce was only reachable
from inside the read, so any test of it was a test of the read, and it only reached
the reduce with rows long enough to fill two chunks. So today pass two got its own
entry point on both backends, and the reads now go through it.

**`flash_decoding.py`.** Four functions:
- `check_reduce_workspace` is Day 64's `check_partials` without a launch to check
  against. The reduce's grid is the workspace's leading axes, so the accumulator's
  `[rows, n_q, splits, head_dim]` is taken as the claim and the other two buffers are
  held to it. Contiguity, fp32 and the device are all still enforced, because pass
  two addresses the partials with the same flat `(row * n_q + head) * splits + split`
  that pass one used to store them.
- `split_reduce_kernel` is the tlsim pass two, moved out of
  `paged_attention_split_kernel` without changes. The read now calls it on views of
  its flat buffers, with `out_dtype=q.dtype`.
- `split_reduce_triton` launches `_split_reduce_fwd` by itself. It refuses host
  tensors with a `ValueError` and a missing Triton with a `RuntimeError`, like every
  other jitted entry point. `paged_attention_split_triton` now calls it for its
  second pass.
- `split_reduce` dispatches on where the partials live, since they are the only
  tensors pass two touches.

**`tests/test_split_reduce.py`.** A `_workspace(live)` fixture builds partials by
hand. Each live chunk is the real partial softmax of its own random keys, and each
dead chunk is `-inf, 0, 0`, so the right answer is the softmax over the union. The
file also includes `_keep_the_winner`, Day 68's mutant written as a function, and
every fixture is checked against it before it's used: a workspace with one live
chunk is kept as a control that the mutant gets right, and the multi-live patterns
are asserted to be ones it gets wrong. One test swaps the mutant in through the
module and shows that the read breaks on a 40-key row split three ways but stays
exact on a 3-key row. That proves the read calls this function, rather than
assuming it.

16 new tests run here and 6 more are GPU-gated. Suite **2233 green** (27 GPU-gated
skips), ruff clean.

The fixtures, reduced two ways (seed 3, 2 rows x 3 heads, max abs error against
the whole softmax):

    live chunks        mutant error   reduce error
    [1, 0, 0, 0]            1.2e-07        1.2e-07
    [0, 0, 1]               2.4e-07        2.4e-07
    [1, 1]                    0.322        1.8e-07
    [1, 0, 1, 1, 0]           0.588        3.6e-07
    [1, 1, 1, 0]               1.38        2.4e-07

With the mutant put into the tlsim reduce, 7 of the 16 new host tests fail. So do 8
of Day 63's and Day 64's in `test_flash_decoding.py`, which was news to me: see What
I learned.

## Why it matters
**Yesterday's lesson says the reduce needs more than one live chunk, and a live
chunk is cheaper to build than a long row.** Through the read, getting two live
partials takes a row longer than one chunk, which at the serving partition means a
514-token request, a server at width 1024, and a tlsim decode loop over 520 keys.
Handed a workspace directly, it takes three tensors and a list of booleans. The
thing under test is `alpha = exp(m_s - M)`, and it's only exercised when two of the
`m_s` are finite. `_workspace([True, True])` is exactly that and nothing more.

**The test is only worth something if the read calls the function it tests.** Before
today the tlsim reduce was a closure inside the read and the Triton reduce was a
launch inside another function. A standalone copy written just for a test would be
one more thing that can drift. So the read's own second pass was moved out, both
reads call it, and one test checks the wiring by putting the mutant in the module
and watching the read go wrong.

**The fixtures are graded before they grade anything.** Day 68's controls passed
every earlier gate and failed only the new one. Here the fixture itself is the
control: `_discriminates` asks whether the mutant gets it wrong, and the one-chunk
workspace is kept around to show what a fixture that fails that check looks like. If
someone later "simplifies" the fixtures down to one live chunk, those assertions
fail before the reduce tests quietly stop meaning anything.

## What I learned
1. **The tlsim reduce was already covered, and Day 68 undersold that.** With the
   mutant in the model's pass two, 8 tests in `test_flash_decoding.py` fail. They
   run the read at toy sizes: `block=8, splits=3` over a 40-key width makes 16-key chunks, so a
   17-key row is already two live partials. The 512-key floor only exists at the
   default partition. What Day 68 found was true for the acceptance run, not for
   the kernel tests. The Triton reduce is still the gap, and today makes it
   callable, not verified.
2. **A reduce needs no launch description, because its grid is its input.** Pass one
   has to be told rows, heads, splits and head_dim, since it reads a pool. Pass two
   reads only the workspace, so all four numbers come off `part_acc.shape`.
   `check_reduce_workspace` is `check_partials` with the accumulator as the claim.
3. **The empty chunk and the padding are the same case in both reduces, on
   purpose.** The jitted ramp pads 5 splits to 8 and loads `-inf` and `0` into the
   surplus lanes, which is exactly what a dead chunk stores. `[True, False, True,
   True, False]` exercises both at once on a card: two dead chunks and three
   padding lanes, all folding to zero with no branch.
4. **Some mutants live in the fixture, not the kernel.** The first version of the
   "maxima far apart" test used `spread=30` and asserted a gap over 60. The widest
   gap it produced was 59.6. The assertion caught that before any reduce ran, which
   is the reason it's there: a test named "far apart" whose inputs weren't far apart
   would pass for the wrong reason.

## Diagram
[split-reduce-door.png](../diagrams/split-reduce-door.png). Top left: before and
after, the reduce as a closure inside the read, then as its own entry point both
reads call. Top right: the five fixtures, with the mutant's error next to the
reduce's. Bottom left: what reaching two live partials costs through the read and
through the door. Bottom right: which mutant is caught where, today and on a card.

## Tomorrow
The same move for pass one. `_paged_attention_split_fwd` can only be reached through
the read too, and its contract is simpler to state than the reduce's: it takes a
pool, a mapping and one chunk, and returns that chunk's `(m, denom, acc)`. That's
what the model's `split_kernel` closure already computes. Lifting it into
`split_partials_kernel` gives the jitted pass one a named target to be held to on a
card, with a test that feeds it one chunk of a row and compares against a direct
softmax over those keys.

The hardware caveat hasn't changed: `graphbench.py --weights ./weights --device cuda
--rates 1,2,4,8` on all three arms, with prompts past 514 tokens, is still the first
run to book. Today adds `pytest tests/test_split_reduce.py` to the list that has to
go green on the first card before any number from it counts.

## Post angle
Day 69 of building an LLM inference engine from scratch. Yesterday a broken merge in
my flash-decoding kernel passed every test, because the tests never produced two live
chunks for it to merge. Today I gave the merge its own entry point. It takes the
per-chunk (max, denominator, accumulator) directly, so a test can say "chunks 0, 2
and 3 saw keys" with a list of booleans instead of a 514-token prompt. Each fixture is
first checked against the broken merge: if the mutant gets it right, it isn't a
test. One-chunk workspaces: mutant error 1e-7. Two or more live chunks: 0.3 to 1.4.
vLLM and SGLang run this same two-pass split for long contexts. The part I'm learning
by building it is how to test the second pass on its own. 2233 green.
