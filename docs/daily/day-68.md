---
title: "Day 68: the third arm passed, on a server that never split"
parent: Daily log
nav_order: 68
---

# Day 68: the third arm passed, on a server that never split

Date: 2026-09-24 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was the acceptance run's third arm: a `--split-read` server
put on the socket next to Day 60's rectangle and streamed servers, with their bytes
compared. Adding it took five lines and every gate went green. Most of the day went
into finding out that the green didn't mean anything, and into a gate that says when
it does.

**`graphbench.py`.** Three small readers and one gate:
- `workspace_from_health` lifts the `split_workspace` section off `/health` (top
  level, where Day 66 put it), or `{}` for the other two reads.
- `longest_read` is the widest context any decode step read for a crowd, taken from
  the clients' own usage blocks: `prompt_tokens + completion_tokens - 1`, maxed over
  the clients that finished.
- `split_merge_floor` is the shortest request, prompt plus completion, whose last
  read reaches a second chunk: `keys_per_split + 2`, or `None` for a one-chunk arena.
- `check_arm_split` refuses a split arm whose rows the split never cut. A one-chunk
  arena and rows that all fit in the first chunk are both `MeasurementUnsound`, and
  the second message names the request length that would fix it. A rectangle or
  streamed arm is an `AcceptanceFailure`, and so is a payload whose boot arena and
  read counters disagree about the chunk count.

`ArmReport` gains `longest_row` and `workspace`, and `run_arm` fills both.

**`tests/test_reads.py`.** The three-arm run, at `max_model_len=1024` with 505- and
510-byte prompts, so every row is past the first 512-key chunk before its first
decode step. Both tiled arms are compared against the rectangle, not against each
other. It has two controls: the toy-width split server and the wide server on
Day 60's short `PLANS`. Each passes every Day 60 gate and fails `check_arm_split`,
one on "one chunk" and one on "first chunk".

`tests/test_split_arm.py` has 19 new tests and `tests/test_reads.py` has 3. Suite
**2217 green** (21 GPU-gated skips), ruff clean. The three live runs take about 35 s,
mostly two tlsim decode loops over 520-key rows.

`armbench.py`, host only, arithmetic only
([day-68-armbench.csv](data/day-68-armbench.csv)):

    config             slots  width splits  chunk merge floor   PLANS  SPLIT_PLANS
    toy, Day 60 width      4     32      1     32           -      no           no
    toy, Day 68 width      4   1024      2    512         514      no          yes
    serving              256   8192     16    512         514      no          yes

And one number that isn't in the repo, because it came from deliberately breaking
the kernel for ten minutes. I replaced the tlsim reduce's rescale,
`alpha = exp(m_s - joint)`, with `alpha = (m_s == joint)`, a combine that keeps the
winning chunk and drops the rest, and ran each configuration against the rectangle:

    run                          splits  widest read  clients that differ
    width 32,   PLANS                 1           20               0 of 6
    width 1024, PLANS                 2           20               0 of 6
    width 1024, SPLIT_PLANS           2          521               3 of 6

## Why it matters
**An acceptance run can only catch a bug on inputs that reach the code.** The
five-line version booted a split server at the file's toy width of 32. `choose_splits`
never cuts a width narrower than twice `DEFAULT_PARTITION`, so the planner picked one
chunk. A one-chunk split is the streamed loop followed by a reduce over one partial,
and a reduce over one partial is the identity. The server answered every byte
correctly, and `/health` said `split x 1 splits`. Nothing in Day 60's gates asks how
many chunks there were.

**Widening the server wasn't enough either.** At 1024 the plan is two 512-key chunks,
and Day 60's prompts are 3 and 5 bytes with 12 and 16 new tokens, so the widest read
is 20 keys. The second chunk walks nothing, stores `-inf, 0, 0`, and drops out of the
reduce with no branch. Day 63 designed that on purpose so a short row costs nothing.
The same design means a short row doesn't test anything. The mutant table shows it:
a combine that throws away every chunk but one is exact whenever only one chunk saw
keys. Two of the three configurations would have shipped it green.

**The gate needs a row length, and the server can't publish one cheaply.** The read's
counters are shapes only, and that was deliberate from Day 60. Counting rows longer
than a chunk would be `(context_lens > keys_per_split).sum().item()`: a device sync
per layer per step, and under a captured graph that Python never runs at all. The
client already sees every row's length, because every streamed answer ends with a
usage block. So `check_arm_split` combines two numbers from two places: the chunk
width, which the server fixed at boot, and the row length, which the client was
billed. Neither side can give the answer alone.

## What I learned
1. **The chunk is 512 keys at every width that splits.** The partition sets it,
   not the width, so the merge floor is 514 tokens for the toy and for the
   8192-token deployment alike. Any split test with prompts shorter than about 500
   tokens tests only the streamed path, whatever the server's width.
2. **The minus one matters.** The last sampled token goes back to the client and
   never into the cache, because the request finishes before a step could feed it in.
   A 510-token prompt with 12 tokens out reads at most 521 keys, not 522. Without the
   minus one, a request that exactly fills its first chunk would be counted as having
   crossed it.
3. **My first mutant didn't die, and it was the wrong mutant.** I broke
   `_split_reduce_fwd`, the jitted Triton reduce, and all four live tests passed. On
   this box the split runs the tlsim passes in `paged_attention_split_kernel`, which
   have their own reduce. The Triton reduce hasn't run once on this machine. That's
   the same lesson as the rest of the day, one level down: a test only covers the
   code it actually reaches.
4. **Only 3 of 6 clients differed under the mutant, and that's expected.** Greedy
   rows took the argmax over logits that the broken combine moved, but not always
   past a tie. The rows that rotated onto sampling drew from a shifted distribution.
   A bug in the attention output is always a numeric difference, but it only
   sometimes changes a token, which is why the comparison runs over a crowd and not
   over one client.
5. **The controls are the tests I'd have written without this day, kept with their
   verdicts inverted.** Each one passes `check_arm_was_crowded`, `check_arm_read` and
   the bytes, and fails only the new gate. If the gate ever gets loosened, they are
   the first tests to notice.

## Diagram
[split-arm-merged.png](../diagrams/split-arm-merged.png). Top left: where each
configuration's last read lands on its chunks. Top right: the "keep the winning
chunk" mutant run through all three. Bottom left: `armbench.py`'s merge floors.
Bottom right: `check_arm_split` and where each of its two numbers comes from.

## Tomorrow
The Triton reduce is the obvious gap: `_split_reduce_fwd` has never run on this box
and its mutant survived every test here. A GPU-gated test that holds the jitted
two-pass split equal to `reduce_partials` over rows that cross a chunk is small and
belongs next to Day 63's kernel tests. It will skip here like the other 21 until
there's a card.

The hardware caveat hasn't changed, and today makes it sharper:
`graphbench.py --weights ./weights --device cuda --rates 1,2,4,8` on all three arms
is still the first run to book, and the split arm needs prompts past 514 tokens or it
measures the streamed read under another name.

## Post angle
Day 68 of building an LLM inference engine from scratch. I added a third server to my
acceptance test: flash-decoding, which splits the KV read into chunks and merges the
partial softmaxes. Five lines, all green. Then I broke the merge on purpose, keeping
only the winning chunk, and it was still green. At the toy width the planner chose one
chunk. At a wider width, my 20-token rows never left the first 512-key chunk, so the
merge only ever saw one live partial, and that's exactly the case where the broken
merge is still correct. A split test only tests the split with rows longer than a
chunk. The new gate takes the chunk width from `/health` and the row length from each
client's usage block, because the server counting long rows would cost a sync per
layer. vLLM and SGLang ship this kernel next to the unsplit one for long contexts;
building it myself showed me how easily a test of it tests nothing. 2217 green.
