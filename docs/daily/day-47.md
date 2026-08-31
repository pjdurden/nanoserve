---
title: "Day 47: the token that stopped coming home one row at a time"
parent: Daily log
nav_order: 47
---

# Day 47: the token that stopped coming home one row at a time

Date: 2026-08-30 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`BatchedSampler.sample_batch_device`, which draws exactly what `sample_batch` drew
and returns it as a `[rows]` int64 tensor on the device the forward ran on instead
of as `list[int]`, with an all-greedy fast path that is one `argmax` over the whole
tensor and no gather, no dict and no per-row loop. `sample_batch` is now that plus
one `.tolist()`, so the two paths cannot drift. Then `nanoserve.output`, the collect
half: `Readback` counting journeys home and the elements they carried, `TokenBatch`
holding one step's tokens plus the request id that owns each row and resolving them
at most once however many callers look, `OutputProcessor` applying them and counting
`transfers_per_step`, the arithmetic (`STRATEGIES`, `syncs_per_step`, `readback_s`,
`saving_per_step`, `strategy_speedup`, `sample_ceiling` pointing Day 46's
`speedup_if` at one phase, `measure_transfer_s`, `render`), and four gates
(`check_single_transfer`, `check_device_resident`, `check_in_row_order`,
`check_measurable`). `Engine._sample` returns a `TokenBatch` and `Engine._collect`
goes through the processor, which moves `syncs=True` off the `sample` phase and onto
`collect`. `syncbench.py` at the root measures both paths over one real
`[slots, 128256]` logits tensor and writes
[day-47-syncbench.csv](data/day-47-syncbench.csv). `tests/test_output.py` is 72
tests and `tests/test_batched_sampling.py` gained 14, including two that monkeypatch
`tolist`, `item` and `__int__` into raising so that "did not read back" is something
a test can fail on rather than a claim in a docstring. Suite **1116 green** (5
GPU-gated skips), ruff clean.

Llama-3.2-1B, cpu fp32, the sample phase over one real logits tensor, median of 25:

    slots   params      per row    one tensor    faster    syncs before/after
      1     greedy     0.193ms       0.145ms      1.33x           1 / 1
      2     greedy     0.231ms       0.151ms      1.54x           1 / 1
      4     greedy     0.299ms       0.151ms      1.99x           1 / 1
      8     greedy     0.680ms       0.358ms      1.90x           1 / 1
      1     top-p     14.525ms      14.504ms      1.00x           1 / 1
      2     top-p     19.481ms      19.208ms      1.01x           2 / 1
      4     top-p     28.993ms      28.708ms      1.01x           4 / 1
      8     top-p     60.900ms      59.592ms      1.02x           8 / 1

And the loop as the engine runs it, greedy, with the transfer count taken live from
`Engine.output` rather than derived: 1.00 transfers per step at every batch size,
`collect` 0.028ms to 0.070ms, `sample` 0.230ms to 0.841ms.

## Why it matters
**A readback is not a copy, it is a stop.** CUDA launches are asynchronous: the host
queues kernels and runs on, which is what makes per-step Python free when it fits
under the arithmetic. Reading a value back ends that, because the number does not
exist until the kernels producing it have finished. `sample_batch` returned
`list[int]`, and every one of those ints was its own `int(tensor)`, so a batch of
eight sampled rows stopped the host nine times a step. That is the mechanism behind
Day 46's finding that `sample` held 86% to 89% of the whole loop, and it is why
`recommended_model` returned `serial` for this engine.

**The bill is per journey, not per byte.** One step's tokens are `[rows]` int64: 64
bytes at eight rows, on a bus that moves gigabytes a second. Essentially all of the
cost is the fixed part, so `readback_s` prices the count and ignores the payload,
and the optimisation it points at is going fewer times rather than carrying less.

**Two savings landed and they are not the same saving.** The Python one is real on
this box and measurable on it: 1.33x to 1.99x on the sample phase, growing with the
batch, because what went away includes a gather of `[rows, 128256]` floats. The
synchronisation one is 8 down to 1 at eight sampled rows, is worth nothing here and
is worth the week on a card. They are reported in separate columns and
`saving_per_step` returns 0.0 for the greedy path on purpose, because adding them
together is how a modest day becomes an overclaim.

**A sync was moved, not removed, and the profile says so.** `collect` still has to
turn the tensor into ints, because `append_token` applies the stop rules and a stop
rule needs a Python int. So `syncs=True` came off `sample` and went onto `collect`,
`sync_points` is still 1, and `recommended_model` still says `serial`. The next step
is vLLM's async output processing: resolve step N's tokens while step N+1 is already
in flight, which costs one token of overshoot past a stop condition and removes the
last synchronisation. `syncs_per_step(..., strategy="deferred", window=k)` is the
arithmetic for it, and it is arithmetic rather than code because the engine does not
do it yet.

## What I learned
1. **The measurement I could not make from the outside was "did it sync".** A
   synchronisation leaves nothing in a return value. It shows up as the host waiting,
   and on a box with no accelerator it does not show up at all, so no assertion about
   tokens can check that the sampler stopped reading back. Two things fixed that.
   `Readback` turns every deliberate journey home into an integer the gates read. And
   the tests monkeypatch `tolist`, `item` and `__int__` on `torch.Tensor` into raising
   for the duration of a call, which turns "did not sync" into a failure rather than a
   comment. That is the first test in this project whose subject is an operation that
   does not happen.
2. **The all-greedy fast path was the whole win on this box, and it is not the win I
   set out to get.** I went after the readback count and the count was already 1 for
   greedy rows: they were one batched `argmax` and one `.tolist()` since Day 40. What
   was actually costing 0.193ms at one row and 0.680ms at eight was `logits[greedy]`,
   a fancy-index gather that copies `[rows, 128256]` floats before an argmax that
   never looks sideways in the first place. `logits.argmax(dim=-1)` is the same answer
   for no copy. Half a millisecond at eight rows, from deleting an indexing expression
   that looked like it was selecting rows and was actually allocating 4MB.
3. **The saving grows with the batch, which is the opposite of what Day 46 taught me
   to expect.** Day 46's lesson was that the loop is roughly flat in the batch and
   therefore cheap per token at high B. This one is not flat: it was a gather whose
   size is the batch, so removing it gets better with more rows, 1.33x at one and
   1.90x at eight. Worth noticing because the flat-loop model is the one I now reach
   for by default, and this is a per-step cost that was not flat and was hiding inside
   a phase whose name suggested arithmetic.
4. **The sampled path shows nothing on a CPU and that is the measurement, not a
   failed one.** top-p at eight rows is 60.9ms before and 59.6ms after: 1.02x, well
   inside the noise. The filters sort a 128256-wide vocabulary per row and that
   dominates everything else by two orders of magnitude, while the eight
   synchronisations I removed cost nothing here because there is no bus to cross.
   `check_measurable` refuses to price them from this box, the same way Day 46's
   `check_device_timed` refuses to price an optimisation from a CPU profile. Reporting
   "1.02x" as the result of removing seven synchronisations would be true and would
   be a lie about a different machine.
5. **I nearly ran the before and after on different days, and it would have been
   worthless.** The obvious comparison was Day 46's profbench numbers against today's.
   I re-ran that exact config and the forward came out at 389ms, 775ms and 1135ms
   against Day 46's 307ms, 534ms and 982ms: the same code, 27% to 43% slower, because
   the box is busier today. Every phase in the loop moved with it. A before and after
   taken two days apart on a shared machine is a measurement of the machine's mood.
   The number that survives is a paired one: both implementations, in one process,
   seconds apart, over the same logits tensor, median of 25. That is why
   `rowwise_sample` lives inside `syncbench.py` as a copy of the Day-40 body rather
   than as a second path left behind in `sampling.py`, where it would quietly stop
   matching.
6. **Memoising the resolve is the difference between one sync a step and one per
   caller.** `TokenBatch.resolve` caches, and the first version of it was an `ids`
   property that recomputed. Nothing about that is visibly wrong: the values are
   identical every time. It just means the log line somebody adds next month costs a
   full round trip, and the counter is the only thing that would ever say so. The gate
   for it, `check_single_transfer`, reads a rate rather than inspecting code, which is
   the only form of this check that survives contact with a future commit.
7. **The `[rows]` tensor is anonymous and that is a real hazard, so the ids ride with
   it.** Between `sample` and `collect` nothing moves, but the thing this is building
   toward is deferring the resolve across a step boundary, and between one step and
   the next the scheduler releases finished rows and admits new ones. A row order that
   shifted in between hands one caller another caller's next token, both answers stay
   grammatical, and no shape check fires because the lengths still agree.
   `TokenBatch` carries `request_ids` and `check_in_row_order` compares them, which
   costs a tuple comparison per step and buys the one bug in this area that would
   otherwise be found by a user.
8. **`index_copy_` needs an index tensor, and building one is a launch rather than a
   sync.** The mixed path writes greedy results and each filter group into one `[rows]`
   buffer by index, and the indices come from Python lists via `torch.tensor(...,
   device=...)`. That is a host to device copy, which the host hands to the driver and
   walks away from; it is the opposite direction from the thing being removed. Easy to
   talk yourself out of on the grounds that it is "still a transfer per step", which is
   true and is not the transfer that costs anything.

## Diagram
[token-readback.png](../diagrams/token-readback.png). Left top is one decode step
drawn twice, four red stop lines through the old sample phase and one through the new
collect. Left bottom is the measured sweep with the greedy and sampled paths kept
apart. Right top is the three strategies and the per-journey cost model, right middle
is the two savings and why they do not add, and right bottom is the four gates plus
why the resolve is memoised.

## Tomorrow
The last synchronisation in a decode step is `collect`, and removing it means not
resolving step N's tokens until step N+1 has already been launched, which the decode
input can support because the next `input_ids` is the previous step's device tensor
and never has to come home at all. The cost is one token of overshoot past a stop
condition, discarded when the resolve catches up. Day 48 is that, priced first with
`syncs_per_step(..., strategy="deferred")` and gated by `check_in_row_order`, which
is exactly the check a one-step lag needs. `torch.compile` on the decode step is
after it, once there is no readback left for it to work around.

## Post angle
Day 47 of building an LLM inference engine from scratch. Yesterday's profile said
the sampler was eating 88% of my engine's Python loop, which made no sense: sampling
is an argmax. Today I found out why, and it was the return type. `sample_batch`
returned `list[int]`, and every one of those ints was its own `int(tensor)`. On a
GPU that is not a copy, it is a stop. Launches are asynchronous, so the host queues
kernels and runs ahead, and Python that fits under the arithmetic is free. Reading a
value back ends that, because the number does not exist yet. Eight sampled rows meant
nine stops a step. So the sampler now returns a `[rows]` tensor that stays on the
device and the step reads it home once. Three things I did not expect. The first is
that on the path my engine actually runs, all greedy, the sync count was already 1,
and the real cost was `logits[greedy]`: a fancy-index gather copying `[rows, 128256]`
floats before an argmax that never looks sideways. Deleting it is 1.33x at one row
and 1.90x at eight, and it gets better with the batch, which is the opposite of what
a loop cost usually does. The second is that the sampled path shows 1.02x here and
drops 8 synchronisations to 1, so the two savings are completely different quantities
and my box can only see one of them; there is a gate that refuses to price the other
from a CPU. And the third: I did not remove a sync, I moved it. `collect` still needs
a Python int to apply a stop rule, so the step still stops once and the profile still
says serial. Resolving it one step late, the way vLLM does, is what removes the last
one, and it costs one token of overshoot. 1116 green.
