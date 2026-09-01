---
title: "Day 48: one journey home for k steps, and the rows it throws away"
parent: Daily log
nav_order: 48
---

# Day 48: one journey home for k steps, and the rows it throws away

Date: 2026-08-31 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.deferred`, the module that makes Day 47's `strategy="deferred"` real
instead of arithmetic. `DeferredOutputProcessor` extends Day 47's `OutputProcessor`
with a window: `defer` takes a step's `TokenBatch` without looking at it, `settle`
keeps the newest batch and brings everything older home in one `torch.cat` and one
`.tolist()`, and `flush` takes the rest. `TokenBatch.adopt_ids` is the other half
of that, so a batch resolved inside somebody else's transfer still memoises and
still refuses to pay twice. Three counters instead of one, because a token can fail
to reach its request for two different reasons: `overshoot_tokens` (the request had
already finished) and `abandoned_tokens` (it went back to the waiting queue and
will sample that token again after its recompute). The arithmetic is
`overshoot_bound`, `expected_overshoot`, `overshoot_waste_fraction`,
`deferred_saving_per_step`, `overshoot_cost_per_step`, `net_saving_per_step`,
`best_window` and `render`; the gates are `check_window_respected`,
`check_overshoot_bounded`, `check_tokens_conserved` and `check_input_is_held`.

`Engine(defer_window=k)` wires it. `_collect` hands the batch over instead of
resolving it, a new `settle` phase runs after both forwards are queued,
`_decode_input_ids` takes the next step's `input_ids` straight off the held tensor
when the rows have not changed under it, and `Engine.flush` closes the accounting at
the end of a run. `Scheduler(lookahead=k)` is the structural half: a held token is
already in the cache row and not yet in `Request.num_tokens`, so the growth rule,
the door check and `used_blocks` all count the headroom. `deferbench.py` at the root
sweeps the windows on one engine in one process and writes
[day-48-deferbench.csv](data/day-48-deferbench.csv). `tests/test_deferred.py` is 59
tests. Suite **1175 green** (5 GPU-gated skips), ruff clean.

Llama-3.2-1B, cpu fp32, 4 slots, 4 requests, 12 tokens each:

    window  transfers/step   collect    settle       step   overshoot   waste
      0          1.000      0.0515ms       -      983.85ms      0        0.00%
      1          1.000      0.0150ms   0.0517ms  1020.54ms      4        7.69%
      2          0.538      0.0149ms   0.0385ms  1017.35ms      4        7.69%
      4          0.308      0.0133ms   0.0257ms  1001.35ms      4        7.69%
      8          0.176      0.0149ms   0.0182ms  1010.58ms     20       29.41%

And one row that never leaves the batch, 40 tokens, which is the steady state the
1/k is a claim about: 0.268 transfers per step at window 4, 0.146 at window 8, and
one row of overshoot for the whole run.

## Why it matters
**Deferring one step removes nothing, and saying otherwise is the easy lie here.**
The measured transfer rate at `window=1` is 1.000, exactly what it was at
`window=0`. That is not a bug, it is what a one-step lag is: the same number of
journeys, taken later. It is worth being blunt about because the mental model that
makes deferral sound free is "the readback hides behind the next forward", and on
one stream it does not. `Tensor.tolist` issues its copy on the current stream and
waits for everything queued ahead of it, so a resolve placed *after* the next
forward is launched waits for that forward too. Hiding a synchronisation behind a
kernel needs a second stream and an event. This engine has one stream, so what
`window=1` actually buys is on the input side, and that is a Python saving rather
than a synchronisation one.

**Deferring k steps removes k-1 journeys, and that part holds on any stream.** The
tokens for k steps are k tiny `[rows]` tensors; `torch.cat` glues them on the device,
which is a launch the host walks away from, and one `.tolist()` brings the lot. That
is the whole mechanism, and `transfers_per_step` measured off `Engine.output` follows
it: 0.538, 0.308, 0.176 at windows 2, 4 and 8.

**The price is Day 29's number coming back off zero.** A request whose stop token is
still on the device is, as far as the scheduler can tell, still running: it stays in
the batch and the engine forwards a row for it that nobody will keep. Continuous
batching drove `waste_fraction` from 79% to 0.0 by construction, and this is the
first thing in the engine to put any of it back: 7.69% at windows 1 to 4, and 29.41%
at window 8 on a 12-token generation. `overshoot_waste_fraction(8, 12)` predicts
27.3% from `(k+1)/2` rows, which is close enough to the measured 29.41% that the
model and the run are describing the same thing.

**So the window has a best value rather than a large one.** The saving is
`latency x (1 - 1/k)` and it saturates: there was only ever one journey per step to
remove. The cost is `(k+1)/2` rows and it is linear. At a 20us journey against a
0.5ms decode row, `best_window` is 4 for a 200-token generation and 1 for a 12-token
one, and at window 16 the net goes negative even at 200 tokens. A chat server with
short answers should not defer at all, and that conclusion is a subtraction rather
than a preference.

## What I learned
1. **The saving I set out to get is not the saving `window=1` gives, and the one it
   gives is on the other end of the step.** I expected the win to be at `collect`.
   `collect` did fall, 0.0515ms to 0.0150ms, and then `settle` appeared at 0.0517ms,
   so at `window=1` the two together are slightly *more* than the one phase they
   replaced. What actually changed for free is `build_inputs`: the decode input is
   `held.tokens.unsqueeze(1)`, a view of the tensor the sampler left, instead of
   `torch.tensor([[r.output_token_ids[-1]] for r in requests])`. That is one fewer
   Python list, one fewer allocation and one fewer host to device copy per step.
   0.0676ms to 0.0620ms here, which is nothing, and it is the phase Day 46 named as
   the one a captured graph exists to delete.
2. **1/k is the floor and not the measurement, and the gap is scheduling.** I
   expected 0.250 at window 4 and measured 0.308. Deferral is only safe while the
   batch is the batch that was sampled: a held `[rows]` tensor whose rows are not
   this step's rows is the right length and the wrong rows, and using it hands every
   row after the change somebody else's token in fluent, plausible, wrong text. So
   every admission and every finish forces a full drain at the top of the next step.
   Four requests that all finish within a couple of steps of each other pay that
   several times in a short run. The single-row sweep is 0.268 at window 4, which is
   the same code with nothing churning, and the difference between those two numbers
   is the honest content of "steady state".
3. **The hazard of this day is not about tokens at all, it is about blocks.** Under
   deferral `Request.num_tokens` lags the cache row's length by however many tokens
   are held, because the row is written during the forward and the request is told
   afterwards. `Scheduler.blocks_needed_for` sizes the reservation off
   `num_tokens`, so it would top a running request up to blocks it has already
   outgrown. Nothing raises when that happens: `BlockTable.append` reaches the
   allocator itself, the pool is booked twice for one sequence, and the release frees
   only the blocks the request knows about, so the pool leaks a block per boundary
   crossed and every test about tokens still passes. `lookahead` is one added term in
   one expression, and finding out it was needed took longer than writing it.
4. **The two ways a token fails to arrive are not the same and one counter would have
   hidden the difference.** A token for a finished request is overshoot: real waste,
   caused by the window, and the thing `waste_fraction` should see. A token for a
   request that was preempted while it was in flight is not waste caused by anything
   I did today: that token was wanted, and the recompute will sample it again,
   because the request comes back over prompt-plus-generated and its position is
   unchanged. Charging it to the window would have made deferral look worse under
   memory pressure for a cost preemption already had.
5. **The property worth testing was not the speedup, it was that nothing moved.**
   Deferral changes when a token is looked at and not what was drawn, so the test
   that matters is a greedy run at `defer_window=k` producing byte for byte what
   `defer_window=0` produced, at every k, through EOS and through preemption. That
   test caught two real bugs (an empty prefill batch held as a row set that matched
   nobody, and the flush ordering inside `_decode_input_ids`) and it is the one
   assertion in the file that would still be worth keeping if the rest were deleted.
   The equality claim has a boundary and I want it written down: it holds for greedy
   because each row's token depends only on its own context. With the shared
   generator and a batch that preempts, deferral changes the order draws are consumed
   in, and the tokens would legitimately differ.
6. **Removing my synchronisation did not make the step free of them.** I wrote a test
   that runs a whole decode step with `tolist`, `item` and `__int__` monkeypatched to
   raise, expecting it to pass. It failed inside
   `paged_attention_batched_reference`, which validates its `context_lens` with
   `int(context_lens.min())` and `int(context_lens.max())` on every layer of every
   step. So the reference kernel reads back more times per step than the output path
   ever did. The test now scopes the claim to `_decode_input_ids`, which is what I
   can honestly assert, and the kernel's own bill is a separate one this day does not
   pay.
7. **A conservation identity is a better gate than a bound.** `check_tokens_conserved`
   says every deferred token is applied, overshot, abandoned or still held. It is
   four counters and one equality, it needs no threshold, and it fails on the exact
   class of bug this design invites: a drain that takes the wrong slice and loses a
   batch. The symptom without it is a request one token short of its budget that
   finishes anyway, which no test about output would notice.
8. **Overshoot depends on where the stop lands relative to the drain boundary, which
   is why the model is an average.** The 40-token single-row run at window 8 overshot
   by exactly 1 row, because 40 is a multiple of 8 and the budget was hit exactly at
   a drain. The 12-token four-request run at window 8 overshot by 20, five per
   request. Same window, same code, an eightfold difference in the price, decided by
   arithmetic between the generation length and the window. `expected_overshoot`
   returns `(k+1)/2` for that reason and not a bound.

## Diagram
[deferred-output.png](../diagrams/deferred-output.png). Left top is four decode steps
drawn twice, four red stop lines against one, with the green arrows showing the held
tensor becoming the next step's input. Left bottom is the measured sweep and why the
rate does not reach 1/k. Right top is the saving saturating against the cost growing,
right middle is `best_window` at two generation lengths, and right bottom is the
lookahead hazard.

## Tomorrow
The window is a policy and it is currently a constant a caller passes in. Everything
needed to choose it is now measurable in the engine: `transfers_per_step` is counted,
the overshoot is counted, and `best_window` is the subtraction. Day 49 is
`torch.compile` on the decode step, which Day 47 said would come once there was no
readback left in the middle of it to work around, and the deferred path is what
finally removes one from between the forward and the next launch. After that, the
second stream and the CUDA event that would let `window=1` actually hide its journey
instead of moving it.

## Post angle
Day 48 of building an LLM inference engine from scratch. Yesterday I moved my decode
loop's last synchronisation from `sample` to `collect` and said out loud that moving
one is not removing one. Today I tried to remove it, the way vLLM does: hold step N's
tokens on the device, launch step N+1 out of that same tensor, and read them home
late. First surprise: a one-step lag removes exactly zero journeys. Measured 1.000
transfers per step before and after. `Tensor.tolist` issues its copy on the current
stream and waits for everything queued ahead of it, so a resolve placed after the next
forward waits for that forward too. Hiding a sync behind a kernel needs a second
stream and an event, and I have one stream. What does work is going less often: k
steps of tokens are k tiny tensors, `torch.cat` glues them on the device (a launch,
not a stop), and one `.tolist()` takes them all. 0.538 transfers per step at window 2,
0.176 at window 8. The price is the interesting part. A request whose stop token is
still on the device is still in the batch, so the engine forwards rows nobody keeps,
which is exactly the waste continuous batching drove to zero on Day 29. 29% of the
forwards at window 8 on a 12-token generation. The saving saturates and the cost is
linear, so there is a best window and it is small: 4 at 200 tokens, 1 at 12. And the
bug that cost me the most time was not about tokens. A held token is already in the
cache row and not yet in the request's length, so the block allocator was sizing
reservations off a number that lags, and nothing raises when a row grows past the
blocks it paid for. 1175 green.
