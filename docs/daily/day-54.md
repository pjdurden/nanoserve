---
title: "Day 54: the step got recorded, and the output turned out to be a buffer too"
parent: Daily log
nav_order: 54
---

# Day 54: the step got recorded, and the output turned out to be a buffer too

Date: 2026-09-08 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.captured`, the decode step recorded once per shape and replayed after
that. A `CapturedGraph` is one recorded shape: the `replay()` that takes no
arguments, the output tensor those kernels write into, the input buffer addresses
they were recorded against, and the row tuple the recorded plan covers.
`CapturedDecode` is the wrapper: first sighting of a shape records, every sighting
after it replays, and a view with no plan falls through to eager and is counted.

The recorder is the seam, the way `torch_compiler` was on Day 49.
`cuda_graph_recorder` is the real one and it is four lines: warm up on a side stream,
`torch.cuda.CUDAGraph()`, record inside `torch.cuda.graph(graph, pool=pool)`, hand
back `graph.replay` and `graph.pool()`. `eager_recorder` records nothing and
reproduces a capture's *semantics* on a box with no device: it keeps the exact
argument objects, takes no arguments, and copies its result into one buffer allocated
at record time. `default_recorder` picks between them off the tensors.

The arithmetic is `score_cells`, `workspace_bytes`, `shared_pool_bytes`,
`private_pool_bytes`, `pool_sharing_ratio`, `capture_cost_s`, `launch_saving_s` and
`capture_breakeven_steps`. The gates are `check_capture_ready`,
`check_capture_preconditions`, `check_replay_rows`, `check_all_shapes_captured`,
`check_replays_dominate`, `check_pool_shared`, `check_pool_budget` and
`check_output_not_held`.

The wiring is small because the five days did it already. `Engine` gains
`capture_decode` and a `decode_graphs` that is always an object, "off" or not, and
sits in front of Day 49's `CompiledDecode` rather than instead of it; `Engine.build`
refuses `capture_decode` without `bucket_decode` and `persist_inputs` by name, and
`_decode`'s forward line goes through it.

`tests/test_captured.py` is 55 tests. Suite **1558 green** (5 GPU-gated skips), ruff
clean.

`capturebench.py`, host only, no weights, 2 rows of a real batched cache from a
16-token prompt ([day-54-capturebench.csv](data/day-54-capturebench.csv)):

    steps   captures: open set -> closed   replays   reuse    run ms eager -> replayed
        8              8 -> 1                    8     88%      9.619 ->   10.889
       32             32 -> 1                   32     97%     36.402 ->   45.251
       64             64 -> 1                   64     98%     74.882 ->   77.126
      128            128 -> 2                  128     98%    152.503 ->  158.508

and the pool table, at 256 rows and 8192 tokens with 32 heads and fp32 scores:

    width multiple    shapes      shared pool      private pools     ratio
              128       576          268.4 MB         17,414.2 MB     64.9x
              512       144          268.4 MB          4,554.5 MB     17.0x
             2048        36          268.4 MB          1,339.6 MB      5.0x
             8192         9          268.4 MB            535.8 MB      2.0x

    what a decode step holds at the top of that list, side by side:
      the [rows, heads, 1, ctx] score rectangle     268,435,456 bytes
      the Day-51 slot table                          16,777,216 bytes
      the Day-53 input buffers                            8,192 bytes

## Why it matters
**Five days of gates got spent, and spending them is what they were for.** Day 49
ended in `check_single_graph`, Day 51 in `check_mapping_is_window` and
`check_window_intact`, Day 52 in `check_shape_in_set` and `check_pad_inert`, Day 53
in `check_step_inputs_persistent` and `check_snapshots_present`. Today they are the
body of `check_capture_preconditions`, in that order, run before a graph exists. The
thing that made this a twenty-line function instead of a day of re-deriving is that
none of them were comments. A comment saying "the rectangle must be a window" would
have needed rewriting as code today; a function saying it needed importing.

**A replay does not hold tensors, it holds a Python object full of windows, and that
distinction is the whole design.** The recorded call keeps the `DecodePlan` and the
cache view from the step it was recorded on, and it keeps them for the life of the
process: step 500 replays with step 1's plan. That is sound for exactly one reason.
Everything the forward reads off a plan is storage that persists and gets written in
place, so the object is frozen and the numbers in it are not. `write_slots`,
`context_lens` and `positions` are Day 53's buffers; `slot_mapping` is Day 51's
table. Which makes `plan.rows` conspicuous: it is a Python tuple, so it says what
step 1 said, forever. `check_replay_rows` is one line and it is the only thing on
this page guarding a field rather than a pointer.

**The gates split into two kinds and the split is a performance decision I did not
expect to have to make.** `check_pad_inert` reads `int(plan.context_lens[i])`.
`check_window_intact` reads `plan.slot_mapping[i, n - 1]`. On a device those are
synchronisations, which is exactly what Day 47 and Day 48 spent two days removing
from this loop. So the value gates run once, at capture, where a sync is unavoidable
anyway, and the ones that run every replay are the two that read an address or a
Python tuple: `check_addresses_stable` and `check_replay_rows`. Writing "check the
preconditions" and then finding out that checking them every step would undo a week
of work is the sort of thing that only shows up when you try it.

**The output is a fixed buffer too, and I spent five days on the input side without
noticing.** A replay writes into the storage it was recorded against, so a captured
step's logits have a lifetime of exactly one step: the next replay writes through
them. Every gate before today points at the inputs, because that is where the loud
failure is. This is the quiet one, and it is the reason Day 48 still works. The
deferred window keeps token tensors across step boundaries on purpose, and it
survives a capture only because the sampler runs *outside* the recorded region, so
what crosses the boundary is the sampler's own allocation. Move the sampling inside
the graph and a window of N steps becomes N views on one buffer, all holding the
newest token, and every test that checks a finished request's text would still pass
until the window was longer than one.

**And the memory number Day 53 flagged came out wrong in a direction worth the
day.** Yesterday's log said "36 recorded graphs each holding a private workspace is a
number nobody has measured". Sharing the pool is right and the multiple is not 36. A
bucket set is geometric on the row axis, so the sum is dominated by its largest term,
and sharing over 36 shapes saves **5.0x**. Over 576 shapes it saves 64.9x, and the
shared column does not move at all between the two, because a pool is sized by the
biggest shape and not by the list. The number that hurts is the absolute one: 268 MB,
and it is one `[rows, heads, 1, ctx]` score rectangle from
`paged_attention_batched_reference`, sixteen times the whole slot table and thirty
thousand times the input buffers. No pool sharing touches it. vLLM's capture pool is
small because its kernel never builds that tensor, and that is the same sentence Day
52 wrote about why its capture list has no width axis.

## What I learned
1. **Writing a gate as a function rather than a comment is worth a day, and today is
   the day it paid.** Six functions imported from four modules, in the order the days
   wrote them, and the whole precondition check was twenty lines. The part I nearly
   got wrong was wrapping their exceptions in `CaptureUnsound` for tidiness. Letting
   `SlotsUnsound` propagate is better: "the rectangle is a gather" should name
   `nanoserve.slots`, because that is the module whose promise broke and the module
   where the fix is. The exception type is the pointer to the day.
2. **The held plan is frozen Python around live storage, and only one field is on
   the wrong side of that.** I expected to have to rebuild something per replay and
   there is nothing to rebuild, because Day 50 through Day 53 already moved every
   number the forward reads into a buffer. `plan.rows` is the exception, it is a
   tuple, and a replay over the same shape but a different set of sequences would
   write this step's tokens into the other batch's slots. Day 51's prefix requirement
   makes it rare rather than impossible, and rare is the reason to spend a line on it
   rather than a paragraph.
3. **Running all the preconditions on every replay would have undone Day 47 and Day
   48.** Two of them index into tensors, which is a device synchronisation, which is
   the exact cost those two days removed. So the checks got sorted by what they read:
   values at capture, addresses always. I would not have found this on CPU, where
   `int(tensor)` is free, if I had not gone looking for what each gate touches.
4. **I spent five days on the input side and the output side is symmetric.** It took
   writing the "a replay computes what an eager forward would" test, finding it
   compared a buffer against itself, and having to `.clone()`. Then the whole thing
   inverted: if the *output* is fixed storage, what else in this engine holds a tensor
   past the end of its step? Day 48's deferred window, and it is fine, and it is fine
   for a reason (the sampler is outside the capture) rather than by luck.
   `check_output_not_held` exists so that reason is checkable.
5. **A bucketed run is not one graph, it is one per bucket boundary it crosses.** The
   128-step arm recorded two, and I read it as a bug for a minute. A 16-token prompt
   plus 128 generated tokens is 144, which crosses the 128-token width bucket, so the
   run legitimately presents two shapes. `width_multiple` is not just a padding knob,
   it is a *rate*: one capture per `width_multiple` tokens of generation, per row
   bucket, and a long generation walks the width axis whatever you do.
6. **A geometric set is its own largest element, and that is why the sharing ratio is
   small.** I expected `pool_sharing_ratio` to be about the length of the capture
   list. It is 5.0x over 36 shapes and 64.9x over 576, and the two numbers have the
   same shared column, because a shared pool is a max and a private one is a sum. The
   useful reading is the other way round: a *coarse* bucket set barely benefits from
   pool sharing, and a fine one benefits enormously, so the two knobs are coupled and
   I would have set them independently.
7. **268 MB is the reference read, not the capture, and pricing it in the capture's
   module is the only reason I know that.** `workspace_bytes` was written to answer
   "what does a graph cost", and the answer is that a graph costs whatever the largest
   intermediate in it costs, and mine is a materialised score rectangle that a real
   paged kernel does not build. Day 22 and Day 23 lowered the single-sequence read to
   Triton; the batched one is still the plain-torch oracle, and today is the first day
   that has had a number for what that is worth in bytes.
8. **The stand-in recorder made the CPU and the device agree about one thing I would
   have got wrong.** A capture step returns a *replay's* output, not the recording
   call's, because the values produced while recording are not a result anybody should
   read. I only wrote it that way because the stand-in made the difference visible: on
   CPU the recording call really does compute the right answer, so it would have
   worked here and been wrong on a GPU. Building the fake to have the real one's
   semantics, and not its speed, is what caught it.

## Diagram
[captured-decode-replay.png](../diagrams/captured-decode-replay.png). Left top is
what a replay is: `replay()` with no arguments, the step-1 plan it holds forever, the
five fields that are windows and say what this step wrote, and the one that is a
Python tuple. Right top is the six preconditions in the order they run, split into
the ones that read tensor values (once, at capture) and the two that read an address
(every replay). Left bottom is the output buffer, two replays writing through one
address, and why Day 48's deferred window survives a capture and would not if the
sampler moved inside it. Right bottom is the pool table, shared against private at
three width multiples, and the three numbers a decode step holds side by side.

## Tomorrow
The capture works and it happens in the wrong place: the first step of each shape
takes a recording, mid-run, in front of a client who is waiting for a token. vLLM
does not do this, and `capture_cost_s`'s docstring already says why. Day 55 is
capturing the list *up front*: walk the bucket set before the server accepts
anything, drive each shape off a synthetic batch rather than a real request, and hand
back a warm engine whose first real decode is a replay. Two things I expect to be
wrong about. One is how to build a synthetic batch that presents a given shape
without writing garbage into the pool, since Day 52's sink slot only covers padded
rows and a fake batch is all padding. The other is the budget: `check_pool_budget`
wants a number for what is left after the weights and the K/V pool, `nanoserve.launch`
already has the memory probe, and nobody has connected the two.

## Post angle
Day 54 of building an LLM inference engine from scratch. A recorded CUDA graph is a
list of kernel launches bound to the addresses they were launched with, and
`replay()` takes no arguments: it re-runs those kernels over whatever those addresses
now hold. Five days went into making every one of those addresses stand still, and
each of those days ended in a gate written as a *function*. Today they became the
body of one twenty-line precondition check, in order, run before a graph exists.
That is the whole argument for writing "the rectangle must be a window" as
`check_mapping_is_window` instead of as a comment: a comment would have needed
rewriting today, and a function needed importing. The mechanism turned out to be
nicer than I expected. A replay holds the step-1 `DecodePlan` for the life of the
process and that is fine, because everything the forward reads off a plan is now
storage written in place, so the Python object is frozen and the numbers in it are
not. One field is on the wrong side of that: `plan.rows` is a tuple, so it says what
step 1 said forever, and it got its own one-line gate. The part I did not see coming
is that the *output* is a fixed buffer too. I spent five days on the input side and
never thought about the symmetric case: a replay writes into the storage it recorded
against, so a captured step's logits live exactly one step. Day 48 keeps sampled
tokens on the device across step boundaries, and it survives a capture only because
the sampler runs outside the recorded region. Move the sampling inside and that
window becomes N views of one buffer, all holding the newest token, and the tests
would pass until the window was longer than one. And the memory number I flagged
yesterday came out wrong in a useful direction: sharing one pool across a 36-shape
capture list saves 5x, not 36x, because a geometric set is dominated by its biggest
element and a shared pool is a max where a private one is a sum. The number that
actually hurts is 268 MB, and it is one score rectangle my reference read
materialises before it masks. That is not the capture's fault and no pool sharing
touches it. 1558 green.
