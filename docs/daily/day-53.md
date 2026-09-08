---
title: "Day 53: the inputs stopped being built, and the witness that shared their storage"
parent: Daily log
nav_order: 53
---

# Day 53: the inputs stopped being built, and the witness that shared their storage

Date: 2026-09-07 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.inputs`, the decode step's four input tensors allocated once and written
in place. An `InputBuffer` is one `[max_rows, *tail]` int64 buffer plus a *staging*
mirror on the host, pinned when the buffer is on CUDA; `write` puts a Python list
into the mirror through its numpy view (which allocates nothing at all) and copies
it down with one `copy_`, or copies a tensor that is already on the device straight
in. `window(rows)` is `buffer[:rows]`, basic slicing, so the same storage at the
same address. `DecodeInputs` is the four of them, sized from the cache's row count:
`input_ids` and `positions` `[rows, 1]`, `write_slots` and `context_lens` `[rows]`.

The arithmetic is `input_cells`, `input_bytes`, `fresh_cells`, `fresh_allocations`,
`per_shape_bytes` and `sharing_ratio`. The gates are `check_addresses_stable`,
`check_plan_inputs_persistent`, `check_step_inputs_persistent`, `check_inputs_fit`,
`check_one_set_covers`, and `check_snapshots_present`, which is the correctness one.

The wiring is five changes. `DecodePlan` grows `context_snapshot` and
`write_snapshot`, host-side copies of the two tensors as they were at build time,
plus `context_list` and `write_list` to read whichever source is the real witness.
`check_plan_current` and `check_window_intact` go through those two properties
instead of through the tensors. `SlotTable.read` takes a `lengths_writer`, so the
lengths land in the caller's buffer instead of a fresh `[rows]` vector. The cache
gains `persist_inputs`, writes `positions`, `write_slots` and `context_lens` into
`self.decode_inputs`, and attaches the snapshots. The engine splits
`_decode_input_ids` into what the token *is* and what tensor it becomes, so the
persistent path can take the sampler's device tensor or a list of host ints, and the
`torch.cat` that padded a bucketed step becomes a wider window on a buffer.

`tests/test_inputs.py` is 72 tests. Suite **1503 green** (5 GPU-gated skips), ruff
clean.

`inputbench.py`, host only, no weights, 3 rows of an 8-row cache from a 16-token
prompt ([day-53-inputbench.csv](data/day-53-inputbench.csv)):

    steps   allocations       writes    cells    plan ms fresh -> buffered
        8        24 -> 0           24       72        0.560 ->    0.666
       64       192 -> 0          192      576        3.762 ->    4.193
      128       384 -> 0          384    1,152        7.406 ->    8.218
      512     1,536 -> 0        1,536    4,608       29.664 ->   32.778

and the second table, which is the memory question the day set out to answer, at
256 rows and 8192 tokens:

    one set of 4 inputs                        8,192 bytes
    a set per captured shape (36 shapes)      65,408 bytes       8.0x
    the Day-51 slot table                 16,777,216 bytes   2,048.0x

## Why it matters
**Four days have been about the same sentence, and this is the last clause of it.**
A replayed CUDA graph takes no arguments. It re-runs the kernels it recorded,
reading the buffers those kernels were recorded against, so every input has to be at
a fixed address, in a fixed shape, holding whatever this step wants said. Day 49
removed the graph breaks, Day 50 moved the addressing out of the forward so the only
guard left is on a tensor shape, Day 51 made the read rectangle a window so its
address stopped moving, Day 52 rounded the shape into a closed set. Everything else
the forward is handed was still a fresh tensor: 1,536 of them over a 512-step run,
none at an address the next step reuses. Today they are four buffers and 0.

**The day costs host time and buys an address, which is the opposite trade to every
other day this week.** 32.8ms against 29.7ms over 512 steps, about 10%. Nothing here
makes a step do less work. A fresh `[3]` int64 is cheap to allocate, and writing one
into an existing buffer still has to get the numbers out of a Python list, so the
buffered arm pays a staging write plus a copy where the fresh arm paid a constructor.
What changes is the second column, and `allocations` is not a performance number: it
is the count that has to reach *zero* rather than the count that has to get small.
One allocation in the middle of a captured run is not a slow run, it is a graph
reading storage that belongs to somebody else.

**Day 47's fastest path is the one this day makes slower, on purpose.** The token a
decode row forwards is the token the previous step sampled, and Day 47 left it on
the device so the input was `tokens.unsqueeze(1)`: a view, zero copy, no journey
home. A replay cannot use a view on the sampler's output, because that allocation is
new every step. So the buffer turns a zero-copy view into a one-copy device-to-device
write. It is a real regression measured against Day 47 and it is the only version of
that line a capture can replay.

**And the part I did not see coming: making an input shared disarmed two gates.**
Day 51 kept `context_lens` a fresh copy on purpose, and wrote down why: Day 50's
`check_plan_current` compares a plan's lengths against the cache's tables, so a
length that lives in a buffer the next step writes through *always* agrees with them.
The gate stops being able to fail rather than starting to. Day 51 avoided that by not
sharing. Day 53 has to share, because a replay needs the address, so the copy has to
become explicit: `context_snapshot` and `write_snapshot` are host-side tuples taken
from the lists the tensors were written from, free because the host already had the
numbers, and `check_snapshots_present` refuses a shared plan that carries none. The
general rule, which cost the most and is worth the most: **when a witness becomes
shared storage, it stops being a witness.**

**vLLM has been doing this the whole time and the reason is the same one.** Its
model runner keeps `input_tokens`, `input_positions` and `slot_mapping` as pinned
host tensors written in place and copied to fixed device buffers, and its cudagraph
replay writes into those buffers and calls `graph.replay()` with no arguments at all.
The interesting difference is what it does *not* need: its kernel reads each row's
length out of a tensor, so its capture list is over batch sizes only and its
per-shape state is smaller than mine. The input buffers are the part that is the
same, and they are the same because the constraint is a property of graph replay
rather than of anybody's attention kernel.

## What I learned
1. **The gate Day 51 wrote down as a reason not to do something was a spec for how
   to do it.** Day 51's module docstring says the lengths cannot be persistent
   because `check_plan_current` would stop working, and it has a test named
   `test_a_length_that_tracked_the_cache_would_disarm_the_staleness_gate` that
   *demonstrates* the failure with a hand-built tensor. Two days later that hand-built
   tensor is the design. The note did not stop me, it told me exactly what I owed:
   one snapshot, one gate, and the older test passes unchanged because it never used
   persistent inputs. Writing down why you are not doing something is worth as much
   as writing down what you did.
2. **`check_window_intact` was disarmed too and I nearly missed it.** I fixed
   `check_plan_current`, felt done, and then read Day 51's docstring line "the
   witnesses are the plan's own copies: `write_slots` and `context_lens` were taken
   at build time and cannot move." That sentence was true when it was written and my
   change made it false. Both gates had the same shape of bug and only one of them
   was in the module I was editing. The fix was one property, `plan.write_list`, and
   the reason I looked was that Day 51 had named its witnesses out loud.
3. **"Written in place" is a claim about the source as well as the destination.** My
   first buffer wrote with `buffer[:n] = torch.tensor(values)`, which allocates the
   tensor the class exists to avoid. The device buffer stops moving and the traffic
   does not. The fix is the staging mirror and its numpy view, `self._flat[:k] =
   values`, which writes into storage that already exists, and on CPU the mirror *is*
   the buffer so there is nothing to copy at all. This is what vLLM's pinned input
   tensors are and I had read that code without understanding what the pinning was
   for.
4. **The padded row's token can be garbage and its slot cannot, and the buffer makes
   that asymmetry a one-word difference.** `write(values, window=graph_rows)` widens
   what comes back past what was written, so a bucketed step's padded `input_ids`
   keeps whatever the buffer last held. That is safe for exactly Day 51's reason:
   nothing but a legal token id is ever written there, and the row's logits are
   dropped. `write_slots` and `context_lens` must not use `window`, because a stale
   slot in a padded row is a made-up token landing in a real sequence's history. Two
   inputs to the same forward, one of which may inherit and one of which may not.
5. **The arithmetic answered the question and then answered a better one I had not
   asked.** How many buffer sets does a 36-shape capture list need? One: a graph is
   per shape, but a buffer is indexed by *batch position*, so every shape's window is
   a prefix of the same storage. Then I priced the alternative and the whole thing
   collapsed: a set per shape is 65 KB and one set is 8 KB, against a slot table that
   is 16 MB. The input side of a decode step was never where the memory went. These
   buffers are worth having for their address and not for their size, and I would not
   have said that before writing the table.
6. **Splitting a function by what it returns made the fast path visible.**
   `_decode_input_ids` did two things: decide *what* the token is (a held device
   tensor or a readback) and build the `[rows, 1]` it becomes. The persistent path
   only wants the first half, because a tensor and a list take different routes into
   a buffer. Splitting it is four lines and it is the first time the two halves of
   Day 47's deferral have been separable. The buffer wants to know which it got; the
   old signature had thrown that away.
7. **A benchmark whose fast arm is slower is the honest one.** I expected the buffered
   arm to win on host time and it loses by 10% at every length. The number is real and
   the day still ships, because the column that matters is `allocations` and it went
   to zero. I nearly went looking for a way to make the timing come out the other way,
   which would have meant optimising a number that is not what the day is for.
8. **`0` is a legal slot and it is why so much of this works.** A fresh buffer, a
   padded column of the Day-51 rectangle, a padded row's position: all of them are 0,
   and 0 is legal in every one of those places by construction rather than by luck.
   Zero-filled allocation means an untouched cell of a new buffer is already valid on
   its very first read, which is the reason none of these classes has an
   initialisation step.

## Diagram
[persistent-decode-inputs.png](../diagrams/persistent-decode-inputs.png). Left top is
the Day-52 input side: three tensors a step at three new addresses. Right top is the
same step over four buffers, with the window the forward is handed, the padded row's
three different destinies, and the address that does not change. Left bottom is the
disarmed gate: a held plan's `context_lens` moving under it, `check_plan_current`
agreeing with everything forever, and the snapshot that re-arms it. Right bottom is
how many buffer sets a capture list needs, and the byte table that says it did not
matter much.

## Tomorrow
Every precondition is now written down and gated: no graph breaks (Day 49), no
Python state in the traced region (Day 50), a fixed rectangle address (Day 51), a
closed shape set (Day 52), fixed input addresses (Day 53). Day 54 is the capture
itself, `torch.cuda.graph` over one bucketed decode step: record per shape, replay
by writing the buffers and calling `replay()`, and a `CapturedDecode` that owns the
pool the graphs share. The gates written over the last four days become the
*preconditions* it checks before it records anything, which is the point of having
written them as functions rather than as comments. The number I expect to be wrong
about is memory: `check_capture_budget` says 36 shapes is a capture list, and 36
recorded graphs each holding a private workspace is a number nobody has measured on
this box yet.

## Post angle
Day 53 of building an LLM inference engine from scratch. A replayed CUDA graph takes
no arguments. It re-runs the kernels it recorded, reading the buffers those kernels
were recorded against, so an input allocated fresh every step is an input the graph
never sees. My decode step was allocating three tensors a step, 1,536 over a 512-step
run, none at an address the next step reuses. Today they are four buffers allocated
once and written in place, and the forward gets `buffer[:graph_rows]`, a window. The
host got 10% *slower* and that is fine: nothing here makes a step do less work, and
the column that matters went from 1,536 to 0. The part I did not see coming is that
sharing storage disarmed two gates. Day 50 checks a plan is not stale by comparing
its `context_lens` against the cache's tables, and Day 51 wrote down that the lengths
could not be a persistent buffer for exactly that reason: a length the next step
writes through always agrees with the tables, so the gate stops being able to fail
rather than starting to. Day 51's answer was not to share. Mine has to, so the copy
became explicit: the plan carries a host-side snapshot, free because the host already
had the numbers. The rule that cost the most: when a witness becomes shared storage,
it stops being a witness, and the second gate with the same bug was in a different
module. Also, I set out to compute how many buffer sets a 36-shape capture list needs
(one: a graph is per shape, a buffer is per batch position, so every window is a
prefix of the same storage) and found out it barely mattered. One set is 8 KB, a set
per shape is 65 KB, the slot table is 16 MB. These buffers are worth having for their
address, not their size. vLLM has kept pinned input buffers all along and now I know
what the pinning was for. 1503 green.
