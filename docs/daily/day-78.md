---
title: "Day 78: a stand-in for a launch has to be opaque like a launch"
parent: Daily log
nav_order: 78
---

# Day 78: a stand-in for a launch has to be opaque like a launch

Date: 2026-10-07 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday ended on the one frame Day 77's gate still caught: the three-arm compiled
toy run (`--split-read --compile dynamic`) failed "every compiled arm stayed
compiled" in the split arm, and the stderr said why:

    torch._dynamo hit config.recompile_limit (8)
       function: 'torch_dynamo_resume_in_split_kernel_at_877'
       last reason: 14/7: i == 0  # qi = q_rows[i, h]

On a laptop the split read is tlsim, and `tlsim.launch` is a Python loop that calls
the kernel body once per program id. Dynamo traced into that loop. The body reads its
row's length as an int (`int(load(len_buf, ...)[0])`), which is a graph break, so the
rest of the body became a resume frame. That frame indexes with `i`, `h` and `s`,
ints dynamo guards on, and it always specialises 0 and 1. Three ids over a
`(row, head, chunk)` grid is more than eight entries, and the ninth was given up.

**The fix is one decorator.** `tlsim.launch` is wrapped in
`torch.compiler.disable`. A compiled forward that reaches a tlsim read now breaks the
graph at the launch, runs the grid in the interpreter, and resumes after it. Every
tlsim kernel goes through `launch` (the split's pass one, its reduce, the streamed
read, Day 21's gather), so the one wrapper covers all of them, and the Triton path a
card takes is untouched.

`tests/test_tlsim_compile.py` has 8 tests in three layers:
- **the launch**: a `(4, 3, 3)` grid with the split kernel's shape (read a length as
  an int, then address the cell with all three ids). Compiled at the default limit it
  gave up 1 frame before the fix and 0 after, and it writes exactly what the eager
  launch writes. The wrapper keeps `launch`'s name and Day 21's docstring.
- **the split read**: `paged_attention_split_kernel` compiled with
  `recompile_limit=2`, four rows in three chunks. 1 frame abandoned before, 0 after,
  and the attention matches `paged_attention_batched_reference`.
- **the process**: the three-arm compiled run that failed yesterday. Exit 0, no
  `recompile_limit` and no `split_kernel` in stderr, `split 0, graphs 0, eager 0`,
  and the compile claim out of `claims_failed` in the CSV.

Suite **2410 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**The stand-in should fail where the real thing fails, and nowhere else.** A card
hands dynamo one call for a Triton launch. It never walks a grid. A failure that
only exists because the CPU model of a kernel is traceable Python is a failure of
the model, and the fix belongs in the model.

**The other option was worse.** Teaching the gate that a frame named `split_kernel`
does not count would have been a gate that knows where its failures come from and
excuses them. The next frame to hit the limit inside a tlsim kernel would have been
excused too, and so would a real one with an unlucky name.

**It was also most of the run.** The three-arm compiled toy run went from 3m52s to
1m22s. The split arm's ITL p50 barely moved (210 ms to 200 ms), because the read is
still tlsim and still slow. What went away was dynamo compiling a resume frame for
every program id it had not seen.

## What I learned
1. **Dynamo specialises 0 and 1 even when an int goes dynamic.** My first test was a
   1-D grid of six programs under a limit of two, and it passed before the fix:
   after one recompile `i` went symbolic and every later id shared an entry. The
   split's grid has three ids, each with its own 0 and 1, and that product is what
   outruns eight. The test grid is `(4, 3, 3)` for that reason.
2. **A graph break inside an inlined function turns the rest of it into a frame of
   its own.** `split_kernel` is a closure called from a loop, and the break on the
   row length is what made its tail a cached frame with guards, once per program.
3. **`torch.compiler.disable` is a boundary, not an off switch.** The forward around
   the read still compiles; the graph stops at the launch and picks up after it. That
   is the same shape a compiled forward has around a custom op on a card.

## Diagram
[tlsim-launch-opaque.png](../diagrams/tlsim-launch-opaque.png). Top left: the split
kernel's body, its graph break, and the resume frame's entries filling up. Top right:
the decorator, and why it is the honest move. Bottom left: the three layers of test.
Bottom right: the three-arm compiled run before and after.

## Tomorrow
The three-arm compiled run is green except for "the arms are comparable", which wants
20 frame gaps and the toy run has 11. That is a sample-size gate doing its job on a
six-request smoke, not a bug, but it means no toy run can pass every claim. Next: a
`--requests` and `--max-tokens` pair for the smoke that clears the floor in the same
wall time budget, or an explicit note in the smoke tests that this one FAIL is
expected, so a new FAIL next to it is not lost.

The card run still comes first when hardware is booked, and now every arm of a
compiled three-arm run reports a compile it kept.

## Post angle
Day 78 of building an LLM inference engine from scratch. My CPU model of a Triton
kernel is a Python loop over program ids, and torch.compile traced into it: one
recompile per program until dynamo gave the frame up. A card hands dynamo one call
per launch. So the model now does too: one `torch.compiler.disable` on launch.
3m52s to 1m22s, every arm compiled. 2410 green.
