---
title: "Day 77: the compiled run passes, and its control arm was not compiled"
parent: Daily log
nav_order: 77
---

# Day 77: the compiled run passes, and its control arm was not compiled

Date: 2026-10-06 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was the last path of the card run no smoke had reached: a toy
run with `--compile dynamic`, so `CompiledDecode` builds behind a booted server and
the capture records over a compiled forward. Both earlier smokes passed `--no-compile`.

**The first compiled run passed, and it was wrong.** Two arms, burst, the toy
checkpoint: exit 0, all eight claims `ok`. Under `loading the eager arm` the stderr
said:

    torch._dynamo hit config.recompile_limit (8)
       function: 'inner' (.../torch/_dynamo/external_utils.py:67)
       last reason: 0/7: tensor 'args[0]' size mismatch at index 0. expected 1, actual 4

Dynamo keeps its compiled entries per *code object*, for the life of the process.
Every arm runs the same `LlamaModel.forward`, and the arms boot one after another in
one process. The graphed arm's warm-up filled entries for its buckets, the eager
arm's compile added to the same list, and at eight dynamo gave the frame up and ran
the eager arm's forward in the interpreter. The control arm of a compiled comparison
was the one arm not compiled, because it booted second. Nothing raised and the
answers matched, so no claim could see it.

**The fix, in three parts.**
- `compiled.dynamo_frames_abandoned()` reads dynamo's own count of frames it gave up
  on at the limit (`counters["unimplemented"]`, keys starting
  `Dynamo recompile limit exceeded`). `compiled.reset_compile_cache()` is
  `torch._dynamo.reset()`, which empties the cache and leaves the counter alone.
- `graphbench.py`'s `measure` resets before each arm boots when the run compiles, and
  charges each arm the frames abandoned between its boot and the end of its load.
- `ArmDelta.abandoned` carries the per-arm counts (`None` when the run did not
  compile), the render prints `compile  frames abandoned at the limit: graphs 0,
  eager 0`, and `compile_claims` adds one gate, "every compiled arm stayed compiled",
  last in the list and only for a compiled run. A `--no-compile` claim list is
  Day 76's, unchanged, and so is the CSV header: the count reaches the CSV through
  `claims_failed`.

`tests/test_compile_smoke.py` has 20 tests: the counter against a real (eager
backend) compile walked past a patched limit, the gate and its render, the script's
`measure` with stubbed servers (resets come before boots, each arm is charged its
own frames, no reset and no count when off), and the compiled smoke as a process.
The smoke is about 90 s, most of it inductor, so it stays in the suite with no marker.
Suite **2402 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**This is the bias that would have shipped in the card run's CSV.** The eager arm
running uncompiled is slower for a reason that has nothing to do with CUDA graphs,
and the table divides by it. Every `itl_p50_speedup` would have carried part of a
compile the other arm got and this one did not.

**Which arm lost depended on boot order.** Three-arm runs boot split, graphs, eager.
Before the reset, the three-arm compiled run lost frames in the graphs arm too (a
`paged_attention` guard on `layer`, inherited from the split arm's entries). A
benchmark whose fairness depends on the order of a loop is not measuring the arms.

**It also cost time.** The three-arm compiled toy run went from 2m45s to 1m30s:
the old run spent its minutes recompiling frames whose lists were already full of
another arm's entries.

## What I learned
1. **Dynamo's cache is per function, not per model.** Two engines in one process
   share the entries for `LlamaModel.forward`. The guards keep their graphs apart,
   and the limit counts them together. A fresh process is what a server gets, so a
   benchmark that boots several servers in one process has to give each one a reset.
2. **The give-up is a counter, not just a warning.** `recompile_limit` prints once on
   stderr and also bumps `counters["unimplemented"]`. Reading it around an arm is the
   measured version of what `CompiledDecode.fell_back` could only predict.
3. **My first counter test counted nothing.** It was the Day 49 failure in miniature,
   an int read off an object, under `dynamic=True`. Read through a closure, the int
   went symbolic and the frame compiled once. A static compile with a new shape per
   call guards on every version. And the helper resets first, because its function
   is one code object however many tests call it.
4. **The remaining FAIL is honest.** With the reset, the three-arm compiled run still
   loses one frame in the split arm: tlsim's `split_kernel`, a Python loop over
   program ids guarded on `i == 0`. That is the CPU stand-in for a triton kernel, so
   it may not happen on a card, and now it prints in the log and in `claims_failed`
   instead of only on stderr.

## Diagram
[compile-arm-reset.png](../diagrams/compile-arm-reset.png). Top left: one entry list
for `LlamaModel.forward`, filled by the graphed arm and finished by the eager arm.
Top right: reset, count, claim. Bottom left: what the compiled smoke checks. Bottom
right: the three-arm compiled run before and after, and the tlsim frame that is left.

## Tomorrow
The split arm's abandoned frame. Under tlsim the kernel is Python that dynamo traces
into, one `program_id` at a time. Either the CPU split read gets
`torch.compiler.disable` around the tlsim launch (the honest move, since it stands in
for a launch that dynamo would not trace), or the gate learns that a tlsim frame is
not the forward. A test pins whichever is chosen to the three-arm compiled run.

The card run still comes first when hardware is booked, and its CSV now says whether
every arm ran the forward it compiled.

## Post angle
Day 77 of building an LLM inference engine from scratch. First benchmark run with
torch.compile on: exit 0, every claim ok. The stderr said the eager arm hit dynamo's
recompile limit. Dynamo caches per function, every arm runs the same forward in one
process, and the second arm inherited the first one's entries. The control arm wasn't
compiled. Fix: reset per arm, count abandoned frames, fail the run if any. 2402 green.
