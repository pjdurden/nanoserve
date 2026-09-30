---
title: "Day 71: the split server grades its own passes before it serves"
parent: Daily log
nav_order: 71
---

# Day 71: the split server grades its own passes before it serves

Date: 2026-09-29 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was a boot probe. Days 69 and 70 gave each pass of the split
read its own entry point and its own test file, but on this box those files only
reach the tlsim programs. The jitted `_paged_attention_split_fwd` and
`_split_reduce_fwd` are gated tests that only run if somebody remembers to run pytest
on a card. A server knows its device before it serves, so now it runs the check
itself.

**`flash_decoding.py`.** Three additions:
- `chunk_partials_reference` is Day 70's test oracle moved into the module
  unchanged: loops over `(row, head, split)`, a direct softmax over exactly the keys
  chunk `s` owns, no launch. It deliberately shares no code with either pass one, so
  a bug in the tile walk or the online rescale would have to be written twice, in two
  different shapes, to get past it.
- `probe_split_passes(device, block, head_dim, n_rep, dtype)` builds a fixed case and
  grades three things in the order a failure should be blamed. Pass one:
  `split_partials` against the reference, slot by slot. Pass two: `split_reduce` over
  the *reference's* partials against `reduce_partials` over the same, so a broken pass
  one can't hide a broken pass two, or the other way round. The read: both passes
  composed, against the fp32 attention over the whole row. It raises `SplitUnsound`
  naming the pass, or returns a `SplitProbe` with the backend, geometry, live chunks
  per row and all three errors.
- `PROBE_ATOL = 1e-3`, the bound all three are held to.

The case is chosen, not sampled. Two rows over a three-tile mapping, with the chunk
one tile wide. Row 0 holds `2 * block + 1` keys, so all three chunks are live and the
last one owns exactly one key. Row 1 holds one key and is the control: every mutant
this repo has planted is exact on it. Two KV heads, each read by `n_rep` query heads.

**`launch.py`.** `probe_split_read(engine)` runs the probe on the engine's device,
using the read's own tile, the model's `head_dim` and GQA factor, and the weights'
dtype. `build_app` calls it right before `arm_split_read`, so a failed probe is a
`BootUnsound` and a refused server never reserved its arena. The report goes on
`app.state.split_probe` and into `/health` as `split_probe`, and `check_boot_info`
refuses a payload whose recorded probe is over its own bound.

**`tests/test_split_probe.py`.** 22 new tests run here and 2 more are GPU-gated. The
mutants are Day 70's overrun and Day 68's keep-the-winner reduce, monkeypatched into
the module where the probe looks them up, and each has to be refused *by name*. A
third plants a 1e-2 nudge into one accumulator to show the bound is the number the
report states. Suite **2269 green** (33 GPU-gated skips), ruff clean.

The probe on the host (tlsim, head_dim 64, n_rep 4, max abs error, bound 1e-3):

    what ran                    pass one   pass two     read    time
    honest, block 4, fp32        9.5e-07          0   2.5e-07   22 ms
    honest, block 16, fp32       1.9e-06          0   2.4e-07   15 ms
    honest, block 16, bf16       9.5e-07          0   2.4e-07   16 ms
    honest, block 64, fp16       2.4e-06          0   1.3e-07   19 ms
    overrun, block 16               6.22          -         -   refused
    keep winner, block 16             ok       2.92         -   refused

## Why it matters
**A test that only runs on request doesn't run.** The four gated tests from Days 69
and 70 are the only thing that grades the jitted passes, and the first card this
project boots on will be a rented box where the first command is `serve.py`, not
`pytest`. The probe moves the check into a place that runs every time: a
`--split-read` server on a card now can't serve without first showing that
`split_partials_triton` and `split_reduce_triton` agree with plain torch on this
device.

**It grades the kernel that will actually serve, not a nearby one.** `BLOCK_N`,
`HEAD_DIM` and `N_REP` are `constexpr` in the jitted pass one, so each combination is
a different compiled kernel. A probe run at `head_dim=64` on a server whose model has
`head_dim=128` would test a specialisation that never launches in production. So the
probe takes all four numbers from the engine rather than defaults.

**Before the arena, not after.** A refused boot that had already reserved its
partials would hold device memory until the process exited. Putting the probe one
line earlier costs nothing and means a failed server never holds the arena.

## What I learned
1. **The control row matters as much as the long row.** Both mutants this repo has
   planted are exact on a row with one live chunk, so a probe case built from
   "realistic" short prompts would pass both. The case needs one row that crosses two
   boundaries, and a test asserts the probe's own `live_chunks` has a 3 and a 1 in
   it, the same "check the control first" pattern as Days 69 and 70.
2. **Grading pass two on the reference's partials separates blame.** If pass two read
   pass one's output, a wrong pass one would make pass two look wrong too, and the
   error message would point at the wrong kernel. Feeding it the reference's
   workspace means each error names the pass that actually broke.
3. **The gap between honest and wrong is six orders of magnitude.** Honest passes sit
   at 1e-6 because both sides upcast the same pool values and fold in fp32; the
   mutants are off by 2 to 6. That's why a single absolute bound works across fp32,
   bf16 and fp16 pools. The whole-row reference also has to run in fp32: in bf16 it
   would be the least accurate number in the comparison.
4. **Monkeypatching needs a module-level lookup.** The probe calls `split_partials`
   and `split_reduce` by their module names at call time, so a mutant planted on the
   module is the one that runs. Binding them as default arguments would have made the
   probe impossible to test against a planted bug.

## Diagram
[split-boot-probe.png](../diagrams/split-boot-probe.png). Top left: where the probe
sits on the boot path, and the refusal it can raise. Top right: the probe's two rows
and which chunks each one makes live. Bottom left: the three grades, honest and under
both mutants. Bottom right: who grades the jitted passes, before today and after.

## Tomorrow
The probe says "correct" and nothing about "worth it". The split read exists to
shorten the longest program in a decode launch, and `SplitPlan.wave_speedup` predicts
that from the tile counts. The next step is to time the probe-sized case both ways
(`paged_attention_split` and the unsplit batched read) at boot and put the measured
ratio in `/health` next to the prediction, so the first boot on a card also says
whether the split pays for itself there.

The hardware caveat hasn't changed: `graphbench.py --weights ./weights --device cuda
--rates 1,2,4,8` on all three arms, with prompts past 514 tokens, is still the first
run to book. Today adds one thing to that list: read `split_probe.backend` off
`/health` and confirm it says `triton`.

## Post angle
Day 71 of building an LLM inference engine from scratch. My flash-decoding kernels
have gated tests that only run if someone remembers to run pytest on a GPU. A server
doesn't need to remember. So a split-read server now runs both passes on a tiny case
at boot, grades each against plain torch, and refuses to start if either is off. The
case matters: one row crosses two chunk boundaries and one fits in a single chunk,
because every bug I've planted is exact on the short row. Honest error is 1e-6, the
bugs are off by 2 to 6, and it all takes 20 ms. vLLM and SGLang ship the split; I'm
learning how to make it prove itself on the card it runs on. 2269 green.
