---
title: "Day 23: the real triton.jit paged-attention kernel"
parent: Daily log
nav_order: 23
---

# Day 23: the real triton.jit paged-attention kernel

Date: 2026-07-13 · Week 6 · Phase 2 Paged memory

## What I added today
`src/nanoserve/kernels/triton_paged_attention.py`: the Day-22 fused loop, transcribed
into an actual `@triton.jit` kernel. Day 22 wrote the read, the score, and the online
softmax as a grid of `tlsim` programs running in a Python `for`, pinned to the Day-18
oracle. Today the same loop streams its tiles from HBM into SRAM and its programs run
in parallel on the card. Two things change on the way down. The grid grows an axis:
program `(i, h)` now owns query position `i` of query head `h`, so the per-head loop
becomes free parallelism and the accumulators collapse from `[n_q]` and `[n_q, d]` to
a scalar and one `[BLOCK_D]` row, small enough to live in registers. And every block
shape has to be a power of two, because that is all `tl.arange` takes, so the head
dimension is padded up to `BLOCK_D` and the key tile to `BLOCK_N` with a mask
discarding the lanes past the real extent.

The layering is the actual design decision. Every integer the kernel turns into a
pointer (the slot stride, `past`, the grid, the padded tile sizes) is computed by
ordinary Python in `check_paged_inputs`, `launch_grid`, and `next_power_of_2`, which
run and are tested on any box, GPU or not. The jitted body is held to
`paged_attention_reference` by five tests gated behind a new `requires_triton_gpu`
mark. `select_backend` dispatches: `paged_attention` launches the kernel on CUDA with
Triton importable and otherwise falls back to the Day-22 CPU model, so the engine runs
on a laptop while the kernel runs on the card. Eighteen new tests in
`tests/test_triton_paged_attention.py`, thirteen of them pure. Suite **176 green**
(5 GPU-gated skips on this box), ruff clean.

## Why it matters
This is the first code in the repo that cannot fully run where it was written, and
that is the whole lesson. A GPU kernel is not debuggable the way a Python loop is:
there is no stepping into a program, the failure mode of a bad pointer is a wrong
number or a segfault, and the thing you most want to inspect (what tile did program
17 actually read) is exactly what the hardware hides. So the kernel is built to push
every bug it can into code that is not the kernel. The addressing arithmetic, which
is where paged-attention bugs actually live, is plain Python with plain tests. The
numerics are pinned to an oracle that ran on the CPU three days ago. What is left
inside `@triton.jit` is a transcription of a loop that was already correct, and when
the GPU version disagrees with the reference, the tlsim model from Day 22 is where
the disagreement gets reproduced in readable torch first. That is the shape vLLM and
SGLang ship their paged-attention kernels in, a jitted body wrapped in ordinary host
code that computes every index: not because the kernel is the hard part, but because
the pointer math is, and pointer math you can single step is pointer math you can fix.

## What I learned
1. **`tl.arange` only takes powers of two, and that leaks into the whole design.**
   A head_dim of 48 or a tile of 13 cannot be a block shape. The fix is not to forbid
   them, it is to round the extent up to `next_power_of_2` and mask the lanes past the
   real one, which is the same mask discipline the CPU model already used for the
   ragged tail of the history, now applied to the channel axis too. The mask is what
   makes an awkward shape legal, so the awkward shapes are the ones worth testing.
2. **The grid is where the per-head loop goes.** On the CPU a program looped its query
   heads because a loop is a loop and it costs nothing to write. On a GPU that loop is
   parallelism I would be throwing away, so the head becomes a second grid axis and
   each program owns exactly one query, one head, one independent softmax, sharing
   nothing. The accumulators shrink to a scalar and a row, and that is precisely what
   lets them stay in registers rather than spilling. The shape of the grid decides the
   shape of the state.
3. **Triton is absent, not broken, on a CPU-only box.** It rides along with the Linux
   GPU torch wheel and simply is not in a CPU wheel, so `import triton` raises
   `ModuleNotFoundError` rather than failing at launch. That means "can I compile" and
   "can I launch" are two different questions with two different answers, and
   `select_backend` has to ask both: `has_triton()` for the package,
   `torch.cuda.is_available()` for the device. Neither implies the other, and a box
   with the package but no card is a real configuration.
4. **A fallback should be the model, not a fast path.** The CPU branch is not a
   degraded approximation of the kernel, it is the loop the kernel was transcribed
   from, correct to the same few ulps and simply slow. Keeping it honest about that is
   what makes it useful: it is the thing the GPU is checked against, so it has to stay
   the reference implementation and never drift into being a shortcut.

## Diagram
[triton-paged-kernel.png](../diagrams/triton-paged-kernel.png). The same fused loop at
two altitudes. On the left, the Day-22 CPU model: one Python program per query
position, looping its heads, holding `[n_q, d]` accumulators. On the right, the jitted
kernel: a 2-D grid where program `(i, h)` owns one query and one head, its tile padded
to a power of two and masked back down, its accumulators a scalar and a row in
registers, its K/V tiles streaming from HBM into SRAM. The band across the bottom is
the layering that makes it debuggable: host Python computes every index and is tested
everywhere, the jitted body is a transcription and is tested on a GPU, and the CPU
model underneath is what a disagreement gets reproduced against.

## Tomorrow
Benchmark it. The kernel is correct and completely unmeasured: the Day-20 `readbench`
harness drew the O(history) curve for the gather, and the fused kernel needs to be put
on the same axes to show the streaming read is actually flat in the history where the
gather was linear. Then wire `select_backend` into `gqa_attention`'s paged read so the
model path uses the kernel on a card instead of calling it only from tests, which is
the last step that turns Week 6 from a kernel in a file into the engine's real
attention.

## Post angle
Day 23 of building an LLM inference engine from scratch. Yesterday's fused paged
attention was a grid of pretend GPU programs running in a Python for loop. Today it is
a real triton.jit kernel, and the interesting part is not the kernel. It is that a GPU
kernel is the least debuggable code in the project: you cannot step into a program, a
bad pointer shows up as a wrong number or a segfault, and the one thing you want to
see, which tile did program 17 actually read, is exactly what the hardware hides. So
the kernel is built to push every bug out of itself. Every integer that becomes a
pointer, the slot stride, the past offset, the grid, the padded tile sizes, is computed
by ordinary Python that runs and is tested on a laptop with no GPU at all. The jitted
body is a transcription of a loop that was already pinned to an oracle on the CPU three
days ago, so when the GPU disagrees, the CPU model is where the disagreement gets
reproduced in readable torch. Two things did change on the way down. The grid grew an
axis, because a per-head Python loop is free parallelism on a card, so program (i, h)
owns one query and one head and its accumulators shrink to a scalar and a row that fit
in registers. And every block shape has to be a power of two, because that is all
tl.arange takes, so a head_dim of 48 or a tile of 13 gets rounded up and masked back
down, the same mask discipline the ragged tail of the history already needed, now on
the channel axis too. Triton is absent rather than broken on a CPU box, so it falls
back to the Day-22 model, which is correct and slow and stays the reference. 176 green.
