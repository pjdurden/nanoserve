---
title: "Day 21: the Triton programming model, modeled on the CPU"
parent: Daily log
nav_order: 21
---

# Day 21: the Triton programming model, modeled on the CPU

Date: 2026-07-06 · Week 6 · Phase 2 Paged memory

## What I added today
`src/nanoserve/kernels/tlsim.py`: a small, stdlib-plus-torch model of the Triton
programming model, run on the CPU with no GPU and no Triton installed. It has the
five primitives a paged-attention kernel is built from: `launch(grid, kernel, *args)`
that runs the same body once per program id (the SPMD loop), a `Program` handle that
answers `program_id`/`num_programs`, masked `load`/`store` over a flat 1-D buffer,
and `arange`/`cdiv` to build offsets and size the grid. On top of it, `paged_gather`
rewrites the Day-18 `pool[slot_mapping]` read as a grid of programs: each program
owns a tile of positions and gathers their K/V through the block table with the
ragged last tile guarded by a mask. Eleven new pure tests in `tests/test_tlsim.py`
pin the grid coverage, the masked-load `other` semantics, the masked-store no-write,
and, the payoff, that `paged_gather` is byte-equal to `pool[slot_mapping]` for every
block size, over physically scattered blocks, feeding the exact `paged_attention_reference`
softmax. Pure suite **141 green**, ruff clean.

## Why it matters
Week 6's headline is a hand-written Triton kernel, and Day 20 put the number on the
wall it has to beat. But a Triton kernel is not the torch code that surrounds it. It
launches a grid of programs, each running the identical body with only its
`program_id` different, and inside it does not slice tensors: it computes flat
integer offsets, forms block pointers, and moves data with masked `tl.load`/`tl.store`.
The mask is the load-bearing idea. A sequence of 10 tokens in tiles of 4 has a last
tile that is only half full, and that tile has to read and write without walking off
the end of the KV pool. In Triton the mask is what lets a lane point anywhere, get
masked off, and quietly return `other` instead of faulting. Get that wrong and the
kernel segfaults or reads garbage that no reference can catch, because the bug is in
the addressing, not the math. So before writing the kernel on rented GPU time, I
wrote the *loop* it runs against a CPU model of exactly those primitives and pinned
it to the reference. Now the kernel next week is a translation of an understood,
tested loop rather than a copied incantation, and the sim is the thing I can single
step when the real kernel disagrees with the oracle.

## What I learned
1. **The mask is the whole game, and it has to be safe on both sides.** A masked
   `load` cannot merely blank the output after gathering, because the masked-off
   lane's offset may be out of bounds and indexing it would fault before any
   blanking happens. So the sim neutralizes the offset first (`where(mask, off, 0)`),
   indexes the safe copy, then overwrites the masked lanes with `other`. The store
   mirror never writes the masked slots at all. That is not a sim detail: it is the
   exact property the ragged tail depends on, and writing it out by hand is what made
   it obvious that "masked" means "never touches memory," not "touches then discards."
2. **SPMD means order must not matter, and a test can prove it.** The programs run
   sequentially here and in any order on hardware, so a correct kernel gives each
   program a disjoint tile. `test_paged_gather_is_transparent_to_block_size` makes
   that concrete: block sizes of 1, 2, 3, 4, 8, 13, 20 over a 13-token history all
   return the identical gather. The tile size is a performance knob, never a
   correctness one, and if it ever were, that test would catch it. A gather that
   changed with the block size would mean the programs were stepping on each other.
3. **A block pointer is just base + arange, and 2-D is the same trick twice.** Each
   token is `n_kv * d` contiguous channels, so viewing the pool and the output as
   flat `[n, C]` matrices lets one program move a `[block, C]` tile: rows are
   positions (`rows[:, None] * C`), columns are channels (`arange(0, C)[None, :]`),
   and the mask broadcasts down the rows. That is the identical index arithmetic a
   Triton kernel writes with `tl.arange` and broadcasting; doing it in torch first
   demystified the pointer math that reads like line noise in real kernels.
4. **Modeling the slow thing is how you understand the fast thing.** `paged_gather`
   is slower than the plain `pool[slot_mapping]` it reproduces: a masked `where`
   gather does strictly more work than a direct index. That is fine, because its job
   is not speed, it is to make the kernel's loop expressible and testable in terms I
   can read. The honest framing, same as the Day-18 reference: the CPU model buys
   understanding and a fixed target, and the speed only arrives when the same loop
   runs as `triton.jit` streaming blocks on the GPU.

## Diagram
[tl-programming-model.png](../diagrams/tl-programming-model.png). The paged read as a
grid of three programs over a 10-token history in tiles of 4. Program 2 owns the
ragged tail (positions 8, 9 valid; 10, 11 masked off), gathers its two live tokens
through their scattered pool slots with a masked `tl.load`, and `tl.store`s them into
the ordered history that feeds the unchanged `paged_attention_reference` softmax. The
three-line takeaway names why the CPU model exists: correct (byte-equal to the torch
index for any block size), understood (`program_id` owns a tile, the mask keeps the
tail in bounds), and honest (this is slow, next week the same loop is a GPU kernel).

## Tomorrow
The kernel itself, now that the loop is understood and pinned. The next step is the
first cut of the `triton.jit` paged-attention kernel behind the same
`paged_attention` signature, whose inner loop is the `paged_gather` tile pattern from
today folded together with the score and softmax so K/V is streamed block by block
and never assembled into a contiguous history. It needs a GPU, so it runs gated like
the weights tests, held byte-close to `paged_attention_reference` and benchmarked
against the Day-20 curve with the same `readbench` harness. Today's sim is the
scaffold it is debugged against: when the kernel disagrees with the oracle, the sim
is where the addressing bug is reproduced in plain torch first.

## Post angle
Day 21 of building an LLM inference engine from scratch. Week 6 is a hand-written
paged-attention kernel, and before renting GPU time to write it I modeled the Triton
programming model on the CPU, no GPU, no Triton installed. Because a Triton kernel is
not the torch around it: it launches a grid of programs, each running the same body
with only its program_id different, and inside it does not slice tensors, it computes
flat offsets, forms block pointers, and moves data with masked tl.load / tl.store.
The mask is the whole game. A 10-token history in tiles of 4 has a last tile that is
only half full, and that tile has to read and write without walking off the end of
the KV pool; the mask is what lets a lane point anywhere, get masked off, and return
other instead of faulting. So I built launch, program_id, arange, cdiv, and masked
load/store as a slow CPU model, then rewrote the paged read pool[slot_mapping] as a
grid of programs gathering tiles through the block table, and pinned it byte-equal to
the torch index for every block size, over scattered blocks, feeding the exact Day-18
softmax. Two gotchas that only show up when you write it by hand: a masked load must
neutralize the out-of-bounds offset before indexing (masked means never touches
memory, not touches then discards), and the gather has to be identical for every
block size or your programs are stepping on each other. It is slower than the index
it reproduces, and that is the point: the CPU model buys understanding and a fixed
target, the speed arrives when the same loop runs as a triton.jit kernel on the GPU,
the shape vLLM and SGLang ship. Now the kernel is a translation of an understood
loop, not a copied incantation. 141 tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
