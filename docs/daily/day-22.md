---
title: "Day 22: the fused paged-attention kernel, modeled on the CPU"
parent: Daily log
nav_order: 22
---

# Day 22: the fused paged-attention kernel, modeled on the CPU

Date: 2026-07-07 · Week 6 · Phase 2 Paged memory

## What I added today
`paged_attention_kernel` in `src/nanoserve/kernels/paged_attention.py`: the fused
paged attention, written as a grid of Day-21 tlsim programs. Day 18's
`paged_attention_reference` gathers the whole history into a contiguous
`[seq_total, n_kv, d]` buffer and runs one softmax; Day 21's `paged_gather`
rewrote the read half as a grid of programs but still assembled that buffer before
scoring. The kernel folds the read, the score, and the softmax into a single
streaming loop. One program owns one query position and walks the history a `block`
of keys at a time: it reads each tile's K/V through the block table with a masked
`load`, scores it against the query, and folds the tile into a flash-attention
style online-softmax accumulator (running max, running denominator, running
weighted-V sum). The full history is never held; peak state is one tile plus the
per-head accumulators. Seven new pure tests in `tests/test_paged_kernel.py` pin it
to the oracle on a decode step, a causal prefill, a partial continuation (past
history plus several new queries, with an explicit `scale`), over physically
scattered blocks, under the config's GQA head map, and independently of the tile
size, plus the batch-1 guard. Pure suite **148 green**, ruff clean.

## Why it matters
Reading each token through its slot and then running a normal softmax, the way the
reference does, quietly rebuilds the contiguous buffer paging exists to avoid. On a
long history that buffer is the dominant memory traffic, and materializing it is
the exact cost the whole paged design was meant to delete. The kernel deletes it by
never scoring the full row at once. It streams the keys in tiles and keeps a
constant-size running softmax, so the memory it touches at any instant is one tile
wide, not the whole past. That is the flash-attention idea applied to a paged cache:
the online softmax lets you renormalize what you have accumulated so far to a new
running max as each tile arrives, so the answer at the end is the same as one big
softmax without ever forming one. Writing it against the tlsim CPU model first means
the loop next week's `triton.jit` kernel runs is already understood and pinned to
the oracle, so the GPU version is a transcription of a tested loop, not a fresh
guess debugged on rented time. This is the shape vLLM and SGLang ship in their
paged-attention kernels; today is that shape on ground I can single step.

## What I learned
1. **The online softmax is what lets the history stream.** A plain softmax needs
   the whole score row to find its max and its normalizer, which forces the
   contiguous buffer. The flash trick keeps a running max `m`, a running
   denominator, and a running weighted-V sum, and when a new tile pushes the max up
   it rescales the old state by `exp(m_old - m_new)` before adding the tile in. The
   accumulators are constant size no matter how long the history is, so the read can
   be a loop over tiles instead of one gather. The rescale is the load-bearing line:
   drop it and every tile after the first is normalized against the wrong max.
2. **A masked load returns zeros, and zero is not the same as absent.** The tail of
   a tile reads masked-off keys as `other=0.0`, and a query scores `0` against a
   zero key, which is a real, finite weight `exp(0 - m)`, not nothing. So after
   scoring I have to force the masked lanes to `-inf` explicitly, so their softmax
   weight is exactly zero. The mask keeps the read in bounds; a second, separate
   mask on the scores keeps the phantom keys out of the softmax. Two different jobs
   that both happen to be called "the mask."
3. **The tile size is a performance knob and a test can prove it.** Block sizes of
   1, 2, 3, 4, 8, 13, and 32 over a 13-token history all return the same attention
   to tolerance. That is the streaming analogue of Day 21's block-size transparency:
   if the answer shifted with the block size, the online rescale would be wrong, the
   way a gather that changed with the block size would mean the programs were
   stepping on each other. The block only decides how much SRAM a tile costs.
4. **Streaming trades bit-exactness for the memory win, honestly.** Where the
   Day-18 gather tests could demand `torch.equal`, the kernel only matches the
   reference to about `1e-5`, because folding the softmax tile by tile reassociates
   the exponent sums. That is not a bug to chase to zero; it is the same accuracy
   trade every flash-attention kernel makes, and the honest bar is equal-to-a-
   few-ulps. Loosening the tolerance is admitting what the algorithm actually is.

## Diagram
[paged-fused-kernel.png](../diagrams/paged-fused-kernel.png). One program folding a
10-token history in tiles of 4 into an online softmax: tiles 0 and 1 are full, tile
2 is the ragged tail (positions 8, 9 valid; 10, 11 masked to `-inf`). Each tile is
read through its scattered pool slots with a masked `load`, scored against the
query, and folded into the running max / denominator / weighted-V accumulators,
which stay one row wide the whole time. The three-line takeaway names why the fused
loop exists: streamed (peak state is one tile, the full history is never assembled),
correct (the online rescale makes it equal to one big softmax, to a few ulps), and
understood (this is the exact loop the GPU kernel runs, pinned to the reference on
the CPU first).

## Tomorrow
The real kernel. Now that the fused loop is pinned to the oracle, the next step is
transcribing `paged_attention_kernel` into an actual `triton.jit` kernel behind the
same signature, its inner loop the same tile read plus online-softmax fold but
streaming K/V from HBM into SRAM and running a grid of programs in parallel. It
needs a GPU, so it runs gated like the weights tests, held byte-close to
`paged_attention_reference` and benchmarked against the Day-20 curve with the same
`readbench` harness. Today's CPU model is the scaffold it is debugged against: when
the GPU kernel disagrees with the oracle, the tlsim loop is where the addressing or
the rescale bug is reproduced in plain torch first.

## Post angle
Day 22 of building an LLM inference engine from scratch. The paged read that gathers
each token through its slot and then runs a normal softmax quietly rebuilds the one
thing paging exists to avoid: the contiguous history buffer. So today I fused the
read, the score, and the softmax into one streaming loop, written on the CPU model
of the Triton programming model I built yesterday. One program owns one query
position and walks the history a tile of keys at a time: it reads each tile's K/V
through the block table with a masked load, scores it, and folds it into a
flash-attention online softmax, a running max, a running denominator, a running
weighted-V sum, all constant size. The full history is never assembled; peak memory
is one tile. The online softmax is the trick that lets it stream: when a new tile
pushes the running max up, you rescale everything accumulated so far by exp(old max
minus new max) before adding the tile in, so the final answer equals one big softmax
without ever forming one. Two gotchas that only show up when you write it by hand: a
masked load returns zeros, and a query scores a real finite weight against a zero
key, so you need a second mask that forces the tail's scores to minus infinity, out
of bounds and out of the softmax are different jobs. And the tile size has to be
invisible: block sizes 1 through 32 over a 13-token history all give the same
attention, or the rescale is wrong. It matches the reference to a few ulps, not bit
for bit, because streaming reassociates the exponent sums, the same accuracy trade
every flash-attention kernel makes, the shape vLLM and SGLang ship. Now the GPU
kernel next is a transcription of a tested loop, not a guess on rented time. 148
tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
