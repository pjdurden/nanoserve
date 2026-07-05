---
title: "Day 20: the number the Triton kernel has to beat"
parent: Daily log
nav_order: 20
---

# Day 20: the number the Triton kernel has to beat

Date: 2026-07-04 · Week 6 · Phase 2 Paged memory

## What I added today
`src/nanoserve/readbench.py`: a small, stdlib-only, clock-injected timing core for
the paged read (`ReadTiming`, `ReadComparison`, `time_read`, `compare_reads`), plus
the runner `readbench.py` that wires a real `PagedKVCache` into it. Six new pure
tests in `tests/test_readbench.py` pin best-of, mean, median, and the speedup ratio
against a scripted fake clock, exactly the way Day 13's benchmark math was pinned.
The runner builds the cache from the default Llama-3.2-1B config (no weights: the
read cost depends on shapes and history, not on which tokens sit in the cache),
prefills it to a history length, then times two read closures over the fixed pool:
the Day-16 **gather** (rebuild the contiguous history, then a normal masked SDPA)
and the Day-19 **fused** read (`paged_attention_reference` over the scattered
blocks). Both are asserted byte-equal before timing, so only their speed is
compared. Pure suite **130 green**, ruff clean. The sweep across history lengths
16, 64, 256, 1024, 4096 is saved to `docs/daily/data/day-20-readbench.csv`.

## Why it matters
Week 6 is the Triton kernel, and you do not write a kernel against a vibe. The
whole reason the Day-18 reference and the Day-19 fused read exist is to give the
kernel a fixed correctness target; today gives it a fixed *performance* target. The
sweep says two honest things. First, the per-step read cost is roughly `O(history)`
and it hurts at long context: a fixed floor of about 0.13 ms at 16 tokens, then it
climbs to 32 ms at 4096, and the last leg alone (1k to 4k, 4x the history) costs
5.4x the time. Every decode step re-reads the entire cache, so the cache being
large is the cost. That climbing tail is precisely what the kernel is for. Second,
gather and fused sit within noise of each other, gather even a hair faster, because
both are plain torch doing the same `O(history)` index gather; the fused reference
folds the read into the score but does not make it cheaper. That is the honest
framing the reference deserves: it is the oracle, not the speedup. The speed only
arrives when a hand-written kernel streams K/V block by block and fuses the read
and the softmax into one pass, and now there is a number on the wall for it to
beat.

## What I learned
1. **Time the read, not the step, and the write has to leave the frame.** Both
   reads share the identical write (`_write`: grow the table, scatter the K/V), and
   both `append` and `paged_attention` *mutate* the cache when called. Timing them
   directly would have measured the write over and over and grown the pool until it
   exhausted. The fix was to fold the write into a one-time prefill and hand the
   timer two *pure* closures over the already-written pool, so each can run 400
   times without touching cache state. What you isolate is what you measure.
2. **Best-of is the honest microbenchmark number, not the mean.** On a shared CPU
   every sample is contaminated *upward* by scheduler jitter and never downward:
   nothing makes fixed work finish faster than the hardware allows. So the fastest
   sample is the cleanest estimate of the true cost, which is why `min_s` is the
   headline and `mean`/`median` are kept only to show the spread. The Day-256 point
   made this concrete: its median was 3x its best-of, pure noise, while the best-of
   stayed put run to run.
3. **A benchmark whose two paths tie is still telling you something.** I half
   expected the fused read to look faster and it did not, because on CPU it is the
   same gather. The tie is the result: it says the reference bought correctness and
   a clean interface, not speed, and it points a finger at exactly where the speed
   has to come from instead (the kernel, at long context). A microbenchmark that
   only ever confirms your hopes is not measuring anything.
4. **The cost curve has two regimes and the kernel only helps one.** At short
   context the per-step cost is a flat overhead floor (projection, softmax, Python),
   which paging and a kernel barely touch. The `O(history)` read only dominates once
   the history is long. That is the same shape production stacks report: paged
   attention and its kernel earn their keep at long context and large batch, not on
   a 16-token toy, and the curve draws exactly where the crossover sits.

## Diagram
[paged-read-benchmark.png](../diagrams/paged-read-benchmark.png). A log-log plot of
per-step read cost against cached history, gather and fused nearly on top of each
other, climbing from the 0.13 ms floor to 32 ms at 4096 tokens. The two side panels
name where the cost lives (a fixed floor at short context, the `O(history)` read at
long context) and what the kernel's job is (bend the long-context tail down by
fusing the read and score into one pass over the blocks). The two lines tracking
each other is the point: the reference is the oracle, and the speedup is still owed.

## Tomorrow
The kernel itself, or the Triton warm-up before it. With the target now measured,
the next step is the first cut of the `triton.jit` paged-attention kernel behind the
same `paged_attention` signature, held byte-close to `paged_attention_reference` and
benchmarked against today's curve with this same `readbench` harness. It needs a
GPU, so it will run gated like the weights tests, and the CPU curve from today is
the baseline it has to bend. If a learning day comes first, it is Triton basics
(program ids, block pointers, `tl.load`/`tl.store`) posted as the warm-up, so the
kernel lands on understood ground rather than copied incantation.

## Post angle
Day 20 of building an LLM inference engine from scratch. Week 6 is the Triton
kernel, and you do not write a kernel against a vibe, you write it against a number.
So today I benchmarked the paged read I already have. Two honest results. One: the
per-step read cost is O(history) and it bites at long context, a flat 0.13 ms floor
at 16 tokens climbing to 32 ms at 4096, and the last stretch alone (1k to 4k, 4x
the history) costs 5.4x the time, because every decode step re-reads the whole
cache. That climbing tail is exactly what a kernel is for. Two: the Day-16 gather
(rebuild the contiguous history, then attend) and the Day-19 fused read (attend over
the scattered blocks in place) tie, gather even a hair faster, because on CPU both
are plain torch doing the same O(history) gather. The fused reference folds the read
into the score but does not make it cheaper. That tie is the honest framing the
reference deserves: it is the correctness oracle, not the speedup. The speed only
shows up when a hand-written kernel streams K/V block by block and fuses read and
softmax into one pass, the shape vLLM and SGLang ship. Now there is a number on the
wall for it to beat. Two gotchas: time the read alone (fold the shared write into a
one-time prefill, or you just measure the write and grow the pool until it dies),
and quote best-of, not the mean (CPU noise only ever pushes a sample slower, so the
fastest one is the cleanest estimate of the real cost). 130 tests green.
#AI #LLM #vLLM #BuildInPublic #Claude #OpenAI
