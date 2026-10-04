---
title: "Day 74: the benchmark script gets its third arm, and pads its own prompt past the chunk"
parent: Daily log
nav_order: 74
---

# Day 74: the benchmark script gets its third arm, and pads its own prompt past the chunk

Date: 2026-10-04 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was the script. Everything Days 68 to 73 built for the split
arm (the arena, the crossing gate, the boot grades and the gate that holds them to
what served) ran inside `tests/test_reads.py`. `graphbench.py`, the script a card run
would actually use, still booted two servers, graphs and eager, and neither split.
Now `--split-read` adds the third.

**`nanoserve/graphbench.py`.** Three additions:
- `split_prompt(prompt, workspace, count=, max_tokens=, max_model_len=)`. It repeats
  the prompt, in whole copies, until the prompt alone is at least `split_merge_floor`
  tokens by the caller's counter. It refuses a one-chunk (or missing) arena, a
  padded prompt plus `max_tokens` past `max_model_len`, an empty prompt, and a
  counter that never grows.
- `ArmDelta.split`, an optional third `ArmReport`. It brings
  `split_itl_p50_speedup` and `split_itl_p99_speedup`, both taken against the
  graphed arm. `row()` appends the split latency columns and `SplitBoot.row()` after
  the two-arm columns, and `render()` prints the split arm, its boot line, and a
  "split vs graphs" ratio line.
- `split_claims(delta)`, the split arm's five gates as `(check, note)` pairs: it
  batched, it gave the graphed arm's answers, the flag reached the cache, the rows
  crossed a chunk, the boot grade is about the read that served. An empty list for a
  two-arm run.

**`graphbench.py` the script.** `--split-read` boots the split arm first (the graphed
arm's flags plus `split_read`), takes its arena off `app.state.workspace`, loads the
checkpoint's tokenizer and pads `--prompt` with `split_prompt`. Only then does it
build the plans, and the same plans go to all three arms. `build_one` gained
`split=` and refuses a split without buckets before any weights load. The parser
moved into `make_parser()` so a test can reach it. `*split_claims(delta)` joins the
claims list, and the sweep table grows two split columns.

**`tests/test_split_script.py`.** 28 new tests: the padding arithmetic and its four
refusals, the three-arm delta (ratios, columns, the two-arm prefix, the render), the
claims (all pass, then each one failing alone), and the script's wiring through a
monkeypatched `build_app`. Suite **2338 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**The first card run can now be one command.** `graphbench.py --weights ./weights
--device cuda --rates 1,2,4,8 --split-read` boots three servers, sends them the same
bytes, and prints the split arm's ITL, its boot grade and its ratio against the
graphed arm in one table. Before today, getting that number meant writing a script.

**The load is decided by the server under test.** The floor is
`keys_per_split + 2`, and `keys_per_split` comes from the plan the split arm made on
this card at this `max_model_len`. A prompt length hard-coded in the script would be
right for one geometry and quietly short for the next, and `check_arm_split` would
fail the run after twenty minutes of serving.

**The comparison changes one thing.** The split arm is the graphed arm plus
`split_read`: the same capture, buckets, persistent inputs and compaction. So its
ratio is against `graphs`, and a ratio against `eager` would credit the read with
what the capture saved.

## What I learned
1. **Count the half of the row you control.** The first version padded until prompt
   plus `max_tokens` cleared the floor. That works on paper and fails at runtime: a
   completion that hits EOS on its second token leaves a 500-token prompt inside a
   512-key chunk. The prompt alone has to clear it.
2. **Count with the server's call.** The completions handler does
   `tokenizer.encode(prompt)`. The script counts with the same call on the same
   checkpoint's tokenizer. A word count would be off by the tokenizer's ratio, and
   off in the direction that fails the gate.
3. **The arm that decides the load boots first.** All three arms have to get the
   same plans or `check_same_answers` compares two workloads, and only the split
   arm knows the floor. So the split arm loads, publishes its arena, sets the
   prompt, serves, and is dropped before the other two load. One engine on the card
   at a time.
4. **A claims list beats a function that raises.** The script reports every claim
   and only stops on `--sanity`. Returning `(check, note)` pairs lets one failure sit
   next to the four that passed, and returning `[]` for two arms means the script
   splices it in with no `if`.

## Diagram
[graphbench-split-arm.png](../diagrams/graphbench-split-arm.png). Top left: the order
`measure()` runs in with `--split-read`. Top right: the padding, a six-token prompt
against a 512-key chunk. Bottom left: the five claims and the gate behind each.
Bottom right: what a three-arm load prints (fixture numbers).

## Tomorrow
The script runs three arms, but the only test of `measure()` end to end is still the
in-process one in `tests/test_reads.py`. Next is a toy-weights smoke for the script
itself: a tiny checkpoint and tokenizer written to a temp directory, then
`graphbench.py --split-read --allow-cpu --arrivals burst` run over it at a width that
splits. That checks the boot order, the padding and the CSV header on a real
process.

The hardware caveat hasn't changed: the card run with `--split-read --rates 1,2,4,8`
is still the first one to book, and it's now a single command.

## Post angle
Day 74 of building an LLM inference engine from scratch. My benchmark script now
boots a third server with flash-decoding style split attention and compares it with
the CUDA-graphed one. The gotcha: to test the split, rows have to cross a chunk, so
the script pads the prompt past the chunk using the server's own tokenizer, and it
counts only the prompt because a completion can stop at EOS early. vLLM and SGLang
ship split decode; I'm learning what it takes to measure one honestly. 2338 green.
