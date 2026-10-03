---
title: "Day 73: the split arm reads its own boot grades and holds them to what served"
parent: Daily log
nav_order: 73
---

# Day 73: the split arm reads its own boot grades and holds them to what served

Date: 2026-10-02 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was a report. Day 71's probe and Day 72's timing both land in
`/health` as `split_probe` and `split_timing`, and the acceptance harness read
neither. `run_arm` lifted the arena off the payload (Day 68) and the read counters
(Day 60) and left behind the two sections that say whether the read is right on this
card and what it bought. Now the split arm carries them, prints them under its row,
and is held to them.

**`graphbench.py`.** Three additions:
- `SplitBoot`, the record: the probe and the timing as published, plain dicts, with
  `graded`, `backend`, `worst_error`, `predicted` and `measured` read off them. A
  `render()` that prints one line ("graded on tlsim: read 3.0e-07 (bound 1e-04);
  predicted 4.00x, measured 0.52x (...), did not pay on the 4-row timed case") and a
  `row()` with five CSV columns that has the same keys whether anything was graded or
  not.
- `split_boot_from_health(payload)`, the lift. It copies, and returns an empty record
  for the other two reads, the same way `workspace_from_health` does.
- `check_arm_split_graded(arm)`, the gate. A split arm has to carry a probe inside its
  bound and a timing on the probe's backend, and that backend has to be the one the
  read counters say served the crowd.

`ArmReport` gains `split_boot`, `run_arm` fills it from the first reading, and
`ArmReport.render()` grows a second line when the arm was graded. Streamed and
rectangle arms still print one line.

**`tests/test_split_report.py`.** 22 new tests. They cover the lift (including the
real `SplitProbe`/`SplitTiming` dataclasses through `as_dict` and JSON, and a real
split boot's `boot_info` on the host), the render, the columns, and every refusal.
The Day 68 three-arm socket test in `tests/test_reads.py` now also runs the new gate
on its split arm and checks that the other two arms graded nothing. Suite **2310
green** (34 GPU-gated skips), ruff clean.

## Why it matters
**The boot record never sees a decode step.** `build_app` holds the probe against
the reference, and `check_boot_info` holds the timing against the probe. Both look
at one record written before the door opened. The read counters taken around a real
crowd name the backend that actually answered, and the harness is the only party
holding both. A grade on tlsim over a crowd served on triton graded a kernel that
answered nobody, and the kernel that did answer was never graded. Today that's a
named refusal rather than a green run.

**The number lands next to the throughput it bought.** The first card run of the
three arms will put "graded on triton, predicted 4x, measured Nx" on the line under
the split arm's ITL. Until now those lived in two places: the latency in the bench
output and the timing in a JSON blob nobody read.

**The verdict is printed, not gated.** Same reason as Day 72: the timed case is
four rows and a serving batch is not. The gate checks that the grade is about the
read that served. It never checks whether the split paid.

## What I learned
1. **An empty backend isn't a mismatch.** A window with no decode reads has
   `backend == ""`, and comparing it to the probe's "tlsim" would fail every idle
   arm with a message about two processes. So an empty window is checked first and
   refused as `MeasurementUnsound`: the run's fault, not the server's, which is what
   `check_arm_split` already calls it.
2. **Keep the CSV keys fixed.** `csv.DictWriter` takes its header from the first
   row. If an empty `row()` returned `{}`, a sweep that writes a split arm after a
   streamed one would crash on the second row, or quietly drop the columns the other
   way round. Same keys every time, `None` where nothing was graded.
3. **Test the fixture against the real producer.** The test payload is hand-written,
   and a hand-written payload drifts. One test builds the real `SplitProbe` and
   `SplitTiming`, sends them through `as_dict` and JSON, and asserts they come out
   equal to the fixture. Another lifts a real host split boot's `boot_info`. If
   `as_dict` renames a key, the unit tests fail instead of passing against a shape
   the server no longer sends.
4. **Under the row, not in it.** The boot grades are fixed for the life of the
   process and a row is one offered load, so a sweep prints the same second line
   under every row. That repetition is the honest picture: one boot, many loads.

## Diagram
[split-arm-report.png](../diagrams/split-arm-report.png). Top left: one `/health`
reading, three sections, into `ArmReport.split_boot` and the gate. Top right: what
the split arm's row prints now. Bottom left: the gate's refusals in order. Bottom
right: which check can see which mismatch, and why only the harness sees the last
one.

## Tomorrow
`graphbench.py` the script still runs two arms, graphs and eager, and neither one
splits. The three-arm run only exists inside `tests/test_reads.py`. The next step is
a `--split-read` arm in the script: a third server with `bucket_decode` on, prompts
past the merge floor, `check_arm_split` and `check_arm_split_graded` in its claims
list, and the `SplitBoot.row()` columns in its table and CSV.

The hardware caveat hasn't changed: `graphbench.py --weights ./weights --device cuda
--rates 1,2,4,8` with prompts past 514 tokens is still the first run to book.

## Post angle
Day 73 of building an LLM inference engine from scratch. My split server grades its
attention kernel at boot and times it. Today the benchmark harness reads both off
/health and prints them under the split arm's latency row. It also checks that the
backend graded at boot is the backend that served the crowd, because the boot record
never sees a decode step and only the harness holds both. vLLM and SGLang ship
flash-decoding; I'm learning to make every number about the kernel that answered.
2310 green.
