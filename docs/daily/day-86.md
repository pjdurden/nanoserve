---
title: "Day 86: one command reads a booking's worth of sweeps"
parent: Daily log
nav_order: 86
---

# Day 86: one command reads a booking's worth of sweeps

Date: 2026-10-16 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Day 85 gave `--table` an exit code: print the sweep's table, then fail if any row is a
`rerun` or `?`. It read one file. A card booking leaves one CSV per day or per card,
so guarding a notebook over five of them meant five commands and five exit codes, and
the first failure hid the other four tables.

**`--table` is `nargs="+"`:**
- **Each sweep's table prints under its path**, in the order the paths were given,
  with a blank line between them.
- **One path prints no heading at all.** Day 83's output for a single file is
  unchanged, down to the line. The heading is there to tell the tables apart, and one
  table has nothing to be told apart from.
- **Every file is parsed before any table prints.** A typo in the third of five paths
  prints no tables, just the sentence. Otherwise the real message lands under two
  tables that look like a partial answer.
- **`--strict` is one verdict over all of them.** Every table prints first, then the
  exit is 1 if any row anywhere is a `rerun` or `?`, and the sentence names each
  offending path with its own loads. `lost` is still a result, in one file or in five.
- **The same path twice is refused.** It would print twice under one heading and
  `--strict` would name it twice.

      graphbench.py --table day-74.csv day-77.csv day-79.csv --strict && plot

`reprint_sweeps` returns the marks keyed by path, in the order given; `reprint` stays
as the one-file call that returns the single list, because `marks[path]` is a layer a
reader with one file should not have to unwrap.

`tests/test_table_sweeps.py` has 10 pure tests: one path prints no heading, two print
under their paths in the order given, `--strict` passes a booking of results, fails on
the second sweep after both tables, names each bad path with its own loads, a bad path
is refused before anything prints, the same path twice is refused, and both functions
return their marks.

Suite **2512 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**A booking is the unit, not a file.** The sweeps from one card session are read
together or not at all: the rows they have in common are the comparison. A flag that
takes one path makes the shell do the joining, and the shell does it badly.

**Fail once, after showing everything.** Day 85's rule was print then fail for one
table. Over several it matters more: the reason to look at all five is that the
rerun in the third one might also be in the fourth.

**The heading appears exactly where it is needed.** One file, no heading, byte for
byte what Day 83 printed. The Day 83 and Day 84 tests that compare stdout line by line
did not change, which is how I know the single-file path did not move.

## What I learned
1. **Two of my ten tests passed before I wrote a line of code, for the wrong reason.**
   `--table a.csv b.csv` with the old single-value flag made argparse exit non-zero on
   "unrecognized arguments", and my refusal tests only asserted non-zero with empty
   stdout. They were green at the red step. I kept them (they assert the right thing
   now) but the lesson is that a test asserting a failure has to assert *which*
   failure, and mine only got specific enough when I checked the path was in stderr.
2. **Read-all-then-print is a different rule from print-then-fail, and both are right.**
   A parse refusal is a usage error, so nothing should have printed. A `--strict`
   failure is a verdict on output the reader wants, so all of it prints. Same command,
   two orders, decided by whether the output is an answer.
3. **A default that is "one" should stay the same shape as before.** Making
   `reprint` a one-line call into `reprint_sweeps` kept Day 85's API intact and meant
   the single-file tests were the regression suite for the new code path.

## Diagram
[table-sweeps.png](../diagrams/table-sweeps.png). Left: the `&&` chain this replaces,
and the two rules that follow from `nargs="+"`. Right: two tables under their paths,
the stderr sentence naming the path, and the exit code.

## Tomorrow
The tables are side by side but nothing compares them. A booking's five sweeps share
loads, and the question a reader actually has is whether `4 rps` moved between day 74
and day 79. Next: with more than one sweep, print a last block that joins the rows by
`load` and shows each sweep's `itl_p50_speedup` in its own column, with a gap where a
sweep did not run that load.

The card run still comes first when hardware is booked.

## Post angle
Day 86 of building an LLM inference engine from scratch. A card booking leaves one
benchmark CSV per day, and my `--table` flag read exactly one of them, so guarding a
notebook over five sweeps was five commands whose first failure hid the other four
tables. Now `--table` takes all of them, prints each table under its path, and
`--strict` is one exit code over the lot. 2512 green.
