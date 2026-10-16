---
title: "Day 85: the table's exit code says whether it is all results"
parent: Daily log
nav_order: 85
---

# Day 85: the table's exit code says whether it is all results

Date: 2026-10-15 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Day 84 marked every row `clean`, `lost`, `rerun` or `?`. The marks were only in the
text, so `graphbench.py --table s.csv` exited 0 for a sweep full of reruns. A script
that guards a notebook had to parse stdout to find that out.

**`--strict` on `--table`:**
- **The table prints first, in full.** Then, if any row is `rerun` or `?`, the process
  exits 1 with one sentence on stderr naming those rows' loads. A sweep of `clean` and
  `lost` rows exits 0 with nothing on stderr.
- **`lost` does not trip it.** A capture that lost, with every failure measured, is a
  result to plot and explain. Only rows that are not comparisons fail the check.
- **`?` trips it.** A Day 76 file with failures cannot say which kind they were, and a
  guard should not guess in the permissive direction.
- **`reprint` returns the marks** in table order, so a caller in Python does not
  re-derive them.
- **`--strict` without `--table` is refused** by the parser, with a sentence naming
  `--table`. A live run has no file to check.

    graphbench.py --table docs/daily/data/day-74-graphbench.csv --strict && plot

`tests/test_table_strict.py` has 6 pure tests: a sweep of results exits 0 with a quiet
stderr, a rerun exits 1 after the table printed and names only its own load, a `?` row
exits 1, no `--strict` still exits 0 on a rerun, `--strict` alone is refused, and
`reprint` returns the marks.

Suite **2502 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**A check that needs parsing is a check nobody writes.** The mark exists so a reader
knows what to do with a row. A shell `&&` is a reader too, and it only reads the exit
code.

**Print first, then fail.** The person who sees the red step still wants to know which
rows to rerun, so the failure carries the full table and a one-line reason.

**The guard follows the table.** It calls `row_mark`, the same function the column
uses, so the table and the exit code cannot disagree about what a result is.

## What I learned
1. **A test fixture has to write the real file.** My first Day 76 test row had
   `claims_unsound` set to empty in the CSV, which reads back as "no unsound notes",
   a `lost` row. A real Day 76 file has no such column. The test failed for the right
   reason and the fixture was the bug.
2. **Strictness is a policy on the mark, not a new mark.** `lost` stays a result under
   `--strict`; the flag chooses which marks fail, and adds no fifth.
3. **Refuse a flag that cannot do anything.** `--strict` with a live run would be
   silently ignored, and a silent no-op in a guard is the worst kind.

## Diagram
[table-strict.png](../diagrams/table-strict.png). Left: the four marks and which of
them `--strict` fails. Right: the table, the stderr sentence and the exit codes.

## Tomorrow
The guard reads one file. A card booking produces several sweeps (one per day or per
card), and checking them one command at a time loses the rows they have in common.
Next: let `--table` take more than one CSV, print each sweep's table under its path,
and make `--strict` fail if any of them has a rerun.

The card run still comes first when hardware is booked.

## Post angle
Day 85 of building an LLM inference engine from scratch. My benchmark table marked
which rows were real results and which were reruns, and still exited 0 on a sweep full
of reruns. Now `--table --strict` prints the table and exits 1 if any row is a rerun,
so `&& plot` refuses a sweep that is not all results. 2502 green.
