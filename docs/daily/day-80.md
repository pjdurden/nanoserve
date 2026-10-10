---
title: "Day 80: a row that lost is not a row that could not measure"
parent: Daily log
nav_order: 80
---

# Day 80: a row that lost is not a row that could not measure

Date: 2026-10-09 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Day 79 got every toy smoke passing every claim but one, and pinned that one to its
reason by reading the line under each FAIL. That made the log able to tell a noisy
tail from a broken one. The CSV could not do the same. `claims_failed` carries notes,
and "the tail did not get worse" is the same note whether the graphed p99 came out
slower (a result) or there were no frame gaps to compare (no result at all).

The gates already say which is which, by the type they raise:
- **`AcceptanceFailure`**: the run measured what it set out to and the answer was no.
  The numbers on the row are good, and they say the capture lost.
- **`MeasurementUnsound`**: the run could not measure it. Too few gaps, a dropped
  request, arms that served different work. The numbers on the row are not a result.

**`verdict_columns`** in `graphbench.py` now returns a third column,
**`claims_unsound`**: the notes refused as `MeasurementUnsound`, in print order,
joined with the same `FAILED_SEP`. It is a subset of `claims_failed`, so a notebook
drops the rows where it is non-empty, then reads `claims_failed` on what is left. It
goes last, so Day 76's header is still a prefix of today's.

`tests/test_sweep_unsound.py` has 9 pure tests: the column order, a clean row, a row
that lost (failed, not unsound), a row that could not measure (both), order and
separator, the subset property over the real claims, `_delta` at the default floor
(three capture claims lose, nothing unsound), the same `_delta` under a floor of 1000
(the comparability gate is the one unsound note), and the table still counting
held over total without counting an unsound note twice.

The toy sweep smoke's header test now expects the three verdict columns at the end,
and a new smoke test asserts every sweep row's `claims_unsound` is empty: the one FAIL
a toy smoke may print is an `AcceptanceFailure`, so every row it writes is usable.

Suite **2436 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**The CSV is the part that leaves the box without the log.** Day 76 put the verdict on
the row because a notebook reads the CSV, not stdout. A verdict that only says *which*
gate refused, and not *how*, sends a plot two kinds of row under one label.

**"Lost" and "could not measure" call for opposite actions.** A row that lost is a
finding: keep it, plot it, explain it. A row that could not measure is a rerun: drop
it, fix the load, run again. Mixing them either hides a regression behind noise or
plots noise as a regression.

**The distinction was already in the code, one layer down.** Day 42 split the two
refusals into two exception types for exactly this reason. The row was the one place
that flattened them back into a string.

## What I learned
1. **A type is a column waiting to happen.** `grade` already kept the exception, not
   just a bool. Keeping the richer value on Day 76 made today one `isinstance`.
2. **A subset column beats a per-note kind column.** A "kind" column parallel to
   `claims_failed` would need a reader to zip two split strings. A subset reads as one
   filter, and `claims_unsound == ""` is the whole question most readers ask.
3. **`_delta`'s capture claims are AcceptanceFailures, not Unsound.** Its graphed arm
   has no capture, and that is a measured fact about the arm. I had half expected
   them to be unsound, and the test that checks it is now the record of why not.

## Diagram
[csv-unsound.png](../diagrams/csv-unsound.png). Top left: two illustrative rows with
the same note and the two different reasons the log printed under them. Top right:
the two refusals and what each means for the row. Bottom left: `verdict_columns` with
the new column. Bottom right: how a card sweep's notebook reads the three columns.

## Tomorrow
The verdict is complete on the row now: held, failed, and unusable. The next gap is on
the reader's side: there is still no code that loads one of these CSVs back. Next: a
small `read_sweep(path)` that parses the three verdict columns into lists and refuses
a CSV whose header is not a known prefix, so a card sweep's notebook starts from the
same parser the tests use.

The card run still comes first when hardware is booked.

## Post angle
Day 80 of building an LLM inference engine from scratch. My benchmark CSV said "the
tail did not get worse: FAIL" on two rows. One was a real slower p99. The other had
no samples to compare at all. Same cell. Now a third column names the claims the run
could not measure, so a notebook drops those rows before it plots anything. 2436 green.
