---
title: "Day 84: the table says which rows are results"
parent: Daily log
nav_order: 84
---

# Day 84: the table says which rows are results

Date: 2026-10-14 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Day 83 made the table print from the row `read_sweep` gives back, so `--table`
reprints a sweep from its CSV. That table still had one number for both kinds of FAIL:
`claims_ok` over the total. A row that measured four claims and lost them printed
`9/13`. A row whose arms served different work also printed `9/13`. Day 80 said the
second one is not a result, and the CSV knew which was which, but the table did not
show it.

**A mark on every row, and the reruns listed under the table:**
- **`row_mark(row)`** reads the two note columns and returns one of `MARKS`:
  `clean` (every claim held), `lost` (some failed, every failure measured: a result),
  `rerun` (at least one failure could not be measured: not a result), or `?` (a Day 76
  file, which did not record which failures were unmeasurable, on a row that has
  failures). A Day 76 row with no failures is `clean`, because no failure means no
  unsound one.
- **`print_table` has a `row` column just before `claims`.** The count stays the last
  cell, so it still reads where it always did.
- **Under the table**, each `rerun` row is listed again with the notes that made it a
  rerun, in table order. Only its unsound notes are listed, not the ones it lost. If
  any row is `?`, there is one more line saying why. A table of results prints nothing
  under it, so a clean sweep looks the same as before.
- **`row_mark` refuses a string in either note column**, with the same `TypeError` as
  Day 83. `"a; c"` is truthy, so it would read as a rerun and its notes would be
  listed letter by letter.

While I wrote the column alignment test, it found an older bug: the `replayed`
header was 9 characters wide and its cell 8, so every column after it was off by one.
The cell is 9 wide now.

`tests/test_sweep_marks.py` has 16 pure tests: the three marks from the note columns,
`?` and `clean` on Day 76 rows, the string refusal, the column's place, each line's
mark next to its count for two-arm and three-arm rows, the columns lining up, nothing
printed under a table of results, the reruns listed with their unsound notes only, the
`?` line, reruns printed before the `?` line, and `--table` as a process printing the
same marks and reruns from the file. Two older tests read the count from the row's own
line now, because the last line printed is the list under the table.

Suite **2496 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**A count cannot tell you what to do with a row.** `9/13` from a run that lost is a
finding to plot and explain. `9/13` from a run that could not measure is a run to do
again. Day 80 put that difference in the CSV. A person reading the table did not see
it, and the table is what they read first.

**The list under the table is the work queue.** A card sweep with a rerun row needs
the hardware again before anything in it is compared. Listing the unsound notes in
table order says what to fix before the next booking: too few gaps means more
requests, and arms that served different work means checking the load.

**A result-only sweep should look unchanged.** All the new output goes under the table
and only shows when something needs attention, so a clean card run reads the same as
it did yesterday.

## What I learned
1. **Name the action, not the cause.** `rerun` tells the reader what to do with the
   row. `unsound` would only repeat the column name and leave them to work out what it
   means.
2. **"Not recorded" has to stay a third answer even in a one-word cell.** A Day 76
   row with failures could be `lost` or `rerun`, and the table should not guess. One
   with no failures can be answered, so it is.
3. **An alignment test checks more than the new column.** I wrote it for the `row`
   column and it found the `replayed` header one character wider than its cell, which
   had been there since the column was added.

## Diagram
[sweep-marks.png](../diagrams/sweep-marks.png). Top left: two rows with the same
count and different meanings. Top right: how `row_mark` reads the two note columns.
Bottom left: the marked table and the rerun list under it. Bottom right: reading a
card sweep with it.

## Tomorrow
The table now says which rows are reruns, and the exit code of `--table` does not. A
script that checks a card sweep before a notebook plots it has to parse stdout to find
out. Next: a `--strict` flag on `--table` that exits non-zero when any row is `rerun`
or `?`, so a CI step or a shell `&&` can refuse to plot a sweep that is not all
results.

The card run still comes first when hardware is booked.

## Post angle
Day 84 of building an LLM inference engine from scratch. My benchmark table printed
`9/13` for a run that measured four claims and lost them, and `9/13` for a run whose
two arms served different work. Only the first is a result. Now every row is marked
clean, lost, rerun, or ?, and the reruns are listed under the table with the claims
they could not measure. 2496 green.
