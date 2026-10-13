---
title: "Day 83: one row shape, so the table reprints from the file"
parent: Daily log
nav_order: 83
---

# Day 83: one row shape, so the table reprints from the file

Date: 2026-10-12 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Day 82 finished typing `read_sweep`, and `print_table` still read the other shape of
the same row: the raw dict `main` built, with `claims_failed` the joined string it
split on `FAILED_SEP` by hand. So a row read back from a CSV could not be printed,
and an old sweep could not be looked at again without running it again.

**One shape, the one the file reads back as:**
- **`as_read(row)`** turns a row `main` just built into what `read_sweep` would return
  from the CSV. Only the two note columns change, to lists. Everything else already
  has the type it is written with. It returns a copy, because `main` writes the CSV
  from the row it built after the table prints.
- **`print_table` takes that shape and only that shape.** The claims cell is
  `claims_ok + len(claims_failed)`. A string in `claims_failed` raises a `TypeError`
  that says to pass the row through `as_read` first, because `len` on a string counts
  characters and would print a wrong total with no error.
- **`graphbench.py --table <csv>`** reprints a finished sweep's table from its file
  through `read_sweep`, with no weights, no device, and nothing booted. Every refusal
  is a sentence and a non-zero exit, not a traceback: a missing file, a file the
  parser refuses (a `--coverage` table, a pre-Day-76 sweep), and a file with a header
  and no rows. `reprint(path)` is the function behind it.

`tests/test_sweep_table.py` has 13 pure tests: `as_read` equals `read_sweep` of the
written row for two-arm and three-arm rows, it changes only the note columns and does
not modify the row it was given, the table counts a typed row and refuses a raw one, a
Day 76 row with `claims_unsound` as `None` still prints, the table printed from the
file matches the table printed from the run, and `--table` run as a process reprints
the same lines, needs no weights, and refuses all three bad files with no traceback.
Three older table tests now send their rows through `as_read`, which is how `main`
sends them.

The toy sweep smoke gets one more test: `--table` on its real three-arm CSV prints
the same lines the sweep printed.

Suite **2480 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**A card run costs hardware time, and its table should not.** The run prints the
table once, in a log that scrolls away. With `--table`, the CSV is enough to print it
again, on any machine, without a GPU.

**Two shapes for one row means two tables that can disagree.** If the live run and
the file each had their own path into the table, a column could print one way
during the run and another way when read back, and no test would see it. With
`as_read` there is one shape, and a test pins the shape: what `as_read` returns
equals what `read_sweep` returns.

**The quiet failure was the dangerous one.** The obvious change, `len(claims_failed)`,
works on a list and on a string, and on a string it is wrong. `"the tail did not get
worse"` is one failed claim and 26 characters. The table now refuses a string rather
than count it.

## What I learned
1. **When two producers feed one consumer, pick one shape and convert the other.**
   The file cannot hold a list, so the live row is the one that converts, and the
   conversion is pinned by a round-trip test instead of by hand.
2. **Duck typing can hide a bug.** `len` takes a string and a list, so the type was
   the only thing that could tell a correct count from a character count. Checking
   it at the top of the loop costs one line.
3. **A reprint test that compares lines catches more than one that checks fields.**
   The toy smoke compares `--table`'s stdout to the live table line by line, so a
   change in a float's format, a column, or the claims count shows up as a diff.

## Diagram
[sweep-table.png](../diagrams/sweep-table.png). Top left: the two shapes and the
character count. Top right: the live run and the old file meeting at the typed row.
Bottom left: what it refuses. Bottom right: `--table` on a card CSV.

## Tomorrow
The table counts claims and does not separate the unsound ones, and Day 80 said a row
with an unsound note is not a result. Next: mark those rows in the table, so a
reprinted card sweep shows which rows can be compared and which could not be
measured, and which claims to look at first.

The card run still comes first when hardware is booked.

## Post angle
Day 83 of building an LLM inference engine from scratch. My benchmark table read
`claims_failed` as a string, and the CSV reader gives it back as a list. The obvious
fix, `len(claims_failed)`, works on both, and on the string it counts characters: one
failed claim showed up as 26. Now there is one row shape, the table refuses the other
one, and `--table run.csv` reprints a GPU sweep with nothing booted. 2480 green.
