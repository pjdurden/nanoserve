---
title: "Day 82: the sweep's numbers come back as numbers"
parent: Daily log
nav_order: 82
---

# Day 82: the sweep's numbers come back as numbers

Date: 2026-10-11 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Day 81 gave a sweep CSV one parser, `read_sweep`, and typed the verdict on the way
back. Everything else in the row still came back as the string the file held, so the
first cell of a card notebook would have been a cleaning step: `float` on every `_ms`
column, `int` on the counts, and a guess about what an empty split cell means.

**`read_sweep` now knows the columns `ArmDelta.row()` writes:**
- **Measurements are floats.** A suffix rule, **`FLOAT_SUFFIXES`** (`_ms`,
  `_speedup`, `_tok_s`), so a new latency column is typed the day it is added, plus
  **`FLOAT_COLUMNS`** for the four that do not follow it: `replay_share` and the split
  boot's `split_read_error`, `split_predicted`, `split_measured`.
- **Counts are ints.** **`INT_COLUMNS`**: `itl_samples`, `graphs_held`,
  `recorded_while_serving`. `12.5` in one is refused, not rounded.
- **`split_pays` is a bool,** read from the two words `csv` writes for one, and any
  third word is refused.
- **Only the split boot's cells may be empty.** **`NULLABLE_COLUMNS`** is
  `tuple(SplitBoot().row())`, the key set of an ungraded boot, so it cannot drift from
  the writer. An empty cell there is `None`. An empty cell anywhere else is a
  measurement that is missing, and it is a `ValueError` naming the column and the row.
- **Anything else is left alone.** `load`, and any column this code did not write.

The conversion lives in one helper, `_typed(name, cell, where)`, and the verdict
columns keep Day 81's handling.

`tests/test_read_sweep_numbers.py` has 14 pure tests: the nullable set is the split
boot's own keys, a written two-arm and three-arm row read back equal in value and in
type for every key, measurements are floats and counts ints, an ungraded boot gives
five `None`s, a graded untimed boot keeps its grade and nulls its clock, `split_pays`
reads `True`, `False` and `""` correctly and refuses `"yes"`, and a hole, a word, and
a fractional count are each refused by column name. One Day 81 test changed its
expectation from `"1.5"` to `1.5`, which is the point of the day.

The toy sweep smoke gets one more test: its real three-arm CSV reads back with float
latencies and ratios, an int sample count, and no split boot cell left as `""`.

Suite **2466 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**`bool("False")` is `True`.** The split boot's `split_pays` is the one column on the
row that says whether the split read earned its keep, and the obvious way to read it
back gets it wrong on exactly the rows where the answer is no. Typing it in the parser
means a notebook cannot plot "did not pay" as "paid".

**An empty cell has two meanings, and only the writer knows which.** In the split boot
columns it means "this boot graded nothing", which is a fact about the arm. In
`graphs_itl_p50_ms` it means the file is broken. A notebook guessing would treat them
the same. The parser does not guess, because the set of columns allowed to be empty is
derived from the code that writes them empty.

**Errors belong where the file is read.** `float("")` raised inside a plotting call is
a stack trace about matplotlib. Raised in `read_sweep`, it is a sentence naming the
file, the row, and the column.

## What I learned
1. **Derive the schema from the writer when you can.** `NULLABLE_COLUMNS` is not a
   list I typed, it is `SplitBoot().row()`'s keys. The day the boot grows a column,
   the parser already knows it may be empty.
2. **A suffix rule is a convention made load-bearing.** Every latency column already
   ended in `_ms`. Leaning on that means the convention now has a test behind it, and
   a column that breaks it comes back as text where someone will notice.
3. **Round-trip equality must check the type too.** `1 == 1.0` in Python, so a test
   that only compares values would pass with `graphs_held` read back as a float. The
   round-trip test asserts `type(row[k]) is type(written[k])` for every key.

## Diagram
[read-sweep-numbers.png](../diagrams/read-sweep-numbers.png). Top left: the string
cells and the three ways reading them by hand goes wrong. Top right: the column
families and `NULLABLE_COLUMNS`. Bottom left: the four cells it refuses. Bottom right:
a card notebook that goes from file to plot with no cleaning cell.

## Tomorrow
The parser now gives back typed rows, and `print_table` still reads the raw row the
script built, splitting `claims_failed` on the separator by hand. Next: let
`print_table` take what `read_sweep` returns, and add a `--table <csv>` mode to
`graphbench.py` that reprints an old sweep's table from its file, so a card run can be
looked at again without being run again.

The card run still comes first when hardware is booked.

## Post angle
Day 82 of building an LLM inference engine from scratch. My benchmark CSV came back as
strings. An ungraded split arm writes None, csv writes that as "", and then
`bool("False")` is True: the row that says a kernel did not pay plots as paid. Now the
parser types every column, and only the columns the writer can leave empty may be
None. 2466 green.
