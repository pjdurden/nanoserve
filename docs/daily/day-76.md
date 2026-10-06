---
title: "Day 76: the sweep runs as a process, and every row carries its own verdict"
parent: Daily log
nav_order: 76
---

# Day 76: the sweep runs as a process, and every row carries its own verdict

Date: 2026-10-05 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was a two-rate sweep smoke: `--rates` with `--arrivals fixed`,
so the multi-row table, `print_table`'s split columns and a two-row CSV run in a real
process. That's the shape of the card run, and the burst smoke never reached that
half of `main`.

**The sweep smoke.** `tests/test_toy_script.py` gets a second module fixture that runs
`graphbench.py --split-read --arrivals fixed --rates 200,400` over the same toy
checkpoint. The rates are high on purpose: six fixed arrivals 2.5 to 5 ms apart land
inside one step, so the split arm still batches. Six tests check that every rate boots
split, graphs, eager again in that order, that both rates pad to the same prompt, that
every split claim and "same answers" print `ok` once per rate, that the table prints
with its split columns, and that the CSV has one row per rate under the burst CSV's
exact header.

**It passed the first time.** There was no bug in the sweep. What the run did show was
in its output: at both rates the log printed `FAIL the arms are comparable` and
`FAIL the tail did not get worse`, and the table and the CSV under it printed the
ratios with no sign of either failure.

**So the row now carries its verdict.** In `graphbench.py`:
- `claims(delta, args)` is the list `main` used to build inline: the eight two-arm
  gates, then `split_claims`.
- `grade(pairs)` runs each one and keeps `(note, exc or None)`. It catches
  `AcceptanceFailure` and `MeasurementUnsound` and nothing else, so a bug inside a
  check still crashes the run instead of printing as a FAIL.
- `verdict_columns(graded)` gives `claims_ok` (a count) and `claims_failed` (the failed
  notes joined with `"; "`). They go at the end of the row, after `replay_share`.
- `print_table` gets a `claims` column that reads held over total: `11/13`.
- `--sanity` now prints every claim before it exits, not just up to the first FAIL.

`tests/test_sweep_verdict.py` covers the four functions in-process (12 tests) and three
more process tests hold the sweep's CSV and table to what its log said. Suite
**2382 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**The CSV travels without the log.** The card run's output that gets plotted is the
CSV. Before today, a `1 rps` row whose p99 came from eleven samples, refused by its
own gate, read exactly like a clean one. Now `claims_failed` is non-empty, and a
notebook can filter on it.

**The table is what a person reads last.** The claims print above the table, one block
per rate. With four rates that's about sixty lines before the table, and the table is
what goes in the post. `11/13` at the end of a row says "go read the log" where the
number is.

**Grading once keeps the log and the row honest with each other.** The log prints from
the graded list and the row is updated from the same list. A second loop over the
gates for the CSV could disagree with the first on a gate that reads the clock.

## What I learned
1. **A smoke that passes still has something to say.** I wrote the sweep tests
   expecting a crash like yesterday's. They went green, and the useful finding was in
   the output, not the exit code: the table printed numbers its own gates had refused.
2. **Catch the refusals, not everything.** `grade` could have caught `Exception` and
   been more forgiving. It would also have turned a `KeyError` in a gate into
   `FAIL the tail did not get worse`, which is a wrong claim about the server and not
   a report of a bug in the harness. There's a test for exactly that.
3. **New columns go at the end.** Day 74's split columns came after the two-arm ones so
   an older CSV stays a prefix. Same rule here: `claims_ok` and `claims_failed` come
   after `replay_share`, and a test pins them as the last two.
4. **Know what a fixture fails before you count on it.** The in-process `_delta`
   carries empty `CaptureStats`, so its graphed arm has no capture and fails the three
   capture claims. My first count said 12 of 13 and it was 10. The tests name those
   three rather than hard-coding a number nobody can explain.

## Diagram
[graphbench-sweep-verdict.png](../diagrams/graphbench-sweep-verdict.png). Top left:
before, the log's FAIL lines and a CSV row that doesn't mention them. Top right:
`claims` to `grade` to `verdict_columns`. Bottom left: what the sweep smoke checks.
Bottom right: the table's `claims` column and the CSV's two new columns.

## Tomorrow
Every part of the script the card run uses has now run in a process on CPU, except
the compile path: both smokes pass `--no-compile`. Next: a toy smoke with
`--compile dynamic` on CPU (one rate, burst), so `CompiledDecode` actually builds
behind a booted server and the capture records over a compiled forward. If inductor
is too slow for the suite on this box, it gets its own marker and runs on demand.

The hardware caveat hasn't changed: the card run with `--split-read --rates 1,2,4,8`
is still the first one to book, and now its CSV will say which rows to trust.

## Post angle
Day 76 of building an LLM inference engine from scratch. I ran the benchmark sweep as a
real process expecting another crash. It passed. The find was in the output: two gates
printed FAIL at every rate, and the table and CSV under them showed the ratios as if
nothing happened. The CSV is what gets plotted, so now every row carries `claims_ok`
and `claims_failed`, and the table says 11/13. vLLM and SGLang are the production
systems I'm learning from; this one is about a benchmark that can't hide its own
refusals. 2382 green.
