---
title: "Day 79: an always-red line hides the next red line"
parent: Daily log
nav_order: 79
---

# Day 79: an always-red line hides the next red line

Date: 2026-10-08 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday ended with the three-arm compiled run green except for one claim, "the arms
are comparable", which wants 20 frame gaps per arm and the toy run had 11. Every
process smoke since Day 75 sent `--requests 6 --max-tokens 8` and every one printed
that FAIL. Day 76 even wrote a test asserting it was there.

That line was a sample-size gate doing its job, and it was also a hiding place.
`check_arms_comparable` checks, in order, that the arms served the same clients, that
no request failed, that they delivered the same token count, and only then the sample
floor. All four print the same `FAIL  the arms are comparable`. If the token counts
had ever diverged on a smoke, every test would still have passed.

**`nanoserve.toysmoke`** holds two things every toy smoke now shares:
- **`SMOKE_LOAD`**, six requests of sixteen tokens. On the byte-level toy a frame is
  not a token: a random byte is often half a UTF-8 character and the server sends no
  empty frame, so eight tokens made under three frames a request. Sixteen gives 32
  gaps per arm, measured, and the same 32 every run because the decode is greedy and
  frames are cut from the text, not the clock. The burst smoke went 13 s to 16 s.
- **`WALL_CLOCK_FAILS`** and **`unexpected_failures(stdout)`**. The one claim a CPU
  smoke may still FAIL is "the tail did not get worse", and only for its own reason
  (the reason line holds "ITL p99 is"). The parser reads each FAIL with the line under
  it and returns every note that is not that, once per rate it failed at.

`tests/test_toy_smoke.py` has 10 pure tests: the load, a clean log, the allowed tail,
the tail refused as `MeasurementUnsound` (caught), the old floor FAIL (caught now), a
new reason under the old note (caught), a new FAIL next to the allowed one (caught),
and a sweep failing the same note at two rates (returned twice).

All four process smokes (burst, sweep, two-arm compiled, three-arm compiled) now send
`*SMOKE_LOAD` and assert `unexpected_failures(stdout) == []`. The burst smoke also
asserts every arm's `(n=...)` clears `DEFAULT_MIN_SAMPLES` and that "the arms are
comparable" printed ok. Day 76's sweep test flipped: the comparability note must now
be absent from `claims_failed`, and whatever is there must be in `WALL_CLOCK_FAILS`.

Run by hand, the three-arm compiled run at the new load printed 14 of 14 ok.

Suite **2426 green** (34 GPU-gated skips), ruff clean.

## Why it matters
**A gate nobody expects to pass is a gate nobody reads.** The point of the smokes is
that a person, or a test, can look at the claims and see what changed. With one line
always red, a reviewer learns to skip red, and a second red line or a second reason
gets skipped with it.

**Expected failures need a reason, not just a name.** Allowing "the tail did not get
worse" by note alone would have let a run with no frame gaps at all through, since
that refusal prints under the same note. The allowlist pins the note to the one
refusal that is noise on a CPU.

**The fix for a sample-size gate is more samples, not a lower floor.** `--min-samples`
already existed, and passing `--min-samples 10` would have turned the line green too.
It would also have made the smoke a run of a different gate than the card run uses.

## What I learned
1. **Count frames, not tokens, before sizing a load.** I expected 6 x 7 = 42 gaps at
   eight tokens and the log said 11. The byte-level tokenizer plus random weights
   means most tokens are partial characters that never become a frame.
2. **The tail gate on a CPU is a coin flip both ways.** Same flags, the graphed p99
   was 2.43x the eager one in one run and passed in the next. On a CPU it gets
   allowlisted, and on a card it stays a hard gate.
3. **Several checks under one note will hide each other.** The order inside
   `check_arms_comparable` meant the floor, which is checked last, was the one we saw,
   and the earlier checks would have printed the same line.

## Diagram
[smoke-floor.png](../diagrams/smoke-floor.png). Top left: the old smoke's FAIL and the
three checks that print under the same note. Top right: why eight tokens is 11 gaps,
and sixteen is 32. Bottom left: `nanoserve.toysmoke`. Bottom right: the one FAIL a CPU
smoke may still print, and the refusal under that note that is still caught.

## Tomorrow
Every toy smoke can now pass every claim, so a FAIL means something. The next gap is
the CSV: `claims_failed` carries notes, not reasons, so the CSV still cannot tell a
noisy tail from a broken one the way the log now can. Next: a `claims_unsound` column
(the notes refused as `MeasurementUnsound`), so a card sweep's notebook can filter
"this row cannot be used" from "this row says the capture lost".

The card run still comes first when hardware is booked.

## Post angle
Day 79 of building an LLM inference engine from scratch. Every smoke run of my
benchmark printed the same FAIL: 11 samples against a floor of 20. Expected, so
ignored. But four different checks print under that note, and any of them would have
hidden behind it. Now the smoke clears the floor and the one allowed FAIL is pinned to
its reason. 2426 green.
