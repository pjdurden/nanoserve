---
title: "Day 66: the arena got a caller, and the caller had one place to stand"
parent: Daily log
nav_order: 66
---

# Day 66: the arena got a caller, and the caller had one place to stand

Date: 2026-09-22 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
`--split-read`, wired from the command line down to the cache, plus the boot-path
call that turns a planned split into reserved memory. Wiring it turned up a gate that
had been refusing every split server with CUDA graphs on.

**`engine.py`.** `Engine.build(split_read=...)` goes to the cache, and the cache's two
refusals (no `bucket_decode`, or both reads at once) come back unchanged. The
constructor's `check_capture_ready` call now passes `armed=False`. The reason is the
main finding of the day and is explained below.

**`captured.py` and `buckets.py`.** `check_capture_ready(cache, *, armed=True)` gains
a fourth precondition. If a split read holds no arena, it raises `CaptureUnsound`
naming `cache.allocate_split_workspace(device)` and `arm_split_read`.
`check_read_matches` gains the same `armed` keyword. It excuses exactly one clause,
"split read, no workspace", which is the only one of its seven that is about *when*
something happens rather than *what* is configured. The chunk-mismatch clause also
stops firing on a read with 0 chunks. Otherwise an unarmed read would get reported as
"0 against 1" and send someone looking for a disagreement that doesn't exist.

**`launch.py`.** Four pieces:
- `arm_split_read(engine, capture=None, *, device=None)` returns `None` for the other
  two reads. For a split read, it allocates the arena through the cache, on the
  weights' device, and refuses a second call.
- `check_arena_matches_capture(capture, workspace)` holds the plan's record and the
  cache's allocation to each other. Chunks, tile and width must be equal. Rows are
  only bounded (the plan can't have more rows than the arena).
- `build_app` now plans the capture with `split_read=engine.cache.read.split`, arms,
  and only then warms. The arming happens outside the capture branch.
- `boot_info` and `boot_lines` take the workspace. `/health` gets a top-level
  `split_workspace` section, and the boot log prints `SplitWorkspace.render` directly
  under the capture line. `check_boot_info` refuses a payload whose capture priced a
  split but names no arena, or whose arena disagrees with the plan.

**`serve.py`.** `--split-read` is mutually exclusive with `--streamed-read` at the
argparse level, so passing both is a usage error, not a traceback after the weights
have loaded. It implies `bucket_decode`. That is the one implication in the launcher
that comes from a read rather than from `--cuda-graphs`, and it exists because the
split can't run without a bucketed width, not because it's convenient.

`tests/test_split_boot.py` is 26 new tests, and one Day 65 test in
`tests/test_width_axis.py` changed its expected exception (explained below). Suite
**2180 green** (21 GPU-gated skips), ruff clean.

`bootbench_split.py`, host only, no weights
([day-66-bootbench.csv](data/day-66-bootbench.csv)):

    where arm_split_read sits on the boot path:
        position  verdict
           never  CaptureUnsound: this cache runs the split read and holds no arena yet
      after warm  CaptureUnsound: this cache runs the split read and holds no arena yet
           twice  BootUnsound: this read already holds an arena
      after plan  boots

    three servers, one flag apart, booted the way serve.py boots them:
      rectangle: 3 boot lines, arena 0 B,    split_workspace in /health: False
      streamed:  3 boot lines, arena 0 B,    split_workspace in /health: False
      split:     4 boot lines, arena 1536 B, split_workspace in /health: True
        CUDA graphs: 3 of 3 shapes, rows <= 4, context <= 1024 (capped by the
          streamed read, which has no width axis), 2-way split, 0.0 MiB of workspace
        split workspace: 4 rows x 8 heads x 2 splits of 512 keys over a
          1024-wide mapping, 0.00 MiB, allocated once

## Why it matters
**The gate I wrote yesterday refused the server I was building today, and it did it
at the earliest possible moment.** Day 54's `Engine.__init__` runs
`check_capture_ready` when the graphs are on, so a capture over an open shape set is
refused at construction. Day 65 added the "does the split read hold an arena" clause
to the gate that function calls. On the boot path, the engine is built before any
device memory exists, and the arena comes later. So as of last night,
`Engine.build(split_read=True, capture_decode=True)` raised a `BucketsUnsound` about
0 chunks against 1, every time, before the only call that could have fixed it. No
test caught it because no test had ever built a split engine with graphs on. Day 65's
tests built the cache, allocated the arena by hand, and called the gate afterwards,
which is the order a test takes and not the order a server does. It took a wiring day
to build one in boot order.

**The fix is a distinction, not a relaxation.** Of the seven clauses in
`check_read_matches`, six are wrong whenever they are true: a width axis under a tiled
read, a split set under an unsplit read, two chunk counts that disagree. Those are
configuration errors, and construction is the right time to refuse them. The seventh,
"a split read with no arena", is only wrong *too late*. It's legitimate from the
constructor until `arm_split_read`, and fatal after. `armed=False` excuses that one
clause and nothing else, and a test covers exactly that: a mis-assembled set is still
refused with `armed=False`.

**The arena has one correct position on the boot path, and a line of code is what
holds it there.** After `plan_capture`, because the plan's budget probe is where the
split's bytes get checked against the card. An arena reserved before the probe would
be counted twice: by the driver as used, and by the plan as needed. Before
`warm_engine`, because a warm batch is a decode step, and an unarmed split read
refuses a decode step. And outside the `if graphs` branch, because the arena belongs
to the read, not to the capture. A split server with graphs off still runs a split on
every decode step, and if it were only armed inside that branch, its first request
would be a refusal. The order table above tries each alternative, and every wrong one
is refused before the door opens.

**A split server reserves two numbers beyond the pool, and now they print together.**
The capture's shared arena is priced by `plan_capture` against a probe. The partials
arena is allocated by the cache from its own limits. Both are held for the life of
the process. They're computed at different moments by different callers, and until
today only the first ever appeared on the boot log. Now the second is the very next
line, and `/health` carries it as `split_workspace`, so someone adding up what the
card holds doesn't have to recompute `rows * heads * splits * (head_dim + 2) * 4`.

## What I learned
1. **`allocate_for(capture)` still has no caller on the boot path, and that's the
   right answer, not a leftover.** Day 65's "Tomorrow" said this day would finally
   give it one. Once I wrote the call, the problem was obvious: a capture plan
   describes what is *recorded*, and `--warm-rows` can trim that below the slot
   count. The arena has to cover what is *served*, and a shape the list skipped still
   runs, eagerly, against the same buffers. Sizing the arena off the plan would have
   given a `--warm-rows 4` server an arena four rows wide and a scheduler that admits
   256. So the cache sizes it, and `check_arena_matches_capture` checks the plan
   against the result instead of treating the plan as the source. That is also why
   the rows clause is an inequality while the other three are equalities.
2. **The refusal changed class, and it had to.** Day 65's
   `test_check_capture_ready_runs_the_split_gate_too` expected `BucketsUnsound` with
   "no workspace". It now gets `CaptureUnsound` naming `allocate_split_workspace`.
   `check_capture_preconditions` has a standing rule that the exception comes from the module where the fix is. An
   unarmed read isn't a bucket set disagreeing with a read, it's a boot path that
   stopped one call short, and the fix is in the boot path. The bucket gate's clause
   is still there for callers who assemble the pair by hand, and the updated test
   checks both.
3. **The toy model never split, and I only noticed because I asserted the count.**
   With 8 heads, 4 rows and a 32-token context, `plan_splits` returns 1 at every
   width up to 512, because the grid already fills `DEFAULT_PARTITION`'s minimum. A
   1-way split runs the whole two-pass machinery and proves nothing a split is for.
   The tests now use a 1024-wide table, which plans 2. The first draft of the token
   test would have passed on a split read that wasn't splitting anything.
4. **`arm_split_read` refuses a second call with `BootUnsound` before allocating,
   rather than letting `attach` refuse after.** `attach` already refuses a second
   arena, but by the time it runs, `allocate_partials` has already reserved the
   memory. A refused attach would leave a live, unreachable tensor on the card until
   the garbage collector gets to it. The check has to come before the allocation.
5. **The launcher implied `bucket_decode` from a read for the first time, and the
   Day 60 comment arguing against bundling reads still holds.** That comment was
   about not switching on a slower read as a side effect of another flag. This goes
   the other direction: the read requires the bucket set, the cache refuses without
   it, and an operator typing `--split-read` has no configuration in which they
   *don't* want it. Implying a requirement isn't bundling a preference.

## Diagram
[split-read-boot-path.png](../diagrams/split-read-boot-path.png). Top left: the five
calls in `build_app`, with the arm between the plan and the warm-up and the reason
for each side. Top right: the three callers of `check_capture_ready` and what each
one asks. Bottom left: the order table, three refusals and one boot. Bottom right:
the split server's boot lines, with the arena right under the capture.

## Tomorrow
The acceptance run gets its third arm. `test_the_same_engine_answers_the_same_bytes_on_either_read`
compares two live servers over a socket. Today's token test compares three engines
through `step`, which settles the attention and the sampler but not the detokenizer
or SSE framing. A third server needs `--split-read`, which exists as of today. It's
slow on this box because every arm is a Python tlsim loop, so Day 67 is either that
run with `PLANS` cut down enough to stay under a minute, or a decision written down
that it waits for the card.

Separately, the capture line still prints `0.0 MiB of workspace` for the toy, with
the split's partials folded into `pool_bytes` from Day 64's pricing, while the
partials now also appear on their own line. At serving size a reader summing the two lines counts the same
76.55 MB twice. Deciding which of the two lines owns the partials is small, and it
belongs to the same day.

And the caveat, unchanged: `graphbench.py --weights ./weights --device cuda --rates
1,2,4,8` on all three arms is the first run to book hardware for. It has been since
Day 58, and today's `--split-read` flag is the last piece of wiring that run was
waiting on.

## Post angle
Day 66 of building an LLM inference engine from scratch. Today the flash-decoding read
became a command-line flag, and wiring it found a bug in yesterday's gate. The split
read holds a partials arena it doesn't allocate, and yesterday I added a check that
refuses a split read without one. Reasonable. But the engine's constructor runs that
check when CUDA graphs are on, and on the boot path the engine is built before
anything is on a device. So every split server with graphs on was refused at
construction, before the one call that could have armed it. No test caught it,
because every test built the cache, allocated the arena by hand, and then called the
gate, which is the order a test takes and not the order a server does. The fix is a
distinction, not a relaxation. Six of the gate's seven clauses are wrong whenever
they're true: config errors, refused at construction. One, "no arena", is only wrong
too late. It's legitimate until the arm call and fatal after. So it's excused at
construction and required at warm-up. That leaves the arena exactly one place on the
boot path: after the capture plan, because its probe is where the arena's bytes met
the card, and before the warm-up, because a warm batch is a decode step. And outside
the graphs branch entirely, because the arena belongs to the read. vLLM and SGLang
both size these workspaces at startup, not per step, and wiring it myself showed me
the startup ordering is the actual design problem. 2180 green, same greedy tokens as
the rectangle through the real boot path, and still not one measured second.
