---
title: "Day 50: the addressing moved out of the forward, and the guard became a shape"
parent: Daily log
nav_order: 50
---

# Day 50: the addressing moved out of the forward, and the guard became a shape

Date: 2026-09-04 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.plan`, the module that decides where a decode step's tokens go before the
forward is called. `DecodePlan` is a frozen dataclass of four tensors and two host
ints: `write_slots` `[rows]` (the flat pool slot this step's K/V goes to),
`slot_mapping` `[rows, max_ctx]` (the read rectangle), `context_lens` `[rows]`, and
`positions` `[rows, 1]` (each new token's absolute position, read *before* the
tables grow, because it is the count of tokens that preceded it). `min_ctx` and
`max_ctx` are on it for the host only, and the whole day is about why nothing inside
the forward may read them. `batch_size`, `width`, `cells`, `real_cells`,
`padding_cells`, `padding_share`, `mapping_bytes` and `render` are what it says about
itself. `plan_decode(cache, rows, device)` builds one; `BatchedPagedKVCache.plan_decode`
is where the mutation lives, because growing the tables is the cache's business.

The wiring is five small changes. `BatchedPagedKVCache.write` takes
a `plan` and, given one, is a single `index_put` over the batch instead of a Python
loop over rows: no table grows, no slot is computed, and the layer-0 handshake is not
consulted, because the handshake existed to tell the other layers where layer 0 put
things and the plan tells everybody at once. `slot_mapping(plan=...)` hands back the
plan's own tensors, identity and not a copy. `paged_attention(plan=...)` calls the
read with `validated=True`. `BatchedCacheRows` carries the plan, so `layers.py` and
`model.py` needed no change at all. `Engine._decode` builds the plan in `build_inputs`
where it used to build `positions` off `tables[row].num_tokens`, and hands the forward
`plan.positions` and `cache.view(rows, plan=plan)`.

`paged_attention_batched_reference` grows a `validated` flag, and Day 49's
`decode_shape` learns to read a plan's width instead of predicting it. The
arithmetic is `mapping_cells`, `mapping_bytes`, `rebuild_cells`, `appended_cells` and
`rebuild_ratio`; the gates are `check_plan_addressing`, `check_plan_rows`,
`check_plan_current` and `check_rebuild_bounded`. `tests/test_plan.py` is 56 tests.
Suite **1313 green** (5 GPU-gated skips), ruff clean.

`compilebench.py` re-run unchanged, Llama-3.2-1B, cpu fp32, 4 rows, 11 decode
steps ([day-50-compilebench.csv](data/day-50-compilebench.csv)):

    mode      shapes  builds  reuse  fell back    compile        step  speedup
    off           11       0   100%         no      0.06s    861.81ms    1.00x
    dynamic       11       1    91%         no     44.01s    863.34ms    1.00x
    static        11       8    27%        yes          -  16357.90ms    0.05x

    Day 49, same script, 9 decode steps:
    dynamic        9       8    11%         no          -  28982.62ms    0.02x

## Why it matters
**The guard changed category, and that is the entire result.** Day 49 ended with one
captured graph, zero breaks, and a decode that was forty-six times slower, because
what dynamo kept invalidating was not a shape:

    kwargs['cache'].cache.tables[0].num_tokens == 13
      # [table.slot(p) for p in range(start, table.num_tokens)], cache.py:729

A tracer cannot make an attribute of an arbitrary Python object symbolic. It bakes
the value in and guards on it, and that value grows by one every decode step for the
whole run, so `dynamic=True` had nothing to offer. With the addressing handed in as
tensors, the reason dynamo now prints is:

    tensor 'kwargs['cache'].plan.slot_mapping' size mismatch at index 1.
      expected 4, actual 5

Same failure, different kind. A size is exactly the thing symbolic shapes exist for,
so the build generalises it and the graph is then kept: 1 build over 11 calls on the
real Llama-3.2-1B where Day 49 had 8 over 9, and no fallback where Day 49 walked past
`cache_size_limit` in eight steps. `check_graph_reused`, written on Day 49 as a gate
that failed, now passes.

**One build over eleven steps, and the regression is gone rather than reversed.**
861.81ms a step eager against 863.34ms compiled is 0.998x: inductor's CPU backend
found nothing in this model that it could not already do, and 44 seconds of compile
bought none of it back, so `breakeven_steps` is still infinite and the honest
recommendation for this box is still "off". What changed is the category of the
result. Day 49 was a 46x regression caused by the compiler fighting the engine; this
is a compiler that ran, kept its graph, and did not help. Those need different next
moves, and only one of them is a bug.

**On the tiny two-layer model the same run sometimes builds twice, and I could not
reduce it to one variable.** `dynamic=True` marks the dimensions of the compiled
function's *arguments*, and the slot mapping is not an argument: it arrives as
`kwargs['cache'].plan.slot_mapping`, two attribute hops off a non-tensor. Whether it
gets a symbol on the first build is therefore left to automatic dynamic, which
specialises, sees a second size, and generalises on the rebuild. I measured 1 build
on some batches and 2 on others over the same code, chased it through row count,
prompt raggedness and a suspected collision with `head_dim`, and falsified all three.
So: sometimes the first build is wasted and the second is the one that lasts, and
either way the run no longer goes near the recompile limit. Writing down a mechanism
I had disproved would have been the easier paragraph.

**Handing the bounds in would have been the same bug in a costume.** Day 49's fix was
to stop calling `int(context_lens.min())` in the kernel and pass the two numbers
down from the cache, where they were already Python ints. The obvious next move was
for the plan to pass `context_bounds=(plan.min_ctx, plan.max_ctx)`, and it is exactly
wrong: those are two Python ints that change every step, so a guard on their values
rebuilds every step whether they were read inside the region or handed into it. That
is why the kernel grew `validated=True` instead. A bool that is True for the whole
run is a guard that holds for the whole run, and the plan does the checking on the
host when it is built. The generalisation, and I nearly shipped the wrong version of
it: this is not about *where* a number is computed, it is about whether the traced
region ever observes it.

**The forward became a function again.** A forward over a KV cache normally is not
one. The paged read writes this step's K/V into the row before it attends, so calling
it twice attends over a history one token longer the second time; Day 49's first
equality test failed by whole units for exactly that reason and I narrowed the test.
With the addressing fixed before the call, replaying the same plan writes the same
slots and reads the same rectangle, so `f(x) == f(x)`. That is not a side benefit, it
is the precondition for everything downstream: comparing compiled against eager,
capturing a CUDA graph and replaying it, retrying a step. The test that a replayed
plan returns identical logits, and its control showing that an unplanned forward does
not, are the two I would keep if I could keep two.

**And the price is quadratic.** The read rectangle is `[rows, max_ctx]` int64 rebuilt
from scratch on the host every step, and `max_ctx` grows by one per step, so step i
writes `rows * (start + i)` cells. Four rows from a 16-token prompt:

    steps    cells rebuilt   cells appended    ratio    host bytes
        8              656               32     20x         5.2 KB
       64           12,416              256     49x        99.3 KB
      128           41,216              512     80x        329 KB
      512          558,080            2,048    272x        4.5 MB

Half a million int64 written to say where two thousand tokens are. `rebuild_cells`
against `appended_cells` is that ratio and `check_rebuild_bounded` is the gate, set
at 32x, which an 8-step run passes and a 512-step run fails by a factor of eight. The
fix is a block table that lives on the device and takes one write per row per step,
which is what vLLM keeps, and it is why vLLM's slot mapping does not appear in a
profile the way this one is about to. Writing the arithmetic on the day the design
incurs the cost is the same order Day 49 used for bucketing.

## What I learned
1. **Moving a computation and hiding it from the tracer are different jobs, and only
   the second one matters.** My first framing of this day was that the host now does
   the addressing "earlier". It does not: the host did all of it before too, one call
   frame deeper. The win is that the traced region no longer *observes* any of the
   values involved. The test that made this concrete is the one that replaces
   `BlockTable.num_tokens` with a property that raises: a planned forward completes,
   an unplanned one does not. A property is a data descriptor and beats the instance
   attribute, which is what makes that trick work on a plain `self.num_tokens = n`.
2. **The layer-0 handshake was a symptom, and the planned path has nothing to hand
   shake about.** `write` grows the tables on layer 0 and stashes `_step_slots` and
   `_step_rows` so layers 1..15 scatter to the same places, with a check that a later
   layer cannot name different rows. That is a sixteen-way agreement about state
   mutated inside the forward, and none of it is needed once the mutation happens
   before the forward: no stash, no check, no ordering requirement between layers.
   The prefill still uses all of it, because a ragged prefill really does decide
   where things go while it runs. What I did not expect is that code written to keep
   a mutation coherent has nothing to do the moment the mutation moves out, and I
   found that by writing a test that a planned write needs no layer 0 first and
   watching it pass with no work.
3. **The per-row Python loop went with it, on that path.** `write` loops over rows
   because a ragged prefill writes different counts to different places. A decode
   step is one token per row by definition, so with a plan the whole batch is
   `k_pool[plan.write_slots] = k[:, :, 0, :]`. That loop had survived four days of
   optimisation work because it was never the bottleneck, and it went away as a side
   effect of a change aimed at something else entirely.
4. **My own shape probe went off by one and the tests caught it, not the benchmark.**
   Day 49's `decode_shape` computed the width as "longest history plus this step's
   token", because the read built its rectangle after the write. With a plan the
   tables have already grown when the wrapper sees the call, so the same arithmetic
   overshoots. The distinct-shape *count* was still right, so every column in
   `compilebench.py` would have looked correct and every width in it would have been
   wrong by one. A measurement that is consistently wrong is the hardest kind to
   notice.
5. **Refusing a decode over an empty row is a check I did not have before and should
   have.** `plan_decode` raises if a row holds no tokens, because a decode over an
   empty row attends over its own token and no prompt: a fluent, plausible, wrong
   continuation with nothing in it that looks like an error. The old path allowed it
   silently. Writing a new entry point is the moment to ask what the old one never
   refused.
6. **A plan is stale the instant a second one is built, and nothing raises.**
   `plan_decode` is what grows the tables, so planning twice and using the first plan
   writes step two's token over step one's slot. Tables consistent, pool legal, one
   token quietly gone. `check_plan_current` compares the plan's `context_lens` with
   the tables' current lengths, which is a two-line gate against a bug I would have
   spent a day on.
7. **`static` is still eight builds and hits the fallback, and that is correct.**
   Static mode specialises per shape by definition, and the shape set is still one
   per step because `max_ctx` still grows. The plan did not close the shape set and
   was never going to; Day 49's bucketing arithmetic is what closes it, and a
   replayed CUDA graph is what will need it closed. Two separate problems that
   looked like one for a day and a half.
8. **The `compile_s` column is still an artefact and I left it in.** It is estimated
   as total decode time minus the median step times the step count, which needs most
   steps not to contain a build. `dynamic` now satisfies that and reports a credible
   44.01s; `static` builds in eight of eleven steps and reports 0.00s, which is the
   same nonsense Day 49 got. An estimator that is right in one row of a table and
   meaningless in another is worse than one that is wrong everywhere, because the
   good row lends the bad one credibility.
9. **The honest scoreboard after two days on the compiler: the pathology is gone and
   the speedup is not there.** 1 build over 11 steps at 0.998x, against 8 builds over
   9 steps at 0.02x. That is the difference between a compiler fighting the engine
   and a compiler that ran and found nothing, and it is worth saying in those words
   rather than picking whichever column reads best.

## Diagram
[decode-plan.png](../diagrams/decode-plan.png). Left top is where the addressing
lives, before and after: inside the traced region as a `table.slot(p)` loop per
layer, then outside it as four tensors built once. Right top is the guard that fails
in each case, an integer against a dimension, with the build counts each produces.
Left bottom is the measured sweep. Right bottom is the price, cells rebuilt against
cells appended, growing quadratically with the run.

## Tomorrow
The plan closed the recompile hole and opened a measurable one: the read rectangle is
rebuilt on the host every step and the rebuild is quadratic in the length of the
generation. Day 51 makes the block table persistent. Allocate `[max_batch_size,
max_model_len]` of slots on the device once, write one entry per row per step, and
hand the forward a *view* of it rather than a freshly built tensor, so
`rebuild_cells` collapses to `appended_cells` and `check_rebuild_bounded` passes at
any run length. That also fixes the row count, which is the other half of what a CUDA
graph capture needs: a fixed-size buffer at a fixed address. Day 49's bucketing
arithmetic is what tells the capture which sizes to hold, and after Day 51 there is
finally something for it to be measured against.

## Post angle
Day 50 of building an LLM inference engine from scratch. Yesterday `torch.compile`
made my decode 46x slower and the culprit was one Python integer read inside the
traced region: `cache.tables[0].num_tokens == 13`. Today I moved the whole of a decode
step's addressing out of the forward, and the fix generalises further than I
expected. A `DecodePlan` is four tensors built on the host before the call: the write
slot per row, the `[rows, ctx]` read rectangle, the context lengths, and the new
tokens' positions. With one in hand the forward reads no Python attribute of the
cache at all, and the guard dynamo fails on changes category, from
`tables[0].num_tokens == 13` to `plan.slot_mapping size at index 1: expected 4,
actual 5`. A size is the thing symbolic shapes exist for. On the real Llama-3.2-1B,
8 builds over 9 steps became 1 over 11, and 0.02x became 0.998x. The near-miss is the part worth stealing: my first move was to have
the plan pass `context_bounds=(min_ctx, max_ctx)` down to the kernel, the way Day 49
had. Those are Python ints that change every step, so a guard on their values rebuilds
every step whether the region reads them or is handed them. It is not about where a
number is computed, it is about whether the traced region ever observes it, so the
kernel takes `validated=True` and the plan checks on the host. Two things fell out
that I was not aiming at. The forward is a *function* again: a cached forward called
twice normally returns different logits, because the read writes this step's K/V
before it attends, and with the addressing fixed up front `f(x) == f(x)`. That is the
precondition for capturing a CUDA graph, not a nicety. And the per-row Python loop in
the write collapsed to one `index_put`, because a decode step is one token per row.
The price is honest and quadratic: the rectangle is rebuilt every step and `max_ctx`
grows every step, so 512 steps over 4 rows rebuild 558,080 int64 cells to append
2,048. That is what a persistent device-side block table fixes, which is what vLLM
keeps, and it is tomorrow. 1313 green.
