---
title: "Day 55: the capture list moved to startup, and the dummy batch was all padding"
parent: Daily log
nav_order: 55
---

# Day 55: the capture list moved to startup, and the dummy batch was all padding

Date: 2026-09-09 · Week 13 · Phase 5 Benchmark and optimize

## What I added today
`nanoserve.warmup`, the capture list recorded before the server accepts anything.
`warmup_plan` builds one shape's addressing over no sequences at all: `rows=()`,
`pad_rows` the whole shape, every `write_slots` entry the sink, every `context_lens`
entry 0, the rectangle a window on the real slot table and the three addressing
tensors written into the real Day-53 buffers. `warmup_batch` is that plus the
made-up tokens and a `BatchedCacheRows` wearing no rows. `warm_decode` walks a list
of shapes, drives one synthetic batch through the Day-54 capture for each, and hands
back a `WarmupReport`.

`warm_shapes` is the bucket set as a capture list, biggest first (a shared pool is
sized by its largest member, so recording that one first means the arena is
allocated once at its final size) with `max_rows` and `max_width` to trim it to what
a server will actually present. The arithmetic is `lazy_captures`, `stall_seconds`,
`warmup_seconds`, `warm_budget_bytes` and `width_ceiling`. The gates are
`check_warm_rows_empty`, `check_warm_writes_sink`, `check_warm_touches_nothing`,
`check_table_stable`, `check_warm_graphs_unbound`, `check_no_cold_captures`,
`check_all_warm` and `check_warm_budget`.

Three small changes elsewhere, and all three are the same shape: an empty row
selection is now legal in exactly the places a warm batch reaches.
`SlotTable.read` takes `rows=()` when `pad_rows` is positive and refuses it
otherwise; `BatchedPagedKVCache.write` and `.paged_attention` read an empty
selection as "the rows this plan addresses", which on a warm plan is none.
`SlotTable` also gains an `address`, which is the number Day 53 has had for its
buffers and Day 51 never had for the larger one. `Engine.warm_decode` is the door.

`tests/test_warmup.py` is 51 tests. Suite **1609 green** (5 GPU-gated skips), ruff
clean.

`warmbench.py`, host only, no weights, 2 rows off a 16-token prompt at width
multiple 32 ([day-55-warmbench.csv](data/day-55-warmbench.csv)):

    generated    cold: lazy   predicted    cold: warmed    warm list
            8             1           1               0     4 graphs,   8.5 ms
           32             2           2               0     4 graphs,   8.0 ms
           64             3           3               0     6 graphs,  12.2 ms
          128             5           5               0    10 graphs,  20.9 ms

and where the same recordings get paid for, at 256 rows and 8192 tokens with width
multiple 128, priced at 50 ms a graph:

    generated   lazy: graphs   in a client's latency    warmed at startup
                                                        full  rows<=8  rows<=8,ctx<=2048
          128              1                  0.05s    28.80s   12.80s        3.20s
          512              4                  0.20s    28.80s   12.80s        3.20s
         2048             16                  0.80s    28.80s   12.80s        3.20s
         8192             64                  3.20s    28.80s   12.80s        3.20s

    what a pool budget buys, 256 rows, 32 heads, fp32 scores:
       64 MB free  ->  widest capture  1,953 tokens,  135 of 576 shapes
      268 MB free  ->  widest capture  8,178 tokens,  567 of 576 shapes
    1,000 MB free  ->  the whole list fits

## Why it matters
**The dummy batch was already designed, three days ago, for something else.** I
spent the start of the day trying to build a fake batch that would not write into
the pool, and the answer is that Day 52 built one. A *padded* row exists to hold a
shape still while touching nothing: its `context_lens` entry is 0 so the whole row is
masked, and its `write_slots` entry is `sink_slot`, one past the pool the allocator
hands out, so its K/V lands at a legal address no `BlockTable` can name. A warm batch
is a step whose rows are **all** padding. It is not a special case of a real plan, it
is the limit of one, and that is why nothing had to be added to make it safe: the
sink was already unconditional, the mask was already a length, and the write was
already one `index_put` over the padded batch.

**Every one of Day 54's six preconditions passes on it, and passes *vacuously*.**
`check_pad_inert`'s third clause is "no real row writes to the sink" and there are no
real rows. `check_mapping_is_window` wants the row set to be a prefix of the table's
and the empty tuple is a prefix of everything. `check_window_intact` and
`check_plan_addressing` both loop over the real rows and find none. Vacuous is a
better outcome than special-cased, because a gate that had to learn about warm-ups
would be a gate whose meaning now depends on who is calling it.

**Warming deletes Day 54's one non-storage field rather than guarding it.**
Yesterday ended on `check_replay_rows`, the only gate on this arc whose subject is a
Python value instead of a pointer: a mid-run capture freezes step 1's `plan.rows`
tuple and holds it for the life of the process, so a replay over a different set of
sequences would write this step's tokens into the other batch's slots. A graph
recorded on a warm batch freezes the *empty* tuple. There is nothing it can be wrong
about. `check_warm_graphs_unbound` is that stated positively, and it is how you tell
that every graph in the list came from the warm-up rather than from a step that
slipped past it. I did not expect the two halves of the week to meet like that.

**A graph recorded over nothing computes real steps, and it is not luck.** The
recorded call keeps its argument objects forever, and every one of them is a window
on storage a real step writes through: `slot_mapping` is Day 51's table,
`input_ids`, `positions`, `write_slots` and `context_lens` are Day 53's buffers. So
the frozen plan reads this step's numbers, and a warm engine's tokens are the
unwarmed engine's tokens. Five days of moving things into fixed storage is what
makes a sentence that sounds like a bug ("the graph holds a plan built over no
sequences at all") into a correctness argument.

**The stall was never a one-off, and warming the whole set is not affordable.** The
lazy cost is one recording per width bucket a run crosses, so it recurs every
`width_multiple` tokens for as long as a request generates: `lazy_captures` predicts
it and the benchmark's `cold` column matches at every length. But the closed set is a
*product*, and the honest table is the one that hurts. At serving size the full list
is 576 shapes, which at 50 ms a graph is 28.8 seconds of startup, against 3.2 seconds
of total stall for a request that generates 8192 tokens. Warming everything is nine
times worse than warming nothing. The warm list has to be trimmed on both axes, and
the trim is a *serving* decision (how many rows will the scheduler admit, how long is
the context this deployment sells) rather than a memory one. That is why
`warm_shapes` takes two ceilings, and it is a design conclusion I would not have
reached without printing the table.

## What I learned
1. **The hardest part of the day was already solved and filed under another name.**
   I went looking for a way to present a shape without owning a row, and the sink
   slot and the zero-length mask were both sitting in `nanoserve.buckets` from Day
   52, written for padded rows in a real batch. A design where the degenerate case is
   the limit of the general one rather than a branch off it is the thing that made
   this a short module.
2. **Vacuous is the outcome to want from a reused gate.** Six preconditions passed on
   a batch with nothing in them, and my first instinct was to check whether they were
   really doing anything. They are: the same six fail on a real plan built the wrong
   way, and they fail on a warm plan built the wrong way (a warm plan whose rows are
   not empty trips `check_pad_inert`'s sink clause). A gate that has nothing to say
   about a legal input is different from a gate that has been switched off.
3. **Day 54 was guarding the small buffer and not the large one.** `check_addresses_stable`
   compares four `[max_batch]` vectors, 8 KB at 256 rows. The rectangles those same
   graphs read are windows on `[max_batch, max_model_len]`, 16 MB at 8192 tokens, and
   `SlotTable.to` reallocates. Nothing was watching it. Warming is what made the hole
   reachable rather than theoretical, because warming is now the first thing in the
   process to hand out a window and therefore the first thing that can hand one out on
   the wrong device. `check_table_stable` is one line and it should have been written
   on Day 51.
4. **A capture list is a product and a stall is a rate, and comparing them needs both
   in seconds.** I had "warming removes the stall" as the day's claim, and the
   benchmark says warming the whole set costs nine times what the stall it removes
   costs. Both numbers are right and the conclusion is neither: warm a *trimmed* list.
   The lesson is that "move the cost to startup" is not free by virtue of being at
   startup, and the only way I found that out was pricing both sides in the same
   units.
5. **The budget is a statement about one number and that number is a width.** Day 54
   left `check_pool_budget` wanting "what is left after the weights and the pool" with
   nobody to ask. Connecting it to the probe was five lines; the useful part was
   realising what to do with a refusal. A shared pool is a max, so a budget does not
   trim the *length* of the list, it caps the widest shape in it. `width_ceiling` is
   the inversion, and `warm_shapes(max_width=width_ceiling(...))` is the whole use.
6. **An empty selection had to become legal in exactly three places, and refusing it
   everywhere else was the right call.** `SlotTable.read` takes it only when something
   is padded; `cache.view(())` still refuses, because a scheduled view over no rows is
   a forward with nothing in it, and the warm-up builds its `BatchedCacheRows`
   directly to say that it means the empty tuple. Widening a guard for one caller is
   how a guard stops meaning anything, and the fix was to widen it by the condition
   that made the caller legitimate.
7. **The pool gets allocated at warm time, which is a second thing warming buys.**
   `_ensure_pool` runs on the first write of a layer, and the first write in a warmed
   process is a synthetic row's throwaway K/V. So the K/V pool is resident before the
   door opens rather than on the first prefill, which is what makes
   `warm_budget_bytes` meaningful: by the time anyone probes the device, everything
   large is already on it.
8. **Recording the biggest shape first is a real ordering and not tidiness.** Every
   capture after the first is handed the first one's pool, and an arena is sized by
   what is allocated out of it. Descending order allocates it once at its final size;
   ascending grows it once per shape on the way up. It costs one `sorted` and I only
   thought about it because Day 54's `pool_sharing_ratio` had already made the arena
   the thing I was watching.

## Diagram
[warm-capture-list.png](../diagrams/warm-capture-list.png). Left top is what a warm
batch is: `warmup_plan` with no `plan_decode` behind it, the five tensors that are
windows on the real buffers, and the row tuple that is empty rather than frozen.
Right top is Day 54's six preconditions passing unchanged, and underneath them the
two address gates side by side with what each one is worth in bytes. Left bottom is
the benchmark's cold column against `lazy_captures`'s prediction. Right bottom is the
startup cost of the full list against two trimmed ones, and the budget read as a
width ceiling.

## Tomorrow
The warm-up exists and nothing in the boot path calls it. `Engine.warm_decode` is a
method a caller has to know about, and `build_engine` does not even pass
`capture_decode` through, so a real server started from `serve.py` runs the Day-47
loop with none of the last six days switched on. Day 56 is the wiring: thread
`bucket_decode`, `persist_inputs`, `capture_decode` and the warm list through
`build_engine` and `create_app`, warm after the pool is planned and before the app
accepts, and print the boot line with the number of graphs and what they cost. Two
things I expect to be wrong about. One is the trim: the warm list needs `max_rows`
from the scheduler and `max_width` from the smaller of what the deployment sells and
what `width_ceiling` allows, and those two numbers arrive from different places at
different times in `build_engine`. The other is `/health`, which reports the
`KVPoolPlan` today and now has a second sizing decision to report next to it, made
after it rather than with it.

## Post angle
Day 55 of building an LLM inference engine from scratch. Day 54's CUDA graph capture
worked and it happened in the wrong place: the first step of each shape takes a
recording, mid-run, in front of a client waiting for a token. And it is not a one-off.
A run crosses a width bucket every `width_multiple` tokens it generates, so the stall
comes back for the whole life of a request. vLLM records its list at startup off dummy
batches, so today was building one. The hard part is what a dummy batch *is*: a graph
is bound to the addresses it was launched with, so a fake batch cannot be built
somewhere quiet and thrown away, it has to go through the real slot table, the real
input buffers and the real K/V pool. Which means the fake batch is about to write K/V
into a pool full of real sequences before anything has been served. The answer turned
out to be three days old. Day 52's *padded* rows exist to hold a shape still while
touching nothing: `context_lens` 0 so the row is masked, `write_slots` at the sink
slot so its K/V lands where no block maps. A warm batch is a step whose rows are all
padding. Not a special case of a real plan, the limit of one, which is why all six of
yesterday's preconditions pass on it unchanged and *vacuously*. The nice part is that
warming deletes yesterday's one weak spot instead of guarding it. A mid-run capture
freezes step 1's row tuple forever, and that tuple was the only thing a replay carried
that was not live storage. A warm graph freezes the empty tuple, so there is no set of
sequences it can be wrong about. The uncomfortable part is the number. The closed
shape set is a product: 576 shapes at serving size, which at 50 ms a graph is 28.8s of
startup, against 3.2s of total stall for a request generating 8192 tokens. Warming
everything is nine times worse than warming nothing. So the warm list gets trimmed on
both axes, and the trim is a serving decision, not a memory one. Moving a cost to
startup is not free by virtue of being at startup. 1609 green.
