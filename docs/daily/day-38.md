---
title: "Day 38: how many blocks fit"
parent: Daily log
nav_order: 38
---

# Day 38: how many blocks fit

Date: 2026-08-18 · Week 10 · Phase 4 Serving layer

## What I added today
`src/nanoserve/launch.py` and `serve.py`. Day 37 built a bridge and an app that
both take their engine and their tokenizer by injection, which is why neither has
ever met Llama-3.2-1B. The launcher is the thing that does: read the config, put
the weights on the device, size the block pool against what the card has left,
build the HF tokenizer, hand the three to `create_app`, and run it under uvicorn.
`tests/test_launch.py` (33 tests) covers all of it without `./weights` and without
a GPU, because the memory probe and the peak allocator stats are injected and the
sizing is arithmetic over a config. Suite **598 green** (5 GPU-gated skips), ruff
clean. `server.py`'s third and last TODO is closed.

It runs. On CPU, fp32, 2 slots, a 64 MiB pool, the real Llama-3.2-1B loads in 4.5s
and `POST /v1/completions` with `"The capital of France is"` comes back in 4.6s
with `" Paris. It is the most populous city in France and the"`. That is the first
time in 38 days that a socket has talked to this engine and gotten English out.

The interesting half is one question with a number for an answer: **how many blocks
fit?** Every day until now took `num_blocks` as an argument and believed it. Every
test, every benchmark, every fixture picked whatever made the scenario work. A
launcher does not get to do that.

## Why it matters
**A block's size is arithmetic over the config, and GQA is a factor of four in it.**
A block holds `block_size` tokens of K and V, for every layer, for every *KV* head:

    2 * num_hidden_layers * block_size * num_key_value_heads * head_dim * itemsize

For Llama-3.2-1B at block_size 16 in bf16 that is exactly 512 KiB, which makes
everything downstream readable: a gibibyte of KV is 2048 blocks, and 2048 blocks is
32768 tokens. Write `num_attention_heads` there instead and a block is 2 MiB, the
pool comes out a quarter the size the card could hold, and nothing raises. The
server is correct, four times less concurrent, and preempts under a load it should
have absorbed. Week 5 built GQA because 8 KV heads is 4x less cache than 32; this
one line is where that saving is either taken or thrown away.

**The budget is what is free after the weights land, minus what a forward will
transiently want.** Two subtractions, and only one of them is obvious:

    budget = free_after_load - (1 - utilization) * total - activation_bytes

The order matters more than the formula. `free` is asked of the driver *after* the
model is on the card, which means the weights, the CUDA context, the allocator's
fragmentation and any co-tenant process are already subtracted by something that
knows the real numbers. Sizing first and subtracting an estimate of the weights
would be remodelling all of that badly. `utilization` is vLLM's
`gpu_memory_utilization` and it is multiplied by *total*, not by free, on purpose:
it is a promise about the whole device, so a co-tenant comes out of your budget
rather than out of your safety margin, which is the conservative direction.

`activation_bytes` is the term people leave out, and on this model it is not small.
Measured by a profile run at the largest shape the scheduler can produce (which is
what vLLM does) or estimated on paper for a box with no CUDA:

| the reserve at 8 x 2048, bf16 | bytes | what it is |
|---|---|---|
| logits | 3.91 GiB | `[batch, len, vocab]`, all 128256 of it, every position |
| scores | 2.00 GiB | `[batch, heads, len, len]`, quadratic in the prompt |
| swiglu | 0.75 GiB | gate, up and their product, all three live at once |
| total | **6.66 GiB** | against 2.30 GiB of weights |

The transient beats the weights by 2.9x, and 59% of it is a tensor the engine
throws away: `LlamaModel.forward` returns logits for every position so a prefill
can be diffed against HF token for token, and `last_token_logits` reads one row per
sequence. Returning only the last row would take the reserve from 6.66 GiB to 2.75
and the pool from 24,645 blocks to 32,657, a **33% larger cache for a slicing
change**. That is a Week 13 optimization with a number on it now.

**Both knobs are memory limits, not just admission limits.** On a 24 GiB card:

| slots x length | reserve | KV pool | blocks | requests at that length |
|---|---|---|---|---|
| 4 x 2048 | 3.33 GiB | 15.37 GiB | 31,469 | 245 |
| 8 x 2048 | 6.66 GiB | 12.03 GiB | 24,645 | 192 |
| 16 x 2048 | 13.33 GiB | 5.37 GiB | 10,997 | 85 |
| 8 x 4096 | 17.33 GiB | 1.37 GiB | 2,805 | 10 |

Doubling the served length doubles the KV each request wants *and* quadruples the
score rectangle it is reserved against, so the pool collapses 8.8x and the number
of full-length requests that fit goes from 192 to 10. `--max-batch-size` and
`--max-model-len` look like policy and are spending the same bytes.

**A pool that cannot hold one request is a server that refuses everything.** Day
33's `Scheduler.add_request` rejects any request whose worst case exceeds the whole
pool, because FIFO admission would otherwise park behind it forever. So a pool
sized under `blocks_for_length(max_model_len)` produces a process whose `/health`
says ok and whose every completion is a 400. `plan_kv_pool` refuses to boot there
and names the length it could have served, so the flag to change is in the failure.

## What I learned
1. **`{k: v.to(device) for k, v in weights.items()}` breaks the tie, and the only
   symptom is a smaller pool.** Llama-3.2-1B ships no `lm_head.weight`; Day 3's
   loader aliases it to `embed_tokens.weight`, one storage under two names. The
   obvious dict comprehension calls `.to` once per *name*, so the tie lands on the
   card as two independent copies. That is 501 MiB of duplicated embedding, which
   at 512 KiB a block is 1002 blocks and 16,032 tokens of KV pool spent on a matrix
   that was already there. Nothing detects it: the model is correct, the logits are
   identical to the last bit, the server is just quietly less concurrent.
   `place_weights` keys on storage so a shared one moves once.
2. **`max_memory_allocated` is a high-water mark, not a delta.** Reading it after
   the profile forward charges the activation reserve for the resident weights too,
   which subtracts them twice: once by the driver in `free`, once by me. Reset,
   read the level, run, read the peak, subtract. On this model that mistake is
   2.30 GiB, 4,714 blocks, and it is invisible because a too-small pool never fails,
   it just preempts.
3. **`max_model_len` is a serving decision that looks like a model property.** The
   config says 131072, and a pool that can hold one such request is 8192 blocks and
   4 GiB before a second caller exists. So it is a flag, clamped to
   `max_position_embeddings` (which is the half that really is a model property:
   more context than the weights were trained for is not a memory question and
   cannot be bought). The plan reports `concurrency_at_max_len` deliberately
   unclamped by the slot count, because the useful thing is knowing which of the
   two limits binds first. At 8 x 2048 it is the slots, and Day 33's preemption is
   not what sheds load. At 8 x 4096 it is the pool, and preemption is the whole
   story.
4. **The profile run happens with no cache, and that is right rather than a
   compromise.** The pool it is sizing does not exist yet, so the dummy forward
   goes down the Week-2 recompute path. Same score rectangle, same MLP rectangles,
   same logits, because a prefill attends over exactly its own context whether it
   reads it from a cache or recomputes it. What it misses is the cache write, which
   is a scatter into the pool being sized and allocates nothing new.
5. **A CPU launch has no VRAM to divide, so it has to say so.** There is no
   per-device free-memory number on CPU and taking a fraction of system RAM is a
   promise the allocator cannot keep. `kv_budget_bytes` raises instead of guessing
   and names the flag (`kv_cache_bytes`) that answers it, which also happens to be
   why the whole sizing path is testable on this box.

## Diagram
[kv-pool-sizing.png](../diagrams/kv-pool-sizing.png). Left is a 24 GiB card drawn
as a stack in the order the launcher subtracts it: weights, CUDA context, the
activation reserve, the utilization reserve, and the KV pool that is what is left,
with the division into blocks and tokens spelled out. Right top is the bytes-per-block
formula with the GQA trap next to it, 512 KiB against 2 MiB. Right bottom is the
slots-by-length sweep and the 8.8x collapse. The banner is the floor rule: a pool
under one request's worth is a server that answers ok on `/health` and 400 to
everybody.

## Tomorrow
Day 39 opens Week 11 with the thing the bridge was already shaped for: streaming.
`stream=true` is currently a 400 that says "Week 11", and the change it needs is
one future becoming one queue, plus an SSE response that yields `data:` frames in
OpenAI's chunk shape and a `[DONE]` sentinel. The gotchas are detokenization
(Llama's BPE emits multi-byte characters across token boundaries, so a naive
per-token `decode` prints replacement characters) and the disconnect path, which
already works for the unary case and has to keep working when the cancellation
arrives mid-stream rather than mid-await.

## Post angle
Day 38 of building an LLM inference engine from scratch. Today the engine got a
launcher, so for the first time an HTTP request reached the real Llama-3.2-1B and
came back with English. The whole day is one question: how many KV blocks fit?
Every day until now took `num_blocks` as an argument and believed whatever number
made the test pass. A launcher does not get to. First: a block costs
`2 * layers * block_size * num_key_value_heads * head_dim * itemsize`, which for
Llama-3.2-1B at block_size 16 in bf16 is exactly 512 KiB. Use `num_attention_heads`
there and you get 2 MiB, a pool a quarter the size the card could hold, and no
error anywhere. That single line is where GQA's saving is taken or thrown away.
Second, and this is the part I got wrong first: the budget is not "card minus
weights". It is what the driver reports free *after* the weights are resident,
minus a reserve against the card total, minus what the biggest forward transiently
wants. That last term is huge. At 8 requests x 2048 tokens this model wants 6.66
GiB of activations against 2.30 GiB of weights, and 3.91 GiB of that is the
`[batch, len, vocab]` logits rectangle, which the engine reads exactly one row of
per sequence. Returning only the last row would grow the pool 33%, from 24,645
blocks to 32,657. Third: `--max-batch-size` and `--max-model-len` look like
admission policy and are actually spending the same bytes. Going from 8x2048 to
8x4096 takes the pool from 12.03 GiB to 1.37, because doubling the length doubles
the KV per request and quadruples the score rectangle it is reserved against. And
the one that would have bitten me silently: moving the weights with
`{k: v.to(device) for k, v in weights.items()}` calls `.to` twice on the tied
`lm_head`/`embed_tokens` tensor and puts two copies on the card. 501 MiB, 1002
blocks, 16,032 tokens of cache, spent on a matrix you already had, with identical
logits and no error. vLLM's profile run is where I stole the measurement idea from.
598 green.
