---
title: "Day 75: the benchmark script runs as a process, and its first run finds three bugs"
parent: Daily log
nav_order: 75
---

# Day 75: the benchmark script runs as a process, and its first run finds three bugs

Date: 2026-10-04 · Week 14 · Phase 5 Benchmark and optimize

## What I added today
Yesterday's "Tomorrow" was a toy-weights smoke for `graphbench.py` itself. Every test
of the script so far either checked its arithmetic or called `build_one` with
`build_app` monkeypatched to a lambda that takes any keyword. Nothing had ever run
the script.

**`nanoserve/toyckpt.py`.** `write_toy_checkpoint(dir, seed=0)` writes what
`./weights` holds, at toy size:
- `config.json` with HuggingFace's field names (2 layers, GQA 8:2, vocab 256).
- `model.safetensors` under HuggingFace's tensor names, with no `lm_head.weight`, so
  the loader takes the tied-head alias path the 1B checkpoint takes.
- `tokenizer.json` and `tokenizer_config.json`: a byte-level BPE with no merges, so
  one byte is one id and `encode` adds no specials. No EOS token, so a random model
  can't end a client at a step nobody chose.

It refuses a directory that already holds any of the four files. The same seed
writes the same bytes.

**`tests/test_toy_script.py`.** 17 tests. Eleven cover the checkpoint: it loads
through `ModelConfig.from_json`, `load_weights` and `AutoTokenizer.from_pretrained`
with nothing injected. Six run `graphbench.py --split-read --allow-cpu --arrivals
burst --max-model-len 1024` once as a subprocess, then check the exit code, the boot
order, the padded length, every split claim, the cross-arm answers and the CSV
columns.

**The three bugs it found**, all there since Day 57:
1. `build_one` passed `compile_decode` to `build_app`, and `build_engine` had no
   parameter for it: a `TypeError` before the first arm loaded. `build_engine` now
   threads it through to `Engine.build`.
2. The "graphs" arm got buckets and persistent inputs but never `capture_decode`, so
   it would have booted with the graphs off. `build_one` now passes
   `capture_decode=graphs`.
3. `--compile` defaulted to `"default"`, which is `torch.compile`'s word.
   `CompiledDecode` takes `off | dynamic | static` and raises on anything else. The
   default is now `"dynamic"`, argparse offers only `MODES`, and `off` maps to no
   compile.

Unit tests guard all three: the script's keywords are held to the real
`build_app`/`build_engine` signatures, the graphed arm must ask for the capture, and
the parser rejects `--compile default`. Suite **2360 green** (34 GPU-gated skips),
ruff clean.

## Why it matters
**The card run would have died in its first second.** Booking a GPU, typing
yesterday's one command, and getting a `TypeError` before the first arm loads is the
cheap version. Bug 2 was worse: if bug 1 had been fixed alone, the run would have
finished, the "capture was used" claim would have failed, and every ratio in the
table would have compared two eager servers.

**A patched launcher accepts anything.** `lambda weights, **kw: calls.append(kw)`
is fine for asserting what the script passes and useless for checking whether the
callee takes it. The Day 74 test `split == graphs` passed with a keyword nothing
downstream knew, because both arms passed the same wrong keyword.

**A smoke on CPU costs 14 seconds.** The script boots three servers over a 2-layer
model, pads the prompt with a real HF tokenizer, serves six clients at a width that
splits, and writes the CSV. That's cheap enough to run on every suite pass.

## What I learned
1. **Test a script by running it.** An in-process test with a patched `build_app`
   checks what the script asks for. Only a process checks that what it asks for
   exists. The three bugs were all at that seam.
2. **A byte tokenizer makes the padding predictable.** `split_prompt` joins copies
   with a space, so n copies of the 24-byte prompt are 25n - 1 tokens, and 21 copies
   (524) is the first count past the 514 floor. The test asserts 524 exactly, not
   "at least 514".
3. **Write the checkpoint the way it ships.** Leaving `lm_head.weight` out of the file
   sends the toy run through the same tied-head alias path as Llama-3.2-1B. A toy
   checkpoint with an untied head would test a path the real run never takes.
4. **A FAIL can be the right output.** On the six-request smoke, "the arms are
   comparable" fails with 11 frame gaps against the 20 it wants. That gate is meant
   to refuse a p99 from a handful of samples, so the smoke asserts the split claims
   and the answers rather than a clean sheet.

## Diagram
[graphbench-toy-smoke.png](../diagrams/graphbench-toy-smoke.png). Top left: the four
files and the real calls that read them. Top right: the three bugs and their fixes.
Bottom left: the command, the boot order and the padding arithmetic. Bottom right:
what the script printed once fixed (CPU, fixture-sized).

## Tomorrow
The script now runs end to end, but on CPU it runs `--no-compile`, so the compile
path it passes to the engine has only been checked at the signature level. Next:
extend the smoke to a two-rate sweep (`--rates` with `--arrivals fixed`) so the
multi-row table, `print_table`'s split columns and a two-row CSV are exercised on a
real process. That's the last part of the script the card run uses that hasn't been
run.

The hardware caveat hasn't changed: the card run with `--split-read --rates 1,2,4,8`
is still the first one to book. Now it has actually been run once.

## Post angle
Day 75 of building an LLM inference engine from scratch. I ran my benchmark script as
a real process for the first time, over a tiny checkpoint written to a temp directory,
and it died at boot. Three bugs from Day 57, all hidden by tests that mocked the
launcher: an unknown kwarg, a "CUDA graphs" arm that never asked for the capture, and
a bad compile default. 14 seconds on CPU found them, before any card time was booked.
vLLM and SGLang are the production systems I'm learning from; this was a lesson about the harness, not the kernels.
2360 green.
