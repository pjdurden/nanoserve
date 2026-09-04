"""Day 49: the decode forward compiled, in graphs built and seconds earned back.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python compilebench.py

    cd ~/nanoserve && .venv/bin/python compilebench.py --modes off,dynamic,static \
        --csv docs/daily/data/day-49-compilebench.csv

Three modes, one engine each, same prompts, same tokens out. What the columns are
and why each of them is a different question:

  **graphs / breaks.** From `torch._dynamo.explain` on one decode call, before any
  timing. This is the column that decides whether the rest of the table means
  anything: a "compiled" function with breaks in it is a chain of fragments with
  the interpreter between them, and it will benchmark like the eager path because
  it mostly is the eager path. This engine read twelve of them until the
  `context_lens` validation stopped going to the device for two numbers it could
  have been handed.

  **shapes / compiles.** Distinct decode shapes the run presented, and builds that
  implies. A decode step is `[rows, 1]` over a `[rows, max_ctx]` mapping, and
  `max_ctx` grows every step, so `static` presents a new shape per step and walks
  past dynamo's recompile limit in eight of them. `dynamic` marks both dimensions
  symbolic and builds once. The `fell_back` column is the one to read first.

  **compile_s.** Measured as the first call minus the median warm call, which is
  crude and is the right order of magnitude: seconds. Everything else in this table
  is microseconds, and that ratio is the whole argument about run length.

  **step_ms / breakeven.** The saving, and how many steps it takes to pay for the
  build. Amdahl is the ceiling and it is not close to the forward's own speedup:
  the step also schedules, syncs rows, builds inputs, samples and collects, and a
  compiler touches none of those.

**A CPU is the wrong machine for the numerator and the right one for the
argument.** Inductor's CPU backend fuses pointwise chains into C++ loops, which is
a real but modest win on a small model; the shape churn, the graph breaks and the
breakeven arithmetic are properties of the engine and read the same anywhere.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from nanoserve.compiled import (
    MODES,
    breakeven_steps,
    dynamo_frames_compiled,
    explain_forward,
    recompiles,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.launch import place_weights
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel
from nanoserve.profiler import StepRecorder, device_timer_for
from nanoserve.scheduler import Request

DEFAULT_PROMPT = "The capital of France is"


def load_engine_model(weights: Path, device: str, dtype: str):
    """This engine's own model on the real weights, exactly as deferbench loads it."""
    import torch

    from transformers import AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(weights)
    config = ModelConfig.from_json(weights)
    loaded = place_weights(
        load_weights(weights, config, dtype=None), torch.device(device), torch_dtype
    )
    return tokenizer, LlamaModel(config, loaded)


def _engine(model, *, mode, slots, num_blocks, block_size, device) -> Engine:
    engine = Engine.build(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=slots,
        compile_decode=None if mode == "off" else mode,
    )
    engine.recorder = StepRecorder(device_timer=device_timer_for(device))
    return engine


def explain_one_decode(model, prompts, *, slots, num_blocks, block_size, device) -> dict:
    """Trace one decode call and report what dynamo would make of it.

    Deliberately on an uncompiled engine and before any timing: `explain` runs the
    tracer and throws the result away, so it answers "how much of this is capturable"
    without leaving a compiled artefact behind to skew the run that follows.
    """
    import torch

    engine = _engine(
        model, mode="off", slots=slots, num_blocks=num_blocks, block_size=block_size,
        device=device,
    )
    for i, prompt in enumerate(prompts):
        engine.add_request(
            Request(request_id=f"x{i}", prompt_token_ids=list(prompt), max_new_tokens=4)
        )
    engine.step()  # the prefill, so the next call is a real decode
    out = engine.scheduler.schedule()
    requests = out.decode
    rows = [r.slot for r in requests]
    engine._sync_rows(requests)
    input_ids = torch.tensor(
        [[r.output_token_ids[-1]] for r in requests], dtype=torch.long, device=device
    )
    positions = torch.tensor(
        [[engine.cache.tables[row].num_tokens] for row in rows],
        dtype=torch.long,
        device=device,
    )
    report = explain_forward(
        model.forward, input_ids, positions, cache=engine.cache.view(rows)
    )
    return {"graphs": report.graphs, "breaks": report.breaks, "ops": report.ops}


def run_mode(model, prompts, *, mode, slots, max_tokens, device, warmup, num_blocks,
             block_size) -> dict:
    """One burst through an engine in one compile mode.

    The compile cost is taken as the total decode time minus the median step times
    the step count: a build happens inside the call that needed it, so the steps it
    lands in are seconds long and the rest are milliseconds.

    **`torch._dynamo.reset()` first, and the first version of this script did not do
    it.** Dynamo's compile cache and its recompile counter are keyed per code object
    per *process*, so a mode that walks past `cache_size_limit` leaves the frame
    marked for every mode that runs after it. The run that found this reported
    `static` at 613ms and a speedup of 1.00x, which was not static being free: it
    was static never compiling at all, because `dynamic` had already exhausted the
    frame's budget three minutes earlier. A benchmark whose third row is a function
    of its second row is worse than no benchmark.
    """
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    builds_before = dynamo_frames_compiled()

    engine = _engine(
        model, mode=mode, slots=slots, num_blocks=num_blocks, block_size=block_size,
        device=device,
    )
    for i, prompt in enumerate(prompts):
        engine.add_request(
            Request(
                request_id=f"r{i}", prompt_token_ids=list(prompt), max_new_tokens=max_tokens
            )
        )
    started = time.perf_counter()
    finished = engine.run_to_completion()
    wall = time.perf_counter() - started

    profile = engine.recorder.profile(mode, warmup=0).select("decode")
    steps = [s.wall for s in profile.samples]
    warm = steps[warmup:] or steps
    median = statistics.median(warm)
    compile_s = max(0.0, sum(steps) - median * len(steps))

    forward = engine.decode_forward
    builds = dynamo_frames_compiled() - builds_before
    return {
        "mode": mode,
        "steps": engine.iterations,
        "decode_steps": len(steps),
        "shapes": forward.distinct,
        "recompiles": recompiles(forward.shapes),
        # Two numbers on purpose. `expected` is what the mode implies from the
        # shapes; `builds` is what dynamo actually did. They disagree here, and the
        # disagreement is the day: the graph is invalidated by a guard on a Python
        # int, not by a shape, so `dynamic` rebuilds as often as `static` does.
        "expected_compiles": forward.expected_compiles,
        "builds": builds,
        "reuse": round((len(steps) - builds) / len(steps), 3) if steps else 0.0,
        "fell_back": int(forward.fell_back),
        "compile_s": round(compile_s, 3),
        "step_ms": round(median * 1e3, 3),
        "wall_s": round(wall, 3),
        "tokens": sum(r.num_output_tokens for r in finished),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("./weights"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.weights.exists():
        print(f"no weights at {args.weights}", file=sys.stderr)
        return 2
    modes = [m.strip() for m in args.modes.split(",")]
    for mode in modes:
        if mode not in MODES:
            print(f"unknown mode {mode!r}; pick from {MODES}", file=sys.stderr)
            return 2

    tokenizer, model = load_engine_model(args.weights, args.device, args.dtype)
    prompt_ids = tokenizer(args.prompt)["input_ids"]
    prompts = [list(prompt_ids) for _ in range(args.requests)]

    shape = dict(
        slots=args.slots,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        device=args.device,
    )
    traced = explain_one_decode(model, prompts, **shape)
    print(
        f"one decode call traces to {traced['graphs']} graph(s) with "
        f"{traced['breaks']} break(s) over {traced['ops']} ops"
    )

    rows = []
    baseline_tokens = None
    baseline_step = None
    print(
        f"\n{'mode':>9}{'shapes':>8}{'builds':>8}{'reuse':>8}{'fell back':>11}"
        f"{'compile':>10}{'step':>12}{'speedup':>10}{'breakeven':>12}"
    )
    for mode in modes:
        row = run_mode(
            model,
            prompts,
            mode=mode,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            **shape,
        )
        # Compiling changes how the forward is executed and must not change what it
        # computes. A run whose token count moved is a bug wearing a benchmark.
        if baseline_tokens is None:
            baseline_tokens, baseline_step = row["tokens"], row["step_ms"]
        elif row["tokens"] != baseline_tokens:
            print(
                f"\nrefusing the sweep: mode {mode} kept {row['tokens']} tokens and "
                f"mode {modes[0]} kept {baseline_tokens}",
                file=sys.stderr,
            )
            return 1
        speedup = baseline_step / row["step_ms"] if row["step_ms"] else float("nan")
        saved = (baseline_step - row["step_ms"]) * 1e-3
        row["speedup"] = round(speedup, 3)
        row["breakeven_steps"] = round(breakeven_steps(row["compile_s"], saved), 1)
        row["traced_graphs"] = traced["graphs"]
        row["traced_breaks"] = traced["breaks"]
        rows.append(row)
        print(
            f"{row['mode']:>9}{row['shapes']:>8}{row['builds']:>8}"
            f"{row['reuse'] * 100:>7.0f}%{'yes' if row['fell_back'] else 'no':>11}"
            f"{row['compile_s']:>9.2f}s{row['step_ms']:>10.2f}ms"
            f"{speedup:>9.2f}x{row['breakeven_steps']:>12.0f}"
        )

    for row in rows:
        if row["mode"] == "dynamic" and row["builds"] > 1:
            # The claim `dynamic` is supposed to rest on: one symbolic build,
            # whatever arrives. It does not hold here, and saying so in the run
            # that measured it is cheaper than a reader deriving it from a column.
            print(
                f"\nnote: dynamic mode built {row['builds']} graphs over "
                f"{row['shapes']} shapes. Symbolic shapes did not stop the "
                "recompiles, so what is invalidating the graph is not a shape",
                file=sys.stderr,
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
