"""Day 46: the engine loop under a stopwatch, one phase at a time.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python profbench.py

    cd ~/nanoserve && .venv/bin/python profbench.py --slots 1,2,4,8 --warmup 2 \
        --csv docs/daily/data/day-46-profbench.csv

This is Week 13's first measurement and the first one taken from *inside* the
engine. Week 12 timed the whole thing from the outside: requests in, tokens out,
a curve. That says how fast the engine is and cannot say why, because a step is
opaque from the request boundary. `Engine.step` now carries a `recorder`, and this
script installs one, runs a burst, and asks where the milliseconds went.

Six phases per decode step, and they are the code rather than a model of it:

    schedule       reap, admit, top up the block tables
    sync_rows      copy new blocks into the cache row's table
    build_inputs   two [rows, 1] int tensors, built from Python lists
    forward        the model
    sample         logits to token ids, and back to Python
    collect        append each token, apply the stop rules

The one to watch is `sample`, because it is where the step synchronises: it returns
`list[int]`, so on a GPU the host stops there and waits for every kernel the step
queued. That is what makes per-step Python overhead land on the critical path in
full instead of hiding under the forward, and it is why `recommended_model` on this
profile says `serial` rather than `overlapped`.

**On a CPU box this script cannot price an optimisation, and says so.** There is no
second clock: the matmuls run on the same core as the Python, so every phase records
zero device time, `idle_fraction` comes out at 1.0 and the ceiling comes out
infinite. `check_device_timed` refuses that, on purpose. What a CPU profile *can* do
is separate the loop from the arithmetic by name, which is `loop_overhead_s`: the
host time in every phase except `forward`. That number does not change when the
forward gets faster, so `--forward-ms` substitutes a GPU-sized forward into this
box's loop and prints the step that would result. It is a projection and is labelled
as one.

The slot sweep is the same knob Day 44 and Day 45 turned, asked from the other side.
Out there, more slots bought throughput with diminishing returns. In here, the reason
is one number: the loop runs once per step whatever the batch, so `overhead/token`
falls like 1/B while the arithmetic per token stays flat.

`nanoserve.profiler` holds all the arithmetic and the four gates and is unit-tested
against phases placed by hand and against Amdahl's law in closed form. This file only
builds an engine, drives it, and prints.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.launch import place_weights
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel
from nanoserve.profiler import (
    ProfileUnsound,
    StepRecorder,
    check_accounted,
    check_attributable,
    check_warmed_up,
    device_timer_for,
    loop_overhead_s,
    recommended_model,
    render,
    speedup_if,
    step_with_compute,
)
from nanoserve.scheduler import Request

DEFAULT_PROMPT = "The capital of France is"


def load_engine_model(weights: Path, device: str, dtype: str):
    """This engine's own model on the real weights. No `transformers` reference here.

    Day 44 needed both systems resident because it was comparing them. This one is
    not comparing anything: the subject is nanoserve's own loop, so the second copy
    would only be memory.
    """
    import torch

    from transformers import AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(weights)
    config = ModelConfig.from_json(weights)
    loaded = place_weights(
        load_weights(weights, config, dtype=None), torch.device(device), torch_dtype
    )
    return tokenizer, LlamaModel(config, loaded)


def profile_run(model, prompts, *, slots: int, max_tokens: int, device: str, warmup: int,
                num_blocks: int, block_size: int):
    """One burst through an instrumented engine, returned as a decode profile.

    Every request is handed over before the first step, which is the offline shape
    Day 44 used: the queue is full from the start, so the batch is as close to
    `slots` as the pool allows and the profile is of a steady loop rather than of a
    ramp.
    """
    engine = Engine.build(
        model, num_blocks=num_blocks, block_size=block_size, max_batch_size=slots
    )
    engine.recorder = StepRecorder(device_timer=device_timer_for(device))
    for i, prompt in enumerate(prompts):
        engine.add_request(
            Request(
                request_id=f"r{i}", prompt_token_ids=list(prompt), max_new_tokens=max_tokens
            )
        )
    engine.run_to_completion()
    return engine.recorder.profile(f"{slots} slots", warmup=warmup).select("decode")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("./weights"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--slots", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--forward-ms",
        type=float,
        default=None,
        help="substitute this forward time into the measured loop and print the step "
        "that would result; a projection, not a measurement",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--no-sanity", action="store_true")
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.slots.split(",") if s]
    tokenizer, model = load_engine_model(args.weights, args.device, args.dtype)
    ids = tokenizer(args.prompt, return_tensors=None)["input_ids"]
    prompts = [list(ids) for _ in range(args.requests)]

    print(
        f"{args.requests} requests x {args.max_tokens} tokens, {len(ids)}-token prompt, "
        f"{args.device}/{args.dtype}, {args.warmup} warmup steps dropped"
    )

    rows = []
    for slots in sizes:
        profile = profile_run(
            model,
            prompts,
            slots=slots,
            max_tokens=args.max_tokens,
            device=args.device,
            warmup=args.warmup,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
        )
        if not args.no_sanity:
            try:
                check_attributable(profile)
                check_accounted(profile)
                check_warmed_up(profile)
            except ProfileUnsound as exc:
                print(f"\nthe profile at {slots} slots cannot be read: {exc}", file=sys.stderr)
                return 2

        loop = loop_overhead_s(profile)
        print()
        print(render(profile, title=f"{slots} slots: {profile.mean_step_s * 1e3:.3f} ms/step"))
        print(
            f"  loop {loop * 1e3:.3f} ms/step outside the forward, "
            f"{loop / profile.mean_step_s:.0%} of it, "
            f"{loop / profile.mean_batch_size * 1e3:.3f} ms/token"
        )
        print("  hotspots (by removable host time):")
        for spot in profile.hotspots(limit=3):
            print(
                f"    {spot.name:<14}{spot.per_step_s * 1e3:>8.3f} ms/step"
                f"{spot.share:>8.1%}{spot.cumulative_share:>8.1%} cumulative"
            )

        ceiling = None
        if profile.total_device_s > 0.0:
            ceiling = speedup_if(profile, overhead_factor=float("inf"))
            print(
                f"  {profile.device_utilisation:.0%} of the step is device time, so the "
                f"ceiling on removing all host overhead is {ceiling:.2f}x "
                f"under the {recommended_model(profile)} model"
            )
        else:
            print(
                "  no device times: this box runs the arithmetic on the same core as the "
                "Python, so the split does not exist and no ceiling is printed"
            )

        projected = None
        if args.forward_ms is not None:
            projected = step_with_compute(profile, args.forward_ms / 1e3)
            print(
                f"  projection: a {args.forward_ms:.2f} ms forward under this same loop is a "
                f"{projected * 1e3:.3f} ms step, {loop / projected:.0%} of it host work"
            )

        rows.append(
            {
                "slots": slots,
                "requests": args.requests,
                "max_tokens": args.max_tokens,
                "device": args.device,
                "steps": profile.steps,
                "warmup_dropped": profile.dropped,
                "mean_batch": round(profile.mean_batch_size, 3),
                "mean_step_ms": round(profile.mean_step_s * 1e3, 4),
                "host_ms": round(profile.total_host_s / profile.steps * 1e3, 4),
                "device_ms": round(profile.total_device_s / profile.steps * 1e3, 4),
                "loop_ms": round(loop * 1e3, 4),
                "loop_share": round(loop / profile.mean_step_s, 4),
                "loop_ms_per_token": round(loop / profile.mean_batch_size * 1e3, 4),
                "tokens_per_second": round(profile.tokens_per_second, 3),
                "sync_points": profile.sync_points,
                "model": recommended_model(profile),
                "top1": profile.hotspots()[0].name,
                "top1_share": round(profile.hotspots()[0].share, 4),
                "top3_share": round(profile.hotspots()[2].cumulative_share, 4),
                "ceiling": None if ceiling is None else round(ceiling, 4),
                "projected_step_ms": None if projected is None else round(projected * 1e3, 4),
            }
        )

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
