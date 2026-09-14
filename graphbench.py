"""Day 57: two servers, one with CUDA graphs and one without, measured per token.

Run from the repo root with the venv python. The whole point is the pair, so this
script starts both arms itself:

    cd ~/nanoserve && .venv/bin/python graphbench.py --weights ./weights \
        --device cuda --requests 16 --rate 4 --max-tokens 128

    cd ~/nanoserve && .venv/bin/python graphbench.py --weights ./weights \
        --device cuda --rates 1,2,4,8 --requests 32 --max-tokens 128 \
        --csv docs/daily/data/day-57-graphbench.csv

Each arm is one process's worth of engine, launched through `build_app` with the
Week 13 flags on or off, served over a real socket by uvicorn, and driven by the
same `ClientPlan` list. Two passes per arm: a crowd, whose answers are compared
across the arms byte for byte, and an open-loop run at an offered rate, which is
where the latency numbers come from.

**The two percentiles are the report and a mean is not.** A capture removes host-side
launch overhead from every step, which moves the middle of the inter-token latency
distribution. A warm-up removes recordings that would otherwise happen inside one
unlucky request, which is only ever the tail. Averaged together, each hides the
other: a 200 ms recording spread over a thousand steps is 0.2 ms of mean and the
whole of the p99.

**Expect the gap to be smaller than the arithmetic says.** The eager arm is not
un-optimised: Day 49's compile already collapses some of the launches a graph would
have saved, so this is a three-way question measured two ways at a time, and
`--no-compile` is the third arm if you want to separate them.

**And expect a share of the steps to not be replaying at all.** Day 57 found that a
recorded read is a window from cache row zero, so a step whose scheduler rows are not
`(0, 1, ... n-1)` cannot replay any graph in the list and runs the forward instead.
That happens the moment one request in a batch finishes before its neighbour, which
is most of the time on a real workload. `replay_share` in the table is the fraction
of decode steps the capture actually covered, and it is the number that says how much
of the graphed arm's speedup was even available.

On CPU this script will still run and it will not report a speedup worth reading: the
recorder is `eager_recorder`, a stand-in with a capture's semantics and none of its
speed, so the "recorded" arm is a forward plus a copy. `--allow-cpu` is there for
checking the harness, and it prints a line saying exactly that.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

import torch

from nanoserve.acceptance import AcceptanceFailure, live_server
from nanoserve.graphbench import (
    ArmDelta,
    check_arm_replayed,
    check_arm_replayed_every_step,
    check_arm_was_crowded,
    check_arms_comparable,
    check_nothing_recorded_while_serving,
    check_same_answers,
    check_tail_not_worse,
    paired_plans,
    run_arm,
)
from nanoserve.servebench import (
    MeasurementUnsound,
    burst_arrivals,
    fixed_arrivals,
    poisson_arrivals,
)

DEFAULT_PROMPT = "The capital of France is"

#: The toy model the coverage study runs on. Two layers over a real batched cache,
#: because the question it asks is about the scheduler's row assignment and not about
#: the model: a 1B checkpoint would produce the same table an hour later.
COVERAGE_BLOCK = 16
COVERAGE_LEN = 512


def _toy_engine(slots: int):
    """A graphed, warmed engine over a two-layer random model. No weights, no device."""
    from nanoserve.captured import eager_recorder
    from nanoserve.config import ModelConfig
    from nanoserve.engine import Engine
    from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
    from nanoserve.model import LlamaModel

    config = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    torch.manual_seed(0)
    tensors = {n: torch.randn(*s) for n, s in expected_shapes(config).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    engine = Engine.build(
        LlamaModel(config, Weights(tensors, config)),
        num_blocks=slots * 64,
        block_size=COVERAGE_BLOCK,
        max_batch_size=slots,
        max_model_len=COVERAGE_LEN,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )
    engine.warm_decode()
    return engine


def coverage_row(*, slots: int, requests: int, lengths: tuple[int, ...]) -> dict:
    """How much of one workload's decode the capture could replay at all.

    The question Day 57 has to answer before any speedup is worth quoting, and it
    needs no device: a step can replay only while the scheduler's rows are
    `(0, 1, ... n-1)`, which is a property of who is holding which slot and not of
    what the kernels do. `lengths` is the generation budget, cycled over the
    requests, so `(16,)` is a batch that finishes together and `(4, 64)` is one where
    every short request leaves a hole in the middle of the row space.
    """
    from nanoserve.captured import CaptureStats
    from nanoserve.scheduler import Request

    engine = _toy_engine(slots)
    graphs = engine.decode_graphs
    # The day's own instrument, used on the day's own benchmark: the counters are
    # cumulative and the warm-up is already in them, so the run is a window.
    before = CaptureStats.of(graphs)
    for i in range(requests):
        engine.add_request(
            Request(f"r{i}", [1 + (i % 7), 2, 3, 4], max_new_tokens=lengths[i % len(lengths)])
        )
    while engine.has_unfinished():
        engine.step()
    served = CaptureStats.of(graphs).since(before)
    return {
        "slots": slots,
        "requests": requests,
        "lengths": "/".join(str(x) for x in lengths),
        "decode_calls": served.calls,
        "replayed": served.replays,
        "scattered": served.scattered_calls,
        "replay_share": round(served.replay_share, 4),
        "graphs_held": graphs.count,
        "recorded_while_serving": served.captures,
    }


def run_coverage(args) -> list[dict]:
    """The table: what share of a real decode loop this engine's capture can cover."""
    plans = [
        (4, 8, (16,)),
        (4, 8, (8, 16)),
        (4, 8, (4, 32)),
        (8, 16, (16,)),
        (8, 16, (8, 16)),
        (8, 16, (4, 32)),
        (8, 16, (4, 8, 16, 32)),
        (16, 32, (4, 32)),
    ]
    rows = [
        coverage_row(slots=slots, requests=requests, lengths=lengths)
        for slots, requests, lengths in plans
    ]
    header = (
        f"{'slots':>6} {'requests':>9} {'gen lengths':>14} {'decode steps':>13} "
        f"{'replayed':>9} {'scattered':>10} {'share':>7}"
    )
    print("what share of a decode loop can replay at all, by workload shape:")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['slots']:>6} {r['requests']:>9} {r['lengths']:>14} "
            f"{r['decode_calls']:>13} {r['replayed']:>9} {r['scattered']:>10} "
            f"{r['replay_share']:>6.0%}"
        )
    return rows


def schedule_for(kind: str, n: int, *, rate: float | None, seed: int) -> list[float]:
    """Turn a flag into arrival offsets. Day 42's three processes, unchanged."""
    if kind == "burst":
        return burst_arrivals(n)
    if rate is None:
        raise SystemExit(f"--arrivals {kind} needs a --rate")
    if kind == "fixed":
        return fixed_arrivals(n, rate=rate)
    if kind == "poisson":
        return poisson_arrivals(n, rate=rate, seed=seed)
    raise SystemExit(f"unknown arrival process {kind!r}")


def build_one(args, *, graphs: bool):
    """One arm's app: the same launcher call with the week's three flags on or off.

    Both arms get the same pool, the same served context and the same compile
    setting. The only difference between the two processes is the capture, which is
    what makes the difference between the two reports attributable to it.
    """
    from nanoserve.launch import build_app

    return build_app(
        args.weights,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
        max_model_len=args.max_model_len,
        num_blocks=args.num_blocks,
        kv_cache_bytes=args.kv_cache_bytes,
        compile_decode=None if args.no_compile else args.compile,
        bucket_decode=graphs,
        persist_inputs=graphs,
        capture_decode=graphs,
        warm=not args.no_warm,
        warm_rows=args.warm_rows,
        warm_width=args.warm_width,
    )


def measure(args, label: str, rate: float | None) -> ArmDelta:
    """Both arms, one offered load, and the comparison between them."""
    plans = paired_plans(
        args.requests,
        prompts=(args.prompt,),
        max_tokens=(args.max_tokens,),
        seed=args.seed,
    )
    arrivals = schedule_for(args.arrivals, len(plans), rate=rate, seed=args.seed)
    reports = {}
    for name, graphs in (("graphs", True), ("eager", False)):
        print(f"\n[{label}] loading the {name} arm ...", file=sys.stderr, flush=True)
        app = build_one(args, graphs=graphs)
        for line in app.state.plan.describe().splitlines():
            print(f"  {line}", file=sys.stderr, flush=True)
        if app.state.capture is not None:
            print(f"  {app.state.capture.describe()}", file=sys.stderr, flush=True)
        with ExitStack() as stack:
            server = stack.enter_context(live_server(app))
            reports[name] = asyncio.run(
                run_arm(
                    server.base_url,
                    plans,
                    arrivals,
                    name=name,
                    target_rate=rate,
                    timeout=args.timeout,
                )
            )
    return ArmDelta(graphs=reports["graphs"], eager=reports["eager"])


def write_csv(where: str, rows: list[dict]) -> None:
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'load':>10} {'ITL p50 graphs':>15} {'eager':>8} {'x':>6} "
        f"{'ITL p99 graphs':>15} {'eager':>8} {'x':>6} {'replayed':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['load']:>10} {r['graphs_itl_p50_ms']:>14.2f}m {r['eager_itl_p50_ms']:>7.2f}m "
            f"{r['itl_p50_speedup']:>5.2f}x {r['graphs_itl_p99_ms']:>14.2f}m "
            f"{r['eager_itl_p99_ms']:>7.2f}m {r['itl_p99_speedup']:>5.2f}x "
            f"{r['replay_share']:>8.0%}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="compare a graphed nanoserve with an eager one")
    target = p.add_argument_group("the two servers")
    target.add_argument("--weights", default=None, help="the checkpoint both arms load")
    target.add_argument(
        "--coverage",
        action="store_true",
        help="skip the servers: how much of a decode loop can replay at all, by workload",
    )
    target.add_argument("--device", default="auto", help="auto | cuda | cpu")
    target.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    target.add_argument("--block-size", type=int, default=16)
    target.add_argument("--max-batch-size", type=int, default=8, help="concurrent slots")
    target.add_argument("--max-model-len", type=int, default=2048)
    target.add_argument("--num-blocks", type=int, default=None, help="skip KV sizing")
    target.add_argument("--kv-cache-bytes", type=int, default=None)
    target.add_argument("--compile", default="default", help="the Day-49 mode, both arms")
    target.add_argument("--no-compile", action="store_true", help="neither arm compiles")
    target.add_argument("--no-warm", action="store_true", help="record lazily, the control")
    target.add_argument("--warm-rows", type=int, default=None)
    target.add_argument("--warm-width", type=int, default=None)
    target.add_argument(
        "--allow-cpu",
        action="store_true",
        help="run without a device, where a replay is a stand-in and no faster",
    )

    load = p.add_argument_group("the load")
    load.add_argument("--requests", type=int, default=16)
    load.add_argument("--rate", type=float, default=None, help="offered requests/second")
    load.add_argument("--rates", default=None, help="sweep, e.g. 1,2,4,8")
    load.add_argument(
        "--arrivals", default="poisson", choices=("poisson", "fixed", "burst")
    )
    load.add_argument("--prompt", default=DEFAULT_PROMPT)
    load.add_argument("--max-tokens", type=int, default=128)
    load.add_argument("--seed", type=int, default=0)
    load.add_argument("--timeout", type=float, default=600.0)
    load.add_argument("--min-samples", type=int, default=20, help="frame gaps a p99 needs")
    load.add_argument("--tolerance", type=float, default=0.10, help="slack on the p99 gate")
    load.add_argument("--csv", default=None, help="write the table here")
    load.add_argument(
        "--sanity",
        action="store_true",
        help="fail if the arms disagree, or the comparison does not support itself",
    )
    args = p.parse_args()

    if args.coverage:
        rows = run_coverage(args)
        if args.csv:
            write_csv(args.csv, rows)
        return
    if args.weights is None:
        raise SystemExit("pass --weights (both arms load the same checkpoint), or --coverage")
    if args.rates is None and args.rate is None and args.arrivals != "burst":
        raise SystemExit("pass --rate, --rates, or --arrivals burst")
    if args.device == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "on CPU there is no torch.cuda.graph, so the recorded arm is a forward "
            "plus a copy and is slower by exactly that copy. Pass --allow-cpu to run "
            "it anyway, for the harness rather than for the number."
        )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("no CUDA device here; pass --allow-cpu to exercise the harness")

    loads: list[tuple[str, float | None]]
    if args.rates:
        loads = [(f"{r} rps", float(r)) for r in args.rates.split(",")]
    else:
        loads = [(f"{args.rate} rps" if args.rate else "burst", args.rate)]

    rows: list[dict] = []
    for label, rate in loads:
        delta = measure(args, label, rate)
        print(f"\n{label}:")
        print(delta.render())
        row = {"load": label, **delta.row()}
        row["replay_share"] = round(delta.graphs.served.replay_share, 4)
        rows.append(row)

        # The claims, reported rather than raised unless --sanity says otherwise.
        for check, note in (
            (lambda: check_arm_was_crowded(delta.graphs), "the graphed arm batched"),
            (lambda: check_arm_was_crowded(delta.eager), "the eager arm batched"),
            (lambda: check_same_answers(delta.graphs, delta.eager), "same answers"),
            (lambda: check_arm_replayed(delta.graphs), "the capture was used"),
            (
                lambda: check_nothing_recorded_while_serving(delta.graphs),
                "nothing recorded while serving",
            ),
            (
                lambda: check_arms_comparable(
                    delta.graphs, delta.eager, min_samples=args.min_samples
                ),
                "the arms are comparable",
            ),
            (
                lambda: check_arm_replayed_every_step(delta.graphs),
                "the capture covered every step",
            ),
            (
                lambda: check_tail_not_worse(delta, tolerance=args.tolerance),
                "the tail did not get worse",
            ),
        ):
            try:
                check()
                print(f"  ok    {note}")
            except (AcceptanceFailure, MeasurementUnsound) as exc:
                print(f"  FAIL  {note}:\n        {exc}")
                if args.sanity:
                    raise SystemExit("\nthis comparison does not support its numbers")

    if len(rows) > 1:
        print()
        print_table(rows)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
