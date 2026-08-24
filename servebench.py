"""Day 42: measure the server. TTFT, inter-token latency, throughput, per rate.

Run from the repo root with the venv python. Against a server this script starts
for itself:

    cd ~/nanoserve && .venv/bin/python servebench.py --weights ./weights \
        --device cpu --num-blocks 512 --requests 8 --rate 0.5 --max-tokens 16

or against one that is already running (`.venv/bin/python serve.py`):

    cd ~/nanoserve && .venv/bin/python servebench.py --url http://127.0.0.1:8000 \
        --requests 64 --rate 4 --max-tokens 128
    cd ~/nanoserve && .venv/bin/python servebench.py --url http://127.0.0.1:8000 \
        --rates 1,2,4,8 --requests 32 --csv docs/daily/data/day-42-servebench.csv

Two modes, and the difference between them is the whole reason this file exists.

  **Open loop (the default).** Requests go out on a schedule, at an offered rate,
  whatever the server is doing with the earlier ones. This is the only way to
  measure a queue, because the queue is made of requests that arrived while the
  server was busy, and a generator that waits for the server never sends them.

  **Closed loop (`--closed N`).** N clients, each sending its next request when
  its previous one returns. Reported here so the two can be compared on one box,
  because a closed loop is what most homegrown benchmarks are and it systematically
  under-reports latency: it stops offering load exactly when the server slows down.

`--rates` sweeps the offered rate and prints the table that has the knee in it. The
shape to look for is TTFT climbing gently and then going vertical between two
neighbouring rates, which is the point where arrivals started outrunning service
and the queue stopped draining. Everything before that knee is a capacity number
you can plan with; the p99 at a rate past it is a number that grows for as long as
you keep the run going, so it describes the run's length, not the server.

`--sanity` runs the two checks the harness owes itself and stops if either fails:
that the run really held more than one request at a time, and that the generator
kept its own schedule. A benchmark that quietly fell back to one-at-a-time still
prints a beautiful p99.

The measurement lives in `nanoserve.servebench` and its arithmetic is unit-tested
against a scripted timeline; this file only builds plans, drives them, and prints.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from contextlib import ExitStack
from pathlib import Path

from nanoserve.acceptance import ClientPlan, live_server
from nanoserve.servebench import (
    LoadReport,
    MeasurementUnsound,
    burst_arrivals,
    check_offered_load,
    check_schedule_kept,
    fixed_arrivals,
    poisson_arrivals,
    run_closed_loop,
    run_open_loop,
)

DEFAULT_PROMPT = "The capital of France is"


def build_plans(n: int, *, prompt: str, max_tokens: int) -> list[ClientPlan]:
    """n identical streaming clients.

    Identical on purpose. Mixed prompt lengths are a more realistic workload and a
    worse instrument: when the rate sweep moves the latency, you want the only
    thing that changed to be the arrival schedule. Day 41's `mixed_crowd` is the
    workload that disagrees with itself, and it is a correctness harness.

    Streaming, because a unary response arrives in one piece and cannot be timed
    per token. The endpoint being measured is the one a chat client uses.
    """
    return [
        ClientPlan(
            client_id=f"b{i}",
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
        )
        for i in range(n)
    ]


def schedule_for(kind: str, n: int, *, rate: float | None, seed: int) -> list[float]:
    """Turn a flag into arrival offsets."""
    if kind == "burst":
        return burst_arrivals(n)
    if rate is None:
        raise SystemExit(f"--arrivals {kind} needs a --rate")
    if kind == "fixed":
        return fixed_arrivals(n, rate=rate)
    if kind == "poisson":
        return poisson_arrivals(n, rate=rate, seed=seed)
    raise SystemExit(f"unknown arrival process {kind!r}")


def run_one(
    base_url: str,
    plans,
    *,
    arrivals: str,
    rate: float | None,
    seed: int,
    closed: int | None,
    timeout: float,
) -> LoadReport:
    """One measurement: either an offered rate or a fixed number of busy clients."""
    if closed is not None:
        return asyncio.run(run_closed_loop(base_url, plans, concurrency=closed, timeout=timeout))
    offsets = schedule_for(arrivals, len(plans), rate=rate, seed=seed)
    return asyncio.run(
        run_open_loop(base_url, plans, offsets, target_rate=rate, timeout=timeout)
    )


def report_row(report: LoadReport, label: str) -> dict:
    """One line of the sweep table, in the units a reader can compare."""
    return {
        "load": label,
        "offered_rps": "" if report.offered_rate is None else f"{report.offered_rate:.2f}",
        "achieved_rps": round(report.achieved_rate, 3),
        "in_flight": round(report.mean_in_flight, 2),
        "output_tps": round(report.output_tps, 2),
        "ttft_p50_ms": round(report.ttft_p50 * 1e3, 1),
        "ttft_p99_ms": round(report.ttft_p99 * 1e3, 1),
        "itl_p50_ms": round(report.itl_p50 * 1e3, 1),
        "itl_p99_ms": round(report.itl_p99 * 1e3, 1),
        "e2e_p50_s": round(report.e2e_p50, 3),
        "ok": report.n_ok,
        "failed": report.n_failed,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'load':>12} {'achieved':>9} {'in flight':>9} {'tok/s':>8} "
        f"{'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p99':>8} {'fail':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['load']:>12} {r['achieved_rps']:>9.2f} {r['in_flight']:>9.2f} "
            f"{r['output_tps']:>8.1f} {r['ttft_p50_ms']:>8.0f}m {r['ttft_p99_ms']:>8.0f}m "
            f"{r['itl_p50_ms']:>7.0f}m {r['itl_p99_ms']:>7.0f}m {r['failed']:>5}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="benchmark a running nanoserve server")
    target = p.add_argument_group("what to measure")
    target.add_argument("--url", default=None, help="a server that is already running")
    target.add_argument("--weights", default=None, help="start one here, from this dir")
    target.add_argument("--device", default="auto", help="auto | cuda | cpu (with --weights)")
    target.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    target.add_argument("--block-size", type=int, default=16)
    target.add_argument("--max-batch-size", type=int, default=8, help="concurrent slots")
    target.add_argument("--max-model-len", type=int, default=2048)
    target.add_argument("--num-blocks", type=int, default=None, help="skip KV sizing")
    target.add_argument("--kv-cache-bytes", type=int, default=None)

    load = p.add_argument_group("the load")
    load.add_argument("--requests", type=int, default=16, help="requests per measurement")
    load.add_argument("--rate", type=float, default=None, help="offered requests/second")
    load.add_argument("--rates", default=None, help="sweep, e.g. 1,2,4,8")
    load.add_argument(
        "--arrivals",
        default="poisson",
        choices=("poisson", "fixed", "burst"),
        help="poisson is bursty like real traffic; fixed is evenly paced",
    )
    load.add_argument("--closed", type=int, default=None, help="closed loop with N clients")
    load.add_argument("--prompt", default=DEFAULT_PROMPT)
    load.add_argument("--max-tokens", type=int, default=32)
    load.add_argument("--seed", type=int, default=0)
    load.add_argument("--timeout", type=float, default=600.0)
    load.add_argument("--csv", default=None, help="write the table here")
    load.add_argument(
        "--sanity",
        action="store_true",
        help="fail if the run was not actually loaded, or the generator fell behind",
    )
    args = p.parse_args()

    if bool(args.url) == bool(args.weights):
        raise SystemExit("pass exactly one of --url (a running server) or --weights (start one)")
    if args.rates is None and args.rate is None and args.closed is None and args.arrivals != "burst":
        raise SystemExit("pass --rate, --rates, --closed, or --arrivals burst")

    plans = build_plans(args.requests, prompt=args.prompt, max_tokens=args.max_tokens)
    rows: list[dict] = []

    with ExitStack() as stack:
        if args.url:
            base_url = args.url.rstrip("/")
        else:
            from nanoserve.launch import build_app

            print(f"loading {args.weights} ...", file=sys.stderr, flush=True)
            app = build_app(
                args.weights,
                device=args.device,
                dtype=args.dtype,
                block_size=args.block_size,
                max_batch_size=args.max_batch_size,
                max_model_len=args.max_model_len,
                num_blocks=args.num_blocks,
                kv_cache_bytes=args.kv_cache_bytes,
            )
            print(app.state.plan.describe(), file=sys.stderr, flush=True)
            server = stack.enter_context(live_server(app))
            base_url = server.base_url

        loads: list[tuple[str, float | None, int | None]] = []
        if args.rates:
            loads = [(f"{r} rps", float(r), None) for r in args.rates.split(",")]
        elif args.closed is not None:
            loads = [(f"{args.closed} clients", None, args.closed)]
        else:
            loads = [(f"{args.rate} rps" if args.rate else "burst", args.rate, None)]

        for label, rate, closed in loads:
            report = run_one(
                base_url,
                plans,
                arrivals=args.arrivals,
                rate=rate,
                seed=args.seed,
                closed=closed,
                timeout=args.timeout,
            )
            print(f"\n{label}:")
            print(report.summary())
            if report.failures:
                worst = report.failures[0]
                print(f"  first failure  {worst.client_id}: {worst.status} {worst.error}")
            if args.sanity:
                try:
                    check_offered_load(report, min_in_flight=1.5)
                    check_schedule_kept(report, max_lag_s=0.1)
                except MeasurementUnsound as exc:
                    raise SystemExit(f"\nthis measurement does not support its numbers:\n  {exc}")
            rows.append(report_row(report, label))

    if len(rows) > 1:
        print()
        print_table(rows)
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
