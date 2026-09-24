"""Day 67: who owns the split's partials, at serving size, before and after.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python ownerbench.py
    cd ~/nanoserve && .venv/bin/python ownerbench.py \
        --csv docs/daily/data/day-67-ownerbench.csv

No weights, no device, no engine: `CapturePlan` is a record and every number here
is a property of one, built by hand over the deployment the week's tables use
(256 slots, 8192 tokens, 32 heads of 128 channels). Arithmetic and nothing else.

**One row per `--warm-rows`.** A split server reserves two things beyond the pool:
the capture's graph pool, and the partials arena. The columns are what the boot log
printed for each, what a reader summing the two lines got, what the budget probe was
asked to hold, and what the process actually holds. On Day 66 those last three were
three different numbers. On Day 67 they are one.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.buckets import DecodeBuckets
from nanoserve.captured import (
    shared_pool_bytes,
    split_partial_bytes,
    split_tile_bytes,
    split_workspace_bytes,
)
from nanoserve.compiled import DecodeShape
from nanoserve.kernels.flash_decoding import plan_splits
from nanoserve.kernels.triton_batched_attention import DEFAULT_BLOCK
from nanoserve.launch import CapturePlan

#: The deployment every arithmetic table this week is stated over.
SERVING_ROWS = 256
SERVING_LEN = 8192
SERVING_HEADS = 32
SERVING_HEAD_DIM = 128

WARM_ROWS = (256, 64, 8, 1)


def _mb(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def plan_for(warm_rows: int, buckets: DecodeBuckets, splits: int) -> CapturePlan:
    """What `plan_capture` builds for a split server started with `--warm-rows`."""
    rows = [r for r in buckets.rows if r <= warm_rows]
    return CapturePlan(
        shapes=tuple(
            DecodeShape(rows=r, context_width=SERVING_LEN) for r in sorted(rows, reverse=True)
        ),
        max_rows=max(rows),
        max_width=SERVING_LEN,
        width_bound_by="read",
        num_heads=SERVING_HEADS,
        full_count=len(buckets.shapes),
        block=buckets.block,
        splits=splits,
        head_dim=SERVING_HEAD_DIM,
        arena_rows=SERVING_ROWS,
    )


def row(warm_rows: int, buckets: DecodeBuckets, splits: int) -> dict:
    capture = plan_for(warm_rows, buckets, splits)
    kw = dict(block=capture.block, splits=splits, head_dim=SERVING_HEAD_DIM)
    widest = capture.widest
    arena = DecodeShape(rows=SERVING_ROWS, context_width=SERVING_LEN)

    # Day 66's accounting, reproduced from the functions it called.
    old_line = shared_pool_bytes(capture.shapes, SERVING_HEADS, **kw)
    old_probe = split_workspace_bytes(
        widest, SERVING_HEADS, capture.block, splits, SERVING_HEAD_DIM
    )
    arena_line = split_partial_bytes(arena, SERVING_HEADS, splits, SERVING_HEAD_DIM)
    held = split_tile_bytes(widest, SERVING_HEADS, capture.block, splits) + arena_line

    return {
        "warm_rows": warm_rows,
        "splits": splits,
        "day66_capture_line": old_line,
        "day66_lines_summed": old_line + arena_line,
        "day66_probe_asked": old_probe,
        "day67_capture_line": capture.pool_bytes,
        "arena_line": arena_line,
        "day67_lines_summed": capture.pool_bytes + arena_line,
        "day67_probe_asked": capture.reserved_bytes,
        "held": held,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    buckets = DecodeBuckets(SERVING_ROWS, SERVING_LEN, streamed=True, block=DEFAULT_BLOCK)
    splits = plan_splits(buckets.rows, SERVING_HEADS, SERVING_LEN, DEFAULT_BLOCK)
    rows = [row(w, buckets, splits) for w in WARM_ROWS]

    print(
        f"\n  a {splits}-way split server, {SERVING_ROWS} slots, {SERVING_LEN} tokens, "
        f"{SERVING_HEADS} heads x {SERVING_HEAD_DIM}, {DEFAULT_BLOCK}-key tiles:",
        file=sys.stderr,
    )
    print(
        f"  {'warm-rows':>9}  {'':>6}  {'capture line':>12}  {'arena line':>11}  "
        f"{'lines summed':>12}  {'probe asked':>12}  {'held':>10}",
        file=sys.stderr,
    )
    for r in rows:
        for day in ("day66", "day67"):
            print(
                f"  {r['warm_rows']:>9}  {day:>6}  {_mb(r[f'{day}_capture_line']):>12}  "
                f"{_mb(r['arena_line']):>11}  {_mb(r[f'{day}_lines_summed']):>12}  "
                f"{_mb(r[f'{day}_probe_asked']):>12}  {_mb(r['held']):>10}",
                file=sys.stderr,
            )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
