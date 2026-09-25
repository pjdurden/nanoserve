"""Day 68: how long a request has to be before a split arm tests the split.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python armbench.py
    cd ~/nanoserve && .venv/bin/python armbench.py \\
        --csv docs/daily/data/day-68-armbench.csv

No weights, no device, no server: `plan_splits` and `partition_width` are the two
functions the cache calls at boot, asked here over the configurations this repo has
actually run a split on. Arithmetic and nothing else.

**One row per configuration.** The columns are the split count the planner picks,
the chunk it cuts the width into, and the merge floor: the shortest request, prompt
plus completion, whose last decode read reaches a second chunk. Below the floor a
split server runs the streamed loop plus a reduce over one live partial, which is
the identity, so its answers can't disagree with the rectangle's whatever the
combine does. `-` means the arena has one chunk and no request the server accepts
crosses it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.buckets import row_axis
from nanoserve.graphbench import split_merge_floor
from nanoserve.kernels.flash_decoding import partition_width, plan_splits

#: (label, slots, max_model_len, query heads, tile). The first two are the toy the
#: acceptance run boots; the last is the deployment every table this week uses.
CONFIGS = (
    ("toy, Day 60 width", 4, 32, 8, 16),
    ("toy, Day 68 width", 4, 1024, 8, 16),
    ("serving", 256, 8192, 32, 32),
)

#: The acceptance run's two request sets, as the longest (prompt, completion) each has.
REQUESTS = (("PLANS", 5, 16), ("SPLIT_PLANS", 510, 12))


def row(label: str, slots: int, width: int, heads: int, block: int) -> dict:
    splits = plan_splits(row_axis(slots), heads, width, block)
    count, chunk = partition_width(width, block, splits=splits)
    workspace = {"splits": count, "keys_per_split": chunk}
    floor = split_merge_floor(workspace)
    out = {
        "config": label,
        "slots": slots,
        "width": width,
        "heads": heads,
        "block": block,
        "splits": count,
        "keys_per_split": chunk,
        "merge_floor": floor if floor is not None else "",
    }
    for name, prompt, completion in REQUESTS:
        read = prompt + completion - 1
        out[f"{name}_crosses"] = bool(count > 1 and read > chunk and read < width)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = [row(*c) for c in CONFIGS]
    print(
        f"\n  {'config':<18} {'slots':>5} {'width':>6} {'splits':>6} {'chunk':>6} "
        f"{'merge floor':>11}  {'PLANS':>6}  {'SPLIT_PLANS':>11}",
        file=sys.stderr,
    )
    for r in rows:
        floor = r["merge_floor"] if r["merge_floor"] != "" else "-"
        print(
            f"  {r['config']:<18} {r['slots']:>5} {r['width']:>6} {r['splits']:>6} "
            f"{r['keys_per_split']:>6} {floor:>11}  "
            f"{'yes' if r['PLANS_crosses'] else 'no':>6}  "
            f"{'yes' if r['SPLIT_PLANS_crosses'] else 'no':>11}",
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
