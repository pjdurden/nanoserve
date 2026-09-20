"""Day 63: what a split buys, what it wastes, and what it costs to store.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python splitbench.py
    cd ~/nanoserve && .venv/bin/python splitbench.py \
        --block 32 --csv docs/daily/data/day-63-splitbench.csv

No weights and no device. Everything below is arithmetic about a launch, plus one
agreement check that runs the split dispatcher against its oracle on toy pools, so a
box with no card can say whether the read is right and how much of Day 62's measured
imbalance a split could recover.

**The first table is the agreement check, and it names the backend and the split
count.** The split is a performance knob: every value returns the same attention to a
few ulps, which is what the deltas here check, and none of them says anything about
speed, which is what nothing here can check.

**The second table is the trade, and it has three columns because a win with two
costs is not a win yet.**

* `wave` is `unsplit_tail / tail`, what a fully resident grid gets back. It is the
  whole point, and it is capped by how many tiles the *longest row* holds rather than
  by the split count: cutting a wide mapping 32 ways does nothing for a batch whose
  longest row is two tiles, because the other thirty chunks are empty.
* `idle` is the share of the grid that walks no tiles at all. A split is an axis, so
  every row gets `splits` programs whether its history reaches them or not, and on a
  long-tail batch most of them do not. This is the column that says a split is not
  free even when it is right.
* `MiB` is the partial workspace, `[rows, heads, splits]` of a max, a denominator and
  a `head_dim`-wide accumulator, in fp32. Day 59 took the 268 MB score rectangle
  away; this is the first thing since that gives memory back, and it grows linearly
  in exactly the number the tail shrinks by.

`imbal` is Day 62's imbalance recomputed on the new grid, and it does not go to 1.00x.
Cutting the width uniformly does not cut a ragged batch uniformly: the short rows'
chunks are empty rather than short, so the spread moves out of the tail and into the
idle count. The two columns together are the honest statement of what happened.

**The third table is `choose_splits` over the batch sizes a server actually sees**,
and it is the reason vLLM ships both kernels rather than replacing one. A long context
on a small batch is 32 programs on a card that holds thousands, and the split is the
only thing that fills it. A 256-row batch is already 8192 programs, the hardware is a
queue, and a split there is a workspace and a second pass bought for nothing.

The split count in every row of this file comes from the mapping's *width* and never
from the batch's longest row. The width is a shape the caller already has; the longest
row is a number inside a tensor, and asking for it is Day 48's synchronisation, Day
49's graph break, and a grid that changes every step, which is Day 61's capture list
all over again.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.kernels.flash_decoding import (
    DEFAULT_PARTITION,
    DEFAULT_TARGET_PROGRAMS,
    choose_splits,
    paged_attention_split,
    split_plan,
)
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.kernels.triton_batched_attention import launch_work, select_backend

#: A serving-sized head count, the same one `gridbench.py` prices with. It multiplies
#: the program count and the workspace and cancels out of every ratio.
SERVING_HEADS = 32

#: A serving-sized head width, for the workspace column only.
SERVING_HEAD_DIM = 128

#: The length spreads a real continuous batch is in, named as in `gridbench.py`, and
#: written longest-first so the batch's longest row is always the mapping's width.
#: That is the case Day 61's bucket set is *not*, and the difference is the day's
#: sharpest column: a chunk of the width is only work if a row reaches into it.
SPREADS: dict[str, tuple[int, ...]] = {
    "uniform": (1,),
    "4x": (8, 4, 2, 1),
    "long tail": (8, 1, 1, 1, 1, 1, 1, 1),
}

#: Split counts to sweep. 1 is Day 62's launch, and it is in the table so every other
#: row has something to be a ratio of.
SPLITS = (1, 2, 4, 8, 16)


def _lengths(rows: int, longest: int, spread: tuple[int, ...]) -> list[int]:
    """Per-row context lengths: the spread's ratios scaled so the longest is `longest`."""
    top = max(spread)
    return [max(1, longest * s // top) for s in (spread * rows)[:rows]]


def agreement(block: int) -> list[dict]:
    """Run the split dispatcher against the oracle, at several split counts."""
    rows = [[1], [7, 3], [0, 4, 6, 2, 9, 12, 5], [5, 8, 11]]
    width = max(len(r) for r in rows)
    mapping = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)
    lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
    torch.manual_seed(0)
    k_pool = torch.randn(16, 2, 8)
    v_pool = torch.randn(16, 2, 8)
    q = torch.randn(len(rows), 8, 1, 8)
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep=4)
    out = []
    for splits in (1, 2, 3, 7):
        got = paged_attention_split(
            q, k_pool, v_pool, mapping, lens, n_rep=4, block=2, splits=splits
        )
        delta = float((got - oracle).abs().max())
        plan = split_plan(lens, n_q=8, head_dim=8, block=2, splits=splits, context_width=width)
        out.append({"splits_asked": splits, "splits_used": plan.splits, "max_abs_delta": delta})
        print(
            f"  asked {splits} splits, launched {plan.splits} of "
            f"{plan.keys_per_split} keys: worst |split - oracle| {delta:.2e}",
            file=sys.stderr,
        )
    return out


def sweep(rows_list: list[int], width: int, block: int) -> list[dict]:
    """The trade, for every (rows, spread, splits)."""
    print(
        f"\n  {'rows':>5} {'spread':>10} {'splits':>7} {'chunk':>6} {'programs':>9} "
        f"{'tail':>6} {'wave':>7} {'idle':>6} {'imbal':>7} {'MiB':>8}",
        file=sys.stderr,
    )
    out = []
    for rows in rows_list:
        # A one-row batch has no spread: every named spread describes the same batch.
        spreads = {"uniform": SPREADS["uniform"]} if rows == 1 else SPREADS
        for spread_name, spread in spreads.items():
            lens = _lengths(rows, width, spread)
            unsplit = launch_work(lens, SERVING_HEADS, block, context_width=width)
            for splits in SPLITS:
                plan = split_plan(
                    lens,
                    n_q=SERVING_HEADS,
                    head_dim=SERVING_HEAD_DIM,
                    block=block,
                    splits=splits,
                    context_width=width,
                )
                assert plan.tiles == unsplit.tiles  # a split changes when, never how much
                out.append(
                    {
                        "rows": rows,
                        "context_width": width,
                        "spread": spread_name,
                        "block": block,
                        "heads": SERVING_HEADS,
                        "head_dim": SERVING_HEAD_DIM,
                        "splits": plan.splits,
                        "keys_per_split": plan.keys_per_split,
                        "programs": plan.programs,
                        "tiles": plan.tiles,
                        "tail_tiles": plan.tail_tiles,
                        "unsplit_tail_tiles": plan.unsplit_tail_tiles,
                        "wave_speedup": round(plan.wave_speedup, 2),
                        "idle_fraction": round(plan.idle_fraction, 3),
                        "imbalance": round(plan.imbalance, 2),
                        "partial_mib": round(plan.partial_mib, 2),
                    }
                )
                print(
                    f"  {rows:>5} {spread_name:>10} {plan.splits:>7} "
                    f"{plan.keys_per_split:>6} {plan.programs:>9} {plan.tail_tiles:>6} "
                    f"{plan.wave_speedup:>6.2f}x {plan.idle_fraction:>5.0%} "
                    f"{plan.imbalance:>6.2f}x {plan.partial_mib:>8.2f}",
                    file=sys.stderr,
                )
    return out


def chosen(rows_list: list[int], widths: list[int], block: int) -> None:
    """What `choose_splits` picks, and why it picks 1 for a batch that is already full."""
    print(
        f"\n  choose_splits at {SERVING_HEADS} heads "
        f"(target {DEFAULT_TARGET_PROGRAMS} programs, {DEFAULT_PARTITION}-key floor):",
        file=sys.stderr,
    )
    print(f"  {'rows':>5} {'unsplit':>8} " + "".join(f"{w:>9}" for w in widths), file=sys.stderr)
    for rows in rows_list:
        picks = "".join(
            f"{choose_splits(rows, SERVING_HEADS, w, block):>9}" for w in widths
        )
        print(f"  {rows:>5} {rows * SERVING_HEADS:>8} {picks}", file=sys.stderr)
    print(
        "  a long context on a small batch is what a split is for; a 256-row batch is\n"
        "  already 8192 programs, so the card is a queue and a split is pure overhead.",
        file=sys.stderr,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rows",
        "context_width",
        "spread",
        "block",
        "heads",
        "head_dim",
        "splits",
        "keys_per_split",
        "programs",
        "tiles",
        "tail_tiles",
        "unsplit_tail_tiles",
        "wave_speedup",
        "idle_fraction",
        "imbalance",
        "partial_mib",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve flash-decoding split benchmark")
    p.add_argument("--rows", type=_int_list, default=[1, 8, 32])
    p.add_argument("--width", type=int, default=8192)
    p.add_argument("--block", type=int, default=32, help="the score tile, not block_size")
    p.add_argument("--csv", default="docs/daily/data/day-63-splitbench.csv")
    args = p.parse_args()

    backend = select_backend(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"the split decode read on this box: backend {backend} "
        f"({SERVING_HEADS} heads, {args.block}-key tiles, {args.width}-wide mapping)",
        file=sys.stderr,
    )
    agreement(args.block)
    rows = sweep(args.rows, args.width, args.block)
    chosen([1, 8, 32, 256], [1024, 4096, 8192, 32768], args.block)
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
