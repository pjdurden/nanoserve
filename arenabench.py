"""Day 64: what the split read's arena costs once somebody prices it.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python arenabench.py
    cd ~/nanoserve && .venv/bin/python arenabench.py \
        --block 32 --csv docs/daily/data/day-64-arenabench.csv

No weights and no device. Everything below is arithmetic over a capture list, plus
one agreement check that reads through a real `SplitWorkspace` on toy pools, so a box
with no card can say whether the planned buffer computes the same attention the
kernel computed when it allocated its own.

**The first table is the agreement check and the address check, and the second one
is the whole reason for the day.** A captured region replays a launch bound to the
pointers it was recorded with. A read that allocates cannot promise that; a read over
a planned buffer can, and the promise is a tuple of three integers that does not move
across a thousand steps.

**The second table is the arena in each of the three reads' currencies.** Day 54
priced the score rectangle at 268 MB and Day 61 took it to a tile. The split read has
been running against an arena nobody charged for since Day 63, and this is that
number: the tiles, once per chunk per row, plus a max, a denominator and a
`head_dim`-wide accumulator per program, in fp32 whatever the pool holds.

**The third table is the surplus, and it is the finding I did not expect.** A capture
list is a graph per shape against one arena, so the split count has to be one number
for the list, and `plan_splits` takes the max over the row buckets. The max lands on
the *narrowest* batch, because a small grid is what a split is for. The arena lands
on the *widest*, because `rows * heads * splits * (head_dim + 2)` is dominated by the
rows once `choose_splits` has bottomed out at 1. So the two maxima are at opposite
ends of the same list, and one rectangular arena is the product of both of them while
no single shape in the list needs more than the max of the product.

That surplus is not free and it is not a bug either. The alternative is a split count
per row bucket, which is a different `tl.constexpr` per bucket, which is a compiled
kernel per bucket: the specialisation Day 61 spent a day collapsing, bought back on a
different axis. One compiled body against a fatter arena is the trade this file
measures, and it is stated rather than hidden.

**The fourth table is the allocator.** Three `torch.empty` calls per read per layer
per step is the thing a planned buffer removes, and on a 16-layer model at 40 tokens
a second it is a number worth writing down once.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.buckets import DecodeBuckets
from nanoserve.captured import (
    shared_pool_bytes,
    split_workspace_bytes,
)
from nanoserve.compiled import DecodeShape
from nanoserve.kernels.flash_decoding import choose_splits, plan_splits
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.partials import allocate_partials

#: The deployment every arithmetic table in this week is stated over, so a number
#: here reads next to Day 54's, Day 61's and Day 63's without converting anything.
SERVING_ROWS = 256
SERVING_LEN = 8192
SERVING_HEADS = 32
SERVING_HEAD_DIM = 128

#: Llama-3.2-1B's depth. The multiplier on everything the allocator is asked for,
#: because the read happens once per layer and the workspace is per read.
SERVING_LAYERS = 16

#: The width multiple Day 52's rectangle set rounds to, kept only so the rectangle
#: arm of the second table is the list Day 54 actually priced.
RECTANGLE_MULTIPLE = 2048


#: The tile the agreement check folds. Two keys, not the serving 32, because the toy
#: mapping below is seven wide: a tile wider than the mapping is one tile, and one
#: tile is one chunk however many splits you ask for, so the check would compare the
#: same launch to itself three times.
AGREEMENT_BLOCK = 2


def agreement() -> None:
    """Read through a planned workspace and check the answer and the addresses."""
    block = AGREEMENT_BLOCK
    rows = [[1], [7, 3], [0, 4, 6, 2, 9, 12, 5], [5, 8, 11]]
    width = max(len(r) for r in rows)
    mapping = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)
    lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
    torch.manual_seed(0)
    k_pool = torch.randn(16, 2, 8)
    v_pool = torch.randn(16, 2, 8)
    q = torch.randn(len(rows), 8, 1, 8)
    oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep=4)

    print(
        f"\n  a planned arena against the oracle ({width}-wide mapping, "
        f"{block}-key tiles):",
        file=sys.stderr,
    )
    for splits in (1, 2, 4):
        workspace = allocate_partials(
            max_rows=8,
            n_q=8,
            head_dim=8,
            context_width=width,
            block=block,
            splits=splits,
        )
        before = workspace.addresses
        worst = 0.0
        for _ in range(64):
            got = workspace.read(q, k_pool, v_pool, mapping, lens, n_rep=4)
            worst = max(worst, float((got - oracle).abs().max()))
        moved = workspace.addresses != before
        print(
            f"    {workspace.splits} splits of {workspace.keys_per_split} keys, "
            f"{workspace.mib:.4f} MiB: worst |planned - oracle| {worst:.2e} over 64 "
            f"reads, addresses moved: {moved}",
            file=sys.stderr,
        )


def by_read(width: int, block: int) -> None:
    """The shared arena in each read's currency, over the list each one implies."""
    rectangle = DecodeBuckets(SERVING_ROWS, width, width_multiple=RECTANGLE_MULTIPLE)
    streamed = DecodeBuckets(SERVING_ROWS, width, streamed=True, block=block)
    splits = plan_splits(streamed.rows, SERVING_HEADS, width, block)

    rect_bytes = shared_pool_bytes(rectangle.shapes, SERVING_HEADS)
    tile_bytes = shared_pool_bytes(streamed.shapes, SERVING_HEADS, block=block)
    split_bytes = shared_pool_bytes(
        streamed.shapes,
        SERVING_HEADS,
        block=block,
        splits=splits,
        head_dim=SERVING_HEAD_DIM,
    )

    print(
        f"\n  the shared capture arena at {SERVING_ROWS} rows, {SERVING_HEADS} heads, "
        f"{width} tokens, {block}-key tiles:",
        file=sys.stderr,
    )
    print(
        f"  {'read':>18} {'shapes':>7} {'arena':>14} {'vs rectangle':>13}",
        file=sys.stderr,
    )
    for name, shapes, size in (
        ("rectangle", len(rectangle.shapes), rect_bytes),
        ("streamed", len(streamed.shapes), tile_bytes),
        (f"split ({splits}-way)", len(streamed.shapes), split_bytes),
    ):
        print(
            f"  {name:>18} {shapes:>7} {size / 1e6:>11.2f} MB "
            f"{rect_bytes / size:>12.1f}x",
            file=sys.stderr,
        )
    print(
        f"  Day 61's plan would have reported {tile_bytes / 1e6:.2f} MB for a process "
        f"running the split\n  read, which spends {split_bytes / 1e6:.2f} MB: an "
        f"under-report of {split_bytes / tile_bytes:.0f}x, and the allocator was the "
        "only thing that knew.",
        file=sys.stderr,
    )


def surplus(width: int, block: int) -> list[dict]:
    """Per-bucket splits, per-shape need, and what one rectangular arena charges."""
    buckets = DecodeBuckets(SERVING_ROWS, width, streamed=True, block=block)
    chosen = plan_splits(buckets.rows, SERVING_HEADS, width, block)
    kw = dict(block=block, head_dim=SERVING_HEAD_DIM)

    print(
        f"\n  one arena over a {len(buckets.rows)}-bucket list, "
        f"{chosen} splits of {width // chosen} keys:",
        file=sys.stderr,
    )
    print(
        f"  {'rows':>5} {'wants':>6} {'programs':>9} {'this shape':>12} {'charged':>10}",
        file=sys.stderr,
    )
    out = []
    peak_need = 0
    for rows in buckets.rows:
        shape = DecodeShape(rows=rows, context_width=width)
        wants = choose_splits(rows, SERVING_HEADS, width, block)
        need = split_workspace_bytes(shape, SERVING_HEADS, splits=wants, **kw)
        charged = split_workspace_bytes(shape, SERVING_HEADS, splits=chosen, **kw)
        peak_need = max(peak_need, need)
        out.append(
            {
                "rows": rows,
                "context_width": width,
                "block": block,
                "heads": SERVING_HEADS,
                "head_dim": SERVING_HEAD_DIM,
                "wants_splits": wants,
                "list_splits": chosen,
                "programs": rows * SERVING_HEADS * wants,
                "shape_bytes": need,
                "charged_bytes": charged,
            }
        )
        print(
            f"  {rows:>5} {wants:>6} {rows * SERVING_HEADS * wants:>9} "
            f"{need / 1e6:>9.2f} MB {charged / 1e6:>7.2f} MB",
            file=sys.stderr,
        )
    arena = max(r["charged_bytes"] for r in out)
    print(
        f"  the widest shape needs {peak_need / 1e6:.2f} MB and the arena is "
        f"{arena / 1e6:.2f} MB: {arena / peak_need:.0f}x, because the split count "
        f"peaks at\n  {min(buckets.rows)} rows and the arena peaks at "
        f"{max(buckets.rows)}. The alternative is a constexpr per bucket, which is a\n"
        "  compiled kernel per bucket, which is the specialisation Day 61 collapsed.",
        file=sys.stderr,
    )
    return out


def allocations(steps: int, layers: int) -> None:
    """What the allocator is asked for, per step and over a run, either way."""
    per_step = 3 * layers
    print(
        f"\n  allocator calls for the partials, {layers} layers, {steps} decode steps:",
        file=sys.stderr,
    )
    print(f"  {'owner':>18} {'per step':>9} {'per run':>10}", file=sys.stderr)
    print(f"  {'the read':>18} {per_step:>9} {per_step * steps:>10}", file=sys.stderr)
    print(f"  {'the plan':>18} {3:>9} {3:>10}", file=sys.stderr)
    print(
        "  the arena is the same bytes either way. What the plan buys is that they "
        "are\n  the same bytes at the same address, which is the only form a graph "
        "can replay.",
        file=sys.stderr,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rows",
        "context_width",
        "block",
        "heads",
        "head_dim",
        "wants_splits",
        "list_splits",
        "programs",
        "shape_bytes",
        "charged_bytes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve split workspace arena benchmark")
    p.add_argument("--width", type=int, default=SERVING_LEN)
    p.add_argument("--block", type=int, default=32, help="the score tile, not block_size")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--layers", type=int, default=SERVING_LAYERS)
    p.add_argument("--csv", default="docs/daily/data/day-64-arenabench.csv")
    args = p.parse_args()

    print(
        f"the split read's workspace, priced: {SERVING_HEADS} heads x "
        f"{SERVING_HEAD_DIM} channels, {args.width}-wide mapping",
        file=sys.stderr,
    )
    agreement()
    by_read(args.width, args.block)
    rows = surplus(args.width, args.block)
    allocations(args.steps, args.layers)
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
