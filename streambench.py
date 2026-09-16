"""Day 59: what the batched decode read holds and walks, rectangle against tiles.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python streambench.py
    cd ~/nanoserve && .venv/bin/python streambench.py \
        --rows 8,32,256 --width 2048,8192 --block 128 \
        --csv docs/daily/data/day-59-streambench.csv

No weights and no device: every column here is either exact host arithmetic or a
correctness check on a toy pool, which is the point. The two reads differ in what
they *materialise*, and that is a shape question, so it is answerable on a laptop
to the byte. What is not answerable here is the wall clock, and this file will not
pretend otherwise: `paged_attention_batched_kernel` is a tlsim loop in Python, so
it is slower than the oracle by two orders of magnitude and always will be. The
same loop in Triton is where the time goes, and a card is where that gets measured.

Three tables.

**Agreement.** The kernel and `paged_attention_batched_reference` on the same small
batches, worst absolute difference reported. Streaming reassociates the exponent
sums the way every flash kernel does, so the bar is a few ulps and not equality.

**The workspace**, which is Day 54's number asked twice. The oracle scores into a
`[rows, heads, 1, ctx]` rectangle, and at 256 rows and an 8192-token width that one
intermediate is 268 MB: the largest live tensor a captured decode region holds, the
thing the shared graph pool is sized by, and the reason the capture list has to
bucket the width axis at all. The kernel scores into a `[rows, heads, 1, block]`
tile. The ratio is `context_width / block` exactly, because rows and heads appear
in both and cancel.

**The walk**, which is the other half and the one the arithmetic above cannot see.
The rectangle gives every row the longest row's history, so a batch pays its spread
once per row. The kernel walks `cdiv(len, block)` tiles per row and stops. On a
uniform batch that is a 1.0x saving and the honest report is 1.0x; on the mixed
generation lengths Day 58 measured, it is the spread.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.captured import (
    streamed_workspace_bytes,
    workspace_bytes,
    workspace_saving,
)
from nanoserve.compiled import DecodeShape
from nanoserve.kernels.paged_attention import (
    paged_attention_batched_kernel,
    paged_attention_batched_reference,
    streamed_work,
)

#: A serving-sized head count, the same one `capturebench.py` prices with.
SERVING_HEADS = 32

#: The length spreads a real continuous batch is in, named the way Day 58's
#: coverage table named them: one length is a synthetic best case, and the mixed
#: rows are what arrives when requests finish at different times.
SPREADS: dict[str, tuple[int, ...]] = {
    "uniform": (1,),
    "2x": (1, 2),
    "4x": (1, 2, 4, 8),
    "long tail": (1, 1, 1, 1, 1, 1, 1, 8),
}


def _lengths(rows: int, width: int, spread: tuple[int, ...]) -> list[int]:
    """Per-row context lengths: the spread's ratios scaled so the longest is `width`."""
    top = max(spread)
    return [max(1, width * s // top) for s in (spread * rows)[:rows]]


def agreement(block: int) -> list[dict]:
    """Run both reads on toy batches and report the worst disagreement."""
    cases = {
        "uniform": [[3, 1, 4], [2, 5, 0], [6, 7, 8]],
        "ragged": [[1], [7, 3], [0, 4, 6, 2, 9], [5, 8, 11]],
        "scattered": [[13, 2, 9, 0], [1, 14, 6], [8, 4, 11, 3, 15]],
    }
    torch.manual_seed(0)
    k_pool = torch.randn(16, 2, 8)
    v_pool = torch.randn(16, 2, 8)
    out = []
    for name, rows in cases.items():
        width = max(len(r) for r in rows)
        mapping = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)
        lens = torch.tensor([len(r) for r in rows], dtype=torch.long)
        q = torch.randn(len(rows), 8, 1, 8)
        kernel = paged_attention_batched_kernel(
            q, k_pool, v_pool, mapping, lens, n_rep=4, block=block
        )
        oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep=4)
        delta = float((kernel - oracle).abs().max())
        out.append({"case": name, "rows": len(rows), "max_abs_delta": delta})
        print(
            f"  {name:<10} {len(rows)} rows, widest {width:>2}: "
            f"worst |kernel - oracle| {delta:.2e}",
            file=sys.stderr,
        )
    return out


def sweep(rows_list: list[int], widths: list[int], block: int) -> list[dict]:
    """The workspace and the walk, for every (rows, width, spread) in the sweep."""
    print(
        f"\n  {'rows':>5} {'width':>6} {'spread':>10} {'rectangle':>12} {'tile':>10} "
        f"{'holds':>7} {'walked':>8} {'rect tiles':>11} {'walks':>6}",
        file=sys.stderr,
    )
    out = []
    for rows in rows_list:
        for width in widths:
            shape = DecodeShape(rows=rows, context_width=width)
            rect = workspace_bytes(shape, SERVING_HEADS)
            tile = streamed_workspace_bytes(shape, SERVING_HEADS, block)
            held = workspace_saving(shape, SERVING_HEADS, block)
            for name, spread in SPREADS.items():
                work = streamed_work(_lengths(rows, width, spread), block, context_width=width)
                out.append(
                    {
                        "rows": rows,
                        "context_width": width,
                        "spread": name,
                        "block": block,
                        "rectangle_bytes": rect,
                        "tile_bytes": tile,
                        "workspace_saving": round(held, 2),
                        "tiles": work.tiles,
                        "rectangle_tiles": work.rectangle_tiles,
                        "ragged_saving": round(work.ragged_saving, 2),
                    }
                )
                print(
                    f"  {rows:>5} {width:>6} {name:>10} {rect / 1e6:>9.1f} MB "
                    f"{tile / 1e6:>7.2f} MB {held:>6.0f}x {work.tiles:>8} "
                    f"{work.rectangle_tiles:>11} {work.ragged_saving:>5.2f}x",
                    file=sys.stderr,
                )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rows",
        "context_width",
        "spread",
        "block",
        "rectangle_bytes",
        "tile_bytes",
        "workspace_saving",
        "tiles",
        "rectangle_tiles",
        "ragged_saving",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve streamed paged read benchmark")
    p.add_argument("--rows", type=_int_list, default=[8, 32, 256])
    p.add_argument("--width", type=_int_list, default=[2048, 8192])
    p.add_argument("--block", type=int, default=128, help="the score tile, not block_size")
    p.add_argument("--csv", default="docs/daily/data/day-59-streambench.csv")
    args = p.parse_args()

    print(
        f"the streamed batched read, host only ({SERVING_HEADS} heads, "
        f"{args.block}-key tiles):",
        file=sys.stderr,
    )
    agreement(args.block)
    rows = sweep(args.rows, args.width, args.block)
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
