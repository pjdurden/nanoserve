"""Day 62: what the batched kernel's grid asks for, and what a grid can give back.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python gridbench.py
    cd ~/nanoserve && .venv/bin/python gridbench.py \
        --block 32 --csv docs/daily/data/day-62-gridbench.csv

No weights and no device. Everything below is arithmetic about a launch, plus one
agreement check that runs the dispatcher against its oracle on toy pools, so a box
with no card can still say whether the read is right and how much of its advertised
saving a real grid could collect.

**The first table names the backend, and that is the point of it.** Day 62 makes
`STREAMED` a dispatch: a CUDA tensor gets the `triton.jit` kernel and everything else
gets Day 59's tlsim loop. Both compute the same attention to a few ulps, which is
what the deltas here check, and they differ in speed by whatever a card is worth,
which is what nothing here can check. So the header prints which one ran rather than
letting a clean agreement table imply the fast one.

**The second table is the day's honest caveat, and it has two savings in it because
one number was never enough.** Day 59's `ragged_saving` counts tiles a streamed read
skips, and a skipped tile is only a saved second if the machine was going to be busy
anyway.

* `work` is `rectangle_tiles / tiles`, the sum. It is what an oversubscribed grid
  collects: when there are far more programs than the card holds at once, the
  hardware is a queue and tiles not walked are waves not run.
* `wave` is `cdiv(width, block) / tail_tiles`, the max. It is what a grid small
  enough to be fully resident collects: every program is in flight, so the launch
  ends when its slowest program does, and the only thing saved is the gap between the
  longest row and the width the mapping was handed.
* `imbalance` is `tail_tiles / mean_tiles`, and it is why the two differ. It rises
  with exactly the length spread that makes `work` rise, so on any batch worth
  streaming the two numbers pull apart.

The `width` column is the mapping's, not the batch's, and that is the load-bearing
choice. Under Day 61's streamed bucket set the width is `max_model_len` on every step
forever, so a batch whose longest row is 512 tokens is compared against the 8192-wide
rectangle a rectangle read would really have been handed. Against a *tight* mapping,
one as wide as its own longest row, `wave` is exactly 1.00x for every batch in this
file, which is the floor and is printed as its own column so nobody has to take it on
faith: a ragged read on a resident grid saves a great deal of memory and no time.

The fix for the imbalance is not in this repo. vLLM and SGLang split one long
sequence across several programs and reduce their partial softmaxes afterwards
(flash-decoding), which turns the tail into more parallelism instead of a longer
wait. `imbalance` is the column that says how much that would be worth here.

**The third table is the address ceiling**, which is `slot_index_dtype`'s whole
reason. The kernel forms `slot * channels + kv_head * head_dim + d`, and in int32
that multiply wraps into a negative offset that is a perfectly legal offset
somewhere else. The ceiling is on *elements*, so in bytes it is the same number for
every head geometry, and it is per layer because this repo allocates one pool per
layer. That is why it is not reachable on hardware that exists, and why the guard is
still worth a comparison: it is a property of the geometry, not of this box.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.kernels.triton_batched_attention import (
    INT32_MAX,
    launch_work,
    paged_attention_batched,
    select_backend,
    slot_index_dtype,
)
from nanoserve.kernels.triton_batched_attention import (
    BatchedGeometry as Geometry,
)

#: A serving-sized head count, the same one `streambench.py` and `capturebench.py`
#: price with. It multiplies both sides of every ratio below and cancels out of all
#: of them, which is exactly why the program count is worth printing next to them.
SERVING_HEADS = 32

#: The length spreads a real continuous batch is in, named as in `streambench.py`.
SPREADS: dict[str, tuple[int, ...]] = {
    "uniform": (1,),
    "2x": (1, 2),
    "4x": (1, 2, 4, 8),
    "long tail": (1, 1, 1, 1, 1, 1, 1, 8),
}

#: How full the batch's longest row is, as a share of the mapping's width. Under a
#: streamed bucket set the width is `max_model_len` and the batch is whatever arrived,
#: so this is the axis that decides whether `wave` is a real number or a 1.00x.
FILLS: dict[str, float] = {"early": 1 / 16, "half": 1 / 2}

#: Head geometries the address ceiling is worth reading for: a GQA model, a
#: multi-head one, and a single compact head with a wide latent.
GEOMETRIES = ((8, 128), (32, 128), (1, 576))


def _lengths(rows: int, longest: int, spread: tuple[int, ...]) -> list[int]:
    """Per-row context lengths: the spread's ratios scaled so the longest is `longest`."""
    top = max(spread)
    return [max(1, longest * s // top) for s in (spread * rows)[:rows]]


def agreement(block: int) -> list[dict]:
    """Run the dispatcher against the oracle on toy batches, and name the backend."""
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
        got = paged_attention_batched(q, k_pool, v_pool, mapping, lens, n_rep=4, block=block)
        oracle = paged_attention_batched_reference(q, k_pool, v_pool, mapping, lens, n_rep=4)
        delta = float((got - oracle).abs().max())
        out.append({"case": name, "rows": len(rows), "max_abs_delta": delta})
        print(
            f"  {name:<10} {len(rows)} rows, widest {width:>2}: "
            f"worst |dispatch - oracle| {delta:.2e}",
            file=sys.stderr,
        )
    return out


def sweep(rows_list: list[int], widths: list[int], block: int) -> list[dict]:
    """The grid, and the two savings, for every (rows, width, fill, spread)."""
    print(
        f"\n  {'rows':>5} {'width':>6} {'longest':>8} {'spread':>10} {'programs':>9} "
        f"{'tiles':>8} {'tail':>6} {'work':>7} {'wave':>8} {'tight':>6} {'imbal':>6}",
        file=sys.stderr,
    )
    out = []
    for rows in rows_list:
        for width in widths:
            for fill_name, fill in FILLS.items():
                longest = max(1, int(width * fill))
                for spread_name, spread in SPREADS.items():
                    lens = _lengths(rows, longest, spread)
                    # Against the mapping the bucket set really hands the read...
                    bucketed = launch_work(lens, SERVING_HEADS, block, context_width=width)
                    # ...and against one as wide as this batch's own longest row.
                    tight = launch_work(lens, SERVING_HEADS, block)
                    out.append(
                        {
                            "rows": rows,
                            "context_width": width,
                            "fill": fill_name,
                            "longest": longest,
                            "spread": spread_name,
                            "block": block,
                            "heads": SERVING_HEADS,
                            "programs": bucketed.programs,
                            "tiles": bucketed.tiles,
                            "tail_tiles": bucketed.tail_tiles,
                            "rectangle_tiles": bucketed.rectangle_tiles,
                            "work_saving": round(bucketed.work_saving, 2),
                            "wave_saving": round(bucketed.wave_saving, 2),
                            "tight_wave_saving": round(tight.wave_saving, 2),
                            "imbalance": round(bucketed.imbalance, 2),
                        }
                    )
                    print(
                        f"  {rows:>5} {width:>6} {longest:>8} {spread_name:>10} "
                        f"{bucketed.programs:>9} {bucketed.tiles:>8} "
                        f"{bucketed.tail_tiles:>6} {bucketed.work_saving:>6.2f}x "
                        f"{bucketed.wave_saving:>7.2f}x {tight.wave_saving:>5.2f}x "
                        f"{bucketed.imbalance:>5.2f}x",
                        file=sys.stderr,
                    )
    return out


def addresses() -> None:
    """Where the kernel's pointer arithmetic outgrows a signed 32-bit integer."""
    print(
        f"\n  the int32 address ceiling, one layer's pool ({INT32_MAX + 1:,} elements):",
        file=sys.stderr,
    )
    print(
        f"  {'n_kv':>5} {'head_dim':>9} {'channels':>9} {'slot ceiling':>14} "
        f"{'fp16 / pool':>12} {'index':>7}",
        file=sys.stderr,
    )
    for n_kv, head_dim in GEOMETRIES:
        channels = n_kv * head_dim
        ceiling = (INT32_MAX + 1) // channels
        at_ceiling = Geometry(
            batch=1,
            n_q=n_kv,
            n_kv=n_kv,
            head_dim=head_dim,
            num_slots=ceiling,
            channels=channels,
            max_ctx=1,
            stride_q_row=channels,
        )
        just_past = Geometry(**{**at_ceiling.__dict__, "num_slots": ceiling + 1})
        gib = ceiling * channels * 2 / 2**30  # one pool, fp16
        print(
            f"  {n_kv:>5} {head_dim:>9} {channels:>9} {ceiling:>14,} "
            f"{gib:>9.2f} GiB {str(slot_index_dtype(at_ceiling)).split('.')[-1]:>7}",
            file=sys.stderr,
        )
        assert slot_index_dtype(just_past) is torch.int64  # the boundary is exact
    print(
        "  the byte ceiling is the same for every geometry because the limit is on\n"
        "  elements, and it is per layer because the pool is per layer: that is why\n"
        "  int32 holds on any card today, and why the check is one comparison anyway.",
        file=sys.stderr,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rows",
        "context_width",
        "fill",
        "longest",
        "spread",
        "block",
        "heads",
        "programs",
        "tiles",
        "tail_tiles",
        "rectangle_tiles",
        "work_saving",
        "wave_saving",
        "tight_wave_saving",
        "imbalance",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve batched paged-attention grid benchmark")
    p.add_argument("--rows", type=_int_list, default=[8, 32, 256])
    p.add_argument("--width", type=_int_list, default=[8192])
    p.add_argument("--block", type=int, default=32, help="the score tile, not block_size")
    p.add_argument("--csv", default="docs/daily/data/day-62-gridbench.csv")
    args = p.parse_args()

    backend = select_backend(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"the streamed decode read on this box: backend {backend} "
        f"({SERVING_HEADS} heads, {args.block}-key tiles)",
        file=sys.stderr,
    )
    agreement(args.block)
    rows = sweep(args.rows, args.width, args.block)
    addresses()
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
