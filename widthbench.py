"""Day 61: the capture list with the width axis in it, and without.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python widthbench.py
    cd ~/nanoserve && .venv/bin/python widthbench.py \
        --block 32 --csv docs/daily/data/day-61-widthbench.csv

No weights and no device. The first table is arithmetic over deployments somebody
would actually type; the second is a measured boot of a toy model down the same path
`serve.py` takes, so the startup seconds are seconds and not a multiplication.

Three columns carry the day.

**`shapes`** is `len(rows) * len(widths)`. Under the rectangle read that product is
the whole problem Day 52 left open: 256 slots and 8192 tokens on a 128-token multiple
is 576 graphs, which is a closed set and a compile bill. Under a streamed read the
second factor is 1, because a read that walks `cdiv(context_lens[row], block)` tiles
of its own row does not care how wide the mapping it was handed is. What is left is
the row axis, which is what vLLM's capture list has always been.

**`arena`** is the shared pool, and it does not shrink because the list did. It
shrinks because the thing being sized changed: a `[rows, heads, 1, width]` score
rectangle at the widest shape, against one `[rows, heads, 1, block]` tile. The ratio
is `width / block` exactly, which is Day 59's `workspace_saving` and is the only
number in this file that is a property of the read rather than of the list.

**`waste`** is the one that is easy to quote wrongly, and the column exists to stop
that. A streamed set rounds every context up to `max_model_len`, so a waste measured
against `DecodeShape.cells` reads 99% and means nothing: the read never builds that
rectangle. `DecodeBuckets.cells_for` prices the tiles a read really walks, and in that
currency the width padding costs zero and the only overshoot left is the last tile of
a ragged row.

**`startup`** is the column that had to be walked back, and the boot table below is
what walked it. A list of 9 graphs instead of 576 looks like 64x off the warm-up, and
it is not, because a streamed capture is not the same price as a rectangle one: every
recording runs the tile loop, which is Python. The measured boot at the bottom of this
file says 6.6 ms a graph on the rectangle arm and 19.2 ms on the streamed one, so the
arithmetic column prices the streamed captures at `STREAMED_CAPTURE_PENALTY` times the
rectangle's and the saving comes out around 22x rather than 64x. A 2400-token toy
boot that records 32 graphs in 0.21s records 4 in 0.077s: real, and not the ratio of
the list lengths.

Nothing here is a latency claim. The streamed read is `tlsim` in Python and Day 60
measured it 8x to 67x slower per call than the torch path on this box. Every number
below is about memory and startup, which is where the width axis was actually being
paid for. The memory half stays true when the loop becomes Triton; the startup
penalty is the half that goes away with it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.buckets import DecodeBuckets, waste
from nanoserve.captured import (
    DEFAULT_CAPTURE_LIMIT,
    eager_recorder,
    shared_pool_bytes,
    workspace_saving,
)
from nanoserve.compiled import DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.launch import (
    boot_info,
    boot_lines,
    build_engine,
    check_boot_info,
    plan_capture,
    warm_engine,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.warmup import warm_shapes, warmup_seconds

#: What one graph costs to record, the same 50 ms Day 55 and Day 56 priced their
#: startup tables with. Kept identical so the three read together.
PER_CAPTURE_S = 0.05

#: What a streamed capture costs to record against a rectangle one, measured by the
#: boot table at the bottom of this file: 19.2 ms a graph against 6.6 at 8 slots and
#: 1024 tokens. It is entirely the tlsim loop running once per warm-up call, so it is
#: a property of this box and of nothing on a card, and it is here because pricing
#: both arms at 50 ms would turn a 22x startup saving into a 64x one on paper.
STREAMED_CAPTURE_PENALTY = 2.9

#: Query heads of Llama-3.2-1B, which is what a score rectangle is counted in.
HEADS = 32

#: Deployments somebody would actually type. The first row is `serve.py` with no
#: flags at all.
SHAPES = (
    (8, 2048),
    (8, 4096),
    (16, 8192),
    (32, 4096),
    (64, 8192),
    (256, 8192),
)

#: A run to price the padding over: one bucket's worth of rows, walking the context
#: from a short prompt out to a thousand tokens the way a real generation does.
RUN = tuple(DecodeShape(rows=6, context_width=w) for w in range(24, 1024))


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _weights(cfg: ModelConfig) -> Weights:
    torch.manual_seed(0)
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, cfg)


# --- table 1: what the two bucket sets ask the process to hold ----------------------


def set_table(shapes, block: int) -> list[dict]:
    """Both sets for each deployment, side by side in every currency there is."""
    rows = []
    for slots, context in shapes:
        for streamed in (False, True):
            buckets = DecodeBuckets(
                slots, context, streamed=streamed, block=block if streamed else 0
            )
            kept = warm_shapes(buckets)
            run = tuple(s for s in RUN if s.rows <= slots and s.context_width <= context)
            rows.append(
                {
                    "slots": slots,
                    "context": context,
                    "read": "streamed" if streamed else "rectangle",
                    "block": block if streamed else 0,
                    "row_buckets": len(buckets.rows),
                    "width_buckets": len(buckets.widths),
                    "shapes": buckets.count,
                    "startup_s": round(
                        warmup_seconds(
                            len(kept),
                            PER_CAPTURE_S
                            * (STREAMED_CAPTURE_PENALTY if streamed else 1.0),
                        ),
                        2,
                    ),
                    "arena_mib": round(
                        shared_pool_bytes(
                            kept, HEADS, block=block if streamed else None
                        )
                        / 1024**2,
                        2,
                    ),
                    "waste": round(waste(run, buckets), 3) if run else 0.0,
                }
            )
    return rows


def print_set_table(rows, block: int) -> None:
    print(
        f"\nthe capture list for one cache, both reads, {HEADS} heads, fp32 scores, "
        f"{block}-key tiles:",
        file=sys.stderr,
    )
    print(
        "  slots  context         read   rows x widths   shapes   startup      "
        "arena   waste",
        file=sys.stderr,
    )
    for r in rows:
        print(
            f"  {r['slots']:>5} {r['context']:>8} {r['read']:>12} "
            f"{r['row_buckets']:>7} x {r['width_buckets']:<5} {r['shapes']:>6} "
            f"{r['startup_s']:>8.2f}s {r['arena_mib']:>9.2f} MiB {r['waste']:>6.1%}",
            file=sys.stderr,
        )


def print_ratio_table(rows) -> None:
    """The three savings, each next to the thing it is actually a ratio of.

    The three columns are three different claims and only one of them is the length
    of the list. `shapes` is that length. `startup` is the list priced at what a
    capture really costs on each arm, which is why it is not the same number.
    `arena` is not about the list at all: it is `width / block` at the widest shape,
    and it would be that ratio if the list had one member.
    """
    print("\nwhat the axis was costing, per deployment:", file=sys.stderr)
    print("  slots  context   shapes   startup    arena", file=sys.stderr)
    pairs = {}
    for r in rows:
        pairs.setdefault((r["slots"], r["context"]), {})[r["read"]] = r
    for (slots, context), arms in pairs.items():
        rect, tile = arms["rectangle"], arms["streamed"]
        print(
            f"  {slots:>5} {context:>8} {rect['shapes'] / tile['shapes']:>7.0f}x "
            f"{rect['startup_s'] / max(tile['startup_s'], 1e-9):>8.0f}x "
            f"{rect['arena_mib'] / max(tile['arena_mib'], 1e-9):>7.0f}x",
            file=sys.stderr,
        )


def print_arena_note(block: int) -> None:
    """One line saying the arena ratio is a property of the read, not of the list."""
    widest = DecodeShape(rows=256, context_width=8192)
    print(
        f"\n  the arena ratio is not the list. At {widest.rows} rows and "
        f"{widest.context_width} tokens it is "
        f"{workspace_saving(widest, HEADS, block):.0f}x, which is "
        f"{widest.context_width} / {block} and nothing else: Day 59's "
        "`workspace_saving`, unchanged.",
        file=sys.stderr,
    )


# --- table 2: a measured boot down the path serve.py takes --------------------------


def boot(slots: int, context: int, block: int, streamed: bool, block_size: int = 4):
    """Build, plan the list, record it, publish. The Day-56 path, both reads."""
    cfg = _config()
    engine, plan = build_engine(
        "unused",
        device="cpu",
        dtype="float32",
        block_size=block_size,
        max_batch_size=slots,
        max_model_len=context,
        num_blocks=-(-context // block_size) + slots,
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
        streamed_read=streamed,
        read_block=block,
        load=lambda _dir, **_kw: _weights(cfg),
        read_config=lambda _dir: cfg,
    )
    capture = plan_capture(engine, plan)
    report = warm_engine(engine, capture)
    check_boot_info(boot_info(plan, capture, report))
    return plan, capture, report


def boot_table(shapes, block: int) -> None:
    print("\na measured boot of the toy model, both reads:", file=sys.stderr)
    for slots, context in shapes:
        for streamed in (False, True):
            plan, capture, report = boot(slots, context, block, streamed)
            read = "streamed" if streamed else "rectangle"
            print(f"\n  {slots} slots x {context} tokens, {read} read", file=sys.stderr)
            for line in boot_lines(plan, capture, report):
                print(f"    {line}", file=sys.stderr)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "slots",
        "context",
        "read",
        "block",
        "row_buckets",
        "width_buckets",
        "shapes",
        "startup_s",
        "arena_mib",
        "waste",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve capture-list width axis benchmark")
    p.add_argument("--block", type=int, default=32, help="keys per score tile")
    p.add_argument("--limit", type=int, default=DEFAULT_CAPTURE_LIMIT)
    p.add_argument("--csv", default="docs/daily/data/day-61-widthbench.csv")
    args = p.parse_args()

    rows = set_table(SHAPES, args.block)
    print_set_table(rows, args.block)
    print_ratio_table(rows)
    print_arena_note(args.block)
    boot_table(((4, 256), (8, 1024)), args.block)

    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
