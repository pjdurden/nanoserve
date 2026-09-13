"""Day 56: what a boot with graphs on costs, and which ceiling picks the list.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python bootbench.py
    cd ~/nanoserve && .venv/bin/python bootbench.py --csv docs/daily/data/day-56-bootbench.csv

No weights and no card. The first two tables are arithmetic over a bucket set, which
is the honest way to price this: a capture list is `row buckets x width buckets` and
both factors are integers a deployment picks. The third table is a real boot of the
two-layer toy model through `build_engine`, `plan_capture` and `warm_engine`, so the
boot lines below are the ones `serve.py` prints.

**The first table is the day's finding.** `serve.py`'s defaults are 8 slots and 2048
tokens, which is 4 row buckets x 16 widths = 64 shapes, and `DEFAULT_CAPTURE_LIMIT` is
64. The default deployment fits with nothing to spare, and the first flag anyone
reaches for (`--max-batch-size 16`) adds a fifth row bucket and puts the list 25% over.
Nothing about the width changed and the width is what gives, because the count is a
product and a deployment cannot move the other factor without changing what it sells.

**The second table is the other side of the trim.** A ceiling below the served
context does not delete those recordings, it moves them back into the decode loop
where Day 55 found them. What makes the trade worth taking is *where* on the width
axis they land: a run walks the axis from the bottom, so the shapes left out at the
top are reached once, late, by a request that has already streamed thousands of
tokens, and the shapes at the bottom are reached in the first seconds of every
request the server ever answers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.buckets import DecodeBuckets
from nanoserve.captured import DEFAULT_CAPTURE_LIMIT, eager_recorder, shared_pool_bytes
from nanoserve.config import ModelConfig
from nanoserve.launch import (
    boot_info,
    boot_lines,
    build_engine,
    check_boot_info,
    plan_capture,
    warm_engine,
    width_from_limit,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.warmup import warm_shapes, warmup_seconds

#: A plausible cost of recording one graph on a real card, in seconds. The same knob
#: Day 55 priced its startup table with, kept identical so the two read together.
PER_CAPTURE_S = 0.05

#: Query heads of Llama-3.2-1B, which is what a score rectangle is counted in.
HEADS = 32

#: Deployments somebody would actually type. The first row is `serve.py` with no
#: flags at all.
SHAPES = (
    (8, 2048),
    (8, 4096),
    (16, 2048),
    (16, 8192),
    (32, 4096),
    (64, 8192),
    (256, 8192),
)


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


# --- table 1: what each deployment shape asks the process to hold -------------------


def limit_table(shapes, limit: int) -> list[dict]:
    """For each deployment, the list it wants and the list the limit lets it have."""
    rows = []
    for slots, context in shapes:
        buckets = DecodeBuckets(slots, context)
        full = len(buckets.shapes)
        width = width_from_limit(buckets, row_count=len(buckets.rows), limit=limit)
        kept = warm_shapes(buckets, max_width=width)
        rows.append(
            {
                "slots": slots,
                "context": context,
                "row_buckets": len(buckets.rows),
                "width_buckets": len(buckets.widths),
                "shapes_wanted": full,
                "over_limit": max(0, full - limit),
                "width_ceiling": width,
                "shapes_kept": len(kept),
                "startup_s": round(warmup_seconds(len(kept), PER_CAPTURE_S), 2),
                "untrimmed_startup_s": round(warmup_seconds(full, PER_CAPTURE_S), 2),
                "workspace_mib": round(shared_pool_bytes(kept, HEADS) / 1024**2, 1),
            }
        )
    return rows


def print_limit_table(rows, limit: int) -> None:
    print(f"\nwhat a deployment asks for, against a {limit}-graph limit:", file=sys.stderr)
    print(
        "  slots  context   rows x widths   wanted   over   ceiling   kept   "
        "startup   arena",
        file=sys.stderr,
    )
    for r in rows:
        mark = " " if r["over_limit"] == 0 else "*"
        print(
            f"  {r['slots']:>5}  {r['context']:>7}   {r['row_buckets']:>4} x "
            f"{r['width_buckets']:<7} {r['shapes_wanted']:>6} {r['over_limit']:>6}{mark}  "
            f"{r['width_ceiling']:>7}  {r['shapes_kept']:>5}   {r['startup_s']:>6.2f}s  "
            f"{r['workspace_mib']:>6.1f} MB",
            file=sys.stderr,
        )
    print(
        "  * the list is over the limit and the width axis is what gives",
        file=sys.stderr,
    )


# --- table 2: what the trim moves back into the decode loop -------------------------


def buckets_above(*, start: int, steps: int, width_multiple: int, ceiling: int) -> int:
    """Width buckets a run crosses that are past the warm list's ceiling.

    One mid-run recording each, taken in front of whoever asked for that context
    first. Everything at or below the ceiling was recorded at boot.
    """
    crossed = {
        -(-token // width_multiple) * width_multiple
        for token in range(start + 1, start + steps + 1)
    }
    return sum(1 for width in crossed if width > ceiling)


def trim_table(rows) -> None:
    print(
        "\nwhat a ceiling moves back into somebody's latency, per request that "
        "generates to the full context:",
        file=sys.stderr,
    )
    print(
        "  slots  context   ceiling   lazy captures   in a client's latency   "
        "saved at startup",
        file=sys.stderr,
    )
    for r in rows:
        buckets = DecodeBuckets(r["slots"], r["context"])
        lazy = buckets_above(
            start=0,
            steps=r["context"],
            width_multiple=buckets.width_multiple,
            ceiling=r["width_ceiling"],
        )
        print(
            f"  {r['slots']:>5}  {r['context']:>7}  {r['width_ceiling']:>8}  "
            f"{lazy:>14}   {lazy * PER_CAPTURE_S:>20.2f}s   "
            f"{r['untrimmed_startup_s'] - r['startup_s']:>14.2f}s",
            file=sys.stderr,
        )


# --- table 3: a real boot, end to end -----------------------------------------------


def boot(slots: int, context: int, block_size: int = 4):
    """The whole Day-56 path over the toy model: build, plan, warm, publish."""
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
        load=lambda _dir, **_kw: _weights(cfg),
        read_config=lambda _dir: cfg,
    )
    capture = plan_capture(engine, plan)
    report = warm_engine(engine, capture)
    check_boot_info(boot_info(plan, capture, report))
    return plan, capture, report


def boot_table(shapes) -> None:
    print("\na measured boot of the toy model, which is the path serve.py takes:", file=sys.stderr)
    for slots, context in shapes:
        plan, capture, report = boot(slots, context)
        print(f"\n  {slots} slots x {context} tokens", file=sys.stderr)
        for line in boot_lines(plan, capture, report):
            print(f"    {line}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, default=DEFAULT_CAPTURE_LIMIT)
    p.add_argument("--csv", default=None, help="write the first table here")
    args = p.parse_args()

    rows = limit_table(SHAPES, args.limit)
    print_limit_table(rows, args.limit)
    trim_table(rows)
    boot_table(((4, 256), (8, 1024)))

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
