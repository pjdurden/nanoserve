"""Day 66: the split read's boot path, one step at a time, and where the arena lands.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python bootbench_split.py
    cd ~/nanoserve && .venv/bin/python bootbench_split.py \
        --csv docs/daily/data/day-66-bootbench.csv

No weights and no device: a tiny random model on CPU with the eager recorder, so
every number here is the wiring and none of it is a second.

**The first table is the order.** `build_app` is five calls and the split read cares
about the position of one of them. Each row is a boot path with `arm_split_read` in a
different place (or missing) and says which call refuses and in what words. Only one
position boots, and the refusals are the reason the other three are not a style
choice: one after the warm-up is one the warm-up already needed, one armed twice is
a second arena or a pointer swap under a recorded graph, and one that never happens
is a server that would be up and refuse its first decode.

**The second table is three servers, one flag apart,** booted exactly the way
`serve.py` boots them, with the lines they print. Only the split prints a fourth line
and only the split's `/health` carries a `split_workspace` section.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.captured import CaptureUnsound, eager_recorder
from nanoserve.config import ModelConfig
from nanoserve.launch import (
    BootUnsound,
    arm_split_read,
    boot_info,
    boot_lines,
    build_app,
    build_engine,
    kv_bytes_per_block,
    plan_capture,
    warm_engine,
)
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes

CONFIG = ModelConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=4,
)
GRAPHS = dict(bucket_decode=True, persist_inputs=True, capture_decode=True)


def _weights() -> Weights:
    torch.manual_seed(0)
    tensors = {n: torch.randn(*s) for n, s in expected_shapes(CONFIG).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, CONFIG)


def _kwargs(**kw) -> dict:
    out = dict(
        weights_dir="unused",
        device="cpu",
        dtype="float32",
        block_size=4,
        max_batch_size=4,
        max_model_len=1024,
        read_block=4,
        kv_cache_bytes=kv_bytes_per_block(CONFIG, 4, torch.float32) * 256,
        load=lambda _d, **_k: _weights(),
        read_config=lambda _d: CONFIG,
        capture_recorder=eager_recorder,
    )
    out.update(kw)
    return out


class _Tok:
    eos_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [b % 64 for b in text.encode()]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


def boot_order(position: str) -> tuple[str, str]:
    """Build, plan, warm, with the arm at `position`. Returns (step reached, verdict)."""
    engine, plan = build_engine(**_kwargs(split_read=True, **GRAPHS))
    try:
        capture = plan_capture(engine, plan, split_read=True, device="cpu")
        if position in ("after plan", "twice"):
            arm_split_read(engine, capture)
        if position == "twice":
            arm_split_read(engine, capture)
        warm_engine(engine, capture)
        if position == "after warm":
            arm_split_read(engine, capture)
        return position, "boots"
    except (BootUnsound, CaptureUnsound) as err:
        kind = type(err).__name__
        return position, f"{kind}: {str(err).split(':')[0][:60]}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    rows = []

    print("where arm_split_read sits on the boot path:")
    print(f"{'position':>12}  verdict")
    for position in ("never", "after warm", "twice", "after plan"):
        where, verdict = boot_order(position)
        print(f"{where:>12}  {verdict}")
        rows.append(dict(table="order", arm=where, verdict=verdict))

    print()
    print("three servers, one flag apart, booted the way serve.py boots them:")
    for name, flags in (("rectangle", {}), ("streamed", {"streamed_read": True}),
                        ("split", {"split_read": True})):
        app = build_app(tokenizer=_Tok(), **_kwargs(**GRAPHS, **flags))
        s = app.state
        info = boot_info(s.plan, s.capture, s.warmup, s.workspace)
        lines = boot_lines(s.plan, s.capture, s.warmup, s.workspace)
        arena = s.workspace.bytes if s.workspace is not None else 0
        print(f"  {name}: {len(lines)} boot lines, arena {arena} B, "
              f"split_workspace in /health: {'split_workspace' in info}")
        for line in lines:
            print(f"    {line}")
        rows.append(dict(table="servers", arm=name, verdict=f"{len(lines)} lines",
                         arena_bytes=arena, splits=s.capture.splits))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["table", "arm", "verdict", "arena_bytes", "splits"]
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
