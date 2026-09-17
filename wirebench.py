"""Day 60: the two decode reads under a whole engine, same tokens, different cells.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python wirebench.py
    cd ~/nanoserve && .venv/bin/python wirebench.py \
        --rows 2,4,8 --prompt 6 --steps 16 --block 8,32 \
        --csv docs/daily/data/day-60-wirebench.csv

No weights and no device. `streambench.py` asked what the two reads *would* hold, on
serving shapes, from arithmetic; this asks what they *did* hold, on a tiny model that
really generates, and the difference between those two questions is the day. Day 59's
table is a statement about `context_width / block`. This one is a quotient of two
counters that a live engine incremented, and the only way it can be wrong is if the
flag never reached the cache, which is precisely the failure the counters exist for.

Three columns are worth reading together.

**`same`** is whether the two reads produced byte-identical token ids through the
whole engine: prefill, decode, sampler, stop rules. Greedy, so the comparison is an
argmax over logits that differ in the last couple of ulps, which is a real chance to
disagree and not a formality: a tie broken the other way is a different word and then
a different continuation forever. It says `yes` on every row below, and the row where
it would say `no` is the one worth having the column for.

**`held`** against **`charged`** is the live version of Day 54's workspace. `charged`
is the cells a `[rows, heads, 1, ctx]` score rectangle would have materialised over
this run, summed per call; `held` is what the read actually had alive. On the default
read they are equal by construction and the saving is 1.0x, which is a measurement of
parity rather than a missing number.

**`ms/read`** is the honest price and it is why this is a flag. The streamed read is
`tlsim` on the CPU, a Python loop over `(row, head)`, so it is slower per call than
the torch path by about an order of magnitude and it will be until the loop is
Triton. Nothing in this file pretends otherwise, and no number here is a claim about
what the same read costs on a card.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.reads import RECTANGLE, STREAMED

#: The prompt lengths a row gets, cycled. Ragged on purpose: the rectangle charges
#: every row the longest row's history, so a batch whose rows are all the same length
#: is the one workload where the streamed read has nothing ragged to skip. Day 58
#: measured what a real continuous batch looks like and it is not that.
SPREAD = (1, 2, 4, 3)


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


def _model(cfg: ModelConfig) -> LlamaModel:
    torch.manual_seed(0)
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _prompts(rows: int, prompt: int) -> list[list[int]]:
    """`rows` prompts whose lengths run over `SPREAD`, scaled so the longest is `prompt`."""
    top = max(SPREAD)
    return [
        list(range(1, max(2, prompt * SPREAD[i % len(SPREAD)] // top) + 1))
        for i in range(rows)
    ]


def run(cfg, model, rows: int, prompt: int, steps: int, block: int, *, streamed: bool):
    """Generate `steps` tokens for `rows` ragged prompts on one read. Returns
    `(tokens, stats, seconds)`, where `stats` is the whole run's window because the
    engine is fresh: a counter is cumulative and this process started it at zero."""
    per_row = -(-(prompt + steps) // 4) + 2
    engine = Engine.build(
        model,
        num_blocks=per_row * rows + 2,
        block_size=4,
        max_batch_size=rows,
        max_model_len=prompt + steps + 8,
        streamed_read=streamed,
        read_block=block,
    )
    started = time.perf_counter()
    out = engine.generate(_prompts(rows, prompt), max_new_tokens=steps)
    seconds = time.perf_counter() - started
    return out, engine.cache.read.stats(), seconds


def sweep(rows_list: list[int], prompt: int, steps: int, blocks: list[int]) -> list[dict]:
    cfg = _config()
    model = _model(cfg)
    print(
        f"\n  {'rows':>5} {'prompt':>7} {'steps':>6} {'block':>6} {'same':>5} "
        f"{'reads':>6} {'charged':>9} {'held':>8} {'saving':>7} "
        f"{'ms/read':>8} {'vs rect':>8}",
        file=sys.stderr,
    )
    out = []
    for rows in rows_list:
        base_tokens, base_stats, base_s = run(
            cfg, model, rows, prompt, steps, blocks[0], streamed=False
        )
        base_ms = base_s / base_stats.calls * 1e3
        for block in blocks:
            tokens, stats, seconds = run(
                cfg, model, rows, prompt, steps, block, streamed=True
            )
            same = tokens == base_tokens
            ms = seconds / stats.calls * 1e3
            for mode, st, per_read in (
                (RECTANGLE, base_stats, base_ms),
                (STREAMED, stats, ms),
            ):
                out.append(
                    {
                        "rows": rows,
                        "prompt": prompt,
                        "steps": steps,
                        "block": st.block,
                        "mode": mode,
                        "same_tokens": "yes" if same else "no",
                        "calls": st.calls,
                        "score_cells": st.score_cells,
                        "held_cells": st.held_cells,
                        "saving": round(st.saving, 2),
                        "ms_per_read": round(per_read, 3),
                    }
                )
            print(
                f"  {rows:>5} {prompt:>7} {steps:>6} {block:>6} "
                f"{('yes' if same else 'NO'):>5} {stats.calls:>6} "
                f"{stats.score_cells:>9} {stats.held_cells:>8} {stats.saving:>6.1f}x "
                f"{ms:>8.2f} {ms / base_ms:>7.1f}x",
                file=sys.stderr,
            )
        print(
            f"  {rows:>5} {prompt:>7} {steps:>6} {'-':>6} {'-':>5} "
            f"{base_stats.calls:>6} {base_stats.score_cells:>9} "
            f"{base_stats.held_cells:>8} {base_stats.saving:>6.1f}x "
            f"{base_ms:>8.2f} {1.0:>7.1f}x   (rectangle)",
            file=sys.stderr,
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rows",
        "prompt",
        "steps",
        "block",
        "mode",
        "same_tokens",
        "calls",
        "score_cells",
        "held_cells",
        "saving",
        "ms_per_read",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve wired decode read benchmark")
    p.add_argument("--rows", type=_int_list, default=[2, 4, 8])
    p.add_argument("--prompt", type=int, default=16, help="the longest prompt in the batch")
    p.add_argument("--steps", type=int, default=48, help="tokens generated per row")
    p.add_argument("--block", type=_int_list, default=[8, 32], help="score tile widths")
    p.add_argument("--csv", default="docs/daily/data/day-60-wirebench.csv")
    args = p.parse_args()

    print("the wired decode read, one tiny engine, both reads:", file=sys.stderr)
    rows = sweep(args.rows, args.prompt, args.steps, args.block)
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
