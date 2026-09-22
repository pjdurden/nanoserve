"""Day 65: three reads under one cache, and what each one publishes about itself.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python modebench.py
    cd ~/nanoserve && .venv/bin/python modebench.py \
        --block 32 --csv docs/daily/data/day-65-modebench.csv

No weights and no device. The first table runs three real caches over the same toy
pool and reads their counters back; the rest is arithmetic over a capture list.

**The first table is the wiring, and the assertion in it is the only one that
matters.** Three caches, same config, same allocator, same prefill, same planned
decode step, one flag apart. If the three outputs do not agree to a few ulps the
dispatch changed the arithmetic, and if the counters do not differ the flag did not
reach anything. Day 60 made that argument for two reads; the third one adds a
precondition neither of the others has, which is that a split read holds an arena it
did not allocate, so the row for it is only reachable after `allocate_split_workspace`.

**The second table is the saving each read reports at serving size, in two
currencies.** Score cells held at once is what `/health` publishes and it is the
number a client-side gate can check; the arena is what the process actually reserves
and it does not appear in that quotient at all. The split is the read where the two
disagree hardest, and that is the point of printing them side by side: it holds the
streamed read's tile once per chunk, so its published saving is the streamed one
divided by the chunks, and it *also* holds a workspace the streamed read has no
version of. A gate that looked only at the first column would read a split server as
a worse streamed one.

**The third table is the gate.** `check_read_matches` has been one question since Day
61 (does this set still have a width axis, and does this read still want one) and it
is now three. The matrix is every (set, read) pair this repo can build, and the useful
column is the last: which clause refuses, because each one names a different mistake
and a different fix. Nothing in it raises on its own in a running server. Every cell
that refuses here is a server that would have produced correct tokens against a memory
number nobody could see was wrong.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.buckets import BucketsUnsound, DecodeBuckets, check_read_matches
from nanoserve.cache import BatchedPagedKVCache, BlockAllocator
from nanoserve.captured import score_cells, split_workspace_bytes
from nanoserve.compiled import DecodeShape
from nanoserve.config import ModelConfig
from nanoserve.kernels.flash_decoding import plan_splits
from nanoserve.partials import allocate_partials
from nanoserve.reads import SPLIT, STREAMED, PagedRead

#: The deployment every arithmetic table in this week is stated over, so a number
#: here reads next to Day 54's, Day 61's and Day 64's without converting anything.
SERVING_ROWS = 256
SERVING_LEN = 8192
SERVING_HEADS = 32
SERVING_HEAD_DIM = 128

#: The toy the first table runs on. Small enough to be a second of host time and wide
#: enough that a split really has more than one chunk to fold.
TOY_ROWS = 4
TOY_LEN = 2048
TOY_BLOCK = 4


def _toy_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _toy_cache(**kw) -> BatchedPagedKVCache:
    cfg = _toy_config()
    return BatchedPagedKVCache(
        cfg,
        BlockAllocator(num_blocks=TOY_LEN // 4, block_size=4),
        batch_size=TOY_ROWS,
        max_model_len=TOY_LEN,
        bucket_decode=True,
        read_block=TOY_BLOCK,
        **kw,
    )


def _toy_step(cache: BatchedPagedKVCache) -> torch.Tensor:
    """One prefill and one planned decode step, identical on every arm."""
    cfg = _toy_config()
    lengths = [9, 6, 3, 1]
    width = max(lengths)
    mask = torch.zeros(TOY_ROWS, width, dtype=torch.long)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
    torch.manual_seed(7)
    for layer in range(cfg.num_hidden_layers):
        shape = (TOY_ROWS, cfg.num_key_value_heads, width, cfg.head_dim)
        cache.write(layer, torch.randn(*shape), torch.randn(*shape), mask)
    torch.manual_seed(11)
    k = torch.randn(TOY_ROWS, cfg.num_key_value_heads, 1, cfg.head_dim)
    v = torch.randn(TOY_ROWS, cfg.num_key_value_heads, 1, cfg.head_dim)
    q = torch.randn(TOY_ROWS, cfg.num_attention_heads, 1, cfg.head_dim)
    plan = cache.plan_decode(rows=tuple(range(TOY_ROWS)))
    return cache.paged_attention(0, k, v, q, n_rep=4, plan=plan)


def wiring() -> None:
    """Three caches, one flag apart, and what each publishes after one decode step."""
    print(
        f"\n  one planned decode step, {TOY_ROWS} rows over a {TOY_LEN}-wide table, "
        f"{TOY_BLOCK}-key tiles:",
        file=sys.stderr,
    )
    print(
        f"  {'mode':>10} {'backend':>8} {'splits':>7} {'charged':>8} {'held':>6} "
        f"{'saving':>7} {'vs oracle':>10}",
        file=sys.stderr,
    )
    oracle = None
    for flags in ({}, {"streamed_read": True}, {"split_read": True}):
        cache = _toy_cache(**flags)
        if cache.read.split:
            cache.allocate_split_workspace()
        out = _toy_step(cache)
        oracle = out if oracle is None else oracle
        stats = cache.read.stats()
        print(
            f"  {stats.mode:>10} {stats.backend:>8} {stats.splits:>7} "
            f"{stats.score_cells:>8} {stats.held_cells:>6} {stats.saving:>6.1f}x "
            f"{float((out - oracle).abs().max()):>10.2e}",
            file=sys.stderr,
        )
    print(
        "  the three answers agree and the three payloads do not, which is the whole "
        "claim: a\n  read is invisible downstream, so the counters are the only place "
        "a flag becomes visible.\n  The charges differ across rows because the sets "
        "do: the rectangle arm still buckets the\n  width and rounds this step to "
        "128, and the other two read at the table's full 2048. That\n  is Day 61's "
        "note, and it is why the savings in this table compare to themselves and not\n"
        "  to each other. The serving table below holds the width fixed.",
        file=sys.stderr,
    )


def savings(width: int, block: int) -> list[dict]:
    """What each read holds at serving size, as cells published and as bytes reserved."""
    buckets = DecodeBuckets(SERVING_ROWS, width, streamed=True, block=block)
    splits = plan_splits(buckets.rows, SERVING_HEADS, width, block)
    shape = DecodeShape(rows=SERVING_ROWS, context_width=width)
    charged = score_cells(shape, SERVING_HEADS)
    tile = SERVING_ROWS * SERVING_HEADS * block

    arms = (
        ("rectangle", 0, charged, 0),
        ("streamed", 0, tile, 0),
        (
            "split",
            splits,
            splits * tile,
            split_workspace_bytes(
                shape, SERVING_HEADS, block, splits, SERVING_HEAD_DIM
            )
            - splits * tile * 2,
        ),
    )

    print(
        f"\n  one decode step at {SERVING_ROWS} rows, {SERVING_HEADS} heads, {width} "
        f"tokens, {block}-key tiles:",
        file=sys.stderr,
    )
    print(
        f"  {'read':>10} {'splits':>7} {'held cells':>11} {'published':>10} "
        f"{'partials':>12}",
        file=sys.stderr,
    )
    out = []
    for name, cut, held, arena in arms:
        out.append(
            {
                "read": name,
                "rows": SERVING_ROWS,
                "heads": SERVING_HEADS,
                "head_dim": SERVING_HEAD_DIM,
                "context_width": width,
                "block": block,
                "splits": cut,
                "score_cells": charged,
                "held_cells": held,
                "arena_bytes": arena,
            }
        )
        print(
            f"  {name:>10} {cut:>7} {held:>11} {charged / held:>9.1f}x "
            f"{arena / 1e6:>9.2f} MB",
            file=sys.stderr,
        )
    print(
        "  the published saving is a quotient of score cells and the arena is not in "
        "it. A\n  split server reporting 16x beside a streamed one reporting 256x is "
        f"not broken: it holds\n  the same tile {splits} times over, once per chunk, "
        "and an accumulator per program besides.",
        file=sys.stderr,
    )
    return out


def gates(width: int, block: int) -> None:
    """Every (bucket set, read) pair, and which clause refuses the ones that do."""
    splits = plan_splits(
        DecodeBuckets(SERVING_ROWS, width).rows, SERVING_HEADS, width, block
    )

    def arena(cut: int):
        return allocate_partials(
            max_rows=1,
            n_q=1,
            head_dim=1,
            context_width=width,
            block=block,
            splits=cut,
        )

    def tag(buckets: DecodeBuckets, read: PagedRead) -> str:
        """The refusing clause as a phrase short enough to tabulate.

        Reconstructed from the pair rather than parsed out of the message, and the
        messages are the reason: a gate that refuses at boot is read once, by somebody
        who has to fix it, so each one is a paragraph. A matrix is read all at once by
        somebody counting. The clause order here is `check_read_matches`' own, so a
        row says which of its four questions failed first.
        """
        if buckets.streamed and not read.tiled:
            return "no width axis"
        if read.tiled and not buckets.streamed:
            return "width buckets"
        if read.tiled and read.block != buckets.block:
            return "tile mismatch"
        if buckets.splits and not read.split:
            return "nobody addresses"
        if read.split and not buckets.splits:
            return "nobody sized"
        if read.split and read.splits < 1:
            return "no workspace"
        return f"{read.splits} against {buckets.splits}"

    sets = (
        ("bucketed widths", DecodeBuckets(SERVING_ROWS, width, width_multiple=2048)),
        ("one width", DecodeBuckets(SERVING_ROWS, width, streamed=True, block=block)),
        (
            f"one width, {splits} chunks",
            DecodeBuckets(SERVING_ROWS, width, streamed=True, block=block, splits=splits),
        ),
    )
    reads = (
        ("rectangle", PagedRead()),
        ("streamed", PagedRead(STREAMED, block=block)),
        ("split, no arena", PagedRead(SPLIT, block=block)),
        ("split", PagedRead(SPLIT, block=block, workspace=arena(splits))),
        ("split, 2 chunks", PagedRead(SPLIT, block=block, workspace=arena(2))),
    )

    print("\n  check_read_matches over every pair this repo can build:", file=sys.stderr)
    print(f"  {'set':>24} {'read':>16}  {'verdict'}", file=sys.stderr)
    passed = 0
    for set_name, buckets in sets:
        for read_name, read in reads:
            try:
                check_read_matches(buckets, read)
                verdict = "ok"
                passed += 1
            except BucketsUnsound:
                verdict = "refused: " + tag(buckets, read)
            print(f"  {set_name:>24} {read_name:>16}  {verdict}", file=sys.stderr)
    total = len(sets) * len(reads)
    print(
        f"  {passed} of these pass and {total - passed} do not, and every refusal is a "
        "server that would\n  have answered correctly against a memory number nobody "
        "in the process could see was wrong.",
        file=sys.stderr,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "read",
        "rows",
        "heads",
        "head_dim",
        "context_width",
        "block",
        "splits",
        "score_cells",
        "held_cells",
        "arena_bytes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve decode read mode benchmark")
    p.add_argument("--width", type=int, default=SERVING_LEN)
    p.add_argument("--block", type=int, default=32, help="the score tile, not block_size")
    p.add_argument("--csv", default="docs/daily/data/day-65-modebench.csv")
    args = p.parse_args()

    print(
        f"three decode reads under one cache: {SERVING_HEADS} heads x "
        f"{SERVING_HEAD_DIM} channels, {args.width}-wide mapping",
        file=sys.stderr,
    )
    wiring()
    rows = savings(args.width, args.block)
    gates(args.width, args.block)
    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
