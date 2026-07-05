"""Day 20: microbenchmark the paged read, gather vs fused, across history lengths.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python readbench.py
    cd ~/nanoserve && .venv/bin/python readbench.py --lens 16,64,256,1024,4096 \
        --repeats 50 --csv docs/daily/data/day-20-readbench.csv

No weights are needed: the read cost depends on shapes and history length, not on
which tokens are present, so this builds a real `PagedKVCache` from the default
Llama-3.2-1B config, fills random K/V, and times only the read. Two closures at a
fixed history length L:

  gather (Day 16): rebuild the contiguous `[1, n_kv, L, d]` history out of the
    scattered pool, then run the ordinary masked-softmax SDPA over it, the buffer
    paging exists to avoid.
  fused (Day 19): `paged_attention_reference` scores the one decode query over the
    scattered blocks in place through the slot mapping, no contiguous history built.

Both are the *read* only; the write is identical and shared, so it is folded into
the one-time prefill and excluded from timing. The two closures are asserted equal
before timing (same math, the Day-18/19 invariant), then each is timed best-of.

The honest result on CPU is close to a tie: both paths are torch and both do an
O(L) index gather, so the reference is not where the speedup lives. What the sweep
exposes is each path's absolute per-step cost and how it climbs with L, which is
the baseline the hand-written Triton kernel has to beat. The timing math lives in
`nanoserve.readbench` and is unit-tested with a fake clock; this file only wires the
real cache into it and prints/saves the numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from nanoserve.cache import BlockAllocator, PagedKVCache
from nanoserve.config import ModelConfig
from nanoserve.kernels.paged_attention import paged_attention_reference
from nanoserve.layers import repeat_kv
from nanoserve.readbench import compare_reads


def build_cache(config: ModelConfig, history: int, block_size: int) -> PagedKVCache:
    """A paged cache for layer 0, prefilled with `history` tokens of random K/V.

    The pool is sized just past `history` so the prefill fits. Only layer 0 is
    written: the read cost is per-layer and identical across layers, so one layer
    is the whole story and keeps the pools small.
    """
    num_blocks = (history + block_size - 1) // block_size + 1
    allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    cache = PagedKVCache(config, allocator)
    n_kv, d = config.num_key_value_heads, config.head_dim
    k = torch.randn(1, n_kv, history, d)
    v = torch.randn(1, n_kv, history, d)
    cache.append(0, k, v)  # one prefill write; grows the table to `history`
    return cache


def make_reads(cache: PagedKVCache, config: ModelConfig, history: int):
    """Two zero-arg read closures over the prefilled pool: (gather, fused).

    Both model a decode step at history length `history`: one query (seq_q=1) at
    absolute position `history-1` scoring the whole cached history. Neither mutates
    the cache, so each can be called thousands of times for a stable best-of.
    """
    n_q, d = config.num_attention_heads, config.head_dim
    n_rep = config.num_kv_groups
    scale = d**-0.5
    q = torch.randn(1, n_q, 1, d)
    slot_mapping = torch.tensor(
        [cache.table.slot(p) for p in range(history)], dtype=torch.long
    )
    k_pool, v_pool = cache.k_pool[0], cache.v_pool[0]

    def gather_read() -> torch.Tensor:
        # Day-16 read: rebuild the contiguous history, then a normal masked SDPA.
        k = k_pool[slot_mapping].transpose(0, 1)[None]  # [1, n_kv, L, d]
        v = v_pool[slot_mapping].transpose(0, 1)[None]
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
        kv_len = k.shape[2]
        scores = torch.matmul(q, k.transpose(2, 3)) * scale
        causal = torch.full((1, kv_len), float("-inf"), dtype=scores.dtype)
        scores = scores + torch.triu(causal, diagonal=kv_len)  # past = kv_len-1
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, v)

    def fused_read() -> torch.Tensor:
        # Day-19 read: score the query over the scattered blocks in place.
        return paged_attention_reference(q, k_pool, v_pool, slot_mapping, n_rep, scale)

    return gather_read, fused_read


def sweep(lens, repeats, warmup, block_size):
    config = ModelConfig()
    rows = []
    for history in lens:
        cache = build_cache(config, history, block_size)
        gather_read, fused_read = make_reads(cache, config, history)
        # Same math either way (the Day-18/19 invariant); prove it before timing.
        assert torch.allclose(gather_read(), fused_read(), atol=1e-5), (
            f"gather and fused disagree at L={history}"
        )
        cmp = compare_reads(gather_read, fused_read, repeats=repeats, warmup=warmup)
        row = {
            "history_len": history,
            "gather_min_ms": round(cmp.gather.min_s * 1e3, 4),
            "gather_median_ms": round(cmp.gather.median_s * 1e3, 4),
            "fused_min_ms": round(cmp.fused.min_s * 1e3, 4),
            "fused_median_ms": round(cmp.fused.median_s * 1e3, 4),
            "speedup": round(cmp.speedup, 3),
            "faster": cmp.faster,
        }
        rows.append(row)
        print(
            f"  L={history:<6} gather {cmp.gather.min_s * 1e3:8.3f}ms  "
            f"fused {cmp.fused.min_s * 1e3:8.3f}ms  "
            f"-> {cmp.speedup:.2f}x ({cmp.faster})",
            file=sys.stderr,
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "history_len",
        "gather_min_ms",
        "gather_median_ms",
        "fused_min_ms",
        "fused_median_ms",
        "speedup",
        "faster",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="nanoserve paged-read microbenchmark")
    p.add_argument("--lens", type=_int_list, default=[16, 64, 256, 1024, 4096])
    p.add_argument("--repeats", type=int, default=50, help="timed calls per read")
    p.add_argument("--warmup", type=int, default=5, help="untimed calls before timing")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--csv", default="docs/daily/data/day-20-readbench.csv")
    args = p.parse_args()

    torch.manual_seed(0)
    print("paged-read microbenchmark (gather vs fused, CPU, fp32):", file=sys.stderr)
    rows = sweep(args.lens, args.repeats, args.warmup, args.block_size)

    out = Path(args.csv)
    write_csv(out, rows)
    print(f"\nwrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
