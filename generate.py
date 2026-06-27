"""One-shot generation with nanoserve: cached decode + sampling, from the CLI.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python generate.py "The test of a"
    cd ~/nanoserve && .venv/bin/python generate.py "Once upon a time" \
        --temperature 0.8 --top-p 0.95 --seed 0 --max-new-tokens 80

This is the non-interactive sibling of chat.py. It prefills the prompt into the
KV cache (Day 11) once, then streams sampled tokens (Day 10) to stdout. Pass
`--temperature 0` for greedy decode (the zero-temperature corner of the sampler).
"""

import argparse
import sys
import time

import torch

from nanoserve.config import ModelConfig
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers not installed in this interpreter; use .venv/bin/python")


def main() -> None:
    p = argparse.ArgumentParser(description="one-shot nanoserve generation")
    p.add_argument("prompt", help="the text to continue")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=0, help="0 = off")
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 = off")
    p.add_argument("--seed", type=int, default=None, help="reproducible sampling")
    p.add_argument("--weights", default="weights", help="path to the weights dir")
    args = p.parse_args()

    print("loading nanoserve (Llama-3.2-1B)...", file=sys.stderr, flush=True)
    model = LlamaModel(ModelConfig.from_json(args.weights), load_weights(args.weights))
    tok = AutoTokenizer.from_pretrained(args.weights)

    ids = tok(args.prompt, return_tensors="pt").input_ids
    sys.stdout.write(tok.decode(ids[0], skip_special_tokens=True))
    sys.stdout.flush()

    n = 0
    t0 = time.perf_counter()
    out = ids
    for nxt in model.generate_stream(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=tok.eos_token_id,
        seed=args.seed,
    ):
        if nxt == tok.eos_token_id:
            break
        prev = tok.decode(out[0], skip_special_tokens=True)
        out = torch.cat([out, torch.tensor([[nxt]])], dim=1)
        full = tok.decode(out[0], skip_special_tokens=True)
        sys.stdout.write(full[len(prev):])
        sys.stdout.flush()
        n += 1
    dt = time.perf_counter() - t0
    rate = n / dt if dt else 0.0
    print(f"\n\n[{n} tokens in {dt:.1f}s, {rate:.2f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
