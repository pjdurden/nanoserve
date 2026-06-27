"""Interactive generation with nanoserve. Type a prompt, watch it stream.

Run from the repo root with the venv python so torch + the weights are found:

    cd ~/nanoserve && .venv/bin/python chat.py
    cd ~/nanoserve && .venv/bin/python chat.py --temperature 0.8 --top-p 0.95

Cached decode (Day 11) plus sampling (Day 10): the prompt is prefilled into the
KV cache once, then one token is produced per step and streamed as it lands.
`--temperature 0` is greedy. Ctrl-C or an empty line quits.
"""

import argparse
import sys

import torch  # noqa: F401  (imported so a missing-torch error is obvious early)

from nanoserve.config import ModelConfig
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers not installed in this interpreter; use .venv/bin/python")

p = argparse.ArgumentParser(description="interactive nanoserve generation")
p.add_argument("--max-new-tokens", type=int, default=100)
p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
p.add_argument("--top-k", type=int, default=0, help="0 = off")
p.add_argument("--top-p", type=float, default=1.0, help="1.0 = off")
p.add_argument("--seed", type=int, default=None, help="reproducible sampling")
args = p.parse_args()

print("loading nanoserve (Llama-3.2-1B)...", flush=True)
model = LlamaModel(ModelConfig.from_json("weights"), load_weights("weights"))
tok = AutoTokenizer.from_pretrained("weights")
eos_id = tok.eos_token_id

mode = "greedy" if args.temperature == 0 else (
    f"sampling (T={args.temperature}, top_k={args.top_k}, top_p={args.top_p})"
)
print(f"ready, {mode}. type a prompt (empty line or Ctrl-C to quit).")
while True:
    try:
        prompt = input("\nprompt> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not prompt.strip():
        break

    ids = tok(prompt, return_tensors="pt").input_ids
    sys.stdout.write(tok.decode(ids[0], skip_special_tokens=True))
    sys.stdout.flush()

    # Stream cached generation: each yielded id is decoded against the running
    # sequence so multi-token characters (and spacing) render correctly.
    out = ids
    for nxt in model.generate_stream(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=eos_id,
        seed=args.seed,
    ):
        if nxt == eos_id:
            break
        prev = tok.decode(out[0], skip_special_tokens=True)
        out = torch.cat([out, torch.tensor([[nxt]])], dim=1)
        full = tok.decode(out[0], skip_special_tokens=True)
        sys.stdout.write(full[len(prev):])
        sys.stdout.flush()
    print()
