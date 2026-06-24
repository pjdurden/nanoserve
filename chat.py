"""Interactive greedy generation with nanoserve. Type a prompt, watch it stream.

Run from the repo root with the venv python so torch + the weights are found:

    cd ~/nanoserve && .venv/bin/python chat.py

Greedy decode, no cache yet (Week 3 adds that), so each new token re-runs the
whole prefix and generation slows as the text grows. Ctrl-C or empty line quits.
"""

import sys

import torch

from nanoserve.config import ModelConfig
from nanoserve.loader import load_weights
from nanoserve.model import LlamaModel

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers not installed in this interpreter; use .venv/bin/python")

MAX_NEW_TOKENS = 100

print("loading nanoserve (Llama-3.2-1B)...", flush=True)
model = LlamaModel(ModelConfig.from_json("weights"), load_weights("weights"))
tok = AutoTokenizer.from_pretrained("weights")
eos_id = tok.eos_token_id

print("ready. type a prompt (empty line or Ctrl-C to quit).")
while True:
    try:
        prompt = input("\nprompt> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not prompt.strip():
        break

    ids = tok(prompt, return_tensors="pt").input_ids
    shown = tok.decode(ids[0], skip_special_tokens=True)
    sys.stdout.write(shown)
    sys.stdout.flush()

    for _ in range(MAX_NEW_TOKENS):
        nxt = model.greedy_token(ids)              # next token id, shape [1]
        if nxt.item() == eos_id:
            break
        ids = torch.cat([ids, nxt[:, None]], dim=1)  # APPEND it (not +=)
        full = tok.decode(ids[0], skip_special_tokens=True)
        sys.stdout.write(full[len(shown):])        # print only the new piece
        sys.stdout.flush()
        shown = full
    print()
