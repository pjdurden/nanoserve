"""Download Llama-3.2-1B weights into ./weights, then inspect them. Week 1, Day 2.

Usage:
    python scripts/download_weights.py

Llama is a gated repo. Accept the license on the model page, then either run
`huggingface-cli login` or set HF_TOKEN before this script. Weights land in
./weights (gitignored). After download it prints config.json and the safetensors
tensor inventory so you can sanity-check shapes against nanoserve.config.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ID = "meta-llama/Llama-3.2-1B"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Skip the optional/original checkpoints; we only need HF-format weights + tokenizer.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# config.json fields worth eyeballing against nanoserve.config.ModelConfig.
CONFIG_KEYS = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "rope_theta",
    "max_position_embeddings",
    "tie_word_embeddings",
    "torch_dtype",
    "rope_scaling",
]


def download() -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(WEIGHTS_DIR),
        allow_patterns=ALLOW_PATTERNS,
    )
    return Path(path)


def inspect(path: Path) -> None:
    cfg = json.loads((path / "config.json").read_text())
    print("\nconfig.json:")
    for key in CONFIG_KEYS:
        print(f"  {key}: {cfg.get(key)}")

    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors"))
    print(f"\n{len(files)} safetensors file(s):")
    total = 0
    sample_shown = False
    for f in files:
        with safe_open(f, framework="pt") as st:
            keys = list(st.keys())
            total += len(keys)
            print(f"  {f.name}: {len(keys)} tensors")
            if not sample_shown:
                print("  sample tensors (name, shape, dtype):")
                for k in keys[:8]:
                    sl = st.get_slice(k)
                    print(f"    {k}  {tuple(sl.get_shape())}  {sl.get_dtype()}")
                sample_shown = True
    print(f"\ntotal tensors: {total}")
    print(
        "\nexpected for Llama-3.2-1B: 1 safetensors file, "
        "weights tied (no separate lm_head.weight), "
        "16 layers x (q,k,v,o,gate,up,down + 2 norms)."
    )


if __name__ == "__main__":
    path = download()
    print(f"weights at: {path}")
    inspect(path)
