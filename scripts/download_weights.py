"""Download Llama-3.2-1B weights into ./weights. Week 1, Day 1.

Usage:
    python scripts/download_weights.py

Requires `huggingface-cli login` (Llama is a gated repo, accept the license on
the model page first). Weights land in ./weights and are gitignored.
"""

# TODO(week1): use huggingface_hub.snapshot_download for meta-llama/Llama-3.2-1B
# into ./weights, then inspect config.json and the safetensors index.

if __name__ == "__main__":
    raise SystemExit("week1: implement snapshot_download for Llama-3.2-1B")
