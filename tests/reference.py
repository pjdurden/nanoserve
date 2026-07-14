"""HF reference harness for numeric-equivalence tests. Week 1, Day 2.

Loads Llama-3.2-1B once via transformers (fp32, CPU) and captures intermediate
activations with forward hooks, so each Week 1 component test can assert its own
tensor is within ~1e-5 of the matching HF activation.

This is a helper module, not a test module (no `test_` prefix), so pytest does
not collect it directly. Component tests import `capture` and `requires_weights`.

Requires ./weights (run scripts/download_weights.py first). `requires_weights`
skips cleanly when the gated weights are absent (e.g. CI without the model).
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# A short fixed prompt so every test compares on the same activations.
# "<|begin_of_text|>The test of a" under the Llama-3 tokenizer.
PROMPT_IDS = [128000, 791, 1296, 315, 264]


def weights_available() -> bool:
    return WEIGHTS_DIR.exists() and any(WEIGHTS_DIR.glob("*.safetensors"))


requires_weights = pytest.mark.skipif(
    not weights_available(),
    reason="Llama-3.2-1B weights not in ./weights (run scripts/download_weights.py)",
)


def triton_gpu_available() -> bool:
    """True when a jitted kernel can actually be launched: Triton importable and a GPU.

    Both halves are needed and neither implies the other. Triton ships with the
    Linux GPU torch wheel, so a CPU-only wheel has no `triton` module at all; and a
    box can have the package while `torch.cuda.is_available()` is false (no device,
    or no driver). A kernel test needs both, so the gate asks for both.
    """
    from nanoserve.kernels.triton_paged_attention import has_triton

    return has_triton() and torch.cuda.is_available()


requires_triton_gpu = pytest.mark.skipif(
    not triton_gpu_available(),
    reason="the Triton paged-attention kernel needs a CUDA device and the triton package",
)


@functools.lru_cache(maxsize=1)
def hf_model():
    """Load the HF reference model once (fp32, CPU, eval)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR, dtype=torch.float32)
    model.eval()
    return model


def _resolve(model, dotted: str):
    """Walk a dotted path, treating integer segments as index access.

    "model.layers.0.input_layernorm" -> model.model.layers[0].input_layernorm
    """
    obj = model
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def capture(module_paths: dict[str, str], input_ids=None) -> dict[str, torch.Tensor]:
    """Run the HF model once and return {name: output tensor} per hooked module.

    `module_paths` maps a friendly name to a dotted attribute path on the HF
    model, e.g. {"rmsnorm0": "model.layers.0.input_layernorm"}. Tuple outputs
    (e.g. attention) are reduced to their first element.
    """
    if input_ids is None:
        input_ids = torch.tensor([PROMPT_IDS])
    model = hf_model()
    acts: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            acts[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    for name, path in module_paths.items():
        handles.append(_resolve(model, path).register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return acts
