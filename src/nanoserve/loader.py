"""Weight loading: safetensors -> your own tensors. Week 1.

Maps HF parameter names to nanoserve's layer tensors. The name-mapping is the
first rabbit hole; keep a dict from HF keys to your module attributes.
"""


def load_weights(model, weights_dir):
    """Load Llama-3.2-1B safetensors into `model` in place."""
    raise NotImplementedError("week1: map safetensors keys to nanoserve layers")
