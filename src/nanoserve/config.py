"""Model config for Llama-3.2-1B. Week 1.

Holds the shapes and hyperparameters read from the HF config.json so the rest of
the engine never reaches back into transformers. Fill the defaults once you have
inspected the real config.json.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # TODO(week1): populate from Llama-3.2-1B config.json after inspecting it.
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0  # GQA: fewer KV heads than query heads
    head_dim: int = 0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 0
