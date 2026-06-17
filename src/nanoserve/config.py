"""Model config for Llama-3.2-1B. Week 1.

Holds the shapes and hyperparameters read from the HF config.json so the rest of
the engine never reaches back into transformers. The defaults are the published
Llama-3.2-1B values; `from_json` reads the real config.json so the numbers can
never drift away from the weights you actually loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RopeScaling:
    """Llama-3.2 'llama3' RoPE frequency rescaling.

    Llama 3.2 does not use plain RoPE: low frequencies are stretched so the 8k
    pretraining context extends to 131k. Day-4 RoPE must apply this, not just
    `rope_theta`, or long-context positions drift from the HF reference.
    """

    rope_type: str = "llama3"
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class ModelConfig:
    # Core shapes, defaulted to Llama-3.2-1B. Confirm against ./weights/config.json
    # via `ModelConfig.from_json` once the gated download is done.
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA: 8 KV heads shared across 32 query heads
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True  # lm_head shares embed_tokens weights
    torch_dtype: str = "bfloat16"
    rope_scaling: RopeScaling = field(default_factory=RopeScaling)

    def __post_init__(self) -> None:
        assert self.num_attention_heads * self.head_dim == self.hidden_size, (
            f"n_heads*head_dim ({self.num_attention_heads}*{self.head_dim}) "
            f"!= hidden_size ({self.hidden_size})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads (GQA)"
        )

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head; the GQA repeat factor (32 / 8 = 4 here)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """Build from a HF config.json (file or its parent directory)."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text())

        rs = raw.get("rope_scaling") or {}
        scaling = RopeScaling(
            rope_type=rs.get("rope_type", rs.get("type", "llama3")),
            factor=rs.get("factor", 32.0),
            low_freq_factor=rs.get("low_freq_factor", 1.0),
            high_freq_factor=rs.get("high_freq_factor", 4.0),
            original_max_position_embeddings=rs.get(
                "original_max_position_embeddings", 8192
            ),
        )
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get(
                "head_dim", raw["hidden_size"] // raw["num_attention_heads"]
            ),
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
            torch_dtype=raw.get("torch_dtype", "bfloat16"),
            rope_scaling=scaling,
        )
