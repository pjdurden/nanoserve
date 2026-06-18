"""Weight loading: safetensors -> nanoserve's own named tensors. Week 1, Day 3.

Half of "loading a model" is just getting the names right. HuggingFace stores
Llama-3.2-1B as 146 tensors under names like
`model.layers.0.self_attn.q_proj.weight`. nanoserve wants its own, slightly
tidier namespace (`layers.0.attn.q_proj.weight`) that matches the module tree it
will build in Weeks 1-2. This module owns that translation, and nothing else in
the engine ever has to think about HF key strings again.

Three rabbit holes live here, and all three are about names, not math:

1. The leading `model.` prefix and `self_attn` -> `attn`, `input_layernorm` ->
   `attn_norm`, `post_attention_layernorm` -> `mlp_norm` renames.
2. Tied embeddings: Llama-3.2-1B ships *no* `lm_head.weight`. The output
   projection reuses the input embedding matrix. So 146 tensors on disk become
   147 named tensors in nanoserve, and the extra one is an alias, not a copy.
3. Separate vs fused QKV: HF keeps q_proj / k_proj / v_proj as three matrices.
   Some engines fuse them into one for a single matmul. nanoserve keeps them
   split for Week 1 (it mirrors HF exactly, which makes the 1e-5 verification
   trivial to reason about); fusing is a deliberate later optimization.

The deliverable is `Weights`: a dict-like container keyed by nanoserve names,
shape- and completeness-checked against `ModelConfig` at load time. Weeks 1-2
layers pull their parameters straight out of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .config import ModelConfig

# --- the explicit HF -> nanoserve name map ----------------------------------
#
# Per-layer parameters share the `model.layers.{i}.` prefix; everything else is
# a one-off. We translate the suffix (everything after the layer index) with
# this table, and the top-level tensors with the one below it. Keeping these as
# literal dicts (rather than clever string surgery) is the point: the mapping is
# meant to be read and audited, not inferred.

_LAYER_SUFFIX_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}

_TOP_LEVEL_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",  # absent on disk when embeddings are tied
}

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# The output projection's canonical name, and the input embedding it ties to.
LM_HEAD = "lm_head.weight"
EMBED = "embed_tokens.weight"


def hf_to_nano(hf_key: str) -> str:
    """Map one HuggingFace parameter name to its nanoserve name.

    Raises KeyError on anything unrecognized, on purpose: a key we do not know
    how to place is a loader bug, not something to silently drop.
    """
    m = _LAYER_RE.match(hf_key)
    if m:
        layer, suffix = m.group(1), m.group(2)
        if suffix not in _LAYER_SUFFIX_MAP:
            raise KeyError(f"unknown per-layer parameter suffix: {suffix!r} (in {hf_key!r})")
        return f"layers.{layer}.{_LAYER_SUFFIX_MAP[suffix]}"
    if hf_key in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[hf_key]
    raise KeyError(f"unrecognized HuggingFace key: {hf_key!r}")


def expected_keys(config: ModelConfig) -> set[str]:
    """The full set of nanoserve names a correctly loaded model must have.

    Includes `lm_head.weight` whether or not it exists on disk: nanoserve always
    has an output projection, even when it is an alias of the embeddings.
    """
    keys = {EMBED, "norm.weight", LM_HEAD}
    for i in range(config.num_hidden_layers):
        for suffix in _LAYER_SUFFIX_MAP.values():
            keys.add(f"layers.{i}.{suffix}")
    return keys


def expected_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Config-derived shape for every nanoserve tensor.

    A linear's weight is stored [out_features, in_features] (the HF/torch
    convention), so e.g. q_proj projects hidden -> n_heads*head_dim and lands as
    [n_heads*head_dim, hidden]. This is what catches a head-count or GQA mistake
    at load time instead of three days later in the attention math.
    """
    h = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    inter = config.intermediate_size

    shapes: dict[str, tuple[int, ...]] = {
        EMBED: (config.vocab_size, h),
        "norm.weight": (h,),
        LM_HEAD: (config.vocab_size, h),
    }
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        shapes[f"{p}.attn_norm.weight"] = (h,)
        shapes[f"{p}.attn.q_proj.weight"] = (q_dim, h)
        shapes[f"{p}.attn.k_proj.weight"] = (kv_dim, h)
        shapes[f"{p}.attn.v_proj.weight"] = (kv_dim, h)
        shapes[f"{p}.attn.o_proj.weight"] = (h, q_dim)
        shapes[f"{p}.mlp_norm.weight"] = (h,)
        shapes[f"{p}.mlp.gate_proj.weight"] = (inter, h)
        shapes[f"{p}.mlp.up_proj.weight"] = (inter, h)
        shapes[f"{p}.mlp.down_proj.weight"] = (h, inter)
    return shapes


def discover_safetensors(weights_dir: str | Path) -> list[Path]:
    """Return the safetensors shard(s) in load order.

    Supports both the single-file layout (Llama-3.2-1B, one `model.safetensors`)
    and the sharded layout (a `model.safetensors.index.json` listing many).
    """
    weights_dir = Path(weights_dir)
    index = weights_dir / "model.safetensors.index.json"
    if index.exists():
        import json

        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({weights_dir / fname for fname in weight_map.values()})
        return list(shards)
    shards = sorted(weights_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors found in {weights_dir}")
    return shards


def _read_raw(weights_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read every tensor from the shard(s) under their original HF names."""
    from safetensors.torch import load_file

    raw: dict[str, torch.Tensor] = {}
    for shard in discover_safetensors(weights_dir):
        raw.update(load_file(shard))
    return raw


class Weights:
    """A shape-checked, dict-like bag of nanoserve-named tensors.

    Built by `load_weights`. Indexing returns the tensor; the helpers below are
    sugar for the layer modules built in later weeks. The container is the
    contract between this file and every layer: layers ask for a canonical name
    and trust that it exists with the right shape.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], config: ModelConfig):
        self._t = tensors
        self.config = config

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._t[name]

    def __contains__(self, name: str) -> bool:
        return name in self._t

    def __len__(self) -> int:
        return len(self._t)

    def keys(self):
        return self._t.keys()

    def layer(self, i: int) -> dict[str, torch.Tensor]:
        """Every tensor belonging to block `i`, keyed by its in-block name.

        `layers.3.attn.q_proj.weight` comes back as `attn.q_proj.weight`.
        """
        prefix = f"layers.{i}."
        return {k[len(prefix):]: v for k, v in self._t.items() if k.startswith(prefix)}

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self._t.values())).dtype

    @property
    def num_params(self) -> int:
        """Distinct parameter elements (the tied lm_head is not double counted)."""
        seen: dict[int, int] = {}
        for t in self._t.values():
            seen[t.data_ptr()] = t.numel()
        return sum(seen.values())


def load_weights(
    weights_dir: str | Path,
    config: ModelConfig | None = None,
    *,
    dtype: torch.dtype | None = torch.float32,
) -> Weights:
    """Load Llama-3.2-1B safetensors into nanoserve-named, validated tensors.

    weights_dir: the ./weights directory (must hold config.json + safetensors).
    config:      ModelConfig; read from weights_dir/config.json if omitted.
    dtype:       cast every tensor to this dtype (default fp32, to match the
                 fp32 HF reference used for Week 1 verification). Pass None to
                 keep the on-disk dtype (bf16) untouched.

    Validates that every HF tensor maps to exactly one nanoserve name, that the
    full expected key set is present, and that every shape matches the config.
    Wires the tied output projection (lm_head -> embed_tokens) as an alias, not
    a copy, so the two never drift and the 250MB embedding is stored once.
    """
    if config is None:
        config = ModelConfig.from_json(weights_dir)

    raw = _read_raw(weights_dir)

    tensors: dict[str, torch.Tensor] = {}
    for hf_key, tensor in raw.items():
        nano_key = hf_to_nano(hf_key)
        if nano_key in tensors:
            raise ValueError(f"two HF keys mapped to {nano_key!r}; mapping is not injective")
        if dtype is not None:
            tensor = tensor.to(dtype)
        tensors[nano_key] = tensor

    # Tied embeddings: synthesize lm_head as a view of the input embedding.
    # `is` identity (shared storage) is the assertion that this is a tie, not a
    # second 128256x2048 matrix sitting in memory.
    if LM_HEAD not in tensors:
        if config.tie_word_embeddings:
            tensors[LM_HEAD] = tensors[EMBED]
        else:
            raise ValueError("lm_head.weight absent but config.tie_word_embeddings is False")

    _validate(tensors, config)
    return Weights(tensors, config)


def _validate(tensors: dict[str, torch.Tensor], config: ModelConfig) -> None:
    """Fail loudly on a missing key, an extra key, or a wrong shape."""
    want = expected_keys(config)
    have = set(tensors)
    missing, extra = want - have, have - want
    if missing:
        raise ValueError(f"missing {len(missing)} tensor(s): {sorted(missing)[:5]}...")
    if extra:
        raise ValueError(f"unexpected {len(extra)} tensor(s): {sorted(extra)[:5]}...")

    shapes = expected_shapes(config)
    for name, want_shape in shapes.items():
        got = tuple(tensors[name].shape)
        if got != want_shape:
            raise ValueError(f"{name}: shape {got} != expected {want_shape}")
