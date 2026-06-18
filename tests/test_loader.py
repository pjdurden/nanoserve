"""Day 3 tests: the weight name-mapping and the loaded tensors.

The pure-mapping tests run anywhere (no weights, but torch is imported by the
module). The end-to-end load test is gated by `requires_weights` and skips
cleanly when ./weights is absent.
"""

from __future__ import annotations

import pytest

from nanoserve.config import ModelConfig
from nanoserve.loader import (
    EMBED,
    LM_HEAD,
    expected_keys,
    expected_shapes,
    hf_to_nano,
    load_weights,
)

from reference import WEIGHTS_DIR, requires_weights, weights_available

CONFIG = ModelConfig()


# --- pure name-mapping logic (no weights needed) ----------------------------


@pytest.mark.parametrize(
    "hf_key,nano_key",
    [
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("model.norm.weight", "norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("model.layers.0.input_layernorm.weight", "layers.0.attn_norm.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "layers.0.attn.q_proj.weight"),
        ("model.layers.0.self_attn.k_proj.weight", "layers.0.attn.k_proj.weight"),
        ("model.layers.7.self_attn.o_proj.weight", "layers.7.attn.o_proj.weight"),
        ("model.layers.15.post_attention_layernorm.weight", "layers.15.mlp_norm.weight"),
        ("model.layers.15.mlp.down_proj.weight", "layers.15.mlp.down_proj.weight"),
    ],
)
def test_hf_to_nano_known_keys(hf_key, nano_key):
    assert hf_to_nano(hf_key) == nano_key


@pytest.mark.parametrize(
    "bad_key",
    [
        "model.layers.0.self_attn.qkv_proj.weight",  # fused name we do not emit
        "model.rotary_emb.inv_freq",  # buffers HF sometimes ships, not a param
        "lm_head.bias",
        "totally.unknown",
    ],
)
def test_hf_to_nano_rejects_unknown(bad_key):
    with pytest.raises(KeyError):
        hf_to_nano(bad_key)


def test_expected_key_count():
    # 16 layers * 9 params + embed + norm + lm_head = 147.
    keys = expected_keys(CONFIG)
    assert len(keys) == CONFIG.num_hidden_layers * 9 + 3 == 147
    assert LM_HEAD in keys and EMBED in keys


def test_expected_shapes_encode_gqa():
    shapes = expected_shapes(CONFIG)
    # q projects to 32*64=2048; k/v to 8*64=512. This asymmetry IS GQA.
    assert shapes["layers.0.attn.q_proj.weight"] == (2048, 2048)
    assert shapes["layers.0.attn.k_proj.weight"] == (512, 2048)
    assert shapes["layers.0.attn.v_proj.weight"] == (512, 2048)
    assert shapes["layers.0.mlp.gate_proj.weight"] == (8192, 2048)
    assert shapes["layers.0.mlp.down_proj.weight"] == (2048, 8192)


# --- end-to-end load against the real weights -------------------------------


@pytest.fixture(scope="module")
def weights():
    return load_weights(WEIGHTS_DIR)  # fp32 by default


@requires_weights
def test_load_is_complete_and_correct_shape(weights):
    cfg = weights.config
    assert set(weights.keys()) == expected_keys(cfg)
    for name, want in expected_shapes(cfg).items():
        assert tuple(weights[name].shape) == want, name


@requires_weights
def test_tied_lm_head_is_alias_not_copy(weights):
    # The whole point of the tie: lm_head shares storage with the embedding.
    assert weights[LM_HEAD].data_ptr() == weights[EMBED].data_ptr()
    # ... so the model does not pay for a second 128256x2048 matrix.
    assert weights.num_params < (weights[EMBED].numel() + sum(
        weights[k].numel() for k in weights.keys() if k not in (LM_HEAD, EMBED)
    )) + weights[EMBED].numel()


@requires_weights
def test_dtype_cast_applied(weights):
    import torch

    assert weights.dtype == torch.float32


@requires_weights
def test_mapping_points_at_the_right_source_tensor():
    """Right name is necessary but not sufficient; prove the bytes match too.

    Read a few tensors straight from the file under their HF names and confirm
    the loader filed them under the expected nanoserve names. This is what
    actually catches a transposed or swapped mapping (e.g. q<->k).
    """
    import torch
    from safetensors import safe_open

    nano = load_weights(WEIGHTS_DIR, dtype=None)  # native bf16, exact compare
    spot = {
        "model.layers.5.self_attn.k_proj.weight": "layers.5.attn.k_proj.weight",
        "model.layers.5.self_attn.q_proj.weight": "layers.5.attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight": "layers.0.mlp.gate_proj.weight",
        "model.norm.weight": "norm.weight",
    }
    path = next(p for p in WEIGHTS_DIR.glob("*.safetensors"))
    with safe_open(path, framework="pt") as f:
        for hf_key, nano_key in spot.items():
            assert torch.equal(f.get_tensor(hf_key), nano[nano_key]), nano_key


def test_load_skips_cleanly_without_weights():
    # Sanity that the gate itself works; only meaningful when weights are gone.
    if not weights_available():
        with pytest.raises(FileNotFoundError):
            load_weights(WEIGHTS_DIR / "does-not-exist")
