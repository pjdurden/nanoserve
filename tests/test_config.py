"""Day-2 tests: config self-consistency and config.json loading. No weights needed."""

import json

from nanoserve.config import ModelConfig, RopeScaling


def test_defaults_self_consistent():
    cfg = ModelConfig()
    assert cfg.num_attention_heads * cfg.head_dim == cfg.hidden_size
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    assert cfg.num_kv_groups == 4
    assert cfg.rope_scaling.rope_type == "llama3"


def test_from_json_matches_published_defaults(tmp_path):
    raw = {
        "vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    assert ModelConfig.from_json(path) == ModelConfig()


def test_from_json_accepts_directory(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 128256,
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
                "max_position_embeddings": 131072,
            }
        )
    )
    # head_dim derived from hidden_size / num_attention_heads when absent.
    cfg = ModelConfig.from_json(tmp_path)
    assert cfg.head_dim == 64
    assert cfg.rope_scaling == RopeScaling()  # defaults when rope_scaling absent
