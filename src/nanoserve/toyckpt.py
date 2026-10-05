"""A checkpoint directory small enough to write in a test. Day 75.

Every server test before today booted through `build_app` with the loader and the
tokenizer injected: `load=lambda _dir, **_kw: _weights()` and a `ByteTokenizer` class
defined in the test file. That is the right trick for testing the wiring, and it is
the wrong one for testing a *script*, because a script cannot be handed a lambda. It
is handed a path, and everything it does after that (`ModelConfig.from_json`,
`load_weights`, `AutoTokenizer.from_pretrained`) is a read of that path.

So this writes one. Three files and the config:

  - `config.json`, the fields `ModelConfig.from_json` reads, with HuggingFace's names.
  - `model.safetensors`, under HuggingFace's tensor names and with no
    `lm_head.weight`, because Llama-3.2-1B ties its head and the loader's alias path
    is the one `./weights` takes.
  - `tokenizer.json` and `tokenizer_config.json`, a byte-level BPE with no merges:
    256 ids, one per byte, and no special tokens added on `encode`. A count is a byte
    count, so a test can say in advance how long a padded prompt will be. The
    vocabulary is the model's whole output space, so any id a random model samples
    decodes to something.

No end-of-sequence token, on purpose. A random model that could sample one would stop
a client at a step nobody chose, and a crowd test that needs every request alive at
once would pass or fail on the seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .loader import LM_HEAD, expected_shapes

#: Two layers, GQA by four, and a vocabulary of exactly one byte. Small enough that
#: three servers boot in seconds on a CPU; shaped like the real one in every way the
#: engine branches on (grouped KV heads, a tied head, more than one layer).
TOY_CONFIG = ModelConfig(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=4,
)

_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def _nano_to_hf(name: str) -> str:
    """The loader's map, read backwards, for the names `expected_shapes` produces."""
    from .loader import _LAYER_SUFFIX_MAP, _TOP_LEVEL_MAP

    top = {nano: hf for hf, nano in _TOP_LEVEL_MAP.items()}
    if name in top:
        return top[name]
    layer, _, suffix = name.partition(".")[2].partition(".")
    back = {nano: hf for hf, nano in _LAYER_SUFFIX_MAP.items()}
    return f"model.layers.{layer}.{back[suffix]}"


def _config_json(config: ModelConfig) -> dict:
    s = config.rope_scaling
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "rope_scaling": {
            "rope_type": s.rope_type,
            "factor": s.factor,
            "low_freq_factor": s.low_freq_factor,
            "high_freq_factor": s.high_freq_factor,
            "original_max_position_embeddings": s.original_max_position_embeddings,
        },
    }


def _write_tokenizer(directory: Path) -> None:
    """A byte-level BPE with an empty merge list: one byte in, one id out."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    # Keyed by the printable stand-in for each byte, so id `b` is byte `b`.
    vocab = {ch: b for b, ch in enumerate(_bytes_to_unicode())}
    assert set(vocab) == set(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1 << 20},
            indent=2,
        )
    )


def _bytes_to_unicode() -> list[str]:
    """GPT-2's printable stand-in for each byte, indexed by the byte.

    The byte-level pre-tokenizer does not see bytes; it sees these 256 characters,
    one per byte, chosen so none is whitespace or a control code. Listed here rather
    than imported so the id of every byte is fixed by this file.
    """
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(
        range(0xAE, 0xFF + 1)
    )
    out, extra = [], 0
    for b in range(256):
        if b in keep:
            out.append(chr(b))
        else:
            out.append(chr(256 + extra))
            extra += 1
    return out


def write_toy_checkpoint(
    directory: str | Path, *, seed: int = 0, config: ModelConfig = TOY_CONFIG
) -> Path:
    """Write a random checkpoint `build_app` loads exactly as it loads `./weights`.

    Refuses a directory that already holds any of the four files, because the one
    way to hurt someone with this function is to point it at the real checkpoint.
    The same seed writes the same bytes, so a run over it is repeatable.
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        if (directory / name).exists():
            raise FileExistsError(f"{directory / name} exists; refusing to overwrite a checkpoint")

    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in sorted(expected_shapes(config).items()):
        if name == LM_HEAD:
            continue  # tied: the loader aliases it to EMBED, as it does for the 1B
        t = torch.randn(*shape, generator=gen)
        if name.endswith("norm.weight"):
            t = 1.0 + 0.1 * t  # a norm near one, as trained norms are
        tensors[_nano_to_hf(name)] = t.contiguous()
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(_config_json(config), indent=2))
    _write_tokenizer(directory)
    return directory
