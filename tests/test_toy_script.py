"""Day 75: `graphbench.py --split-read` run as a process, over a checkpoint on disk.

Day 74 gave the script its third arm and tested it the cheap way: the padding's
arithmetic against a word counter, and the wiring through a monkeypatched
`build_app`. Neither of those is the script. The script parses flags, loads a
tokenizer with `AutoTokenizer.from_pretrained`, boots three servers in an order that
matters, and writes a CSV, and every one of those is a place a card run could fail
twenty minutes in for a reason no unit test saw.

So this file runs it. Two pieces:

  1. **A checkpoint the script cannot tell from a real one.** `write_toy_checkpoint`
     puts a `config.json`, a `model.safetensors` under HuggingFace's names and a
     byte-level `tokenizer.json` in a directory. Everything `build_app` and
     `padded_prompt` read comes off that directory through the same calls they make
     on `./weights`, and nothing is injected. The tokenizer maps one byte to one id,
     so a count is a byte count and the test can say what the padding should be.
  2. **The script, once, at a width that splits.** `--max-model-len 1024` is the
     narrowest width with a second 512-key chunk, `--arrivals burst` needs no rate,
     and `--allow-cpu` is the flag the script already has for exactly this. The
     smoke reads what a person would read: the order the arms loaded in, the
     padding line, the claims, and the CSV's header.

Nothing here needs a card, and nothing here says anything about speed: on CPU the
recorder is a stand-in and the split is tlsim. It is the harness being checked.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nanoserve.config import ModelConfig
from nanoserve.loader import EMBED, LM_HEAD, expected_keys, load_weights
from nanoserve.toyckpt import TOY_CONFIG, write_toy_checkpoint

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return write_toy_checkpoint(tmp_path_factory.mktemp("toy"))


# --- 1. a checkpoint the loader takes as a real one ------------------------------------


def test_the_directory_holds_what_a_downloaded_checkpoint_holds(checkpoint):
    names = {p.name for p in checkpoint.iterdir()}
    assert {"config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json"} <= names


def test_the_config_reads_back_as_the_toy_config(checkpoint):
    cfg = ModelConfig.from_json(checkpoint)
    for field in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "head_dim"):
        assert getattr(cfg, field) == getattr(TOY_CONFIG, field)
    assert cfg.tie_word_embeddings


def test_the_weights_load_through_the_real_loader_complete(checkpoint):
    weights = load_weights(checkpoint, dtype=torch.float32)
    assert set(weights.keys()) == expected_keys(TOY_CONFIG)


def test_the_head_is_tied_on_disk_the_way_the_1b_checkpoint_ties_it(checkpoint):
    """No `lm_head.weight` in the file, so the loader's alias path is the one taken,
    which is the path `./weights` takes."""
    from safetensors import safe_open

    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as fh:
        on_disk = set(fh.keys())
    assert "lm_head.weight" not in on_disk
    weights = load_weights(checkpoint, dtype=torch.float32)
    assert weights[LM_HEAD].data_ptr() == weights[EMBED].data_ptr()


def test_the_same_seed_writes_the_same_bytes(tmp_path):
    a = write_toy_checkpoint(tmp_path / "a", seed=3)
    b = write_toy_checkpoint(tmp_path / "b", seed=3)
    c = write_toy_checkpoint(tmp_path / "c", seed=4)
    blob = lambda d: (d / "model.safetensors").read_bytes()  # noqa: E731
    assert blob(a) == blob(b)
    assert blob(a) != blob(c)


def test_a_directory_that_already_holds_a_checkpoint_is_refused(checkpoint):
    """The writer's one way to hurt someone is `--weights ./weights` by mistake."""
    with pytest.raises(FileExistsError, match="config.json"):
        write_toy_checkpoint(checkpoint)


# --- 1b. a tokenizer the script loads the way it loads the real one --------------------


@pytest.fixture(scope="module")
def tokenizer(checkpoint):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint))


def test_one_byte_is_one_token(tokenizer):
    text = "The capital of France is"
    assert len(tokenizer.encode(text)) == len(text.encode())


def test_encode_adds_nothing_the_prompt_did_not_have(tokenizer):
    """A BOS here would make every count one more than the byte count, and the
    padding test below would be off by one in the direction that hides a bug."""
    assert tokenizer.encode("a") == tokenizer.encode("a", add_special_tokens=False)


def test_every_id_is_inside_the_model_vocabulary(tokenizer):
    ids = tokenizer.encode("café ☃ ok")
    assert ids and max(ids) < TOY_CONFIG.vocab_size


def test_text_survives_the_round_trip(tokenizer):
    text = "the quick brown fox"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_any_id_the_model_can_sample_decodes(tokenizer):
    """A random model samples every id in its vocabulary, and the detokenizer is
    downstream of all of them."""
    out = tokenizer.decode(list(range(TOY_CONFIG.vocab_size)))
    assert isinstance(out, str)


# --- 2. the script, as a process -------------------------------------------------------


@pytest.fixture(scope="module")
def smoke(checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke") / "split.csv"
    cmd = [
        sys.executable, str(REPO / "graphbench.py"),
        "--weights", str(checkpoint),
        "--device", "cpu", "--dtype", "float32", "--allow-cpu",
        "--split-read", "--arrivals", "burst",
        "--max-model-len", "1024", "--block-size", "16",
        "--max-batch-size", "4", "--num-blocks", "160",
        "--requests", "6", "--max-tokens", "8",
        "--no-compile", "--warm-rows", "4",
        "--csv", str(out),
    ]
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    return done, out


def test_the_script_exits_cleanly(smoke):
    done, _ = smoke
    assert done.returncode == 0, done.stderr[-3000:]


def test_the_split_arm_loads_before_the_other_two(smoke):
    """Its arena decides the prompt, and the other two must be sent the same one."""
    err = smoke[0].stderr
    split = err.index("loading the split arm")
    assert split < err.index("loading the graphs arm") < err.index("loading the eager arm")


def test_the_prompt_was_padded_past_the_chunk_before_the_graphed_arm_loaded(smoke):
    err = smoke[0].stderr
    line = next(x for x in err.splitlines() if "prompt padded to" in x)
    assert "512-key chunk" in line
    padded = int(line.split("prompt padded to ")[1].split()[0])
    # Byte-level, so the count is said in advance: `split_prompt` joins copies of the
    # 24-byte default prompt with a space, n copies are 25n - 1 bytes, and 21 is the
    # first n past the floor of 514.
    n = len(b"The capital of France is") + 1
    assert padded == 21 * n - 1 >= 514 > 20 * n - 1
    assert err.index("prompt padded to") < err.index("loading the graphs arm")


def test_every_split_claim_passed(smoke):
    out = smoke[0].stdout
    for note in ("the split arm batched", "the split arm gave the graphed arm's same answers",
                 "the split flag reached the cache", "the split arm's rows crossed a chunk",
                 "the split arm's boot grade is about the read that served"):
        assert f"ok    {note}" in out, out[-3000:]


def test_the_answers_matched_across_all_three_arms(smoke):
    assert "ok    same answers" in smoke[0].stdout


def test_the_csv_carries_the_split_columns(smoke):
    _, path = smoke
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    for column in ("graphs_itl_p50_ms", "eager_itl_p50_ms", "split_itl_p50_ms",
                   "split_itl_p50_speedup", "split_itl_p99_speedup", "replay_share"):
        assert column in rows[0]
    assert rows[0]["load"] == "burst"

