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
from nanoserve.graphbench import DEFAULT_MIN_SAMPLES
from nanoserve.toyckpt import TOY_CONFIG, write_toy_checkpoint
from nanoserve.toysmoke import SMOKE_LOAD, WALL_CLOCK_FAILS, unexpected_failures

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
        *SMOKE_LOAD,
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


def test_every_arm_cleared_the_sample_floor(smoke):
    """Day 79: the load was picked so this holds. Three arms, three `(n=...)`."""
    counts = [int(x.split("(n=")[1].split(")")[0])
              for x in smoke[0].stdout.splitlines() if "(n=" in x]
    assert len(counts) == 3
    assert min(counts) >= DEFAULT_MIN_SAMPLES


def test_the_arms_are_comparable_on_the_smoke(smoke):
    assert "ok    the arms are comparable" in smoke[0].stdout, smoke[0].stdout[-3000:]


def test_the_smoke_printed_no_failure_it_was_not_allowed(smoke):
    assert unexpected_failures(smoke[0].stdout) == [], smoke[0].stdout[-3000:]


def test_the_csv_carries_the_split_columns(smoke):
    _, path = smoke
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    for column in ("graphs_itl_p50_ms", "eager_itl_p50_ms", "split_itl_p50_ms",
                   "split_itl_p50_speedup", "split_itl_p99_speedup", "replay_share"):
        assert column in rows[0]
    assert rows[0]["load"] == "burst"



# --- 3. Day 76: the sweep, as a process ------------------------------------------------
#
# The card run is `--rates 1,2,4,8 --arrivals ...`, not `--arrivals burst`, and the
# burst smoke above never reached the half of `main` a sweep takes: the label made
# from each rate, the loop that boots all three arms again per rate, `print_table`
# (skipped for one row), and a CSV with more than one row under one header. Two rates
# are the fewest that reach all of it. They are high, 200 and 400 per second, so six
# fixed arrivals land inside one step's worth of time and the split arm still batches.

SWEEP_RATES = ("200", "400")


@pytest.fixture(scope="module")
def sweep(checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep") / "sweep.csv"
    cmd = [
        sys.executable, str(REPO / "graphbench.py"),
        "--weights", str(checkpoint),
        "--device", "cpu", "--dtype", "float32", "--allow-cpu",
        "--split-read", "--arrivals", "fixed", "--rates", ",".join(SWEEP_RATES),
        "--max-model-len", "1024", "--block-size", "16",
        "--max-batch-size", "4", "--num-blocks", "160",
        *SMOKE_LOAD,
        "--no-compile", "--warm-rows", "4",
        "--csv", str(out),
    ]
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    return done, out


def _sweep_rows(sweep) -> list[dict]:
    with sweep[1].open() as fh:
        return list(csv.DictReader(fh))


def test_the_sweep_exits_cleanly(sweep):
    done, _ = sweep
    assert done.returncode == 0, done.stderr[-3000:]


def test_every_rate_boots_all_three_arms_in_order(sweep):
    """Each rate is its own `measure`, so each boots its own three servers, split
    first. A sweep that reused a server across rates would carry one rate's warm pool
    into the next rate's numbers."""
    err = sweep[0].stderr
    at = 0
    for rate in SWEEP_RATES:
        for arm in ("split", "graphs", "eager"):
            at = err.index(f"[{rate} rps] loading the {arm} arm", at)


def test_every_rate_pads_the_prompt_to_the_same_length(sweep):
    """The padding comes off the split arm's arena, which is booted again per rate.
    Same flags, same arena, same prompt: a different count would mean the two rows of
    the table were not sent the same work."""
    lines = [x for x in sweep[0].stderr.splitlines() if "prompt padded to" in x]
    assert len(lines) == len(SWEEP_RATES)
    assert len(set(lines)) == 1


def test_every_rate_reports_every_split_claim(sweep):
    out = sweep[0].stdout
    for note in ("the split arm batched", "the split arm gave the graphed arm's same answers",
                 "the split flag reached the cache", "the split arm's rows crossed a chunk",
                 "the split arm's boot grade is about the read that served", "same answers"):
        assert out.count(f"ok    {note}") == len(SWEEP_RATES), out[-3000:]


def test_the_sweep_prints_the_table_with_the_split_columns(sweep):
    out = sweep[0].stdout
    header = next(x for x in out.splitlines() if "ITL p50 graphs" in x)
    assert "ITL p50 split" in header and "vs graphs" in header
    below = out.splitlines()[out.splitlines().index(header) + 2:]
    for rate in SWEEP_RATES:
        assert any(x.strip().startswith(f"{rate} rps") for x in below), out[-3000:]


def test_the_sweep_csv_has_one_row_per_rate_under_one_header(sweep):
    rows = _sweep_rows(sweep)
    assert [r["load"] for r in rows] == [f"{rate} rps" for rate in SWEEP_RATES]
    assert all(set(r) == set(rows[0]) for r in rows)
    for r in rows:
        assert float(r["split_itl_p50_ms"]) > 0
        assert float(r["graphs_itl_p50_ms"]) > 0


def test_the_sweep_csv_header_is_the_burst_csv_header(smoke, sweep):
    """The two loads take different branches of `main`, and the columns must not
    depend on which: a sweep's CSV and a burst's CSV go into the same notebook."""
    with smoke[1].open() as fh:
        burst = next(csv.reader(fh))
    with sweep[1].open() as fh:
        swept = next(csv.reader(fh))
    assert swept == burst


def test_the_sweep_csv_names_the_gates_its_log_failed(sweep):
    """Day 76 wrote this against a smoke that failed the comparability gate on purpose
    (11 gaps against 20). Day 79's load clears the floor, so the only note a row may
    carry is one `WALL_CLOCK_FAILS` lets through, and the log must have printed it."""
    out = sweep[0].stdout
    for r in _sweep_rows(sweep):
        failed = [x for x in r["claims_failed"].split("; ") if x]
        assert "the arms are comparable" not in failed
        assert set(failed) <= set(WALL_CLOCK_FAILS)
        assert all(f"FAIL  {note}:" in out for note in failed)
        assert int(r["claims_ok"]) + len(failed) == 13


def test_the_sweep_printed_no_failure_it_was_not_allowed(sweep):
    assert unexpected_failures(sweep[0].stdout) == [], sweep[0].stdout[-3000:]


def test_the_sweep_table_shows_each_row_held_over_total(sweep):
    out = sweep[0].stdout
    lines = out.splitlines()
    header = next(x for x in lines if "ITL p50 graphs" in x)
    assert header.rstrip().endswith("claims")
    # Below the header, so the per-rate log heading ("200 rps:") is not the match.
    table = lines[lines.index(header) + 2:]
    for r in _sweep_rows(sweep):
        line = next(x for x in table if x.strip().startswith(r["load"]))
        assert line.rstrip().endswith(f"{r['claims_ok']}/13")


def test_the_verdict_columns_come_after_every_column_yesterday_had(sweep):
    header = list(_sweep_rows(sweep)[0])
    assert header[-2:] == ["claims_ok", "claims_failed"]
    assert header.index("replay_share") < header.index("claims_ok")
