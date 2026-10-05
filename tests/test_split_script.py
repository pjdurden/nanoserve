"""Day 74: `graphbench.py` the script gets its third arm.

Day 68 put a split server on a socket next to the other two reads, and Day 73 made the
harness read its boot grades and hold them to what served. All of that ran inside
`tests/test_reads.py`. The script a card run would actually use still booted two
servers, graphs and eager, and neither one split. So the first real three-arm run
could not happen from the command line.

Three pieces, and each one is a thing the script would otherwise have got wrong:

  1. **The prompt has to clear the merge floor on its own.** `check_arm_split` refuses
     an arm whose rows all fit in the first chunk, and `DEFAULT_PROMPT` is six tokens.
     `split_prompt` repeats the prompt until it is at least `split_merge_floor` tokens
     by the server's own tokenizer. The prompt alone, and not prompt plus `max_tokens`:
     a completion can stop at EOS on its second token, and the client can only promise
     the half of the row it sends.
  2. **The split arm sits beside the graphed arm, not the eager one.** The split server
     boots with the same capture, buckets and persistent inputs as the graphed arm, and
     differs from it in the read alone. So `ArmDelta.split` is compared with
     `ArmDelta.graphs`, and a ratio against eager would be two changes at once.
  3. **The claims list grows by five, and only when the arm exists.** `split_claims`
     returns the split arm's gates in the order a reader should see them fail, and an
     empty list for a two-arm run, so the script needs no `if` around them.

Nothing here needs a card. The live three-arm run is still `tests/test_reads.py`;
this file is the script's arithmetic and its wiring.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_split_report import PROBE, TIMING, _payload

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.captured import CaptureStats
from nanoserve.launch import build_app, build_engine
from nanoserve.graphbench import (
    ArmDelta,
    ArmReport,
    SplitBoot,
    split_boot_from_health,
    split_claims,
    split_prompt,
)
from nanoserve.reads import RECTANGLE, SPLIT, ReadStats
from nanoserve.servebench import LoadReport, MeasurementUnsound, RequestRecord

#: Two 512-key chunks over a 1024-key width: the arena the script's split arm boots.
WORKSPACE = {
    "max_rows": 8,
    "heads": 32,
    "splits": 2,
    "keys_per_split": 512,
    "context_width": 1024,
    "block": 16,
    "partial_bytes": 4096,
}

ONE_CHUNK = {**WORKSPACE, "splits": 1, "keys_per_split": 1024}


def words(text: str) -> int:
    """A tokenizer that counts words. Enough to test the padding's arithmetic."""
    return len(text.split())


# --- 1. the prompt clears the floor by itself ------------------------------------------


def test_a_short_prompt_is_repeated_until_it_reaches_the_merge_floor():
    text = split_prompt(
        "The capital of France is", WORKSPACE, count=words, max_tokens=16,
        max_model_len=1024,
    )
    assert words(text) >= WORKSPACE["keys_per_split"] + 2
    # Whole copies, so the text is still the prompt and not a truncated tail of it.
    assert text.startswith("The capital of France is")
    assert words(text) % 5 == 0


def test_the_padding_stops_at_the_first_copy_past_the_floor():
    text = split_prompt("a b c", WORKSPACE, count=words, max_tokens=16, max_model_len=1024)
    assert words(text) - 3 < WORKSPACE["keys_per_split"] + 2 <= words(text)


def test_a_prompt_already_past_the_floor_is_sent_unchanged():
    long = " ".join(["w"] * 600)
    assert split_prompt(long, WORKSPACE, count=words, max_tokens=16,
                        max_model_len=1024) == long


def test_the_floor_is_cleared_by_the_prompt_alone_and_not_by_max_tokens():
    """A 500-word prompt plus 128 new tokens is past the floor on paper, and a
    completion that hits EOS on its second token leaves the row inside chunk one."""
    prompt = " ".join(["w"] * 500)
    text = split_prompt(prompt, WORKSPACE, count=words, max_tokens=128,
                        max_model_len=2048)
    assert words(text) >= 514


def test_a_one_chunk_arena_is_refused_before_any_load_is_built():
    with pytest.raises(MeasurementUnsound, match="one chunk"):
        split_prompt("hi", ONE_CHUNK, count=words, max_tokens=16, max_model_len=1024)


def test_an_empty_arena_is_refused_the_same_way():
    """A server that published no arena is not a split server, and has no floor."""
    with pytest.raises(MeasurementUnsound, match="one chunk"):
        split_prompt("hi", {}, count=words, max_tokens=16, max_model_len=1024)


def test_a_padded_prompt_that_would_not_fit_the_served_length_is_refused():
    """Server-side this is a 400 on every request, and the run would measure refusals."""
    with pytest.raises(MeasurementUnsound, match="max_model_len"):
        split_prompt("hi there", WORKSPACE, count=words, max_tokens=512,
                     max_model_len=1024)


def test_the_padding_fits_exactly_at_the_served_length():
    text = split_prompt("w", WORKSPACE, count=words, max_tokens=1024 - 514,
                        max_model_len=1024)
    assert words(text) == 514


def test_an_empty_prompt_cannot_be_padded():
    with pytest.raises(ValueError, match="empty"):
        split_prompt("", WORKSPACE, count=words, max_tokens=16, max_model_len=1024)


def test_a_counter_that_never_grows_is_refused_instead_of_looping_forever():
    with pytest.raises(ValueError, match="did not grow"):
        split_prompt("hi", WORKSPACE, count=lambda _: 1, max_tokens=16,
                     max_model_len=1024)


# --- 2. the split arm beside the graphed one -------------------------------------------


def _load(gap: float, n: int = 4, frames: int = 9) -> LoadReport:
    """`n` requests at one steady cadence, so p50 and p99 are both `gap`."""
    times = tuple(0.05 + k * gap for k in range(frames))
    records = tuple(
        RequestRecord(client_id=f"c{i}", scheduled_at=0.0, sent_at=0.0,
                      frame_times=times, done_at=times[-1], prompt_tokens=514,
                      output_tokens=frames, status=200)
        for i in range(n)
    )
    return LoadReport(records=records, started_at=0.0, finished_at=times[-1])


def _arm(name, mode, load, *, texts=None, boot=None, longest=600, peak=4):
    splits = 2 if mode == SPLIT else 0
    block = 0 if mode == RECTANGLE else 16
    return ArmReport(
        name=name,
        texts={"a0": "x", "a1": "y"} if texts is None else texts,
        load=load,
        before=CaptureStats(),
        after=CaptureStats(),
        boot={},
        peak_running=peak,
        read_before=ReadStats(mode=mode, block=block, splits=splits),
        read_after=ReadStats(mode=mode, block=block, splits=splits,
                             backend="tlsim" if mode == SPLIT else "torch",
                             calls=40, rows=120),
        longest_row=longest,
        workspace=dict(WORKSPACE) if mode == SPLIT else {},
        split_boot=boot if boot is not None else (
            split_boot_from_health(_payload()) if mode == SPLIT else SplitBoot()
        ),
    )


def _delta(split=True, **kw):
    graphs = _arm("graphs", RECTANGLE, _load(0.020))
    eager = _arm("eager", RECTANGLE, _load(0.030))
    arm = _arm("split", SPLIT, _load(0.016), **kw) if split else None
    return ArmDelta(graphs=graphs, eager=eager, split=arm)


def test_a_two_arm_delta_is_what_it_was():
    delta = _delta(split=False)
    assert delta.split is None
    assert not any(key.startswith("split") for key in delta.row())
    assert len(delta.render().splitlines()) == 5


def test_the_split_ratio_is_against_the_graphed_arm():
    delta = _delta()
    assert delta.split_itl_p50_speedup == pytest.approx(0.020 / 0.016)
    assert delta.split_itl_p99_speedup == pytest.approx(0.020 / 0.016)


def test_a_two_arm_delta_has_no_split_ratio():
    assert _delta(split=False).split_itl_p50_speedup == 0.0


def test_the_split_columns_carry_the_latency_and_the_boot_grades():
    delta = _delta()
    row = delta.row()
    assert row["split_itl_p50_ms"] == pytest.approx(16.0)
    assert row["split_itl_p99_ms"] == pytest.approx(16.0)
    assert row["split_tok_s"] == pytest.approx(delta.split.load.output_tps, abs=0.01)
    assert row["split_tok_s"] > row["graphs_tok_s"]
    assert row["split_itl_p50_speedup"] == pytest.approx(1.25)
    assert row["split_backend"] == PROBE["backend"]
    assert row["split_predicted"] == TIMING["predicted"]


def test_the_two_arm_columns_are_unchanged_by_the_third():
    """A CSV from yesterday's two-arm run and today's three-arm one share a prefix."""
    two, three = _delta(split=False).row(), _delta().row()
    assert list(three)[: len(two)] == list(two)


def test_every_row_of_a_three_arm_sweep_has_the_same_keys():
    """`csv.DictWriter` takes its header from the first row (Day 73)."""
    graded, ungraded = _delta().row(), _delta(boot=SplitBoot()).row()
    assert list(graded) == list(ungraded)


def test_the_render_puts_the_split_arm_and_its_grade_under_the_other_two():
    lines = _delta().render().splitlines()
    assert lines[0].strip().startswith("graphs")
    assert lines[1].strip().startswith("eager")
    assert lines[2].strip().startswith("split")
    assert "graded on tlsim" in lines[3]
    assert any("split vs graphs" in line and "1.25x" in line for line in lines)


# --- 3. the claims ---------------------------------------------------------------------


def _run(claims):
    """What the script does with the list: each one ok or the exception it raised."""
    out = []
    for check, note in claims:
        try:
            check()
            out.append((note, None))
        except (AcceptanceFailure, MeasurementUnsound) as exc:
            out.append((note, exc))
    return out


def test_a_two_arm_run_has_no_split_claims():
    assert split_claims(_delta(split=False)) == []


def test_a_good_split_arm_passes_all_five():
    results = _run(split_claims(_delta()))
    assert len(results) == 5
    assert all(exc is None for _, exc in results)


def test_every_claim_has_a_note_that_names_the_split_arm():
    for _, note in split_claims(_delta()):
        assert "split" in note


def test_the_split_answers_are_held_to_the_graphed_arm():
    results = dict(_run(split_claims(_delta(texts={"a0": "x", "a1": "z"}))))
    failed = [note for note, exc in results.items() if exc is not None]
    assert len(failed) == 1 and "same answers" in failed[0]


def test_rows_inside_the_first_chunk_fail_only_the_crossing_claim():
    results = _run(split_claims(_delta(longest=40)))
    failed = [(note, exc) for note, exc in results if exc is not None]
    assert len(failed) == 1
    assert isinstance(failed[0][1], MeasurementUnsound)
    assert "chunk" in failed[0][0]


def test_an_ungraded_split_arm_fails_the_grade_claim():
    results = _run(split_claims(_delta(boot=SplitBoot())))
    failed = [note for note, exc in results if exc is not None]
    assert failed == [n for n, _ in results if "grade" in n]


def test_a_split_arm_alone_in_its_batch_fails_the_crowd_claim():
    results = _run(split_claims(_delta(peak=1)))
    failed = [note for note, exc in results if exc is not None]
    assert len(failed) == 1 and "batched" in failed[0]


# --- the script's wiring ---------------------------------------------------------------


def _script():
    path = Path(__file__).resolve().parent.parent / "graphbench.py"
    spec = importlib.util.spec_from_file_location("graphbench_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("graphbench_script", None)
    spec.loader.exec_module(module)
    return module


def _args(**kw):
    base = dict(
        weights="./weights", device="cpu", dtype="auto", block_size=16,
        max_batch_size=8, max_model_len=2048, num_blocks=None, kv_cache_bytes=None,
        compile="dynamic", no_compile=False, no_compact=False, no_warm=False,
        warm_rows=None, warm_width=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def launched(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nanoserve.launch.build_app", lambda weights, **kw: calls.append(kw) or kw
    )
    return calls


def test_the_split_arm_is_the_graphed_arm_with_the_split_read_on(launched):
    script = _script()
    script.build_one(_args(), graphs=True)
    script.build_one(_args(), graphs=True, split=True)
    graphs, split = launched
    assert split.pop("split_read") is True
    assert graphs.pop("split_read") is False
    assert split == graphs


def test_the_eager_arm_never_splits(launched):
    script = _script()
    script.build_one(_args(), graphs=False)
    assert launched[0]["split_read"] is False


def test_a_split_arm_without_buckets_is_refused_by_the_script():
    """The split refuses to boot without `bucket_decode` (Day 65); say so first."""
    script = _script()
    with pytest.raises(SystemExit, match="bucket"):
        script.build_one(_args(), graphs=False, split=True)


def test_the_script_takes_the_flag():
    script = _script()
    parser = script.make_parser()
    assert parser.parse_args(["--split-read"]).split_read is True
    assert parser.parse_args([]).split_read is False


# --- Day 75: what the patched `build_app` above could not see -------------------------


def test_every_flag_the_script_hands_the_launcher_is_one_it_takes(launched):
    """The patch above takes any keyword, so `build_one` could pass a name nothing
    downstream knew and every test here still passed. Held to the real signatures:
    a keyword `build_app` does not name itself goes on to `build_engine`."""
    takes = set(inspect.signature(build_app).parameters) | set(
        inspect.signature(build_engine).parameters
    )
    script = _script()
    for graphs, split in ((True, True), (True, False), (False, False)):
        script.build_one(_args(), graphs=graphs, split=split)
    for kw in launched:
        assert set(kw) <= takes, set(kw) - takes


def test_the_graphed_arm_asks_for_the_capture_and_the_eager_arm_does_not(launched):
    """Buckets and persistent inputs are what a capture needs, not the capture. Without
    `capture_decode` the "graphs" arm boots with the graphs off and replays nothing."""
    script = _script()
    script.build_one(_args(), graphs=True)
    script.build_one(_args(), graphs=False)
    graphs, eager = launched
    assert graphs["capture_decode"] is True
    assert eager["capture_decode"] is False


def test_the_default_compile_mode_is_one_the_engine_accepts():
    """`--compile` defaulted to "default", which is `torch.compile`'s word and not
    `CompiledDecode`'s, so a card run without `--no-compile` would have refused to boot
    its first arm. The parser now only offers the engine's own modes."""
    from nanoserve.compiled import MODES

    parser = _script().make_parser()
    assert parser.parse_args([]).compile in MODES
    with pytest.raises(SystemExit):
        parser.parse_args(["--compile", "default"])
