"""Day 77: `graphbench.py --compile dynamic` run as a process, and the arm it shortchanged.

Days 75 and 76 ran the script in a process and both passed `--no-compile`, so the
one path the card run takes that no smoke had reached was the compile: a
`CompiledDecode` built behind a booted server, a capture recorded over a compiled
forward. Today's smoke runs it, two arms, burst, the same toy checkpoint.

It exited 0 and every claim printed `ok`. The stderr said something else:

    torch._dynamo hit config.recompile_limit (8)
       function: 'inner' (.../torch/_dynamo/external_utils.py:67)
       last reason: 0/7: tensor 'args[0]' size mismatch at index 0. expected 1, actual 4

under `loading the eager arm`. Dynamo keeps its compiled entries per *code object*,
for the life of the process, and every arm runs the same `LlamaModel.forward`. The
graphed arm's warm-up filled entries for its buckets, the eager arm's own compile
added to the same list, and at eight dynamo gave the frame up and ran the eager
arm's forward in the interpreter for the rest of the run. So the arm that is the
control in a compiled comparison was the one arm not compiled, and only because it
booted second. No claim saw it, because no claim asked.

Three pieces:

  1. **Count it.** `dynamo_frames_abandoned` reads dynamo's own counter of frames it
     gave up on at the limit. Read around one arm, it says whether that arm ran the
     forward it was booted with.
  2. **Start each arm from an empty cache.** `measure` calls `torch._dynamo.reset()`
     before each compiled arm boots, which is what a process of its own would have
     given it. The order the arms boot in stops deciding which of them gets compiled.
  3. **Hold the run to it.** `ArmDelta.abandoned` carries the per-arm count, the
     render prints it, and `compile_claims` adds one gate, "every compiled arm stayed
     compiled", only when the run compiled at all. A `--no-compile` run's claim list
     is the one Day 76 had.

The process smoke here is about a minute and a half on this box, most of it inductor.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_split_script import _delta, _script

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.compiled import dynamo_frames_abandoned
from nanoserve.graphbench import ArmDelta, check_no_frame_abandoned, compile_claims
from nanoserve.toyckpt import write_toy_checkpoint

REPO = Path(__file__).resolve().parent.parent
ARGS = SimpleNamespace(min_samples=20, tolerance=0.10)
HELD = "every compiled arm stayed compiled"


def _with(delta: ArmDelta, abandoned) -> ArmDelta:
    return ArmDelta(graphs=delta.graphs, eager=delta.eager, split=delta.split,
                    abandoned=abandoned)


# --- 1. the counter ----------------------------------------------------------------------


def _walk_past_the_limit(limit: int, calls: int) -> None:
    """A static compile, the cheapest backend, a new shape every call: one entry per
    call until the limit, and then the frame is given up.

    Static and not an int guard, which was the first draft: under `dynamic=True` an
    int read through a closure goes symbolic, so the frame compiled once and the test
    counted nothing. A shape under `dynamic=False` is a guard on every version.

    It starts with a reset for the reason the day is about: `f` is one code object
    however many times this runs, and the first test to walk it past the limit would
    leave every later one a frame dynamo had already given up on.
    """
    torch._dynamo.reset()

    def f(x):
        return x * 2

    g = torch.compile(f, backend="eager", dynamic=False)
    with torch._dynamo.config.patch(recompile_limit=limit):
        for i in range(calls):
            g(torch.ones(i + 1))


def test_a_frame_walked_past_the_limit_is_counted_once():
    before = dynamo_frames_abandoned()
    _walk_past_the_limit(limit=2, calls=6)
    assert dynamo_frames_abandoned() - before == 1


def test_a_frame_that_stays_under_the_limit_is_not_counted():
    before = dynamo_frames_abandoned()
    _walk_past_the_limit(limit=8, calls=3)
    assert dynamo_frames_abandoned() == before


def test_a_reset_empties_the_cache_and_keeps_the_count():
    """The two halves of the fix: the arm after a reset gets the whole limit again,
    and the count read across it still says what the arm before it lost."""
    before = dynamo_frames_abandoned()
    _walk_past_the_limit(limit=2, calls=6)
    torch._dynamo.reset()
    assert dynamo_frames_abandoned() - before == 1


# --- 2. the delta and its gate -----------------------------------------------------------


def test_an_uncompiled_run_carries_no_count():
    assert _delta(split=False).abandoned is None


def test_no_compile_means_no_compile_claim():
    """Not a pass: there is nothing for it to be about."""
    assert compile_claims(_delta(split=False)) == []


def test_a_compiled_run_gets_the_one_compile_claim():
    delta = _with(_delta(split=False), {"graphs": 0, "eager": 0})
    assert [note for _, note in compile_claims(delta)] == [HELD]


def test_every_arm_held_passes():
    check_no_frame_abandoned(_with(_delta(), {"split": 0, "graphs": 0, "eager": 0}))


def test_the_failure_names_the_arm_that_lost_its_compile():
    delta = _with(_delta(split=False), {"graphs": 0, "eager": 1})
    with pytest.raises(AcceptanceFailure, match="eager arm") as info:
        check_no_frame_abandoned(delta)
    assert "graphs arm" not in str(info.value)


def test_the_render_says_what_each_arm_lost():
    plain = _delta(split=False)
    counted = _with(plain, {"graphs": 0, "eager": 1})
    extra = counted.render().splitlines()[len(plain.render().splitlines()):]
    assert extra == ["  compile  frames abandoned at the limit: graphs 0, eager 1"]


def test_the_row_does_not_change():
    """The count goes to the verdict columns through the claim, so a compiled CSV and
    a `--no-compile` one keep the same header."""
    plain = _delta()
    assert _with(plain, {"split": 0, "graphs": 1, "eager": 0}).row() == plain.row()


# --- 3. the script ------------------------------------------------------------------------


def test_the_script_appends_the_compile_claim_last():
    script = _script()
    plain = [n for _, n in script.claims(_delta(), ARGS)]
    compiled = [n for _, n in script.claims(_with(_delta(), {"split": 0, "graphs": 0,
                                                             "eager": 0}), ARGS)]
    assert compiled == [*plain, HELD]


@pytest.fixture
def staged(monkeypatch):
    """`measure` with the servers stubbed out and a counter that each boot can bump.

    `abandons` says how many frames each arm's boot gives up. The resets are logged in
    the same list as the boots, so the test can say which came first.
    """
    script = _script()
    events: list[str] = []
    state = {"count": 0}
    abandons = {"split": 0, "graphs": 0, "eager": 0}

    def build_one(args, *, graphs, split=False):
        name = "split" if split else ("graphs" if graphs else "eager")
        events.append(f"boot {name}")
        state["count"] += abandons[name]
        return SimpleNamespace(name=name)

    def serve_arm(args, app, name, plans, arrivals, rate):
        return getattr(_delta(), "split" if name == "split" else name)

    monkeypatch.setattr(script, "build_one", build_one)
    monkeypatch.setattr(script, "describe", lambda app: None)
    monkeypatch.setattr(script, "serve_arm", serve_arm)
    monkeypatch.setattr(script, "padded_prompt", lambda args, app: "p")
    monkeypatch.setattr(script, "dynamo_frames_abandoned", lambda: state["count"])
    monkeypatch.setattr(script, "reset_compile_cache", lambda: events.append("reset"))
    return script, events, abandons


def _measure_args(**kw):
    base = dict(split_read=True, prompt="p", requests=2, max_tokens=4, seed=0,
                arrivals="burst", no_compile=False, compile="dynamic")
    base.update(kw)
    return SimpleNamespace(**base)


def test_each_arm_is_charged_its_own_abandoned_frames(staged):
    script, _, abandons = staged
    abandons.update(graphs=0, eager=2)
    delta = script.measure(_measure_args(), "burst", None)
    assert delta.abandoned == {"split": 0, "graphs": 0, "eager": 2}


def test_every_compiled_arm_boots_after_a_reset(staged):
    script, events, _ = staged
    script.measure(_measure_args(), "burst", None)
    assert events == ["reset", "boot split", "reset", "boot graphs", "reset", "boot eager"]


@pytest.mark.parametrize("off", [dict(no_compile=True), dict(compile="off")])
def test_an_uncompiled_run_neither_resets_nor_counts(staged, off):
    script, events, _ = staged
    delta = script.measure(_measure_args(**off), "burst", None)
    assert "reset" not in events
    assert delta.abandoned is None


# --- 4. the script, compiled, as a process -----------------------------------------------


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    checkpoint = write_toy_checkpoint(tmp_path_factory.mktemp("toy"))
    out = tmp_path_factory.mktemp("compiled") / "compiled.csv"
    cmd = [
        sys.executable, str(REPO / "graphbench.py"),
        "--weights", str(checkpoint),
        "--device", "cpu", "--dtype", "float32", "--allow-cpu",
        "--arrivals", "burst",
        "--max-model-len", "1024", "--block-size", "16",
        "--max-batch-size", "4", "--num-blocks", "160",
        "--requests", "6", "--max-tokens", "8",
        "--compile", "dynamic", "--warm-rows", "4",
        "--csv", str(out),
    ]
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
    return done, out


def test_the_compiled_script_exits_cleanly(compiled):
    done, _ = compiled
    assert done.returncode == 0, done.stderr[-3000:]


def test_no_arm_hit_the_recompile_limit(compiled):
    """The line that started the day, read where it was found."""
    assert "recompile_limit" not in compiled[0].stderr, compiled[0].stderr[-3000:]


def test_the_log_reports_every_arm_held_its_compile(compiled):
    out = compiled[0].stdout
    assert "frames abandoned at the limit: graphs 0, eager 0" in out, out[-3000:]
    assert f"ok    {HELD}" in out


def test_the_compiled_capture_answered_like_the_compiled_forward(compiled):
    """The capture recorded over a compiled forward, and the two arms still agree."""
    out = compiled[0].stdout
    for note in ("same answers", "the capture was used", "nothing recorded while serving",
                 "the capture covered every step"):
        assert f"ok    {note}" in out, out[-3000:]


def test_the_compiled_csv_holds_the_claim(compiled):
    with compiled[1].open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert HELD not in row["claims_failed"]
    assert int(row["claims_ok"]) + len([x for x in row["claims_failed"].split("; ") if x]) == 9
