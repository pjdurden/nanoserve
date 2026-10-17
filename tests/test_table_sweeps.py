"""Day 86: `--table` takes more than one CSV, and `--strict` checks all of them.

Day 85 gave `--table` an exit code: print the sweep's table, then fail if any row is
a `rerun` or `?`. It reads one file. A card booking produces several sweeps, one per
day or per card, and checking them one command at a time means one exit code per
command: the shell `&&` that guards a notebook over five sweeps has to be written
five times, and the first failure hides the other four tables.

So `--table` is `nargs="+"`:

  - **Each sweep's table prints under its path**, in the order the paths were given.
    One path prints no heading at all, exactly as Day 83 printed it: the heading is
    there to tell the tables apart, and with one table there is nothing to tell apart.
  - **Every file is parsed before any table prints.** A typo in the third path would
    otherwise leave two tables on stdout and the real message under them.
  - **`--strict` is one verdict over all the sweeps.** Every table prints, then the
    exit is 1 if any row anywhere is a `rerun` or `?`, and the sentence names each
    offending path with its own loads.
  - **The same path twice is refused.** The sweep would print twice under one heading
    and `--strict` would name it twice, and neither says anything the single path did
    not.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_split_script import _script
from test_sweep_marks import CLEAN, LOST, UNSOUND, _day76, _row
from test_table_strict import _csv as _one_csv

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def script():
    return _script()


def _csv(script, tmp_path, name, rows):
    """A sweep's file under its own name, so several of them can sit side by side."""
    where = tmp_path / name
    where.mkdir()
    return _one_csv(script, where, rows)


def _run(*args):
    return subprocess.run([sys.executable, str(REPO / "graphbench.py"), *args],
                          cwd=REPO, capture_output=True, text=True, timeout=120)


def _table(script, capsys, rows):
    capsys.readouterr()
    script.print_table(rows)
    return capsys.readouterr().out.splitlines()


# --- 1. the tables ---------------------------------------------------------------------


def test_one_path_prints_no_heading(script, tmp_path, capsys):
    """Day 83's output, unchanged: the path is not news when there is one of them."""
    path = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    want = _table(script, capsys, [_row(script, "1 rps", CLEAN)])
    done = _run("--table", str(path))
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == want


def test_each_sweep_prints_under_its_path(script, tmp_path, capsys):
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    second = _csv(script, tmp_path, "b", [_row(script, "2 rps", LOST)])
    done = _run("--table", str(first), str(second))
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == f"{first}:"
    assert lines[1:4] == _table(script, capsys, [_row(script, "1 rps", CLEAN)])
    assert lines[4] == ""
    assert lines[5] == f"{second}:"
    assert lines[6:9] == _table(script, capsys, [_row(script, "2 rps", LOST)])
    assert len(lines) == 9


def test_the_tables_come_in_the_order_the_paths_did(script, tmp_path):
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    second = _csv(script, tmp_path, "b", [_row(script, "2 rps", CLEAN)])
    out = _run("--table", str(second), str(first)).stdout
    assert out.index(f"{second}:") < out.index(f"{first}:")


# --- 2. one verdict over all of them ---------------------------------------------------


def test_strict_passes_when_every_sweep_is_results(script, tmp_path):
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    second = _csv(script, tmp_path, "b", [_row(script, "2 rps", LOST)])
    done = _run("--table", str(first), str(second), "--strict")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_strict_fails_on_the_second_sweep_after_both_tables(script, tmp_path):
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    second = _csv(script, tmp_path, "b", [_row(script, "2 rps", UNSOUND)])
    done = _run("--table", str(first), str(second), "--strict")
    assert done.returncode == 1
    assert f"{first}:" in done.stdout and f"{second}:" in done.stdout
    assert str(second) in done.stderr
    assert "2 rps" in done.stderr
    assert str(first) not in done.stderr


def test_the_strict_sentence_names_each_bad_path_with_its_own_loads(script, tmp_path):
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", UNSOUND)])
    second = _csv(script, tmp_path, "b", [_day76(script, "9 rps")])
    err = _run("--table", str(first), str(second), "--strict").stderr
    assert err.index("1 rps") < err.index("9 rps")
    assert err.index(str(first)) < err.index("1 rps")
    assert err.index(str(second)) < err.index("9 rps")


# --- 3. the refusals -------------------------------------------------------------------


def test_a_bad_path_is_refused_before_any_table_prints(script, tmp_path):
    """Two good sweeps and a third that is not a sweep: nothing printed, one sentence."""
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    second = _csv(script, tmp_path, "b", [_row(script, "2 rps", CLEAN)])
    done = _run("--table", str(first), str(second), str(tmp_path / "nope.csv"))
    assert done.returncode != 0
    assert done.stdout == ""
    assert "nope.csv" in done.stderr


def test_the_same_path_twice_is_refused(script, tmp_path):
    path = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    done = _run("--table", str(path), str(path))
    assert done.returncode != 0
    assert done.stdout == ""
    assert str(path) in done.stderr


# --- 4. the marks, per sweep -----------------------------------------------------------


def test_reprint_sweeps_returns_the_marks_per_path(script, tmp_path):
    first = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN)])
    second = _csv(script, tmp_path, "b", [_row(script, "2 rps", LOST),
                                          _row(script, "4 rps", UNSOUND)])
    got = script.reprint_sweeps([first, second])
    assert list(got) == [first, second]
    assert got == {first: ["clean"], second: ["lost", "rerun"]}


def test_reprint_still_returns_one_sweeps_marks(script, tmp_path):
    path = _csv(script, tmp_path, "a", [_row(script, "1 rps", CLEAN),
                                        _row(script, "2 rps", LOST)])
    assert script.reprint(path) == ["clean", "lost"]
