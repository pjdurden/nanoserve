"""Day 83: the table is printed from a row as `read_sweep` gives it back.

Day 82 finished typing the parser, and `print_table` still read the other shape: the
raw row `main` built, with `claims_failed` a string it split on `FAILED_SEP` by hand.
Two shapes for one row is one too many, and the hand split is where it shows: a typed
row's `claims_failed` is a list, and the same expression on a list would have asked a
list for `.split`. Worse the other way round: `len()` on the string counts characters.

So there is one shape, the one the file reads back as:

  1. **`as_read(row)`** is a row `main` just built, as `read_sweep` would return it
     from the CSV: the two note columns become lists, nothing else changes, because
     everything else is already the type it is written as. The test below is the
     round trip: `as_read(row) == read_sweep(write_csv(row))`.
  2. **`print_table` takes that shape and only that shape.** `claims_failed` is
     counted with `len`, and a string there is a `TypeError` naming the fix, rather
     than a count of characters.
  3. **`graphbench.py --table <csv>`** reprints an old sweep's table from its file,
     through `read_sweep`, with no weights and no device. A card run can be looked at
     again without being run again. A file the parser refuses, or one with no rows,
     exits with the parser's sentence and no traceback.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_split_script import _delta, _script

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def script():
    return _script()


def _built(script, delta, load="1 rps", graded=(("a", None),)):
    """A row the way `main` builds it, before anything reads it back."""
    row = {"load": load, **delta.row(), "replay_share": 0.875}
    row.update(script.verdict_columns(list(graded)))
    return row


def _written(script, tmp_path, rows, name="s.csv"):
    path = tmp_path / name
    script.write_csv(str(path), rows)
    return path


def _run(*argv):
    return subprocess.run(
        [sys.executable, str(REPO / "graphbench.py"), *argv],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )


SOME_FAIL = (("a", None), ("b", RuntimeError("x")), ("c", None))


# --- 1. one shape ----------------------------------------------------------------------


@pytest.mark.parametrize("split", [False, True], ids=["two-arm", "three-arm"])
def test_as_read_is_what_the_file_gives_back(script, tmp_path, split):
    from nanoserve.servebench import MeasurementUnsound

    graded = [("a", None), ("b", MeasurementUnsound("x")), ("c", RuntimeError("y"))]
    built = _built(script, _delta(split=split), graded=graded)
    (back,) = script.read_sweep(_written(script, tmp_path, [built]))
    assert script.as_read(built) == back


def test_as_read_types_the_notes_and_nothing_else(script):
    built = _built(script, _delta(), graded=SOME_FAIL)
    read = script.as_read(built)
    assert read["claims_failed"] == ["b"]
    assert read["claims_unsound"] == []
    assert {k: v for k, v in read.items() if not k.startswith("claims_")} == {
        k: v for k, v in built.items() if not k.startswith("claims_")
    }


def test_as_read_does_not_touch_the_row_main_writes(script):
    """`main` writes the CSV from the built rows after it prints the table."""
    built = _built(script, _delta(), graded=SOME_FAIL)
    script.as_read(built)
    assert built["claims_failed"] == "b"


# --- 2. the table reads the typed row ---------------------------------------------------


def test_the_table_counts_a_typed_row(script, capsys):
    script.print_table([script.as_read(_built(script, _delta(), graded=SOME_FAIL))])
    assert capsys.readouterr().out.splitlines()[-1].rstrip().endswith("2/3")


def test_the_table_refuses_the_raw_row_rather_than_count_its_characters(script):
    """`len("b")` is 1 by luck. `len("the tail did not get worse")` is 26."""
    with pytest.raises(TypeError, match="as_read"):
        script.print_table([_built(script, _delta(), graded=SOME_FAIL)])


def test_a_day_76_row_prints_without_an_unsound_column(script, capsys):
    """A pre-Day-80 file reads with `claims_unsound` as `None`. The table never read
    that column, so it must not start needing it."""
    row = script.as_read(_built(script, _delta(split=False), graded=SOME_FAIL))
    row["claims_unsound"] = None
    script.print_table([row])
    assert capsys.readouterr().out.splitlines()[-1].rstrip().endswith("2/3")


def test_the_table_from_the_file_is_the_table_from_the_run(script, tmp_path, capsys):
    built = [_built(script, _delta(), "1 rps", SOME_FAIL), _built(script, _delta(), "2 rps")]
    path = _written(script, tmp_path, built)
    capsys.readouterr()
    script.print_table([script.as_read(r) for r in built])
    live = capsys.readouterr().out
    script.print_table(script.read_sweep(path))
    assert capsys.readouterr().out == live


# --- 3. --table, as a process ----------------------------------------------------------


def test_table_mode_reprints_a_sweep_from_its_file(script, tmp_path, capsys):
    built = [_built(script, _delta(), "1 rps", SOME_FAIL), _built(script, _delta(), "2 rps")]
    path = _written(script, tmp_path, built)
    capsys.readouterr()
    script.print_table([script.as_read(r) for r in built])
    want = capsys.readouterr().out
    done = _run("--table", str(path))
    assert done.returncode == 0, done.stderr
    assert done.stdout == want


def test_table_mode_needs_no_weights_and_no_device(script, tmp_path):
    """No `--weights`, no `--allow-cpu`: the run is over, nothing boots."""
    path = _written(script, tmp_path, [_built(script, _delta(split=False))])
    done = _run("--table", str(path))
    assert done.returncode == 0, done.stderr
    assert "ITL p50 split" not in done.stdout
    assert "loading" not in done.stderr


def test_table_mode_refuses_a_file_the_parser_refuses(script, tmp_path):
    rows = [script.coverage_row(slots=2, requests=2, lengths=(4,))]
    path = _written(script, tmp_path, rows, "coverage.csv")
    done = _run("--table", str(path))
    assert done.returncode != 0
    assert "known verdict" in done.stderr
    assert "Traceback" not in done.stderr


def test_table_mode_refuses_a_file_with_no_rows(script, tmp_path):
    path = _written(script, tmp_path, [_built(script, _delta())])
    path.write_text(path.read_text().splitlines()[0] + "\n")
    done = _run("--table", str(path))
    assert done.returncode != 0
    assert "no rows" in done.stderr
    assert "Traceback" not in done.stderr


def test_table_mode_refuses_a_file_that_is_not_there(tmp_path):
    done = _run("--table", str(tmp_path / "nope.csv"))
    assert done.returncode != 0
    assert "nope.csv" in done.stderr
    assert "Traceback" not in done.stderr
