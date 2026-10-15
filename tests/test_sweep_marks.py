"""Day 84: the table says which rows are results and which are reruns.

Day 80 put `claims_unsound` on the row: the failed notes the run could not measure,
as opposed to the ones it measured and lost. Day 83 made the table print from the row
`read_sweep` gives back. The table still showed one number for both kinds of FAIL,
`claims_ok` over the total, so a reprinted card sweep read `9/13` on a row that lost
four claims and on a row whose arms served different work, and only the second one is
not a result.

So each row gets a mark, in a `row` column just before `claims`:

  - **`clean`**: every claim held.
  - **`lost`**: some claims failed and every failure was measured. A result: compare
    it, plot it, explain it.
  - **`rerun`**: at least one failure could not be measured. Not a result: the row's
    numbers are not a comparison, so the run has to go again.
  - **`?`**: a Day 76 file, which did not record which failures were unmeasurable, on
    a row that has failures. A row with none is `clean` in any file: no failure means
    no unsound one.

Under the table, the rerun rows are listed again with the notes that made them
reruns, in table order: that is which claims to look at first. A `?` gets one line
saying why. A table with only results prints nothing under it, so a clean sweep reads
exactly as it did.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_split_script import _delta, _script

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.servebench import MeasurementUnsound

REPO = Path(__file__).resolve().parent.parent

CLEAN = (("a", None), ("b", None))
LOST = (("a", None), ("b", AcceptanceFailure("slower")))
UNSOUND = (("a", MeasurementUnsound("few gaps")), ("b", AcceptanceFailure("slower")),
           ("c", MeasurementUnsound("dropped")))


@pytest.fixture(scope="module")
def script():
    return _script()


def _row(script, load="1 rps", graded=CLEAN, split=False):
    """A row as `main` sends it to the table: built, then through `as_read`."""
    row = {"load": load, **_delta(split=split).row(), "replay_share": 0.875}
    row.update(script.verdict_columns(list(graded)))
    return script.as_read(row)


def _day76(script, load="1 rps", graded=LOST):
    row = _row(script, load, graded)
    row["claims_unsound"] = None
    return row


def _printed(script, capsys, rows):
    capsys.readouterr()
    script.print_table(rows)
    return capsys.readouterr().out.splitlines()


# --- 1. the mark ----------------------------------------------------------------------


@pytest.mark.parametrize(("graded", "mark"), [(CLEAN, "clean"), (LOST, "lost"),
                                              (UNSOUND, "rerun")],
                         ids=["clean", "lost", "rerun"])
def test_the_mark_reads_the_two_note_columns(script, graded, mark):
    assert script.row_mark(_row(script, graded=graded)) == mark


def test_a_day_76_row_with_failures_cannot_say(script):
    assert script.row_mark(_day76(script, graded=LOST)) == "?"


def test_a_day_76_row_with_no_failures_is_clean(script):
    """No failure means no unsound failure, whatever the file did not record."""
    assert script.row_mark(_day76(script, graded=CLEAN)) == "clean"


def test_the_mark_refuses_the_unsound_string(script):
    """`"few gaps"` is truthy and would read as a rerun, and its notes as letters."""
    row = _row(script, graded=UNSOUND)
    row["claims_unsound"] = "a; c"
    with pytest.raises(TypeError, match="as_read"):
        script.row_mark(row)


# --- 2. the column ---------------------------------------------------------------------


def test_the_row_column_sits_just_before_claims(script, capsys):
    header = _printed(script, capsys, [_row(script)])[0].split()
    assert header[-2:] == ["row", "claims"]


@pytest.mark.parametrize("split", [False, True], ids=["two-arm", "three-arm"])
def test_each_line_carries_its_mark_next_to_its_count(script, capsys, split):
    rows = [_row(script, "1 rps", CLEAN, split), _row(script, "2 rps", LOST, split),
            _row(script, "4 rps", UNSOUND, split)]
    lines = _printed(script, capsys, rows)[2:5]
    assert [line.split()[-2:] for line in lines] == [["clean", "2/2"], ["lost", "1/2"],
                                                    ["rerun", "0/3"]]


def test_the_columns_line_up(script, capsys):
    rows = [_row(script, "1 rps", CLEAN), _row(script, "2 rps", UNSOUND),
            _day76(script, "4 rps")]
    lines = _printed(script, capsys, rows)
    edge = lines[0].index(" row ") + len(" row")
    for line in [lines[0]] + lines[2:2 + len(rows)]:
        assert line[edge - 1] != " " and line[edge] == " "


# --- 3. under the table ----------------------------------------------------------------


def test_a_table_of_results_prints_nothing_under_it(script, capsys):
    rows = [_row(script, "1 rps", CLEAN), _row(script, "2 rps", LOST)]
    assert len(_printed(script, capsys, rows)) == 2 + len(rows)


def test_the_reruns_are_listed_with_the_notes_that_made_them(script, capsys):
    rows = [_row(script, "1 rps", UNSOUND), _row(script, "2 rps", LOST),
            _row(script, "4 rps", ((("z", MeasurementUnsound("x")),)))]
    under = _printed(script, capsys, rows)[2 + len(rows):]
    assert under[0] == ""
    assert "rerun" in under[1]
    assert [line.split(":", 1) for line in under[2:]] == [
        ["    1 rps", " a; c"],
        ["    4 rps", " z"],
    ]


def test_a_lost_note_is_not_listed_as_a_rerun(script, capsys):
    """`b` lost on the rerun row. Only the notes that could not measure are the
    reason it is a rerun."""
    under = _printed(script, capsys, [_row(script, graded=UNSOUND)])[3:]
    assert not any(line.rstrip().endswith("b") or "; b" in line for line in under)


def test_a_question_mark_gets_one_line_saying_why(script, capsys):
    under = _printed(script, capsys, [_row(script, "1 rps"), _day76(script, "2 rps")])[4:]
    assert under[0] == ""
    assert len(under) == 2
    assert under[1].startswith("?")
    assert "Day 80" in under[1]


def test_the_reruns_come_before_the_question_mark(script, capsys):
    rows = [_day76(script, "1 rps"), _row(script, "2 rps", UNSOUND)]
    under = _printed(script, capsys, rows)[4:]
    assert "rerun" in under[1]
    assert under[2].strip().startswith("2 rps")
    assert under[-1].startswith("?")


# --- 4. from the file ------------------------------------------------------------------


def test_table_mode_marks_a_rerun_from_its_file(script, tmp_path, capsys):
    built = [_row(script, "1 rps", LOST), _row(script, "2 rps", UNSOUND)]
    raw = [{**r, "claims_failed": script.FAILED_SEP.join(r["claims_failed"]),
            "claims_unsound": script.FAILED_SEP.join(r["claims_unsound"])} for r in built]
    path = tmp_path / "s.csv"
    script.write_csv(str(path), raw)
    want = _printed(script, capsys, built)
    done = subprocess.run(
        [sys.executable, str(REPO / "graphbench.py"), "--table", str(path)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == want
    assert "    2 rps: a; c" in done.stdout
