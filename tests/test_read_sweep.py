"""Day 81: a sweep CSV comes back through the parser the tests use.

Day 80 finished the verdict on the row: `claims_ok`, `claims_failed`, and the subset
of it the run could not measure, `claims_unsound`. Everything that read one of these
files back so far was a test doing `csv.DictReader` and `.split("; ")` by hand, and
the card sweep's notebook would have been one more copy of that.

So `graphbench.py` gets **`read_sweep(path)`**, the one place a CSV turns back into
rows:

  - **The verdict comes back typed.** `claims_ok` is an int, and the two note columns
    are lists, so a reader never splits on the separator and never mistakes `""` for
    one empty note.
  - **A header it does not know is refused.** The verdict columns have only ever been
    appended, so a file this code wrote ends in a prefix of `VERDICT_COLUMNS` of
    length two or three. Anything else (no verdict, a column moved, something after
    it) is a file whose rows cannot be filtered, and a notebook that plotted it
    anyway would be plotting rows nobody graded.
  - **A Day 76 file reads, and says what it does not know.** It has no
    `claims_unsound`, so that key comes back `None` and not `[]`: "not recorded" is a
    different answer from "nothing was unsound", and a notebook that wants usable
    rows has to decide what to do with the first.
  - **The subset is checked on the way in.** An unsound note that is not a failed
    note is a file that was edited, or written by something else.
"""

from __future__ import annotations

import csv

import pytest
from test_split_script import _delta, _script
from test_sweep_unsound import ARGS, STARVED

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.servebench import MeasurementUnsound


@pytest.fixture(scope="module")
def script():
    return _script()


def _write(path, header, rows):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _row(script, graded, load="1 rps"):
    return {"load": load, "graphs_itl_p50_ms": 1.5, **script.verdict_columns(graded)}


def test_the_verdict_columns_are_named_once(script):
    assert script.VERDICT_COLUMNS == ("claims_ok", "claims_failed", "claims_unsound")
    assert tuple(script.verdict_columns([("a", None)])) == script.VERDICT_COLUMNS


def test_a_written_sweep_reads_back_as_the_verdict_it_was_given(script, tmp_path):
    graded = [("a", None), ("b", AcceptanceFailure("slower")),
              ("c", MeasurementUnsound("11 gaps"))]
    path = tmp_path / "s.csv"
    script.write_csv(str(path), [_row(script, graded)])
    (row,) = script.read_sweep(path)
    assert row["claims_ok"] == 1
    assert row["claims_failed"] == ["b", "c"]
    assert row["claims_unsound"] == ["c"]


def test_a_clean_row_reads_as_empty_lists_not_one_empty_note(script, tmp_path):
    path = tmp_path / "s.csv"
    script.write_csv(str(path), [_row(script, [("a", None), ("b", None)])])
    (row,) = script.read_sweep(path)
    assert row["claims_ok"] == 2
    assert row["claims_failed"] == []
    assert row["claims_unsound"] == []


def test_the_other_columns_are_left_as_the_file_had_them(script, tmp_path):
    path = tmp_path / "s.csv"
    script.write_csv(str(path), [_row(script, [("a", None)], load="burst")])
    (row,) = script.read_sweep(path)
    assert row["load"] == "burst"
    assert row["graphs_itl_p50_ms"] == "1.5"
    assert list(row) == ["load", "graphs_itl_p50_ms", *script.VERDICT_COLUMNS]


def test_rows_keep_file_order(script, tmp_path):
    path = tmp_path / "s.csv"
    rows = [_row(script, [("a", None)], load=f"{r} rps") for r in (1, 2, 4)]
    script.write_csv(str(path), rows)
    assert [r["load"] for r in script.read_sweep(path)] == ["1 rps", "2 rps", "4 rps"]


def test_the_real_claims_round_trip_at_both_floors(script, tmp_path):
    for i, args in enumerate((ARGS, STARVED)):
        graded = script.grade(script.claims(_delta(), args))
        path = tmp_path / f"s{i}.csv"
        script.write_csv(str(path), [_row(script, graded)])
        (row,) = script.read_sweep(path)
        assert row["claims_failed"] == [n for n, e in graded if e is not None]
        assert row["claims_unsound"] == [
            n for n, e in graded if isinstance(e, MeasurementUnsound)
        ]
        assert row["claims_ok"] + len(row["claims_failed"]) == len(graded)


def test_a_day_76_file_reads_and_says_it_did_not_record_unsound(script, tmp_path):
    path = _write(tmp_path / "d76.csv", ["load", "claims_ok", "claims_failed"],
                  [["1 rps", "12", "the tail did not get worse"]])
    (row,) = script.read_sweep(path)
    assert row["claims_failed"] == ["the tail did not get worse"]
    assert row["claims_unsound"] is None


@pytest.mark.parametrize(
    "header",
    [
        ["load", "graphs_itl_p50_ms"],
        ["load", "claims_ok"],
        ["load", "claims_failed", "claims_ok", "claims_unsound"],
        ["load", "claims_ok", "claims_failed", "claims_unsound", "replay_share"],
        ["load", "claims_ok", "claims_unsound"],
    ],
    ids=["no-verdict", "half-verdict", "reordered", "column-after", "skipped-one"],
)
def test_a_header_that_is_not_a_known_prefix_is_refused(script, tmp_path, header):
    path = _write(tmp_path / "bad.csv", header, [["x"] * len(header)])
    with pytest.raises(ValueError, match="verdict"):
        script.read_sweep(path)


def test_an_unsound_note_that_did_not_fail_is_refused(script, tmp_path):
    path = _write(tmp_path / "edited.csv", ["load", *script.VERDICT_COLUMNS],
                  [["1 rps", "1", "a", "b"]])
    with pytest.raises(ValueError, match="subset"):
        script.read_sweep(path)


def test_an_empty_file_is_refused(script, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="header"):
        script.read_sweep(path)
