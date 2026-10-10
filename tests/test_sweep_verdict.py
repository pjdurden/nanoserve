"""Day 76: every row of the sweep carries its own verdict.

Day 75's smoke ran the burst branch of `graphbench.py`. Today's runs the sweep branch,
`--arrivals fixed --rates 200,400`, which is the shape of the card run, and it passed
the first time: the labels, the per-rate boots, `print_table`'s split columns and a
two-row CSV were all right. What it showed instead was what the table *leaves out*.
At both rates "the arms are comparable" and "the tail did not get worse" printed FAIL
in the log above the table, and the table and the CSV under it carried the ratios
alone. A row out of a run whose own gates refused it looked the same as a clean one,
and the CSV is the part that goes into a notebook without the log.

So the claims move out of `main` into two functions the script and this file share:

  1. **`claims(delta, args)`** is the list `main` used to build inline, two-arm gates
     then `split_claims`, as `(check, note)` pairs in the order they print.
  2. **`grade(pairs)`** runs each one and keeps `(note, exc or None)`. It catches the
     two refusals the gates raise and nothing else, so a bug in a check is still a
     crash and not a quiet FAIL.

The row gets two columns at its end, `claims_ok` (a count) and `claims_failed` (the
failed notes, joined), so yesterday's CSV header is still a prefix of today's. The
table gets a `claims` column that reads `11/13`. And `--sanity` now prints every claim
before it exits, rather than stopping at the first, because the second FAIL is often
the one that explains the first.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_split_script import _delta, _script

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.servebench import MeasurementUnsound

ARGS = SimpleNamespace(min_samples=20, tolerance=0.10)

#: What `_delta` fails and why: its arms carry empty `CaptureStats`, mode "off", so the
#: graphed arm has no capture for the three capture claims to be about. Everything else
#: in it is clean, and 4 clients of 8 gaps each clear a floor of 20.
NO_CAPTURE = ("the capture was used", "nothing recorded while serving",
              "the capture covered every step")


@pytest.fixture(scope="module")
def script():
    return _script()


# --- 1. the list the script grades -----------------------------------------------------


def test_a_two_arm_run_has_the_eight_two_arm_claims(script):
    notes = [note for _, note in script.claims(_delta(split=False), ARGS)]
    assert len(notes) == 8
    assert notes[0] == "the graphed arm batched"
    assert "same answers" in notes and "the tail did not get worse" in notes


def test_a_three_arm_run_appends_the_five_split_claims_after_them(script):
    two = [n for _, n in script.claims(_delta(split=False), ARGS)]
    three = [n for _, n in script.claims(_delta(), ARGS)]
    assert three[:8] == two
    assert len(three) == 13
    assert all("split" in n for n in three[8:])


def test_the_comparability_gate_reads_its_floor_off_the_args(script):
    """`_delta`'s arms have 32 frame gaps each, so the floor decides the verdict."""
    loose = dict(script.grade(script.claims(_delta(), ARGS)))
    strict = dict(script.grade(script.claims(_delta(), SimpleNamespace(min_samples=1000,
                                                                      tolerance=0.10))))
    assert loose["the arms are comparable"] is None
    assert isinstance(strict["the arms are comparable"], MeasurementUnsound)


# --- 2. grading ------------------------------------------------------------------------


def test_grade_keeps_every_note_in_order(script):
    pairs = script.claims(_delta(), ARGS)
    assert [n for n, _ in script.grade(pairs)] == [n for _, n in pairs]


def test_grade_keeps_both_kinds_of_refusal(script):
    def refuse(kind):
        def check():
            raise kind("no")
        return check

    out = script.grade([(refuse(AcceptanceFailure), "a"), (refuse(MeasurementUnsound), "b"),
                        (lambda: None, "c")])
    assert isinstance(out[0][1], AcceptanceFailure)
    assert isinstance(out[1][1], MeasurementUnsound)
    assert out[2] == ("c", None)


def test_a_bug_in_a_check_is_a_crash_and_not_a_fail(script):
    def broken():
        raise KeyError("itl_p99")

    with pytest.raises(KeyError):
        script.grade([(broken, "broken")])


# --- 3. the verdict on the row ---------------------------------------------------------


def test_the_verdict_columns_count_and_name_the_failures(script):
    graded = script.grade(script.claims(_delta(), ARGS))
    cols = script.verdict_columns(graded)
    assert cols["claims_ok"] == 13 - len(NO_CAPTURE)
    assert cols["claims_failed"] == "; ".join(NO_CAPTURE)


def test_a_clean_row_says_so_with_an_empty_cell(script):
    graded = [("a", None), ("b", None)]
    assert script.verdict_columns(graded) == {"claims_ok": 2, "claims_failed": "",
                                              "claims_unsound": ""}


def test_several_failures_are_joined_in_the_order_they_printed(script):
    graded = [("a", MeasurementUnsound("x")), ("b", None), ("c", AcceptanceFailure("y"))]
    assert script.verdict_columns(graded)["claims_failed"] == "a; c"


def test_a_note_never_holds_the_separator(script):
    """Otherwise `claims_failed.split("; ")` would not give the notes back."""
    for _, note in script.claims(_delta(), ARGS):
        assert ";" not in note


# --- 4. the table ----------------------------------------------------------------------


def _row(script, delta, load="1 rps"):
    graded = script.grade(script.claims(delta, ARGS))
    return {"load": load, **delta.row(), "replay_share": 1.0,
            **script.verdict_columns(graded)}


def test_the_table_has_a_claims_column_out_of_the_total(script, capsys):
    script.print_table([_row(script, _delta(), "1 rps"), _row(script, _delta(), "2 rps")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].rstrip().endswith("claims")
    assert all(line.rstrip().endswith("10/13") for line in lines[2:])


def test_a_two_arm_table_counts_out_of_eight(script, capsys):
    script.print_table([_row(script, _delta(split=False))])
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.rstrip().endswith("5/8")
