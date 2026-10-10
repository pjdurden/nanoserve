"""Day 80: the CSV tells a row that cannot be used from a row that lost.

Day 76 put the verdict on the row as `claims_ok` and `claims_failed`, and Day 79 made
the toy smokes pass every claim but one. The log can now tell the two kinds of FAIL
apart, because the reason is printed under the note: "the tail did not get worse"
over a slower p99 is a result, and the same note over "no frame gaps to compare" is a
broken run. The CSV could not. `claims_failed` carries notes, and a note does not say
which of the two refusals it was.

The two refusals mean different things to whoever reads the notebook:

  - **`AcceptanceFailure`**: the run measured what it set out to and the answer was
    no. The row's numbers are good, and they say the capture lost.
  - **`MeasurementUnsound`**: the run could not measure it. Too few gaps, a dropped
    request, arms that served different work. The row's numbers are not a result at
    all, and a plot that keeps them is plotting noise.

So the row gets a third verdict column, **`claims_unsound`**: the notes refused as
`MeasurementUnsound`, in print order, joined the same way. It is a subset of
`claims_failed`, so a reader filters `claims_unsound == ""` to keep the usable rows
and then reads `claims_failed` on what is left. It goes last, so Day 76's header is
still a prefix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_split_script import _delta, _script

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.servebench import MeasurementUnsound

ARGS = SimpleNamespace(min_samples=20, tolerance=0.10)

#: `_delta`'s arms have 32 frame gaps each, so a floor this high is a sample-size
#: refusal on a row that is otherwise the same.
STARVED = SimpleNamespace(min_samples=1000, tolerance=0.10)


@pytest.fixture(scope="module")
def script():
    return _script()


def test_the_verdict_has_a_third_column_and_it_goes_last(script):
    cols = script.verdict_columns([("a", None)])
    assert list(cols) == ["claims_ok", "claims_failed", "claims_unsound"]


def test_a_clean_row_has_an_empty_unsound_cell(script):
    assert script.verdict_columns([("a", None), ("b", None)])["claims_unsound"] == ""


def test_a_row_that_lost_is_failed_but_not_unsound(script):
    graded = [("a", AcceptanceFailure("slower")), ("b", None)]
    cols = script.verdict_columns(graded)
    assert cols["claims_failed"] == "a"
    assert cols["claims_unsound"] == ""


def test_a_row_that_could_not_measure_is_both(script):
    graded = [("a", None), ("b", MeasurementUnsound("11 gaps"))]
    cols = script.verdict_columns(graded)
    assert cols["claims_failed"] == "b"
    assert cols["claims_unsound"] == "b"


def test_unsound_notes_keep_print_order_and_the_same_separator(script):
    graded = [("a", MeasurementUnsound("x")), ("b", AcceptanceFailure("y")),
              ("c", MeasurementUnsound("z"))]
    cols = script.verdict_columns(graded)
    assert cols["claims_unsound"] == "a; c"
    assert cols["claims_unsound"].split(script.FAILED_SEP) == ["a", "c"]


def test_unsound_is_always_a_subset_of_failed(script):
    for args in (ARGS, STARVED):
        cols = script.verdict_columns(script.grade(script.claims(_delta(), args)))
        failed = [x for x in cols["claims_failed"].split(script.FAILED_SEP) if x]
        unsound = [x for x in cols["claims_unsound"].split(script.FAILED_SEP) if x]
        assert set(unsound) <= set(failed)


def test_the_real_claims_split_by_kind_on_the_real_delta(script):
    """`_delta`'s graphed arm has no capture, so the three capture claims lose. That is
    a result about the capture, not a broken measurement, and the row stays usable."""
    cols = script.verdict_columns(script.grade(script.claims(_delta(), ARGS)))
    assert cols["claims_failed"] != ""
    assert cols["claims_unsound"] == ""


def test_the_same_row_under_a_floor_it_cannot_clear_is_unsound(script):
    cols = script.verdict_columns(script.grade(script.claims(_delta(), STARVED)))
    assert cols["claims_unsound"] == "the arms are comparable"
    assert "the arms are comparable" in cols["claims_failed"]


def test_the_table_still_counts_held_over_total(script, capsys):
    """The new column is for the CSV. The table's `claims` cell reads off
    `claims_ok` and `claims_failed`, and an unsound note must not be counted twice."""
    graded = script.grade(script.claims(_delta(), STARVED))
    row = {"load": "1 rps", **_delta().row(), "replay_share": 1.0,
           **script.verdict_columns(graded)}
    script.print_table([row])
    assert capsys.readouterr().out.splitlines()[-1].rstrip().endswith("9/13")
