"""Day 82: a sweep CSV's numbers come back as numbers, and its holes as `None`.

Day 81 typed the verdict and left every other column as the file had it, which for a
CSV means a string. So the first cell of a card notebook was a cleaning step: `float`
on every `_ms` column, `int` on the counts, and a guess about what an empty split cell
means. That guess is the bug. An ungraded split boot writes `None`, `csv` writes that
as `""`, and `float("")` raises while `bool("False")` is `True`.

So `read_sweep` now knows the columns `ArmDelta.row()` writes:

  - **Measurements are floats.** Every `_ms`, `_speedup` and `_tok_s` column, plus
    `replay_share` and the split boot's three numbers.
  - **Counts are ints.** `itl_samples`, `graphs_held`, `recorded_while_serving`.
  - **`split_pays` is a bool,** read from the two words `csv` writes, and refused if
    it is any third word.
  - **The split boot's cells are nullable,** and only those: `NULLABLE_COLUMNS` is the
    key set of an ungraded `SplitBoot().row()`, so an empty one comes back `None`. An
    empty cell anywhere else is a measurement that is missing, and refused.
  - **Everything else is left alone.** `load`, and any column this file did not write.
"""

from __future__ import annotations

import csv

import pytest
from test_split_script import _delta, _script

from nanoserve.graphbench import SplitBoot, split_boot_from_health
from test_split_report import _payload


@pytest.fixture(scope="module")
def script():
    return _script()


def _sweep_row(script, delta, load="1 rps"):
    row = {"load": load, **delta.row(), "replay_share": 0.875}
    row.update(script.verdict_columns([("a", None)]))
    return row


def _round_trip(script, tmp_path, written):
    path = tmp_path / "s.csv"
    script.write_csv(str(path), [written])
    (row,) = script.read_sweep(path)
    return row


def _write(path, header, rows):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_nullable_columns_are_the_split_boots_own_keys(script):
    assert script.NULLABLE_COLUMNS == tuple(SplitBoot().row())


@pytest.mark.parametrize("split", [False, True], ids=["two-arm", "three-arm"])
def test_a_written_row_reads_back_equal_to_what_was_written(script, tmp_path, split):
    written = _sweep_row(script, _delta(split=split))
    row = _round_trip(script, tmp_path, written)
    for key in written:
        if key in script.VERDICT_COLUMNS:
            continue
        assert row[key] == written[key], key
        assert type(row[key]) is type(written[key]), key


def test_measurements_are_floats_and_counts_are_ints(script, tmp_path):
    row = _round_trip(script, tmp_path, _sweep_row(script, _delta()))
    for key in ("graphs_itl_p50_ms", "itl_p99_speedup", "eager_tok_s",
                "split_itl_p50_speedup", "replay_share", "split_read_error"):
        assert isinstance(row[key], float), key
    for key in ("itl_samples", "graphs_held", "recorded_while_serving"):
        assert type(row[key]) is int, key


def test_an_ungraded_split_boot_reads_as_none_not_empty_strings(script, tmp_path):
    row = _round_trip(script, tmp_path, _sweep_row(script, _delta(boot=SplitBoot())))
    for key in script.NULLABLE_COLUMNS:
        assert row[key] is None, key
    assert isinstance(row["split_itl_p50_ms"], float)


def test_a_graded_untimed_boot_keeps_its_grade_and_nulls_its_clock(script, tmp_path):
    boot = split_boot_from_health(_payload(timing=None))
    row = _round_trip(script, tmp_path, _sweep_row(script, _delta(boot=boot)))
    assert row["split_backend"] == boot.backend
    assert isinstance(row["split_read_error"], float)
    assert row["split_predicted"] is None
    assert row["split_measured"] is None
    assert row["split_pays"] is None


@pytest.mark.parametrize("cell,want", [("True", True), ("False", False), ("", None)])
def test_split_pays_reads_the_words_csv_writes(script, tmp_path, cell, want):
    path = _write(tmp_path / "p.csv", ["load", "split_pays", "claims_ok", "claims_failed"],
                  [["1 rps", cell, "1", ""]])
    (row,) = script.read_sweep(path)
    assert row["split_pays"] is want


def test_split_pays_refuses_a_third_word(script, tmp_path):
    path = _write(tmp_path / "p.csv", ["load", "split_pays", "claims_ok", "claims_failed"],
                  [["1 rps", "yes", "1", ""]])
    with pytest.raises(ValueError, match="split_pays"):
        script.read_sweep(path)


def test_an_empty_measurement_is_refused_and_named(script, tmp_path):
    path = _write(tmp_path / "hole.csv",
                  ["load", "graphs_itl_p50_ms", "claims_ok", "claims_failed"],
                  [["2 rps", "", "1", ""]])
    with pytest.raises(ValueError, match=r"graphs_itl_p50_ms.*'2 rps'|'2 rps'.*graphs_itl"):
        script.read_sweep(path)


def test_a_count_that_is_not_an_integer_is_refused(script, tmp_path):
    path = _write(tmp_path / "c.csv", ["load", "itl_samples", "claims_ok", "claims_failed"],
                  [["1 rps", "12.5", "1", ""]])
    with pytest.raises(ValueError, match="itl_samples"):
        script.read_sweep(path)


def test_a_word_in_a_measurement_is_refused(script, tmp_path):
    path = _write(tmp_path / "w.csv", ["load", "eager_tok_s", "claims_ok", "claims_failed"],
                  [["1 rps", "fast", "1", ""]])
    with pytest.raises(ValueError, match="eager_tok_s"):
        script.read_sweep(path)


def test_columns_it_did_not_write_are_left_as_strings(script, tmp_path):
    path = _write(tmp_path / "x.csv", ["load", "note", "claims_ok", "claims_failed"],
                  [["burst", "12", "1", ""]])
    (row,) = script.read_sweep(path)
    assert row["load"] == "burst"
    assert row["note"] == "12"
