"""Day 85: `--table --strict` exits non-zero when any row is not a result.

Day 84 marked each row `clean`, `lost`, `rerun` or `?`, and the table prints the
marks. The exit code of `--table` still said 0 for a sweep full of reruns, so a script
that guards a notebook (`graphbench.py --table s.csv --strict && plot`) had to parse
stdout to find out. With `--strict`, a sweep with any `rerun` or `?` row still prints
its whole table, then exits 1 with one sentence on stderr naming the rows. `lost` is a
result and does not trip it: a capture that lost is a finding to plot, not a run to
do again. Without `--table`, `--strict` has nothing to check and is refused.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_split_script import _script
from test_sweep_marks import CLEAN, LOST, UNSOUND, _day76, _row

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def script():
    return _script()


def _csv(script, tmp_path, rows):
    """The file a sweep writes. A Day 76 row has no `claims_unsound` column at all."""
    raw = []
    for r in rows:
        r = {**r, "claims_failed": script.FAILED_SEP.join(r["claims_failed"])}
        if r["claims_unsound"] is None:
            del r["claims_unsound"]
        else:
            r["claims_unsound"] = script.FAILED_SEP.join(r["claims_unsound"])
        raw.append(r)
    path = tmp_path / "s.csv"
    script.write_csv(str(path), raw)
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(REPO / "graphbench.py"), *args],
                          cwd=REPO, capture_output=True, text=True, timeout=120)


def test_strict_passes_a_sweep_of_results(script, tmp_path):
    path = _csv(script, tmp_path, [_row(script, "1 rps", CLEAN), _row(script, "2 rps", LOST)])
    done = _run("--table", str(path), "--strict")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_strict_fails_a_rerun_after_printing_the_table(script, tmp_path):
    path = _csv(script, tmp_path, [_row(script, "1 rps", CLEAN), _row(script, "2 rps", UNSOUND)])
    done = _run("--table", str(path), "--strict")
    assert done.returncode == 1
    assert "rerun" in done.stdout
    assert "2 rps" in done.stderr
    assert "1 rps" not in done.stderr


def test_strict_fails_a_question_mark(script, tmp_path):
    path = _csv(script, tmp_path, [_day76(script, "4 rps")])
    done = _run("--table", str(path), "--strict")
    assert done.returncode == 1
    assert "4 rps" in done.stderr


def test_without_strict_a_rerun_still_exits_zero(script, tmp_path):
    path = _csv(script, tmp_path, [_row(script, "2 rps", UNSOUND)])
    assert _run("--table", str(path)).returncode == 0


def test_strict_without_table_is_refused():
    done = _run("--strict")
    assert done.returncode != 0
    assert "--table" in done.stderr


def test_reprint_returns_the_marks(script, tmp_path):
    path = _csv(script, tmp_path, [_row(script, "1 rps", CLEAN), _row(script, "2 rps", LOST)])
    assert script.reprint(path) == ["clean", "lost"]
