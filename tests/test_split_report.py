"""Day 73: the split arm reads its own boot grades off `/health` and prints them.

Day 71 made a split server grade both passes before it serves, and Day 72 made it
time them. Both land in `/health` as `split_probe` and `split_timing`, and until
today the acceptance harness read neither. `run_arm` took the arena off the payload
(Day 68) and the read counters (Day 60) and left the two sections that say whether
the read is correct on this card and what it bought.

So `split_boot_from_health` lifts them into a `SplitBoot`, `ArmReport` carries it, and
the arm's row in the report grows a second line: graded on which backend, how far off,
predicted how much, measured how much. The first card run then puts "graded on
triton, predicted 4x, measured Nx" under the same row as the latency it paid for.

And one gate, `check_arm_split_graded`, for the one thing the harness can see that the
server can't: whether the backend that was graded at boot is the backend that served.
`check_boot_info` already holds the probe and the timing against each other. Only the
harness holds them against the read counters taken around a real crowd, because those
are two readings of one process, and a mismatch is a payload stitched from two.

The timing is reported and never enforced, for Day 72's reason: the timed case is four
rows and a serving batch is not, so a split that loses on it can still win.
"""

from __future__ import annotations

import json

import pytest
import torch
from test_split_boot import GRAPHS, _app

from nanoserve.acceptance import AcceptanceFailure
from nanoserve.captured import CaptureStats
from nanoserve.graphbench import (
    ArmReport,
    SplitBoot,
    check_arm_split_graded,
    split_boot_from_health,
)
from nanoserve.kernels.flash_decoding import SplitProbe, SplitTiming
from nanoserve.launch import boot_info
from nanoserve.reads import RECTANGLE, SPLIT, STREAMED, ReadStats
from nanoserve.servebench import LoadReport, MeasurementUnsound

#: What `SplitProbe.as_dict` publishes for a tlsim boot, after a trip through JSON.
PROBE = {
    "backend": "tlsim",
    "block": 16,
    "chunk": 16,
    "splits": 4,
    "head_dim": 64,
    "n_rep": 4,
    "dtype": "float32",
    "live_chunks": [1, 3, 4],
    "pass_one_error": 2.4e-7,
    "pass_two_error": 1.2e-7,
    "read_error": 3.0e-7,
    "atol": 1e-4,
}

#: What `SplitTiming.as_dict` publishes for the same boot: Day 72's host row.
TIMING = {
    "backend": "tlsim",
    "rows": 4,
    "splits": 4,
    "block": 16,
    "head_dim": 64,
    "n_rep": 4,
    "dtype": "float32",
    "tail_tiles": 1,
    "unsplit_tail_tiles": 4,
    "predicted": 4.0,
    "unsplit_ms": 11.64,
    "split_ms": 22.24,
    "measured": 11.64 / 22.24,
    "pays": False,
    "repeats": 5,
}


def _payload(probe=PROBE, timing=TIMING):
    payload = {"status": "ok", "split_workspace": {"splits": 2}}
    if probe is not None:
        payload["split_probe"] = dict(probe)
    if timing is not None:
        payload["split_timing"] = dict(timing)
    return payload


def _arm(mode=SPLIT, backend="tlsim", boot=None):
    splits = 2 if mode == SPLIT else 0
    block = 0 if mode == RECTANGLE else 16
    before = ReadStats(mode=mode, block=block, splits=splits)
    after = ReadStats(mode=mode, block=block, splits=splits, backend=backend,
                      calls=10, rows=30)
    return ArmReport(
        name=mode,
        texts={"a0": "x"},
        load=LoadReport(),
        before=CaptureStats(),
        after=CaptureStats(),
        boot={},
        read_before=before,
        read_after=after,
        split_boot=split_boot_from_health(_payload()) if boot is None else boot,
    )


# --- lifting it off the payload --------------------------------------------------------


def test_the_probe_and_the_timing_are_lifted_off_the_top_of_the_payload():
    boot = split_boot_from_health(_payload())
    assert boot.probe == PROBE
    assert boot.timing == TIMING
    assert boot.backend == "tlsim"


def test_a_payload_with_neither_lifts_as_empty():
    """The other two reads publish neither section, and that is a configuration."""
    boot = split_boot_from_health(_payload(probe=None, timing=None))
    assert boot == SplitBoot()
    assert not boot.graded
    assert boot.backend == ""
    assert boot.render() == ""


def test_the_lift_copies_so_the_report_does_not_alias_the_payload():
    payload = _payload()
    boot = split_boot_from_health(payload)
    payload["split_probe"]["backend"] = "triton"
    assert boot.backend == "tlsim"


def test_the_boot_numbers_survive_the_real_dataclasses_and_json():
    """The fixture is only worth testing against if it is what the server publishes."""
    probe = SplitProbe(
        backend="tlsim", block=16, chunk=16, splits=4, head_dim=64, n_rep=4,
        dtype=torch.float32, live_chunks=(1, 3, 4), pass_one_error=2.4e-7,
        pass_two_error=1.2e-7, read_error=3.0e-7, atol=1e-4,
    )
    timing = SplitTiming(
        backend="tlsim", rows=4, splits=4, block=16, head_dim=64, n_rep=4,
        dtype=torch.float32, tail_tiles=1, unsplit_tail_tiles=4, predicted=4.0,
        unsplit_ms=11.64, split_ms=22.24, repeats=5,
    )
    payload = json.loads(json.dumps(
        {"split_probe": probe.as_dict(), "split_timing": timing.as_dict()}
    ))
    boot = split_boot_from_health(payload)
    assert boot.probe == PROBE
    assert boot.timing == pytest.approx(TIMING)


def test_a_split_server_s_own_boot_info_lifts_whole():
    """A real split boot on the host, through `boot_info`, into the harness's record."""
    app = _app(split_read=True, **GRAPHS)
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace, app.state.split_probe, app.state.split_timing)
    boot = split_boot_from_health(json.loads(json.dumps(info)))
    assert boot.backend == "tlsim"
    assert boot.timing["backend"] == "tlsim"
    assert boot.predicted >= 2.0
    assert boot.measured > 0.0


# --- what the report prints ----------------------------------------------------------


def test_the_boot_line_names_the_grade_and_both_speedups():
    line = split_boot_from_health(_payload()).render()
    assert "graded on tlsim" in line
    assert "read 3.0e-07" in line
    assert "bound 1e-04" in line
    assert "predicted 4.00x" in line
    assert "measured 0.52x" in line


def test_a_split_that_lost_on_the_timed_case_says_so():
    line = split_boot_from_health(_payload()).render()
    assert "did not pay" in line


def test_a_split_that_won_on_the_timed_case_says_so():
    timing = dict(TIMING, unsplit_ms=40.0, split_ms=12.5, measured=3.2, pays=True)
    line = split_boot_from_health(_payload(timing=timing)).render()
    assert "measured 3.20x" in line
    assert "paid" in line and "did not pay" not in line


def test_a_graded_boot_with_no_timing_prints_that_it_was_not_timed():
    """The render is the report and never the gate: it prints what it has."""
    line = split_boot_from_health(_payload(timing=None)).render()
    assert "graded on tlsim" in line
    assert "not timed" in line


def test_a_split_arm_s_row_grows_a_second_line():
    lines = _arm().render().splitlines()
    assert len(lines) == 2
    assert lines[0].lstrip().startswith(SPLIT)
    assert "graded on tlsim" in lines[1]


def test_an_arm_with_no_boot_grades_is_one_line_as_before():
    arm = _arm(mode=STREAMED, boot=SplitBoot())
    assert len(arm.render().splitlines()) == 1


def test_the_row_columns_are_the_same_keys_either_way():
    """A CSV writer takes its header from the first row, so a sweep that mixes a split
    arm with another one needs the same columns on both, empty where nothing was
    graded."""
    graded = split_boot_from_health(_payload()).row()
    empty = SplitBoot().row()
    assert list(graded) == list(empty)
    assert graded["split_backend"] == "tlsim"
    assert graded["split_predicted"] == 4.0
    assert graded["split_measured"] == pytest.approx(0.523, abs=1e-3)
    assert graded["split_pays"] is False
    assert all(v is None for v in empty.values())


# --- the gate ---------------------------------------------------------------------------


def test_a_split_arm_graded_and_served_on_one_backend_passes():
    check_arm_split_graded(_arm())


def test_a_split_that_did_not_pay_still_passes_the_gate():
    """Reported, never enforced: the timed case is four rows and the batch is not."""
    timing = dict(TIMING, unsplit_ms=1.0, split_ms=100.0, measured=0.01, pays=False)
    check_arm_split_graded(_arm(boot=split_boot_from_health(_payload(timing=timing))))


@pytest.mark.parametrize("mode", [RECTANGLE, STREAMED])
def test_the_other_two_reads_fail_the_graded_gate(mode):
    with pytest.raises(AcceptanceFailure, match="not the split one"):
        check_arm_split_graded(_arm(mode=mode, backend="torch", boot=SplitBoot()))


def test_a_split_arm_with_no_probe_is_refused():
    boot = split_boot_from_health(_payload(probe=None, timing=None))
    with pytest.raises(AcceptanceFailure, match="no split_probe"):
        check_arm_split_graded(_arm(boot=boot))


def test_a_split_arm_graded_but_never_timed_is_refused():
    boot = split_boot_from_health(_payload(timing=None))
    with pytest.raises(AcceptanceFailure, match="no split_timing"):
        check_arm_split_graded(_arm(boot=boot))


def test_a_probe_over_its_bound_is_refused():
    """`build_app` refuses that boot, so a server that is up and publishing it is not
    the process that ran the probe."""
    probe = dict(PROBE, read_error=3e-3)
    boot = split_boot_from_health(_payload(probe=probe))
    with pytest.raises(AcceptanceFailure, match="3e-03 off"):
        check_arm_split_graded(_arm(boot=boot))


def test_a_timing_on_another_backend_than_the_probe_is_refused():
    boot = split_boot_from_health(_payload(timing=dict(TIMING, backend="triton")))
    with pytest.raises(AcceptanceFailure, match="timed on triton.*graded on tlsim"):
        check_arm_split_graded(_arm(boot=boot))


def test_a_boot_graded_on_one_backend_and_served_on_another_is_refused():
    """The one mismatch only the harness can see: the boot's grade against the read
    counters taken around a real crowd. `check_boot_info` never sees the counters."""
    with pytest.raises(AcceptanceFailure, match="graded on tlsim.*served on triton"):
        check_arm_split_graded(_arm(backend="triton"))


def test_a_split_arm_that_served_nothing_cannot_be_held_to_its_grade():
    """A window with no read in it has an empty backend: nothing to compare against."""
    arm = _arm(backend="")
    arm = ArmReport(**{**arm.__dict__, "read_after": arm.read_before})
    with pytest.raises(MeasurementUnsound, match="no decode reads"):
        check_arm_split_graded(arm)
