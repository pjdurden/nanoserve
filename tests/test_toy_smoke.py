"""Day 79: the toy smoke clears the sample floor, and its one allowed FAIL is pinned.

Every process smoke since Day 75 sent six requests of eight tokens, and every one of
them printed `FAIL  the arms are comparable`: 11 frame gaps against a floor of 20. The
smokes knew, and Day 76 even wrote a test that the FAIL was there. That made the line
noise, and a line that is always red hides two things:

  - **a new FAIL next to it.** A reader skims past the red line they expect.
  - **a new reason under it.** `check_arms_comparable` checks the clients, the failed
    requests and the token counts *before* the floor. If the token counts ever
    differed, the smoke would print the same `FAIL  the arms are comparable` and every
    test would still pass.

So two pieces, both in `nanoserve.toysmoke`:

  1. **A load that clears the floor.** `SMOKE_LOAD` is six requests of sixteen tokens.
     On the byte-level toy a frame is not a token (a random byte is often half a UTF-8
     character, and the server sends no empty frame), so the count is measured, not
     multiplied out: 32 gaps per arm, deterministic because the decode is greedy.
  2. **The FAILs a smoke may print, each with its reason.** On a CPU the tail gate is a
     comparison of two p99s over 32 wall-clock samples, a coin flip, so it is allowed to
     fail, and only as the slower-tail refusal. `unexpected_failures` reads the log and
     returns every FAIL that is not that one. Every smoke asserts it is empty.
"""

from __future__ import annotations

from nanoserve.toysmoke import (
    SMOKE_LOAD,
    WALL_CLOCK_FAILS,
    failures,
    smoke_max_tokens,
    unexpected_failures,
)

TAIL = "the tail did not get worse"
TAIL_NOISE = (
    "  FAIL  the tail did not get worse:\n"
    "        the recorded arm's ITL p99 is 39.60 ms against the eager arm's 16.30 ms, "
    "which is 2.43x and past the 10% this comparison allows for noise\n"
)
TAIL_UNSOUND = (
    "  FAIL  the tail did not get worse:\n"
    "        one of these arms has no inter-token latency to compare: a run with no "
    "frame gaps in it cannot support a claim about a tail\n"
)
FLOOR = (
    "  FAIL  the arms are comparable:\n"
    "        the eager arm has 11 frame gap(s) and a p99 needs 20\n"
)
TOKENS = (
    "  FAIL  the arms are comparable:\n"
    "        the arms delivered different token counts (graphs 96, eager 95): "
    "inter-token latency is a ratio\n"
)
CLEAN = "  ok    the graphed arm batched\n  ok    same answers\n"


# --- 1. the load ----------------------------------------------------------------------


def test_the_smoke_load_is_six_requests_of_sixteen_tokens():
    assert SMOKE_LOAD == ("--requests", "6", "--max-tokens", "16")
    assert smoke_max_tokens() == 16


def test_the_load_is_flags_a_command_line_can_take():
    assert all(isinstance(x, str) for x in SMOKE_LOAD)
    assert SMOKE_LOAD[0::2] == ("--requests", "--max-tokens")


# --- 2. reading the log ---------------------------------------------------------------


def test_a_clean_log_has_no_failures():
    assert failures(CLEAN) == []
    assert unexpected_failures(CLEAN) == []


def test_a_failure_is_read_with_its_reason():
    assert failures(CLEAN + FLOOR) == [
        ("the arms are comparable", "the eager arm has 11 frame gap(s) and a p99 needs 20")
    ]


def test_the_tail_on_a_cpu_is_allowed_to_fail_for_its_own_reason():
    assert TAIL in WALL_CLOCK_FAILS
    assert unexpected_failures(CLEAN + TAIL_NOISE) == []


def test_the_tail_failing_for_another_reason_is_not_allowed():
    """No frame gaps at all is a broken run, not a slow one."""
    assert unexpected_failures(TAIL_UNSOUND) == [TAIL]


def test_the_sample_floor_is_no_longer_an_expected_failure():
    assert unexpected_failures(FLOOR) == ["the arms are comparable"]


def test_a_new_reason_under_the_old_note_is_caught():
    """The case the always-red line hid: same note, different refusal."""
    assert unexpected_failures(TOKENS) == ["the arms are comparable"]


def test_a_new_failure_next_to_the_allowed_one_is_caught():
    new = "  FAIL  the capture was used:\n        the graphed arm replayed 0 steps\n"
    assert unexpected_failures(TAIL_NOISE + new) == ["the capture was used"]


def test_every_rate_of_a_sweep_is_read():
    """A sweep prints its claims once per rate, so the same note can fail twice."""
    log = "200 rps:\n" + FLOOR + TAIL_NOISE + "400 rps:\n" + TAIL_NOISE + FLOOR
    assert unexpected_failures(log) == ["the arms are comparable"] * 2

