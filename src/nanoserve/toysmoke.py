"""The load a toy smoke sends, and the FAILs it is allowed to print. Day 79.

Days 75 to 78 ran `graphbench.py` as a process over `write_toy_checkpoint`, four
smokes, every one with `--requests 6 --max-tokens 8`, and every one printed

    FAIL  the arms are comparable:
          the eager arm has 11 frame gap(s) and ...

The floor is 20 (`DEFAULT_MIN_SAMPLES`), it is a sample-size gate doing its job, and
it meant no smoke could pass every claim. A line that is always red stops being read,
and it hid two failures a smoke exists to catch: a new FAIL printed next to it, and a
new *reason* under the same note. `check_arms_comparable` refuses unequal clients,
dropped requests and unequal token counts before it gets to the floor, and all of
them print the same `FAIL  the arms are comparable`.

**A frame is not a token on the toy.** Its tokenizer is one id per byte, a random
model samples bytes, a lone byte is often half a UTF-8 character, and the server sends
no empty frame. So eight tokens were under three frames a request, and the gap count
is measured rather than multiplied out: sixteen tokens give 32 gaps per arm. It is the
same 32 on every run, because the decode is greedy and the frames are cut from the
text, not from the clock. The burst smoke went from 13 s to 16 s.

**What is still allowed to fail, and why only that.** "the tail did not get worse"
compares two p99s over 32 wall-clock samples on a CPU that is also running the
recorder's stand-in: the same run gave the graphed arm 2.43x the eager arm's p99 at
eight tokens and 0.59x at sixteen. That is a coin flip, so a smoke may print it, but
only as the slower-tail refusal. The same note refused as `MeasurementUnsound` (no gaps
to compare) is a broken run and is not allowed.
"""

from __future__ import annotations

#: The command-line load every toy smoke sends. Six requests, so four slots batch and
#: two wait; sixteen tokens, so each arm has 32 frame gaps against a floor of 20.
SMOKE_LOAD: tuple[str, ...] = ("--requests", "6", "--max-tokens", "16")

#: Note -> a fragment of the one refusal it may print in a smoke. Matched against the
#: reason line under the FAIL, so the same note failing for another reason is caught.
WALL_CLOCK_FAILS: dict[str, str] = {
    "the tail did not get worse": "ITL p99 is",
}


def smoke_max_tokens() -> int:
    """`--max-tokens` out of `SMOKE_LOAD`, for a test that wants the number."""
    return int(SMOKE_LOAD[SMOKE_LOAD.index("--max-tokens") + 1])


def failures(log: str) -> list[tuple[str, str]]:
    """Every `(note, reason)` the script's claims printed as FAIL, in order.

    `main` prints a failure as `  FAIL  <note>:` with the refusal on the next line, so
    the reason is the line after. A sweep prints its claims once per rate, and a note
    that failed at two rates is here twice.
    """
    lines = log.splitlines()
    out: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        head = line.strip()
        if not head.startswith("FAIL  "):
            continue
        note = head[len("FAIL  "):].removesuffix(":")
        reason = lines[i + 1].strip() if i + 1 < len(lines) else ""
        out.append((note, reason))
    return out


def unexpected_failures(log: str) -> list[str]:
    """The notes of every FAIL a toy smoke is not allowed to print.

    Empty is a clean smoke. A note in `WALL_CLOCK_FAILS` is let through only when its
    reason holds that note's fragment; anything else is returned, once per time it
    failed.
    """
    return [
        note
        for note, reason in failures(log)
        if note not in WALL_CLOCK_FAILS or WALL_CLOCK_FAILS[note] not in reason
    ]
