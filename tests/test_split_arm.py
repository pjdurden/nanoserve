"""Day 68: the acceptance run's third arm, and the gate that says it really split.

Day 60 put two servers on a socket and compared their bytes: the rectangle read and
the streamed one. Day 66 made `--split-read` a thing a server can boot with, and
Day 67's "Tomorrow" left the third arm open. Adding it took five lines. Finding out
whether it tested anything took the day, because the obvious version doesn't:

  1. **At the file's own toy width the split has one chunk.** `choose_splits` never
     cuts a width shorter than twice `DEFAULT_PARTITION`, and `PLANS` runs at
     `max_model_len=32`. So the split server boots, plans `splits=1`, answers every
     byte correctly and reports `split x 1 splits` on `/health`. A one-chunk split is
     the streamed read followed by a reduce over one partial, and a reduce over one
     partial is the identity. The arm passes by never doing the thing it's named for.
  2. **At a width that does split, the rows still fit in the first chunk.** At
     `max_model_len=1024` the plan is two chunks of 512 keys, and a 5-byte prompt
     with 16 new tokens reads at most 20 of them. The second chunk walks nothing,
     stores `-inf, 0, 0`, and drops out of the reduce with no branch (Day 63 made
     that free on purpose). So the reduce is handed two partials and one of them is
     empty. Every row is still answered by a single chunk.
  3. **So the gate needs a row length, and the counters can't give it one.**
     `PagedRead._charge` counts shapes and never contents, because
     `(context_lens > keys_per_split).sum().item()` is a sync, and under a captured
     graph the Python doesn't run at all. The harness is the one party that knows how
     long its rows got: every streamed answer ends with a usage block, and
     `prompt_tokens + completion_tokens - 1` is the longest context the last decode
     step read. That number is what `ArmReport.longest_row` carries, and
     `check_arm_split` holds it against the chunk the server published.

The live runs are in `tests/test_reads.py`, next to Day 60's pair. This file is the
arithmetic, the refusals and the plumbing, none of which needs a socket.
"""

from __future__ import annotations

import pytest

from nanoserve.acceptance import AcceptanceFailure, ClientResult, CrowdReport
from nanoserve.captured import CaptureStats
from nanoserve.graphbench import (
    ArmReport,
    check_arm_split,
    longest_read,
    split_merge_floor,
    workspace_from_health,
)
from nanoserve.reads import RECTANGLE, SPLIT, STREAMED, ReadStats
from nanoserve.servebench import LoadReport, MeasurementUnsound

#: The split arm the live test runs: two 512-key chunks over a 1024-key width.
WORKSPACE = {
    "max_rows": 4,
    "heads": 8,
    "splits": 2,
    "keys_per_split": 512,
    "context_width": 1024,
    "block": 16,
    "partial_bytes": 1536,
}


def _read(mode=SPLIT, splits=2, calls=10):
    block = 0 if mode == RECTANGLE else 16
    before = ReadStats(mode=mode, block=block, splits=splits if mode == SPLIT else 0)
    after = ReadStats(
        mode=mode, block=block, splits=splits if mode == SPLIT else 0,
        backend="tlsim", calls=calls, rows=calls * 3,
    )
    return before, after


def _arm(mode=SPLIT, splits=2, longest_row=530, workspace=WORKSPACE, calls=10):
    read_before, read_after = _read(mode, splits, calls)
    return ArmReport(
        name=mode,
        texts={"a0": "x"},
        load=LoadReport(),
        before=CaptureStats(),
        after=CaptureStats(),
        boot={},
        peak_running=4,
        read_before=read_before,
        read_after=read_after,
        longest_row=longest_row,
        workspace=dict(workspace) if workspace is not None else {},
    )


# --- the row length the harness can see ---------------------------------------------


def _result(cid, prompt, completion, ok=True):
    return ClientResult(
        client_id=cid,
        status=200 if ok else 500,
        text="x",
        usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        error=None if ok else "boom",
    )


def test_the_longest_read_is_the_total_minus_the_token_nobody_read():
    """The last token a request samples is never written to the cache: the stream
    ends before a step could feed it back in. So a 510-token prompt with 16 tokens
    out had its widest decode read at 525 keys, not 526, and the one-off is the whole
    difference between a row that crossed a 525-key chunk and one that filled it."""
    crowd = CrowdReport(results={"a0": _result("a0", 510, 16)})
    assert longest_read(crowd) == 525


def test_the_longest_read_is_the_max_over_the_crowd():
    crowd = CrowdReport(results={
        "a0": _result("a0", 5, 16),
        "a1": _result("a1", 510, 12),
        "a2": _result("a2", 3, 12),
    })
    assert longest_read(crowd) == 521


def test_a_failed_client_s_usage_does_not_count():
    """Its row may never have reached that length: a 500 is a request that stopped
    somewhere, and the bill it was sent is not evidence of where."""
    crowd = CrowdReport(results={
        "a0": _result("a0", 5, 16),
        "a1": _result("a1", 900, 16, ok=False),
    })
    assert longest_read(crowd) == 20


def test_a_crowd_with_no_usage_read_nothing_the_harness_can_see():
    """0 and not an exception, so the gate is the one that refuses and it says why.
    An older server that streams no usage block is a harness that can't see its rows,
    which is the gate's business rather than the reader's."""
    crowd = CrowdReport(results={"a0": ClientResult(client_id="a0", status=200, text="x")})
    assert longest_read(crowd) == 0


# --- the workspace the server published ------------------------------------------------


def test_the_workspace_is_lifted_off_the_top_of_the_payload():
    """Top level, not under `cuda_graphs`, which is Day 66's placement: the arena
    belongs to the read, and a split server with the graphs off holds it all the same."""
    payload = {"status": "ok", "split_workspace": WORKSPACE, "cuda_graphs": {"shapes": 3}}
    assert workspace_from_health(payload) == WORKSPACE


def test_a_payload_with_no_workspace_lifts_as_empty():
    assert workspace_from_health({"status": "ok"}) == {}


def test_the_merge_floor_is_one_key_past_the_chunk_plus_the_unread_token():
    """The shortest request, prompt plus completion, whose last read reaches a second
    chunk. With 512-key chunks that read has to be 513 keys wide, and one more token
    was sampled and never read, so 514."""
    assert split_merge_floor(WORKSPACE) == 514


def test_the_merge_floor_of_a_one_chunk_arena_does_not_exist():
    """A single chunk covers the whole width, so no row that fits the server crosses
    it. `None` rather than `context_width + 2`, which would be a length the server
    refuses."""
    assert split_merge_floor({**WORKSPACE, "splits": 1, "keys_per_split": 1024}) is None


# --- the gate --------------------------------------------------------------------------


def test_a_split_arm_whose_rows_crossed_a_chunk_passes():
    check_arm_split(_arm(longest_row=530))


def test_a_one_chunk_split_is_refused_by_name():
    """The first finding of the day. The server's answers are right, its `/health`
    says `split`, and it ran the streamed loop plus a reduce over one partial, which
    is the identity. A `MeasurementUnsound` and not an `AcceptanceFailure`: nothing
    about the server is wrong, the configuration just can't be used to say anything
    about the combine."""
    with pytest.raises(MeasurementUnsound, match="one chunk"):
        check_arm_split(_arm(splits=1, workspace={**WORKSPACE, "splits": 1,
                                                  "keys_per_split": 1024}))


def test_a_split_arm_whose_rows_all_fit_in_the_first_chunk_is_refused():
    """The second finding. Two chunks planned, and every row read at most 20 keys of
    a 512-key first chunk. The second chunk stored `-inf, 0, 0` on every call and the
    reduce merged one live partial per row, every time."""
    with pytest.raises(MeasurementUnsound, match="first chunk"):
        check_arm_split(_arm(longest_row=20))


def test_a_row_that_exactly_fills_the_first_chunk_did_not_cross_it():
    with pytest.raises(MeasurementUnsound, match="first chunk"):
        check_arm_split(_arm(longest_row=512))


def test_a_row_one_key_into_the_second_chunk_did():
    check_arm_split(_arm(longest_row=513))


def test_an_arm_with_no_visible_rows_is_refused_as_unwitnessed():
    """0 is what `longest_read` returns for a server that streamed no usage. That's
    a harness that can't see its rows, and passing it would be the gate agreeing with
    a number it never got."""
    with pytest.raises(MeasurementUnsound, match="usage"):
        check_arm_split(_arm(longest_row=0))


def test_a_split_arm_that_published_no_workspace_is_refused():
    with pytest.raises(MeasurementUnsound, match="split_workspace"):
        check_arm_split(_arm(workspace=None))


@pytest.mark.parametrize("mode", [RECTANGLE, STREAMED])
def test_the_other_two_reads_fail_the_split_gate(mode):
    """The control, in the same shape `check_arm_replayed` has for the eager arm. A
    gate the rectangle arm passes is a gate that stopped distinguishing anything."""
    with pytest.raises(AcceptanceFailure, match="split"):
        check_arm_split(_arm(mode=mode, workspace=None))


def test_a_split_arm_that_ran_no_reads_is_refused():
    with pytest.raises(MeasurementUnsound, match="no decode reads"):
        check_arm_split(_arm(calls=0))


def test_the_published_chunk_count_must_match_the_read_s():
    """Two sections of one payload, one fixed at boot and one counted since. They
    come from the same arena, so a disagreement is a payload assembled from two
    processes, and a merge floor computed from the wrong one moves by a factor of
    the ratio."""
    with pytest.raises(MeasurementUnsound, match="disagree"):
        check_arm_split(_arm(splits=4))
