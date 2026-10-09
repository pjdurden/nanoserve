"""Day 78: a tlsim launch is a launch, and dynamo does not get to unroll it.

Day 77 gave every compiled arm a fresh dynamo cache and a count of the frames it gave
up on. The three-arm compiled toy run still failed the new gate, in the split arm:

    torch._dynamo hit config.recompile_limit (8)
       function: 'torch_dynamo_resume_in_split_kernel_at_877'
       last reason: 14/7: i == 0  # qi = q_rows[i, h]

On a laptop the split read is tlsim: `launch` is a Python loop that calls the kernel
body once per program id. Under `torch.compile` dynamo traced *into* that loop. The
body graph-breaks on its row length (`int(load(...))`, a tensor read as an int), so
the rest of the body became a resume frame, and that frame read `program_id`, a
Python int dynamo guards on. Every program was a new guard, and the eighth was the
limit. A card never sees any of it: a Triton launch is one call dynamo hands to the
kernel, not a grid it walks.

Two ways out, and the day takes the honest one. The gate could learn that a tlsim
frame is not the forward, which would be a gate that knows where its failures come
from and excuses them. Or the stand-in can behave like what it stands in for:
`tlsim.launch` is wrapped in `torch.compiler.disable`, so a compiled forward breaks
at the launch, runs the grid in the interpreter the way a card runs it on the
device, and resumes after it. Every tlsim kernel goes through `launch`, so the
split's two passes and the streamed read are all covered by the one wrapper.

Three layers of test: `launch` under a compile with a body shaped like the split's,
the split read itself under a compile, and the three-arm compiled run as a process.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_split_partials import _case

from nanoserve.compiled import dynamo_frames_abandoned
from nanoserve.kernels.flash_decoding import paged_attention_split_kernel
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.kernels.tlsim import arange, launch, load, store
from nanoserve.toyckpt import write_toy_checkpoint
from nanoserve.toysmoke import SMOKE_LOAD, unexpected_failures

REPO = Path(__file__).resolve().parent.parent
HELD = "every compiled arm stayed compiled"


# --- 1. the launch -----------------------------------------------------------------------


#: The split's pass-one grid, small: rows, query heads, chunks.
GRID = (4, 3, 3)


def _scaled_by_row(src: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """A grid with the split kernel's shape. Each program reads its row's length off a
    buffer as an int, which is the graph break, then addresses its own cell with all
    three program ids after it, which is the guard. `out[i, h, s] = src[i, h, s] *
    lens[i]`, written through a flat buffer the way the kernel writes its workspace."""
    rows, heads, chunks = GRID
    out = torch.zeros(src.numel())

    def kernel(prog, src_buf, len_buf, out_buf) -> None:
        i, h, s = prog.program_id(0), prog.program_id(1), prog.program_id(2)
        n = int(load(len_buf, arange(i, i + 1))[0])
        cell = (i * heads + h) * chunks + s
        store(out_buf, arange(cell, cell + 1), src_buf[i, h, s] * n)

    launch(GRID, kernel, src, lens, out)
    return out.reshape(GRID)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(1.0, 1.0 + GRID[0] * GRID[1] * GRID[2]).reshape(GRID)
    return src, torch.tensor([3, 1, 4, 1])


def _compiled_launch() -> torch.Tensor:
    """`_scaled_by_row` under the cheapest compile, from an empty cache, at the
    default limit. Thirty-six programs and three ids each, every one of which dynamo
    specialises at 0 and 1: traced into, that is more than eight frames."""
    torch._dynamo.reset()
    return torch.compile(_scaled_by_row, backend="eager")(*_inputs())


def test_a_compiled_launch_gives_up_no_frame():
    """Day 77's failure in miniature: a launch dynamo steps over has no frames to lose."""
    before = dynamo_frames_abandoned()
    _compiled_launch()
    assert dynamo_frames_abandoned() == before


def test_a_compiled_launch_writes_what_an_eager_one_does():
    assert torch.equal(_compiled_launch(), _scaled_by_row(*_inputs()))


def test_the_launch_keeps_its_name_and_its_doc():
    """The wrapper is invisible to a reader: `help(launch)` is still Day 21's."""
    assert launch.__name__ == "launch"
    assert "once for every program id" in launch.__doc__


# --- 2. the split read under a compile ---------------------------------------------------


def test_the_split_read_compiled_gives_up_no_frame_and_answers_the_same():
    """The read the split arm serves, two passes and four rows of two heads in three
    chunks: twenty-four pass-one programs, well past a limit of two."""
    case = _case([5, 33, 70, 12], width=96)
    want = paged_attention_batched_reference(*case, 1)
    torch._dynamo.reset()
    before = dynamo_frames_abandoned()
    read = torch.compile(paged_attention_split_kernel, backend="eager")
    with torch._dynamo.config.patch(recompile_limit=2):
        got = read(*case, 1, block=16, splits=3)
    assert dynamo_frames_abandoned() == before
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


# --- 3. the three-arm compiled run, as a process -----------------------------------------


@pytest.fixture(scope="module")
def three_arms(tmp_path_factory):
    """Day 77's compiled smoke with `--split-read`: the run whose split arm lost a
    frame. A few minutes on this box, most of it inductor and the split's tlsim."""
    checkpoint = write_toy_checkpoint(tmp_path_factory.mktemp("toy"))
    out = tmp_path_factory.mktemp("three") / "three.csv"
    cmd = [
        sys.executable, str(REPO / "graphbench.py"),
        "--weights", str(checkpoint),
        "--device", "cpu", "--dtype", "float32", "--allow-cpu",
        "--arrivals", "burst",
        "--max-model-len", "1024", "--block-size", "16",
        "--max-batch-size", "4", "--num-blocks", "160",
        *SMOKE_LOAD,
        "--compile", "dynamic", "--warm-rows", "4", "--split-read",
        "--csv", str(out),
    ]
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200)
    return done, out


def test_the_three_arm_compiled_run_exits_cleanly(three_arms):
    done, _ = three_arms
    assert done.returncode == 0, done.stderr[-3000:]


def test_no_tlsim_frame_reached_the_limit(three_arms):
    """The line Day 77 left, read where it was found."""
    err = three_arms[0].stderr
    assert "recompile_limit" not in err, err[-3000:]
    assert "split_kernel" not in err, err[-3000:]


def test_the_split_arm_held_its_compile(three_arms):
    out = three_arms[0].stdout
    assert "frames abandoned at the limit: split 0, graphs 0, eager 0" in out, out[-3000:]
    assert f"ok    {HELD}" in out


def test_the_three_arm_csv_holds_the_claim(three_arms):
    with three_arms[1].open() as fh:
        (row,) = list(csv.DictReader(fh))
    assert HELD not in row["claims_failed"]


def test_the_three_arm_run_printed_no_failure_it_was_not_allowed(three_arms):
    """Day 79: at `SMOKE_LOAD` the comparability gate holds, so the only FAIL left
    is the tail on a CPU, and only for its own reason."""
    assert unexpected_failures(three_arms[0].stdout) == [], three_arms[0].stdout[-3000:]
