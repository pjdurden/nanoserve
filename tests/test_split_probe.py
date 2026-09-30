"""Day 71: the split server grades its own two passes before it serves.

Days 69 and 70 gave each pass of the split read its own entry point and a test file
that grades it by name. On this box those tests only ever run the tlsim programs:
the jitted `_paged_attention_split_fwd` and `_split_reduce_fwd` are four gated tests
that go green or red on the first card, and only if somebody remembers to run pytest
there. A server does not have to remember. It picks its read at construction and
gets its arena before the first step (Days 65 and 66), so there is a moment on the
boot path when the backend that will serve is known and nothing is serving yet.

This file is that moment. `chunk_partials_reference` is Day 70's plain-torch oracle
moved into the module, `probe_split_passes` runs `split_partials` and `split_reduce`
on a tiny case with one row that crosses two chunk boundaries and one that fits in
one chunk, and grades each pass against the reference on its own. `probe_split_read`
runs it on the engine's device, with the read's tile and the model's head geometry,
and `build_app` refuses to boot when it fails. The mutants are Day 70's overrun and
Day 68's keep-the-winner reduce, planted into the module where the probe will call
them, and the probe has to name the pass that broke.
"""

from __future__ import annotations

import pytest
import torch
from reference import requires_triton_gpu
from test_split_boot import GRAPHS, _app, _engine, _tiny_config
from test_split_partials import _case, _chunk_partials

from nanoserve.kernels import flash_decoding
from nanoserve.kernels.flash_decoding import (
    SplitProbe,
    SplitUnsound,
    chunk_partials_reference,
    probe_split_passes,
    reduce_partials,
    split_partials_kernel,
)
from nanoserve.kernels.paged_attention import paged_attention_batched_reference
from nanoserve.launch import BootUnsound, boot_info, check_boot_info, probe_split_read

# --- the reference, moved into the module ---------------------------------------------


@pytest.mark.parametrize(
    "lens, width, chunk, count, n_rep",
    [
        ([20, 3], 24, 8, 3, 1),
        ([17, 16, 1], 32, 16, 2, 1),
        ([40, 9], 40, 16, 3, 2),
    ],
)
def test_the_module_reference_is_day_70_s_oracle(lens, width, chunk, count, n_rep):
    case = _case(lens, width=width, n_q=2, n_rep=n_rep, seed=4)
    got = chunk_partials_reference(*case, n_rep=n_rep, chunk=chunk, count=count)
    want = _chunk_partials(*case, n_rep, chunk, count)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_the_reference_reduces_to_the_attention_over_the_whole_row():
    case = _case([20, 3], width=24, n_q=4, n_rep=2, seed=5)
    parts = chunk_partials_reference(*case, n_rep=2, chunk=8, count=3)
    want = paged_attention_batched_reference(*case, n_rep=2)
    assert torch.allclose(reduce_partials(*parts), want, atol=1e-5)


def test_the_reference_is_the_model_pass_one_slot_for_slot():
    case = _case([20, 3], width=24, n_q=2, seed=6)
    got = split_partials_kernel(*case, n_rep=1, block=4, splits=3)
    want = chunk_partials_reference(*case, n_rep=1, chunk=8, count=3)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-5, equal_nan=True)


def test_the_reference_takes_the_scale_it_is_given():
    case = _case([9], width=16, seed=7)
    half = chunk_partials_reference(*case, n_rep=1, chunk=8, count=2, scale=0.5)
    auto = chunk_partials_reference(*case, n_rep=1, chunk=8, count=2)
    assert not torch.allclose(half[0], auto[0])


# --- the probe, on the host -----------------------------------------------------------


def test_the_probe_passes_the_honest_passes_on_the_host():
    report = probe_split_passes(torch.device("cpu"), block=4, head_dim=8, n_rep=2)
    assert isinstance(report, SplitProbe)
    assert report.backend == "tlsim"
    assert report.pass_one_error < report.atol
    assert report.pass_two_error < report.atol
    assert report.read_error < report.atol


def test_the_probe_s_case_has_a_row_across_chunks_and_a_row_inside_one():
    """The probe's own controls. A row that fits in one chunk is where both mutants
    are exact, and a row that crosses chunks is where both break, so a probe case
    missing either one is a probe that passes the mutants."""
    report = probe_split_passes(torch.device("cpu"), block=4, head_dim=8)
    assert max(report.live_chunks) >= 3
    assert min(report.live_chunks) == 1
    assert report.splits == max(report.live_chunks)


def test_the_probe_walks_whole_tiles_of_the_block_it_is_given():
    for block in (1, 4, 16):
        report = probe_split_passes(torch.device("cpu"), block=block, head_dim=4)
        assert report.block == block
        assert report.chunk % block == 0


def test_the_probe_runs_the_pool_s_dtype():
    report = probe_split_passes(
        torch.device("cpu"), block=4, head_dim=8, dtype=torch.bfloat16
    )
    assert report.dtype == torch.bfloat16
    assert report.pass_one_error < report.atol


def _overrun(q, k_pool, v_pool, slot_mapping, context_lens, n_rep, scale=None, *,
             block, splits, **_):
    """Day 70's mutant as a pass one: every chunk walks to its row's end."""
    _, chunk = flash_decoding.partition_width(slot_mapping.shape[1], block, splits=splits)
    return _chunk_partials(q, k_pool, v_pool, slot_mapping, context_lens, n_rep, chunk,
                           splits, overrun=True)


def _keep_winner(partials, out_dtype=None):
    """Day 68's mutant as a pass two: keep the chunk with the largest max, drop the rest."""
    part_max, part_denom, part_acc = partials
    win = part_max.argmax(dim=2, keepdim=True)
    denom = part_denom.gather(2, win)
    acc = part_acc.gather(2, win[..., None].expand(-1, -1, -1, part_acc.shape[-1]))
    out = (acc / denom[..., None])[:, :, 0, :][:, :, None, :]
    return out if out_dtype is None else out.to(out_dtype)


def test_a_pass_one_that_overruns_its_chunk_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(flash_decoding, "split_partials", _overrun)
    with pytest.raises(SplitUnsound, match="pass one"):
        probe_split_passes(torch.device("cpu"), block=4, head_dim=8)


def test_a_pass_two_that_keeps_only_the_winner_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(flash_decoding, "split_reduce", _keep_winner)
    with pytest.raises(SplitUnsound, match="pass two"):
        probe_split_passes(torch.device("cpu"), block=4, head_dim=8)


def test_the_overrun_would_have_passed_a_probe_of_short_rows_only():
    """Why the probe's case is shaped the way it is: on its one-chunk row, the
    overrun's slots are the reference's."""
    case = _case([3], width=12, seed=0)
    wrong = _chunk_partials(*case, 1, 4, 3, overrun=True)
    right = chunk_partials_reference(*case, n_rep=1, chunk=4, count=3)
    for a, b in zip(wrong, right):
        assert torch.equal(a, b)


def test_the_probe_s_tolerance_is_a_real_bound(monkeypatch):
    """A pass one off by 1e-2 in one accumulator is caught at the default tolerance
    and passed at a loose one: the bound is the number the report states."""
    honest = flash_decoding.split_partials

    def nudged(*args, **kwargs):
        m, denom, acc = honest(*args, **kwargs)
        acc = acc.clone()
        acc[0, 0, 1, 0] += 1e-2
        return m, denom, acc

    monkeypatch.setattr(flash_decoding, "split_partials", nudged)
    with pytest.raises(SplitUnsound, match="pass one"):
        probe_split_passes(torch.device("cpu"), block=4, head_dim=8)
    report = probe_split_passes(torch.device("cpu"), block=4, head_dim=8, atol=1.0)
    assert 5e-3 < report.pass_one_error < 1.0


def test_the_probe_report_says_what_it_graded():
    report = probe_split_passes(torch.device("cpu"), block=4, head_dim=8, n_rep=2)
    d = report.as_dict()
    assert d["backend"] == "tlsim"
    assert d["splits"] == report.splits and d["block"] == 4
    assert set(d) >= {"pass_one_error", "pass_two_error", "read_error", "atol", "dtype"}
    assert "tlsim" in report.render() and "pass one" in report.render()


# --- the boot path --------------------------------------------------------------------


def test_probing_a_non_split_engine_is_a_no_op():
    assert probe_split_read(_engine(**GRAPHS)) is None


def test_the_boot_probe_uses_the_read_s_tile_and_the_model_s_heads():
    engine = _engine(split_read=True, **GRAPHS)
    report = probe_split_read(engine)
    cfg = _tiny_config()
    assert report.block == engine.cache.read.block
    assert report.head_dim == cfg.head_dim
    assert report.n_rep == cfg.num_attention_heads // cfg.num_key_value_heads
    assert report.dtype == torch.float32


def test_a_failed_probe_is_a_refused_boot_before_the_arena_exists(monkeypatch):
    monkeypatch.setattr(flash_decoding, "split_partials", _overrun)
    engine = _engine(split_read=True, **GRAPHS)
    with pytest.raises(BootUnsound, match="pass one"):
        probe_split_read(engine)
    assert engine.cache.read.workspace is None


def test_a_split_server_boots_with_its_probe_on_record():
    app = _app(split_read=True, **GRAPHS)
    assert app.state.split_probe.backend == "tlsim"
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace, app.state.split_probe)
    assert info["split_probe"]["splits"] >= 3
    check_boot_info(info)


def test_a_server_on_another_read_runs_no_probe():
    app = _app(**GRAPHS)
    assert app.state.split_probe is None


def test_a_split_server_whose_pass_two_is_wrong_does_not_boot(monkeypatch):
    monkeypatch.setattr(flash_decoding, "split_reduce", _keep_winner)
    with pytest.raises(BootUnsound, match="pass two"):
        _app(split_read=True, bucket_decode=True)


def test_a_payload_whose_probe_failed_is_refused():
    app = _app(split_read=True, **GRAPHS)
    info = boot_info(app.state.plan, app.state.capture, app.state.warmup,
                     app.state.workspace, app.state.split_probe)
    info["split_probe"]["pass_one_error"] = 1.0
    with pytest.raises(BootUnsound, match="probe"):
        check_boot_info(info)


# --- the card -------------------------------------------------------------------------


@requires_triton_gpu
def test_the_probe_grades_the_jitted_passes_on_a_card():
    report = probe_split_passes(torch.device("cuda"), block=16, head_dim=64, n_rep=4,
                                dtype=torch.float16)
    assert report.backend == "triton"
    assert report.pass_one_error < report.atol


@requires_triton_gpu
def test_the_probe_refuses_an_overrun_on_a_card(monkeypatch):
    monkeypatch.setattr(flash_decoding, "split_partials", _overrun)
    with pytest.raises(SplitUnsound, match="pass one"):
        probe_split_passes(torch.device("cuda"), block=16, head_dim=64)
