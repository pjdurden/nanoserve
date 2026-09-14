"""Day 57 tests: the Week 13 acceptance test, two servers, one recorded.

Week 13 built the replay one property at a time. Day 49 compiled the decode step,
Day 52 closed its shape set into buckets, Day 53 moved its inputs into buffers that
do not move, Day 54 recorded it, Day 55 warmed the list off synthetic batches and
Day 56 wired all of it to a command-line flag. Every one of those days asserted its
own property against an engine object it was holding.

None of them asked a *running server* anything. `/health` reports what the boot
decided and says nothing about what the loop has done since, so a process whose
trimmed list is quietly recording at the top of the width axis, or worse, falling
through to eager, is indistinguishable from one replaying everything. That is the
failure this week can have with no symptom: the tokens are right, the counters
nobody published are wrong, and the only evidence is a latency histogram.

It found one, which is the reason this file reads the way it does.

**A warm capture replayed a window over the wrong rows, and returned another
request's continuation.** A recorded read is `slots[:rows, :width]`, a window from
row zero, while the persistent input buffers are written in batch order. The two
line up only while the scheduler's rows are `(0, 1, ... n-1)`. Two requests of
different lengths break that on the step the shorter one finishes: the survivor
stays in cache row 1, the replay reads cache row 0's history under its token, and
nothing raises anywhere. Day 54's `check_replay_rows` is not this gate, and on a
warm list it cannot be: a warm graph records over no rows at all, so the field it
guards is empty exactly where the check was needed. `rows_are_a_prefix` is the gate,
and a step that fails it now runs the forward instead of a replay.

Four claims:

  1. **A replay answers what an eager forward would, across two processes.** Two
     servers from the same weights, one launched with the graphs and one without,
     every client's text byte-identical, **under a crowd**. The first version of this
     file compared answers from a pass where each client ran alone and it passed
     against the bug above, because a solo request is one row in cache row zero,
     which is the one batch shape a misaligned replay gets right.
  2. **A running process can say what its capture did.** `/health` carries the live
     counters next to the boot decision, and the two gates Day 54 wrote work on a
     reading taken over a socket exactly as they work on the object.
  3. **A claim about a run is a difference of two readings.** A counter is cumulative
     and a health check is an instant, so "this server recorded nothing while it was
     serving" is a subtraction, not a number. The lazy server is the control: same
     engine, same list, and the recordings land in front of clients.
  4. **The comparison is only worth quoting if both arms did the same work.** Two
     arms that delivered different token counts have two different denominators, and
     the ITL between them is a comparison of workloads wearing the name of a
     speedup. That is a `MeasurementUnsound`, which is a different morning from a
     capture that made the tail worse.

The speed claim is asserted on scripted reports rather than on a live pair, and
that is not a shortcut. On a box with no CUDA the recorder is `eager_recorder`, a
stand-in with a capture's semantics and none of its speed, so a graphed arm here is
a forward plus a copy and is honestly *slower*. The arithmetic of the comparison is
what these tests can check; `graphbench.py` is where the two arms meet a card.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import torch

from nanoserve.acceptance import AcceptanceFailure, ClientPlan, live_server
from nanoserve.captured import (
    CaptureStats,
    CaptureUnsound,
    CapturedDecode,
    check_all_shapes_captured,
    check_no_scattered_rows,
    check_replays_dominate,
    eager_recorder,
    rows_are_a_prefix,
)
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.graphbench import (
    ArmDelta,
    ArmReport,
    capture_from_health,
    check_arm_replayed,
    check_arm_replayed_every_step,
    check_arm_was_crowded,
    check_arms_comparable,
    check_nothing_recorded_while_serving,
    check_same_answers,
    check_tail_not_worse,
    paired_plans,
    run_arm,
)
from nanoserve.launch import build_app, kv_bytes_per_block
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import Request
from nanoserve.server import health_payload
from nanoserve.servebench import LoadReport, MeasurementUnsound, RequestRecord, burst_arrivals

# --- a capture's counters as a value ----------------------------------------------


def _fake_decode(calls=10, captures=3, replays=7, eager_calls=0, graphs=3, mode="capture"):
    """A `CapturedDecode` whose counters were set rather than earned.

    Every property under test here reads counters and nothing else, so a run that
    produced them would be a slower way of writing the same six integers down.
    """
    captured = CapturedDecode(lambda *a, **k: None, mode=mode, recorder=eager_recorder)
    captured.calls = calls
    captured.captures = captures
    captured.replays = replays
    captured.eager_calls = eager_calls
    captured.graphs = dict.fromkeys(range(graphs), object())
    return captured


def test_the_stats_are_the_object_s_counters_without_the_object():
    stats = CaptureStats.of(_fake_decode())
    assert (stats.calls, stats.captures, stats.replays, stats.eager_calls) == (10, 3, 7, 0)
    assert stats.graphs == 3
    assert stats.mode == "capture"


def test_the_reuse_is_the_same_number_the_object_reports():
    captured = _fake_decode()
    assert CaptureStats.of(captured).reuse == captured.reuse


def test_reuse_with_no_calls_is_zero_rather_than_an_error():
    """A server nobody has asked anything of has not failed to reuse its graphs."""
    assert CaptureStats(mode="capture").reuse == 0.0


def test_a_reading_survives_json():
    """The whole reason this is a value: it goes over a socket and comes back."""
    stats = CaptureStats.of(_fake_decode())
    assert CaptureStats.from_dict(stats.as_dict()) == stats


def test_the_payload_names_the_mode_so_a_reader_can_tell_off_from_idle():
    """Zero replays means two different things and they need different mornings:
    a process with the capture switched off, and one that is switched on and has
    not been asked for a token yet."""
    assert CaptureStats().as_dict()["mode"] == "off"
    assert CaptureStats(mode="capture").as_dict()["mode"] == "capture"


def test_a_window_is_the_difference_between_two_readings():
    before = CaptureStats(mode="capture", graphs=3, calls=10, captures=3, replays=7)
    after = CaptureStats(mode="capture", graphs=3, calls=40, captures=3, replays=37)
    window = after.since(before)
    assert window.calls == 30
    assert window.captures == 0
    assert window.replays == 30


def test_a_window_with_no_recordings_in_it_is_full_reuse():
    """The number a warm server is supposed to have while it is serving. Every
    decode in the window replayed a graph the boot paid for."""
    before = CaptureStats(mode="capture", graphs=3, calls=10, captures=3, replays=7)
    after = CaptureStats(mode="capture", graphs=3, calls=110, captures=3, replays=107)
    assert after.since(before).reuse == 1.0


def test_a_window_counts_the_graphs_that_appeared_during_it():
    before = CaptureStats(mode="capture", graphs=1, calls=4, captures=1, replays=3)
    after = CaptureStats(mode="capture", graphs=3, calls=20, captures=3, replays=17)
    assert after.since(before).graphs == 2


def test_a_reading_earlier_than_the_one_before_it_is_refused():
    """Counters only go up, so a negative window is not a small number, it is a
    reading taken from a different process: a restart, or two servers whose
    readings got swapped."""
    before = CaptureStats(mode="capture", calls=40)
    after = CaptureStats(mode="capture", calls=10)
    with pytest.raises(ValueError, match="cannot have run"):
        after.since(before)


def test_a_window_across_a_mode_change_is_refused():
    with pytest.raises(ValueError, match="mode"):
        CaptureStats(mode="capture", calls=10).since(CaptureStats(mode="off"))


def test_a_reading_renders_as_a_line_a_human_reads_at_3am():
    line = CaptureStats.of(_fake_decode()).render()
    assert "3 graphs" in line
    assert "70%" in line


# --- the Day-54 gates, asked of a reading rather than of an object ------------------


def test_the_eager_gate_works_on_a_reading():
    stats = CaptureStats(mode="capture", graphs=3, calls=10, replays=8, eager_calls=2)
    with pytest.raises(CaptureUnsound, match="ran eager"):
        check_all_shapes_captured(stats)


def test_the_eager_gate_passes_a_reading_that_replayed_everything():
    check_all_shapes_captured(CaptureStats(mode="capture", graphs=3, calls=10, replays=10))


def test_the_reuse_gate_works_on_a_reading():
    stats = CaptureStats(mode="capture", graphs=9, calls=10, captures=9, replays=10)
    with pytest.raises(CaptureUnsound, match="reuse"):
        check_replays_dominate(stats)


def test_the_gates_read_an_interface_and_not_a_class():
    """The point of the value. One gate, two callers: the process holding the
    object, and a harness holding a dict that came off a socket."""
    captured = _fake_decode(calls=10, captures=9, replays=10, graphs=9)
    for subject in (captured, CaptureStats.of(captured)):
        with pytest.raises(CaptureUnsound):
            check_replays_dominate(subject)


# --- what /health says about a capture ---------------------------------------------


def test_the_health_merge_keeps_both_halves_of_one_key():
    """Two dicts want `cuda_graphs` and `**` would silently keep the second. The
    boot decision and the live counters are not alternatives: one says what this
    process promised and the other says what it has done."""
    info = {"num_blocks": 32, "cuda_graphs": {"shapes": 3, "graphs_held": 3}}
    stats = {"running": 0, "cuda_graphs": {"mode": "capture", "calls": 9}}
    payload = health_payload("nanoserve", info, stats)
    assert payload["cuda_graphs"]["graphs_held"] == 3
    assert payload["cuda_graphs"]["runtime"]["calls"] == 9


def test_a_server_with_no_capture_has_no_section_at_all():
    payload = health_payload("nanoserve", {"num_blocks": 32}, {"running": 0})
    assert "cuda_graphs" not in payload
    assert payload["status"] == "ok"


def test_the_live_half_alone_still_reaches_the_payload():
    """An engine built with the capture by hand, served by an app nobody launched:
    there is no boot section to nest under, and the counters still have to arrive."""
    payload = health_payload("nanoserve", {}, {"cuda_graphs": {"mode": "capture", "calls": 9}})
    assert payload["cuda_graphs"]["calls"] == 9


def test_a_reading_is_lifted_out_of_the_payload_by_one_function():
    payload = health_payload(
        "nanoserve",
        {"cuda_graphs": {"shapes": 3}},
        {"cuda_graphs": CaptureStats(mode="capture", calls=9, replays=9, graphs=3).as_dict()},
    )
    assert capture_from_health(payload) == CaptureStats(
        mode="capture", calls=9, replays=9, graphs=3
    )


def test_a_server_without_graphs_reads_back_as_a_capture_that_is_off():
    assert capture_from_health({"num_blocks": 32}).mode == "off"


def test_a_payload_that_names_a_capture_and_will_not_say_what_it_did_is_refused():
    """The Day-56 complaint, as a gate. A boot section with no runtime next to it
    is a server reporting a decision it made once and nothing about whether the
    decision is still true."""
    with pytest.raises(AcceptanceFailure, match="what it has done"):
        capture_from_health({"cuda_graphs": {"shapes": 3, "graphs_held": 3}})


# --- one arm of the comparison -------------------------------------------------------


def _record(client_id: str, *, gaps, start=0.0, tokens=None) -> RequestRecord:
    """One request whose frames landed at exactly these intervals."""
    times = [start]
    for gap in gaps:
        times.append(times[-1] + gap)
    return RequestRecord(
        client_id=client_id,
        scheduled_at=0.0,
        sent_at=0.0,
        frame_times=tuple(times),
        done_at=times[-1],
        prompt_tokens=3,
        output_tokens=len(times) if tokens is None else tokens,
        status=200,
    )


def _load(*, gaps, n=4, tokens=None) -> LoadReport:
    """A run of `n` identical requests with this cadence, and nothing else real."""
    records = tuple(_record(f"c{i}", gaps=gaps, tokens=tokens) for i in range(n))
    span = max(r.done_at for r in records)
    return LoadReport(records=records, started_at=0.0, finished_at=span, target_rate=None)


def _arm(
    name="graphs",
    *,
    texts=None,
    gaps=(0.010, 0.010, 0.010),
    n=4,
    tokens=None,
    before=None,
    after=None,
    boot=None,
    peak_running=4,
) -> ArmReport:
    texts = {f"c{i}": "hello" for i in range(n)} if texts is None else texts
    mode = "off" if name == "eager" else "capture"
    before = before if before is not None else CaptureStats(mode=mode, graphs=3, calls=0)
    after = after if after is not None else CaptureStats(
        mode=mode, graphs=3, calls=100, captures=0, replays=100
    )
    return ArmReport(
        name=name,
        texts=texts,
        load=_load(gaps=gaps, n=n, tokens=tokens),
        before=before,
        after=after,
        boot=boot if boot is not None else {"shapes": 3, "graphs_held": 3},
        peak_running=peak_running,
    )


def test_an_arm_reports_the_window_it_served_in():
    arm = _arm(
        before=CaptureStats(mode="capture", graphs=3, calls=12, captures=3, replays=9),
        after=CaptureStats(mode="capture", graphs=3, calls=112, captures=3, replays=109),
    )
    assert arm.served.calls == 100
    assert arm.served.captures == 0


def test_an_arm_knows_whether_it_was_graphed():
    assert _arm().graphed
    assert not _arm("eager").graphed


def test_an_arm_renders_its_name_its_cadence_and_its_capture():
    line = _arm().render()
    assert "graphs" in line
    assert "10.0 ms" in line


# --- the two arms, compared ----------------------------------------------------------


def test_the_delta_is_a_ratio_in_the_direction_a_reader_expects():
    """Above one means the recorded arm was faster, which is the only reading of
    "speedup" nobody has to look up."""
    delta = ArmDelta(graphs=_arm(gaps=(0.005,) * 4), eager=_arm("eager", gaps=(0.010,) * 4))
    assert delta.itl_p50_speedup == pytest.approx(2.0)
    assert delta.itl_p50_saved_ms == pytest.approx(5.0)


def test_a_capture_that_made_the_step_slower_reads_as_a_speedup_below_one():
    delta = ArmDelta(graphs=_arm(gaps=(0.020,) * 4), eager=_arm("eager", gaps=(0.010,) * 4))
    assert delta.itl_p50_speedup == pytest.approx(0.5)
    assert delta.itl_p50_saved_ms == pytest.approx(-10.0)


def test_the_tail_is_reported_apart_from_the_middle():
    """The whole reason this day reports two percentiles. A capture's claim is
    about per-step launch overhead, which is the middle; a warm-up's claim is about
    the recordings that would otherwise land in front of a client, which is only
    ever the tail."""
    slow_tail = _arm(gaps=(0.010, 0.010, 0.010, 0.200))
    delta = ArmDelta(graphs=slow_tail, eager=_arm("eager", gaps=(0.010,) * 4))
    assert delta.itl_p50_speedup == pytest.approx(1.0)
    assert delta.itl_p99_speedup < 0.1


def test_a_delta_with_no_samples_is_zero_rather_than_a_division():
    delta = ArmDelta(graphs=_arm(gaps=()), eager=_arm("eager", gaps=()))
    assert delta.itl_p50_speedup == 0.0
    assert delta.itl_p99_speedup == 0.0


def test_the_delta_is_a_row_a_csv_can_hold():
    row = ArmDelta(graphs=_arm(gaps=(0.005,) * 4), eager=_arm("eager", gaps=(0.010,) * 4)).row()
    assert row["graphs_itl_p50_ms"] == pytest.approx(5.0)
    assert row["eager_itl_p50_ms"] == pytest.approx(10.0)
    assert row["itl_p50_speedup"] == pytest.approx(2.0)


# --- claim 1: the same answers -------------------------------------------------------


def test_two_arms_that_agree_pass():
    check_same_answers(_arm(), _arm("eager"))


def test_one_different_token_fails_and_names_the_client():
    graphs = _arm(texts={"c0": "hello", "c1": "wxrld"})
    eager = _arm("eager", texts={"c0": "hello", "c1": "world"})
    with pytest.raises(AcceptanceFailure, match="c1"):
        check_same_answers(graphs, eager)


def test_a_client_missing_from_one_arm_is_a_failure_and_not_a_skip():
    """The way a comparison like this rots into a no-op: the loop body never runs
    and the function returns successfully having asserted nothing."""
    graphs = _arm(texts={"c0": "hello"})
    eager = _arm("eager", texts={"c0": "hello", "c1": "world"})
    with pytest.raises(AcceptanceFailure, match="c1"):
        check_same_answers(graphs, eager)


def test_two_empty_arms_are_a_failure_rather_than_a_pass():
    with pytest.raises(AcceptanceFailure, match="no answers"):
        check_same_answers(_arm(texts={}), _arm("eager", texts={}))


# --- claim 2 and 3: what the capture did while it was serving ------------------------


def test_an_arm_that_replayed_everything_passes():
    check_arm_replayed(_arm())


def test_an_arm_that_fell_through_to_eager_fails():
    """The failure with no symptom. The tokens are right, the steps are slow, and
    nothing in the process raises: a decode whose cache view carried no plan just
    runs the forward."""
    arm = _arm(after=CaptureStats(mode="capture", graphs=3, calls=100, replays=60, eager_calls=40))
    with pytest.raises(AcceptanceFailure, match="ran eager"):
        check_arm_replayed(arm)


def test_an_arm_that_recorded_as_often_as_it_replayed_fails():
    arm = _arm(after=CaptureStats(mode="capture", graphs=90, calls=100, captures=90, replays=100))
    with pytest.raises(AcceptanceFailure, match="reuse"):
        check_arm_replayed(arm)


def test_an_arm_with_the_capture_switched_off_cannot_pass_this_claim():
    """Not a vacuous pass. An eager arm has no graphs to replay, and a check that
    said nothing about it would make the control arm look like a compliant one."""
    with pytest.raises(AcceptanceFailure, match="no capture"):
        check_arm_replayed(_arm("eager", after=CaptureStats(mode="off", calls=100)))


def test_an_arm_that_served_nothing_cannot_support_the_claim_either():
    arm = _arm(after=CaptureStats(mode="capture", graphs=3, calls=0))
    with pytest.raises(AcceptanceFailure, match="no decode"):
        check_arm_replayed(arm)


def test_an_arm_that_replayed_every_step_passes_the_coverage_claim():
    check_arm_replayed_every_step(_arm())


def test_an_arm_whose_rows_scattered_reports_the_share_it_lost():
    """The finding, as the number it is. Those steps ran the right forward and got
    the right tokens; what they did not get is the launch overhead this week was
    spent removing, and a capture nobody can use is memory held for nothing."""
    arm = _arm(
        after=CaptureStats(
            mode="capture", graphs=3, calls=100, replays=60, scattered_calls=40
        )
    )
    with pytest.raises(AcceptanceFailure, match="40%"):
        check_arm_replayed_every_step(arm)


def test_the_coverage_claim_and_the_replay_claim_are_different_claims():
    """A scattered run passes `check_arm_replayed`: it recorded nothing new, it
    reused what it had, and every step it did replay was a step it was entitled to.
    Collapsing the two would make "the capture is on" and "the capture covers the
    loop" one green tick, and they are one day apart."""
    arm = _arm(
        after=CaptureStats(
            mode="capture", graphs=3, calls=100, replays=60, scattered_calls=40
        )
    )
    check_arm_replayed(arm)
    with pytest.raises(AcceptanceFailure):
        check_arm_replayed_every_step(arm)


def test_an_arm_that_served_its_clients_one_at_a_time_is_not_admissible():
    with pytest.raises(MeasurementUnsound, match="row zero"):
        check_arm_was_crowded(_arm(peak_running=1))


def test_an_arm_that_really_batched_is_admissible():
    check_arm_was_crowded(_arm(peak_running=4))


def test_a_warm_arm_recorded_nothing_in_front_of_a_client():
    check_nothing_recorded_while_serving(_arm())


def test_a_lazy_arm_recorded_in_front_of_clients_and_says_how_many():
    """The control, and the thing a production process would page on. Day 55's
    warm-up is exactly the difference between these two arms."""
    arm = _arm(
        before=CaptureStats(mode="capture", graphs=0, calls=0),
        after=CaptureStats(mode="capture", graphs=3, calls=100, captures=3, replays=100),
    )
    with pytest.raises(AcceptanceFailure, match="3 graph"):
        check_nothing_recorded_while_serving(arm)


# --- claim 4: the comparison is between two of the same thing ------------------------


def test_two_arms_that_did_the_same_work_are_comparable():
    check_arms_comparable(_arm(), _arm("eager"), min_samples=4)


def test_arms_that_delivered_different_token_counts_are_not_comparable():
    """Not a regression and not a bug: a ratio between two different denominators.
    A graphed arm that answered half the tokens has a beautiful ITL."""
    with pytest.raises(MeasurementUnsound, match="token counts"):
        check_arms_comparable(_arm(tokens=8), _arm("eager", tokens=16), min_samples=4)


def test_arms_that_served_different_clients_are_not_comparable():
    with pytest.raises(MeasurementUnsound, match="clients"):
        check_arms_comparable(_arm(n=4), _arm("eager", n=3), min_samples=3)


def test_a_comparison_with_too_few_frame_gaps_is_refused():
    """A p99 of four samples is the worst of four samples wearing a name it did not
    earn, and this is the day whose whole claim is about a tail."""
    with pytest.raises(MeasurementUnsound, match="frame gap"):
        check_arms_comparable(_arm(), _arm("eager"), min_samples=100)


def test_a_failed_request_in_either_arm_invalidates_the_comparison():
    broken = _load(gaps=(0.01, 0.01, 0.01))
    records = (*broken.records[:-1], RequestRecord(client_id="c3", scheduled_at=0.0, sent_at=0.0))
    arm = ArmReport(
        name="graphs",
        texts={f"c{i}": "hello" for i in range(4)},
        load=LoadReport(records=records, started_at=0.0, finished_at=1.0),
        before=CaptureStats(mode="capture", graphs=3),
        after=CaptureStats(mode="capture", graphs=3, calls=100, replays=100),
        boot={"shapes": 3},
    )
    with pytest.raises(MeasurementUnsound, match="failed"):
        check_arms_comparable(arm, _arm("eager"), min_samples=3)


def test_a_capture_that_did_not_move_the_tail_backwards_passes():
    delta = ArmDelta(graphs=_arm(gaps=(0.005,) * 6), eager=_arm("eager", gaps=(0.010,) * 6))
    check_tail_not_worse(delta)


def test_a_capture_that_made_the_tail_worse_is_a_regression():
    delta = ArmDelta(graphs=_arm(gaps=(0.030,) * 6), eager=_arm("eager", gaps=(0.010,) * 6))
    with pytest.raises(AcceptanceFailure, match="p99"):
        check_tail_not_worse(delta)


def test_the_tolerance_is_what_a_noisy_box_is_allowed_to_be_slower_by():
    """A p99 is the noisiest number in the report and a gate on it with no slack is
    a test that fails on a box that was busy, which is how a real regression ends up
    being explained away."""
    delta = ArmDelta(graphs=_arm(gaps=(0.0105,) * 6), eager=_arm("eager", gaps=(0.010,) * 6))
    check_tail_not_worse(delta, tolerance=0.10)
    with pytest.raises(AcceptanceFailure):
        check_tail_not_worse(delta, tolerance=0.01)


# --- the plans both arms run ---------------------------------------------------------


def test_every_plan_streams_because_a_unary_answer_has_no_cadence():
    assert all(p.stream for p in paired_plans(6, prompts=("abc", "abcd")))


def test_the_plans_mix_greedy_and_seeded_requests():
    """A capture is recorded around a forward and the sampler runs outside it, so
    the interesting claim is that a seed means the same thing on both arms."""
    plans = paired_plans(6, prompts=("abc",))
    assert any(p.is_sampled for p in plans)
    assert any(not p.is_sampled for p in plans)


def test_nobody_hangs_up_because_a_departed_client_has_no_answer_to_compare():
    assert not any(p.hangs_up for p in paired_plans(6, prompts=("abc",)))


def test_the_plans_are_the_same_list_every_time_they_are_built():
    assert paired_plans(6, prompts=("abc", "abcd")) == paired_plans(6, prompts=("abc", "abcd"))


# --- what the crowd found: a replay is a window from row zero --------------------------

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."


class ByteTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(ch) for ch in text]

    def decode(self, token_ids) -> str:
        return "".join(ALPHABET[i] for i in token_ids)


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )


def _tiny_weights(config: ModelConfig, seed: int = 0) -> Weights:
    torch.manual_seed(seed)
    tensors = {n: torch.randn(*s) for n, s in expected_shapes(config).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return Weights(tensors, config)


def _engine(**kw) -> Engine:
    """A launched-by-hand engine over the tiny model, with the week's flags optional."""
    cfg = _tiny_config()
    model = LlamaModel(cfg, _tiny_weights(cfg))
    defaults = dict(num_blocks=64, block_size=4, max_batch_size=2, max_model_len=16)
    defaults.update(kw)
    return Engine.build(model, **defaults)


def _graphed_engine(warm: bool = True, **kw) -> Engine:
    engine = _engine(
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
        **kw,
    )
    if warm:
        engine.warm_decode()
    return engine


def _uneven_pair(engine: Engine) -> tuple[list[int], list[int]]:
    """Two requests of different lengths, run together to the end.

    The smallest batch that breaks a replay, and it is the shape of every real
    server: `r0` stops after two tokens, its row comes back, and `r1` keeps going
    *in cache row 1* because nothing moves a running request's row. From that step on
    the scheduler's rows are `(1,)`, and a recorded read is the window `slots[:1]`,
    which is cache row 0.
    """
    short = Request("r0", [1, 2, 3], max_new_tokens=2)
    long = Request("r1", [4, 5, 6], max_new_tokens=8)
    engine.add_request(short)
    engine.add_request(long)
    while engine.has_unfinished():
        engine.step()
    return short.output_token_ids, long.output_token_ids


def test_a_replay_does_not_hand_a_request_its_neighbour_s_history():
    """The bug this day's acceptance test was written to find, as the test that
    would have found it. Before `rows_are_a_prefix`, `r1`'s tokens diverged from the
    eager engine's on exactly the step `r0` finished, and every gate in the repo
    passed: the addresses were stable, the shape was in the set, the pool was shared
    and the graph was warm."""
    assert _uneven_pair(_engine()) == _uneven_pair(_graphed_engine())


def test_the_step_that_cannot_replay_runs_the_forward_and_is_counted():
    engine = _graphed_engine()
    _uneven_pair(engine)
    graphs = engine.decode_graphs
    assert graphs.scattered_calls > 0
    assert graphs.replays > 0
    assert graphs.eager_calls == 0


def test_a_scattered_step_is_not_a_recording():
    """It cannot be. A graph recorded over rows `(1,)` would be refused at record
    time by `check_mapping_is_window`, because a non-prefix rectangle is an
    index_select and not a window: there is nothing at a fixed address to record."""
    engine = _graphed_engine()
    before = engine.decode_graphs.count
    _uneven_pair(engine)
    assert engine.decode_graphs.count == before


def test_the_gate_reads_the_rows_and_nothing_else():
    """Cheap enough to run on every step, which is the whole reason it can be the
    gate: a tuple comparison on the host, with no tensor read and therefore no
    synchronisation."""

    class _Plan:
        def __init__(self, rows):
            self.rows = rows

    assert rows_are_a_prefix(_Plan((0, 1, 2)))
    assert rows_are_a_prefix(_Plan(()))
    assert not rows_are_a_prefix(_Plan((1, 2)))
    assert not rows_are_a_prefix(_Plan((0, 2)))


def test_a_prefix_of_a_different_length_still_replays_its_bucket_s_graph():
    """The other half of the correction. Row equality was never the property: a
    three-row step and a four-row step of the same bucket read the same window and
    write the same buffers, so one graph serves both. Equality would have refused
    this, and refusing it is what would push a server back onto the eager path for
    every batch that is not exactly the size it was recorded at."""
    engine = _graphed_engine(max_batch_size=4)
    graphs = engine.decode_graphs
    warmed = graphs.captures
    engine.add_request(Request("r0", [1, 2, 3], max_new_tokens=6))
    engine.add_request(Request("r1", [4, 5, 6], max_new_tokens=6))
    engine.step()
    engine.add_request(Request("r2", [7, 8, 9], max_new_tokens=4))
    while engine.has_unfinished():
        engine.step()
    assert graphs.replays > 0
    assert graphs.captures == warmed


def test_the_scattered_gate_prices_the_loss_as_a_share():
    stats = CaptureStats(mode="capture", graphs=3, calls=200, replays=150, scattered_calls=50)
    with pytest.raises(CaptureUnsound, match="25%"):
        check_no_scattered_rows(stats)


def test_the_replay_share_is_not_the_reuse():
    """Two numbers that agree on every run until this week, and the day needs both:
    a server that records nothing and replays half its steps is 100% reuse and 50%
    covered, and quoting the first one makes the second invisible."""
    stats = CaptureStats(mode="capture", graphs=3, calls=100, replays=50, scattered_calls=50)
    assert stats.reuse == 1.0
    assert stats.replay_share == 0.5


# --- the acceptance run, over two real servers ---------------------------------------


def _app(**kw):
    cfg = _tiny_config()
    defaults = dict(
        weights_dir="unused",
        device="cpu",
        dtype="float32",
        block_size=4,
        max_batch_size=4,
        max_model_len=32,
        kv_cache_bytes=kv_bytes_per_block(cfg, 4, torch.float32) * 64,
        load=lambda _dir, **_kw: _tiny_weights(cfg),
        read_config=lambda _dir: cfg,
        tokenizer=ByteTokenizer(),
        bucket_decode=True,
        persist_inputs=True,
        capture_decode=True,
        capture_recorder=eager_recorder,
    )
    defaults.update(kw)
    return build_app(**defaults)


def _eager_app(**kw):
    return _app(bucket_decode=False, persist_inputs=False, capture_decode=False, **kw)


#: Eight clients over four slots, generating long enough that the crowd is still a
#: crowd by the time `/health` is polled. A tiny model answers a four-token request
#: in single-digit milliseconds, and a run that finishes between two polls reports
#: `peak_running` of zero: not a serialised crowd, an unobserved one.
PLANS = paired_plans(8, prompts=("abc", "abcde"), max_tokens=(10, 14))


def _both_arms(graphed_app, eager_app, plans=PLANS):
    """Run the same plans against two live servers and report each arm."""

    async def scenario():
        arms = []
        for name, app in (("graphs", graphed_app), ("eager", eager_app)):
            with live_server(app) as server:
                arms.append(
                    await run_arm(
                        server.base_url,
                        plans,
                        burst_arrivals(len(plans)),
                        name=name,
                    )
                )
        return arms

    return asyncio.run(asyncio.wait_for(scenario(), 120.0))


def test_the_week_s_acceptance_run_over_two_sockets():
    """Claims 1 to 4 at once, which is what an acceptance test is for.

    One run, because the run is the expensive part: two uvicorns, two crowds and two
    bursts against a tiny model. Splitting it into four tests would quadruple that to
    assert four things about the same two arms.

    The coverage claim is asserted as the failure it currently is, with the share in
    the message. That is not a test written around a bug: it is this engine's number,
    and a green tick here would say the capture covers a loop it does not cover.
    """
    graphs, eager = _both_arms(_app(), _eager_app())

    check_arm_was_crowded(graphs)
    check_arm_was_crowded(eager)
    check_same_answers(graphs, eager)
    check_arm_replayed(graphs)
    check_nothing_recorded_while_serving(graphs)
    check_arms_comparable(graphs, eager, min_samples=4)

    assert graphs.served.calls > 0
    assert graphs.boot["graphs_held"] == graphs.boot["shapes"]
    assert not eager.graphed

    with pytest.raises(AcceptanceFailure, match="not a prefix"):
        check_arm_replayed_every_step(graphs)
    assert 0.0 < graphs.served.replay_share < 1.0


def test_a_lazily_recorded_server_is_caught_by_the_window_and_not_by_its_tokens():
    """The control, over a socket. Same engine, same list, `--no-warm`: the answer is
    identical, every check about correctness passes, and the only evidence that
    somebody's first token paid for a recording is a difference of two readings.

    One client, and the reason is Day 55's argument arriving as a constraint. A
    mid-run recording freezes that step's row tuple, and `check_replay_rows` refuses
    the next batch whose rows differ, which takes the loop down rather than answering
    slowly. A lazy capture is a single-shape toy; the warm list is the deployment.
    """
    solo = paired_plans(1, prompts=("abc",), max_tokens=(6,))
    lazy, eager = _both_arms(_app(warm=False), _eager_app(), plans=solo)

    check_same_answers(lazy, eager)
    assert lazy.served.captures >= 1
    with pytest.raises(AcceptanceFailure, match="graph"):
        check_nothing_recorded_while_serving(lazy)


def test_health_carries_the_live_counters_while_the_loop_is_running():
    """Asked through the app rather than off the engine, because the whole point of
    publishing them is that the process holding the object is not the one asking."""

    async def scenario():
        app = _app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                before = capture_from_health((await client.get("/health")).json())
                body = {"model": "nanoserve", "prompt": "abc", "max_tokens": 4, "stream": False}
                await client.post("/v1/completions", json=body)
                after = capture_from_health((await client.get("/health")).json())
        return before, after

    before, after = asyncio.run(asyncio.wait_for(scenario(), 30.0))
    window = after.since(before)
    assert window.calls > 0
    assert window.captures == 0
    assert window.eager_calls == 0
    check_all_shapes_captured(window)


def test_a_server_launched_without_graphs_publishes_no_runtime_section():
    """And it reads back as a capture that is off rather than as one that is
    broken, which is the distinction the eager arm of every comparison rests on."""

    async def scenario():
        app = _eager_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://nano") as client:
            async with app.router.lifespan_context(app):
                return (await client.get("/health")).json()

    payload = asyncio.run(asyncio.wait_for(scenario(), 30.0))
    assert "cuda_graphs" not in payload
    assert capture_from_health(payload).mode == "off"


def test_the_plans_are_the_same_request_on_both_arms():
    """The one property that makes the comparison a comparison. Not a statement
    about the harness being tidy: a plan is data with no clock in it, so the only
    thing that differs between the two arms is which engine answered."""
    bodies = [p.body("nanoserve") for p in PLANS]
    assert bodies == [p.body("nanoserve") for p in PLANS]
    assert all(isinstance(p, ClientPlan) for p in PLANS)
