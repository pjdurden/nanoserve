"""Day 44 tests: the outside baseline, and the identity that factorises a speedup.

Three tiers, the same shape Day 43 used, because the module has the same shape:
a stdlib-only arithmetic core, a pairing gate on top of it, and a thin runner that
wires the core to a real model.

  1. **The arithmetic**, over records placed by hand on a ruler. Occupancy is the
     area under the in-flight curve divided by the wall clock, throughput is tokens
     over the wall clock, and the two are related by an identity that has to hold on
     every input rather than on the ones a benchmark happens to produce.
  2. **The gate.** `compare` exists to refuse. Two runs that generated different
     text, or different amounts of it, produce a ratio that is not about speed, and
     the tests here are mostly about that refusal firing.
  3. **The wiring**, over the tiny random model: the engine at concurrency four
     against the same model driven one prompt at a time, which is the shape of the
     real HF comparison with the second system swapped out for one the pure suite
     can build. The weights-gated test at the bottom is the real thing.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.baseline import (
    Comparison,
    RunRecord,
    RunUnsound,
    SystemRun,
    UnfairComparison,
    check_concurrent,
    check_serial,
    compare,
    first_divergence,
    hf_generate_one,
    run_concurrent,
    run_serial,
)
from nanoserve.cache import KVCacheExhausted
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel

from reference import WEIGHTS_DIR, requires_weights


def _rec(
    rid: str,
    started: float,
    finished: float,
    *,
    first: float | None = None,
    prompt: int = 4,
    out: tuple[int, ...] = (11, 12, 13, 14),
    prefill: float | None = None,
    served: float | None = None,
) -> RunRecord:
    return RunRecord(
        request_id=rid,
        prompt_tokens=prompt,
        output_token_ids=out,
        started_s=started,
        finished_s=finished,
        first_token_s=first,
        prefill_s=prefill,
        served_s=served,
    )


# --- tier 1: the arithmetic ---------------------------------------------------


def test_record_duration_and_token_count():
    rec = _rec("a", 1.0, 3.5, out=(1, 2, 3))
    assert rec.duration_s == pytest.approx(2.5)
    assert rec.output_tokens == 3


def test_ttft_and_decode_sum_to_the_duration():
    """The record's own two-part split, and it is an identity like Day 43's five."""
    rec = _rec("a", 1.0, 5.0, first=2.0)
    assert rec.ttft_s == pytest.approx(1.0)
    assert rec.decode_s == pytest.approx(3.0)
    assert rec.ttft_s + rec.decode_s == pytest.approx(rec.duration_s)


def test_decode_rate_counts_the_tokens_after_the_first():
    """The first token is bought by TTFT, so it is not decode's to be credited with.

    Four tokens across three seconds of decode is three tokens per second, not
    four over three. Counting the prefill's token in the decode rate is the normal
    way an inter-token number comes out too flattering.
    """
    rec = _rec("a", 0.0, 5.0, first=2.0, out=(1, 2, 3, 4))
    assert rec.decode_s == pytest.approx(3.0)
    assert rec.decode_tps == pytest.approx(1.0)


def test_a_record_with_no_first_token_has_no_ttft():
    rec = _rec("a", 0.0, 2.0)
    assert rec.ttft_s is None
    assert rec.decode_s is None
    assert rec.decode_tps == 0.0


def test_a_record_cannot_finish_before_it_started():
    with pytest.raises(ValueError, match="before it started"):
        _rec("a", 3.0, 1.0)


def test_a_first_token_outside_the_record_is_refused():
    with pytest.raises(ValueError, match="first token"):
        _rec("a", 1.0, 2.0, first=5.0)


def test_run_totals_count_every_record():
    run = SystemRun.from_records("x", [_rec("a", 0.0, 1.0, out=(1, 2)), _rec("b", 1.0, 2.0)])
    assert run.n_requests == 2
    assert run.output_tokens == 6
    assert run.prompt_tokens == 8


def test_wall_clock_is_the_first_start_to_the_last_finish():
    run = SystemRun.from_records("x", [_rec("a", 10.0, 12.0), _rec("b", 11.0, 15.0)])
    assert run.wall_clock_s == pytest.approx(5.0)


def test_back_to_back_requests_have_an_occupancy_of_one():
    """Three requests, one second each, run in series: exactly one in flight, always."""
    run = SystemRun.from_records(
        "serial",
        [_rec("a", 0.0, 1.0), _rec("b", 1.0, 2.0), _rec("c", 2.0, 3.0)],
    )
    assert run.occupancy == pytest.approx(1.0)


def test_occupancy_is_the_mean_number_in_flight():
    """Four requests over a two-second window, each alive for one second: mean two."""
    run = SystemRun.from_records(
        "batched",
        [
            _rec("a", 0.0, 1.0),
            _rec("b", 0.0, 1.0),
            _rec("c", 1.0, 2.0),
            _rec("d", 1.0, 2.0),
        ],
    )
    assert run.resident_s == pytest.approx(4.0)
    assert run.occupancy == pytest.approx(2.0)


def test_a_gap_between_requests_drops_the_occupancy_below_one():
    """Idle time is in the wall clock, so the baseline's own overhead shows up here."""
    run = SystemRun.from_records("serial", [_rec("a", 0.0, 1.0), _rec("b", 3.0, 4.0)])
    assert run.occupancy == pytest.approx(0.5)


def test_throughput_is_occupancy_times_the_per_request_rate():
    """The identity the whole module is built on, on a run whose numbers are round.

    Four requests, four tokens each, two at a time for two seconds: sixteen tokens
    over two seconds is 8 tokens/s, and it is two requests in flight each running
    at four tokens per request-second. Neither factor alone is the throughput and
    the product is, exactly.
    """
    run = SystemRun.from_records(
        "batched",
        [
            _rec("a", 0.0, 1.0),
            _rec("b", 0.0, 1.0),
            _rec("c", 1.0, 2.0),
            _rec("d", 1.0, 2.0),
        ],
    )
    assert run.throughput_tps == pytest.approx(8.0)
    assert run.occupancy == pytest.approx(2.0)
    assert run.per_request_tps == pytest.approx(4.0)
    assert run.throughput_tps == pytest.approx(run.occupancy * run.per_request_tps)


def test_the_identity_holds_on_a_lopsided_run_too():
    """No round numbers, overlapping spans, ragged outputs: still exact."""
    run = SystemRun.from_records(
        "ragged",
        [
            _rec("a", 0.0, 3.7, out=(1, 2, 3, 4, 5)),
            _rec("b", 1.3, 2.1, out=(1,)),
            _rec("c", 2.9, 9.4, out=tuple(range(11))),
        ],
    )
    run.check()
    assert run.throughput_tps == pytest.approx(run.occupancy * run.per_request_tps, rel=1e-12)


def test_an_empty_run_reports_zeroes_rather_than_dividing_by_zero():
    """A window in which nothing was answered prints a report; it does not raise."""
    run = SystemRun.from_records("nothing", [])
    assert run.n_requests == 0
    assert run.throughput_tps == 0.0
    assert run.occupancy == 0.0
    assert run.per_request_tps == 0.0
    run.check()


def test_a_record_outside_the_wall_clock_is_unsound():
    """The window has to contain the run, or every rate below it is denominated wrong."""
    run = SystemRun("x", (_rec("a", 0.0, 10.0),), wall_clock_s=2.0)
    with pytest.raises(RunUnsound, match="outside the measured window"):
        run.check()


def test_mean_latency_and_mean_ttft_average_over_the_right_denominator():
    """A request that never spoke has no TTFT, and a zero for it would be a lie."""
    run = SystemRun.from_records(
        "x",
        [_rec("a", 0.0, 4.0, first=1.0), _rec("b", 0.0, 6.0)],
    )
    assert run.n_answered == 1
    assert run.mean_latency_s == pytest.approx(5.0)
    assert run.mean_ttft_s == pytest.approx(1.0)


def test_a_record_with_no_service_stamp_counts_all_of_it_as_service():
    """The right default for a system with no queue: it never waited for a slot."""
    rec = _rec("a", 0.0, 4.0)
    assert rec.service_s == pytest.approx(4.0)
    assert rec.queued_s == pytest.approx(0.0)


def test_a_record_cannot_be_served_for_longer_than_it_existed():
    with pytest.raises(ValueError, match="longer than it existed"):
        _rec("a", 0.0, 2.0, served=5.0)


def test_batch_occupancy_ignores_the_queue_and_system_occupancy_does_not():
    """The day's own first bug, pinned. Four requests handed over at once against
    one slot: four in the system, one in the batch, and only one of those two
    numbers may be called batching."""
    run = SystemRun.from_records(
        "one-slot",
        [_rec(f"r{i}", 0.0, 4.0, served=1.0) for i in range(4)],
    )
    assert run.occupancy == pytest.approx(4.0)
    assert run.batch_occupancy == pytest.approx(1.0)
    assert run.queue_share == pytest.approx(0.75)


def test_both_factorisations_give_the_same_throughput():
    """Two exact splits of one number: on the system, and on service alone."""
    run = SystemRun.from_records(
        "one-slot",
        [_rec(f"r{i}", 0.0, 4.0, served=1.0, out=(1, 2)) for i in range(4)],
    )
    run.check()
    assert run.throughput_tps == pytest.approx(2.0)
    assert run.throughput_tps == pytest.approx(run.occupancy * run.per_request_tps)
    assert run.throughput_tps == pytest.approx(run.batch_occupancy * run.per_served_tps)


def test_a_serial_run_has_no_queue_share():
    run = SystemRun.from_records("hf", [_rec("a", 0.0, 1.0), _rec("b", 1.0, 2.0)])
    assert run.queue_share == pytest.approx(0.0)
    assert run.batch_occupancy == pytest.approx(run.occupancy)


# --- tier 2: the gate ---------------------------------------------------------


def test_check_serial_accepts_requests_that_never_overlap():
    run = SystemRun.from_records("hf", [_rec("a", 0.0, 1.0), _rec("b", 1.0, 2.0)])
    check_serial(run)


def test_check_serial_rejects_an_overlap():
    """A "one at a time" baseline that batched is not the baseline anybody meant."""
    run = SystemRun.from_records("hf", [_rec("a", 0.0, 2.0), _rec("b", 1.0, 3.0)])
    with pytest.raises(RunUnsound, match="overlap"):
        check_serial(run)


def test_check_concurrent_rejects_a_run_that_was_accidentally_serial():
    """The mirror of Day 42's `check_offered_load`, and it fails the same way.

    An engine benchmark whose requests happened to run one at a time still prints
    a throughput number. It is a measurement of the wrong system.
    """
    run = SystemRun.from_records("nanoserve", [_rec("a", 0.0, 1.0), _rec("b", 1.0, 2.0)])
    with pytest.raises(RunUnsound, match="one at a time"):
        check_concurrent(run, min_occupancy=2.0)


def test_check_concurrent_is_not_fooled_by_a_queue():
    """Eight requests handed over at once against one slot: the system occupancy is
    high, the batch is one, and gating on the wrong one waves this through."""
    run = SystemRun.from_records(
        "nanoserve",
        [_rec(f"r{i}", 0.0, 8.0, served=1.0) for i in range(8)],
    )
    assert run.occupancy == pytest.approx(8.0)
    with pytest.raises(RunUnsound, match="one at a time"):
        check_concurrent(run, min_occupancy=2.0)


def test_first_divergence_is_none_when_the_tokens_agree():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_first_divergence_reports_the_index_they_split_at():
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_compare_refuses_different_request_counts():
    a = SystemRun.from_records("hf", [_rec("a", 0.0, 1.0)])
    b = SystemRun.from_records("ns", [_rec("a", 0.0, 1.0), _rec("b", 0.0, 1.0)])
    with pytest.raises(UnfairComparison, match="same number of requests"):
        compare(a, b)


def test_compare_refuses_different_prompts():
    a = SystemRun.from_records("hf", [_rec("a", 0.0, 1.0, prompt=4)])
    b = SystemRun.from_records("ns", [_rec("a", 0.0, 1.0, prompt=9)])
    with pytest.raises(UnfairComparison, match="prompt"):
        compare(a, b)


def test_compare_refuses_different_output_lengths():
    """Fewer tokens is not faster. This is the easiest way to fake a speedup."""
    a = SystemRun.from_records("hf", [_rec("a", 0.0, 1.0, out=(1, 2, 3, 4))])
    b = SystemRun.from_records("ns", [_rec("a", 0.0, 1.0, out=(1, 2))])
    with pytest.raises(UnfairComparison, match="tokens"):
        compare(a, b)


def test_compare_refuses_runs_that_produced_different_text():
    """The correctness gate: a speedup against a system that said something else."""
    a = SystemRun.from_records("hf", [_rec("a", 0.0, 1.0, out=(1, 2, 3, 4))])
    b = SystemRun.from_records("ns", [_rec("a", 0.0, 1.0, out=(1, 2, 7, 4))])
    with pytest.raises(UnfairComparison, match="index 2"):
        compare(a, b)


def test_compare_refuses_a_run_with_no_measured_time():
    a = SystemRun("hf", (_rec("a", 0.0, 0.0, out=(1,)),), wall_clock_s=0.0)
    b = SystemRun.from_records("ns", [_rec("a", 0.0, 1.0, out=(1,))])
    with pytest.raises(UnfairComparison, match="no measured time"):
        compare(a, b)


def _paired_runs() -> Comparison:
    """One prompt-set, run serially and then two at a time. Same tokens both sides."""
    out = (1, 2, 3, 4)
    serial = SystemRun.from_records(
        "hf",
        [_rec(f"r{i}", float(i), float(i) + 1.0, first=float(i) + 0.5, out=out) for i in range(4)],
    )
    batched = SystemRun.from_records(
        "nanoserve",
        [
            _rec("r0", 0.0, 1.5, first=0.6, out=out),
            _rec("r1", 0.0, 1.5, first=0.6, out=out),
            _rec("r2", 1.5, 3.0, first=2.1, out=out),
            _rec("r3", 1.5, 3.0, first=2.1, out=out),
        ],
    )
    return compare(serial, batched)


def test_the_speedup_factorises_exactly():
    """speedup == occupancy gain x per-request rate ratio, and it is an identity.

    Four seconds serially against three seconds batched: a 1.33x throughput win.
    It is not 1.33x of anything getting faster. It is 2x the requests in flight,
    times 0.67x the rate each one ran at.
    """
    cmp = _paired_runs()
    assert cmp.throughput_speedup == pytest.approx(4.0 / 3.0)
    assert cmp.occupancy_gain == pytest.approx(2.0)
    assert cmp.per_request_ratio == pytest.approx(2.0 / 3.0)
    cmp.check()
    assert cmp.throughput_speedup == pytest.approx(
        cmp.occupancy_gain * cmp.per_request_ratio, rel=1e-12
    )


def test_the_batch_factorisation_is_exact_too():
    """The same speedup, split the other way: mean batch x what a row produced."""
    cmp = _paired_runs()
    cmp.check()
    assert cmp.throughput_speedup == pytest.approx(
        cmp.batch_gain * cmp.per_served_ratio, rel=1e-12
    )


def test_the_batch_gain_is_not_the_occupancy_gain_when_there_is_a_queue():
    """The two factorisations disagree exactly where the queue is, which is the
    point of carrying both: only the batch one is a claim about hardware."""
    out = (1, 2, 3, 4)
    serial = SystemRun.from_records(
        "hf", [_rec(f"r{i}", float(i), float(i) + 1.0, out=out) for i in range(4)]
    )
    queued = SystemRun.from_records(
        "ns", [_rec(f"r{i}", 0.0, 4.0, served=1.0, out=out) for i in range(4)]
    )
    cmp = compare(serial, queued)
    cmp.check()
    assert cmp.occupancy_gain == pytest.approx(4.0)
    assert cmp.batch_gain == pytest.approx(1.0)
    assert cmp.throughput_speedup == pytest.approx(1.0)
    assert cmp.per_served_ratio == pytest.approx(1.0)


def test_a_per_request_ratio_below_one_means_every_caller_waited_longer():
    """The half of a batching win that a throughput headline never mentions."""
    cmp = _paired_runs()
    assert cmp.per_request_ratio < 1.0
    assert cmp.mean_latency_ratio < 1.0


def test_the_comparison_prefers_prefill_over_ttft_when_both_carry_it():
    """TTFT under load is mostly queue, so pitting it against a queueless baseline
    measures the queue. The prefill spans are the apples-to-apples pair."""
    out = (1, 2)
    serial = SystemRun.from_records(
        "hf", [_rec("a", 0.0, 2.0, first=1.0, prefill=1.0, out=out)]
    )
    batched = SystemRun.from_records(
        "ns", [_rec("a", 0.0, 4.0, first=3.0, prefill=0.5, out=out)]
    )
    cmp = compare(serial, batched)
    assert cmp.ttft_ratio == pytest.approx(1.0 / 3.0)
    assert cmp.prefill_ratio == pytest.approx(2.0)


def test_the_prefill_ratio_is_zero_when_a_side_did_not_measure_one():
    out = (1, 2)
    serial = SystemRun.from_records("hf", [_rec("a", 0.0, 2.0, first=1.0, out=out)])
    batched = SystemRun.from_records("ns", [_rec("a", 0.0, 4.0, first=3.0, prefill=0.5, out=out)])
    assert compare(serial, batched).prefill_ratio == 0.0


def test_the_summary_names_both_factors():
    text = "\n".join(_paired_runs().summary_lines())
    assert "occupancy" in text
    assert "per-request" in text
    assert "batch" in text
    assert "per-served" in text
    assert "hf" in text and "nanoserve" in text


# --- tier 3: the wiring -------------------------------------------------------

PROMPTS = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15]]
MAX_NEW = 6


def _tiny_model() -> LlamaModel:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _stream_one(model: LlamaModel):
    """A `generate_one` over the single-sequence cached loop: the stand-in baseline.

    It stamps on the first yielded token for the same reason the HF closure hangs
    a logits processor off `generate`: the prefill is the forward that produced it,
    and stamping at the end of the call would report the whole generation as TTFT.
    """

    def generate_one(prompt, stamp):
        out = []
        for token in model.generate_stream(
            torch.tensor([prompt]), max_new_tokens=MAX_NEW, temperature=0.0
        ):
            if not out:
                stamp()
            out.append(token)
        return out

    return generate_one


def _engine() -> Engine:
    return Engine.build(_tiny_model(), num_blocks=32, block_size=16, max_batch_size=4)


def test_run_serial_records_every_prompt_in_order():
    run = run_serial(PROMPTS, _stream_one(_tiny_model()), name="stream")
    assert run.n_requests == 4
    assert [r.prompt_tokens for r in run.records] == [len(p) for p in PROMPTS]
    assert all(r.output_tokens == MAX_NEW for r in run.records)
    run.check()


def test_run_serial_really_is_serial():
    run = run_serial(PROMPTS, _stream_one(_tiny_model()), name="stream")
    check_serial(run)
    assert run.occupancy <= 1.0


def test_run_serial_reports_ttft_as_its_prefill():
    """No queue means TTFT is prefill, and saying so is what makes it comparable."""
    run = run_serial(PROMPTS, _stream_one(_tiny_model()), name="stream")
    for rec in run.records:
        assert rec.prefill_s == pytest.approx(rec.ttft_s)


def test_run_concurrent_holds_more_than_one_row():
    run = run_concurrent(_engine(), PROMPTS, max_new_tokens=MAX_NEW)
    assert run.n_requests == 4
    assert run.occupancy > 1.0
    check_concurrent(run, min_occupancy=2.0)
    run.check()


def test_run_concurrent_carries_the_engine_prefill_from_the_timeline():
    """Day 43's stamps, read back out. The engine already knew; nothing new is timed."""
    run = run_concurrent(_engine(), PROMPTS, max_new_tokens=MAX_NEW)
    for rec in run.records:
        assert rec.prefill_s is not None
        assert 0.0 < rec.prefill_s <= rec.ttft_s + 1e-9


def test_run_concurrent_recovers_the_batch_size_from_the_residency():
    """Four prompts on four slots: the mean batch is near four and the queue is near
    nothing, and both come out of Day 43's accumulators rather than a new timer."""
    run = run_concurrent(_engine(), PROMPTS, max_new_tokens=MAX_NEW)
    assert run.batch_occupancy > 2.0
    assert run.batch_occupancy <= 4.0 + 1e-9
    assert run.queue_share < 0.25


def test_one_slot_makes_the_two_occupancies_disagree():
    """The measured version of the day's gotcha, on a real engine: the system
    occupancy climbs with the queue while the batch stays at one."""
    engine = Engine.build(_tiny_model(), num_blocks=32, block_size=16, max_batch_size=1)
    run = run_concurrent(engine, PROMPTS, max_new_tokens=MAX_NEW)
    assert run.batch_occupancy < 1.05
    assert run.occupancy > 1.5
    assert run.queue_share > 0.25
    run.check()
    with pytest.raises(RunUnsound, match="one at a time"):
        check_concurrent(run, min_occupancy=2.0)


def test_the_two_runs_agree_token_for_token():
    """The gate passing is the day's real claim: same text, so the ratio is speed."""
    serial = run_serial(PROMPTS, _stream_one(_tiny_model()), name="stream")
    batched = run_concurrent(_engine(), PROMPTS, max_new_tokens=MAX_NEW)
    for a, b in zip(serial.records, batched.records):
        assert a.output_token_ids == b.output_token_ids
    compare(serial, batched)


def test_the_identity_survives_a_real_clock():
    """Floats out of `perf_counter`, not a ruler. Still exact to twelve digits."""
    for run in (
        run_serial(PROMPTS, _stream_one(_tiny_model()), name="stream"),
        run_concurrent(_engine(), PROMPTS, max_new_tokens=MAX_NEW),
    ):
        assert run.throughput_tps == pytest.approx(
            run.occupancy * run.per_request_tps, rel=1e-12
        )


def test_a_real_comparison_factorises():
    cmp = compare(
        run_serial(PROMPTS, _stream_one(_tiny_model()), name="stream"),
        run_concurrent(_engine(), PROMPTS, max_new_tokens=MAX_NEW),
    )
    cmp.check()
    assert cmp.occupancy_gain > 1.0
    assert len(cmp.summary_lines()) > 3


def test_run_concurrent_refuses_a_prompt_it_cannot_fit():
    """The engine's own door check, not silently a shorter run."""
    engine = Engine.build(_tiny_model(), num_blocks=1, block_size=16, max_batch_size=2)
    with pytest.raises(KVCacheExhausted):
        run_concurrent(engine, [list(range(40))], max_new_tokens=MAX_NEW)


# --- the real baseline, when the weights are here -----------------------------


@requires_weights
def test_hf_generate_and_this_engine_produce_the_same_tokens():
    """The claim underneath every number in today's log, on the real model.

    Greedy on both sides, same prompt, same budget: if the tokens differ, the
    speedup is comparing two different computations and no arithmetic above this
    line means anything. Four tokens, one prompt, because this runs on CPU.
    """
    from transformers import AutoModelForCausalLM

    from nanoserve.loader import load_weights

    prompt = [128000, 791, 1296, 315, 264]
    hf = AutoModelForCausalLM.from_pretrained(WEIGHTS_DIR, torch_dtype=torch.float32).eval()
    serial = run_serial([prompt], hf_generate_one(hf, max_new_tokens=4), name="hf")

    config = ModelConfig.from_json(WEIGHTS_DIR)
    model = LlamaModel(config, load_weights(WEIGHTS_DIR, config))
    engine = Engine.build(model, num_blocks=64, block_size=16, max_batch_size=2)
    batched = run_concurrent(engine, [prompt], max_new_tokens=4)

    assert serial.records[0].output_token_ids == batched.records[0].output_token_ids
    compare(serial, batched)
