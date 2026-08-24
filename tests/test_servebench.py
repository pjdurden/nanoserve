"""Day 42 tests: the serving benchmark, and the load it can prove it offered.

Two tiers, the same split every measurement day in this repo has used. The
arithmetic tier builds `RequestRecord`s and `LoadReport`s out of scripted numbers
and pins TTFT, inter-token latency, the throughput denominator and the percentile
definition to the decimal, with no server anywhere near it. The live tier runs the
real app under uvicorn on a real port and drives it with the two load generators,
because the interesting claim of the day is not that the arithmetic is right, it
is that a *closed* loop cannot see the queue it is standing in.

Nothing here needs `./weights`. The tiny random model is the one every serving
test since Day 37 has used: the question is what the scheduler does to latency
when more requests arrive than there are slots, and a 1B model would only make
the same shape take longer to draw.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from nanoserve.acceptance import ClientPlan, live_server
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.server import create_app
from nanoserve.servebench import (
    LoadReport,
    MeasurementUnsound,
    RequestRecord,
    burst_arrivals,
    check_offered_load,
    check_schedule_kept,
    fixed_arrivals,
    percentile,
    poisson_arrivals,
    run_closed_loop,
    run_open_loop,
)
from nanoserve.serving import AsyncEngine

# --- the same tiny model and byte tokenizer every serving test uses -------------

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."


class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(ch) for ch in text]

    def decode(self, token_ids) -> str:
        return "".join(ALPHABET[i] for i in token_ids)


TOKENIZER = ByteTokenizer()


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


def _model(seed: int = 0) -> LlamaModel:
    torch.manual_seed(seed)
    cfg = _tiny_config()
    tensors = {name: torch.randn(*shape) for name, shape in expected_shapes(cfg).items()}
    tensors[LM_HEAD] = tensors[EMBED]
    return LlamaModel(cfg, Weights(tensors, cfg))


def _app(max_batch_size: int = 2):
    """Two slots, so a burst of eight has to queue for them."""
    engine = Engine.build(
        _model(), num_blocks=64, block_size=4, max_batch_size=max_batch_size
    )
    return create_app(AsyncEngine(engine), TOKENIZER, model_name="nanoserve")


def _plans(n: int, *, max_tokens: int = 6) -> list[ClientPlan]:
    """n identical streaming clients. Identical on purpose: the only thing this
    benchmark is allowed to vary between two runs is when the requests arrive."""
    return [
        ClientPlan(client_id=f"b{i}", prompt="the test of a", max_tokens=max_tokens, stream=True)
        for i in range(n)
    ]


def _record(
    client_id: str = "r0",
    *,
    scheduled_at: float = 0.0,
    sent_at: float = 0.0,
    frame_times=(1.0, 1.5, 2.0),
    done_at: float = 2.0,
    output_tokens: int = 3,
    prompt_tokens: int = 5,
    status: int = 200,
    error: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        client_id=client_id,
        scheduled_at=scheduled_at,
        sent_at=sent_at,
        frame_times=tuple(frame_times),
        done_at=done_at,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        status=status,
        error=error,
    )


# --- the percentile definition ----------------------------------------------------


def test_percentile_is_nearest_rank_so_every_number_reported_really_happened():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # nearest rank: ceil(q/100 * n), 1-indexed. p50 of ten samples is the fifth,
    # which is 5.0, and not the 5.5 an interpolating median would invent.
    assert percentile(values, 50) == 5.0
    assert percentile(values, 90) == 9.0
    assert percentile(values, 100) == 10.0
    assert percentile(values, 0) == 1.0


def test_percentile_does_not_need_sorted_input_and_survives_an_empty_run():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([], 99) == 0.0


def test_percentile_rejects_a_q_outside_the_scale():
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 101)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], -1)


def test_a_p99_of_twenty_samples_is_just_the_worst_one():
    # the reason LoadReport reports n next to every percentile. With twenty
    # samples there is no 99th of anything: ceil(0.99 * 20) is 20, so the p99 is
    # the worst sample, and p95 is the one below it. Quoting a p99 from a run this
    # small is quoting the max under a name that promises a distribution.
    values = [float(i) for i in range(20)]
    assert percentile(values, 99) == max(values)
    assert percentile(values, 95) == 18.0


# --- one request's timings --------------------------------------------------------


def test_record_splits_the_wait_into_send_lag_ttft_and_the_gaps():
    r = _record(scheduled_at=0.0, sent_at=0.5, frame_times=(2.0, 2.5, 2.9), done_at=3.0)

    assert r.send_lag_s == pytest.approx(0.5)
    # what the server owes: from the byte going out to the first byte coming back.
    assert r.ttft_s == pytest.approx(1.5)
    # what the user waited: the schedule said 0.0 and the first token landed at 2.0.
    assert r.arrival_ttft_s == pytest.approx(2.0)
    assert r.itls == pytest.approx([0.5, 0.4])
    assert r.median_itl_s == pytest.approx(0.45)
    assert r.e2e_s == pytest.approx(2.5)
    assert r.arrival_e2e_s == pytest.approx(3.0)


def test_frames_are_not_tokens_so_the_two_cadences_differ():
    # three frames, four tokens: the detokenizer held one back as half a character,
    # exactly the case Day 39 built the incremental detokenizer for.
    r = _record(sent_at=0.0, frame_times=(1.0, 1.3, 1.6), done_at=1.7, output_tokens=4)

    assert r.n_frames == 3
    assert r.median_itl_s == pytest.approx(0.3)  # what the screen did
    # what the engine did: the same span divided by the gaps between real tokens.
    assert r.decode_span_s == pytest.approx(0.6)
    assert r.mean_token_itl_s == pytest.approx(0.2)
    assert r.tokens_per_frame == pytest.approx(4 / 3)


def test_a_one_frame_request_has_no_inter_token_latency_and_does_not_divide_by_zero():
    r = _record(sent_at=0.0, frame_times=(1.0,), done_at=1.0, output_tokens=1)

    assert r.ttft_s == pytest.approx(1.0)
    assert r.itls == []
    assert r.median_itl_s == 0.0
    assert r.mean_token_itl_s == 0.0
    assert r.decode_span_s == 0.0


def test_a_record_rejects_a_clock_that_ran_backwards():
    with pytest.raises(ValueError, match="before it was scheduled"):
        _record(scheduled_at=1.0, sent_at=0.5)
    with pytest.raises(ValueError, match="before the request was sent"):
        _record(sent_at=1.0, frame_times=(0.5, 1.5), done_at=2.0)
    with pytest.raises(ValueError, match="out of order"):
        _record(sent_at=0.0, frame_times=(1.0, 0.9), done_at=2.0)
    with pytest.raises(ValueError, match="before its last frame"):
        _record(sent_at=0.0, frame_times=(1.0, 2.0), done_at=1.5)


def test_a_200_that_delivered_nothing_is_not_ok():
    # a request with no frames has no TTFT, and counting it as a zero would drag
    # every percentile in the report down towards a latency nobody experienced.
    empty = _record(frame_times=(), done_at=2.0, output_tokens=0)
    assert not empty.ok
    assert _record().ok
    assert not _record(status=503).ok
    assert not _record(error="ReadTimeout").ok


# --- the run's report -------------------------------------------------------------


def _overlapping_report() -> LoadReport:
    """Four requests, each 0.9s long, all inside a one-second window."""
    records = tuple(
        _record(
            client_id=f"b{i}",
            scheduled_at=0.0,
            sent_at=0.0,
            frame_times=(0.5, 0.6, 0.7, 0.8),
            done_at=0.9,
            output_tokens=10,
        )
        for i in range(4)
    )
    return LoadReport(records=records, started_at=0.0, finished_at=1.0, target_rate=10.0)


def test_throughput_denominator_is_the_run_window_not_the_sum_of_requests():
    report = _overlapping_report()

    assert report.duration_s == pytest.approx(1.0)
    assert report.output_tokens == 40
    # 40 tokens came out of this server in one second, because four requests were
    # in flight at once. Dividing by the summed per-request time (3.6s) reports
    # 11.1 tok/s, which is one request's rate wearing the run's name.
    assert report.output_tps == pytest.approx(40.0)
    assert report.request_tps == pytest.approx(4.0)


def test_mean_in_flight_is_the_admissibility_number():
    # Little's law: the average number of requests in the system is the summed
    # residency divided by the window. 4 x 0.9s of residency in 1.0s = 3.6.
    assert _overlapping_report().mean_in_flight == pytest.approx(3.6)

    sequential = LoadReport(
        records=(
            _record(client_id="b0", sent_at=0.0, frame_times=(0.1, 0.5), done_at=1.0),
            _record(client_id="b1", scheduled_at=1.0, sent_at=1.0, frame_times=(1.1, 1.5), done_at=2.0),
        ),
        started_at=0.0,
        finished_at=2.0,
    )
    assert sequential.mean_in_flight == pytest.approx(1.0)


def test_a_report_knows_whether_it_kept_up_with_its_own_schedule():
    report = _overlapping_report()
    assert report.offered_rate == pytest.approx(10.0)
    assert report.achieved_rate == pytest.approx(4.0)
    # offered ten a second and served four: the server is behind, so its latency
    # numbers are a measurement of an overloaded system, which is the point.
    assert not report.kept_up
    assert report.max_send_lag_s == 0.0


def test_failures_are_counted_and_kept_out_of_the_latency_numbers():
    report = LoadReport(
        records=(
            _record(client_id="b0", sent_at=0.0, frame_times=(1.0, 2.0), done_at=2.0),
            _record(client_id="b1", sent_at=0.0, frame_times=(), done_at=0.1, status=503),
        ),
        started_at=0.0,
        finished_at=2.0,
    )

    assert report.n_ok == 1
    assert report.n_failed == 1
    assert report.ttfts == pytest.approx([1.0])
    assert report.ttft_p50 == pytest.approx(1.0)


def test_pooled_itls_weight_the_long_requests_and_the_report_says_so():
    long_one = _record(client_id="long", sent_at=0.0, frame_times=tuple(
        1.0 + 0.1 * i for i in range(11)
    ), done_at=2.1, output_tokens=11)
    short_one = _record(client_id="short", sent_at=0.0, frame_times=(1.0, 3.0), done_at=3.0,
                        output_tokens=2)
    report = LoadReport(records=(long_one, short_one), started_at=0.0, finished_at=3.0)

    # ten gaps from the long request, one from the short one: the pool is not a
    # per-request average, and the p50 belongs to whoever generated most tokens.
    assert report.n_itl_samples == 11
    assert report.itl_p50 == pytest.approx(0.1)
    assert report.itl_p99 == pytest.approx(2.0)


def test_the_summary_never_calls_an_open_loop_a_closed_one():
    # three different experiments, and the header line has to tell them apart: a
    # burst is an open loop with no nominal rate, not a closed loop, and reading
    # one as the other is reading a queueing measurement as a service-time one.
    burst = LoadReport(records=(_record(),), started_at=0.0, finished_at=2.0)
    paced = LoadReport(records=(_record(),), started_at=0.0, finished_at=2.0, target_rate=4.0)
    closed = LoadReport(records=(_record(),), started_at=0.0, finished_at=2.0, concurrency=2)

    assert "burst" in burst.summary()
    assert "closed loop" not in burst.summary()
    assert "4.00 req/s" in paced.summary()
    assert "closed loop, 2 clients" in closed.summary()
    # n travels with every percentile, so a p99 of one sample cannot be quoted
    # without the reader seeing that it is one sample.
    assert "n=1" in burst.summary()


def test_a_report_with_a_backwards_window_is_a_bug_not_a_negative_throughput():
    with pytest.raises(ValueError, match="cannot finish before it started"):
        LoadReport(records=(), started_at=5.0, finished_at=4.0)


# --- the checks a benchmark owes itself ------------------------------------------


def test_check_offered_load_fires_when_the_run_was_never_crowded():
    lonely = LoadReport(
        records=(_record(sent_at=0.0, frame_times=(0.1, 0.5), done_at=1.0),),
        started_at=0.0,
        finished_at=1.0,
    )
    with pytest.raises(MeasurementUnsound, match="one request at a time"):
        check_offered_load(lonely, min_in_flight=2.0)

    check_offered_load(_overlapping_report(), min_in_flight=2.0)


def test_check_schedule_kept_fires_when_the_generator_itself_was_the_bottleneck():
    late = LoadReport(
        records=(
            _record(client_id="b0", scheduled_at=0.0, sent_at=0.0,
                    frame_times=(0.5,), done_at=0.6, output_tokens=1),
            _record(client_id="b1", scheduled_at=0.1, sent_at=0.9,
                    frame_times=(1.4,), done_at=1.5, output_tokens=1),
        ),
        started_at=0.0,
        finished_at=1.5,
    )
    assert late.max_send_lag_s == pytest.approx(0.8)
    with pytest.raises(MeasurementUnsound, match="behind its own schedule"):
        check_schedule_kept(late, max_lag_s=0.05)

    check_schedule_kept(_overlapping_report(), max_lag_s=0.05)


# --- arrival schedules ------------------------------------------------------------


def test_fixed_arrivals_are_one_over_the_rate_apart():
    assert fixed_arrivals(5, rate=10.0) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_burst_arrivals_all_land_at_zero():
    assert burst_arrivals(3) == [0.0, 0.0, 0.0]


def test_poisson_arrivals_are_deterministic_for_a_seed_and_hit_the_rate():
    a = poisson_arrivals(500, rate=10.0, seed=7)
    b = poisson_arrivals(500, rate=10.0, seed=7)
    c = poisson_arrivals(500, rate=10.0, seed=8)

    assert a == b
    assert a != c
    assert a[0] == 0.0
    assert all(y >= x for x, y in zip(a, a[1:]))
    # 500 exponential gaps at rate 10 should land near 50 seconds of schedule.
    assert a[-1] == pytest.approx(50.0, rel=0.2)
    # and they are bursty, which is the whole reason to use them: some gaps are
    # far shorter than the mean, so requests really do pile onto each other.
    gaps = [y - x for x, y in zip(a, a[1:])]
    assert min(gaps) < 0.02


def test_arrival_schedules_reject_a_rate_that_cannot_arrive():
    with pytest.raises(ValueError, match="positive"):
        fixed_arrivals(3, rate=0.0)
    with pytest.raises(ValueError, match="positive"):
        poisson_arrivals(3, rate=-1.0, seed=0)
    with pytest.raises(ValueError, match="negative"):
        fixed_arrivals(-1, rate=1.0)


# --- against a live server --------------------------------------------------------


def test_open_loop_measures_a_real_server_over_a_real_socket():
    with live_server(_app()) as server:
        report = asyncio.run(run_open_loop(server.base_url, _plans(8), burst_arrivals(8)))

    assert report.n_failed == 0, report.failures
    assert report.n_ok == 8
    assert report.output_tokens == 8 * 6
    assert report.duration_s > 0.0
    assert report.output_tps > 0.0
    # a burst of eight against two slots really was a crowd.
    check_offered_load(report, min_in_flight=2.0)
    # and the harness kept up with a schedule that asked for everything at once.
    check_schedule_kept(report, max_lag_s=0.5)
    assert report.n_itl_samples > 0
    assert report.ttft_p99 >= report.ttft_p50


def test_the_closed_loop_cannot_see_the_queue_the_open_loop_measures():
    plans = _plans(8)
    with live_server(_app()) as server:
        closed = asyncio.run(run_closed_loop(server.base_url, plans, concurrency=1))
        burst = asyncio.run(run_open_loop(server.base_url, plans, burst_arrivals(8)))

    assert closed.n_failed == 0, closed.failures
    assert burst.n_failed == 0, burst.failures
    # one client at a time never queues: its TTFT is the server's service time and
    # nothing else, and it is the number a closed-loop benchmark quotes.
    assert closed.mean_in_flight == pytest.approx(1.0, abs=0.3)
    assert burst.mean_in_flight > 1.5
    # the same eight requests, the same server, arriving together: six of them wait
    # for a slot, and that wait is latency the closed loop never offered the load
    # to see.
    assert burst.ttft_p99 > closed.ttft_p99
    # and the server that looked slower per request served the run faster.
    assert burst.output_tps > closed.output_tps


def test_a_closed_loop_report_is_never_late_which_is_exactly_the_criticism():
    with live_server(_app()) as server:
        closed = asyncio.run(run_closed_loop(server.base_url, _plans(4), concurrency=2))

    # a closed loop has no schedule to fall behind, so this check passes for free.
    # It is reported anyway so that a reader can see it means nothing here.
    assert closed.max_send_lag_s == 0.0
    check_schedule_kept(closed, max_lag_s=0.0)
    assert closed.concurrency == 2


def test_a_paced_open_loop_below_capacity_barely_queues():
    with live_server(_app()) as server:
        paced = asyncio.run(
            run_open_loop(server.base_url, _plans(4), fixed_arrivals(4, rate=4.0), target_rate=4.0)
        )

    assert paced.n_failed == 0, paced.failures
    assert paced.offered_rate == pytest.approx(4.0)
    # the run took at least as long as the schedule it was paced by.
    assert paced.duration_s >= 0.75


def test_the_benchmark_refuses_a_unary_plan_because_it_cannot_time_tokens():
    unary = [ClientPlan(client_id="u0", prompt="the test", max_tokens=4, stream=False)]
    with live_server(_app()) as server:
        with pytest.raises(ValueError, match="stream"):
            asyncio.run(run_open_loop(server.base_url, unary, burst_arrivals(1)))


def test_a_request_the_pool_can_never_hold_is_a_failure_not_a_crash():
    # 4 blocks of 4 tokens against a prompt that wants more: the server answers
    # 400 and the report counts it as a failure with every other record intact.
    engine = Engine.build(_model(), num_blocks=4, block_size=4, max_batch_size=2)
    app = create_app(AsyncEngine(engine), TOKENIZER, model_name="nanoserve")
    plans = [
        ClientPlan(client_id="ok", prompt="ab", max_tokens=2, stream=True),
        ClientPlan(client_id="huge", prompt="a" * 40, max_tokens=32, stream=True),
    ]
    with live_server(app) as server:
        report = asyncio.run(run_open_loop(server.base_url, plans, burst_arrivals(2)))

    assert report.n_failed == 1
    assert [r.client_id for r in report.failures] == ["huge"]
    assert report.failures[0].status == 400
    assert report.n_ok == 1
