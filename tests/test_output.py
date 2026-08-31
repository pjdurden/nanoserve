"""Day 47 tests: the sampled token, and the journey home it does not have to make.

Day 46 put a stopwatch inside one decode step and the answer was awkward. The
`forward` is the longest phase and almost none of it is removable; the `sample`
phase is a fraction of its length and 86% to 89% of the entire Python loop. Not
because sampling is arithmetic-heavy, but because `sample_batch` returned
`list[int]`, and every one of those ints was a separate `int(tensor)`: a copy back
from the device, and on a GPU a *synchronisation*, because you cannot read a value
the kernels have not produced yet.

That is why Day 46's `recommended_model` said `serial`. CUDA launches are async, so
a host that never reads back runs ahead and its Python hides under the kernels. A
host that reads back stops. One readback per sampled row per step means the engine
stops N times a step, and every microsecond of launch overhead lands on the
critical path in full.

This file is about the count. Three failure modes are worth testing:

  1. **A readback you cannot see is a readback you cannot remove.** So `Readback`
     is an instrument, `TokenBatch` resolves through it exactly once, and the gates
     read the counter rather than the intention.
  2. **Moving a sync is not removing a sync.** The engine still needs Python ints
     to apply a stop rule, so one transfer per step remains. The arithmetic here
     says what that costs and what the next step (deferring it) would buy, and it
     refuses to pretend the remaining one is free.
  3. **A CPU cannot price this.** There is no bus to cross, so a readback is a
     memcpy and a synchronisation is nothing at all. `check_measurable` refuses to
     turn this box's numbers into a claim about a card, the same way Day 46's
     `check_device_timed` refuses to price an optimisation from a CPU profile.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.output import (
    STRATEGIES,
    OutputProcessor,
    OutputUnsound,
    Readback,
    TokenBatch,
    check_device_resident,
    check_in_row_order,
    check_measurable,
    check_single_transfer,
    measure_transfer_s,
    readback_s,
    render,
    sample_ceiling,
    saving_per_step,
    strategy_speedup,
    syncs_per_step,
)
from nanoserve.sampling import GREEDY, BatchedSampler, SamplingParams
from nanoserve.scheduler import Request, RequestState

US = 1e-6


def _tokens(*ids: int) -> torch.Tensor:
    return torch.tensor(list(ids), dtype=torch.long)


def _request(request_id: str = "r", max_new_tokens: int = 8) -> Request:
    """A request in the one state a token can be appended to."""
    request = Request(
        request_id=request_id, prompt_token_ids=[1, 2, 3], max_new_tokens=max_new_tokens
    )
    request.transition_to(RequestState.RUNNING)
    return request


# --- the instrument -------------------------------------------------------------


def test_a_fresh_readback_has_carried_nothing():
    counter = Readback()
    assert (counter.transfers, counter.elements) == (0, 0)


def test_reading_a_tensor_back_counts_one_transfer():
    counter = Readback()
    assert counter.tolist(_tokens(4, 5, 6)) == [4, 5, 6]
    assert counter.transfers == 1


def test_the_elements_are_counted_as_well_as_the_journeys():
    """Both halves of the model: a transfer has a fixed cost and a size."""
    counter = Readback()
    counter.tolist(_tokens(1, 2, 3, 4))
    counter.tolist(_tokens(9))
    assert (counter.transfers, counter.elements) == (2, 5)


def test_a_readback_can_be_reset_between_measurements():
    counter = Readback()
    counter.tolist(_tokens(1))
    counter.reset()
    assert (counter.transfers, counter.elements) == (0, 0)


# --- one step's tokens, still on the device -------------------------------------


def test_a_token_batch_holds_the_tensor_it_was_given():
    tokens = _tokens(7, 8)
    batch = TokenBatch(tokens, ("a", "b"))
    assert batch.tokens is tokens


def test_a_token_batch_knows_how_many_rows_it_carries():
    assert TokenBatch(_tokens(7, 8, 9), ("a", "b", "c")).num_rows == 3


def test_a_token_batch_is_unresolved_until_somebody_reads_it():
    """The property the whole day rests on: holding it costs nothing."""
    assert not TokenBatch(_tokens(7), ("a",)).is_resolved


def test_resolving_a_token_batch_gives_python_ints():
    batch = TokenBatch(_tokens(7, 8), ("a", "b"))
    assert batch.resolve() == [7, 8]
    assert batch.is_resolved


def test_a_token_batch_resolves_exactly_once_however_often_it_is_asked():
    """Memoised on purpose. A second `.tolist()` is a second synchronisation.

    This is easy to get wrong by writing `ids` as a plain property: every caller
    that touches it pays a full round trip, and the count goes back up without a
    single line of the engine changing.
    """
    counter = Readback()
    batch = TokenBatch(_tokens(1, 2), ("a", "b"))
    for _ in range(5):
        assert batch.resolve(counter) == [1, 2]
    assert counter.transfers == 1


def test_a_token_batch_reports_the_device_its_tokens_are_on():
    batch = TokenBatch(_tokens(1), ("a",))
    assert batch.device == torch.device("cpu")


def test_the_row_ids_and_the_tokens_must_be_the_same_length():
    with pytest.raises(ValueError, match="rows"):
        TokenBatch(_tokens(1, 2), ("a",))


def test_a_token_batch_refuses_a_two_dimensional_tensor():
    """`[rows, 1]` is what `multinomial` hands back and it is not one token per row."""
    with pytest.raises(ValueError, match="one dimension"):
        TokenBatch(torch.zeros(2, 1, dtype=torch.long), ("a", "b"))


def test_a_token_batch_refuses_floats():
    """A token id is an index. A float here is a sampler that forgot to draw."""
    with pytest.raises(ValueError, match="integer"):
        TokenBatch(torch.zeros(2), ("a", "b"))


def test_an_empty_token_batch_is_legal_and_resolves_to_nothing():
    batch = TokenBatch.empty()
    assert batch.num_rows == 0
    assert batch.resolve() == []


# --- applying them --------------------------------------------------------------


def test_applying_a_batch_appends_one_token_to_each_request():
    a, b = _request("a"), _request("b")
    processor = OutputProcessor()
    processor.apply(TokenBatch(_tokens(11, 22), ("a", "b")), [a, b])
    assert a.output_token_ids == [11] and b.output_token_ids == [22]


def test_applying_a_batch_costs_exactly_one_transfer():
    """The headline. One journey home per step, whatever the batch size."""
    requests = [_request(f"r{i}") for i in range(6)]
    processor = OutputProcessor()
    processor.apply(TokenBatch(_tokens(*range(6)), tuple(r.request_id for r in requests)), requests)
    assert processor.transfers == 1


def test_the_transfers_per_step_of_a_run_is_one():
    requests = [_request(f"r{i}") for i in range(3)]
    ids = tuple(r.request_id for r in requests)
    processor = OutputProcessor()
    for step in range(4):
        processor.apply(TokenBatch(_tokens(step, step, step), ids), requests)
    assert processor.steps == 4
    assert processor.transfers_per_step == pytest.approx(1.0)


def test_a_processor_counts_the_tokens_it_handed_out():
    requests = [_request(f"r{i}") for i in range(3)]
    ids = tuple(r.request_id for r in requests)
    processor = OutputProcessor()
    for step in range(2):
        processor.apply(TokenBatch(_tokens(step, step, step), ids), requests)
    assert processor.tokens == 6


def test_applying_returns_the_ids_it_applied():
    a, b = _request("a"), _request("b")
    assert OutputProcessor().apply(TokenBatch(_tokens(4, 5), ("a", "b")), [a, b]) == [4, 5]


def test_a_batch_whose_rows_do_not_match_its_requests_is_refused():
    """The bug this exists to catch, and the reason `TokenBatch` carries the ids.

    A `[rows]` tensor is anonymous. Between sampling and collecting, the engine
    releases finished rows and admits new ones, and a row order that has shifted in
    between hands one caller another caller's token. No shape check sees that: the
    lengths still agree.
    """
    a, b = _request("a"), _request("b")
    with pytest.raises(OutputUnsound, match="row order"):
        OutputProcessor().apply(TokenBatch(_tokens(4, 5), ("a", "b")), [b, a])


def test_a_batch_with_the_wrong_number_of_requests_is_refused():
    a = _request("a")
    with pytest.raises(OutputUnsound, match="row order"):
        OutputProcessor().apply(TokenBatch(_tokens(4, 5), ("a", "b")), [a])


def test_an_empty_batch_applies_to_nobody_and_is_not_a_step():
    """A prefill-only iteration still calls collect; it should not skew the count."""
    processor = OutputProcessor()
    processor.apply(TokenBatch.empty(), [])
    assert (processor.steps, processor.transfers) == (0, 0)


def test_a_processor_shares_a_readback_when_it_is_given_one():
    counter = Readback()
    a = _request("a")
    OutputProcessor(counter).apply(TokenBatch(_tokens(3), ("a",)), [a])
    assert counter.transfers == 1


def test_the_transfers_per_step_of_a_processor_that_never_ran_is_zero():
    assert OutputProcessor().transfers_per_step == 0.0


# --- the arithmetic: what a sync costs and how many there are -------------------


def test_the_strategies_are_the_three_this_engine_can_be_in():
    assert STRATEGIES == ("per_row", "one_transfer", "deferred")


def test_the_old_path_synced_once_per_sampled_row_plus_once_for_the_greedy_block():
    """Day 40's `sample_batch`, counted. Four sampled rows and two greedy ones.

    The greedy rows are one batched `argmax` and therefore one `.tolist()`; every
    sampled row was its own `int(multinomial(...))`.
    """
    assert syncs_per_step(6, num_sampled=4, strategy="per_row") == 5.0


def test_an_all_greedy_batch_on_the_old_path_synced_once():
    assert syncs_per_step(8, num_sampled=0, strategy="per_row") == 1.0


def test_an_all_sampled_batch_on_the_old_path_synced_once_per_row():
    assert syncs_per_step(8, num_sampled=8, strategy="per_row") == 8.0


def test_the_new_path_syncs_once_whatever_the_batch():
    for rows in (1, 4, 32):
        assert syncs_per_step(rows, num_sampled=rows, strategy="one_transfer") == 1.0


def test_deferring_the_readback_amortises_it_over_a_window():
    assert syncs_per_step(4, strategy="deferred", window=8) == pytest.approx(0.125)


def test_a_window_of_one_is_the_same_as_not_deferring():
    assert syncs_per_step(4, strategy="deferred", window=1) == 1.0


def test_a_step_with_no_rows_syncs_not_at_all():
    for strategy in STRATEGIES:
        assert syncs_per_step(0, strategy=strategy) == 0.0


def test_more_sampled_rows_than_rows_is_refused():
    with pytest.raises(ValueError, match="sampled"):
        syncs_per_step(2, num_sampled=3)


def test_a_negative_row_count_is_refused():
    with pytest.raises(ValueError, match="rows"):
        syncs_per_step(-1)


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="strategy"):
        syncs_per_step(4, strategy="magic")


def test_a_window_below_one_is_refused():
    with pytest.raises(ValueError, match="window"):
        syncs_per_step(4, strategy="deferred", window=0)


def test_a_readback_costs_its_count_times_the_latency():
    """The model of the day: the bill is per journey, not per byte.

    A `[rows]` int64 readback at rows=8 is 64 bytes. Nothing on a bus that moves
    gigabytes a second. What it costs is the fixed part: a synchronise, a small
    DMA, a launch of the next thing. So the count is the quantity to reduce and the
    size is not.
    """
    assert readback_s(5.0, latency_s=20 * US) == pytest.approx(100 * US)


def test_a_readback_of_nothing_costs_nothing():
    assert readback_s(0.0, latency_s=20 * US) == 0.0


def test_a_negative_latency_is_refused():
    with pytest.raises(ValueError, match="latency"):
        readback_s(1.0, latency_s=-1.0)


def test_the_saving_is_the_difference_between_two_strategies():
    """Six rows, four of them sampled: five syncs before, one after."""
    saving = saving_per_step(6, num_sampled=4, latency_s=20 * US)
    assert saving == pytest.approx(4 * 20 * US)


def test_an_all_greedy_batch_saves_nothing_by_batching_the_readback():
    """The honest corner, and the engine's own default.

    Every `Request` is greedy unless it asks not to be, and the old path already
    read the greedy block back in one `.tolist()`. So the sync count was already
    one and this optimisation buys zero synchronisations on the path the engine
    actually runs. What it buys there is the Python: no gather of `[rows, vocab]`
    floats, no dict, no per-row loop. Two different savings, and conflating them is
    how a benchmark ends up claiming a speedup the profile cannot show.
    """
    assert saving_per_step(8, num_sampled=0, latency_s=20 * US) == 0.0


def test_deferring_saves_more_than_batching_did():
    batched = saving_per_step(4, num_sampled=4, latency_s=20 * US)
    deferred = saving_per_step(
        4, num_sampled=4, latency_s=20 * US, after="deferred", window=8
    )
    assert deferred > batched


def test_a_saving_can_be_asked_for_between_any_two_strategies():
    saving = saving_per_step(
        4, latency_s=20 * US, before="one_transfer", after="deferred", window=4
    )
    assert saving == pytest.approx(0.75 * 20 * US)


def test_a_step_speedup_is_the_step_over_what_is_left_of_it():
    assert strategy_speedup(1.0, 0.2) == pytest.approx(1.25)


def test_saving_nothing_is_a_speedup_of_one():
    assert strategy_speedup(1.0, 0.0) == 1.0


def test_saving_the_whole_step_is_refused_as_a_speedup():
    """An optimisation that leaves zero step left is arithmetic, not a measurement."""
    with pytest.raises(ValueError, match="whole step"):
        strategy_speedup(1.0, 1.0)


def test_a_negative_saving_is_refused():
    with pytest.raises(ValueError, match="saving"):
        strategy_speedup(1.0, -0.1)


# --- the bridge back to Day 46's profile ----------------------------------------


def _profile(sample_host_ms: float = 2.2):
    from nanoserve.profiler import Phase, StepProfile, StepSample

    ms = 1e-3
    phases = (
        Phase("schedule", 0.5 * ms, 0.0),
        Phase("build_inputs", 1.5 * ms, 0.0),
        Phase("forward", 5.0 * ms, 4.6 * ms),
        Phase("sample", sample_host_ms * ms, 0.4 * ms, syncs=True),
        Phase("collect", 0.8 * ms, 0.0),
    )
    total = sum(p.host_s for p in phases)
    samples = [
        StepSample(index=i, kind="decode", batch_size=4, phases=phases, wall_s=total * 1.001)
        for i in range(4)
    ]
    return StepProfile.from_samples("decode", samples)


def test_the_ceiling_is_what_removing_the_sample_phase_would_buy():
    """The number Day 46 left as the one to beat, pointed at one phase."""
    from nanoserve.profiler import speedup_if

    profile = _profile()
    assert sample_ceiling(profile) == pytest.approx(speedup_if(profile, eliminate=("sample",)))


def test_the_ceiling_of_a_bigger_sample_phase_is_higher():
    assert sample_ceiling(_profile(4.0)) > sample_ceiling(_profile(1.0))


def test_the_ceiling_needs_the_phase_to_be_in_the_profile():
    from nanoserve.profiler import ProfileUnsound

    with pytest.raises(ProfileUnsound):
        sample_ceiling(_profile(), phase="readback")


# --- the gates ------------------------------------------------------------------


def test_a_run_that_read_back_once_a_step_passes_the_gate():
    requests = [_request("a")]
    processor = OutputProcessor()
    for _ in range(3):
        processor.apply(TokenBatch(_tokens(1), ("a",)), requests)
    check_single_transfer(processor)


def test_a_run_that_read_back_twice_a_step_is_refused():
    """The regression a helpful debug line reintroduces in one commit."""
    requests = [_request("a")]
    processor = OutputProcessor()
    for _ in range(3):
        batch = TokenBatch(_tokens(1), ("a",))
        processor.apply(batch, requests)
        processor.readback.tolist(_tokens(1))
    with pytest.raises(OutputUnsound, match="per step"):
        check_single_transfer(processor)


def test_the_transfer_gate_has_nothing_to_say_about_a_run_that_never_stepped():
    check_single_transfer(OutputProcessor())


def test_a_sampler_that_returned_a_tensor_passes_the_residency_gate():
    logits = torch.randn(3, 16)
    check_device_resident(BatchedSampler().sample_batch_device(logits, [("a", GREEDY)] * 3), logits)


def test_a_sampler_that_returned_a_list_is_refused():
    """The whole optimisation, expressed as one type check."""
    logits = torch.randn(3, 16)
    with pytest.raises(OutputUnsound, match="tensor"):
        check_device_resident([1, 2, 3], logits)


def test_tokens_on_a_different_device_from_the_logits_are_refused():
    """A `.cpu()` somebody added to make a print work is exactly this failure.

    `meta` stands in for a second device on a box that has one: it is a real
    `torch.device` that is not `cpu`, which is all the gate compares.
    """
    logits = torch.randn(2, 8, device="meta")
    tokens = torch.tensor([1, 2], dtype=torch.long)
    with pytest.raises(OutputUnsound, match="device"):
        check_device_resident(tokens, logits)


def test_a_batch_in_row_order_passes_the_order_gate():
    a, b = _request("a"), _request("b")
    check_in_row_order(TokenBatch(_tokens(1, 2), ("a", "b")), [a, b])


def test_a_batch_out_of_row_order_is_refused():
    a, b = _request("a"), _request("b")
    with pytest.raises(OutputUnsound, match="row order"):
        check_in_row_order(TokenBatch(_tokens(1, 2), ("a", "b")), [b, a])


def test_a_cpu_cannot_be_asked_what_a_readback_costs():
    """Day 46's `check_device_timed`, in the currency of transfers.

    On a CPU box `.tolist()` is a memcpy of 64 bytes out of the same RAM the
    interpreter lives in. There is no bus, no synchronisation and no host running
    ahead of anything, so a latency measured here says nothing whatever about what
    the same call costs on a card. Measuring it anyway and quoting the saving is
    the mistake this refuses.
    """
    with pytest.raises(OutputUnsound, match="no device"):
        check_measurable("cpu")


def test_a_cuda_device_can_be_asked():
    check_measurable("cuda")
    check_measurable("cuda:1")


# --- measuring it ---------------------------------------------------------------


def test_a_transfer_can_be_timed_and_is_positive():
    assert measure_transfer_s("cpu", rows=4, repeats=20) > 0.0


def test_timing_a_transfer_needs_at_least_one_repeat():
    with pytest.raises(ValueError, match="repeats"):
        measure_transfer_s("cpu", rows=4, repeats=0)


# --- the table ------------------------------------------------------------------


def test_the_render_names_every_strategy():
    table = render(8, num_sampled=8, latency_s=20 * US, window=8)
    for strategy in STRATEGIES:
        assert strategy in table


def test_the_render_shows_the_sync_count_of_each_strategy():
    table = render(8, num_sampled=8, latency_s=20 * US, window=8)
    assert "8.00" in table and "1.00" in table


def test_the_render_carries_a_title_when_it_is_given_one():
    assert "decode x 8" in render(8, num_sampled=8, latency_s=20 * US, title="decode x 8")


# --- through the engine ---------------------------------------------------------


def _tiny_engine(max_batch_size: int = 4):
    from nanoserve.config import ModelConfig
    from nanoserve.engine import Engine
    from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
    from nanoserve.model import LlamaModel

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
    model = LlamaModel(cfg, Weights(tensors, cfg))
    return Engine.build(model, num_blocks=64, block_size=4, max_batch_size=max_batch_size)


def _run(engine, max_new_tokens: int = 5, sampling=None):
    for i, prompt in enumerate([[1, 2, 3], [4, 5], [6]]):
        engine.add_request(
            Request(
                request_id=f"r{i}",
                prompt_token_ids=prompt,
                max_new_tokens=max_new_tokens,
                sampling=sampling or SamplingParams(),
            )
        )
    engine.run_to_completion()
    return engine


def test_an_engine_carries_an_output_processor():
    assert isinstance(_tiny_engine().output, OutputProcessor)


def test_an_engine_reads_its_tokens_back_once_per_step():
    engine = _run(_tiny_engine())
    assert engine.output.transfers == engine.iterations
    check_single_transfer(engine.output)


def test_an_engine_reads_back_one_element_per_issued_token():
    """The elements are the tokens; the transfers are the steps. Two counters."""
    engine = _run(_tiny_engine())
    assert engine.output.readback.elements == engine.issued_tokens


def test_a_sampled_engine_still_reads_back_once_per_step():
    """The case that used to cost one sync per row, now one for the batch."""
    engine = _run(_tiny_engine(), sampling=SamplingParams(temperature=1.0, top_p=0.9))
    assert engine.output.transfers == engine.iterations


def test_the_engine_collects_every_token_it_issued():
    engine = _run(_tiny_engine())
    assert engine.output.tokens == engine.collected_tokens == engine.issued_tokens


def test_the_engine_still_generates_what_it_generated_before():
    """The invariant under all of this: the same requests, the same tokens.

    Greedy, so it is deterministic and comparable across the change. The whole day
    is a change to how a token gets from the device to a list, and if the list
    changed, the day is a bug.
    """
    def generated() -> dict[str, list[int]]:
        engine = _tiny_engine()
        finished = []
        for i, prompt in enumerate([[1, 2, 3], [4, 5], [6]]):
            engine.add_request(
                Request(request_id=f"r{i}", prompt_token_ids=prompt, max_new_tokens=5)
            )
        finished = engine.run_to_completion()
        return {r.request_id: r.output_token_ids for r in finished}

    assert generated() == generated()


def test_the_engine_s_sync_point_moved_from_sampling_to_collecting():
    """The result to state carefully: a sync was moved, not removed.

    `sample` no longer comes home, so it is no longer the phase that stops the
    host. `collect` is, because appending a token and applying a stop rule needs
    the integer. The step still has exactly one synchronisation in it and
    `recommended_model` still says `serial`, which is the honest reading: the
    overlapped model does not apply to this engine yet.
    """
    from nanoserve.profiler import StepRecorder, recommended_model

    engine = _tiny_engine()
    engine.recorder = StepRecorder()
    _run(engine)
    profile = engine.recorder.profile("engine").select("decode")
    totals = profile.phase_totals()
    assert profile.sync_points == 1
    assert recommended_model(profile) == "serial"
    assert set(totals) >= {"sample", "collect"}
