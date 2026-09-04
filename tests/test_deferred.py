"""Day 48 tests: the readback that covers k steps, and the tokens it overshoots.

Day 47 moved the step's synchronisation from `sample` to `collect` and said so
plainly: a sync moved is not a sync removed. The host still stops once a step,
because a stop rule needs a Python int and the int does not exist until the
kernels have run. This file is about the only way to stop fewer times, which is
to stop *later*: hold step N's tokens on the device, launch step N+1 out of that
same tensor, and bring several steps home in one journey.

Four things are worth testing and they are not the same thing:

  1. **The output must not change.** Deferral reorders when a token is looked at,
     not what was drawn. A greedy run with `defer_window=k` has to produce exactly
     the tokens, the finish reasons and the lengths of the same run with deferral
     off, for every k, through preemption and through EOS. Everything else in this
     file is worthless if that one fails.
  2. **The count really drops, and it drops by the right factor.** k batches home
     in one `torch.cat` and one `.tolist()` is 1/k transfers per step, measured off
     `Engine.output` rather than derived.
  3. **The overshoot is real and it is bounded.** A request whose stop token is
     still on the device is still in the batch, so the engine forwards rows for it
     that nobody keeps. That is Day 29's `waste_fraction` coming back off zero,
     and the bound is `window` rows per finished request.
  4. **The blocks have to be there before the token is.** Under deferral the
     request's length lags the cache row's, so the scheduler must reserve
     `lookahead` tokens of headroom. Without it `BlockTable.append` reaches the
     allocator itself, the pool is booked twice for one sequence, and the damage
     surfaces later as a stranger's K/V.
"""

from __future__ import annotations

import pytest
import torch

from nanoserve.cache import BlockAllocator
from nanoserve.config import ModelConfig
from nanoserve.deferred import (
    DeferredOutputProcessor,
    best_window,
    check_input_is_held,
    check_overshoot_bounded,
    check_tokens_conserved,
    check_window_respected,
    deferred_saving_per_step,
    expected_overshoot,
    net_saving_per_step,
    overshoot_bound,
    overshoot_waste_fraction,
    overshoot_cost_per_step,
    render,
)
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.output import OutputUnsound, Readback, TokenBatch
from nanoserve.sampling import SamplingParams
from nanoserve.scheduler import Request, RequestState, Scheduler

# --- fixtures -------------------------------------------------------------------


def _tiny_config() -> ModelConfig:
    """The same small-but-structurally-real config the other model tests use."""
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


def _engine(model, *, defer_window=0, num_blocks=64, block_size=4, max_batch_size=4) -> Engine:
    return Engine.build(
        model,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=max_batch_size,
        defer_window=defer_window,
    )


def _req(request_id, prompt, max_new_tokens=6, eos_token_id=None) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        sampling=SamplingParams(),
    )


def _running(request_id: str, max_new_tokens: int = 8, eos_token_id=None) -> Request:
    """A request already in the state a sampled row's owner is in."""
    request = _req(request_id, [1, 2, 3], max_new_tokens, eos_token_id)
    request.transition_to(RequestState.RUNNING)
    return request


def _batch(requests, tokens) -> TokenBatch:
    return TokenBatch(
        torch.tensor(tokens, dtype=torch.long), [r.request_id for r in requests]
    )


# --- the arithmetic -------------------------------------------------------------


class TestOvershootArithmetic:
    """What a window of k costs, in rows nobody keeps."""

    def test_the_bound_is_the_window(self):
        # A batch is drained at most `window` steps after it was sampled, and every
        # step in between forwards a row for a request that has already stopped.
        assert overshoot_bound(1) == 1
        assert overshoot_bound(8) == 8

    def test_the_expected_overshoot_is_half_a_window_plus_a_half(self):
        # The stop token is equally likely to land in any of the `window` batches
        # that go home together: the first costs `window` rows, the last costs 1.
        assert expected_overshoot(1) == pytest.approx(1.0)
        assert expected_overshoot(2) == pytest.approx(1.5)
        assert expected_overshoot(8) == pytest.approx(4.5)

    def test_a_window_of_one_still_overshoots_by_a_row(self):
        # The point Day 47 left implicit. Even a one-step lag means the stop is seen
        # after the next forward has already been launched.
        assert overshoot_bound(1) == 1
        assert expected_overshoot(1) > 0.0

    def test_the_waste_fraction_is_the_day_29_number_coming_back(self):
        # 4.5 wasted rows on a request that kept 100 tokens.
        assert overshoot_waste_fraction(8, 100) == pytest.approx(4.5 / 104.5)
        # And a short request pays it in full: continuous batching drove this to
        # zero and deferral puts some of it back.
        assert overshoot_waste_fraction(8, 5) > 0.4

    def test_a_window_of_zero_is_not_a_window(self):
        with pytest.raises(ValueError, match="at least one"):
            overshoot_bound(0)
        with pytest.raises(ValueError, match="at least one"):
            expected_overshoot(0)

    def test_a_request_that_keeps_nothing_has_no_waste_to_divide(self):
        with pytest.raises(ValueError, match="output tokens"):
            overshoot_waste_fraction(4, 0)


class TestTheTrade:
    """Synchronisations saved against rows wasted: the two sides of the window."""

    def test_the_saving_is_the_syncs_a_window_removes(self):
        # 1 transfer a step becomes 1/k, at `latency_s` each.
        assert deferred_saving_per_step(1, latency_s=20e-6) == pytest.approx(0.0)
        assert deferred_saving_per_step(2, latency_s=20e-6) == pytest.approx(10e-6)
        assert deferred_saving_per_step(8, latency_s=20e-6) == pytest.approx(17.5e-6)

    def test_the_saving_saturates_and_the_cost_does_not(self):
        # The whole shape of the day. The saving is bounded by one transfer per
        # step no matter how large the window gets; the wasted rows are linear in
        # it, so there is a window past which every extra step of lag is a loss.
        big = deferred_saving_per_step(1024, latency_s=20e-6)
        assert big < 20e-6
        assert overshoot_cost_per_step(1024, row_s=1e-3, output_tokens=100) > big

    def test_the_cost_is_the_wasted_rows_spread_over_the_run(self):
        # 4.5 rows at 1ms, over a request that ran 100 steps.
        assert overshoot_cost_per_step(8, row_s=1e-3, output_tokens=100) == pytest.approx(
            4.5e-5
        )

    def test_the_net_is_a_difference_and_it_can_be_negative(self):
        cheap_rows = net_saving_per_step(8, latency_s=20e-6, row_s=1e-6, output_tokens=100)
        assert cheap_rows > 0.0
        dear_rows = net_saving_per_step(8, latency_s=20e-6, row_s=1e-2, output_tokens=4)
        assert dear_rows < 0.0

    def test_the_best_window_is_one_when_the_rows_cost_more_than_the_stop(self):
        # A cheap readback and an expensive forward: do not defer at all.
        assert best_window(latency_s=1e-9, row_s=1e-2, output_tokens=8) == 1

    def test_the_best_window_grows_when_the_stop_costs_more_than_the_rows(self):
        # An expensive synchronisation and a long generation: lag is nearly free.
        assert best_window(latency_s=1e-3, row_s=1e-6, output_tokens=1000) > 8

    def test_the_best_window_is_reported_with_its_own_arithmetic(self):
        window = best_window(latency_s=50e-6, row_s=2e-3, output_tokens=200)
        best = net_saving_per_step(window, latency_s=50e-6, row_s=2e-3, output_tokens=200)
        for other in (1, 2, 4, 8, 16, 32):
            assert (
                net_saving_per_step(other, latency_s=50e-6, row_s=2e-3, output_tokens=200)
                <= best + 1e-15
            )


class TestRender:
    def test_the_table_names_the_window_the_syncs_and_the_waste(self):
        text = render(latency_s=20e-6, row_s=1e-3, output_tokens=100, windows=(1, 2, 8))
        assert "window" in text
        assert "overshoot" in text
        # 1/8 of a transfer a step, in the honest fractional unit.
        assert "0.12" in text or "0.13" in text

    def test_the_table_can_carry_a_title(self):
        text = render(
            latency_s=20e-6,
            row_s=1e-3,
            output_tokens=100,
            windows=(1,),
            title="a 20us readback",
        )
        assert text.startswith("a 20us readback")


# --- the processor --------------------------------------------------------------


class TestHolding:
    """What the deferred processor keeps, and for how long."""

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            DeferredOutputProcessor(window=0)

    def test_nothing_is_held_before_anything_is_deferred(self):
        processor = DeferredOutputProcessor(window=2)
        assert processor.held == 0
        assert processor.newest is None

    def test_deferring_costs_no_transfer(self):
        processor = DeferredOutputProcessor(window=1)
        requests = [_running("a"), _running("b")]
        processor.defer(_batch(requests, [5, 6]), requests)
        assert processor.transfers == 0
        assert processor.held == 1
        assert requests[0].output_token_ids == []

    def test_settling_leaves_the_newest_behind(self):
        # The newest batch is the next step's decode input, so it stays on the
        # device: draining it would be exactly the readback this day removes.
        processor = DeferredOutputProcessor(window=1)
        requests = [_running("a")]
        processor.defer(_batch(requests, [5]), requests)
        processor.defer(_batch(requests, [6]), requests)
        processor.settle()
        assert processor.held == 1
        assert processor.newest.request_ids == ("a",)
        assert requests[0].output_token_ids == [5]

    def test_a_window_of_k_holds_k_batches_before_it_pays(self):
        processor = DeferredOutputProcessor(window=3)
        requests = [_running("a")]
        for token in (1, 2, 3):
            processor.defer(_batch(requests, [token]), requests)
            processor.settle()
        assert processor.transfers == 0
        assert processor.held == 3
        processor.defer(_batch(requests, [4]), requests)
        processor.settle()
        assert processor.transfers == 1
        assert processor.held == 1
        assert requests[0].output_token_ids == [1, 2, 3]

    def test_k_steps_go_home_in_exactly_one_journey(self):
        # The whole point. `torch.cat` is a launch, not a synchronisation, so k
        # batches concatenate on the device and one `.tolist()` brings them all.
        readback = Readback()
        processor = DeferredOutputProcessor(window=4, readback=readback)
        requests = [_running("a"), _running("b")]
        for step in range(9):
            processor.defer(_batch(requests, [step, 100 + step]), requests)
            processor.settle()
        assert readback.transfers == 2
        assert readback.elements == 16
        assert requests[0].output_token_ids == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_the_transfer_rate_is_one_over_the_window(self):
        processor = DeferredOutputProcessor(window=4)
        requests = [_running("a", max_new_tokens=64)]
        for step in range(17):
            processor.defer(_batch(requests, [step]), requests)
            processor.settle()
        assert processor.transfers_per_step == pytest.approx(0.25)

    def test_flush_drains_the_newest_too(self):
        processor = DeferredOutputProcessor(window=4)
        requests = [_running("a")]
        for token in (1, 2, 3):
            processor.defer(_batch(requests, [token]), requests)
        assert processor.flush() == [1, 2, 3]
        assert processor.held == 0
        assert processor.transfers == 1

    def test_flushing_nothing_is_not_a_transfer(self):
        processor = DeferredOutputProcessor(window=2)
        assert processor.flush() == []
        assert processor.transfers == 0

    def test_an_empty_batch_is_not_held(self):
        # A prefill-only iteration with nothing sampled would otherwise sit in the
        # deque as a row set that matches nobody and force a drain next step.
        processor = DeferredOutputProcessor(window=2)
        processor.defer(TokenBatch.empty(), [])
        assert processor.held == 0

    def test_a_batch_whose_rows_are_not_its_requests_is_refused(self):
        processor = DeferredOutputProcessor(window=1)
        requests = [_running("a"), _running("b")]
        with pytest.raises(OutputUnsound, match="row order"):
            processor.defer(_batch(requests, [1, 2]), list(reversed(requests)))

    def test_the_tokens_reach_their_requests_in_row_order(self):
        processor = DeferredOutputProcessor(window=1)
        a, b, c = _running("a"), _running("b"), _running("c")
        processor.defer(_batch([a, b, c], [7, 8, 9]), [a, b, c])
        processor.flush()
        assert (a.output_token_ids, b.output_token_ids, c.output_token_ids) == (
            [7],
            [8],
            [9],
        )

    def test_a_held_batch_resolves_once_however_many_look(self):
        readback = Readback()
        processor = DeferredOutputProcessor(window=1, readback=readback)
        requests = [_running("a")]
        batch = _batch(requests, [5])
        processor.defer(batch, requests)
        processor.flush()
        assert batch.is_resolved
        batch.resolve(readback)
        batch.resolve(readback)
        assert readback.transfers == 1


class TestOvershoot:
    """The rows nobody keeps, and the two different reasons a token is dropped."""

    def test_a_token_for_a_finished_request_is_discarded_and_counted(self):
        processor = DeferredOutputProcessor(window=1)
        request = _running("a", max_new_tokens=1)
        processor.defer(_batch([request], [5]), [request])
        processor.defer(_batch([request], [6]), [request])
        processor.settle()  # 5 lands, the budget is spent, the request finishes
        assert request.is_finished
        processor.flush()  # 6 arrives for a request that is already done
        assert request.output_token_ids == [5]
        assert processor.overshoot_tokens == 1
        assert processor.abandoned_tokens == 0

    def test_a_token_for_a_preempted_request_is_discarded_separately(self):
        # Not overshoot: this token was really wanted, and the request will sample
        # it again when its prefill recomputes the K/V it just lost. Counting it as
        # overshoot would blame the deferral for a cost preemption already had.
        processor = DeferredOutputProcessor(window=1)
        request = _running("a")
        processor.defer(_batch([request], [5]), [request])
        request.preempt()
        processor.flush()
        assert request.output_token_ids == []
        assert processor.overshoot_tokens == 0
        assert processor.abandoned_tokens == 1

    def test_every_deferred_token_is_applied_dropped_or_still_held(self):
        processor = DeferredOutputProcessor(window=2)
        a = _running("a", max_new_tokens=2)
        b = _running("b", max_new_tokens=8)
        for token in (1, 2, 3, 4):
            processor.defer(_batch([a, b], [token, token]), [a, b])
            processor.settle()
        check_tokens_conserved(processor)
        assert processor.deferred_tokens == 8
        assert processor.tokens + processor.overshoot_tokens + processor.abandoned_tokens + (
            sum(batch.num_rows for batch in processor.held_batches)
        ) == 8

    def test_a_broken_conservation_is_a_raise(self):
        processor = DeferredOutputProcessor(window=1)
        request = _running("a")
        processor.defer(_batch([request], [5]), [request])
        processor.flush()
        processor.deferred_tokens += 3  # a batch that went somewhere unaccounted
        with pytest.raises(OutputUnsound, match="conserv"):
            check_tokens_conserved(processor)


class TestGates:
    def test_a_processor_holding_more_than_its_window_is_a_raise(self):
        processor = DeferredOutputProcessor(window=1)
        requests = [_running("a")]
        processor.defer(_batch(requests, [1]), requests)
        check_window_respected(processor)
        processor.defer(_batch(requests, [2]), requests)
        with pytest.raises(OutputUnsound, match="holding"):
            check_window_respected(processor)

    def test_overshoot_above_the_bound_is_a_raise(self):
        processor = DeferredOutputProcessor(window=1)
        processor.overshoot_tokens = 2
        check_overshoot_bounded(processor, finished_requests=2)
        with pytest.raises(OutputUnsound, match="overshot"):
            check_overshoot_bounded(processor, finished_requests=1)

    def test_a_host_built_decode_input_is_a_raise(self):
        requests = [_running("a"), _running("b")]
        batch = _batch(requests, [4, 9])
        check_input_is_held(batch.tokens.unsqueeze(1), batch)
        with pytest.raises(OutputUnsound, match="rebuilt"):
            check_input_is_held(torch.tensor([[4], [9]], dtype=torch.long), batch)

    def test_a_decode_input_of_the_wrong_shape_is_a_raise(self):
        requests = [_running("a")]
        batch = _batch(requests, [4])
        with pytest.raises(OutputUnsound, match="one column"):
            check_input_is_held(batch.tokens, batch)


# --- the engine -----------------------------------------------------------------


class TestEngineEquivalence:
    """The only property that matters: deferral changes when, not what."""

    @pytest.mark.parametrize("window", [1, 2, 4])
    def test_a_deferred_run_generates_exactly_what_an_immediate_one_does(self, window):
        prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        plain = _engine(_model()).generate(prompts, max_new_tokens=6)
        deferred = _engine(_model(), defer_window=window).generate(prompts, max_new_tokens=6)
        assert deferred == plain

    def test_a_stop_token_stops_at_the_same_token(self):
        model = _model()
        plain = _engine(model)
        deferred = _engine(model, defer_window=3)
        first = plain.generate([[1, 2, 3]], max_new_tokens=8)[0]
        eos = first[-1]
        assert plain.generate([[1, 2, 3]], max_new_tokens=8, eos_id=eos)[0] == first
        assert deferred.generate([[1, 2, 3]], max_new_tokens=8, eos_id=eos)[0] == first

    def test_deferral_survives_preemption(self):
        # A pool too small for three requests, so somebody is evicted and comes back
        # over prompt-plus-generated. Its held token is abandoned rather than
        # overshot, and the recompute samples it again.
        model = _model()
        prompts = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        plain = _engine(model, num_blocks=6, block_size=4, max_batch_size=3)
        deferred = _engine(
            model, defer_window=2, num_blocks=6, block_size=4, max_batch_size=3
        )
        assert deferred.generate(prompts, max_new_tokens=5) == plain.generate(
            prompts, max_new_tokens=5
        )
        assert deferred.scheduler.num_preemptions > 0

    def test_the_finish_reasons_are_the_same(self):
        model = _model()
        plain = _engine(model)
        deferred = _engine(model, defer_window=2)
        for engine in (plain, deferred):
            for i, prompt in enumerate(([1, 2], [3, 4, 5])):
                engine.add_request(_req(f"r{i}", prompt, max_new_tokens=4))
        assert [r.finish_reason for r in plain.run_to_completion()] == [
            r.finish_reason for r in deferred.run_to_completion()
        ]


class TestEngineTransfers:
    def test_the_engine_reads_back_once_every_window_steps(self):
        engine = _engine(_model(), defer_window=4, max_batch_size=1)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=16))
        engine.run_to_completion()
        # A single row that never leaves the batch is the steady state: no row set
        # changes, so no step is forced to drain early.
        assert engine.output.transfers_per_step == pytest.approx(0.25, abs=0.1)

    def test_a_window_of_one_does_not_reduce_the_count(self):
        # The honest half of the day. One step of lag hides nothing on one stream
        # and removes no journey; what it buys is on the input side.
        engine = _engine(_model(), defer_window=1, max_batch_size=1)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=16))
        engine.run_to_completion()
        assert engine.output.transfers_per_step == pytest.approx(1.0, abs=0.05)

    def test_the_decode_input_is_the_tensor_the_sampler_left(self):
        engine = _engine(_model(), defer_window=2, max_batch_size=1)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=4))
        engine.step()  # prefill: samples and holds
        held = engine.output.newest
        seen = {}
        original = engine.model.forward

        def spy(input_ids, *args, **kwargs):
            seen["input_ids"] = input_ids
            return original(input_ids, *args, **kwargs)

        engine.model.forward = spy
        engine.step()
        check_input_is_held(seen["input_ids"], held)

    def test_building_the_decode_input_never_brings_a_tensor_home(self):
        # Day 47's trick, pointed at the input side: `tolist`, `item` and `__int__`
        # raise for the duration of the call, so "did not read back" is something a
        # test can fail on rather than a claim in a docstring.
        #
        # The call and not the whole step, and the difference was a finding rather
        # than a convenience. `paged_attention_batched_reference` validated its
        # `context_lens` with `int(context_lens.min())` on every layer of every
        # step, so on Day 48 the reference kernel read back more times per step
        # than the output path ever did, and this test could only honestly claim
        # `_decode_input_ids`. Day 49 paid that bill: the bounds come down from
        # `BatchedPagedKVCache.context_bounds`, where they are already Python ints,
        # and `test_compiled.py` makes the wider claim over the whole forward. The
        # scope here stays as it was, because it is this day's claim.
        engine = _engine(_model(), defer_window=4, max_batch_size=1)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=8))
        engine.step()  # prefill: samples and holds
        requests = list(engine.scheduler.running)
        with _no_readback():
            input_ids = engine._decode_input_ids(requests, None)
        check_input_is_held(input_ids, engine.output.newest)

    def test_a_row_change_makes_the_input_cost_a_readback(self):
        # The other half of the same claim. When the held rows are not this step's
        # rows the engine has to have Python ints, and it pays for them rather than
        # using a tensor that means something else.
        engine = _engine(_model(), defer_window=4, max_batch_size=2)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=8))
        engine.step()
        engine.add_request(_req("b", [4, 5, 6], max_new_tokens=8))
        engine.step()  # admits b: prefill and decode in one iteration
        requests = list(engine.scheduler.running)
        before = engine.output.transfers
        engine._decode_input_ids(requests, None)
        assert engine.output.transfers == before + 1

    def test_a_row_set_change_forces_the_readback_early(self):
        # Deferral is only safe while the batch is the batch that was sampled. A
        # request finishing or arriving means the held tensor is not this step's
        # input, and the engine pays for the ints it now needs.
        engine = _engine(_model(), defer_window=8, max_batch_size=2)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=2))
        engine.add_request(_req("b", [4, 5, 6], max_new_tokens=8))
        engine.run_to_completion()
        assert engine.output.transfers >= 2


class TestEngineOvershoot:
    def test_a_deferred_run_forwards_rows_nobody_keeps(self):
        engine = _engine(_model(), defer_window=4, max_batch_size=1)
        engine.add_request(_req("a", [1, 2, 3], max_new_tokens=8))
        engine.run_to_completion()
        assert engine.output.overshoot_tokens > 0
        assert engine.waste_fraction > 0.0

    def test_the_waste_is_exactly_the_rows_that_were_dropped(self):
        engine = _engine(_model(), defer_window=3, max_batch_size=2)
        for i, prompt in enumerate(([1, 2, 3], [4, 5, 6])):
            engine.add_request(_req(f"r{i}", prompt, max_new_tokens=6))
        engine.run_to_completion()
        dropped = engine.output.overshoot_tokens + engine.output.abandoned_tokens
        assert engine.issued_tokens - engine.collected_tokens == dropped

    def test_an_immediate_run_still_wastes_nothing(self):
        engine = _engine(_model(), defer_window=0, max_batch_size=2)
        for i, prompt in enumerate(([1, 2, 3], [4, 5, 6])):
            engine.add_request(_req(f"r{i}", prompt, max_new_tokens=6))
        engine.run_to_completion()
        assert engine.waste_fraction == 0.0

    def test_the_overshoot_stays_inside_the_window(self):
        engine = _engine(_model(), defer_window=2, max_batch_size=2)
        for i, prompt in enumerate(([1, 2, 3], [4, 5, 6])):
            engine.add_request(_req(f"r{i}", prompt, max_new_tokens=5))
        finished = engine.run_to_completion()
        check_overshoot_bounded(engine.output, finished_requests=len(finished))

    def test_the_run_ends_with_nothing_left_on_the_device(self):
        engine = _engine(_model(), defer_window=4, max_batch_size=2)
        for i, prompt in enumerate(([1, 2, 3], [4, 5, 6])):
            engine.add_request(_req(f"r{i}", prompt, max_new_tokens=5))
        engine.run_to_completion()
        assert engine.output.held == 0
        check_tokens_conserved(engine.output)


class TestLookahead:
    """The blocks have to exist before the token the request has not seen yet."""

    def test_the_scheduler_reserves_the_windows_worth_of_headroom(self):
        allocator = BlockAllocator(num_blocks=32, block_size=4)
        scheduler = Scheduler(allocator, max_batch_size=2, lookahead=2)
        request = _req("a", [1, 2, 3, 4])
        request.transition_to(RequestState.RUNNING)
        request.block_ids = list(allocator.allocate_for(4))
        # 4 tokens fit one block; two tokens of headroom need a second.
        assert scheduler.blocks_needed_for(request) == 1

    def test_no_lookahead_is_the_day_33_rule_unchanged(self):
        allocator = BlockAllocator(num_blocks=32, block_size=4)
        scheduler = Scheduler(allocator, max_batch_size=2)
        request = _req("a", [1, 2, 3, 4])
        request.transition_to(RequestState.RUNNING)
        request.block_ids = list(allocator.allocate_for(4))
        assert scheduler.blocks_needed_for(request) == 0

    def test_a_deferring_engine_wires_its_window_into_the_scheduler(self):
        engine = _engine(_model(), defer_window=3)
        assert engine.scheduler.lookahead == 3

    def test_the_pool_comes_back_whole_after_a_deferred_run(self):
        # The bug lookahead exists to stop. Without the headroom the cache row grows
        # past the blocks the request holds, `BlockTable.append` allocates one
        # itself, and the release frees only the request's list: the pool leaks a
        # block per boundary crossed and nothing raises.
        engine = _engine(_model(), defer_window=4, num_blocks=32, block_size=4)
        for i, prompt in enumerate(([1, 2, 3, 4, 5], [6, 7, 8])):
            engine.add_request(_req(f"r{i}", prompt, max_new_tokens=8))
        engine.run_to_completion()
        assert engine.allocator.num_free == engine.allocator.num_blocks

    def test_every_row_holds_the_blocks_the_cache_wrote_into(self):
        engine = _engine(_model(), defer_window=4, num_blocks=32, block_size=4)
        engine.add_request(_req("a", [1, 2, 3, 4], max_new_tokens=8))
        for _ in range(6):
            engine.step()
            for request in engine.scheduler.running:
                table = engine.cache.tables[request.slot]
                assert set(table.block_ids) <= set(request.block_ids)

    def test_a_door_check_counts_the_headroom_too(self):
        # A request that fits the pool exactly does not fit it with a window's
        # worth of overshoot on top, and the door is the only place that can say so
        # before the engine hangs on it.
        allocator = BlockAllocator(num_blocks=2, block_size=4)
        scheduler = Scheduler(allocator, max_batch_size=1, lookahead=4)
        from nanoserve.cache import KVCacheExhausted

        with pytest.raises(KVCacheExhausted):
            scheduler.add_request(_req("a", [1, 2, 3, 4], max_new_tokens=4))


# --- the no-readback harness ----------------------------------------------------


class _no_readback:
    """Make every route home from a tensor raise, for the duration of a block.

    Day 47's instrument, reused. A synchronisation leaves nothing in a return
    value, so the only way to assert one did not happen is to make it impossible.
    """

    _NAMES = ("tolist", "item", "__int__")

    def __enter__(self):
        self._saved = {name: getattr(torch.Tensor, name) for name in self._NAMES}

        def refuse(*args, **kwargs):
            raise AssertionError("the decode step brought a tensor home")

        for name in self._NAMES:
            setattr(torch.Tensor, name, refuse)
        return self

    def __exit__(self, *exc):
        for name, original in self._saved.items():
            setattr(torch.Tensor, name, original)
        return False
