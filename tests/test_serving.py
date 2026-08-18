"""Day 37 tests: many coroutines, one synchronous engine loop, one GPU.

`Engine.step()` is a blocking call that owns the model, the cache and the
scheduler. An HTTP server is a pile of coroutines on one event loop. Wiring them
together is not "call the engine from the handler": that would serialise every
request behind the one holding the loop, which throws away the exact property
Week 8 built. So there is one loop task that steps the engine, an inbox the
handlers push arrivals onto, and one future per request that is resolved when
*that* request finishes rather than when the batch does.

Everything in this file is a claim about that bridge, and they fall into three
groups.

  1. **The answer is unchanged.** Serving is plumbing. A request that goes
     through the bridge must emit exactly the tokens it emits offline, alone or
     sharing the batch with three others, or this is a different model with a
     socket in front of it.
  2. **The loop is never blocked.** The step runs off the event loop thread, so
     a handler can accept a new request, and a waiter can be woken, while the GPU
     is busy. The gate test proves it the only way that is not a stopwatch: it
     makes the step wait for something only the event loop can provide.
  3. **One request's bad day is its own.** A rejected prompt, an abort, a
     disconnect, a step that raises. Each of those has to land on the right
     future, and a per-request error raised inside the shared loop must not take
     the other callers down with it. This is where a serving layer is actually
     hard, and none of it exists offline: `Engine.generate` has one caller and
     may simply raise at it.

Every test runs under `run()`, which is `asyncio.run` with a hard timeout, because
the failure mode of a wrong bridge is a hang, not an assertion. A deadlocked loop
would otherwise wedge the whole suite instead of failing one test.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
import torch

from nanoserve.cache import KVCacheExhausted
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.scheduler import SchedulerOutput
from nanoserve.serving import AsyncEngine, EngineStopped, EngineStuck

# --- the same tiny model the engine tests use -----------------------------------


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


def _engine(num_blocks=64, block_size=4, max_batch_size=4) -> Engine:
    return Engine.build(
        _model(), num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def _offline(prompt: list[int], max_new_tokens: int, **kw) -> list[int]:
    """What a fresh offline engine emits for this prompt on its own.

    The reference every serving test compares against. Same weights, same seed,
    same greedy argmax, so any difference is the bridge inventing something.
    """
    return _engine(**kw).generate([list(prompt)], max_new_tokens=max_new_tokens)[0]


def run(coro, timeout: float = 15.0):
    """Run one coroutine under a hard timeout.

    A bridge bug is usually a hang: a waiter nobody resolves, a loop parked on an
    event nobody sets. `asyncio.run` alone would wait for that forever and take
    the suite with it, so every test here gets a deadline and a failed test
    instead of a wedged process.
    """

    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


async def _until(predicate, tries: int = 5000):
    """Give the event loop turns until `predicate()` holds.

    The bridge's work happens on a task the test does not await, so "has the loop
    got there yet?" is a real question with no `await` that answers it. Polling a
    predicate is honest about that and, unlike a fixed sleep, it neither flakes on
    a slow box nor costs anything on a fast one.
    """
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the condition never held")


async def _quiesce(serving):
    """Wait until the loop has nothing left to do and has parked on its event."""
    await _until(lambda: serving.parked)


class _SpyEngine:
    """Every call the bridge is allowed to make on an engine, plus a record of it.

    The bridge must touch the scheduler only between steps, and this is what
    makes that observable: `in_step` is true exactly while the worker thread is
    inside a forward, so an `add_request` that arrives then is a data race the
    test can count rather than a race the suite hits once a month.
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.in_step = False
        self.step_threads: list[int] = []
        self.arrivals_during_step = 0
        self.aborts_during_step = 0

    def step(self):
        self.in_step = True
        self.step_threads.append(threading.get_ident())
        try:
            return self.engine.step()
        finally:
            self.in_step = False

    def add_request(self, request):
        if self.in_step:
            self.arrivals_during_step += 1
        return self.engine.add_request(request)

    def abort(self, request_id):
        if self.in_step:
            self.aborts_during_step += 1
        return self.engine.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.engine.has_unfinished()

    @property
    def iterations(self) -> int:
        return self.engine.iterations

    @property
    def scheduler(self):
        return self.engine.scheduler


# --- 1. the answer is unchanged -------------------------------------------------


def test_one_request_through_the_bridge_matches_offline():
    prompt = [1, 2, 3]
    expected = _offline(prompt, max_new_tokens=6)

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            return await serving.generate(prompt, max_new_tokens=6)

    request = run(scenario())
    assert request.token_ids == expected
    assert request.finish_reason == "length"


def test_four_concurrent_requests_each_match_their_solo_run():
    """The claim continuous batching makes, now made through a socket's worth of
    indirection: who you shared the forward with does not change your tokens."""
    prompts = [[1, 2, 3], [5, 5], [7, 8, 9, 10], [11]]
    expected = [_offline(p, max_new_tokens=5) for p in prompts]

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            return await asyncio.gather(
                *(serving.generate(p, max_new_tokens=5) for p in prompts)
            )

    got = run(scenario())
    assert [r.token_ids for r in got] == expected


def test_concurrent_requests_share_iterations_rather_than_queueing():
    """Four requests through one loop must cost far fewer iterations than four
    runs, or the bridge has serialised them and the batch is decorative."""

    async def scenario():
        serving = AsyncEngine(_engine())
        async with serving:
            await asyncio.gather(
                *(serving.generate([1, 2, 3], max_new_tokens=5, request_id=f"r{i}")
                  for i in range(4))
            )
            return serving.engine.iterations

    iterations = run(scenario())
    # Each request alone is 1 prefill + 4 decodes = 5 iterations, so 20 serialised.
    assert iterations <= 8


def test_each_waiter_wakes_at_its_own_finish_not_the_batchs():
    """Day 29's head-of-line bill, paid off at the layer the caller sees."""
    order: list[str] = []

    async def scenario():
        async with AsyncEngine(_engine()) as serving:

            async def one(name, budget):
                await serving.generate([1, 2, 3], max_new_tokens=budget, request_id=name)
                order.append(name)

            await asyncio.gather(one("long", 12), one("short", 1))

    run(scenario())
    assert order == ["short", "long"]


# --- 2. the loop is never blocked -----------------------------------------------


def test_the_step_runs_off_the_event_loop_thread():
    spy = _SpyEngine(_engine())

    async def scenario():
        async with AsyncEngine(spy) as serving:
            await serving.generate([1, 2, 3], max_new_tokens=3)
            return threading.get_ident()

    loop_thread = run(scenario())
    assert spy.step_threads, "the engine was never stepped"
    assert all(ident != loop_thread for ident in spy.step_threads)


def test_the_loop_is_free_while_a_step_runs():
    """The point of the thread, asserted without a stopwatch.

    The step blocks on an event that only the event loop sets. If the step ran on
    the loop, the loop would be inside it, the gate would never open, and this
    test would fail on its deadline instead of passing.
    """
    gate = threading.Event()

    class _Gated(_SpyEngine):
        def step(self):
            gate.wait(timeout=5.0)
            return super().step()

    spy = _Gated(_engine())

    async def scenario():
        async with AsyncEngine(spy) as serving:
            task = asyncio.ensure_future(serving.generate([1, 2, 3], max_new_tokens=3))
            await asyncio.sleep(0)  # let the loop task reach its first step
            gate.set()
            return await task

    request = run(scenario())
    assert request.num_output_tokens == 3


def test_the_answer_arrives_on_the_step_that_emitted_it():
    """The engine's `finished` list is a *release* event, not a completion one.

    A request that emits its last token on step N is only reported by
    `SchedulerOutput.finished` on step N+1, because releasing its slot and blocks
    is the next schedule's first job. Waiting for that costs every caller a whole
    iteration of latency for something the engine already knew, so the bridge
    delivers on `request.is_finished` instead. Here the reap step is held shut in
    a worker thread: an answer that arrives anyway is an answer that did not wait
    for it.
    """
    gate = threading.Event()

    class _HoldsTheReap(_SpyEngine):
        def __init__(self, engine):
            super().__init__(engine)
            self.calls = 0

        def step(self):
            if self.calls == 3:
                gate.wait(timeout=5.0)
            self.calls += 1
            return super().step()

    spy = _HoldsTheReap(_engine())

    async def scenario():
        async with AsyncEngine(spy) as serving:
            request = await serving.generate([1, 2, 3], max_new_tokens=3)
            emitted = spy.engine.iterations
            gate.set()
            return request, emitted

    request, emitted = run(scenario())
    assert request.num_output_tokens == 3
    assert emitted == 3


def test_arrivals_are_never_applied_mid_step():
    """The reason there is an inbox at all: two threads, one scheduler."""
    spy = _SpyEngine(_engine())

    async def scenario():
        async with AsyncEngine(spy) as serving:
            tasks = [
                asyncio.ensure_future(
                    serving.generate([1, 2, 3], max_new_tokens=4, request_id=f"r{i}")
                )
                for i in range(4)
            ]
            # Submit two more once the loop is already stepping the first four.
            await _until(lambda: spy.engine.iterations >= 1)
            tasks += [
                asyncio.ensure_future(
                    serving.generate([4, 5], max_new_tokens=4, request_id=f"late{i}")
                )
                for i in range(2)
            ]
            await asyncio.gather(*tasks)

    run(scenario())
    assert spy.arrivals_during_step == 0
    assert spy.aborts_during_step == 0


def test_the_loop_parks_when_idle_and_a_new_arrival_wakes_it():
    """An idle server must not spin the engine, and must not add latency for it."""

    async def scenario():
        serving = AsyncEngine(_engine())
        async with serving:
            await serving.generate([1, 2, 3], max_new_tokens=2)
            await _quiesce(serving)
            parked = (serving.idle_waits, serving.engine.iterations)
            await asyncio.sleep(0.05)  # real time, with nothing to do in it
            spun = serving.engine.iterations
            # And it is asleep, not dead: the next request still runs.
            await serving.generate([4, 5, 6], max_new_tokens=2)
            await _quiesce(serving)
            return parked, spun, serving.idle_waits

    (idle_waits, iterations), spun, later_waits = run(scenario())
    assert idle_waits >= 1
    assert spun == iterations, "the loop stepped an empty engine"
    assert later_waits > idle_waits


# --- 3. one request's bad day is its own ----------------------------------------


def test_a_rejected_prompt_fails_only_its_own_caller():
    """`add_request` refuses a request too large for the whole pool. That refusal
    happens inside the shared loop, and it has to come back out on one future."""

    async def scenario():
        async with AsyncEngine(_engine(num_blocks=8, block_size=4)) as serving:
            big = asyncio.ensure_future(serving.generate([1, 2, 3], max_new_tokens=400))
            small = asyncio.ensure_future(serving.generate([1, 2, 3], max_new_tokens=4))
            with pytest.raises(KVCacheExhausted):
                await big
            return await small

    request = run(scenario())
    assert request.num_output_tokens == 4


def test_an_empty_prompt_is_refused_at_submit():
    """Some errors do not need the loop at all. A malformed request is one, and
    surfacing it synchronously is what lets a handler answer 400 immediately."""

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            with pytest.raises(ValueError):
                serving.submit([], max_new_tokens=4)
            with pytest.raises(ValueError):
                serving.submit([1, 2], max_new_tokens=0)
            return serving.num_active

    assert run(scenario()) == 0


def test_a_duplicate_request_id_is_refused_at_submit():
    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            task = asyncio.ensure_future(
                serving.generate([1, 2, 3], max_new_tokens=6, request_id="dup")
            )
            await asyncio.sleep(0)
            with pytest.raises(ValueError):
                serving.submit([1, 2, 3], max_new_tokens=6, request_id="dup")
            await task

    run(scenario())


def test_abort_while_running_returns_the_partial_answer():
    async def scenario():
        serving = AsyncEngine(_engine())
        async with serving:
            task = asyncio.ensure_future(
                serving.generate([1, 2, 3], max_new_tokens=64, request_id="doomed")
            )
            await _until(lambda: serving.engine.iterations >= 3)  # a few tokens first
            serving.abort("doomed")
            request = await task
            await _quiesce(serving)
            return request, serving.engine.scheduler.num_running

    request, running = run(scenario())
    assert request.finish_reason == "abort"
    assert request.num_output_tokens < 64
    assert running == 0


def test_abort_before_admission_is_not_an_error():
    """The id is in the inbox and unknown to the scheduler, which would raise
    KeyError at it. A handler whose client vanished before the first step is a
    normal event, not a bug."""

    async def scenario():
        serving = AsyncEngine(_engine())
        async with serving:
            # submit is synchronous, so both of these happen before the loop
            # task gets a turn: the id is the bridge's and nobody else's yet.
            future = serving.submit([1, 2, 3], max_new_tokens=8, request_id="gone")
            serving.abort("gone")
            return await future

    request = run(scenario())
    assert request.finish_reason == "abort"
    assert request.output_token_ids == []


def test_a_cancelled_waiter_aborts_its_request():
    """A disconnected client must stop costing GPU. Without this the request runs
    to its budget, holding a slot and blocks nobody is waiting for."""

    async def scenario():
        serving = AsyncEngine(_engine())
        async with serving:
            task = asyncio.ensure_future(
                serving.generate([1, 2, 3], max_new_tokens=64, request_id="hangup")
            )
            await _until(lambda: serving.engine.iterations >= 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await _quiesce(serving)  # the abort lands at the next iteration boundary
            return serving.num_active, serving.engine.has_unfinished()

    active, unfinished = run(scenario())
    assert active == 0
    assert not unfinished


def test_a_step_that_raises_fails_every_waiter():
    """The worst failure mode a shared loop has: one exception, and every open
    connection waits forever for a future nobody will ever resolve."""

    class _Broken(_SpyEngine):
        def step(self):
            raise RuntimeError("the forward blew up")

    serving_ref = {}

    async def scenario():
        serving = AsyncEngine(_Broken(_engine()))
        serving_ref["it"] = serving
        async with serving:
            tasks = [
                asyncio.ensure_future(
                    serving.generate([1, 2, 3], max_new_tokens=4, request_id=f"r{i}")
                )
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # And it stays failed: a later caller is told immediately, not hung.
            with pytest.raises(RuntimeError):
                serving.submit([1, 2, 3], max_new_tokens=4)
            return results

    results = run(scenario())
    assert len(results) == 3
    assert all(isinstance(r, RuntimeError) for r in results)
    assert isinstance(serving_ref["it"].failure, RuntimeError)


def test_a_schedule_that_never_progresses_fails_the_callers():
    """The offline loop's hang guard, rewritten for a server. `run_to_completion`
    may raise into a script; a server has to fail the sockets, because a caller
    that never gets an answer is worse than one that gets an error."""

    class _Stuck(_SpyEngine):
        def step(self):
            return SchedulerOutput()  # nothing scheduled, ever

        def has_unfinished(self) -> bool:
            return True

    async def scenario():
        async with AsyncEngine(_Stuck(_engine()), max_idle_schedules=8) as serving:
            with pytest.raises(EngineStuck):
                await serving.generate([1, 2, 3], max_new_tokens=4)

    run(scenario())


def test_stopping_fails_whatever_is_still_in_flight():
    """Shutdown is not a hang either. Whoever is mid-generation gets told."""

    async def scenario():
        serving = AsyncEngine(_engine())
        await serving.start()
        task = asyncio.ensure_future(serving.generate([1, 2, 3], max_new_tokens=64))
        await asyncio.sleep(0)
        await serving.stop()
        with pytest.raises(EngineStopped):
            await task
        assert not serving.running
        with pytest.raises(EngineStopped):
            serving.submit([1, 2, 3], max_new_tokens=4)

    run(scenario())


def test_start_is_idempotent_and_stop_without_start_is_a_no_op():
    async def scenario():
        serving = AsyncEngine(_engine())
        await serving.stop()  # never started
        await serving.start()
        await serving.start()
        assert serving.running
        await serving.stop()
        assert not serving.running

    run(scenario())


def test_stats_report_what_the_loop_has_done():
    async def scenario():
        serving = AsyncEngine(_engine())
        async with serving:
            await serving.generate([1, 2, 3], max_new_tokens=3)
            await _quiesce(serving)
            return serving.stats()

    stats = run(scenario())
    # One prefill and two decodes emit the three tokens; the fourth step is the
    # reap, which schedules nothing and so is a step the engine does not count.
    assert stats["iterations"] == 3
    assert stats["steps"] == 4
    assert stats["active"] == 0
    assert stats["running"] == 0
    assert stats["waiting"] == 0
    assert stats["idle_waits"] == 1
    assert stats["parked"] is True
