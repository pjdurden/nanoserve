"""Day 39 tests: one future becomes one queue, and the queue becomes SSE.

Day 37's bridge parks each caller on a future that resolves once, at that
request's last token. That is the right shape for `/v1/completions` and the wrong
shape for streaming, which wants the *first* token as soon as it exists and every
one after it. So the future becomes a queue: the loop pushes each new output
token onto it at the same `_settle` that would have resolved the future, and the
caller reads them with `async for`.

Nothing about the engine changes, which is the point worth checking hardest. The
tokens a streamed request emits must be exactly the tokens it emits unary, and
exactly the tokens it emits offline. Streaming is a delivery schedule.

Three groups again, mirroring Day 37:

  1. **The answer is unchanged, and it arrives early.** Same ids as offline, in
     order, and the first one is readable while the request is still running,
     which is the only thing that distinguishes this from `generate` with extra
     steps.
  2. **A stream is one row of a shared batch.** Streams run next to unary
     requests and next to each other, and a consumer that reads slowly must not
     slow the loop down for anybody else. There is no backpressure here on
     purpose: you cannot pause a GPU batch for one socket.
  3. **Ending badly.** A client that hangs up mid-stream, an abort, a loop that
     dies, a shutdown. Each has to end the stream rather than hang it, and a
     disconnect has to free the slot and the blocks the way Day 37's cancel does.

Then the wire. SSE has one property that shapes the whole handler: **the status
code is chosen before the answer exists.** Once the first byte of a 200 is out,
a failure cannot become a 400 any more. So the handler pulls the first update
before it returns the response, which makes admission errors real HTTP errors and
leaves only mid-stream failures to be reported inside the stream.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import aclosing

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from nanoserve.cache import KVCacheExhausted
from nanoserve.config import ModelConfig
from nanoserve.engine import Engine
from nanoserve.loader import EMBED, LM_HEAD, Weights, expected_shapes
from nanoserve.model import LlamaModel
from nanoserve.server import create_app
from nanoserve.serving import AsyncEngine, EngineStopped, StreamUpdate

# --- the same tiny model and byte tokenizer the Day 37 tests use ----------------

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


def _engine(num_blocks=64, block_size=4, max_batch_size=4) -> Engine:
    return Engine.build(
        _model(), num_blocks=num_blocks, block_size=block_size, max_batch_size=max_batch_size
    )


def _offline(prompt: list[int], max_new_tokens: int, **kw) -> list[int]:
    """Generation only, which is what a completions stream shows."""
    full = _engine(**kw).generate([list(prompt)], max_new_tokens=max_new_tokens)[0]
    return full[len(prompt) :]


def run(coro, timeout: float = 20.0):
    async def guarded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(guarded())


async def _until(predicate, tries: int = 5000):
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("the condition never held")


async def _drain(stream) -> list[StreamUpdate]:
    return [update async for update in stream]



class _GatedEngine:
    """An engine that runs one step per ticket, so "when" is the test's decision.

    The claim streaming makes is about time: the first token is readable before
    the last one exists. A stopwatch cannot prove that without flaking, and
    counting `serving.steps` from the consumer races the loop. A gate can: hold
    the engine after the step that produced token one, look, then let it go.
    """

    def __init__(self, engine, tickets: int = 1):
        self.engine = engine
        self._tickets = threading.Semaphore(tickets)
        self._open = False

    def allow(self, n: int = 1) -> None:
        self._tickets.release(n)

    def open(self) -> None:
        self._open = True
        self._tickets.release(4096)

    def step(self):
        if not self._open:
            self._tickets.acquire()
        return self.engine.step()

    def add_request(self, request):
        return self.engine.add_request(request)

    def abort(self, request_id):
        return self.engine.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.engine.has_unfinished()

    @property
    def iterations(self) -> int:
        return self.engine.iterations

    @property
    def scheduler(self):
        return self.engine.scheduler


# --- 1. the answer is unchanged, and it arrives early ---------------------------


def test_a_stream_yields_the_offline_tokens_in_order():
    prompt = [1, 2, 3]
    expected = _offline(prompt, max_new_tokens=6)

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            return await _drain(serving.stream(prompt, max_new_tokens=6))

    updates = run(scenario())
    assert [u.token_id for u in updates[:-1]] == expected
    assert updates[-1].token_id is None


def test_the_last_update_is_final_and_carries_the_finished_request():
    """The stream's closing frame is where `finish_reason` and usage come from."""

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            return await _drain(serving.stream([1, 2, 3], max_new_tokens=4))

    updates = run(scenario())
    assert [u.is_final for u in updates] == [False] * 4 + [True]
    final = updates[-1]
    assert final.request.is_finished
    assert final.request.finish_reason == "length"
    assert final.request.num_output_tokens == 4
    # `token_id` is the snapshot; `request` is the live object, so it is only
    # safe to read the terminal fields off the final update. That is the deal.
    assert all(u.request is final.request for u in updates)


def test_a_stream_and_a_unary_call_agree_token_for_token():
    """The only thing that changed is when the caller is told."""
    prompt = [4, 5, 6]

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            unary = await serving.generate(prompt, max_new_tokens=5)
            streamed = await _drain(serving.stream(prompt, max_new_tokens=5))
            return unary.output_token_ids, [u.token_id for u in streamed[:-1]]

    unary, streamed = run(scenario())
    assert unary == streamed


def test_the_first_token_is_readable_before_the_request_finishes():
    """The whole reason this exists. If the first update only arrived at the end
    this would be `generate` with a more expensive interface.

    The gate holds the engine after the prefill, so the assertion is about the
    bridge's ordering rather than about whether this box was fast enough.
    """

    async def scenario():
        gated = _GatedEngine(_engine())
        async with AsyncEngine(gated) as serving:
            stream = serving.stream([1, 2, 3], max_new_tokens=8).__aiter__()
            first = await stream.__anext__()  # arrives on the one step allowed
            observed = (
                first.token_id is not None,
                first.request.is_finished,
                first.request.num_output_tokens,
                serving.num_active,
                serving.steps,
            )
            gated.open()
            rest = [u async for u in stream]
            return observed, len(rest)

    observed, remaining = run(scenario())
    assert observed == (True, False, 1, 1, 1)
    assert remaining == 8  # seven more tokens plus the final update


def test_one_step_yields_exactly_one_update():
    """A decode step emits one token, so a stream of it emits one update.

    Ticket by ticket, with the engine stopped in between, so an implementation
    that buffered and flushed in bursts would show up as a queue that is empty
    when it should have one thing in it.
    """

    async def scenario():
        gated = _GatedEngine(_engine())
        async with AsyncEngine(gated) as serving:
            stream = serving.stream([1, 2, 3], max_new_tokens=5).__aiter__()
            counts = []
            for _ in range(5):
                update = await stream.__anext__()
                counts.append((serving.steps, update.request.num_output_tokens))
                gated.allow()
            gated.open()
            return counts, [u.is_final async for u in stream]

    counts, tail = run(scenario())
    assert counts == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert tail == [True]


# --- 2. a stream is one row of a shared batch -----------------------------------


def test_three_concurrent_streams_each_match_their_solo_run():
    prompts = [[1, 2, 3], [5, 5], [7, 8, 9, 10]]
    expected = [_offline(p, max_new_tokens=5) for p in prompts]

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            streams = await asyncio.gather(
                *(_drain(serving.stream(p, max_new_tokens=5)) for p in prompts)
            )
            return [[u.token_id for u in s[:-1]] for s in streams]

    assert run(scenario()) == expected


def test_a_stream_and_a_unary_request_share_the_batch_correctly():
    prompts = [[1, 2, 3], [9, 9, 9]]
    expected = [_offline(p, max_new_tokens=5) for p in prompts]

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            streamed, unary = await asyncio.gather(
                _drain(serving.stream(prompts[0], max_new_tokens=5)),
                serving.generate(prompts[1], max_new_tokens=5),
            )
            return [u.token_id for u in streamed[:-1]], unary.output_token_ids

    assert list(run(scenario())) == expected


def test_a_slow_consumer_does_not_stall_the_loop():
    """There is no backpressure and there must not be.

    A queue with a bound would make one socket's reader the pace of a forward
    pass that three other requests are rows of. So the queue is unbounded, the
    engine runs to the request's budget whether or not anyone is reading, and the
    tokens wait in memory. This test reads nothing until the request has finished
    and then asserts that nothing was lost.
    """
    expected = _offline([1, 2, 3], max_new_tokens=6)

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            stream = serving.stream([1, 2, 3], max_new_tokens=6).__aiter__()
            first = await stream.__anext__()
            await _until(lambda: serving.parked)  # the engine ran to the end alone
            rest = [u async for u in stream]
            return [first.token_id] + [u.token_id for u in rest[:-1]]

    assert run(scenario()) == expected


def test_streams_and_unary_requests_can_be_interleaved_on_one_loop():
    """Six callers of two kinds, one engine, nobody's answer bleeds into anybody's."""
    prompts = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    expected = [_offline(p, max_new_tokens=4) for p in prompts]

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            jobs = []
            for i, prompt in enumerate(prompts):
                if i % 2:
                    jobs.append(serving.generate(prompt, max_new_tokens=4))
                else:
                    jobs.append(_drain(serving.stream(prompt, max_new_tokens=4)))
            done = await asyncio.gather(*jobs)
            return [
                r.output_token_ids if i % 2 else [u.token_id for u in r[:-1]]
                for i, r in enumerate(done)
            ]

    assert run(scenario()) == expected


# --- 3. ending badly ------------------------------------------------------------


def test_hanging_up_mid_stream_aborts_the_request_and_frees_its_blocks():
    """A closed socket must stop costing a slot, a row and a block.

    The unary path gets this from `CancelledError` at the await. A stream gets it
    from the generator being closed, which is a different event arriving at a
    different place, and it has to do the same thing.
    """

    async def scenario():
        async with AsyncEngine(_engine(num_blocks=16, block_size=4)) as serving:
            stream = serving.stream([1, 2, 3], max_new_tokens=40).__aiter__()
            await stream.__anext__()
            await stream.aclose()  # the client went away
            await _until(lambda: serving.num_active == 0)
            await _until(lambda: serving.parked)
            return serving.stats(), serving.engine.scheduler.allocator.num_free

    stats, free = run(scenario())
    assert stats["active"] == 0
    assert stats["running"] == 0
    assert free == 16  # every block back
    # The step count is the assertion that has teeth. A request that merely ran
    # out of budget would also end with an empty pool and zero active, forty
    # forwards later; this one stopped because nobody was listening.
    assert stats["steps"] < 10


def test_breaking_out_of_the_async_for_also_aborts():
    """`break` is how a real handler stops reading, so it has to work too."""

    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            async with aclosing(serving.stream([1, 2, 3], max_new_tokens=40)) as stream:
                async for _ in stream:
                    break
            await _until(lambda: serving.num_active == 0)
            await _until(lambda: serving.parked)
            return serving.stats()

    stats = run(scenario())
    assert stats["active"] == 0
    assert stats["steps"] < 10


def test_abort_by_id_ends_the_stream_with_an_abort_reason():
    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            stream = serving.stream(
                [1, 2, 3], max_new_tokens=40, request_id="r1"
            ).__aiter__()
            await stream.__anext__()
            serving.abort("r1")
            rest = [u async for u in stream]
            return rest

    rest = run(scenario())
    assert rest[-1].is_final
    assert rest[-1].request.finish_reason == "abort"


def test_a_prompt_too_large_for_the_pool_raises_on_the_first_read():
    """Admission is refused inside the shared loop and has to reach this caller.

    Raising at the first `__anext__` rather than mid-stream is what lets the HTTP
    layer still choose a status code, because no byte of the body has gone out.
    """

    async def scenario():
        async with AsyncEngine(_engine(num_blocks=4, block_size=4)) as serving:
            stream = serving.stream([1, 2, 3], max_new_tokens=100).__aiter__()
            with pytest.raises(KVCacheExhausted):
                await stream.__anext__()
            return serving.num_active

    assert run(scenario()) == 0


def test_a_step_that_raises_ends_every_stream_rather_than_hanging_it():
    class _Exploding:
        def __init__(self, engine):
            self.engine = engine
            self.n = 0

        def step(self):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("the forward blew up")
            return self.engine.step()

        def add_request(self, request):
            return self.engine.add_request(request)

        def abort(self, request_id):
            return self.engine.abort(request_id)

        def has_unfinished(self):
            return self.engine.has_unfinished()

        @property
        def iterations(self):
            return self.engine.iterations

        @property
        def scheduler(self):
            return self.engine.scheduler

    async def scenario():
        async with AsyncEngine(_Exploding(_engine())) as serving:
            with pytest.raises(RuntimeError, match="the forward blew up"):
                await _drain(serving.stream([1, 2, 3], max_new_tokens=20))

    run(scenario())


def test_stopping_the_engine_mid_stream_raises_engine_stopped():
    async def scenario():
        serving = AsyncEngine(_engine())
        await serving.start()
        stream = serving.stream([1, 2, 3], max_new_tokens=40).__aiter__()
        await stream.__anext__()
        await serving.stop()
        with pytest.raises(EngineStopped):
            async for _ in stream:
                pass

    run(scenario())


def test_streaming_on_a_stopped_engine_refuses_at_the_first_read():
    async def scenario():
        serving = AsyncEngine(_engine())
        stream = serving.stream([1, 2, 3]).__aiter__()
        with pytest.raises(EngineStopped):
            await stream.__anext__()

    run(scenario())


def test_a_duplicate_request_id_is_refused_before_the_engine_sees_it():
    async def scenario():
        async with AsyncEngine(_engine()) as serving:
            first = serving.stream([1, 2, 3], max_new_tokens=20, request_id="dup")
            await first.__aiter__().__anext__()
            with pytest.raises(ValueError, match="already in flight"):
                await serving.stream(
                    [1, 2, 3], max_new_tokens=4, request_id="dup"
                ).__aiter__().__anext__()
            await first.aclose()

    run(scenario())


# --- the wire: SSE --------------------------------------------------------------


def _app(serving: AsyncEngine, **kw):
    return create_app(serving, TOKENIZER, vocab_size=64, **kw)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://nanoserve"
    )


def _frames(body: str) -> list[str]:
    """The `data:` payloads of an SSE body, in order, including the sentinel."""
    return [
        line[len("data: ") :]
        for line in body.split("\n\n")
        if line.startswith("data: ")
    ]


def _chunks(body: str) -> list[dict]:
    return [json.loads(f) for f in _frames(body) if f != "[DONE]"]


def _stream_body(app, payload: dict) -> tuple[int, str, str]:
    """POST and read the whole SSE response. Returns (status, content-type, body)."""
    with TestClient(app) as client:
        response = client.post("/v1/completions", json=payload)
        return response.status_code, response.headers.get("content-type", ""), response.text


def test_stream_true_returns_an_event_stream():
    serving = AsyncEngine(_engine())
    status, content_type, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abc", "max_tokens": 5, "stream": True},
    )
    assert status == 200
    assert content_type.startswith("text/event-stream")
    assert body.endswith("data: [DONE]\n\n")


def test_the_streamed_text_equals_the_unary_text_for_the_same_prompt():
    """The claim that makes streaming safe to turn on: same answer, drip-fed."""
    payload = {"model": "nanoserve", "prompt": "abc", "max_tokens": 6}
    with TestClient(_app(AsyncEngine(_engine()))) as client:
        unary = client.post("/v1/completions", json=payload).json()
    _, _, body = _stream_body(_app(AsyncEngine(_engine())), {**payload, "stream": True})
    streamed = "".join(c["choices"][0]["text"] for c in _chunks(body))
    assert streamed == unary["choices"][0]["text"]


def test_every_chunk_has_the_openai_shape():
    serving = AsyncEngine(_engine())
    _, _, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abc", "max_tokens": 4, "stream": True},
    )
    chunks = _chunks(body)
    assert len(chunks) == 5  # four tokens plus the closing chunk
    for chunk in chunks:
        assert chunk["object"] == "text_completion"
        assert chunk["model"] == "nanoserve"
        assert chunk["id"].startswith("cmpl-")
        assert isinstance(chunk["created"], int)
        assert chunk["choices"][0]["index"] == 0
    assert all(c["id"] == chunks[0]["id"] for c in chunks)


def test_finish_reason_is_null_until_the_last_chunk():
    """A client watching `finish_reason` must see it exactly once, at the end."""
    serving = AsyncEngine(_engine())
    _, _, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abc", "max_tokens": 4, "stream": True},
    )
    reasons = [c["choices"][0]["finish_reason"] for c in _chunks(body)]
    assert reasons[:-1] == [None] * 4
    assert reasons[-1] == "length"


def test_the_last_chunk_carries_usage_and_no_text():
    serving = AsyncEngine(_engine())
    _, _, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abcd", "max_tokens": 3, "stream": True},
    )
    last = _chunks(body)[-1]
    assert last["choices"][0]["text"] == ""
    assert last["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
    assert all(c.get("usage") is None for c in _chunks(body)[:-1])


def test_an_admission_refusal_before_the_first_byte_is_a_400_not_a_200():
    """The status code is chosen before the answer exists, so the handler pulls
    the first update before it commits to one."""
    serving = AsyncEngine(_engine(num_blocks=4, block_size=4))
    status, content_type, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abc", "max_tokens": 100, "stream": True},
    )
    assert status == 400
    assert "event-stream" not in content_type
    assert "data:" not in body


def test_a_bad_model_is_still_a_404_when_streaming():
    serving = AsyncEngine(_engine())
    status, _, _ = _stream_body(
        _app(serving),
        {"model": "gpt-4", "prompt": "abc", "stream": True},
    )
    assert status == 404


def test_temperature_is_still_refused_when_streaming():
    """Streaming is a delivery schedule. It does not make a lie about sampling
    into the truth."""
    serving = AsyncEngine(_engine())
    status, _, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abc", "temperature": 0.7, "stream": True},
    )
    assert status == 400
    assert "temperature" in body


def test_a_failure_after_the_first_chunk_becomes_an_error_frame():
    """A 200 cannot be taken back, so a mid-stream failure is reported in-band.

    The alternative is a truncated body that looks exactly like a complete one,
    which is the worst outcome available: a caller that silently believes the
    model stopped there.
    """

    class _Exploding:
        def __init__(self, engine):
            self.engine = engine
            self.n = 0

        def step(self):
            self.n += 1
            if self.n == 3:
                raise RuntimeError("the forward blew up")
            return self.engine.step()

        def add_request(self, request):
            return self.engine.add_request(request)

        def abort(self, request_id):
            return self.engine.abort(request_id)

        def has_unfinished(self):
            return self.engine.has_unfinished()

        @property
        def iterations(self):
            return self.engine.iterations

        @property
        def scheduler(self):
            return self.engine.scheduler

    serving = AsyncEngine(_Exploding(_engine()))
    status, _, body = _stream_body(
        _app(serving),
        {"model": "nanoserve", "prompt": "abc", "max_tokens": 20, "stream": True},
    )
    assert status == 200
    frames = _frames(body)
    assert frames[-1] == "[DONE]"
    error = json.loads(frames[-2])["error"]
    assert "blew up" in error["message"]


def test_the_stop_token_is_counted_but_not_printed_in_a_stream():
    """The same split `/v1/completions` makes: charged for, not shown.

    The engine really computed it and the cache really holds it, so hiding it
    from usage would misreport the work; printing it would put a control token in
    the caller's text.
    """
    engine = _engine()
    prompt = [1, 2, 3]
    ids = _offline(prompt, max_new_tokens=6)
    serving = AsyncEngine(engine)
    app = _app(serving, eos_token_id=ids[2])
    _, _, body = _stream_body(
        app,
        {"model": "nanoserve", "prompt": TOKENIZER.decode(prompt), "max_tokens": 6,
         "stream": True},
    )
    chunks = _chunks(body)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 3
    assert "".join(c["choices"][0]["text"] for c in chunks) == TOKENIZER.decode(ids[:2])


def test_unary_completions_are_unchanged_by_all_of_this():
    """The regression that matters: Day 37's endpoint still behaves exactly so."""
    serving = AsyncEngine(_engine())
    with TestClient(_app(serving)) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "nanoserve", "prompt": "abc", "max_tokens": 5},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["text"] == TOKENIZER.decode(
        _offline([0, 1, 2], max_new_tokens=5)
    )


def test_two_streams_over_one_app_do_not_interleave_their_text():
    """Two sockets, one engine loop, one batch. Each body is its own answer."""
    serving = AsyncEngine(_engine())
    app = _app(serving)

    async def scenario():
        async with _client(app) as client:
            async with app.router.lifespan_context(app):
                bodies = await asyncio.gather(
                    *(
                        client.post(
                            "/v1/completions",
                            json={
                                "model": "nanoserve",
                                "prompt": prompt,
                                "max_tokens": 5,
                                "stream": True,
                            },
                        )
                        for prompt in ("abc", "xyz")
                    )
                )
                return [r.text for r in bodies]

    bodies = run(scenario())
    for body, prompt in zip(bodies, ("abc", "xyz")):
        text = "".join(c["choices"][0]["text"] for c in _chunks(body))
        assert text == TOKENIZER.decode(
            _offline(TOKENIZER.encode(prompt), max_new_tokens=5)
        )


async def _hang_up_after_one_chunk(app, payload: dict) -> list[bytes]:
    """Drive the ASGI app by hand and disconnect once a chunk has been sent.

    `httpx`'s ASGI transport runs the app to completion before it hands back a
    response, so it cannot express a client that stops reading: every test
    written through it would pass whether or not the disconnect path exists.
    Driving the callable directly can, because a hang-up is not a socket
    mystery, it is one message: `http.disconnect` on `receive`.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [(b"content-type", b"application/json"), (b"host", b"nanoserve")],
        "server": ("nanoserve", 80),
        "client": ("127.0.0.1", 5555),
    }
    body = json.dumps(payload).encode()
    sent_request = False
    gone = asyncio.Event()
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            chunks.append(message["body"])
            gone.set()  # the reader walks away with one chunk in hand

    await app(scope, receive, send)
    return chunks


def test_a_client_that_hangs_up_mid_body_frees_the_slot_and_the_blocks():
    """The disconnect path, end to end, through the real response object.

    A stopped reader is not a stopped request: without the abort the row keeps
    generating to its budget, holding a slot and blocks that nothing will ever
    collect, and a server under flaky clients leaks its whole pool one hang-up at
    a time. Day 37 got this from `CancelledError` at an await. A stream gets it
    from the response generator being closed when Starlette cancels it, which is
    a different event, at a different place, doing the same job.
    """
    serving = AsyncEngine(_engine(num_blocks=16, block_size=4))
    app = _app(serving)

    async def scenario():
        async with app.router.lifespan_context(app):
            chunks = await _hang_up_after_one_chunk(
                app,
                {
                    "model": "nanoserve",
                    "prompt": "abc",
                    "max_tokens": 40,
                    "stream": True,
                },
            )
            await _until(lambda: serving.num_active == 0)
            await _until(lambda: serving.parked)
            return chunks, serving.stats(), serving.engine.scheduler.allocator.num_free

    chunks, stats, free = run(scenario())
    assert chunks and chunks[0].startswith(b"data: ")
    assert stats["active"] == 0
    assert stats["running"] == 0
    assert free == 16
    assert stats["steps"] < 10  # stopped on the hang-up, not on the budget


def test_a_multi_token_character_is_not_split_across_chunks():
    """The detokenizer, wired. A byte-level tokenizer whose ids are bytes makes
    this the same pathology Llama-3 has with emoji."""

    class Utf8Tokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

        def decode(self, token_ids) -> str:
            return bytes(token_ids).decode("utf-8", errors="replace")

    serving = AsyncEngine(_engine())
    app = create_app(serving, Utf8Tokenizer(), vocab_size=64)
    _, _, body = _stream_body(
        app, {"model": "nanoserve", "prompt": "abc", "max_tokens": 6, "stream": True}
    )
    texts = [c["choices"][0]["text"] for c in _chunks(body)]
    assert "�" not in "".join(texts)
