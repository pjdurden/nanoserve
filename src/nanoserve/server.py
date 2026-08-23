"""The HTTP surface: OpenAI-compatible /v1/completions over the bridge. Weeks 10-11.

`serving.py` is the hard half of this phase and it is finished by the time this
file starts: there is a loop, requests go in on an inbox, and each caller has a
future that resolves at its own last token. What is left is translation. The wire
speaks JSON and a schema somebody else designed; the engine speaks token ids and
`Request`. This file is the adapter, and it is deliberately thin, because
everything interesting about serving an LLM happens on the other side of it.

Two decisions are worth more than the code.

**Compatible means the fields that change the answer, not the fields that fit.**
It is easy to accept OpenAI's whole schema and honour the third of it that is
implemented, and it produces a server that looks compatible and lies. Day 37
wrote the rule with `temperature=0.7` as its example: an engine that only knows
`argmax` cannot serve it, and handing back greedy tokens with a 200 is a wrong
answer dressed as a right one that nothing downstream can tell. So every
parameter here is either honoured or refused with a 400 that names it and says
when it lands. Two of those refusals are now features. Day 39 served
`stream=true`; Day 40 serves `temperature`, `top_k`, `top_p` and `seed`, which
leaves `n` as the last of them. A 400 is a bug report the caller can act on; a
silently ignored parameter is one they will find in production.

**Serving a parameter and policing its value are the same job.** `top_p=1.5` is
not a nucleus, and `temperature=-1` is a distribution turned inside out. Those
are refused here, by constructing the request's `SamplingParams` in the handler
and turning its `ValueError` into a 400 that carries the field name. Building it
in the handler rather than on the loop thread is what makes the refusal cheap and
attributable: it costs its own caller a status code before the engine has spent
an iteration on it, and no other row of the batch ever hears about it.

**The status code is a claim about whose fault it is.** A prompt that carries a
token id the model does not have is a 400, because without the check it is an
`IndexError` in the embedding lookup and therefore a 500, which tells the caller
the server broke when the caller did. A generation budget too large for the whole
KV pool is a 400 for the same reason, and that one is interesting because the
refusal is made by `Scheduler.add_request` deep inside a loop shared with every
other request, arrives back here on one future, and has to become one caller's
status code without any of the others noticing. An unknown model is a 404. The
loop having stopped is a 503, which is the only one of these that is the server's
fault and says so.

**And a stream has to choose its status code before it has an answer.** Day 39.
A response's status line goes out with its first byte, so the moment the first
SSE frame is written the server has committed to a 200 and every later failure
has to be reported inside a body that already claims success. The handler
therefore pulls the *first* update from the bridge before it constructs the
response, which is free (the first update is the first token) and which is what
keeps "this prompt can never fit the pool" a 400 instead of an error frame. What
is left after that (a step that raises, a loop that dies) becomes an error frame
followed by `[DONE]`, because the alternative is a body that simply stops and is
indistinguishable from a complete one.

**A unary caller who leaves is only heard from if somebody listens.** Day 41.
Day 37 wrote the disconnect path as `AsyncEngine.generate` aborting inside its own
`except CancelledError`, and tested it through `httpx.ASGITransport`, where the
client cancelling a request cancels the handler's coroutine directly. Over a real
socket nothing does that. uvicorn turns a closed connection into an
`http.disconnect` *message*, delivered on `receive`, and a handler parked on
`await serving.generate(...)` never calls `receive`, so it never finds out: the
request keeps its slot, keeps its blocks, and spends a row of every forward until
`max_tokens` runs out, for a caller who left. The acceptance run measured it at
800 tokens generated for a client that hung up after two. So the handler races the
generation against a watcher on `receive`, and a caller who leaves cancels their
own request. Streaming was never affected, because Starlette watches for the
disconnect itself while a response body is being produced, and that difference is
the whole reason this bug could live under a green test suite.

`prompt` accepts a string or a list of token ids, both of which the OpenAI schema
allows, and the id form is what lets a caller drive this server without agreeing
with it about a tokenizer. Everything else in the response is the standard shape:
one choice, a finish reason, a usage block. Streamed text comes from
`detokenizer.py` rather than from `decode` of one id at a time, which is a
correctness requirement and not a nicety: on a byte-level tokenizer the naive
version prints replacement characters where the unary endpoint prints emoji.

Not here yet, on purpose: `/v1/chat/completions` and its template, `logprobs`,
`stop` strings, multiple prompts per request, and any authentication at all. This is a localhost server for a single-GPU engine, and pretending otherwise
would be its own kind of lie.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Response
from fastapi import Request as HTTPRequest  # not `scheduler.Request`, which is the engine's
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .cache import KVCacheExhausted
from .detokenizer import IncrementalDetokenizer
from .sampling import SamplingParams
from .scheduler import Request
from .serving import AsyncEngine, EngineStopped, StreamUpdate


class Tokenizer(Protocol):
    """All this server asks of a tokenizer, which is why the tests can fake it."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


# nginx's code for a connection the caller closed before a response was sent. Not
# in the IANA registry and universally understood, which is the useful kind of
# non-standard: it puts "this caller left" in the access log without claiming a
# success or blaming the server.
CLIENT_CLOSED_REQUEST = 499


class ClientGone(Exception):
    """The caller's connection closed before their answer was ready. Day 41.

    Not an error condition of the server and not of the request: it is the normal
    end of a request nobody is waiting for any more. It exists as an exception
    rather than a sentinel return so that the cancellation and the reporting are
    the same control flow, and so a handler cannot accidentally treat "the client
    left" as "here is the answer".
    """


async def await_client_disconnect(receive) -> None:
    """Resolve when the ASGI connection reports the client has gone. Day 41.

    The loop is the whole subtlety. uvicorn's `receive` returns `http.request` for
    anything that arrives on the socket, including the tail of a body and a
    pipelined follow-up on a keep-alive connection, and it blocks on an event that
    is set either by new data or by `connection_lost`. A watcher that resolved on
    the first message it saw would call every ordinary client gone and abort their
    request mid-generation, so only `http.disconnect` counts.
    """
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return


async def run_until_client_leaves(work: Awaitable, receive) -> Any:
    """Await `work`, cancelling it if the caller disconnects first. Day 41.

    The generation and the disconnect watcher are two tasks and the first one to
    finish decides. If it is the work, its result (or its exception, which is how a
    `KVCacheExhausted` stays a 400) comes straight back. If it is the watcher, the
    work is cancelled and `ClientGone` is raised.

    Two details are load-bearing.

    **The cancelled task is awaited, not merely cancelled.** `AsyncEngine.generate`
    frees its request inside `except CancelledError`, and that only runs when the
    cancellation is actually delivered. Returning without awaiting would leave the
    handler finished and the request still running: the same leak with a shorter
    window and a much harder test.

    **The `finally` covers the handler being cancelled too.** Starlette cancels
    handlers on shutdown, and a watcher left polling `receive` on a connection that
    is going away is a task that outlives its request.
    """
    generation = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(await_client_disconnect(receive))
    try:
        await asyncio.wait({generation, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if generation.done():
            return generation.result()
        raise ClientGone("the client disconnected before its answer was ready")
    finally:
        for task in (generation, watcher):
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task


# --- the wire schema ------------------------------------------------------------


class CompletionRequest(BaseModel):
    """The subset of OpenAI's /v1/completions body this engine can mean.

    Fields that are declared but unimplemented are here so they can be *refused*
    by name. Leaving them undeclared would let them through as ignored extras,
    which is the failure this file exists to avoid.
    """

    model: str
    prompt: str | list[int]
    max_tokens: int = Field(default=16, ge=1)
    stream: bool = False
    # Day 40. `temperature` defaults to 0, which is greedy, which is what every
    # client of this server got before today: a body that names none of these
    # fields keeps the answer it has always had. `top_k` is not in OpenAI's schema
    # and is in vLLM's, and it is here because the sampler has had it since Week 6.
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    n: int = 1


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Any = None
    finish_reason: str


class CompletionUsage(BaseModel):
    """Tokens the caller is charged for, which is not the same as tokens shown.

    A generation that stopped on EOS counts that token here and does not print it:
    the model really computed it and the cache really holds it, so hiding it from
    the bill would misreport the work. `Request.append_token` makes the same split
    and leaves the display decision to this layer.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class CompletionChunkChoice(BaseModel):
    """A choice in a streamed chunk. `text` is a delta, not the whole answer.

    `finish_reason` is null on every chunk but the last, which is how a client
    knows the stream ended because the model stopped rather than because the
    connection did. A truncated body and a complete one look identical otherwise,
    which is why this field carries more weight than its size suggests.
    """

    text: str
    index: int = 0
    logprobs: Any = None
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    """One SSE frame's payload. Same `object` and `id` as the unary response.

    Every chunk in a stream repeats the id, so a client multiplexing several
    streams over one log can tell them apart, and `usage` rides only on the last
    one, which is what OpenAI's `stream_options={"include_usage": true}` gets you.
    """

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]
    usage: CompletionUsage | None = None


# --- the app --------------------------------------------------------------------


def create_app(
    serving: AsyncEngine,
    tokenizer: Tokenizer,
    *,
    model_name: str = "nanoserve",
    eos_token_id: int | None = None,
    vocab_size: int | None = None,
    info: dict | None = None,
) -> FastAPI:
    """Build the ASGI app around one already-constructed bridge.

    serving:      the `AsyncEngine`. Injected rather than built here so a test can
                  hold the same object the app is holding and ask it what the loop
                  did, and so one process could serve two apps over one engine.
    tokenizer:    anything with `encode`/`decode`.
    eos_token_id: server configuration, not a request parameter. The stop token is
                  a property of the model the server loaded, and letting a caller
                  choose it is how you get a request that never stops.
    vocab_size:   when known, token ids outside it are refused at the door.
    info:         static facts about this process, merged into `/health`. Day 38's
                  launcher puts the pool it chose here, because the block count is
                  the one number nothing in the code can be read off and the thing
                  you want when a server is preempting more than you expected.

    The lifespan owns `start`/`stop`, because under uvicorn nobody else can: the
    loop has to exist before the first request and has to be told to leave when
    the server drains, and a background task that outlives its app is a hang.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await serving.start()
        try:
            yield
        finally:
            await serving.stop()

    app = FastAPI(title="nanoserve", lifespan=lifespan)
    info = dict(info or {})

    def _prompt_ids(prompt: str | list[int]) -> list[int]:
        """Token ids for either prompt form, or a 400 explaining which rule broke."""
        ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise HTTPException(status_code=400, detail="prompt is empty")
        if vocab_size is not None and any(not 0 <= i < vocab_size for i in ids):
            raise HTTPException(
                status_code=400,
                detail=f"prompt has a token id outside the model's vocab of {vocab_size}",
            )
        return ids

    def _check(body: CompletionRequest) -> None:
        """Refuse everything this engine cannot honestly do, by name."""
        if body.model != model_name:
            raise HTTPException(
                status_code=404, detail=f"model {body.model!r} is not served here"
            )
        if body.n != 1:
            raise HTTPException(status_code=400, detail="n must be 1")

    def _sampling(body: CompletionRequest) -> SamplingParams:
        """The request's sampling parameters, or a 400 naming the one that is wrong.

        The validation lives in `SamplingParams.__post_init__`, which is where the
        rules are, and this turns its `ValueError` into a status code. Doing it
        here and not on the loop thread is what makes the refusal cheap and
        attributable: a bad `top_p` costs its own caller a 400 before the engine
        has spent an iteration on it, and no other request in the batch ever hears
        about it. The message carries the field name because a 400 a caller cannot
        act on is barely better than a 500.
        """
        try:
            return SamplingParams(
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                seed=body.seed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(body: CompletionRequest, http_request: HTTPRequest) -> Any:
        """One request, one answer, delivered when *this* request finishes.

        The `await` here is the whole serving layer in one line: the handler is
        suspended, the loop keeps stepping a batch this request is one row of, and
        the coroutine is resumed with its own tokens.

        `http_request` is here for one reason and it is Day 41's: it carries the
        `receive` this handler has to listen on to find out that its caller left.
        Nothing cancels an ASGI handler when a connection drops; the disconnect is
        a message, and a coroutine parked on a future is not reading messages. So
        the generation is raced against a watcher, and a client that hangs up
        cancels its own request, which is what stops a dead socket from holding a
        slot and its blocks until `max_tokens` runs out.

        The status code for that is 499, nginx's "client closed request", written
        to a socket nobody is reading. It exists for the access log, which is the
        only place it can ever be seen, and it is a real number rather than a 200
        because a log line saying this request succeeded would be false.

        `stream=true` takes the other road and returns a `StreamingResponse`,
        which FastAPI hands back untouched, so `response_model` describes the
        unary body only.
        """
        _check(body)
        sampling = _sampling(body)
        prompt_ids = _prompt_ids(body.prompt)
        if body.stream:
            return await _stream_completion(body, prompt_ids, sampling)
        try:
            request = await run_until_client_leaves(
                serving.generate(
                    prompt_ids,
                    max_new_tokens=body.max_tokens,
                    eos_token_id=eos_token_id,
                    sampling=sampling,
                ),
                http_request.receive,
            )
        except ClientGone:
            return Response(status_code=CLIENT_CLOSED_REQUEST)
        except KVCacheExhausted as exc:
            # The caller asked for more than the pool can ever hold. Their error,
            # decided inside a loop that a dozen other requests are sharing.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EngineStopped as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _as_response(request, model_name, tokenizer, eos_token_id)

    async def _stream_completion(
        body: CompletionRequest, prompt_ids: list[int], sampling: SamplingParams
    ) -> Response:
        """Turn one bridge stream into an SSE body, after choosing the status code.

        The first update is pulled *here*, before the response exists, and that is
        the entire reason this function is not just a `StreamingResponse` wrapped
        around the generator. A response's status line goes out with its first
        byte, so anything that can still be an error has to happen before it: a
        prompt the pool can never hold is refused by `Scheduler.add_request` deep
        inside the shared loop, and arriving on the first read is what lets it
        come back as a 400 rather than as an error buried in a 200.

        It costs nothing in latency. The first update *is* the first token, so the
        headers leave at the instant the caller had something to read anyway.
        """
        stream = serving.stream(
            prompt_ids,
            max_new_tokens=body.max_tokens,
            eos_token_id=eos_token_id,
            sampling=sampling,
        )
        updates = stream.__aiter__()
        try:
            first = await updates.__anext__()
        except KVCacheExhausted as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EngineStopped as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return StreamingResponse(
            _frames(first, updates, prompt_ids),
            media_type="text/event-stream",
            headers={
                # Anything that buffers this response destroys the product. Both
                # headers are addressed to middleboxes, not to the client.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _frames(
        first: StreamUpdate,
        updates: AsyncIterator[StreamUpdate],
        prompt_ids: list[int],
    ) -> AsyncIterator[str]:
        """The SSE body: one `data:` frame per readable delta, then `[DONE]`.

        Three things here are not obvious.

        **Text comes from an `IncrementalDetokenizer`, not from `decode` of one
        id.** Llama-3's tokenizer is byte-level, so a token can be half a
        character, and decoding it alone prints a replacement character that the
        unary endpoint would never print for the same ids. A frame with no text is
        not emitted at all, which is what a held-back partial character looks like
        from out here.

        **A failure after the first byte cannot be a status code any more.** The
        200 is already sent, so a loop that dies mid-stream is reported as an
        error frame and then `[DONE]`. The alternative is a body that simply
        stops, which is indistinguishable from a complete one and leaves the
        caller believing the model chose to end there.

        **The `finally` closes the bridge stream.** A client that hangs up
        cancels this generator, and the close is what propagates that into an
        abort, freeing the slot and the blocks instead of leaving a request
        generating into a queue nobody will read.
        """
        detokenizer = IncrementalDetokenizer(tokenizer, prompt_ids)
        completion_id = f"cmpl-{first.request.request_id}"
        created = int(time.time())
        failure: BaseException | None = None
        try:
            update = first
            while not update.is_final:
                # The stop token is counted and not shown, the same split the
                # unary response makes.
                if update.token_id != eos_token_id:
                    text = detokenizer.append(update.token_id)
                    if text:
                        yield _sse(_chunk(completion_id, created, model_name, text))
                update = await updates.__anext__()
            yield _sse(
                _chunk(
                    completion_id,
                    created,
                    model_name,
                    detokenizer.flush(),
                    finish_reason=update.request.finish_reason or "length",
                    usage=_usage(update.request),
                )
            )
        except Exception as exc:  # noqa: BLE001 - the status code is already spent
            failure = exc
        finally:
            await updates.aclose()
        if failure is not None:
            yield _sse(
                {"error": {"message": str(failure), "type": type(failure).__name__}}
            )
        yield "data: [DONE]\n\n"

    @app.get("/health")
    async def health() -> dict:
        """Liveness plus what the loop is doing, which is what you want at 3am."""
        return {"status": "ok", "model": model_name, **info, **serving.stats()}

    return app


def _sse(payload: dict) -> str:
    """One server-sent event. The blank line is the frame delimiter, not padding.

    SSE is text: `data: <line>` and an empty line to end the event. A client that
    reads on `\n\n` and gets a frame without it will hold the whole message
    waiting for a delimiter that already went past, which presents as a stream
    that is one chunk behind forever.
    """
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chunk(
    completion_id: str,
    created: int,
    model_name: str,
    text: str,
    finish_reason: str | None = None,
    usage: CompletionUsage | None = None,
) -> dict:
    """One streamed chunk, as the dict that goes on the wire."""
    return CompletionChunk(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[CompletionChunkChoice(text=text, finish_reason=finish_reason)],
        usage=usage,
    ).model_dump()


def _usage(request: Request) -> CompletionUsage:
    """The bill, identical for a stream and for the unary answer of the same ids."""
    return CompletionUsage(
        prompt_tokens=request.num_prompt_tokens,
        completion_tokens=request.num_output_tokens,
        total_tokens=request.num_prompt_tokens + request.num_output_tokens,
    )


def _as_response(
    request: Request,
    model_name: str,
    tokenizer: Tokenizer,
    eos_token_id: int | None,
) -> CompletionResponse:
    """Turn a finished `Request` into the completion body.

    The one judgement call is the trailing stop token: kept in the count, dropped
    from the text. `finish_reason` passes through as the scheduler wrote it, so
    "stop" and "length" are OpenAI's own vocabulary and "abort" is the honest
    third case, reachable only when something stopped the request while it ran.
    """
    output_ids = list(request.output_token_ids)
    shown = output_ids
    if eos_token_id is not None and shown and shown[-1] == eos_token_id:
        shown = shown[:-1]
    return CompletionResponse(
        id=f"cmpl-{request.request_id}",
        created=int(time.time()),
        model=model_name,
        choices=[
            CompletionChoice(
                text=tokenizer.decode(shown),
                finish_reason=request.finish_reason or "length",
            )
        ],
        usage=_usage(request),
    )
