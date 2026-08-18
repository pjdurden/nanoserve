"""The HTTP surface: OpenAI-compatible /v1/completions over the bridge. Week 10.

`serving.py` is the hard half of this phase and it is finished by the time this
file starts: there is a loop, requests go in on an inbox, and each caller has a
future that resolves at its own last token. What is left is translation. The wire
speaks JSON and a schema somebody else designed; the engine speaks token ids and
`Request`. This file is the adapter, and it is deliberately thin, because
everything interesting about serving an LLM happens on the other side of it.

Two decisions are worth more than the code.

**Compatible means the fields that change the answer, not the fields that fit.**
It is easy to accept OpenAI's whole schema and honour the third of it that is
implemented, and it produces a server that looks compatible and lies. The engine
samples with `argmax`; a request that asks for `temperature=0.7` and gets greedy
tokens back with a 200 has been given a wrong answer dressed as a right one, and
nothing downstream can tell. So every parameter here is either honoured or
refused with a 400 that names it and says when it lands. Right now that is
`stream`, `temperature` and `n`, and Week 11 turns two of those refusals into
features. A 400 is a bug report the caller can act on; a silently ignored
parameter is one they will find in production.

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

`prompt` accepts a string or a list of token ids, both of which the OpenAI schema
allows, and the id form is what lets a caller drive this server without agreeing
with it about a tokenizer. Everything else in the response is the standard shape:
one choice, a finish reason, a usage block.

Not here yet, on purpose: `stream=true` and SSE (Week 11, and the bridge is
already the right shape for it, one future becomes one queue), sampling
parameters, `/v1/chat/completions` and its template, `logprobs`, multiple prompts
per request, and any authentication at all. This is a localhost server for a
single-GPU engine, and pretending otherwise would be its own kind of lie.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cache import KVCacheExhausted
from .scheduler import Request
from .serving import AsyncEngine, EngineStopped


class Tokenizer(Protocol):
    """All this server asks of a tokenizer, which is why the tests can fake it."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


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
    temperature: float = 0.0
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


# --- the app --------------------------------------------------------------------


def create_app(
    serving: AsyncEngine,
    tokenizer: Tokenizer,
    *,
    model_name: str = "nanoserve",
    eos_token_id: int | None = None,
    vocab_size: int | None = None,
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
        if body.stream:
            raise HTTPException(
                status_code=400,
                detail="stream=true is not implemented yet; it lands in Week 11",
            )
        if body.temperature != 0.0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "temperature is not implemented yet: this engine decodes "
                    "greedily, so serving a non-zero temperature would return "
                    "argmax tokens under a name that promises otherwise"
                ),
            )
        if body.n != 1:
            raise HTTPException(status_code=400, detail="n must be 1")

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(body: CompletionRequest) -> CompletionResponse:
        """One request, one answer, delivered when *this* request finishes.

        The `await` here is the whole serving layer in one line: the handler is
        suspended, the loop keeps stepping a batch this request is one row of, and
        the coroutine is resumed with its own tokens. If the client hangs up
        first, this coroutine is cancelled and `AsyncEngine.generate` aborts the
        request on the way out, which is what stops a dead socket from holding a
        slot until its budget runs out.
        """
        _check(body)
        prompt_ids = _prompt_ids(body.prompt)
        try:
            request = await serving.generate(
                prompt_ids, max_new_tokens=body.max_tokens, eos_token_id=eos_token_id
            )
        except KVCacheExhausted as exc:
            # The caller asked for more than the pool can ever hold. Their error,
            # decided inside a loop that a dozen other requests are sharing.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EngineStopped as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _as_response(request, model_name, tokenizer, eos_token_id)

    @app.get("/health")
    async def health() -> dict:
        """Liveness plus what the loop is doing, which is what you want at 3am."""
        return {"status": "ok", "model": model_name, **serving.stats()}

    return app


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
        usage=CompletionUsage(
            prompt_tokens=request.num_prompt_tokens,
            completion_tokens=len(output_ids),
            total_tokens=request.num_prompt_tokens + len(output_ids),
        ),
    )
