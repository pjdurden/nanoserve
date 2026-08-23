"""The Week 11 acceptance test: a real socket, a crowd, and four claims. Day 41.

Phase 4 built the serving layer one guarantee at a time. Day 37 gave each caller a
future that resolves with that caller's tokens. Day 39 turned the future into a
queue and the queue into SSE. Day 40 gave each request its own generator so a seed
means the same thing in a batch as it does alone. Every one of those days tested
its guarantee in isolation, through `httpx.ASGITransport`, which calls the app as a
Python coroutine.

That transport is a very good lie. There is no TCP, no HTTP parser, no chunked
encoding, and above all **no client that can vanish**: a "disconnect" there is a
generator being closed by the same event loop the server is running on, three
layers below the socket where a real one happens. Day 39 already found one bug that
only exists on the far side of that boundary, and an acceptance test that ran
through the same transport would be testing the same fiction the unit tests test.

So this module runs uvicorn. A real server, on a real port the kernel chose, with
real clients that really hang up. What it asserts is the phase's whole contract in
four claims, in the order they fail quietly:

  1. **No request receives another request's tokens.** Checked in the strongest
     form available: each client's answer *in the crowd* is byte-identical to the
     answer the same server gave that same request when it ran *alone*. That covers
     crosstalk, mis-indexed rows, a cache row addressing a neighbour's blocks, and
     a sampler that lets batch composition leak into a draw, without needing a
     separate test for each.
  2. **A seed survives the crowd.** The same comparison, applied to requests that
     sample. It is a strictly harder claim than the first, because a greedy answer
     is a function of the logits and a sampled one is a function of an RNG whose
     state Day 40 had to key by request id to keep out of the batch's hands.
  3. **The pool comes back.** When the crowd leaves, every block is free, every
     slot is free, and Day 35's ledger audit passes. Blocks belonging to clients
     that hung up included, since nothing frees those for you.
  4. **The process survives clients that disappear.** One caller's dropped
     connection must cost that caller and nobody else: every client that stayed
     gets its answer, and the server is still serving afterwards.

Three things about the shape of this file are decisions rather than convenience.

**The server runs in a thread of this process, on a real socket.** A subprocess
would be one notch more faithful and would cost claim 3 entirely: across a process
boundary the only thing that can be asked about the pool is whatever `/health`
chooses to print, and "every block came back" is a statement about the allocator's
ledger, not about a summary line. In a thread the socket, the HTTP parser, the
chunked encoding and the disconnects are all real, and the test still holds the
`Engine` object it can audit afterwards. What is given up is process isolation,
which is not what any of the four claims are about.

**The baseline is the same server, one client at a time.** Not a fresh engine, not
an offline `generate`. The comparison has to isolate exactly one variable, which is
concurrency, so everything else has to be held identical: same weights, same
tokenizer, same pool, same code path from socket to sampler. A baseline computed
offline would also be testing that the HTTP layer agrees with the engine, which is
Day 37's test and not this one's.

**The crowd reports how crowded it was.** `peak_running` is sampled from `/health`
while the clients are in flight, and it exists for the same reason Day 35's soak
reports `max_preemptions`: a run that never put two rows in one batch has tested
the serving layer one request at a time, would pass every check in this file, and
would mean nothing. A test that cannot prove it was hard is a test that will get
easier without anybody noticing.

Everything here is a test and benchmark tool. It imports `httpx` and `uvicorn`,
which are the dev and server extras, and nothing in the engine imports it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from .audit import InvariantViolation, audit_engine


class AcceptanceFailure(AssertionError):
    """A claim this phase makes about itself, found to be false under a crowd.

    An `AssertionError` because that is what it is: not a bug in the harness and
    not an exception the server raised, but a statement in the docstring above that
    the running system just contradicted. The message names the client, because in
    a crowd of a dozen "the answers differ" is not actionable and "c7 got c3's
    tokens" is.
    """


# --- a live server ---------------------------------------------------------------


@dataclass(frozen=True)
class LiveServer:
    """A uvicorn serving `app` on a port that exists, for as long as the block runs."""

    app: Any
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@contextlib.contextmanager
def live_server(
    app,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    startup_timeout: float = 30.0,
    shutdown_timeout: float = 30.0,
    log_level: str = "warning",
) -> Iterator[LiveServer]:
    """Run `app` under uvicorn in a background thread, on a socket bound here.

    The socket is created and bound by this function rather than by uvicorn, for
    one reason: `port=0` asks the kernel to choose, and the caller needs the number
    it chose *before* anything can connect. Binding here means the port is known the
    instant the socket exists, and uvicorn is handed a socket it did not have to
    find. Ephemeral ports also mean two of these can run back to back, and in a test
    suite that matters more than it sounds: a fixed port turns one leaked server
    into every later test failing with `address already in use`.

    Waiting on `server.started` rather than on the port accepting connections is
    what makes the yield safe. Uvicorn sets that flag after the lifespan has run,
    and the lifespan is where `AsyncEngine.start` happens, so a request sent the
    moment this returns finds a loop that is already stepping. Polling the port
    instead would race the loop's startup and produce a first request that times out
    once in a hundred runs.

    Shutdown is `should_exit` plus a join with a deadline, and the deadline is the
    point. `server.run` is on a daemon thread, so a server that refuses to stop
    would otherwise be invisible: the interpreter would exit at the end of the run
    and the hang would present as a suite that is mysteriously slow. Joining turns
    it into one failed test with a name.
    """
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    bound_host, bound_port = sock.getsockname()[:2]

    config = uvicorn.Config(app, log_level=log_level, access_log=False, lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        name=f"nanoserve-acceptance-{bound_port}",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + startup_timeout
    while not server.started:
        if not thread.is_alive():
            sock.close()
            raise RuntimeError("the uvicorn thread died before the server started")
        if time.monotonic() > deadline:
            server.should_exit = True
            sock.close()
            raise TimeoutError(f"uvicorn did not start within {startup_timeout}s")
        time.sleep(0.005)

    try:
        yield LiveServer(app=app, host=bound_host, port=bound_port)
    finally:
        server.should_exit = True
        thread.join(shutdown_timeout)
        with contextlib.suppress(OSError):
            sock.close()
        if thread.is_alive():
            raise TimeoutError(f"uvicorn did not shut down within {shutdown_timeout}s")


# --- what one client intends to do -----------------------------------------------


@dataclass(frozen=True)
class ClientPlan:
    """One caller's whole intention, as a value the harness can run twice.

    Twice is the requirement that shaped it. The acceptance run executes every plan
    alone and then again in a crowd, and the two runs have to be the same request in
    every respect a server can observe, or the comparison proves nothing. So a plan
    is data, with no state and no clock in it, and the only thing that differs
    between the two executions is who else was on the socket.

    client_id:  the harness's name for this caller, which is *not* the server's
                request id. The server assigns those, and they differ between the
                solo pass and the crowd pass, which is exactly why the comparison
                cannot be keyed on them.
    hang_up_after_frames:  streaming only, and the deterministic way to leave: read
                exactly this many SSE frames, then close the connection.
    hang_up_after_seconds: the only handle a unary caller has, since it has nothing
                to count. A deadline is a worse test instrument than a frame count
                and it is the honest model of the case: a unary client that goes
                away goes away at a wall-clock moment, not at a token.
    """

    client_id: str
    prompt: str | list[int]
    max_tokens: int = 16
    stream: bool = False
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    hang_up_after_frames: int | None = None
    hang_up_after_seconds: float | None = None

    @property
    def is_sampled(self) -> bool:
        """True when this request draws rather than takes the argmax."""
        return self.temperature > 0.0 or self.top_k > 0 or self.top_p < 1.0

    @property
    def hangs_up(self) -> bool:
        return self.hang_up_after_frames is not None or self.hang_up_after_seconds is not None

    def body(self, model: str) -> dict:
        """The JSON this plan puts on the wire.

        `seed` is omitted when it is None rather than sent as null. Both are legal
        bodies and they mean the same thing to the server, but a greedy plan has to
        produce *exactly* the body every client of this server sent before Day 40,
        or the acceptance run is quietly exercising the sampled path for requests
        that asked for greedy and the greedy path is never covered at all.
        """
        body = {
            "model": model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        return body


def mixed_crowd(
    size: int,
    *,
    prompts: Sequence[str],
    max_tokens: Sequence[int] = (4, 12, 24),
    hang_up_every: int = 5,
    hang_up_after_frames: int = 2,
    hang_up_after_seconds: float = 0.15,
    start: int = 0,
) -> list[ClientPlan]:
    """A deterministic crowd that disagrees with itself about everything.

    Every axis is driven by the index and every stride is different, which is the
    only interesting property this function has. Streaming alternates on 2, sampling
    rotates on 3, prompt length rotates on `len(prompts)`, budget rotates on
    `len(max_tokens)`, and the hang-ups fall on `hang_up_every`. Coprime-ish strides
    mean the flavours cross: with the defaults, the clients that leave early are two
    streams and a unary, one greedy and two sampled, rather than all of them being
    whatever the first stride selected. A crowd where every disconnect happens to be
    a streaming greedy request tests one cell of the table and passes.

    Nothing here is random. A random crowd that fails once is a bug report nobody
    can act on, and the variety wanted is coverage, not entropy.

    `start` offsets the client ids so a caller can build noise around a request it
    named itself without colliding.
    """
    plans: list[ClientPlan] = []
    for i in range(size):
        g = start + i
        sampling: dict[str, Any] = {}
        if i % 3 == 1:
            sampling = {"temperature": 0.8, "top_p": 0.9, "seed": g}
        elif i % 3 == 2:
            sampling = {"temperature": 1.0, "top_k": 4, "seed": g}
        stream = i % 2 == 0
        leaves = hang_up_every > 0 and i % hang_up_every == 0
        plans.append(
            ClientPlan(
                client_id=f"c{g}",
                prompt=prompts[i % len(prompts)],
                max_tokens=max_tokens[i % len(max_tokens)],
                stream=stream,
                hang_up_after_frames=hang_up_after_frames if leaves and stream else None,
                hang_up_after_seconds=hang_up_after_seconds if leaves and not stream else None,
                **sampling,
            )
        )
    return plans


# --- what one client got ----------------------------------------------------------


@dataclass(frozen=True)
class ClientResult:
    """What actually happened to one caller, with the three outcomes kept apart.

    Answered, left, and failed are three different things and collapsing any two of
    them breaks a claim. A disconnect folded into `error` makes claim 4 unaskable,
    because "the caller left" and "the server broke" become the same row. A 400
    reported as a hang-up makes claim 1 vacuous, because a client that was refused
    has no answer to compare and would be silently skipped.
    """

    client_id: str
    status: int | None = None
    text: str = ""
    frames: int = 0
    hung_up: bool = False
    usage: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and not self.hung_up


@dataclass(frozen=True)
class CrowdReport:
    """The crowd's results, plus the evidence that it was a crowd.

    `peak_running` and `peak_active` are sampled from `/health` while the clients
    are in flight. They are not statistics about the run, they are the run's
    admissibility: every check in this file passes trivially against a crowd of
    one, so a report that cannot show two rows in one batch is a report whose green
    means nothing. Day 35's `max_preemptions` is the same idea for the soak.
    """

    results: dict[str, ClientResult] = field(default_factory=dict)
    peak_running: int = 0
    peak_active: int = 0
    health_polls: int = 0

    @property
    def completed(self) -> list[ClientResult]:
        return [r for r in self.results.values() if r.ok]

    @property
    def hung_up(self) -> list[ClientResult]:
        return [r for r in self.results.values() if r.hung_up]

    @property
    def failed(self) -> list[ClientResult]:
        return [r for r in self.results.values() if not r.ok and not r.hung_up]

    @property
    def texts(self) -> dict[str, str]:
        return {r.client_id: r.text for r in self.completed}


# --- driving one client ------------------------------------------------------------


async def run_client(
    client: httpx.AsyncClient, plan: ClientPlan, *, model: str = "nanoserve"
) -> ClientResult:
    """Execute one plan and report which of the three things happened.

    Never raises for anything the server or the network did. A crowd is run with
    `asyncio.gather`, and one client raising there would cancel or mask the others,
    which would turn "c7 got a 500" into "the acceptance test errored" and lose
    every other client's result in the same run. Failures are values here so that
    the checks can be run over the whole crowd at once.
    """
    if plan.stream:
        return await _run_stream(client, plan, model)
    return await _run_unary(client, plan, model)


async def _run_unary(client: httpx.AsyncClient, plan: ClientPlan, model: str) -> ClientResult:
    """POST and wait for the whole answer, unless the plan says to leave first.

    The deadline is `asyncio.wait_for` around the post, and the cancellation it
    raises is what closes the connection: httpx cannot return a half-read response
    to its pool, so it drops the socket, and the server sees a client that vanished
    mid-generation. That is the disconnect a unary caller can actually produce, and
    it is a genuinely different event from a streaming one, because the server has
    written nothing yet and is parked on a future rather than inside a generator.
    """
    request = client.post("/v1/completions", json=plan.body(model))
    try:
        if plan.hang_up_after_seconds is not None:
            response = await asyncio.wait_for(request, plan.hang_up_after_seconds)
        else:
            response = await request
    except (asyncio.TimeoutError, TimeoutError):
        return ClientResult(client_id=plan.client_id, hung_up=True)
    except Exception as exc:  # noqa: BLE001 - a transport failure is this client's result
        return ClientResult(client_id=plan.client_id, error=f"{type(exc).__name__}: {exc}")
    if response.status_code != 200:
        return ClientResult(
            client_id=plan.client_id, status=response.status_code, error=response.text
        )
    payload = response.json()
    return ClientResult(
        client_id=plan.client_id,
        status=200,
        text=payload["choices"][0]["text"],
        frames=1,
        usage=payload["usage"],
    )


async def _run_stream(client: httpx.AsyncClient, plan: ClientPlan, model: str) -> ClientResult:
    """Read SSE frames, reassemble the text, and optionally stop reading mid-answer.

    Leaving is `break`, and the connection close is the `async with` unwinding over
    a response that was never finished: httpx will not reuse a socket whose body is
    incomplete, so it closes it, uvicorn turns that into an ASGI `http.disconnect`,
    and Starlette cancels the streaming response. Three hops, each of which is a
    place the disconnect can be dropped, and none of which exist under
    `ASGITransport`.

    `[DONE]` is the sentinel, not the last data frame, so a body that stopped early
    is not mistaken for one that ended: `complete` stays False and the result is
    reported as an error rather than as a shorter answer, which is the difference
    between finding a truncation bug and averaging over it.
    """
    parts: list[str] = []
    frames = 0
    usage: dict | None = None
    error: str | None = None
    complete = False
    try:
        async with client.stream("POST", "/v1/completions", json=plan.body(model)) as response:
            if response.status_code != 200:
                await response.aread()
                return ClientResult(
                    client_id=plan.client_id, status=response.status_code, error=response.text
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    complete = True
                    break
                payload = json.loads(data)
                if "error" in payload:
                    error = payload["error"]["message"]
                    continue
                frames += 1
                parts.append(payload["choices"][0]["text"])
                usage = payload.get("usage") or usage
                if plan.hang_up_after_frames is not None and frames >= plan.hang_up_after_frames:
                    return ClientResult(
                        client_id=plan.client_id,
                        text="".join(parts),
                        frames=frames,
                        hung_up=True,
                    )
    except Exception as exc:  # noqa: BLE001 - a transport failure is this client's result
        return ClientResult(
            client_id=plan.client_id,
            status=200,
            text="".join(parts),
            frames=frames,
            error=f"{type(exc).__name__}: {exc}",
        )
    if error is None and not complete:
        error = "the stream ended without [DONE]"
    return ClientResult(
        client_id=plan.client_id,
        status=200,
        text="".join(parts),
        frames=frames,
        usage=usage,
        error=error,
    )


# --- driving a crowd ----------------------------------------------------------------


async def wait_until_idle(
    client: httpx.AsyncClient, *, timeout: float = 30.0, poll_every: float = 0.005
) -> dict:
    """Poll `/health` until the loop has nothing left, and return that reading.

    Asked over HTTP rather than off the engine object, and not because the object is
    out of reach. An abort is *queued*: a disconnect marks a request and the loop
    applies it at the top of its next turn, so the moment `run_client` returns, the
    departed client may still be a running row holding blocks. Reading the
    allocator's counters at that instant is a race that fails one run in twenty and
    looks like a leak. `/health` is answered by the same event loop that owns the
    scheduler, so a reading of zero is a statement made by the thread that would
    know.
    """
    deadline = time.monotonic() + timeout
    while True:
        health = (await client.get("/health")).json()
        if health["active"] == 0 and health["running"] == 0 and health["waiting"] == 0:
            return health
        if time.monotonic() > deadline:
            raise TimeoutError(f"the server was still busy after {timeout}s: {health}")
        await asyncio.sleep(poll_every)


async def run_solo(
    base_url: str,
    plans: Sequence[ClientPlan],
    *,
    model: str = "nanoserve",
    timeout: float = 60.0,
) -> dict[str, ClientResult]:
    """Run every plan alone, on this server, and return the baseline answers.

    Sequential, with an idle wait between each, and both halves matter. Sequential
    is what "alone" means. The idle wait is what makes it true: without it, a plan
    that hung up leaves a request the loop has not aborted yet, and the next plan
    starts as row two of a batch, which is the exact condition the baseline is
    supposed to be free of.
    """
    results: dict[str, ClientResult] = {}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        for plan in plans:
            await wait_until_idle(client)
            results[plan.client_id] = await run_client(client, plan, model=model)
        await wait_until_idle(client)
    return results


async def run_crowd(
    base_url: str,
    plans: Sequence[ClientPlan],
    *,
    model: str = "nanoserve",
    timeout: float = 60.0,
    poll_every: float = 0.005,
) -> CrowdReport:
    """Run every plan at once and record how crowded it actually got.

    The watcher is a task polling `/health` next to the clients, and it is measuring
    the one thing the test cannot arrange: how many of these requests the scheduler
    really had in one batch. That is decided by uvicorn's accept order, the kernel,
    and `max_batch_size`, and no amount of care in the harness can force it. So it
    is observed instead, and asserted on afterwards.

    The watcher is cancelled in a `finally` because a crowd that raises must not
    leave a task polling a server that is about to be shut down: that turns one test
    failure into a shutdown that hangs, which is the failure mode hardest to read.
    """
    peak = {"running": 0, "active": 0, "polls": 0}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        watcher = asyncio.create_task(_watch(client, peak, poll_every))
        try:
            results = await asyncio.gather(
                *(run_client(client, plan, model=model) for plan in plans)
            )
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
    return CrowdReport(
        results={r.client_id: r for r in results},
        peak_running=peak["running"],
        peak_active=peak["active"],
        health_polls=peak["polls"],
    )


async def _watch(client: httpx.AsyncClient, peak: dict, poll_every: float) -> None:
    """Sample `/health` forever, keeping the high-water marks. Cancelled by the caller."""
    while True:
        try:
            health = (await client.get("/health")).json()
        except Exception:  # noqa: BLE001 - the watcher must never fail the run
            await asyncio.sleep(poll_every)
            continue
        peak["running"] = max(peak["running"], health["running"])
        peak["active"] = max(peak["active"], health["active"])
        peak["polls"] += 1
        await asyncio.sleep(poll_every)


# --- the claims ---------------------------------------------------------------------


def check_answers(crowd: CrowdReport, solo: dict[str, ClientResult]) -> None:
    """Claim 1 and claim 2: every answer is a function of its own request.

    One comparison covers both because a seed is not a special case of anything: if
    a client's crowd text equals its solo text, then nothing about who else was in
    the batch reached it, whether the token was chosen by an argmax or drawn from a
    generator. The sampled clients are simply the ones where this can fail for a
    reason greedy cannot have, which is Day 40's shared-generator coupling.

    A client with no baseline is a failure, not a skip. Comparing against a dict
    that happens to be empty is the way a check like this rots into a no-op, and it
    would do so silently: the loop body would never run and the function would
    return successfully having asserted nothing.
    """
    mismatched: list[str] = []
    missing: list[str] = []
    for client_id, result in sorted(crowd.results.items()):
        if result.hung_up:
            continue
        baseline = solo.get(client_id)
        if baseline is None:
            missing.append(client_id)
            continue
        if baseline.hung_up:
            continue
        if result.text != baseline.text:
            mismatched.append(
                f"{client_id}: alone {baseline.text!r}, in the crowd {result.text!r}"
            )
    if missing:
        raise AcceptanceFailure(
            f"no solo run to compare against for {', '.join(missing)}: the comparison "
            "would have passed by having nothing to check"
        )
    if mismatched:
        raise AcceptanceFailure(
            f"{len(mismatched)} client(s) got a different answer in a crowd than alone, "
            "which means a request's tokens depend on who it shared a batch with:\n  "
            + "\n  ".join(mismatched)
        )


def check_survivors(crowd: CrowdReport) -> None:
    """Claim 4: the only clients without an answer are the ones that left.

    The claim is about blast radius. A disconnect is a normal event and the server
    is allowed to notice it; what it is not allowed to do is let it become somebody
    else's 500, somebody else's truncated stream, or a loop that stops. So every
    client that stayed is required to have a 200 and a complete body, and the
    failure message carries each one's status and error because "the crowd failed"
    and "c3 got a 503 because the loop died" are different mornings.
    """
    failed = crowd.failed
    if failed:
        detail = "\n  ".join(f"{r.client_id}: status {r.status}, {r.error!r}" for r in failed)
        raise AcceptanceFailure(
            f"{len(failed)} client(s) that did not hang up failed anyway:\n  {detail}"
        )


def check_pool_returned(engine) -> None:
    """Claim 3: the crowd left and took nothing with it.

    Two questions in one function, asked in this order for a reason.

    First the ledger, via Day 35's `audit_engine`: is the pool *consistent*? A block
    held by two requests, or allocated and owned by nobody, is a corruption whose
    symptom is a wrong answer much later, and it has to be reported as itself rather
    than as a count that happens to be off. Its `InvariantViolation` is re-raised in
    this layer's vocabulary so a caller has one exception type to catch.

    Then the counts: is the pool *empty*? These are genuinely different. A request
    still running after the crowd has left passes the audit, because holding blocks
    is exactly what a running request is entitled to do, and fails this, which is
    what a leaked abort looks like: the socket is gone, the caller got its result or
    its disconnect, and a row is still generating into nothing.
    """
    try:
        audit_engine(engine)
    except InvariantViolation as exc:
        raise AcceptanceFailure(f"the engine's invariants did not survive the crowd: {exc}") from exc

    allocator = engine.allocator
    if allocator.num_free != allocator.num_blocks:
        raise AcceptanceFailure(
            f"{allocator.num_blocks - allocator.num_free} block(s) never came back: the "
            f"pool has {allocator.num_free} of {allocator.num_blocks} free with nothing "
            "left to serve"
        )
    scheduler = engine.scheduler
    if len(scheduler.free_slots) != scheduler.max_batch_size:
        raise AcceptanceFailure(
            f"{scheduler.max_batch_size - len(scheduler.free_slots)} slot(s) never came "
            f"back: {len(scheduler.free_slots)} of {scheduler.max_batch_size} cache rows "
            "are free and the engine is idle"
        )
    if scheduler.num_running or scheduler.num_waiting:
        raise AcceptanceFailure(
            f"the crowd has left and the scheduler still has {scheduler.num_running} "
            f"running and {scheduler.num_waiting} waiting"
        )
