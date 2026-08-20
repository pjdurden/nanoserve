"""The bridge: many HTTP coroutines, one synchronous engine loop. Weeks 10-11.

Every day so far had one caller. `Engine.generate` takes a list of prompts, drives
`step()` in a `while` loop, and hands back a list; the loop owns the process for
as long as it runs, which is exactly right for a benchmark and useless for a
server. A server's requests arrive one at a time from sockets that already exist,
at moments nobody chose, and the whole point of Week 8 is that a request admitted
now joins a batch already in flight. So the loop cannot be owned by a caller. It
has to be a thing that is always running, and callers have to be able to reach it.

There are two hard facts and the design is what falls out of them.

**`Engine.step()` blocks and owns the GPU.** It is one forward pass over whatever
the scheduler picked: no `await` anywhere in it, and no way to give one. Called
from a coroutine, it stops the event loop for its whole duration, which on a real
model is tens of milliseconds. During that window nothing else on the loop runs:
no socket is read, no arrival is accepted, no finished answer is written back. The
server would still be *correct*, and every request in the batch would still get
the right tokens, but the concurrency that continuous batching buys would be spent
on a thread that is not allowed to use it. So the step runs in a worker thread
(`asyncio.to_thread`), and the loop is free while it does. Torch drops the GIL for
the ops that matter, and while the GPU is busy the loop is genuinely idle.

**A thread means two writers.** The moment the step is off the loop, the scheduler
has two threads reaching for it: the worker inside `schedule()`, and the event loop
inside a handler that wants to add a request. `Scheduler` is plain Python
bookkeeping with no lock in it, on purpose, because everything it does is supposed
to happen at one point in the iteration. Adding a request to `waiting` while
`schedule()` walks that same deque is the kind of bug that costs a week. So
arrivals do not touch the engine at all: they go on an inbox, and the loop drains
the inbox at the top of an iteration, which is the one instant when nothing is
mid-forward. Aborts take the same road for the same reason.

That is the whole bridge, and the rest is where a caller's answer goes.

**One future per request, resolved at that request's own finish.** Not at the
batch's, which is the Day-29 bill this project already paid off inside the engine
and would otherwise re-introduce at the socket. And not at the *reap* either:
`SchedulerOutput.finished` reports a request one iteration after its last token,
because releasing its slot and its blocks is the next schedule's first job.
Waiting for that would charge every caller an extra iteration of latency for a
fact the engine already had, so the loop settles on `request.is_finished` and lets
the reap happen behind the answer.

**A per-request failure must stay per-request.** This is the part that has no
offline equivalent. `Engine.generate` may raise at its one caller and be done;
here, a prompt too large for the pool is one caller's 400 while nine other
requests keep generating, and the refusal happens inside the shared loop, in the
drain, on behalf of somebody who is not there. Every error in this file is
therefore routed to a future rather than raised: the rejected arrival to its own,
a step that blew up to all of them (a loop that dies quietly is a server where
every open connection hangs forever), a schedule that stops progressing to all of
them too. `Engine.run_to_completion` answers that last one with a hang guard that
raises into a script. A server has to answer it by failing sockets.

**And one future becomes one queue when the caller wants the tokens early.**
Day 39. `stream` is the same request on a different delivery schedule: the loop
pushes each new output token onto a queue at the same `_settle` that would have
resolved the future, and the caller reads them with `async for`. Nothing about
the engine, the scheduler or the batch changes, which is the property worth
guarding, because the moment streaming changes a token it has stopped being a
delivery decision. The queue is unbounded on purpose, and that is the one design
call here that is not obvious: a bound would make one slow socket the pace of a
forward pass that a dozen other requests are rows of, and there is no way to
pause a single row of a batch. So backpressure on a stream would be backpressure
on everybody, and the tokens wait in memory instead.

This object is loop-affine and not thread-safe: `submit`, `stream` and `abort`
must be called from the event loop that ran `start`, which is where request
handlers run anyway. vLLM's `AsyncLLMEngine` is this same shape: a background
loop, a queue of arrivals, one stream per request.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .scheduler import Request


class EngineStopped(RuntimeError):
    """The loop is not running, so nothing submitted now can ever be answered.

    Raised at submit and handed to whatever was still in flight when `stop` was
    called. Shutdown does not drain: a caller told the server is going away can
    retry, a caller left holding a future nobody will resolve cannot.
    """


class EngineStuck(RuntimeError):
    """The scheduler produced nothing, repeatedly, while claiming to have work.

    The serving equivalent of `Engine.run_to_completion`'s iteration cap. Day 36's
    watermark deadlock was exactly this shape (both queues stopped and no future
    iteration could change it), and offline it presents as a wedged process, which
    is a fine way to learn about it from a terminal and an unacceptable way to
    learn about it from a request that never returns.
    """


@dataclass(frozen=True)
class StreamUpdate:
    """One thing the loop has to say about a streaming request.

    token_id: the token this update is about, or None on the final update. There
              is exactly one final update per stream and it is always last, so a
              consumer can treat it as the closing frame without counting.
    request:  the live `Request`. On the final update it is finished, which is
              where `finish_reason` and the token counts come from; before that it
              is a running request being read between iterations, so anything
              other than "how many tokens so far" is a snapshot.

    Frozen, and the token is a copy rather than an index into the request, because
    the request keeps mutating after this update is queued. A consumer reading a
    slow socket must see the token that was current when the update was made, not
    the one that is current when it gets around to looking.
    """

    request: Request
    token_id: int | None = None

    @property
    def is_final(self) -> bool:
        return self.token_id is None


@dataclass
class _Waiter:
    """One in-flight request and whatever its caller is parked on.

    A unary caller has a future, resolved once. A streaming caller has a queue,
    pushed to at every token. Exactly one of the two is set, and everything below
    branches on which, because the difference between `/v1/completions` and its
    `stream=true` twin is entirely this field.

    admitted: the loop has handed this request to the engine. Until then it lives
              only in the inbox, which is why an abort that arrives early cannot
              go to `Scheduler.abort`: the scheduler has never heard of it.
    delivered: output tokens already pushed onto the queue. The stream's cursor,
              kept here rather than read off the consumer because the loop is the
              only thread allowed to look at the request at all.
    """

    request: Request
    future: asyncio.Future | None = None
    queue: asyncio.Queue | None = None
    admitted: bool = False
    delivered: int = 0
    closed: bool = False

    @property
    def streaming(self) -> bool:
        return self.queue is not None

    @property
    def abandoned(self) -> bool:
        """Nobody is on the other end any more, so the request is pure cost."""
        return self.future is not None and self.future.cancelled()

    def deliver(self) -> None:
        """Push every output token the consumer has not been shown yet.

        Unbounded, on purpose, and this is the one design decision in streaming
        that is not obvious. A bounded queue would make a slow reader's socket the
        pace of a forward pass that three other requests are rows of: there is no
        way to pause one row of a batch, so backpressure on a stream is
        backpressure on everybody. The tokens wait in memory instead, which is a
        few bytes per token for a request that is already holding blocks.
        """
        output_ids = self.request.output_token_ids
        while self.delivered < len(output_ids):
            self.queue.put_nowait(StreamUpdate(self.request, output_ids[self.delivered]))
            self.delivered += 1

    def settle(self) -> None:
        """The request finished: resolve the future, or close the stream."""
        if self.queue is not None:
            self.deliver()
            self.close()
        elif self.future is not None and not self.future.done():
            self.future.set_result(self.request)

    def close(self) -> None:
        """Put the final update on the queue, once, whatever ended the stream."""
        if self.closed:
            return
        self.closed = True
        self.queue.put_nowait(StreamUpdate(self.request, None))

    def fail(self, exc: BaseException) -> None:
        """Hand the bad news to whichever kind of caller this is.

        A stream carries the exception *in* the queue rather than raising it into
        the loop, because the consumer may be several tokens behind: raising
        immediately would drop tokens that were already produced and paid for.
        """
        if self.queue is not None:
            if not self.closed:
                self.closed = True
                self.queue.put_nowait(exc)
        elif self.future is not None and not self.future.done():
            self.future.set_exception(exc)


@dataclass
class AsyncEngine:
    """A running `Engine` with an inbox in front of it and a future per request.

    engine:             any object with the `Engine` surface this uses, which is
                        four methods (`add_request`, `abort`, `has_unfinished`,
                        `step`) plus its counters. Kept narrow so the bridge can
                        be tested against a spy that records when it was called.
    max_idle_schedules: consecutive empty schedules tolerated while the engine
                        still claims unfinished work, before the callers are told
                        the loop is stuck rather than left waiting on it.
    """

    engine: object
    max_idle_schedules: int = 256

    _inbox: deque = field(default_factory=deque, init=False, repr=False)
    _live: dict = field(default_factory=dict, init=False, repr=False)
    _aborted: set = field(default_factory=set, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _wakeup: asyncio.Event | None = field(default=None, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

    #: The exception that killed the loop, or None. Sticky: once the loop is dead
    #: every later submit is refused with it instead of being accepted and hung.
    failure: BaseException | None = field(default=None, init=False)
    #: True while the loop is asleep on its wakeup event with nothing to do.
    parked: bool = field(default=False, init=False)
    steps: int = field(default=0, init=False)
    idle_waits: int = field(default=0, init=False)

    # --- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Start the loop task. Idempotent, because a lifespan can be re-entered."""
        if self._task is not None:
            return
        self._closing = False
        self._wakeup = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        """Stop the loop after the step in flight, and fail whatever is left.

        The loop is asked to leave rather than cancelled, so a step already inside
        a worker thread finishes and the scheduler is never abandoned halfway
        through an iteration. What it does not do is drain: outstanding requests
        get `EngineStopped`, which is a shutdown a caller can see.
        """
        task, self._task = self._task, None
        self._closing = True
        if self._wakeup is not None:
            self._wakeup.set()
        if task is not None:
            await task
        self.parked = False
        self._fail_all(EngineStopped("the engine loop has stopped"))

    async def __aenter__(self) -> AsyncEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def num_active(self) -> int:
        """Requests the bridge owes an answer for, admitted or still in the inbox."""
        return len(self._live)

    # --- what a handler calls -------------------------------------------------

    def submit(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 16,
        eos_token_id: int | None = None,
        request_id: str | None = None,
    ) -> asyncio.Future:
        """Queue a request and return the future its answer will arrive on.

        Synchronous, and that is the useful half: everything checkable without the
        engine is checked here, in the handler, so a malformed request is a 400
        before it has cost the loop an iteration. `Request.__post_init__` rejects
        an empty prompt and a budget below one; a repeated id is rejected here
        because ids are the bridge's map key and a collision would hand one
        caller another's tokens.

        What is *not* checked here is anything the engine owns. Whether the pool
        can ever hold this request is `Scheduler.add_request`'s answer, and asking
        it would mean touching the scheduler from the loop thread while a step may
        be running in another. So that refusal comes back later, on this future.
        """
        waiter = self._accept(
            prompt_token_ids, max_new_tokens, eos_token_id, request_id, streaming=False
        )
        return waiter.future

    def _accept(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int | None,
        request_id: str | None,
        *,
        streaming: bool,
    ) -> _Waiter:
        """Everything `submit` and `stream` share: check, build, queue, wake.

        The only difference between the two is the last line of it, which is
        whether the caller gets a future or a queue. Everything before that
        (the loop is alive, the id is free, the `Request` is well formed) is one
        set of rules, because a streaming caller is not owed weaker ones.
        """
        if self.failure is not None:
            raise self.failure
        if self._task is None or self._closing:
            raise EngineStopped("the engine loop is not running")
        request_id = self._new_id() if request_id is None else request_id
        if request_id in self._live:
            raise ValueError(f"request id {request_id!r} is already in flight")
        request = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        if streaming:
            waiter = _Waiter(request, queue=asyncio.Queue())
        else:
            waiter = _Waiter(request, future=asyncio.get_running_loop().create_future())
        self._live[request_id] = waiter
        self._inbox.append(request_id)
        self._wakeup.set()
        return waiter

    async def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 16,
        eos_token_id: int | None = None,
        request_id: str | None = None,
    ) -> Request:
        """Submit and await one request. The coroutine an HTTP handler is.

        The `CancelledError` arm is the disconnect path and it is not optional. A
        client that hangs up cancels this coroutine, and a request nobody is
        waiting for still holds a slot, still holds blocks, and still spends a row
        of every forward until it reaches its budget. Aborting on the way out is
        what turns a closed socket into freed KV.
        """
        request_id = self._new_id() if request_id is None else request_id
        future = self.submit(prompt_token_ids, max_new_tokens, eos_token_id, request_id)
        try:
            return await future
        except asyncio.CancelledError:
            self.abort(request_id)
            raise

    async def stream(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int = 16,
        eos_token_id: int | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[StreamUpdate]:
        """Submit one request and yield its tokens as the loop produces them.

        The same request `generate` would run, delivered on a different schedule:
        one `StreamUpdate` per output token, then exactly one final update
        carrying the finished request. Nothing about the engine, the scheduler or
        the batch changes, which is the property the tests hammer, because the
        moment streaming changes a token it stops being a delivery decision and
        starts being a different model.

        Two things about the shape are deliberate.

        **The submit happens when this is first iterated, not when it is called.**
        An async generator's body does not run until `__anext__`, so a caller that
        builds one and drops it has not queued anything. That also puts admission
        errors on the first read, which is what lets the HTTP layer choose a
        status code: no byte of the body has gone out yet, so `KVCacheExhausted`
        can still be a 400 rather than an error frame inside a 200.

        **The `finally` is the disconnect path and it is not optional.** A stream
        can end from the consumer's side, by `break`, by `aclose`, or by the whole
        handler being cancelled when a socket dies, and none of those look like
        Day 37's `CancelledError` at an await. All of them run this `finally`, and
        the abort in it is what turns a hung-up client into freed KV instead of a
        row that keeps generating into a queue nobody will ever read.
        """
        waiter = self._accept(
            prompt_token_ids, max_new_tokens, eos_token_id, request_id, streaming=True
        )
        queue = waiter.queue
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.is_final:
                    return
        finally:
            self.abort(waiter.request.request_id)

    def abort(self, request_id: str) -> None:
        """Ask for a request to stop. Applied at the next iteration boundary.

        Unknown ids are ignored rather than raised at, unlike `Scheduler.abort`.
        By the time a disconnect is noticed the request may have finished and been
        answered, and a server that turns "too late to cancel" into an exception
        has invented an error out of a race it cannot win.
        """
        if request_id not in self._live:
            return
        self._aborted.add(request_id)
        if self._wakeup is not None:
            self._wakeup.set()

    def stats(self) -> dict:
        """What a health endpoint reports: the loop's work and the engine's."""
        scheduler = self.engine.scheduler
        return {
            "iterations": self.engine.iterations,
            "steps": self.steps,
            "idle_waits": self.idle_waits,
            "active": self.num_active,
            "running": scheduler.num_running,
            "waiting": scheduler.num_waiting,
            "parked": self.parked,
            "loop_running": self.running,
        }

    def _new_id(self) -> str:
        self._next_id += 1
        return f"req-{self._next_id}"

    # --- the loop -------------------------------------------------------------

    async def _run(self) -> None:
        """Drain, abort, settle, step. Forever, and asleep when there is nothing.

        The order is the whole invariant. Everything that mutates the scheduler
        happens in the first half of a turn, on this thread, with no `await`
        between the drain and the step; the step itself is the only thing that
        happens elsewhere. So there is exactly one writer at any instant, without
        a lock, and "when is it safe to add a request?" has one answer: here.

        Parking is the other half. An idle server must not spin the engine, and
        must not make the next arrival wait for a poll either, so the loop sleeps
        on an event that `submit` sets. The clear-then-check order matters and is
        the classic wakeup race: clear the event first, then look at the inbox,
        because a request that arrived before the clear is already *in* the inbox
        and one that arrives after it will set the event again.
        """
        idle_schedules = 0
        try:
            while not self._closing:
                self._drain_inbox()
                self._apply_aborts()
                self._settle()
                if self._closing:
                    break
                if not self.engine.has_unfinished():
                    self._wakeup.clear()
                    if self._inbox or self._aborted:
                        continue
                    self.idle_waits += 1
                    self.parked = True
                    await self._wakeup.wait()
                    self.parked = False
                    continue
                # The only line that leaves this thread, and the reason for every
                # queue above it.
                out = await asyncio.to_thread(self.engine.step)
                self.steps += 1
                self._settle()
                idle_schedules = idle_schedules + 1 if out.is_empty else 0
                if idle_schedules >= self.max_idle_schedules:
                    raise EngineStuck(
                        f"the scheduler produced nothing for {idle_schedules} "
                        f"consecutive steps with work outstanding: "
                        f"{self.engine.scheduler.num_running} running, "
                        f"{self.engine.scheduler.num_waiting} waiting"
                    )
        except Exception as exc:  # noqa: BLE001 - the loop is the last line of defence
            self.failure = exc
            self.parked = False
            self._fail_all(exc)

    def _drain_inbox(self) -> None:
        """Hand every arrival to the engine, or fail it, at an iteration boundary.

        The `try` is the point. `Scheduler.add_request` refuses a request whose
        worst case cannot fit an empty pool, and that refusal arrives here, in a
        loop shared by every other caller. Letting it propagate would kill the
        loop and hang everybody; it belongs to one future.
        """
        while self._inbox:
            request_id = self._inbox.popleft()
            waiter = self._live.get(request_id)
            if waiter is None:
                continue
            if waiter.abandoned or request_id in self._aborted:
                # Cancelled or aborted before the engine ever saw it: it holds no
                # slot and no blocks, so finishing it is bookkeeping and nothing
                # else. `Scheduler.abort` could not do this, it has never heard
                # of the request.
                self._aborted.discard(request_id)
                waiter.request.finish("abort")
                waiter.settle()
                del self._live[request_id]
                continue
            try:
                self.engine.add_request(waiter.request)
            except Exception as exc:  # noqa: BLE001 - one caller's error, not the loop's
                waiter.fail(exc)
                del self._live[request_id]
                continue
            waiter.admitted = True

    def _apply_aborts(self) -> None:
        """Push queued aborts into the scheduler, now that nothing is mid-forward."""
        while self._aborted:
            request_id = self._aborted.pop()
            waiter = self._live.get(request_id)
            if waiter is None or not waiter.admitted:
                continue
            if not waiter.request.is_finished:
                self.engine.abort(request_id)

    def _settle(self) -> None:
        """Hand every new token, and every finished request, to its own caller.

        On `is_finished` rather than on the scheduler's `finished` list, so an
        answer is delivered on the step that produced its last token instead of on
        the step that released its blocks. A cancelled future is the disconnect
        case arriving from the other direction: nobody wants the answer, so the
        request is aborted here instead of being allowed to run to its budget.
        """
        for request_id, waiter in list(self._live.items()):
            if waiter.abandoned:
                if waiter.admitted and not waiter.request.is_finished:
                    self.engine.abort(request_id)
                del self._live[request_id]
                continue
            if not waiter.admitted:
                continue
            if waiter.streaming:
                # The one line streaming adds to the loop: push whatever this
                # request emitted since the last look. Every token goes out on the
                # step that produced it, which is the whole product.
                waiter.deliver()
            if waiter.request.is_finished:
                waiter.settle()
                del self._live[request_id]

    def _fail_all(self, exc: BaseException) -> None:
        """Give everyone the bad news. A hung socket is worse than an error."""
        for waiter in list(self._live.values()):
            waiter.fail(exc)
        self._live.clear()
        self._inbox.clear()
