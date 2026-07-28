"""Scheduler: static then continuous batching. Weeks 7-9.

Week 7 batched a fixed set of sequences and Day 29 measured what that cost. The
numbers are the reason this module exists. Seven rows of 4 tokens batched behind
one row of 32: the decode issued 248 tokens and collected 52, so 79% of the
forward computed tokens for rows that were already done, and a short row's answer
was handed back at 6.6x its own latency because a static batch returns every row
when its slowest row finishes. Both bills come from the same structural fact, that
the batch is a rectangle chosen once, and a row is an index into it.

Week 8 replaces the index with an object. A `Request` has a *state*, so it can
leave the batch the moment it finishes and a waiting request can take the slot it
just gave up. Today is the first half: the state machine and the two queues.

  waiting  --admit-->  running  --stop/length-->  finished
     \\                                              ^
      \\----------------- abort ---------------------/

Two resources gate admission, and keeping them separate is most of the design:

  1. **A slot.** The batched cache is built with `batch_size` rows, so at most
     that many requests can be in one forward. A slot is a row index, handed out
     at admission and returned at finish. Lowest free slot wins, so slots are
     recycled rather than left as holes and the tests can name them.
  2. **KV blocks.** A request needs somewhere to put its K/V before it can run a
     single step, and this is the resource that actually runs out. Admission
     reserves `worst_case_tokens` (the prompt plus every token the request is
     still allowed to emit) atomically through the Day-14 allocator.

Reserving the worst case is a deliberate, temporary choice, and it is the debt
Week 9 is built to pay. Its virtue is that a running request can never fail
mid-flight: it already owns every block it could possibly need, so there is no
state in which the pool is dry and a running row must be rolled back. That is what
makes a scheduler safe to build before preemption exists. Its cost is real and
measured here rather than hidden: `reservation_waste` reports the share of
reserved blocks holding no tokens yet, and a request that stops at token 3 of a
possible 200 held ~99% of its reservation for nothing, which is concurrency the
pool could have sold to someone else. vLLM allocates a block at a time and
preempts (recompute or swap) when the pool cannot grow a running sequence, which
is strictly better and needs the machinery Week 9 adds.

Admission is FIFO and does not skip over a blocked head. If the request at the
front cannot fit, scheduling stops there for this iteration even when something
smaller behind it would fit. Skipping ahead raises utilisation and starves large
requests indefinitely, which trades a visible queue for an invisible one, and a
serving stack that never runs the big prompt is worse than one that makes small
prompts wait a step.

There is no forward pass in this file, on purpose. Everything here is bookkeeping
over integers and object states, which is the part that is subtly wrong when it is
wrong, so it is tested without a model the same way the Day-14 allocator and the
Day-29 timing core were. Day 31 wires it to `greedy_generate_batch`.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .cache import BlockAllocator, KVCacheExhausted


class RequestState(Enum):
    """Where a request is in its life. The whole state space, and it is small.

    WAITING:  accepted by the engine, owns no slot and no blocks, costs nothing
              but a queue entry.
    RUNNING:  holds a slot and its block reservation, is in every forward until it
              finishes.
    FINISHED: terminal. Its resources are released at the next scheduling step and
              its tokens are handed back to the caller.

    Week 9 adds preemption, which is the missing RUNNING -> WAITING edge: a
    running request pushed back to the queue with its blocks freed, to be
    recomputed later.
    """

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class IllegalTransition(RuntimeError):
    """A request was asked to do something its current state does not allow.

    Loud on purpose. Every one of these is a scheduler bug with a quiet failure
    mode: appending a token to a finished request silently corrupts its output,
    and re-admitting a finished one double-allocates blocks that were already
    freed. A raise costs one crashed test, the alternative costs a pool.
    """


# The state machine as data. Every edge the engine is allowed to take, and nothing
# else: no self-loops (a no-op transition is a caller bug, not a nothing), no
# resurrection from FINISHED, and no RUNNING -> WAITING until Week 9 has a
# recompute path to make it mean something.
_LEGAL_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.WAITING: frozenset({RequestState.RUNNING, RequestState.FINISHED}),
    RequestState.RUNNING: frozenset({RequestState.FINISHED}),
    RequestState.FINISHED: frozenset(),
}


@dataclass
class Request:
    """One generation request, and everything the scheduler knows about it.

    request_id:       caller's handle, unique within a scheduler.
    prompt_token_ids: the prompt, already tokenized. Non-empty: there has to be a
                      token to attend to.
    max_new_tokens:   the generation budget. It is also the admission quantity,
                      since the reservation is sized by what the request *could*
                      still emit rather than what it will.
    eos_token_id:     stop token, or None to always run the full budget.

    Mutable, unlike the frozen `PaddedBatch` of Day 27, and for the opposite
    reason. A padded batch describes one forward pass and the next step builds a
    new one; a request outlives every batch it appears in, and its whole job is to
    accumulate state across them.
    """

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int = 16
    eos_token_id: int | None = None
    state: RequestState = RequestState.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    # Assigned at admission, returned at finish. `slot` is the row this request
    # occupies in the batched cache; `block_ids` is its reservation from the pool.
    slot: int | None = None
    block_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError(f"request {self.request_id!r} has an empty prompt")
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be at least 1, got {self.max_new_tokens} for "
                f"request {self.request_id!r}"
            )

    # --- length ---------------------------------------------------------------

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Tokens this request currently occupies in the cache."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def worst_case_tokens(self) -> int:
        """The longest this request can ever get: prompt plus its whole budget.

        The number admission reserves against. It is an upper bound and usually a
        loose one, which is exactly what `Scheduler.reservation_waste` measures.
        """
        return self.num_prompt_tokens + self.max_new_tokens

    @property
    def token_ids(self) -> list[int]:
        """Prompt and generation as one sequence, which is what the caller wanted."""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_finished(self) -> bool:
        return self.state is RequestState.FINISHED

    # --- transitions ----------------------------------------------------------

    def transition_to(self, state: RequestState) -> None:
        """Move to `state`, or raise if that edge does not exist."""
        if state not in _LEGAL_TRANSITIONS[self.state]:
            raise IllegalTransition(
                f"request {self.request_id!r} cannot go {self.state.value} -> "
                f"{state.value}"
            )
        self.state = state

    def finish(self, reason: str) -> None:
        """Mark this request done. Its resources come back at the next schedule.

        The release is deferred rather than immediate because a request usually
        finishes in the middle of the engine's step, right after sampling, while
        its row is still being read. The scheduler reaps at a point where nothing
        is mid-forward, which keeps "when are these blocks safe to hand to someone
        else?" a question with one answer.
        """
        self.transition_to(RequestState.FINISHED)
        self.finish_reason = reason

    def append_token(self, token_id: int) -> None:
        """Record one sampled token and apply the two stopping rules.

        Stop first, then length: a request that emits EOS as its final budgeted
        token stopped, it did not run out of room. The EOS token itself is kept in
        the output because it was really sampled and the cache really holds it;
        whether to show it is the detokenizer's decision, not the scheduler's.
        """
        if self.state is not RequestState.RUNNING:
            raise IllegalTransition(
                f"cannot append to request {self.request_id!r}: it is "
                f"{self.state.value}, not running"
            )
        self.output_token_ids.append(token_id)
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self.finish("stop")
        elif self.num_output_tokens >= self.max_new_tokens:
            self.finish("length")


@dataclass(frozen=True)
class SchedulerOutput:
    """What one iteration decided: who runs, who just joined, who just left.

    scheduled: every request in this iteration's forward, in slot order.
    admitted:  the subset that joined this iteration. They still hold only their
               prompt, so they need a prefill; everything else needs one decode
               step. Day 31 uses this split, and a later day makes it the
               prefill/decode scheduling decision rather than a consequence.
    finished:  requests whose resources were released at the start of this
               iteration. This is how the engine learns an answer is ready, and it
               is where the Day-29 head-of-line delay goes to die: a row leaves at
               its own last token instead of at the batch's.

    Frozen, like `PaddedBatch`: a decision about one iteration. The next iteration
    is a new one.
    """

    scheduled: tuple[Request, ...] = ()
    admitted: tuple[Request, ...] = ()
    finished: tuple[Request, ...] = ()
    num_waiting: int = 0

    @property
    def batch_size(self) -> int:
        """Rows in this iteration's forward. Varies step to step, which is the point."""
        return len(self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def prefill(self) -> tuple[Request, ...]:
        """Rows whose whole prompt has to be run: the newly admitted ones."""
        return self.admitted

    @property
    def decode(self) -> tuple[Request, ...]:
        """Rows that only need their next token."""
        new = {id(r) for r in self.admitted}
        return tuple(r for r in self.scheduled if id(r) not in new)


class Scheduler:
    """Two queues over a block pool: what runs next, and what it costs.

    allocator:      the Day-14 `BlockAllocator`. The scheduler never touches K/V
                    tensors, only the block ids that stand for them.
    max_batch_size: how many requests may be in one forward, i.e. how many slots
                    exist. The cache is built with this many rows.

    The engine's loop is three lines and this object is the first of them:

        out = scheduler.schedule()      # reap the finished, admit what fits
        logits = model(out.scheduled)   # one forward over a batch that changed
        for r, t in zip(...): r.append_token(t)

    Everything the class does happens in `schedule`, and it happens in one order:
    release first, admit second. Reversed, a slot freed this iteration could not
    be filled until the next one, which is the bug the whole week exists to avoid.
    """

    def __init__(self, allocator: BlockAllocator, max_batch_size: int = 8):
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be at least 1, got {max_batch_size}")
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # Free slots as a heap so the lowest index is always reused. Any order is
        # correct; a deterministic one makes a scheduling trace readable and lets
        # the tests assert that a finished row's slot is the one refilled.
        self._free_slots: list[int] = list(range(max_batch_size))
        heapq.heapify(self._free_slots)
        self._by_id: dict[str, Request] = {}

    # --- queue state ----------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_unfinished(self) -> bool:
        """Whether the engine has any reason to run another iteration."""
        return bool(self.waiting or self.running)

    # --- admission ------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Accept a request onto the waiting queue. It reserves nothing yet.

        The pool-capacity check is here rather than in `schedule` because a
        request too large for an empty pool is not "waiting", it is stuck: FIFO
        admission would never get past it and every request behind it would starve
        forever. Rejecting at the door turns a hang into an error the caller can
        act on.
        """
        if request.state is not RequestState.WAITING:
            raise ValueError(
                f"only a waiting request can be queued; {request.request_id!r} is "
                f"{request.state.value}"
            )
        if request.request_id in self._by_id:
            raise ValueError(f"request id {request.request_id!r} is already known")
        need = self.allocator.blocks_for_length(request.worst_case_tokens)
        if need > self.allocator.num_blocks:
            raise KVCacheExhausted(
                f"request {request.request_id!r} needs {need} blocks for "
                f"{request.worst_case_tokens} tokens; the whole pool is "
                f"{self.allocator.num_blocks}"
            )
        self._by_id[request.request_id] = request
        self.waiting.append(request)

    def abort(self, request_id: str) -> None:
        """Finish a request early, wherever it is. Its resources come back next step."""
        request = self._by_id.get(request_id)
        if request is None:
            raise KeyError(f"unknown request {request_id!r}")
        if not request.is_finished:
            request.finish("abort")

    # --- one iteration --------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Release what finished, admit what fits, and return the next batch."""
        finished = self._release_finished()
        admitted = self._admit()
        self.running.sort(key=lambda r: r.slot)
        return SchedulerOutput(
            scheduled=tuple(self.running),
            admitted=tuple(admitted),
            finished=tuple(finished),
            num_waiting=self.num_waiting,
        )

    def _release_finished(self) -> list[Request]:
        """Take back the slots and blocks of everything that finished.

        Both queues, because a request can be aborted before it ever ran. A
        waiting request owns nothing, so dropping it from the queue is the whole
        release; a running one gives back its blocks and its slot.
        """
        finished: list[Request] = []

        still_waiting = deque(r for r in self.waiting if not r.is_finished)
        if len(still_waiting) != len(self.waiting):
            finished.extend(r for r in self.waiting if r.is_finished)
            self.waiting = still_waiting

        still_running: list[Request] = []
        for request in self.running:
            if not request.is_finished:
                still_running.append(request)
                continue
            self.allocator.free_all(request.block_ids)
            request.block_ids = []
            heapq.heappush(self._free_slots, request.slot)
            request.slot = None
            finished.append(request)
        self.running = still_running

        for request in finished:
            self._by_id.pop(request.request_id, None)
        return finished

    def _admit(self) -> list[Request]:
        """Move requests from waiting to running while a slot and blocks are free.

        The loop stops at the first request that does not fit rather than looking
        past it: see the module docstring on why FIFO beats best-fit here. The
        allocation is `allocate_for`, which is atomic, so a request either owns its
        whole worst case or is left untouched on the queue.
        """
        admitted: list[Request] = []
        while self.waiting and self._free_slots:
            request = self.waiting[0]
            if not self.allocator.can_allocate(request.worst_case_tokens):
                break
            self.waiting.popleft()
            request.block_ids = self.allocator.allocate_for(request.worst_case_tokens)
            request.slot = heapq.heappop(self._free_slots)
            request.transition_to(RequestState.RUNNING)
            self.running.append(request)
            admitted.append(request)
        return admitted

    # --- what the reservation costs -------------------------------------------

    @property
    def reserved_blocks(self) -> int:
        """Blocks the running set holds."""
        return sum(len(r.block_ids) for r in self.running)

    @property
    def used_blocks(self) -> int:
        """Blocks the running set's real tokens actually need right now."""
        return sum(self.allocator.blocks_for_length(r.num_tokens) for r in self.running)

    @property
    def reservation_waste(self) -> float:
        """Share of the reservation holding no tokens: the price of no preemption.

        High right after a prefill and falling as the requests generate, which is
        the shape to expect: the reservation is sized by the budget and the tokens
        arrive one per step. It is the cost of admitting on the worst case, and it
        is the number Week 9's incremental allocation has to beat. Zero with
        nothing running, which is a statement about an empty pool rather than a
        perfect one.
        """
        reserved = self.reserved_blocks
        return (reserved - self.used_blocks) / reserved if reserved else 0.0
