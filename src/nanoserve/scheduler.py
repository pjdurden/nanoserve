"""Scheduler: static then continuous batching. Weeks 7-9.

Week 7 batched a fixed set of sequences and Day 29 measured what that cost. The
numbers are the reason this module exists. Seven rows of 4 tokens batched behind
one row of 32: the decode issued 248 tokens and collected 52, so 79% of the
forward computed tokens for rows that were already done, and a short row's answer
was handed back at 6.6x its own latency because a static batch returns every row
when its slowest row finishes. Both bills come from the same structural fact, that
the batch is a rectangle chosen once, and a row is an index into it.

Week 8 replaced the index with an object. A `Request` has a *state*, so it leaves
the batch the moment it finishes and a waiting request takes the slot it gave up.

  waiting  --admit-->  running  --stop/length-->  finished
     ^  \\                 |                          ^
     |   \\--------------- | ---- abort --------------/
     \\------ preempt -----/

Two resources gate admission, and keeping them separate is most of the design:

  1. **A slot.** The batched cache is built with `batch_size` rows, so at most
     that many requests can be in one forward. A slot is a row index, handed out
     at admission and returned at finish. Lowest free slot wins, so slots are
     recycled rather than left as holes and the tests can name them.
  2. **KV blocks.** A request needs somewhere to put its K/V before it can run a
     single step, and this is the resource that actually runs out.

Day 33 changes how the second one is bought, which is the whole of Week 9. Week 8
admitted by reserving `worst_case_tokens`, the prompt plus every token the request
was still allowed to emit, and that made a running request unkillable: it already
owned every block it could ever need, so no forward could fail. It also meant a
request that stopped at token 3 of a possible 200 held ~99% of its reservation for
nothing, which is concurrency the pool could have sold to somebody else.

Now the rule is one line and it holds for every request in every state:

    a request must hold `blocks_for_length(num_tokens)` blocks before it forwards

Admission buys that many, and the top of every iteration tops each running request
up by the (at most one) block its next token needs. Nothing is held for a maybe,
so `reservation_waste` is 0.0 by construction and what is left is
`fragmentation_waste`, the unfilled tail of each sequence's last block, which is
bounded by `block_size - 1` tokens per sequence and is the standard paging bound.

The price is the failure the worst case ruled out. Growth can find the pool dry,
so the RUNNING -> WAITING edge Day 30 deliberately left out of the transition table
now exists: **preemption by recompute**. A victim's blocks go back to the pool, its
slot goes back to the free list, its *tokens are kept*, and it returns to the head
of the waiting queue to be prefilled again later over prompt-plus-generated. The
K/V is thrown away and bought a second time; the text is not. vLLM's other option
is swapping those blocks out to host memory and back, which pays PCIe instead of
FLOPs and needs a second pool to swap into; recompute is the simpler policy and
the one that fits a single-GPU engine, so it is the one here.

Two rules keep that safe:

  * **Victims are the youngest running request.** Seniority, not size. The oldest
    request can therefore never be preempted, so it always reaches its own last
    token, so it always frees its blocks: somebody finishing is what makes the
    pool drain, and it is the only reason this loop cannot livelock. Preempting
    the largest or the most expensive would utilise the pool slightly better and
    would let two requests take turns evicting each other forever.
  * **A request too large for an empty pool is refused at the door.** That check
    was about a blocked FIFO head on Day 30; it is now also what makes
    self-preemption terminate, since a request that has evicted everyone else
    holds the whole pool and must be able to fit in it.

Admission is FIFO and does not skip over a blocked head. If the request at the
front cannot fit, scheduling stops there for this iteration even when something
smaller behind it would fit. Skipping ahead raises utilisation and starves large
requests indefinitely, which trades a visible queue for an invisible one, and a
serving stack that never runs the big prompt is worse than one that makes small
prompts wait a step.

There is no forward pass in this file, on purpose. Everything here is bookkeeping
over integers and object states, which is the part that is subtly wrong when it is
wrong, so it is tested without a model the same way the Day-14 allocator and the
Day-29 timing core were. Day 31 wires it to a model; `Engine` supplies the
`on_release` hook that empties a cache row when its slot comes back.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .cache import BlockAllocator, KVCacheExhausted


class RequestState(Enum):
    """Where a request is in its life. The whole state space, and it is small.

    WAITING:  accepted by the engine, owns no slot and no blocks, costs nothing
              but a queue entry. A preempted request is back here, holding tokens
              whose K/V no longer exists.
    RUNNING:  holds a slot and the blocks its current tokens need, is in every
              forward until it finishes or is preempted.
    FINISHED: terminal. Its resources are released at the next scheduling step and
              its tokens are handed back to the caller.

    RUNNING -> WAITING is Day 33's edge, and it is the only one that destroys work
    rather than recording it: the request keeps its tokens and loses their K/V.
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
# else: no self-loops (a no-op transition is a caller bug, not a nothing) and no
# resurrection from FINISHED. RUNNING -> WAITING opened on Day 33, when preemption
# gave it a meaning and a recompute path to make it safe.
_LEGAL_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.WAITING: frozenset({RequestState.RUNNING, RequestState.FINISHED}),
    RequestState.RUNNING: frozenset({RequestState.WAITING, RequestState.FINISHED}),
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
    # Assigned at admission, returned at finish or at preemption. `slot` is the row
    # this request occupies in the batched cache; `block_ids` is what it holds of
    # the pool, which grows a block at a time rather than being reserved up front.
    slot: int | None = None
    block_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    # Set by the scheduler when the request is queued. Seniority: it decides who is
    # admitted first and, inverted, who is preempted first.
    arrival: int | None = None
    num_preemptions: int = 0

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

        Week 8 reserved this much at admission. Day 33 does not, and the number
        survives for one job: the door check in `Scheduler.add_request`, which
        refuses a request that could outgrow an empty pool. That check is what
        makes preemption terminate, since a request that has evicted every other
        one is alone with the whole pool and has to fit in it.
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

    def preempt(self) -> None:
        """Go back to the queue and lose the K/V, but not the text. Day 33.

        The recompute policy in one method. `output_token_ids` is untouched, so the
        request resumes as a longer version of itself: its next prefill runs over
        `token_ids` (prompt plus what it has already emitted) and samples the token
        that would have come next anyway. Nothing about the answer changes, which
        is the only property that makes a memory policy allowed to exist.

        The blocks and the slot are not touched here either. They belong to the
        scheduler, which frees them around this call, for the same reason `finish`
        does not free them: one component talks to the allocator.
        """
        self.transition_to(RequestState.WAITING)
        self.num_preemptions += 1

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
    admitted:  the subset that joined this iteration. They hold their tokens and
               no K/V, so they need a prefill; everything else needs one decode
               step. Day 31 uses this split, and a later day makes it the
               prefill/decode scheduling decision rather than a consequence.
    finished:  requests whose resources were released at the start of this
               iteration. This is how the engine learns an answer is ready, and it
               is where the Day-29 head-of-line delay goes to die: a row leaves at
               its own last token instead of at the batch's.
    preempted: requests pushed back to the queue this iteration to free blocks for
               an older one. Reported rather than kept quiet because it is the one
               event that costs the caller latency for somebody else's benefit,
               and because a rising count is the signal that the pool is too small
               for the offered load.

    Frozen, like `PaddedBatch`: a decision about one iteration. The next iteration
    is a new one.
    """

    scheduled: tuple[Request, ...] = ()
    admitted: tuple[Request, ...] = ()
    finished: tuple[Request, ...] = ()
    preempted: tuple[Request, ...] = ()
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
    watermark:      the share of the pool admission refuses to spend. See
                    `watermark_blocks`.
    on_release:     called with a slot index whenever that slot goes back to the
                    free list, whether the request finished or was preempted. The
                    scheduler owns slot lifetime and has no tensors; the engine
                    passes its cache-row reset here so the row is emptied while
                    the scheduler still knows whose it was and before anybody else
                    can be handed it.

    The engine's loop is three lines and this object is the first of them:

        out = scheduler.schedule()      # reap, grow, preempt, admit
        logits = model(out.scheduled)   # one forward over a batch that changed
        for r, t in zip(...): r.append_token(t)

    Everything the class does happens in `schedule`, and it happens in one order:
    release, then grow, then admit. Release first because a slot freed this
    iteration must be fillable in this iteration, which is the bug Week 8 exists to
    avoid. Grow before admit because a running request's next token has already
    been sampled and has nowhere to go, while a waiting request has waited before
    and can wait again: admitting into the block an older row needs turns admission
    itself into a preemption.
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        max_batch_size: int = 8,
        watermark: float = 0.01,
        on_release: Callable[[int], None] | None = None,
    ):
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be at least 1, got {max_batch_size}")
        if not 0.0 <= watermark < 1.0:
            raise ValueError(f"watermark must be in [0, 1), got {watermark}")
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        self.on_release = on_release
        # Blocks admission leaves on the table. A newcomer that takes the pool's
        # last block is a preemption waiting to happen: the running rows are one
        # token from a boundary and the newcomer is the youngest, so it is the
        # first victim, and the engine pays a prefill and a recompute to have
        # changed nothing. A small brake here converts that thrash into a wait.
        # Growth ignores it: it is an admission policy, not a reserve, and a
        # running row must never be starved by an accounting rule. vLLM keeps the
        # same 1% default for the same reason.
        self.watermark_blocks = int(watermark * allocator.num_blocks)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # Free slots as a heap so the lowest index is always reused. Any order is
        # correct; a deterministic one makes a scheduling trace readable and lets
        # the tests assert that a finished row's slot is the one refilled.
        self._free_slots: list[int] = list(range(max_batch_size))
        heapq.heapify(self._free_slots)
        self._by_id: dict[str, Request] = {}
        self._arrivals = 0
        # What preemption has cost, in the two currencies it is paid in.
        self.num_preemptions = 0
        self.preempted_tokens = 0

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
        request.arrival = self._arrivals
        self._arrivals += 1
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

    def blocks_needed_for(self, request: Request) -> int:
        """Blocks this request must be given before it may forward again.

        The one allocation rule in the file, and it does not care what state the
        request is in or how it got there. A request needs somewhere to put the K/V
        of every token it holds, so it needs `blocks_for_length(num_tokens)` blocks,
        and it needs however many of those it does not already have. For a fresh
        request that is its whole prompt; for a running one it is 0 on most steps
        and 1 on the step that crosses a block boundary; for a preempted one coming
        back it is prompt plus everything it generated before it lost its blocks.

        Never more than 1 for a running request, because a running request adds
        exactly one token per iteration. That bound is why the growth loop can
        preempt one victim at a time and know it is making progress.
        """
        return self.allocator.blocks_for_length(request.num_tokens) - len(request.block_ids)

    def schedule(self) -> SchedulerOutput:
        """Reap, grow (preempting if the pool is dry), admit, and hand back the batch."""
        finished = self._release_finished()
        preempted = self._grow_running()
        admitted = self._admit()
        self.running.sort(key=lambda r: r.slot)
        return SchedulerOutput(
            scheduled=tuple(self.running),
            admitted=tuple(admitted),
            finished=tuple(finished),
            preempted=tuple(preempted),
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
            self._release_resources(request)
            finished.append(request)
        self.running = still_running

        for request in finished:
            self._by_id.pop(request.request_id, None)
        return finished

    def _release_resources(self, request: Request) -> None:
        """Give a running request's blocks and slot back. Finish and preempt share it.

        Blocks first, then the slot, and the slot through `on_release` so the cache
        row is emptied before the free list can hand that index to anybody else. The
        two failure modes on either side of this call are both silent: a block
        returned twice corrupts the pool, and a row handed on dirty makes its next
        tenant attend over a stranger's K/V.
        """
        self.allocator.free_all(request.block_ids)
        request.block_ids = []
        slot, request.slot = request.slot, None
        if slot is not None:
            if self.on_release is not None:
                self.on_release(slot)
            heapq.heappush(self._free_slots, slot)

    def _grow_running(self) -> list[Request]:
        """Top every running request up to the blocks its next token needs.

        Oldest first, and the victim is always the youngest still to be considered,
        so the request being grown is never younger than the one paying for it. The
        loop is short because `blocks_needed_for` is at most 1 here: a request adds
        one token per iteration, so at worst it wants one more block, and freeing
        one victim releases at least one block. Progress is therefore guaranteed on
        every pass.

        The last case is the one that looks strange and is not: a request with
        nobody younger to evict preempts *itself*. It is the honest outcome. There
        is no block for its next token and no one to take one from, so the only
        alternatives are crashing the engine or writing that token over somebody
        else's K/V. Going back to the queue costs it a recompute and costs the
        older requests nothing, and it terminates, because the oldest request can
        never reach this branch: everyone else would have been preempted first,
        leaving it alone with a pool that `add_request` already proved big enough
        for it.
        """
        preempted: list[Request] = []
        pending = deque(sorted(self.running, key=lambda r: r.arrival))
        while pending:
            request = pending.popleft()
            need = self.blocks_needed_for(request)
            while need > self.allocator.num_free:
                victim = pending.pop() if pending else request
                self._preempt(victim)
                preempted.append(victim)
                if victim is request:
                    break
            if request.state is RequestState.RUNNING and need > 0:
                request.block_ids.extend(
                    self.allocator.allocate_for(need * self.allocator.block_size)
                )
        return preempted

    def _preempt(self, request: Request) -> None:
        """Evict a running request by recompute: blocks and slot back, tokens kept.

        It returns to the *head* of the waiting queue rather than the back. It is
        older than everything already queued (admission is FIFO, so anything still
        waiting arrived after it), it has work invested that the queue behind it
        does not, and sending it to the back would let a newcomer take the blocks it
        was just evicted from, which is a starvation loop rather than a policy.
        Preempting youngest first and pushing each victim onto the front leaves the
        queue in arrival order, which is the invariant admission assumes.
        """
        self.num_preemptions += 1
        self.preempted_tokens += request.num_tokens
        self._release_resources(request)
        request.preempt()
        self.running.remove(request)
        self.waiting.appendleft(request)

    def _admit(self) -> list[Request]:
        """Move requests from waiting to running while a slot and blocks are free.

        The loop stops at the first request that does not fit rather than looking
        past it: see the module docstring on why FIFO beats best-fit here. The
        allocation is `allocate_for`, which is atomic, so a request either holds
        every block its tokens need or is left untouched on the queue.

        The quantity is `blocks_needed_for`, not the worst case, which is the Day-33
        change and the reason a pool that fit one request now fits four. A resumed
        request pays for its generated tokens here too: it is admitted as a longer
        version of itself.
        """
        admitted: list[Request] = []
        while self.waiting and self._free_slots:
            request = self.waiting[0]
            need = self.blocks_needed_for(request)
            if need > self.allocator.num_free - self.watermark_blocks:
                break
            self.waiting.popleft()
            request.block_ids = self.allocator.allocate_for(need * self.allocator.block_size)
            request.slot = heapq.heappop(self._free_slots)
            request.transition_to(RequestState.RUNNING)
            self.running.append(request)
            admitted.append(request)
        return admitted

    # --- what the pool is really holding --------------------------------------

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
        """Share of the held blocks holding no tokens: 0.0 since Day 33.

        Week 8's headline cost, kept as a measurement rather than deleted as a
        solved problem. Admission bought `worst_case_tokens`, so a request that
        stopped at token 3 of a possible 200 held ~99% of its blocks for text it
        never generated. Incremental allocation buys exactly what the tokens need,
        so held and used are the same number and this is zero by construction.

        Clamped at zero rather than allowed to go negative: between a sample and
        the next iteration a request holds one token more than its blocks cover,
        and the scheduler closes that gap at the top of the next `schedule` before
        anything forwards. What is left after this number is
        `fragmentation_waste`, which paging cannot remove.
        """
        reserved = self.reserved_blocks
        if not reserved:
            return 0.0
        return max(0.0, (reserved - self.used_blocks) / reserved)

    @property
    def fragmentation_waste(self) -> float:
        """Share of the *token slots* in held blocks that hold no token.

        The waste that survives incremental allocation: a block is the unit of
        allocation, so a sequence of 17 tokens with `block_size` 16 holds two blocks
        and leaves 15 slots empty until it grows into them. Internal fragmentation,
        bounded by `block_size - 1` tokens per sequence however long the sequence
        gets, which is why vLLM quotes a few percent for a real block size and a
        real workload, and why the block size is a tuning knob rather than a
        constant: larger blocks mean fewer tables and more dead tail.
        """
        capacity = self.reserved_blocks * self.allocator.block_size
        if not capacity:
            return 0.0
        live = sum(r.num_tokens for r in self.running)
        return max(0.0, (capacity - live) / capacity)
