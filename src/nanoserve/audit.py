"""The engine's invariants, written down and checked every iteration. Day 35.

Week 9 gave the scheduler the right to take a running request's blocks away and
hand them to somebody else (Day 33), and then priced what that costs (Day 34).
Both days were tested the way a feature is tested: script a situation, assert the
outcome. That style cannot find the failures preemption actually has, because all
of them are *accumulative*. One block that goes out of the pool and never comes
back does not change a single answer; it makes the pool one block smaller, forever,
and the run that finally notices is some other run, much later, that preempts more
than it should and cannot say why.

So this module writes the invariants down instead, as four questions asked of the
engine's whole state at once:

  * **The pool is a closed ledger.** Every block id is free, or held by exactly one
    request, and never both and never neither. `audit_pool`.
  * **A slot is a cache row, and there are exactly `max_batch_size` of them.** Free
    or held by one running request, same rule. `audit_slots`.
  * **A request holds what its state entitles it to.** Waiting means nothing
    reserved; running means a slot and exactly the blocks its current tokens need.
    The queues are in arrival order, which is the thing FIFO admission assumes and
    never checks. `audit_requests`.
  * **A cache row addresses its tenant's blocks and nobody else's.** The row's
    table is a prefix of the request's reservation, and the K/V it holds is one
    token behind the request's text. `audit_rows`.

`soak` is those checks in a loop around `Engine.step`, which is the Week-9 stress
test: far more requests than slots, through a pool far too small for them, run
until requests have been evicted and resumed several times each. What makes it a
test rather than a demo is that the audit runs *between* iterations, so a violation
is reported by the iteration that caused it instead of by the symptom three hundred
iterations later.

**The observation point matters and is part of the contract.** These checks are
true of a settled engine: after a `step` has completed, or before the first one.
They are deliberately not true in the middle of a `schedule`, where a slot is taken
from one request and given to another, nor between a schedule and its forward,
where a newly admitted request holds blocks its cache row has not adopted yet. That
is not a weakness of the invariants, it is what an invariant is: the state a system
returns to, not a state it never leaves.

Two consequences of that show up as slack in the checks below, and both are real
work in flight rather than sloppiness:

  * A running request's blocks must cover `num_tokens - 1`, not `num_tokens`,
    because between a sample and the next `schedule` a request holds one token
    more than its blocks do. The scheduler closes that gap before anything
    forwards.
  * A cache row may be one block behind its request's reservation, because the
    scheduler grows the request at the top of an iteration and the engine syncs the
    row just before the write that needs it.

The pool ledger has no such slack, which is exactly why it is the check that finds
things.

This is a debug and test tool, not something the engine does in production: every
call is O(blocks + slots + requests) and the whole point is to run it far more
often than a real server could afford. vLLM keeps the same kind of checking behind
a flag for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .scheduler import Request, RequestState, Scheduler


class InvariantViolation(RuntimeError):
    """The engine's state contradicts something that must always be true.

    A separate exception from `IllegalTransition` and `KVCacheExhausted` because it
    means something different. Those two are refusals: a caller asked for something
    impossible and nothing bad happened. This one is a discovery, and by the time it
    is raised the damage is already in the state being described. The message names
    the object and both sides of the contradiction, since "an invariant failed" with
    no id in it sends you back to the same three hundred iterations this module
    exists to avoid reading.
    """


# --- the pool -----------------------------------------------------------------


def _holders(scheduler: Scheduler) -> list[Request]:
    """Every request that could be holding pool blocks, in both queues.

    Waiting requests are included even though a waiting request must hold nothing:
    if one does, that is the violation, and gathering it here is what turns "a
    block vanished" into "request r4 is waiting and still holding block 7".
    """
    return list(scheduler.running) + list(scheduler.waiting)


def audit_pool(scheduler: Scheduler) -> None:
    """Every block is free, or held by exactly one request. Never both, never neither.

    The Day-14 allocator keeps two halves of this itself (a free list and an
    allocated set) and guards the transitions between them. What it cannot see is
    the third party: the requests. A block can be perfectly allocated as far as the
    pool is concerned and owned by nobody at all, which is the stranded block, the
    only pool bug that cannot be recovered from at runtime and the one preemption
    makes easy to write, since a preemption frees blocks the request keeps the ids
    of and a finish frees ids the cache row is still addressing.

    Checked in the order the messages are most useful, not the order the sets were
    built: two owners first, because that is the one that corrupts an answer rather
    than a count.
    """
    allocator = scheduler.allocator
    held: dict[int, str] = {}
    for request in _holders(scheduler):
        for block_id in request.block_ids:
            owner = held.get(block_id)
            if owner is not None:
                raise InvariantViolation(
                    f"block {block_id} is held by both {owner!r} and "
                    f"{request.request_id!r}: two sequences writing K/V to one place"
                )
            held[block_id] = request.request_id

    free = allocator.free_blocks
    free_set = set(free)
    if len(free_set) != len(free):
        duplicated = next(b for b in free if free.count(b) > 1)
        raise InvariantViolation(
            f"block {duplicated} is free twice: the pool will hand it to two requests"
        )

    allocated = allocator.allocated_blocks
    both = free_set & allocated
    if both:
        raise InvariantViolation(
            f"the allocator has block {min(both)} free and allocated at once"
        )

    for block_id, owner in sorted(held.items()):
        if block_id in allocated:
            continue
        if block_id in free_set:
            raise InvariantViolation(
                f"block {block_id} is free and held at once: {owner!r} is using it and "
                "the pool is about to hand it to somebody else"
            )
        raise InvariantViolation(
            f"request {owner!r} holds block {block_id}, which this pool has no record of"
        )

    stranded = allocated - held.keys()
    if stranded:
        raise InvariantViolation(
            f"block {min(stranded)} is allocated but no request holds it: it left the "
            f"pool and nothing will ever return it ({len(stranded)} in total)"
        )

    accounted = free_set | allocated
    for block_id in range(allocator.num_blocks):
        if block_id not in accounted:
            raise InvariantViolation(
                f"block {block_id} is neither free nor held by anybody: the pool is "
                "smaller than it thinks it is"
            )
    foreign = accounted - set(range(allocator.num_blocks))
    if foreign:
        raise InvariantViolation(
            f"block {min(foreign)} is in the ledger but is not a block in this pool"
        )


# --- the slots ----------------------------------------------------------------


def audit_slots(scheduler: Scheduler) -> None:
    """The same closed-ledger rule for slots, which are cache rows.

    Blocks are the resource that runs out; slots are the resource that is *reused*,
    and the two failures are not the same shape. A leaked block costs capacity. A
    leaked slot costs concurrency, quietly, since the engine simply never fills that
    row again. A slot handed to two requests at once costs correctness immediately:
    both of them address the same cache row, and the second one's decode attends
    over the first one's K/V.
    """
    held: dict[int, str] = {}
    for request in scheduler.running:
        if request.slot is None:
            raise InvariantViolation(
                f"request {request.request_id!r} is running with no slot: it has no "
                "cache row to put its K/V in"
            )
        owner = held.get(request.slot)
        if owner is not None:
            raise InvariantViolation(
                f"slot {request.slot} is held by both {owner!r} and "
                f"{request.request_id!r}: one cache row, two tenants"
            )
        held[request.slot] = request.request_id

    free = scheduler.free_slots
    free_set = set(free)
    if len(free_set) != len(free):
        duplicated = next(s for s in free if free.count(s) > 1)
        raise InvariantViolation(f"slot {duplicated} is free twice")

    both = free_set & held.keys()
    if both:
        slot = min(both)
        raise InvariantViolation(
            f"slot {slot} is free and held at once: {held[slot]!r} is running in a row "
            "the scheduler is about to hand to somebody else"
        )

    for slot in range(scheduler.max_batch_size):
        if slot not in free_set and slot not in held:
            raise InvariantViolation(
                f"slot {slot} is neither free nor held: a cache row that will never be "
                "used again"
            )


# --- the requests --------------------------------------------------------------


def audit_requests(scheduler: Scheduler) -> None:
    """What each request is holding, against what its state entitles it to hold.

    Three families, and the third is the one nothing else checks.

    **States and resources.** WAITING costs a queue entry and nothing else, which is
    the property that makes a preempted request cheap and is the property a
    preemption that forgets to free breaks. RUNNING owns a slot and the blocks its
    current tokens need. FINISHED is allowed in either queue, holding anything: it
    is the one legal in-between, since a request finishes in the middle of a step
    and is reaped at the top of the next one.

    **Blocks against tokens**, in both directions. Too few is the write that lands
    in somebody else's block. Too many is Day 33's whole claim (`reservation_waste`
    is 0.0 by construction) turned into something that can fail. The lower bound
    carries one token of slack for the sample that has not been grown into yet; see
    the module docstring.

    **Arrival order**, which is the invariant FIFO admission assumes and never
    states. `_admit` only ever looks at `waiting[0]`, so if the queue is not in
    arrival order the scheduler is not FIFO any more and nothing raises: it just
    starves somebody. Preemption is what makes this checkable and worth checking,
    because it is the only thing that puts a request back on the queue, and it puts
    it at the *head*. The two halves are that the queue itself ascends, and that
    nothing waiting is older than anything running, which is what "does not skip a
    blocked head" means from the outside.
    """
    allocator = scheduler.allocator
    known = scheduler.known_ids

    for request in scheduler.running:
        if request.state is RequestState.WAITING:
            raise InvariantViolation(
                f"request {request.request_id!r} is in the running set but is waiting"
            )
        need = allocator.blocks_for_length(request.num_tokens)
        holds = len(request.block_ids)
        if holds > need:
            raise InvariantViolation(
                f"request {request.request_id!r} holds {holds} more blocks than its "
                f"{request.num_tokens} tokens need ({need}): a reservation for a maybe"
            )
        if holds * allocator.block_size < request.num_tokens - 1:
            raise InvariantViolation(
                f"request {request.request_id!r} holds {request.num_tokens} tokens but "
                f"its {holds} block covers only {holds * allocator.block_size} of them"
            )

    for request in scheduler.waiting:
        if request.state is RequestState.RUNNING:
            raise InvariantViolation(
                f"request {request.request_id!r} is in the waiting queue but is running"
            )
        if request.block_ids:
            raise InvariantViolation(
                f"request {request.request_id!r} is waiting but holds "
                f"{len(request.block_ids)} blocks: waiting is supposed to cost nothing"
            )
        if request.slot is not None:
            raise InvariantViolation(
                f"request {request.request_id!r} is waiting but holds slot "
                f"{request.slot}: a cache row nobody can use"
            )

    arrivals = [r.arrival for r in scheduler.waiting]
    for earlier, later in zip(arrivals, arrivals[1:]):
        if earlier > later:
            raise InvariantViolation(
                f"the waiting queue is out of arrival order ({earlier} before {later}): "
                "admission only ever looks at the head, so this is a starved request"
            )
    if scheduler.running and scheduler.waiting:
        youngest_running = max(r.arrival for r in scheduler.running)
        oldest_waiting = min(arrivals)
        if oldest_waiting < youngest_running:
            raise InvariantViolation(
                f"a request that arrived at {youngest_running} is running while one that "
                f"arrived at {oldest_waiting} waits: somebody jumped the queue"
            )

    queued = {r.request_id for r in _holders(scheduler)}
    missing = queued - known
    if missing:
        raise InvariantViolation(
            f"request {min(missing)!r} is queued but not in the id index: it cannot be "
            "aborted and its id can be handed out again"
        )
    stale = known - queued
    if stale:
        raise InvariantViolation(
            f"the id index still has {min(stale)!r}, which is in neither queue"
        )


def audit_scheduler(scheduler: Scheduler) -> None:
    """Every check that needs no tensors: pool, slots, requests."""
    audit_pool(scheduler)
    audit_slots(scheduler)
    audit_requests(scheduler)


# --- the cache rows -------------------------------------------------------------


def audit_rows(engine) -> None:
    """Each cache row addresses its tenant's blocks, and only its tenant's tokens.

    The tensor half of the slot ledger, and the one that catches the failure Day 31
    named and Day 33 made frequent: a row handed on without being emptied. Nothing
    about that is detectable from the answer, which stays fluent, plausible and
    wrong, so it has to be detectable from the state.

    Three rules per row. A *free* slot's row is empty, because the next tenant
    adopts into it and an adoption onto a dirty table is a stranger's history. A
    *held* row's block list is a prefix of its request's reservation, allowed to be
    one block short (the growth the engine syncs just before it writes) and never
    one block long or one block different, since a block the request does not hold
    is a block somebody else does. And a held row's token count is behind its
    request's, never ahead: the cache is written before the token is sampled, so it
    trails the text by exactly one, and a row holding more tokens than its tenant
    has is holding somebody else's.
    """
    cache = engine.cache
    block_size = cache.block_size
    tenants = {r.slot: r for r in engine.scheduler.running if r.slot is not None}

    for slot, table in enumerate(cache.tables):
        request = tenants.get(slot)
        if request is None:
            if table.block_ids or table.num_tokens:
                raise InvariantViolation(
                    f"slot {slot} is free but its cache row still holds "
                    f"{table.num_tokens} tokens in {len(table.block_ids)} blocks: the "
                    "next request admitted here would attend over them"
                )
            continue

        holds = len(table.block_ids)
        if table.block_ids != request.block_ids[:holds]:
            raise InvariantViolation(
                f"the cache row for {request.request_id!r} addresses "
                f"{table.block_ids} which does not match the blocks it holds, "
                f"{request.block_ids}"
            )
        behind = len(request.block_ids) - holds
        if behind > 1:
            raise InvariantViolation(
                f"the cache row for {request.request_id!r} is {behind} blocks behind "
                "its reservation: at most one block of growth can be in flight"
            )
        if table.num_tokens > request.num_tokens:
            raise InvariantViolation(
                f"the cache row for {request.request_id!r} holds {table.num_tokens} "
                f"tokens, ahead of the request's {request.num_tokens}: K/V for text "
                "this request never had"
            )
        if table.num_tokens > holds * block_size:
            raise InvariantViolation(
                f"the cache row for {request.request_id!r} holds {table.num_tokens} "
                f"tokens in {holds} blocks, which is {holds * block_size} slots"
            )


def audit_engine(engine) -> None:
    """Every invariant there is: the scheduler's bookkeeping and the cache's rows."""
    audit_scheduler(engine.scheduler)
    audit_rows(engine)


# --- the stress test ------------------------------------------------------------


@dataclass(frozen=True)
class SoakReport:
    """What a soak did, and what the pressure cost, in one object.

    Deliberately not a timing. Day 34 already measures what preemption costs in
    seconds and in recomputed positions; this reports whether the engine survived
    it, and the numbers here exist to prove the run was hard enough to be worth
    believing. A soak that reports `max_preemptions` of 0 has tested nothing but
    the happy path with extra steps, which is why the tests assert on it.

    iterations:       how many `step`s it took to drain.
    audits:           how many times the whole state was checked. With the default
                      `audit_every=1` this is `iterations + 1`: every step, plus the
                      state before any of them.
    texts:            request id to prompt-plus-generation. The soak does not know
                      what the right answer is; the caller compares this against the
                      same workload through a pool big enough never to preempt.
    finish_order:     ids in the order they completed, which under pressure is not
                      submission order.
    preemptions:      request id to how many times it was evicted and resumed.
    """

    iterations: int
    audits: int
    num_blocks: int
    free_blocks: int
    max_batch_size: int
    free_slots: int
    finish_order: tuple[str, ...] = ()
    texts: dict[str, list[int]] = field(default_factory=dict)
    preemptions: dict[str, int] = field(default_factory=dict)
    forward_tokens: int = 0
    recomputed_tokens: int = 0

    @property
    def num_requests(self) -> int:
        return len(self.texts)

    @property
    def num_preemptions(self) -> int:
        return sum(self.preemptions.values())

    @property
    def max_preemptions(self) -> int:
        """The worst-treated request's eviction count. The stress test's own thermometer."""
        return max(self.preemptions.values(), default=0)

    @property
    def resumed_requests(self) -> int:
        return sum(1 for n in self.preemptions.values() if n)

    @property
    def resumed_fraction(self) -> float:
        return self.resumed_requests / self.num_requests if self.num_requests else 0.0

    @property
    def recompute_fraction(self) -> float:
        """Day 34's number for this run: the share of forward positions bought twice."""
        return self.recomputed_tokens / self.forward_tokens if self.forward_tokens else 0.0

    @property
    def pool_returned(self) -> bool:
        """Every block back in the pool at the end. The claim a whole run makes at once."""
        return self.free_blocks == self.num_blocks

    @property
    def slots_returned(self) -> bool:
        return self.free_slots == self.max_batch_size


def soak(
    engine,
    requests,
    *,
    max_iterations: int = 10_000,
    audit_every: int = 1,
) -> SoakReport:
    """Drive an engine to empty under pressure, auditing the whole state as it goes.

    The Week-9 stress test as a function, so a test can say what it wants (twelve
    requests, three slots, ten blocks) instead of writing the loop again. It submits
    every request, steps until both queues are empty, and calls `audit_engine`
    between iterations. Three things can stop it, and all three are the point:

      * an invariant violated, raised by the iteration that caused it;
      * an iteration that made no progress at all, which is a livelock and would
        otherwise be indistinguishable from a slow run until `max_iterations`;
      * the drain guard itself, which is a hang caught rather than a hang.

    Progress is deliberately loose: a token collected, a request finished, or a
    request preempted. Preemption counts because an engine thrashing its pool is
    still moving, just badly, and the number that says so is `max_preemptions`, not
    an exception. What is *not* progress is an iteration that scheduled nothing and
    released nothing while work was still queued, which is the deadlock shape: a
    waiting queue whose head cannot fit and nobody running to free anything.

    `audit_every` is the cost knob. Checking every iteration is O(pool) per step and
    that is the right default for a test; a longer soak can check every k-th and
    still get the final state, since the drain always audits what it ends with.
    """
    if audit_every < 1:
        raise ValueError(f"audit_every must be at least 1, got {audit_every}")

    requests = list(requests)
    for request in requests:
        engine.add_request(request)

    audit_engine(engine)
    audits = 1
    iterations = 0
    finish_order: list[str] = []
    collected = engine.collected_tokens

    while engine.has_unfinished():
        if iterations >= max_iterations:
            raise InvariantViolation(
                f"the soak did not drain in {max_iterations} iterations: "
                f"{engine.scheduler.num_running} running, "
                f"{engine.scheduler.num_waiting} waiting"
            )
        out = engine.step()
        iterations += 1
        finish_order.extend(r.request_id for r in out.finished)
        moved = engine.collected_tokens > collected or out.finished or out.preempted
        collected = engine.collected_tokens
        if not moved:
            raise InvariantViolation(
                f"iteration {iterations} made no progress: no token collected, nothing "
                f"finished, nothing preempted, and {engine.scheduler.num_waiting} "
                f"requests still waiting"
            )
        if iterations % audit_every == 0:
            audit_engine(engine)
            audits += 1

    if iterations % audit_every:
        audit_engine(engine)
        audits += 1

    unfinished = [r.request_id for r in requests if not r.is_finished]
    if unfinished:
        raise InvariantViolation(
            f"the engine has nothing left to do but {len(unfinished)} requests never "
            f"finished, starting with {unfinished[0]!r}"
        )

    return SoakReport(
        iterations=iterations,
        audits=audits,
        num_blocks=engine.allocator.num_blocks,
        free_blocks=engine.allocator.num_free,
        max_batch_size=engine.scheduler.max_batch_size,
        free_slots=len(engine.scheduler.free_slots),
        finish_order=tuple(finish_order),
        texts={r.request_id: list(r.token_ids) for r in requests},
        preemptions={r.request_id: r.num_preemptions for r in requests},
        forward_tokens=engine.forward_tokens,
        recomputed_tokens=engine.recomputed_tokens,
    )
