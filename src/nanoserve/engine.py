"""The engine: the scheduler's decisions turned into forward passes. Weeks 8-9.

Day 30 built the half that thinks. `Scheduler` reaps what finished, admits what
fits, and hands back a `SchedulerOutput` saying who is in this iteration's batch,
who just joined and who just left. It has no model in it and no tensors, which is
why it could be tested over plain integers. This file is the half that runs. One
`step()` is:

    out = scheduler.schedule()      # who runs this iteration
    prefill(out.admitted)           # the ones that arrived: their whole prompt
    decode(out.decode)              # the ones already running: one token each
    for r, t in ...: r.append_token(t)

and `run_to_completion` is that in a loop. The interesting part is not the loop, it
is that **the batch changes shape every iteration**, and three things follow that
Week 7's static path never had to answer.

**A row is a slot, and a slot is reused.** The cache is built with
`max_batch_size` rows because a row index is a physical thing: it addresses a block
table. The scheduler hands out those indices as slots and takes them back, so cache
row 3 belongs to whichever request currently holds slot 3, and over a long run that
is many different requests. Which makes the reset load-bearing. When a request
finishes, its row has to be emptied before the next one lands on it, or the new
tenant's first decode attends over the previous tenant's K/V. That failure is
silent: no shape is wrong, no index is out of range, the model simply continues
somebody else's sentence in fluent, plausible, wrong text.

**A forward covers some rows, not all of them.** Four rows this iteration, two the
next, five after that. `BatchedPagedKVCache.view(rows)` is what makes that
expressible: it presents an arbitrary subset as if it were a whole batch, so
`layers.py` needs no notion of scheduling at all. It also keeps the Day-28 guard
honest. That guard refuses a masked prefill onto a cache that already holds tokens,
which is correct and, seen across the whole cache, would refuse every prefill after
the first, since under continuous batching the other rows are always mid-generation.
Through a view it asks the right question: are *these* rows empty?

**The pool has one owner.** The scheduler holds this request's blocks, so the
cache row borrows them (`adopt_row`) instead of allocating its own. Two
reservations for one sequence would book the pool twice and the second booking
fails mid-flight. Day 33 makes that borrowing continuous rather than one-off: the
scheduler hands a running request another block whenever its next token crosses a
boundary, so before every decode the engine syncs the row's table with the
request's list (`extend_row`). The alternative, letting `BlockTable.append` reach
the allocator when it runs short, is the double-booking bug wearing a hat.

**A row can be taken away.** Preemption is the Week-9 edge, and the tensor half of
it is one rule: the cache row must be emptied at the moment its slot goes back to
the free list, not at the engine's convenience, because the very next thing the
scheduler does is hand that slot to somebody else. So the engine installs
`Scheduler.on_release` and the reset happens inside `schedule`. The request that
comes back is not a new one: it kept its tokens and lost their K/V, so its prefill
runs over `token_ids` (prompt plus everything it generated) rather than over the
prompt, and the token it samples is the one it would have sampled anyway. Every
prefill in this file uses `token_ids` for that reason, and for a request that has
never been preempted the two are the same list.

What the loop buys is Day 29's two bills, paid off and counted. `issued_tokens` is
the token-slots the forwards actually computed and `collected_tokens` is the ones a
request kept; here they are equal, so `waste_fraction` is 0.0 rather than the 79%
static batching paid, because a row is in the forward only while it still wants a
token. What it does *not* buy is on the same object and reported next to it:
`prefill_padding_waste`, because the prefill is still one padded rectangle and a
short prompt batched with a long one still pays for the difference, and
`recomputed_tokens`, the K/V a preempted request has to buy twice. Ragged (varlen)
prefill and chunked prefill are what fix the first; a bigger pool is the only thing
that fixes the second. vLLM and SGLang both run this loop; the parts still missing
here are swap-out as an alternative to recompute, prefix caching, and mixing
prefill and decode tokens into one forward.
"""

from __future__ import annotations

import torch

from .batch import last_token_logits, pad_prompts
from .cache import BatchedPagedKVCache, BlockAllocator
from .scheduler import Request, Scheduler, SchedulerOutput


class Engine:
    """Model + paged cache + scheduler, stepped one iteration at a time.

    model:     a `LlamaModel`. The engine only ever calls `forward`, and hands it a
               row view of the cache instead of the cache itself.
    scheduler: the Day-30 `Scheduler`. It owns the queues, the slots and the block
               reservations; the engine owns the tensors those decisions describe.
    cache:     a `BatchedPagedKVCache` with one row per slot, over the *same*
               allocator the scheduler admits against. Same pool, one bookkeeper.
    pad_id:    filler for the prefill rectangle. Never attended to and never
               written to the cache, so any in-vocab id works.

    The two objects are deliberately not merged. The scheduler is pure bookkeeping
    and stays testable without a GPU, a model, or a tensor; this class is the only
    place where a decision becomes a forward pass. `Engine.build` wires a matching
    pair when a caller does not want to construct the three pieces itself.
    """

    def __init__(
        self,
        model,
        scheduler: Scheduler,
        cache: BatchedPagedKVCache,
        pad_id: int = 0,
    ):
        if cache.batch_size != scheduler.max_batch_size:
            raise ValueError(
                f"the cache has {cache.batch_size} rows but the scheduler hands out "
                f"{scheduler.max_batch_size} slots: a slot is a cache row, so they "
                "must be the same number"
            )
        if cache.allocator is not scheduler.allocator:
            raise ValueError(
                "the cache and the scheduler must share one block pool: the "
                "scheduler reserves the blocks the cache rows run on"
            )
        self.model = model
        self.scheduler = scheduler
        self.cache = cache
        self.pad_id = pad_id
        # The scheduler owns slot lifetime and has no tensors; this is the tensor
        # half of releasing one. Installed rather than passed to the constructor so
        # a caller who built the pair by hand cannot forget it, and because the
        # engine is the only object that knows what a slot means physically.
        scheduler.on_release = self._release_row
        # What the run cost, in the vocabulary Day 29 measured static batching in.
        self.iterations = 0
        self.issued_tokens = 0
        self.collected_tokens = 0
        self.prefill_slots = 0
        self.prefill_tokens = 0
        self.recomputed_tokens = 0
        self._next_id = 0

    @classmethod
    def build(
        cls,
        model,
        num_blocks: int,
        block_size: int = 16,
        max_batch_size: int = 8,
        pad_id: int = 0,
    ) -> Engine:
        """Wire a scheduler and a matching cache over one fresh pool."""
        allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        return cls(
            model,
            Scheduler(allocator, max_batch_size=max_batch_size),
            BatchedPagedKVCache(model.config, allocator, batch_size=max_batch_size),
            pad_id=pad_id,
        )

    @property
    def allocator(self) -> BlockAllocator:
        return self.scheduler.allocator

    # --- what the caller does -------------------------------------------------

    def add_request(self, request: Request) -> Request:
        """Queue a request. It reserves nothing until a `step` admits it."""
        self.scheduler.add_request(request)
        return request

    def abort(self, request_id: str) -> None:
        """Stop a request wherever it is. Its row and blocks come back next step."""
        self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # --- one iteration --------------------------------------------------------

    def step(self) -> SchedulerOutput:
        """Schedule, sync the rows, forward, sample, and feed the tokens back.

        The reap is entirely inside `schedule` now. It releases the *ids* (a slot to
        the free list, blocks to the pool) and calls back into `_release_row` for
        the *tensors*, which is the only ordering that survives preemption: a slot
        can be taken from one request and given to another within a single
        `schedule` call, so a tidy-up loop before or after it would be either too
        late or too early. It covers the aborted case for free, which is the one
        that never comes back through sampling.

        Prefill and decode are two forwards here, not one. Newly admitted rows run
        their whole context through the Day-27 padded rectangle; already running
        rows run one token each through the Day-28 batched decode. Both emit exactly
        one token per row, so every scheduled request advances by one token per
        iteration whichever half it was in, and a resumed request rejoins in the
        prefill half. Mixing the two into a single flattened forward (chunked
        prefill) is a real optimisation and a later one.
        """
        out = self.scheduler.schedule()
        if out.is_empty:
            return out

        if out.prefill:
            self._prefill(out.prefill)
        if out.decode:
            self._decode(out.decode)

        self.issued_tokens += out.batch_size
        self.iterations += 1
        return out

    def _release_row(self, slot: int) -> None:
        """Empty a cache row whose slot the scheduler just took back.

        Called from inside `Scheduler.schedule`, for a request that finished, was
        aborted, or was preempted. The blocks are not freed here: they belong to the
        request and the scheduler hands them back to the pool around this call, so
        exactly one component ever talks to the allocator about a given block. What
        this owns is the row's table, and emptying it is what stops the next tenant
        of the slot attending over a stranger's K/V.
        """
        self.cache.reset_row(slot)

    def _prefill(self, requests) -> None:
        """Run the admitted rows' context, and emit one token each.

        `token_ids`, not `prompt_token_ids`, because a preempted request comes back
        as a longer version of itself: prompt plus every token it generated before
        it lost its blocks. Its K/V has to exist again before it can decode, and
        recomputing it is exactly a prefill over that longer context. The two lists
        are identical for a request that has never been preempted, which is why
        there is no branch here.
        """
        rows = [r.slot for r in requests]
        for request in requests:
            # The scheduler already took these blocks out of the pool; the row runs
            # on that reservation rather than making a second one.
            self.cache.adopt_row(request.slot, request.block_ids)
            self.recomputed_tokens += request.num_tokens if request.num_preemptions else 0

        batch = pad_prompts([r.token_ids for r in requests], pad_id=self.pad_id, side="left")
        logits = self.model.forward(
            batch.input_ids,
            batch.position_ids,
            cache=self.cache.view(rows),
            attention_mask=batch.attention_mask,
        )
        self.prefill_slots += batch.batch_size * batch.max_length
        self.prefill_tokens += int(batch.lengths.sum().item())
        self._collect(requests, last_token_logits(logits, batch).argmax(dim=-1))

    def _decode(self, requests) -> None:
        """Run one token for every already-running row.

        Each row is at its own absolute position, which is its own cached length: no
        two rows in this forward are at the same point in their sequence, and that
        is what the per-row block tables are for. The token forwarded is the one
        sampled last iteration, which is why the cache is exactly one token behind
        the request at the top of every decode step.
        """
        self._sync_rows(requests)
        rows = [r.slot for r in requests]
        device = self.cache.k_pool[0].device if self.cache.k_pool[0] is not None else None
        input_ids = torch.tensor(
            [[r.output_token_ids[-1]] for r in requests], dtype=torch.long, device=device
        )
        positions = torch.tensor(
            [[self.cache.tables[row].num_tokens] for row in rows],
            dtype=torch.long,
            device=device,
        )
        logits = self.model.forward(input_ids, positions, cache=self.cache.view(rows))
        self._collect(requests, logits[:, -1].argmax(dim=-1))

    def _sync_rows(self, requests) -> None:
        """Copy any block the scheduler added this iteration into the row's table.

        The row was given its blocks at adoption and the scheduler has been topping
        the request up a block at a time since, so the two lists drift by at most one
        entry per iteration and this closes the gap before the write that needs it.
        Skipping it does not raise: `BlockTable.append` would quietly allocate a
        block of its own, the pool would be booked twice for one sequence, and the
        damage would surface much later as a stranger's K/V in somebody's context.
        """
        for request in requests:
            table = self.cache.tables[request.slot]
            held = len(table.block_ids)
            if held < len(request.block_ids):
                self.cache.extend_row(request.slot, request.block_ids[held:])

    def _collect(self, requests, tokens: torch.Tensor) -> None:
        """Hand each row its sampled token. `append_token` applies the stop rules."""
        for request, token in zip(requests, tokens.tolist()):
            request.append_token(int(token))
            self.collected_tokens += 1

    # --- driving it -----------------------------------------------------------

    def run_to_completion(self, max_iterations: int = 100_000) -> list[Request]:
        """Step until both queues are empty. Returns the requests in *finish* order.

        Finish order, not submission order, because that is the order a server hands
        answers back and the order that shows the day's point: a short request
        returns at its own last token instead of at the batch's.

        `max_iterations` is a hang guard, not a policy. Every iteration either emits
        a token for a running row or releases one, so a loop that does not drain is
        a bug in this file and a raise is a better way to learn about it than a
        wedged process.
        """
        finished: list[Request] = []
        loops = 0
        while self.has_unfinished():
            out = self.step()
            finished.extend(out.finished)
            loops += 1
            if loops > max_iterations:
                raise RuntimeError(
                    f"the engine did not drain in {max_iterations} iterations: "
                    f"{self.scheduler.num_running} running, "
                    f"{self.scheduler.num_waiting} waiting"
                )
        return finished

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> list[list[int]]:
        """Offline convenience: submit N prompts, drain, return prompt+generation.

        In *submission* order, unlike `run_to_completion`, because a caller handing
        in a list wants a list back. Rows are ragged: each stops at its own EOS or
        its own budget, which is the whole difference from `greedy_generate_batch`,
        where every row runs until the slowest one is done.
        """
        requests = [
            self.add_request(
                Request(
                    request_id=self._new_id(),
                    prompt_token_ids=list(prompt),
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                )
            )
            for prompt in prompts
        ]
        self.run_to_completion()
        return [r.token_ids for r in requests]

    def _new_id(self) -> str:
        self._next_id += 1
        return f"req-{self._next_id}"

    # --- what the run cost ----------------------------------------------------

    @property
    def waste_fraction(self) -> float:
        """Share of the issued token-slots no request kept. Day 29's number.

        Zero on this loop by construction, and that is the claim rather than a
        measurement artefact: a finished row leaves the batch, so every row in a
        forward is a row that still wants a token. Static batching paid 79% on the
        same ragged shape because its batch was chosen once.
        """
        issued = self.issued_tokens
        return (issued - self.collected_tokens) / issued if issued else 0.0

    @property
    def prefill_padding_waste(self) -> float:
        """Share of the prefill rectangles spent on pad slots. The debt still owed.

        Continuous batching fixes the decode bill and not this one: prompts admitted
        together are still padded to the longest of them, and every pad slot pays a
        full row of attention and MLP for a token that does not exist. Ragged
        (varlen) prefill and chunked prefill are what remove it.
        """
        slots = self.prefill_slots
        return (slots - self.prefill_tokens) / slots if slots else 0.0
