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
from .compiled import CompiledDecode
from .deferred import DeferredOutputProcessor
from .output import OutputProcessor, TokenBatch
from .plan import plan_decode
from .profiler import NULL_RECORDER
from .sampling import BatchedSampler, SamplingParams
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
    defer_window: Day 48. How many steps the sampled tokens may stay on the device
               before the host looks at them. 0 is Day 47's loop: resolve every
               step, one journey home per step. 1 is a one-step lag, which does not
               reduce that count and does let the next decode take its `input_ids`
               straight from the tensor the sampler left. k brings k steps home in
               one journey, and costs up to k rows per finished request that
               nobody keeps. The scheduler is given a matching `lookahead`,
               because a token still on the device is in the cache row and not in
               `Request.num_tokens`. See `nanoserve.deferred`.
    compile_decode: Day 49. Which bet to take on `torch.compile` for the decode
               forward, and only the decode forward. `None` (or "off") leaves the
               eager path alone. "dynamic" compiles once with the row count and the
               context width symbolic, which is the only setting that survives a
               decode loop: the context width grows by one every step, so a graph
               specialised on it is rebuilt every step until dynamo gives up.
               "static" is that specialisation, kept because measuring it is the
               only way the claim above is a measurement. The prefill stays eager
               either way, because its rectangle is a different shape on almost
               every admission. See `nanoserve.compiled`.
    seed:      seeds the sampler's *shared* generator, the one requests that gave
               no seed of their own draw from. It makes a fixed set of requests
               repeatable; it cannot make one request repeatable, because with a
               shared generator that request's draw depends on how many of its
               batchmates drew before it in the same step. Per-request `seed` in
               `SamplingParams` is what buys that.

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
        seed: int | None = None,
        defer_window: int = 0,
        compile_decode: str | None = None,
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
        # Day 40. The rows of one forward no longer agree about how to sample, and
        # this is what holds the disagreement: per-request parameters come in on
        # the `Request`, per-request RNG state lives here, keyed by request id and
        # dropped when the request finishes.
        self.sampler = BatchedSampler(seed=seed)
        # Day 47. The collect half, and the only place in this loop allowed to bring
        # a tensor back to the host. The sampler now returns a `[rows]` device tensor,
        # so a step reads its tokens home exactly once, here, instead of once per
        # sampled row inside `sample`. It counts that too, which is what makes
        # `output.check_single_transfer` a check rather than a comment.
        #
        # Day 48 makes "once" adjustable. With a window the tokens stay on the
        # device across step boundaries and several steps go home together, so the
        # counter this reports is a rate below one rather than exactly one, and the
        # two processors are kept as two classes rather than one with a flag
        # because the deferred one has three ways for a token not to reach its
        # request and the immediate one has none.
        # Day 49. The forward the compiler is pointed at, wrapped whether or not
        # anything was asked for, so `_decode` has one call site rather than a
        # branch and so `engine.decode_forward.calls` is always a number.
        #
        # Two different callables go in, and the difference is the honest content
        # of "off". Uncompiled, the wrapper is handed `_eager_forward`, which looks
        # `forward` up on the model every call, so replacing `engine.model.forward`
        # (which the Day-48 tests do, to spy on the decode input) still reaches the
        # decode path exactly as it did before this class existed. Compiled, it is
        # handed the bound method itself, because that is what compiling *is*: the
        # function has been traced and the trace is what runs, and a later
        # assignment to the attribute cannot reach inside it.
        self.decode_forward = CompiledDecode(
            self._eager_forward if not compile_decode else model.forward,
            mode=compile_decode or "off",
        )
        self.defer_window = defer_window
        self.output = (
            DeferredOutputProcessor(window=defer_window) if defer_window else OutputProcessor()
        )
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
        self.prefill_rows = 0
        self.recomputed_tokens = 0
        self._next_id = 0
        # Day 46. Where the seconds of a step go. `NULL_RECORDER` is a real object
        # with the real shape rather than a `None` to branch on, so the loop below
        # reads the same whether anybody is watching or not, and the phase names are
        # written down in one place instead of living in a separate profiling copy
        # of `step` that would drift from this one. Swap in a `StepRecorder` to
        # collect. See `nanoserve.profiler`.
        self.recorder = NULL_RECORDER

    @classmethod
    def build(
        cls,
        model,
        num_blocks: int,
        block_size: int = 16,
        max_batch_size: int = 8,
        pad_id: int = 0,
        seed: int | None = None,
        defer_window: int = 0,
        compile_decode: str | None = None,
        max_model_len: int | None = None,
        bucket_decode: bool = False,
        persist_inputs: bool = False,
    ) -> Engine:
        """Wire a scheduler and a matching cache over one fresh pool.

        `defer_window` reaches the scheduler as `lookahead` and nowhere else. The
        two numbers are the same number seen from two sides: one is how many steps
        of tokens are still on the device, the other is how many tokens of block
        headroom that means every running row needs. Letting a caller set them
        apart is letting them set them wrong.

        `max_model_len` is Day 51, and it sizes one thing: the cache's persistent
        slot table, `[max_batch_size, max_model_len]` int64 held for the process.
        Left unset it falls back to the pool, which is the bound that always holds
        and is far larger than any server actually serves. `build_engine` passes the
        length it planned the pool around, which is where the number belongs.

        `bucket_decode` is Day 52 and it changes one thing: every decode step is
        padded up to a row bucket and read at a width multiple, so the forward sees
        one of a small closed set of shapes rather than a new one every step. It
        costs cells the kernel computes and nobody reads, and it buys the only
        property a replayed capture cannot do without. See `nanoserve.buckets`.

        `persist_inputs` is Day 53 and it is the other half of the same want. The
        step's four input tensors are allocated once and written in place, so what
        the forward is handed is a window at an address that does not change for the
        life of the process. A replay takes no arguments; it reads the buffers it
        was recorded against. See `nanoserve.inputs`.
        """
        allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        return cls(
            model,
            Scheduler(allocator, max_batch_size=max_batch_size, lookahead=defer_window),
            BatchedPagedKVCache(
                model.config,
                allocator,
                batch_size=max_batch_size,
                max_model_len=max_model_len,
                bucket_decode=bucket_decode,
                persist_inputs=persist_inputs,
            ),
            pad_id=pad_id,
            seed=seed,
            defer_window=defer_window,
            compile_decode=compile_decode,
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
        with self.recorder.step() as timing:
            with timing.phase("schedule"):
                out = self.scheduler.schedule()
                # Whatever the schedule just reaped is done drawing: finished, or
                # aborted, which arrives here as finished too. Preempted requests are
                # deliberately not in this list, because they come back and their
                # generator has to be where they left it. Dropping the rest is not
                # tidiness: a server that keeps one `torch.Generator` per request it
                # has ever served leaks for as long as it runs.
                for request in out.finished:
                    self.sampler.release(request.request_id)
            if out.is_empty:
                # An iteration that admitted nobody is not a step of this loop, so it
                # is thrown away rather than averaged in as a very fast one.
                timing.drop()
                return out

            timing.describe("prefill" if out.prefill else "decode", out.batch_size)
            if out.prefill:
                self._prefill(out.prefill, timing)
            if out.decode:
                self._decode(out.decode, timing)

            if self.defer_window:
                # After both forwards have been queued, which is the whole placement
                # argument: the host has handed the device this step's work before it
                # stops to look at an older step's answer. On one stream that is a
                # reordering rather than an overlap, and what it actually buys is the
                # batching: everything but the newest batch goes home together.
                with timing.phase("settle", syncs=True):
                    self.collected_tokens += len(self.output.settle())

            self.issued_tokens += out.batch_size
            self.iterations += 1
            return out

    def _eager_forward(self, *args, **kwargs):
        """The decode forward, resolved on the model at call time. Day 49.

        One line, and it exists so that "off" means off. `CompiledDecode` holds a
        callable, and holding `model.forward` directly would freeze the attribute
        at construction: the prefill would follow a later reassignment and the
        decode would not, which is a difference nobody would look for.
        """
        return self.model.forward(*args, **kwargs)

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

    def _prefill(self, requests, timing) -> None:
        """Run the admitted rows' context, and emit one token each.

        `token_ids`, not `prompt_token_ids`, because a preempted request comes back
        as a longer version of itself: prompt plus every token it generated before
        it lost its blocks. Its K/V has to exist again before it can decode, and
        recomputing it is exactly a prefill over that longer context. The two lists
        are identical for a request that has never been preempted, which is why
        there is no branch here.
        """
        with timing.phase("adopt_rows"):
            rows = [r.slot for r in requests]
            for request in requests:
                # The scheduler already took these blocks out of the pool; the row
                # runs on that reservation rather than making a second one.
                self.cache.adopt_row(request.slot, request.block_ids)
                self.recomputed_tokens += request.num_tokens if request.num_preemptions else 0

        with timing.phase("build_inputs"):
            batch = pad_prompts([r.token_ids for r in requests], pad_id=self.pad_id, side="left")

        with timing.phase("forward", device=True):
            logits = self.model.forward(
                batch.input_ids,
                batch.position_ids,
                cache=self.cache.view(rows),
                attention_mask=batch.attention_mask,
            )

        with timing.phase("account", syncs=True):
            self.prefill_slots += batch.batch_size * batch.max_length
            # `.item()` on a device tensor: a synchronisation, on the prefill path.
            self.prefill_tokens += int(batch.lengths.sum().item())
            self.prefill_rows += batch.batch_size

        with timing.phase("sample", device=True):
            tokens = self._sample(requests, last_token_logits(logits, batch))

        with timing.phase("collect", syncs=not self.defer_window):
            self._collect(requests, tokens)

    def _decode(self, requests, timing) -> None:
        """Run one token for every already-running row.

        Each row is at its own absolute position, which is its own cached length: no
        two rows in this forward are at the same point in their sequence, and that
        is what the per-row block tables are for. The token forwarded is the one
        sampled last iteration, which is why the cache is exactly one token behind
        the request at the top of every decode step.
        """
        with timing.phase("sync_rows"):
            self._sync_rows(requests)

        with timing.phase("build_inputs"):
            rows = [r.slot for r in requests]
            device = self.cache.k_pool[0].device if self.cache.k_pool[0] is not None else None
            # One host-to-device copy of one integer per row, built out of a Python
            # list, every step. Tiny in bytes and not tiny in time: this is the phase
            # a captured graph makes disappear by writing into a fixed input buffer.
            # Day 48 removes the other one. See `_decode_input_ids`.
            values = self._decode_input_values(requests, device)
            inputs = self.cache.decode_inputs
            input_ids = None if inputs is not None else self._as_input_ids(values, device)
            # Day 50. The step's whole addressing, decided here rather than inside
            # the forward: this grows every row's table by one and hands back the
            # write slots, the read rectangle, the context lengths and the new
            # tokens' positions. `positions` used to be built right here off
            # `tables[row].num_tokens`, and the plan is the same list read one line
            # earlier, so the phase does not gain a build. What it gains is that the
            # forward no longer does one.
            plan = plan_decode(self.cache, rows, device)
            view = self.cache.view(rows, plan=plan)
            if inputs is not None:
                # Day 53. The tokens go into the buffer the forward already reads
                # from, and the window is widened to the bucketed row count rather
                # than concatenated onto. The padded rows keep whatever the buffer
                # last held, which is a legal token id because nothing but one is
                # ever written here, and those rows' logits are dropped below.
                input_ids = inputs.set_input_ids(values, window=plan.graph_rows)
            elif plan.pad_rows:
                # Day 52. The rows the plan invented need a token each, and the
                # forward wants one tensor. This is the last per-step allocation on
                # the input side and it is `[pad, 1]` of zeros; Day 53's buffers are
                # what replace it, which is the next thing this padding exists for.
                input_ids = torch.cat(
                    [input_ids, input_ids.new_zeros(plan.pad_rows, 1)], dim=0
                )

        with timing.phase("forward", device=True):
            # Through the Day-49 wrapper rather than straight at the model. It is
            # the same call when nothing is compiled, and when something is it is
            # the one place that knows how many distinct shapes this run has asked
            # a compiler to build for.
            logits = self.decode_forward(input_ids, plan.positions, cache=view)

        with timing.phase("sample", device=True):
            # Day 46 measured this phase at 86% to 89% of the whole host loop and
            # Day 47 found out why: it used to end in one `int(tensor)` per row. Now
            # it ends in a `[rows]` tensor that stays where it was computed, so
            # nothing here waits on a kernel and `syncs` is gone from this line.
            # `logits` is `[graph_rows, seq, vocab]`, and on a bucketed step the
            # padded rows are on the end of it. Dropping them here rather than
            # inside the forward keeps the traced region's shapes constant, which
            # is the whole reason they are there.
            tokens = self._sample(requests, logits[: len(requests), -1])

        with timing.phase("collect", syncs=not self.defer_window):
            # Where Day 47's annotation went, and the honest reading of that day: a
            # sync moved rather than a sync removed. A stop rule needs a Python int,
            # so the step still stops the host exactly once. Day 48 is what removes
            # it: with `defer_window` set, this phase only hands the tensor over and
            # the stop is `settle`, one step or more later. The annotation follows
            # the sync rather than the phase name, which is the only way
            # `sync_points` stays a statement about the code that ran.
            self._collect(requests, tokens)

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

    def _sample(self, requests, logits: torch.Tensor) -> TokenBatch:
        """One token per row, each under the params its own request asked for.

        Until Day 40 this line was `.argmax(dim=-1)` and it was the same call for
        every row, because every row was greedy. Now the rows of one `[rows,
        vocab]` tensor can disagree: greedy, `top_p=0.9`, `temperature=1.4` with a
        seed. What is handed down is a request id per row as well as its params,
        because the RNG state is keyed by id and has to survive from this step to
        the next one, and across a preemption that puts the row back in a prefill.

        Day 47 changes what comes back and not what is drawn. `sample_batch_device`
        returns the tokens as a tensor on the device the forward ran on, and the
        request ids ride along in the `TokenBatch` so that `collect` can check the
        rows are still the rows it sampled. They can stop being that: between these
        two phases nothing moves, but between one step and the next the scheduler
        releases and admits, and a `[rows]` tensor on its own is anonymous.
        """
        rows = [(r.request_id, r.sampling) for r in requests]
        return TokenBatch(self.sampler.sample_batch_device(logits, rows), [i for i, _ in rows])

    def _decode_input_ids(self, requests, device) -> torch.Tensor:
        """This step's input tokens as the `[rows, 1]` the model wants.

        Day 53 splits this in two. What the token *is* is `_decode_input_values`
        below and has not changed; what is here is the tensor it becomes, which is
        the half a persistent input buffer replaces. Nothing outside this class
        calls the pair, so the split costs a line and keeps the fast path readable.
        """
        return self._as_input_ids(self._decode_input_values(requests, device), device)

    def _as_input_ids(self, values, device) -> torch.Tensor:
        """A fresh `[rows, 1]` from either form the values come in. Day 52's path."""
        if isinstance(values, torch.Tensor):
            return values.unsqueeze(1)
        return torch.tensor([[v] for v in values], dtype=torch.long, device=device)

    def _decode_input_values(self, requests, device):
        """This step's input tokens, from the device if last step's are still on it.

        The fast path is the day. The token a decode row forwards is exactly the
        token the previous step sampled for that row, so if the previous step's
        `[rows]` tensor is still held and its rows are still these rows, the input
        is `tensor.unsqueeze(1)`: a view, no allocation, no Python list, and above
        all no journey home to build one out of. The slow path is Day 33's line,
        and it needs `output_token_ids[-1]`, which is a Python int, which is a
        readback that has not happened yet.

        The row check is not a formality. Between one step and the next the
        scheduler releases finished rows and admits new ones, so a held `[rows]`
        tensor can be the right length and the wrong rows, and using it then hands
        every row after the change somebody else's token in fluent, plausible,
        wrong text. When the rows differ the held tokens are not this step's input
        and the requests below need ints the engine does not have, so it pays for
        them here and goes back to deferring on the next step.

        Returns a device tensor on the fast path and a list of host ints on the slow
        one, and the caller decides what to do with each. Day 53 wants that
        distinction: a tensor already on the device is copied straight into the
        input buffer, and a list goes through the buffer's pinned staging mirror.
        """
        held = self.output.newest if self.defer_window else None
        if held is not None and held.request_ids == tuple(r.request_id for r in requests):
            return held.tokens
        if self.defer_window:
            self.collected_tokens += len(self.output.flush())
        return [r.output_token_ids[-1] for r in requests]

    def _collect(self, requests, tokens: TokenBatch) -> None:
        """Hand the step's tokens to the output path, and let it decide when to look.

        Day 47's version was the whole collect phase: one `.tolist()` and one
        `append_token` per row, the single synchronisation left in a step. With a
        deferral window it is a handover instead, and the transfer happens in
        `settle` after the next forward has been queued, or later still if the
        window covers more steps than one.
        """
        if self.defer_window:
            self.output.defer(tokens, requests)
        else:
            self.collected_tokens += len(self.output.apply(tokens, requests))

    def flush(self) -> int:
        """Bring every held token home. The end of a run, and a no-op without a window.

        A run drains itself in the ordinary case, because a request is only finished
        once its stop token has been applied and `has_unfinished` is what ends the
        loop. What is left afterwards is overshoot: rows forwarded for requests that
        were already done. They still have to be counted, or `waste_fraction` reports
        a run that issued more tokens than it can account for.
        """
        if not self.defer_window:
            return 0
        applied = len(self.output.flush())
        self.collected_tokens += applied
        return applied

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
        self.flush()
        return finished

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        eos_id: int | None = None,
        sampling: SamplingParams | None = None,
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
                    sampling=sampling or SamplingParams(),
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
    def forward_tokens(self) -> int:
        """Token positions the forwards computed: the honest denominator. Day 34.

        `issued_tokens` counts rows, which is the right unit for the decode bill and
        the wrong one for everything a prefill does. A decode row is one position; a
        prefill row is its whole context, and a *resumed* prefill row is the whole
        context of a request that already had one. Counting rows would hide the
        recompute surcharge in exactly the place it is biggest, so this counts
        positions: every real prefill token, plus one for each decode row.

        Pad slots stay out on purpose. They are a separate bill, reported next door
        as `prefill_padding_waste`, and folding them in would make the recompute
        share move with how the prompts happened to line up in a rectangle.
        """
        return self.prefill_tokens + self.issued_tokens - self.prefill_rows

    @property
    def recompute_fraction(self) -> float:
        """Share of the forward positions that were bought a second time.

        The price of preemption in the currency the engine pays it in. Zero when the
        pool is big enough for the offered load, and it climbs as the pool shrinks,
        which is the trade Day 33 made when it stopped reserving the worst case.
        The other half of the bill is not here: it is one caller's latency, and
        `ContinuousTiming.preemption_latency_penalty` is where that lands.
        """
        forwarded = self.forward_tokens
        return self.recomputed_tokens / forwarded if forwarded else 0.0

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
