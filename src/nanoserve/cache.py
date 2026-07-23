"""Paged KV cache and block allocator. Weeks 3-5.

Week 3 ships a naive contiguous cache that grows per step. Weeks 4-5 replace it
with a paged cache: a fixed pool of physical blocks plus a per-sequence block
table that maps logical token positions to physical blocks. This is the OS-paging
analogy at the heart of the engine.

Why a cache at all? Without one, generating token n re-runs attention over the
whole 0..n-1 prefix every step, because attention needs every past key and value.
But those keys and values only depend on tokens that are already fixed: token 3's
K and V never change once token 3 is in the sequence. So the recompute is pure
waste. The cache is the one idea that turns an O(n^2) decode (every step redoes
all the prefix work) into O(n) (every step does one token's work and reads the
rest): compute each token's K/V exactly once, keep them, and append.
"""

from __future__ import annotations

import torch

from .kernels.paged_attention import paged_attention_batched_reference
from .kernels.triton_paged_attention import paged_attention as paged_attention_dispatch


class NaiveKVCache:
    """Week 3: contiguous per-layer K/V store that grows by concatenation.

    One (K, V) pair per transformer layer, each shaped
    `[batch, num_kv_heads, seq_so_far, head_dim]`. "Naive" and "contiguous" mean
    the store is literally one growing tensor per layer: every decode step
    `torch.cat`s the new token's K/V onto the end and hands the whole thing back.
    That is the simplest thing that works and the thing Weeks 4-5 will replace,
    because a contiguous per-sequence buffer is exactly what fragments VRAM and
    forces you to pre-reserve max-length space for every request. For a single
    sequence on Week 3 it is perfect, and it makes the speedup visible before the
    paging machinery arrives.

    Crucially the cache holds the *compact* GQA K/V (8 KV heads here), not the
    repeated 32-head version: the head expansion is a cheap view applied at read
    time in attention, so storing the repeat would quadruple the cache for nothing
    and throw away the entire reason GQA exists.
    """

    def __init__(self, num_layers: int):
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K/V for `layer` and return the full running K/V.

        k, v: [batch, num_kv_heads, new_seq, head_dim]. `new_seq` is the prompt
              length on the prefill call and 1 on each decode step. The first
              append for a layer seeds the store; later ones concatenate on the
              sequence axis (dim 2), oldest token first.

        Returns the complete (K, V) for the layer so attention can score the new
        query against the entire history without the model holding the past
        itself.
        """
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def seq_len(self) -> int:
        """Tokens cached so far (read off layer 0; all layers stay in lockstep)."""
        return 0 if self.k[0] is None else self.k[0].shape[2]


class KVCacheExhausted(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation.

    This is the engine's out-of-memory signal, scoped to KV cache. On a single
    sequence it just means "context too long for the pool", but it is the same
    event that, once many sequences share the pool (Weeks 8-9), triggers
    scheduling decisions: stop admitting new requests, or preempt and recompute a
    running one to reclaim its blocks. Naming it now means the later scheduler can
    catch one specific thing instead of a bare RuntimeError.
    """


class BlockAllocator:
    """Week 4: a fixed pool of physical KV blocks, with alloc and free.

    This is the OS-paging analogy made literal. Physical memory is carved into
    `num_blocks` fixed-size blocks, each holding `block_size` tokens' worth of
    K/V. The allocator owns nothing about tensors yet (Week 5 hangs the real K/V
    storage off these ids); it is pure bookkeeping over integer block ids, and
    keeping it that way is the point. The hard part of paging is not the storage,
    it is never losing or double-handing-out a block, so the whole component is a
    free list plus an allocated set, and the invariant (every block is in exactly
    one of them) is what the tests pin.

    Why fixed blocks at all? The Week-3 contiguous cache forces each sequence to
    reserve one growing buffer sized to its worst case, which fragments memory and
    caps how many sequences fit. Blocks decouple a sequence's *logical* length
    from *physical* placement: a 40-token sequence with block_size 16 takes three
    blocks that can sit anywhere in the pool, and frees them independently when it
    finishes. That is what lets the pool stay packed under many concurrent
    sequences, which is the entire reason paged attention exists.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError(
                f"num_blocks and block_size must be positive, got "
                f"num_blocks={num_blocks}, block_size={block_size}"
            )
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Free ids kept as a stack. Seeded high-to-low so pop() hands out 0, 1, 2,
        # ...: not required for correctness, but it makes allocation order
        # deterministic and the tests legible. `_allocated` is the other half of
        # the invariant: a block is in exactly one of these two at all times.
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        """Physical blocks available right now."""
        return len(self._free)

    def blocks_for_length(self, num_tokens: int) -> int:
        """How many blocks a sequence of `num_tokens` needs (ceiling division).

        Zero tokens need zero blocks; otherwise the last partial block still costs
        a whole physical block, because a block is the unit of allocation. This is
        the only place the token count meets the block count.
        """
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        """Whether the pool can currently hold a `num_tokens` sequence."""
        return self.blocks_for_length(num_tokens) <= self.num_free

    def allocate(self) -> int:
        """Hand out one free physical block, or raise if the pool is dry."""
        if not self._free:
            raise KVCacheExhausted(
                f"block pool exhausted: all {self.num_blocks} blocks allocated"
            )
        block_id = self._free.pop()
        self._allocated.add(block_id)
        return block_id

    def allocate_for(self, num_tokens: int) -> list[int]:
        """Reserve all blocks a `num_tokens` sequence needs, atomically.

        Either every needed block is reserved and returned, or the pool is left
        exactly as it was and `KVCacheExhausted` is raised. The atomicity matters:
        a half-filled allocation would strand blocks that no sequence owns and no
        one ever frees, so the count is checked up front before any block moves.
        """
        need = self.blocks_for_length(num_tokens)
        if need > self.num_free:
            raise KVCacheExhausted(
                f"need {need} blocks for {num_tokens} tokens, only "
                f"{self.num_free} free"
            )
        return [self.allocate() for _ in range(need)]

    def free(self, block_id: int) -> None:
        """Return one block to the pool. Rejects anything not currently allocated.

        Freeing a block that is already free (a double free) or never belonged to
        this pool is always a bug in the caller, and a silent one corrupts the
        pool: the same block would be handed to two sequences. So it raises rather
        than guessing.
        """
        if block_id not in self._allocated:
            raise ValueError(
                f"block {block_id} is not allocated (double free or foreign id)"
            )
        self._allocated.discard(block_id)
        self._free.append(block_id)

    def free_all(self, block_ids: list[int]) -> None:
        """Return a whole sequence's blocks to the pool, e.g. when it finishes."""
        for block_id in block_ids:
            self.free(block_id)


class BlockTable:
    """Week 4: one sequence's map from logical token position to physical block.

    The allocator owns the pool but is blind to meaning: it hands out block ids
    and takes them back, nothing more. The block table is the per-sequence layer
    on top that remembers *which* physical block holds *which* logical token. It
    is the address-translation table of the paging analogy, the exact counterpart
    of an OS page table: logical position -> (physical block, offset within it).

    A sequence is just an ordered list of block ids, `block_ids[i]` holding
    logical positions `i*block_size .. i*block_size + block_size - 1`. Those
    physical blocks can sit anywhere in the pool and need not be contiguous (the
    allocator hands out whatever is free), which is the entire freedom paging buys
    over the Week-3 contiguous cache. Translation stays trivial regardless:
    position `p` lives in `block_ids[p // block_size]` at offset `p % block_size`.

    Still torch-free. No K/V is stored here; the table only computes *where* each
    token's K/V will go. Week 5 hangs the real per-block K/V tensors off the pool
    and writes/reads them at the `slot` this table returns.
    """

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.block_ids: list[int] = []
        self.num_tokens = 0

    @property
    def capacity(self) -> int:
        """Tokens the currently held blocks can hold (>= num_tokens)."""
        return len(self.block_ids) * self.block_size

    def append(self, num_new_tokens: int = 1) -> None:
        """Grow the sequence by `num_new_tokens`, pulling blocks only as needed.

        New blocks are reserved exactly when growth crosses a block boundary: a
        token at position `capacity` has no block yet, so one is allocated. On the
        common decode step (one token, mid-block) no allocation happens at all,
        which is the whole efficiency story: physical memory is grabbed a block at
        a time, only at the moment it is first written.

        Atomic like `allocate_for`: if the growth needs more blocks than the pool
        can supply, none are taken and `KVCacheExhausted` is raised, leaving the
        table exactly as it was. A half-grown table would strand blocks and desync
        `num_tokens` from the blocks actually held.
        """
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be non-negative, got {num_new_tokens}")
        target = self.num_tokens + num_new_tokens
        blocks_needed = self.allocator.blocks_for_length(target) - len(self.block_ids)
        if blocks_needed > 0:
            # allocate_for is itself atomic: it reserves all or raises, taking
            # none, so a failure here cannot leave a partially grown table.
            self.block_ids.extend(self.allocator.allocate_for(blocks_needed * self.block_size))
        self.num_tokens = target

    def position(self, logical_pos: int) -> tuple[int, int]:
        """Translate a logical token position to (physical block id, offset).

        Raises `IndexError` for any position not currently live (negative or
        `>= num_tokens`), because a translation past the sequence would point at a
        block that holds no K/V for this token yet.
        """
        if logical_pos < 0 or logical_pos >= self.num_tokens:
            raise IndexError(
                f"position {logical_pos} out of range for {self.num_tokens} tokens"
            )
        block_id = self.block_ids[logical_pos // self.block_size]
        offset = logical_pos % self.block_size
        return block_id, offset

    def slot(self, logical_pos: int) -> int:
        """Flat index of a token's K/V in a `[num_blocks * block_size]` pool.

        This is the single integer Week 5's paged read/write uses: rather than
        index a `[block][offset]` tensor, flatten the pool to one axis and address
        it as `block_id * block_size + offset`. Distinct live positions always map
        to distinct slots, even when the sequence's blocks are scattered.
        """
        block_id, offset = self.position(logical_pos)
        return block_id * self.block_size + offset

    def free(self) -> None:
        """Return every block to the pool and reset to an empty, reusable table."""
        self.allocator.free_all(self.block_ids)
        self.block_ids = []
        self.num_tokens = 0


class PagedKVCache:
    """Week 5: real K/V stored in a block pool, read/written through a BlockTable.

    This is where the Day-14 allocator and Day-15 table finally hold tensors. It
    is a drop-in for `NaiveKVCache` at the one interface attention uses: the same
    `append(layer, k, v) -> (full_k, full_v)` and the same `seq_len`. Attention
    does not change at all; only where the K/V physically live does.

    The naive cache keeps one growing contiguous tensor per layer. The paged
    cache keeps, per layer, a *fixed* flat pool of shape
    `[num_blocks * block_size, num_kv_heads, head_dim]` and writes each token's
    K/V at the flat `slot` the block table assigns. Those slots can be anywhere in
    the pool, so a sequence's K/V may be physically scattered, which is the whole
    point: physical placement is decoupled from logical position, so the pool
    stays packed under many sequences (Weeks 8-9) instead of each sequence
    reserving a worst-case contiguous buffer.

    One sequence, one table. `BatchedPagedKVCache` below is the Phase-3 sibling that
    gives each row of a batch its own table over the same pool; this class stays as
    the single-sequence path the batched one is graded against, row by row.

    One block table, shared across all layers. Every layer stores the same tokens
    at the same logical positions, so the logical->physical map is identical for
    all of them; a physical block id therefore names a slot in *every* layer's
    pool at once, exactly as it does in vLLM. The table is grown once per step, on
    the first layer's append, and the other layers reuse the slots it computed.

    Two reads live here. `append` is the Day-16 gather: rebuild the contiguous
    history so the output is verifiably identical to the naive path, the reference
    the tests still compare against. `paged_attention` is the Week-6 fused read: it
    attends directly over the scattered blocks through the Day-18 reference and
    hands `gqa_attention` the attention output, never rebuilding the history. The
    model runs on the fused read now, and since Day 25 that read dispatches through
    `select_backend`: the Triton kernel on a card, the tlsim CPU model on a laptop,
    both held to `paged_attention_reference` as the oracle. The reference is the
    check the dispatch is graded against, no longer the path the model runs.
    """

    def __init__(self, config, allocator: BlockAllocator):
        self.config = config
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.num_layers = config.num_hidden_layers
        # One table for the sequence (all layers share the logical->physical map).
        self.table = BlockTable(allocator)
        # Per-layer flat K/V pools, allocated lazily on first append so their dtype
        # and device match the data attention actually produces.
        self.k_pool: list[torch.Tensor | None] = [None] * self.num_layers
        self.v_pool: list[torch.Tensor | None] = [None] * self.num_layers
        # Flat slots this step writes to, computed once on layer 0 and reused by
        # the remaining layers (they store the same positions).
        self._step_slots: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        """Tokens cached so far (the shared table's length; all layers agree)."""
        return self.table.num_tokens

    def _slots_for(self, positions: range, device) -> torch.Tensor:
        """Flat pool indices for a range of logical positions."""
        return torch.tensor(
            [self.table.slot(p) for p in positions], dtype=torch.long, device=device
        )

    def _write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store this step's K/V for `layer` into the block pool. No read.

        k, v: [1, num_kv_heads, new_seq, head_dim]. `new_seq` is the prompt length
              on prefill and 1 on a decode step. Batch is 1: one sequence per cache
              until Phase 3 gives each sequence its own table.

        On the first layer of a step the shared table grows by `new_seq` (pulling
        physical blocks only when a token crosses a block boundary, raising
        `KVCacheExhausted` atomically if the pool is dry), and the slots for the new
        tokens are recorded. Every layer then scatters its `new_seq` K/V into those
        slots in its own pool. This is the write half both reads share: `append`
        gathers the contiguous history after it, `paged_attention` attends over the
        scattered blocks after it. The cache ends in the identical state either way.
        """
        if k.shape[0] != 1:
            raise ValueError(
                "PagedKVCache handles one sequence; batch must be 1 "
                "(per-sequence tables arrive in Phase 3)"
            )
        new_seq = k.shape[2]

        if layer == 0:
            start = self.table.num_tokens
            self.table.append(new_seq)  # atomic; raises KVCacheExhausted if dry
            self._step_slots = self._slots_for(range(start, start + new_seq), k.device)

        if self.k_pool[layer] is None:
            pool = (self.allocator.num_blocks * self.block_size,
                    self.config.num_key_value_heads, self.config.head_dim)
            self.k_pool[layer] = torch.zeros(pool, dtype=k.dtype, device=k.device)
            self.v_pool[layer] = torch.zeros(pool, dtype=v.dtype, device=v.device)

        # Write [1, n_kv, new_seq, d] -> [new_seq, n_kv, d] at this step's slots.
        self.k_pool[layer][self._step_slots] = k[0].transpose(0, 1)
        self.v_pool[layer][self._step_slots] = v[0].transpose(0, 1)

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this step's K/V for `layer` and return the full running K/V.

        The gather read: write the K/V (see `_write`), then reassemble the whole
        history (positions `0..seq_len-1`) into a contiguous
        `[1, num_kv_heads, seq, head_dim]`, oldest token first, exactly the tensor
        `NaiveKVCache.append` returns. This is the Day-16 drop-in that let the paged
        cache prove itself equal to the naive one before any kernel existed. Week 6
        moves the model onto `paged_attention` below, which never rebuilds this
        contiguous buffer; `append` stays as the reference read the tests compare to.
        """
        self._write(layer, k, v)
        # Gather the full history back, oldest first, into [1, n_kv, seq, d].
        hist = self._slots_for(range(self.table.num_tokens), k.device)
        full_k = self.k_pool[layer][hist].transpose(0, 1)[None]
        full_v = self.v_pool[layer][hist].transpose(0, 1)[None]
        return full_k, full_v

    def paged_attention(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        n_rep: int,
        scale: float | None = None,
    ) -> torch.Tensor:
        """The fused read: store this step's K/V and attend over the block table.

        This is Week 6's payoff wired into the cache. Rather than hand attention the
        contiguous history to score itself (`append`), the cache does the read: it
        writes the K/V (see `_write`), then dispatches the paged read over its own
        scattered pool, reading each past token through its slot and never rebuilding
        the contiguous buffer. `gqa_attention` calls this and takes the attention
        output straight back.

        Since Day 25 the read goes through `paged_attention_dispatch`, the Day-23
        entry point that asks `select_backend` which read the query's device can run:
        the Triton kernel on a CUDA tensor, the Day-22 tlsim model on the CPU. The
        engine gets the kernel on a card and the correct-and-slow model on a laptop
        without the cache knowing which, and `paged_attention_reference` steps back to
        being the oracle both backends are graded against, not the path either runs.

        k, v: [1, num_kv_heads, new_seq, head_dim] this step's compact K/V.
        q:    [1, num_attention_heads, new_seq, head_dim] this step's rotated query.
        n_rep: GQA repeat factor (`config.num_kv_groups`).
        scale: softmax scale; defaults to `head_dim ** -0.5`, matching `gqa_attention`.

        Returns [1, num_attention_heads, new_seq, head_dim], the attention output
        before o_proj. It matches a contiguous SDPA over the same K/V to a few ulps
        rather than bit for bit: both backends stream the online softmax, which
        reassociates the exponent sums, the accuracy trade every flash-attention
        kernel makes. The pool and table are left exactly as `append` would leave
        them; only the return value differs.
        """
        self._write(layer, k, v)
        slot_mapping = self._slots_for(range(self.table.num_tokens), q.device)
        return paged_attention_dispatch(
            q, self.k_pool[layer], self.v_pool[layer], slot_mapping, n_rep, scale
        )

    def free(self) -> None:
        """Return every block to the pool and reset to an empty, reusable cache."""
        self.table.free()
        self.k_pool = [None] * self.num_layers
        self.v_pool = [None] * self.num_layers
        self._step_slots = None


class BatchedPagedKVCache:
    """Week 7: one block table per sequence, one pool for all of them. Day 28.

    `PagedKVCache` above is one sequence with one table, which is what Weeks 4-6
    needed and what Day 27's padded batch ran into: many prompts in one forward, one
    table between them, so the fused read had no way to tell whose blocks were whose
    and refused. This class is that missing piece. N sequences, N `BlockTable`s, and
    a single `BlockAllocator` and per-layer pool underneath all of them.

    The shared pool is the point, not an implementation detail. Blocks are handed to
    whichever row needs one next, so rows interleave physically and a short sequence
    occupies exactly what it uses instead of a slice sized to the batch's longest
    member. That is the difference between paging and simply putting N contiguous
    caches side by side, and it is what the Weeks 8-9 scheduler will allocate out of.

    Two shapes coexist here, and keeping them straight is most of the work:

      - **The rectangle is the input's, not the cache's.** A prefill arrives as a
        padded `[batch, n_kv, max_len, d]`, and `write` takes the batch's key mask
        and stores only the real columns. A row of length 2 gets 2 slots. Nothing
        that is not a token ever enters the pool.
      - **The read gets a rectangle back, but only for addressing.**
        `slot_mapping()` returns `[batch, max_ctx]` plus a `context_lens` vector,
        because a kernel wants one tensor, not a ragged list. The entries past a
        row's context are inert padding whose one job is to remain a legal index.

    Which is why the decode read needs no attention mask at all: Day 27's key mask
    existed to hide pad tokens that were sitting in the K/V, and per-sequence tables
    mean there are none to hide. The mask became `context_lens`.

    The read is `paged_attention_batched_reference` for now, the same way Week 6
    started from a plain-torch reference before Day 22's tlsim model and Day 23's
    `triton.jit` kernel. Lowering the batched read is the next step; the reference is
    the oracle it will be graded against.
    """

    def __init__(self, config, allocator: BlockAllocator, batch_size: int):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.config = config
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.num_layers = config.num_hidden_layers
        self.batch_size = batch_size
        # One table per sequence, all drawing on the same pool. The tables are the
        # only thing that is per-row: the physical blocks are common property.
        self.tables = [BlockTable(allocator) for _ in range(batch_size)]
        self.k_pool: list[torch.Tensor | None] = [None] * self.num_layers
        self.v_pool: list[torch.Tensor | None] = [None] * self.num_layers
        # This step's per-row destination slots, computed once on layer 0 and reused
        # by the rest (every layer stores the same tokens at the same positions).
        self._step_slots: list[torch.Tensor] | None = None
        # The read's `[batch, max_ctx]` addressing, rebuilt whenever the tables grow.
        self._mapping: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def seq_lens(self) -> list[int]:
        """Cached tokens per row. Ragged by construction: no padding is stored."""
        return [table.num_tokens for table in self.tables]

    @property
    def cached_tokens(self) -> int:
        """Total real tokens held across every row, i.e. slots actually in use."""
        return sum(self.seq_lens)

    def _reserve(self, counts: list[int]) -> None:
        """Check the pool can take every row's growth *before* any row grows.

        `BlockTable.append` is already atomic for one sequence, and with N sequences
        that is not enough: rows 0 and 1 can succeed and row 2 find the pool dry,
        which leaves the batch half written, the tables disagreeing with the tokens
        actually stored, and blocks reserved for a step that never happened. So the
        whole batch's block demand is summed and checked once, here.

        This is also the exact signal the Week-8 scheduler will act on rather than
        propagate: it means "this batch does not fit", and the answer is to admit
        fewer requests or preempt a running one, not to crash.
        """
        need = 0
        for table, n_new in zip(self.tables, counts):
            need += self.allocator.blocks_for_length(table.num_tokens + n_new) - len(
                table.block_ids
            )
        if need > self.allocator.num_free:
            raise KVCacheExhausted(
                f"the batch needs {need} more blocks, only {self.allocator.num_free} free "
                f"(rows hold {self.seq_lens}, adding {counts})"
            )

    def write(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor | None = None
    ) -> None:
        """Store this step's K/V into each row's own blocks. No read.

        k, v: [batch, num_kv_heads, seq, head_dim]. `seq` is the padded prompt length
              on a prefill and 1 on a decode step.
        valid: optional [batch, seq] bool, True where a column is a real token. This
              is the `PaddedBatch` key mask, and passing it is what makes the write
              ragged: row i contributes `valid[i].sum()` tokens to its table, taken in
              column order, so left and right padding both work. `None` means every
              column is real, which is the decode case.

        On layer 0 the tables grow (atomically across the whole batch, see `_reserve`)
        and each row's destination slots are recorded; every layer then scatters its
        own K/V into those same slots in its own pool. The scatter is a Python loop
        over rows because the rows write different counts to different places, which
        is honest for a reference and is exactly the loop a real engine flattens into
        one `[total_new_tokens]` slot vector.
        """
        if k.ndim != 4 or k.shape != v.shape:
            raise ValueError(
                "k and v must both be [batch, num_kv_heads, seq, head_dim]; got "
                f"{tuple(k.shape)} and {tuple(v.shape)}"
            )
        batch, _, seq, _ = k.shape
        if batch != self.batch_size:
            raise ValueError(
                f"this cache holds {self.batch_size} sequences, got a batch of {batch}"
            )
        if valid is not None and tuple(valid.shape) != (batch, seq):
            raise ValueError(
                f"valid must be the batch's [batch, seq] key mask = {(batch, seq)}; "
                f"got {tuple(valid.shape)}"
            )

        if layer == 0:
            counts = (
                [seq] * batch if valid is None else [int(n) for n in valid.sum(dim=1).tolist()]
            )
            self._reserve(counts)  # all rows fit, or none of them moves
            slots = []
            for table, n_new in zip(self.tables, counts):
                start = table.num_tokens
                table.append(n_new)
                slots.append(
                    torch.tensor(
                        [table.slot(p) for p in range(start, table.num_tokens)],
                        dtype=torch.long,
                        device=k.device,
                    )
                )
            self._step_slots = slots
            self._mapping = None  # the tables grew; the read's addressing is stale
        if self._step_slots is None:
            raise ValueError("layer 0 must be written first: it is what grows the tables")

        if self.k_pool[layer] is None:
            pool = (
                self.allocator.num_blocks * self.block_size,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            self.k_pool[layer] = torch.zeros(pool, dtype=k.dtype, device=k.device)
            self.v_pool[layer] = torch.zeros(pool, dtype=v.dtype, device=v.device)

        for row in range(batch):
            cols = (
                torch.arange(seq, device=k.device)
                if valid is None
                else valid[row].nonzero(as_tuple=True)[0]
            )
            if cols.numel() == 0:
                continue
            # [n_kv, n_new, d] -> [n_new, n_kv, d], scattered to this row's slots.
            self.k_pool[layer][self._step_slots[row]] = k[row][:, cols, :].transpose(0, 1)
            self.v_pool[layer][self._step_slots[row]] = v[row][:, cols, :].transpose(0, 1)

    def slot_mapping(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The read's addressing: `[batch, max_ctx]` slots and `[batch]` real lengths.

        Ragged histories, one rectangle, because a kernel wants a tensor with a stride
        and not a list of lists. Row i's first `context_lens[i]` entries are its own
        slots, oldest first; the rest are padding.

        The padding is slot 0, and the choice matters more than it looks. This is not
        a "won't be read" value: the reference gathers the whole rectangle and masks
        afterwards, so every padded entry is dereferenced. It has to be a legal index,
        and it must not be -1, which torch would silently wrap onto the last slot of
        the pool rather than reject. Zero is always in range, and whatever it holds is
        masked away by `context_lens`.

        Rebuilt only when the tables have grown (`write` on layer 0 invalidates it),
        so the 15 remaining layers of a decode step reuse one construction.
        """
        lens = self.seq_lens
        if max(lens) == 0:
            raise ValueError("nothing is cached yet: write a prefill before reading")
        stale = self._mapping is None or (
            device is not None and self._mapping[0].device != torch.device(device)
        )
        if stale:
            max_ctx = max(lens)
            rows = [
                [table.slot(p) for p in range(table.num_tokens)] + [0] * (max_ctx - n)
                for table, n in zip(self.tables, lens)
            ]
            self._mapping = (
                torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(lens, dtype=torch.long, device=device),
            )
        return self._mapping

    def paged_attention(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        n_rep: int,
        scale: float | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The batched fused read: store one new token per row, attend per row.

        Same contract as `PagedKVCache.paged_attention`, one axis wider. Write this
        step's K/V into each row's own blocks, then attend, each row over exactly its
        own `context_lens[i]` slots of the shared pool.

        k, v: [batch, num_kv_heads, 1, head_dim] this step's compact K/V.
        q:    [batch, num_attention_heads, 1, head_dim] this step's rotated queries.
        attention_mask: must be None. A key mask is what a padded batch needed when
              the pads were sitting in the K/V; per-sequence tables mean no pad was
              ever written, so a mask here would either be redundant or be hiding a
              bug, and both are worth refusing over.

        Returns [batch, num_attention_heads, 1, head_dim], the attention output
        before o_proj, with row i equal to what row i's own `PagedKVCache` would give.
        """
        if attention_mask is not None:
            raise ValueError(
                "the batched paged read takes no key mask: per-sequence tables hold "
                "only real tokens, so the padding is a context length, not a mask"
            )
        if k.shape[2] != 1:
            raise ValueError(
                f"the batched paged read is the decode read: one new token per row, "
                f"got {k.shape[2]}. A ragged prefill writes and attends densely"
            )
        self.write(layer, k, v)
        slot_mapping, context_lens = self.slot_mapping(q.device)
        return paged_attention_batched_reference(
            q, self.k_pool[layer], self.v_pool[layer], slot_mapping, context_lens, n_rep, scale
        )

    def free(self) -> None:
        """Return every row's blocks to the pool and reset to an empty, reusable cache."""
        for table in self.tables:
            table.free()
        self.k_pool = [None] * self.num_layers
        self.v_pool = [None] * self.num_layers
        self._step_slots = None
        self._mapping = None
