"""The launcher: a directory of weights and a card, turned into a running engine. Week 10, Day 38.

`serving.py` and `server.py` take their engine and their tokenizer by injection,
and have therefore never met Llama-3.2-1B. Everything they are tested against is a
two-layer random model over a 64-symbol alphabet with `num_blocks=64` written into
a fixture. That is deliberate and it is also the last piece missing: somebody has
to load the real weights, decide how big the pool is, build the real tokenizer, and
hand the three of them to `create_app`. This file is that somebody.

The interesting half is one question with a number for an answer: **how many blocks
fit?** Every day until now took `num_blocks` as an argument and believed it. A
launcher cannot. It gets a card with a fixed number of bytes on it and has to turn
that into an integer, and if it gets the integer wrong the failure is not a
traceback, it is a server that runs at a quarter of the concurrency it could, or
one that dies mid-forward under load.

**A block's size is arithmetic over the config, and GQA is a factor of four in it.**
A block holds `block_size` tokens of K *and* V, for every layer, for every *KV*
head:

    2 * num_hidden_layers * block_size * num_key_value_heads * head_dim * itemsize

For Llama-3.2-1B at block_size 16 in bf16 that is exactly 512 KiB, which makes the
rest of the sizing readable: a gibibyte of KV is 2048 blocks, and 2048 blocks is
32768 tokens. Use `num_attention_heads` there and every number is 4x too big, the
pool comes out a quarter of the size the card could hold, and nothing raises: the
server is correct, slower, and preempts under a load it should have absorbed. The
whole reason Week 5 built GQA is that 8 KV heads is 4x less cache than 32, and this
is the line where that saving is either taken or thrown away.

**The budget is what is free after the weights land, minus what a forward will
transiently want.** Not the card's total, and not "total minus weights" either.
Two subtractions matter and only one of them is obvious:

    budget = free_after_load - (1 - utilization) * total - activation_bytes

`utilization` is vLLM's `gpu_memory_utilization` and it is a promise about the
*whole device*, not a fraction of what happens to be free, which is why it is
multiplied by `total`: a co-tenant process comes out of your budget rather than out
of your headroom, and that is the conservative direction. `activation_bytes` is the
one people leave out. It is measured here, by running the largest forward this
server can be asked for and reading the allocator's peak (`profile_prefill_bytes`),
which is what vLLM's profile run does, and it can be estimated on paper
(`estimate_activation_bytes`) for a box with no CUDA to profile on. Either way the
number is large: at 8 requests x 2048 tokens Llama-3.2-1B wants about 6.7 GiB of
transient tensors against 2.3 GiB of weights, and the single biggest line in it is
the full-sequence logits rectangle, 3.9 GiB of a 128k vocab for every position of
every row, of which the engine reads one row per sequence and discards the rest.

**A pool that cannot hold one request is a server that refuses everything.** Day
33's `Scheduler.add_request` rejects a request whose worst case exceeds the whole
pool, because FIFO admission would otherwise park behind it forever. So a pool
sized under `blocks_for_length(max_model_len)` produces a process whose health
check says ok and whose every completion is a 400. `plan_kv_pool` refuses to boot
there, and says what length it could have served, so the flag to change is obvious
from the failure.

Two smaller things this file owns because nobody else can.

**`max_model_len` is a serving decision, not a model property.** The config says
131072, and reserving a pool that can hold one such request is 8192 blocks, 4 GiB,
before a second caller exists. So the length is a parameter, clamped to what the
model was actually trained for, and the plan reports how many requests of that
length fit at once.

**Moving the weights to the device must not break the tie.** Llama-3.2-1B ships no
`lm_head.weight`; the loader aliases it to `embed_tokens.weight`, one storage under
two names. The obvious move, `{k: v.to(device) for k, v in weights.items()}`, calls
`.to` twice on the same tensor and gets two copies, which is 501 MiB of duplicated
embedding on the card. That is 1002 blocks, 16032 tokens of KV, spent on a matrix
you already had, and the only symptom is a smaller pool. `place_weights` keys on
storage so a shared one is moved once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .captured import DEFAULT_CAPTURE_LIMIT, shared_pool_bytes
from .compiled import DecodeShape
from .reads import DEFAULT_BLOCK
from .config import ModelConfig
from .engine import Engine
from .loader import EMBED, Weights, load_weights
from .model import LlamaModel
from .server import create_app
from .serving import AsyncEngine
from .warmup import WarmupReport, warm_budget_bytes, warm_shapes, width_ceiling


class PoolTooSmall(RuntimeError):
    """The bytes left over cannot hold a pool this server could honestly serve from.

    Raised at launch, on purpose, rather than survived. A pool below the floor is
    not a degraded server, it is a server that admits nothing: `Scheduler`
    rejects any request whose worst case exceeds the whole pool, so every caller
    gets a 400 and `/health` says ok. Failing at boot puts the error where the
    person who chose the numbers is standing.
    """


class CaptureTooSmall(RuntimeError):
    """The capture list cannot be trimmed to anything this process could record.

    Day 56's `PoolTooSmall`, raised for the same reason: what is left over does not
    hold the *smallest* member of the thing being sized. A byte budget under one
    width bucket is not a server without graphs, it is an out-of-memory error
    scheduled for the first decode, because a lazy recording wants the same arena in
    front of a client rather than at boot.
    """


class BootUnsound(AssertionError):
    """A boot step taken in an order, or against an object, that makes it a lie.

    An assertion rather than an error, because every one of these is a thing
    `build_app` does correctly by construction. They are checked so that the next
    caller who wires a process by hand finds out at the call instead of in a token.
    """


# --- what a block costs ----------------------------------------------------------


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def kv_bytes_per_block(config: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
    """Bytes one physical block occupies, across every layer's K and V pool.

    The unit the whole launcher divides by. `num_key_value_heads`, not
    `num_attention_heads`: the cache stores what attention *reads*, and under GQA
    that is one K/V per group of query heads. Getting this wrong is a factor of
    `num_kv_groups` (4 on Llama-3.2-1B) applied to every number downstream, in the
    direction that silently shrinks the pool.
    """
    return (
        2  # K and V
        * config.num_hidden_layers
        * block_size
        * config.num_key_value_heads
        * config.head_dim
        * _itemsize(dtype)
    )


# --- what a forward transiently wants --------------------------------------------


@dataclass(frozen=True)
class ActivationEstimate:
    """The three rectangles a worst-case prefill puts on the card at once.

    An estimate, not a measurement, and it exists for the box that cannot profile:
    `torch.cuda.max_memory_allocated` is the honest number and it needs CUDA.
    These three terms are the ones that scale with the batch rectangle and dwarf
    everything else; the per-layer hidden states and the residual stream are
    linear in `batch * len * hidden` and small beside them.
    """

    scores_bytes: int
    mlp_bytes: int
    logits_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.scores_bytes + self.mlp_bytes + self.logits_bytes


def estimate_activation_bytes(
    config: ModelConfig, *, max_batch_size: int, max_model_len: int, dtype: torch.dtype
) -> ActivationEstimate:
    """Paper estimate of the peak transient allocation of the largest prefill.

    scores: `[batch, heads, len, len]`, quadratic in the prompt length and the
            reason a long-context server reserves so much. Query heads here, not
            KV heads: GQA saves cache, not attention arithmetic.
    mlp:    three `[batch, len, intermediate]` rectangles live at once, because
            SwiGLU holds the gate projection, the up projection and their product
            before `down` collapses them.
    logits: `[batch, len, vocab]`. The largest of the three on any real vocab, and
            the one that is pure overhead here: `LlamaModel.forward` returns logits
            for every position so a prefill can be diffed against HF token for
            token, and the engine reads one row per sequence.
    """
    itemsize = _itemsize(dtype)
    b, length = max_batch_size, max_model_len
    return ActivationEstimate(
        scores_bytes=b * config.num_attention_heads * length * length * itemsize,
        mlp_bytes=3 * b * length * config.intermediate_size * itemsize,
        logits_bytes=b * length * config.vocab_size * itemsize,
    )


def measure_activation_bytes(run, *, device=None, reset=None, peak=None) -> int:
    """Run something and report how much *more* memory it peaked at than it started.

    The subtraction is the point. `max_memory_allocated` is a high-water mark over
    everything the allocator holds, so the resident weights are inside it; reading
    it straight after the forward would charge the activation budget for the model
    a second time and shrink the pool by exactly the size of the weights.

    `reset` and `peak` are injected so the arithmetic can be tested on a box with
    no CUDA, which is also the box this project is written on.
    """
    if reset is None:
        reset = lambda: torch.cuda.reset_peak_memory_stats(device)  # noqa: E731
    if peak is None:
        peak = lambda: torch.cuda.max_memory_allocated(device)  # noqa: E731
    reset()
    before = peak()
    run()
    return max(0, peak() - before)


def profile_prefill_bytes(
    model,
    *,
    max_batch_size: int,
    max_model_len: int,
    device,
    pad_id: int = 0,
    reset=None,
    peak=None,
) -> int:
    """Measure the peak of the biggest forward the scheduler can produce.

    vLLM calls this the profile run and it is the only honest way to get the
    number: activation memory depends on the kernels the shapes actually dispatch
    to, and no formula tracks that across a torch version bump.

    Run with `cache=None`, which is the Week-2 recompute path, because the pool
    this is sizing does not exist yet. That is the right shape anyway: a prefill
    attends over exactly its own context whether it reads it from a cache or
    recomputes it, so the score rectangle, the MLP rectangles and the logits are
    the same. What it misses is the cache write itself, which is a scatter into
    the pool being sized and allocates nothing new.
    """

    def run() -> None:
        ids = torch.full(
            (max_batch_size, max_model_len), pad_id, dtype=torch.long, device=device
        )
        positions = torch.arange(max_model_len, device=device).repeat(max_batch_size, 1)
        model.forward(ids, positions)

    return measure_activation_bytes(run, device=device, reset=reset, peak=peak)


# --- the budget ------------------------------------------------------------------


def kv_budget_bytes(
    device,
    *,
    activation_bytes: int,
    utilization: float = 0.90,
    probe=None,
) -> int:
    """Bytes available for the KV pool, given a device already holding the weights.

    Call this *after* the model is on the card. `free` is then the honest number:
    weights, CUDA context, allocator fragmentation and any co-tenant are all
    already subtracted by the driver, and none of them need to be modelled here.

    `utilization` is a reserve against the device *total*, matching vLLM's
    `gpu_memory_utilization`. A fraction of what is free would mean a card shared
    with another process quietly hands this engine a smaller absolute headroom
    exactly when it needs a larger one; against the total, a co-tenant eats into
    the budget and the safety margin stays the size it was chosen to be.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            f"cannot size a KV pool by probing a {device.type} device: there is no "
            "per-device free-memory number to divide, and taking a fraction of "
            "system RAM is a promise the allocator cannot keep. Pass an explicit "
            "kv_cache_bytes (or num_blocks) instead"
        )
    if probe is None:
        probe = torch.cuda.mem_get_info
    free, total = probe(device)
    reserve = int(total * (1.0 - utilization))
    return max(0, free - reserve - activation_bytes)


# --- the plan --------------------------------------------------------------------


@dataclass(frozen=True)
class KVPoolPlan:
    """The sizing decision, as a value a human can read and a test can assert on.

    Kept as a record rather than being applied straight to an `Engine` because it
    is the one number in this project that cannot be derived from the code, and a
    server that will not start should be able to explain itself in one line
    without having built anything.
    """

    num_blocks: int
    block_size: int
    max_batch_size: int
    max_model_len: int
    bytes_per_block: int
    budget_bytes: int
    dtype: torch.dtype

    @property
    def pool_bytes(self) -> int:
        """What the pool actually takes, which is the budget rounded down."""
        return self.num_blocks * self.bytes_per_block

    @property
    def capacity_tokens(self) -> int:
        """Total tokens of K/V the pool can hold, over all sequences at once."""
        return self.num_blocks * self.block_size

    @property
    def blocks_per_request(self) -> int:
        """Blocks one request at `max_model_len` needs. The admission floor."""
        return math.ceil(self.max_model_len / self.block_size)

    @property
    def concurrency_at_max_len(self) -> int:
        """Requests of the full length that fit at once, before preemption starts.

        Not clamped to `max_batch_size` on purpose: the two limits bind
        independently, and knowing which one binds first is the whole reason to
        print this. Below the slot count, the pool is the constraint and Day 33's
        preemption is the load-shedding mechanism; above it, the slots are.
        """
        return self.num_blocks // self.blocks_per_request

    def as_dict(self) -> dict:
        """The shape `/health` reports, so the pool is visible without a restart."""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "max_model_len": self.max_model_len,
            "max_batch_size": self.max_batch_size,
            "kv_pool_bytes": self.pool_bytes,
            "capacity_tokens": self.capacity_tokens,
            "kv_dtype": str(self.dtype).replace("torch.", ""),
        }

    def describe(self) -> str:
        """One line, printed at boot, holding every number that was chosen for you."""
        gib = self.pool_bytes / 1024**3
        return (
            f"KV pool: {self.num_blocks} blocks x {self.block_size} tokens = "
            f"{self.capacity_tokens} tokens ({gib:.2f} GiB of "
            f"{str(self.dtype).replace('torch.', '')}), "
            f"{self.concurrency_at_max_len} concurrent requests at "
            f"{self.max_model_len} tokens, {self.max_batch_size} slots"
        )


def plan_kv_pool(
    config: ModelConfig,
    *,
    block_size: int,
    max_batch_size: int,
    max_model_len: int,
    dtype: torch.dtype,
    budget_bytes: int | None = None,
    num_blocks: int | None = None,
) -> KVPoolPlan:
    """Turn a byte budget into a block count, or refuse and say why.

    Exactly one of `budget_bytes` and `num_blocks` is given. The override exists
    because a benchmark wants the same pool on every box and should not have to
    care what card it landed on; it still goes through the floor check, since a
    hand-picked number is at least as capable of being too small as a computed one.

    The floor is `Scheduler.add_request`'s rule read backwards. It refuses a
    request needing more blocks than the pool has, so a pool below
    `blocks_for_length(max_model_len)` accepts no request of that length, ever,
    and a server whose advertised context cannot be served is lying in its own
    error messages.
    """
    if (budget_bytes is None) == (num_blocks is None):
        raise ValueError("give exactly one of budget_bytes and num_blocks")

    # A serving decision, bounded by a model fact. Asking for more context than the
    # weights were trained for is not a memory question and cannot be bought.
    max_model_len = min(max_model_len, config.max_position_embeddings)

    bytes_per_block = kv_bytes_per_block(config, block_size, dtype)
    if num_blocks is None:
        num_blocks = budget_bytes // bytes_per_block
    else:
        budget_bytes = num_blocks * bytes_per_block

    if num_blocks < 1:
        raise PoolTooSmall(
            f"{budget_bytes} bytes is not one {bytes_per_block}-byte block: there is "
            "nothing left for the KV cache after the weights and the forward. Lower "
            "max_batch_size or max_model_len, or raise utilization"
        )

    plan = KVPoolPlan(
        num_blocks=int(num_blocks),
        block_size=block_size,
        max_batch_size=max_batch_size,
        max_model_len=max_model_len,
        bytes_per_block=bytes_per_block,
        budget_bytes=int(budget_bytes),
        dtype=dtype,
    )
    if plan.num_blocks < plan.blocks_per_request:
        raise PoolTooSmall(
            f"a pool of {plan.num_blocks} blocks holds {plan.capacity_tokens} tokens, "
            f"and one request at max_model_len={plan.max_model_len} needs "
            f"{plan.blocks_per_request}. Every completion would be refused by the "
            f"scheduler. Serve max_model_len={plan.capacity_tokens} or smaller, or "
            "give the pool more bytes"
        )
    return plan


# --- device, dtype, and getting the weights there --------------------------------


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(spec: str = "auto", cuda_available=None) -> torch.device:
    """"auto" means the GPU if there is one. Anything else is taken literally."""
    if spec != "auto":
        return torch.device(spec)
    if cuda_available is None:
        cuda_available = torch.cuda.is_available
    return torch.device("cuda" if cuda_available() else "cpu")


def resolve_dtype(spec: str, device: torch.device, config: ModelConfig) -> torch.dtype:
    """The dtype the weights and the KV pool are held in.

    "auto" is the config's on a GPU (bf16, which is what the weights were published
    in) and float32 on a CPU. The CPU branch is not caution, it is the reason every
    earlier day in this project loaded fp32: bf16 matmul on CPU falls off the fast
    path and the whole test suite would slow to a crawl for no accuracy gained.
    """
    if spec == "auto":
        spec = config.torch_dtype if device.type == "cuda" else "float32"
    if spec not in _DTYPES:
        raise ValueError(f"unknown dtype {spec!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[spec]


def weights_bytes(weights: Weights) -> int:
    """Bytes the loaded weights occupy, counting a shared storage once.

    The same rule as `Weights.num_params` and for the same reason: the tied output
    projection is one matrix under two names, and adding it up per name reports
    501 MiB the card is not holding.
    """
    seen: dict[int, int] = {}
    for name in weights.keys():
        tensor = weights[name]
        seen[tensor.data_ptr()] = tensor.numel() * tensor.element_size()
    return sum(seen.values())


def place_weights(
    weights: Weights, device: torch.device, dtype: torch.dtype | None = None
) -> Weights:
    """Move (and optionally cast) every tensor, moving a shared storage once.

    The whole function is the dict on the second line. `{k: v.to(device) for k, v
    in ...}` calls `.to` once per *name*, and `lm_head.weight` and
    `embed_tokens.weight` are two names for one tensor, so the tie becomes two
    independent copies on the card. On Llama-3.2-1B that is 501 MiB, which at 512
    KiB a block is 1002 blocks and 16032 tokens of KV pool, lost to a duplicate of
    a matrix that was already there. Nothing detects it: the model is correct, the
    logits are identical, the pool is just smaller than it should be.
    """
    moved: dict[tuple, torch.Tensor] = {}
    tensors: dict[str, torch.Tensor] = {}
    for name in weights.keys():
        tensor = weights[name]
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        if key not in moved:
            moved[key] = tensor.to(device=device, dtype=dtype)
        tensors[name] = moved[key]
    return Weights(tensors, weights.config)


# --- the second sizing decision: which shapes get a graph -------------------------


@dataclass(frozen=True)
class CapturePlan:
    """The capture list this process will hold, as a value a human can read.

    `KVPoolPlan`'s sibling, and the resemblance is the point: both turn numbers that
    arrive from different places into one integer nothing in the code can be read
    off, and both are kept as a record so a server that will not start can explain
    itself in a line without having built anything.

    What is different is *when*. The pool is planned before a block exists, off a
    probe of a card holding only the weights. The list is planned after that, and its
    own budget is a second probe with the pool already spoken for, which is why this
    is a separate record made by a separate call rather than three more fields on the
    first one.

    shapes:         the list, biggest first, exactly as `warm_decode` will walk it.
    max_rows:       the row ceiling. The scheduler's slot count unless a flag is
                    lower, because no batch wider than a slot count can be presented.
    max_width:      the context ceiling, and the one number the other three
                    constraints all turn into.
    width_bound_by: which constraint set it. "served", "flag", "budget" or "limit".
    num_heads:      query heads, which is what a score rectangle is counted in.
    full_count:     shapes in the untrimmed bucket set, so the trim is visible.
    budget_bytes:   what the probe said was left, or None when nobody asked.
    limit:          graphs this process will hold at all.
    """

    shapes: tuple[DecodeShape, ...]
    max_rows: int
    max_width: int
    width_bound_by: str
    num_heads: int
    full_count: int
    budget_bytes: int | None = None
    limit: int = DEFAULT_CAPTURE_LIMIT

    @property
    def count(self) -> int:
        return len(self.shapes)

    @property
    def trimmed(self) -> int:
        """Shapes the bucket set holds that this list will not record."""
        return self.full_count - self.count

    @property
    def widest(self) -> DecodeShape:
        """The shape the arena is sized by. First, because the list is descending."""
        return self.shapes[0]

    @property
    def pool_bytes(self) -> int:
        """What the shared capture arena costs: a max over the list, not a sum."""
        return shared_pool_bytes(self.shapes, self.num_heads)

    def as_dict(self) -> dict:
        """The shape `/health` reports, under its own key rather than beside the pool."""
        return {
            "shapes": self.count,
            "shapes_in_set": self.full_count,
            "max_rows": self.max_rows,
            "max_width": self.max_width,
            "width_bound_by": self.width_bound_by,
            "workspace_bytes": self.pool_bytes,
            "capture_limit": self.limit,
        }

    def describe(self) -> str:
        """One line, printed at boot, next to the pool's."""
        why = {
            "served": "the context this server sells",
            "flag": "the width you asked for",
            "budget": "what the card had left",
            "limit": "the graph limit",
        }[self.width_bound_by]
        return (
            f"CUDA graphs: {self.count} of {self.full_count} shapes, rows <= "
            f"{self.max_rows}, context <= {self.max_width} (capped by {why}), "
            f"{self.pool_bytes / 1024**2:.1f} MiB of workspace"
        )


def width_from_limit(buckets, *, row_count: int, limit: int) -> int:
    """The widest context bucket that keeps the list inside `limit` graphs.

    Day 55 ended on "a budget is not a statement about the length of the capture
    list, it is a statement about one width", because a shared arena is sized by its
    largest member. The graph limit turns out to be the same kind of statement from
    the opposite direction: the list is `rows x widths`, so a cap on the *count* is a
    cap on how far up the width axis it may go. Two constraints with nothing in
    common reduce to the same knob, which is why `CapturePlan` has one ceiling and a
    field saying who set it.

    The width axis is truncated in whole rows. Keeping half a row would warm a shape
    for four rows and not for eight at the same context, and the eight-row shape is
    exactly what the scheduler presents the moment it admits one more request: the
    hole would be in the part of the list a busy server uses most.

    Dropping the *widest* widths rather than the narrowest is the direction that
    costs least. A run crosses the width axis from the bottom, so a shape left out at
    the top is recorded once, late, by a request that has already streamed thousands
    of tokens; a shape left out at the bottom is recorded in the first seconds of
    every request the server ever answers.
    """
    if row_count < 1:
        raise ValueError(f"a bucket set has at least one row bucket; got {row_count}")
    if limit < 1:
        raise ValueError(f"a capture holds at least one graph; got {limit}")
    keep = limit // row_count
    if keep < 1:
        raise CaptureTooSmall(
            f"a limit of {limit} graphs cannot hold one row bucket's worth of this "
            f"set, which is {row_count} rows x 1 width: every width bucket has to be "
            "recorded for every row bucket or the list has a hole in it at the "
            "context where the scheduler admits one more request"
        )
    return buckets.widths[min(keep, len(buckets.widths)) - 1]


def plan_capture(
    engine: Engine,
    plan: KVPoolPlan,
    *,
    max_rows: int | None = None,
    max_width: int | None = None,
    budget_bytes: int | None = None,
    utilization: float = 0.90,
    probe=None,
    device=None,
    limit: int | None = None,
) -> CapturePlan:
    """Decide which shapes this process records, from four numbers and three sources.

    The rows come from the scheduler, because a slot count is the widest batch that
    can ever be presented and nothing else in the process knows it. The width comes
    from the smallest of three things that each cap it, and the day's finding is that
    all three *are* width caps:

    served: `plan.max_model_len`, the context this deployment sells. The default, and
            the only one of the four that is a promise to a caller.
    flag:   what the operator asked for, when they know their traffic is shorter than
            their limit and would rather have the startup seconds back.
    budget: `width_ceiling` of what the card has left. Day 55's inversion.
    limit:  `width_from_limit` of how many graphs this process will hold.

    The budget is probed here rather than taken from `build_engine`'s, and the
    difference is the whole reason this is a second call. `kv_budget_bytes` ran on a
    card holding the weights; this one runs on a card that is also about to hold the
    pool. It is not holding it *yet*: `BatchedPagedKVCache` allocates a layer's pool
    on its first write, so at this moment the driver cannot see it and would report
    it as free. That is what `reserved_bytes` is for, and `plan.pool_bytes` is
    exactly the number it wants.
    """
    buckets = getattr(engine.cache, "decode_buckets", None)
    if buckets is None:
        raise ValueError(
            "this engine was built without bucket_decode, so its decode steps do not "
            "land on a closed set of shapes and there is no list to capture: a graph "
            "per distinct shape is a graph per step"
        )
    limit = engine.decode_graphs.limit if limit is None else limit
    num_heads = engine.model.config.num_attention_heads

    rows = plan.max_batch_size if max_rows is None else min(max_rows, plan.max_batch_size)
    row_count = sum(1 for r in buckets.rows if r <= rows)
    if row_count < 1:
        raise CaptureTooSmall(
            f"no row bucket in this set is within max_rows={rows}: the set rounds up, "
            "so a ceiling below the smallest bucket excludes every batch a step "
            "could present"
        )

    if budget_bytes is None and device is None:
        device = engine.model.weights[EMBED].device
    if budget_bytes is None and torch.device(device).type == "cuda":
        budget_bytes = warm_budget_bytes(
            device,
            reserved_bytes=plan.pool_bytes,
            utilization=utilization,
            probe=probe,
        )

    # Four candidates, strictly improving, so a ceiling that ties with the served
    # context is not reported as the one that bit. A boot line blaming a flag that
    # changed nothing sends somebody to the wrong knob.
    candidates = [("served", plan.max_model_len)]
    if max_width is not None:
        candidates.append(("flag", int(max_width)))
    if budget_bytes is not None:
        candidates.append(
            ("budget", width_ceiling(rows=rows, num_heads=num_heads, budget_bytes=budget_bytes))
        )
    candidates.append(("limit", width_from_limit(buckets, row_count=row_count, limit=limit)))
    bound_by, width = candidates[0]
    for name, value in candidates[1:]:
        if value < width:
            bound_by, width = name, value

    # Down to a bucket, because the ceiling is a number and the list is a set. A
    # budget that buys 300 tokens of context buys the 256 bucket, and reporting 300
    # would name a width no graph in the list was recorded at.
    snapped = [w for w in buckets.widths if w <= width]
    width = snapped[-1] if snapped else width

    if width < buckets.widths[0]:
        raise CaptureTooSmall(
            f"a context ceiling of {width} tokens ({bound_by}) is under this set's "
            f"narrowest width bucket of {buckets.widths[0]}: there is no shape left to "
            "record, and the first real decode would want the same arena in front of "
            "a client. Raise the budget, or build without capture_decode"
        )
    shapes = warm_shapes(buckets, max_rows=rows, max_width=width)
    return CapturePlan(
        shapes=shapes,
        max_rows=rows,
        max_width=width,
        width_bound_by=bound_by,
        num_heads=num_heads,
        full_count=len(buckets.shapes),
        budget_bytes=budget_bytes,
        limit=limit,
    )


def warm_engine(
    engine: Engine, capture: CapturePlan, *, device=None, serving=None
) -> WarmupReport:
    """Record the planned list now, before anything is serving. Day 56.

    Three lines of work and two gates, and the gates are the content. A warm-up
    writes the persistent input buffers and reads a window on the slot table, which
    are the same addresses a real decode step uses, so a walk taken while the bridge
    is running is two writers on one buffer and the symptom is a token rather than a
    traceback. `build_app` cannot reach that state, because it warms before the
    `AsyncEngine` exists; `serving` is here for a caller who warms a process that has
    one already.
    """
    check_capture_matches_cache(capture, engine.cache)
    if serving is not None:
        check_warm_before_serving(serving)
    check_capture_limit(capture, engine.decode_graphs)
    return engine.warm_decode(capture.shapes, device=device)


# --- what the process says about itself -------------------------------------------


def boot_info(
    plan: KVPoolPlan, capture: CapturePlan | None = None, report: WarmupReport | None = None
) -> dict:
    """The `/health` payload: every decision this launch made on somebody's behalf.

    The capture decision is nested rather than merged. Flattened, `max_width` would
    sit next to `max_model_len` and invite the reading that one is derived from the
    other, when in fact they are a promise to a caller and a memory ceiling that
    happen to be measured in the same unit.

    The warm-up's numbers go in the same place because "how many shapes did you
    decide to record" and "how many did you actually record" are only useful next to
    each other. A server whose list is 60 and whose graphs are 0 is a server that was
    started with warming off, and that is a thing you want to find out at 3am from a
    health check rather than from a latency histogram.
    """
    info = plan.as_dict()
    if capture is not None:
        graphs = capture.as_dict()
        if report is not None:
            graphs.update(
                graphs_held=report.graphs,
                cold=len(report.cold),
                warmup_seconds=round(report.seconds, 3),
                ms_per_graph=round(report.per_capture_s * 1e3, 1),
            )
        info["cuda_graphs"] = graphs
    return info


def boot_lines(
    plan: KVPoolPlan, capture: CapturePlan | None = None, report: WarmupReport | None = None
) -> tuple[str, ...]:
    """What `serve.py` prints, built here so the CLI stays flags and `uvicorn.run`."""
    lines = [plan.describe()]
    if capture is not None:
        lines.append(capture.describe())
    if report is not None:
        lines.append(report.render())
    return tuple(lines)


# --- gates -------------------------------------------------------------------------


def check_warm_before_serving(serving) -> None:
    """Refuse a warm-up behind a bridge that is already stepping.

    The one ordering rule of the boot path. Day 53 moved the step's inputs into
    buffers that do not move, which is what makes a replay possible and also means
    every decode in the process writes the same four tensors. A warm batch writes
    them too. Run the two at once and the graph records a step whose `input_ids` were
    overwritten halfway through by a real one, or a real step reads the warm batch's
    made-up tokens, and neither of those raises anything.
    """
    if getattr(serving, "running", False):
        raise BootUnsound(
            "this bridge's loop is already running, so a warm-up would write the "
            "persistent decode buffers underneath a step in flight: warm between "
            "building the engine and starting the loop, which is the window "
            "build_app does it in"
        )


def check_capture_matches_cache(capture: CapturePlan, cache) -> None:
    """Refuse a list that was planned against some other cache's bucket set.

    Two arguments, no type says they belong together, and a mismatch is not a
    tidiness problem: a shape wider than this cache's slot table is a rectangle that
    is not a window on anything, and a shape with more rows than the cache has rows
    is a plan addressing slots that do not exist.
    """
    buckets = getattr(cache, "decode_buckets", None)
    if buckets is None:
        raise BootUnsound(
            "this cache has no bucket set, so no capture list belongs to it: a shape "
            "it never rounds to is a graph no step will ever replay"
        )
    if capture.max_rows > cache.batch_size:
        raise BootUnsound(
            f"this list is planned for up to {capture.max_rows} rows and the cache "
            f"has {cache.batch_size}: a plan cannot address a slot that does not exist"
        )
    stray = [s for s in capture.shapes if s not in buckets.shapes]
    if stray:
        raise BootUnsound(
            f"{len(stray)} shape(s) in this list, {stray[0]} first, are not in this "
            "cache's bucket set: the list was planned against a different cache, and "
            "a shape this one never rounds to is a graph no step will replay"
        )


def check_capture_limit(capture: CapturePlan, captured) -> None:
    """Refuse a list longer than the graphs this process will hold.

    `CapturedDecode` refuses past its limit, which mid-warm-up means a boot that
    fails on the 65th shape after paying for 64 recordings. Asked here it is a
    division. Graphs already held count, because warming is not always the first
    thing in the process to record one.
    """
    held = set(getattr(captured, "graphs", {}))
    total = len(held | set(capture.shapes))
    if total > captured.limit:
        raise BootUnsound(
            f"this list needs {total} graphs and the capture's limit is "
            f"{captured.limit}: trim it with max_rows or max_width, or raise the "
            "limit and pay the arena for it"
        )


def check_boot_info(info: dict) -> None:
    """Refuse a health payload that does not say what this process decided.

    Both failures it catches are a server that is up and lying about itself. One is
    a payload with no pool in it, which is the number nothing in the code can be read
    off. The other is subtler and is the reason this gate exists: a payload that says
    the graphs are on while some shape is still cold. Nothing else in the process
    would notice, because a cold shape is not an error, it is a recording that has
    not happened yet and will happen in front of whoever asks for that context first.
    """
    if "num_blocks" not in info:
        raise BootUnsound(
            "this payload names no pool: the block count is the one number a reader "
            "cannot derive, and it is the first thing you want when a server is "
            "preempting more than you expected"
        )
    graphs = info.get("cuda_graphs")
    if graphs is None:
        return
    if graphs.get("shapes", 0) < 1:
        raise BootUnsound(
            "this payload reports a capture with no shapes in it: an engine built "
            "with capture_decode and an empty list records a graph per step"
        )
    if "graphs_held" in graphs and graphs.get("cold"):
        raise BootUnsound(
            f"this payload says the capture is warm and {graphs['cold']} shape(s) are "
            "still cold: the server is up, the graphs are on, and somebody's first "
            "token at that context still pays for a recording"
        )


# --- wiring it together ----------------------------------------------------------


def build_engine(
    weights_dir: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    block_size: int = 16,
    max_batch_size: int = 8,
    max_model_len: int = 2048,
    utilization: float = 0.90,
    kv_cache_bytes: int | None = None,
    num_blocks: int | None = None,
    profile: bool = True,
    bucket_decode: bool = False,
    persist_inputs: bool = False,
    capture_decode: bool = False,
    compact_rows: bool = False,
    streamed_read: bool = False,
    read_block: int = DEFAULT_BLOCK,
    capture_recorder=None,
    load=load_weights,
    read_config=ModelConfig.from_json,
    probe=None,
    cuda_available=None,
) -> tuple[Engine, KVPoolPlan]:
    """Load the model onto a device and size a pool for what is left.

    The order is the design. Config, then device and dtype, then the weights *onto
    the card*, and only then the budget: `kv_budget_bytes` asks the driver what is
    free, and that answer is only worth anything once the thing that will occupy
    most of the card is occupying it. Sizing first and subtracting an estimate of
    the weights would reintroduce every term the driver already knows, including
    the CUDA context and the allocator's own fragmentation.

    `load` and `read_config` are injected so this function can be tested against a
    tiny random model, which is the same trick `create_app` uses for its engine and
    for the same reason: the wiring is what is under test, not the weights.

    Returns the engine and the plan, because the plan is what `/health` reports and
    what the boot line prints, and recovering it from the engine afterwards would
    mean rederiving a decision that was already made.

    Day 56 adds the three flags of Weeks 12 and 13, and Day 58 a fourth, passed
    straight through to `Engine.build` without being bundled. `Engine.build` refuses two of the three
    loudly and this function does not soften that: the missing halves are not
    performance settings, and a capture over an open shape set or a moving input is
    not a slower engine, it is a wrong one. The bundling lives in `serve.py`, one
    layer up, where `--cuda-graphs` is a thing an operator can reasonably mean.

    What it does *not* do is warm. The capture list is a second sizing decision made
    against a second probe, and it is `plan_capture` and `warm_engine`, called after
    this returns. See `build_app`.
    """
    resolved_device = resolve_device(device, cuda_available=cuda_available)
    config = read_config(weights_dir)
    resolved_dtype = resolve_dtype(dtype, resolved_device, config)

    weights = load(weights_dir, config=config, dtype=None)
    weights = place_weights(weights, resolved_device, resolved_dtype)
    model = LlamaModel(config, weights)

    if num_blocks is None and kv_cache_bytes is None:
        if resolved_device.type != "cuda":
            raise RuntimeError(
                "a CPU launch has no VRAM to divide: pass kv_cache_bytes or "
                "num_blocks explicitly"
            )
        activation = (
            profile_prefill_bytes(
                model,
                max_batch_size=max_batch_size,
                max_model_len=min(max_model_len, config.max_position_embeddings),
                device=resolved_device,
            )
            if profile
            else estimate_activation_bytes(
                config,
                max_batch_size=max_batch_size,
                max_model_len=min(max_model_len, config.max_position_embeddings),
                dtype=resolved_dtype,
            ).total_bytes
        )
        kv_cache_bytes = kv_budget_bytes(
            resolved_device,
            activation_bytes=activation,
            utilization=utilization,
            probe=probe,
        )

    plan = plan_kv_pool(
        config,
        block_size=block_size,
        max_batch_size=max_batch_size,
        max_model_len=max_model_len,
        dtype=resolved_dtype,
        budget_bytes=kv_cache_bytes if num_blocks is None else None,
        num_blocks=num_blocks,
    )
    engine = Engine.build(
        model,
        num_blocks=plan.num_blocks,
        block_size=plan.block_size,
        max_batch_size=plan.max_batch_size,
        # Day 51. The cache's persistent slot table is sized from the same length
        # the pool was planned around, not from the pool: `[max_batch_size,
        # num_blocks * block_size]` int64 would be hundreds of megabytes of pure
        # addressing on a real card. See `nanoserve.slots.check_table_fits`.
        max_model_len=plan.max_model_len,
        bucket_decode=bucket_decode,
        persist_inputs=persist_inputs,
        capture_decode=capture_decode,
        # Day 58. Passed through like the other three and bundled like none of them.
        # A persistent batch is what makes a capture cover the whole decode loop
        # instead of the quarter of it where the rows happened to line up, and it is
        # still a scheduler decision an operator can hold separately: `serve.py` is
        # where `--cuda-graphs` turns it on, one layer up. See `nanoserve.compact`.
        compact_rows=compact_rows,
        # Day 60. The fifth flag and the first that changes the arithmetic. Passed
        # through unbundled like the rest, and bundled by nothing at all: no other
        # flag implies it, because the streamed read is slower on this box and a
        # launcher that switched it on as a side effect of asking for CUDA graphs
        # would be trading correctness-preserving memory for wall clock without
        # saying so. See `nanoserve.reads`.
        streamed_read=streamed_read,
        read_block=read_block,
        capture_recorder=capture_recorder,
    )
    return engine, plan


def build_app(
    weights_dir: str | Path,
    *,
    model_name: str = "nanoserve",
    tokenizer=None,
    eos_token_id: int | None = None,
    max_idle_schedules: int = 256,
    warm: bool = True,
    warm_rows: int | None = None,
    warm_width: int | None = None,
    warm_bytes: int | None = None,
    **engine_kwargs,
):
    """The whole server, from a path. What `serve.py` calls and nothing else does.

    The tokenizer comes from the same directory as the weights, which is the only
    place it can honestly come from: a tokenizer that does not match the checkpoint
    produces fluent text out of the wrong ids, and no test downstream of here can
    tell. `eos_token_id` is read off it rather than accepted per request, because a
    stop token is a property of the model this process loaded and letting a caller
    choose one is how a request never stops.

    Day 56 puts three more steps between the engine and the app, and their *position*
    is the day. The list is planned after the pool, because its budget is what is left
    once the pool is spoken for. The warm-up runs after that and before the
    `AsyncEngine` is constructed, which is not a preference: a warm batch writes the
    same persistent decode buffers a real step does, so the only safe window is the
    one where no loop exists to race. This function cannot pass a `serving` to
    `warm_engine` because there is not one yet, and that is the proof the order is
    right rather than a comment claiming it.

    `warm=False` keeps the graphs and records them lazily, the Day-54 way. It is the
    control a benchmark needs and not a flag a deployment should want.
    """
    engine, plan = build_engine(weights_dir, **engine_kwargs)

    capture = report = None
    if engine.decode_graphs.mode != "off":
        capture = plan_capture(
            engine,
            plan,
            max_rows=warm_rows,
            max_width=warm_width,
            budget_bytes=warm_bytes,
            utilization=engine_kwargs.get("utilization", 0.90),
            probe=engine_kwargs.get("probe"),
        )
        if warm:
            report = warm_engine(engine, capture)

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    serving = AsyncEngine(engine, max_idle_schedules=max_idle_schedules)
    app = create_app(
        serving,
        tokenizer,
        model_name=model_name,
        eos_token_id=eos_token_id,
        vocab_size=engine.model.config.vocab_size,
        info=boot_info(plan, capture, report),
    )
    # Hung off the app so a test (and a debugger attached to a live server) can
    # reach the same objects the handlers are holding.
    app.state.plan = plan
    app.state.capture = capture
    app.state.warmup = report
    app.state.engine = engine
    app.state.serving = serving
    return app
