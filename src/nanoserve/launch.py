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

from .config import ModelConfig
from .engine import Engine
from .loader import Weights, load_weights
from .model import LlamaModel
from .server import create_app
from .serving import AsyncEngine


class PoolTooSmall(RuntimeError):
    """The bytes left over cannot hold a pool this server could honestly serve from.

    Raised at launch, on purpose, rather than survived. A pool below the floor is
    not a degraded server, it is a server that admits nothing: `Scheduler`
    rejects any request whose worst case exceeds the whole pool, so every caller
    gets a 400 and `/health` says ok. Failing at boot puts the error where the
    person who chose the numbers is standing.
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
    )
    return engine, plan


def build_app(
    weights_dir: str | Path,
    *,
    model_name: str = "nanoserve",
    tokenizer=None,
    eos_token_id: int | None = None,
    max_idle_schedules: int = 256,
    **engine_kwargs,
):
    """The whole server, from a path. What `serve.py` calls and nothing else does.

    The tokenizer comes from the same directory as the weights, which is the only
    place it can honestly come from: a tokenizer that does not match the checkpoint
    produces fluent text out of the wrong ids, and no test downstream of here can
    tell. `eos_token_id` is read off it rather than accepted per request, because a
    stop token is a property of the model this process loaded and letting a caller
    choose one is how a request never stops.
    """
    engine, plan = build_engine(weights_dir, **engine_kwargs)
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
        info=plan.as_dict(),
    )
    # Hung off the app so a test (and a debugger attached to a live server) can
    # reach the same objects the handlers are holding.
    app.state.plan = plan
    app.state.engine = engine
    app.state.serving = serving
    return app
