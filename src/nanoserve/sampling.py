"""Token sampling: greedy, temperature, top-k, top-p. Week 3, Day 10.

Greedy decode (Week 2) always takes the single largest logit, so a tiny float
difference is invisible: only the *order* of the top logit matters. Sampling is
the less forgiving case the Day-9 log promised. Here the actual probabilities
matter, because the next token is *drawn* from them, so each transform has to
match the reference exactly or the distribution drifts.

The whole module is pure: a logits vector goes in, a token id comes out, and the
three knobs are independent functions composed in one fixed order. That order is
the same one HuggingFace's `generate` uses, and it is not arbitrary:

    temperature  ->  top-k  ->  top-p  ->  softmax  ->  draw

  1. **temperature** reshapes the distribution before anything is thrown away.
     It is a plain divide of the logits: `< 1` sharpens toward the argmax, `> 1`
     flattens toward uniform. It comes first because top-k and top-p decide what
     to keep based on probabilities, and temperature is what sets those.
  2. **top-k** keeps the k most likely tokens and masks the rest to `-inf`. A
     hard cap on the candidate set, regardless of how the mass is spread.
  3. **top-p** (nucleus) keeps the smallest set of most likely tokens whose
     probabilities sum to at least `p`, masking the long improbable tail. Unlike
     top-k it adapts: a confident step keeps few tokens, an uncertain one keeps
     many. It runs after top-k so the nucleus is measured over what survived.

Masking means setting a logit to `-inf` so its softmax probability is exactly 0
and `torch.multinomial` can never draw it. The filters always keep at least one
token, so the surviving distribution is never empty. The masks are pinned to
transformers' own `TopKLogitsWarper` / `TopPLogitsWarper` in the tests, the same
way Day-4 RoPE was pinned to HF's `apply_rotary_pos_emb`: match the reference,
do not reinvent the convention.

The filters operate on the last dim, so they work on a single `[vocab]` vector
or a `[batch, vocab]` batch unchanged. `sample` itself takes one `[vocab]` vector
(one sequence's next-token logits) and returns a Python int.

**Week 11, Day 40** adds the half a server needs. `SamplingParams` is what one
request asked for, and `BatchedSampler` turns a whole `[rows, vocab]` tensor into
one token per row when the rows disagree: one greedy, one at `top_p=0.9`, one at
`temperature=1.4` with its own seed, all columns of the same forward. The three
rules it is built on are that greedy is a branch and not a zero temperature, that
rows sharing a filter setting are filtered together, and that the draw itself is
per row because a seed belongs to a request rather than to a batch.

**Week 13, Day 47** changes what comes out and nothing about what is drawn.
`sample_batch_device` returns the tokens as a `[rows]` tensor on the device they
were computed on; `sample_batch` is that plus one `.tolist()`. The old path turned
every row into a Python int on the spot, which on a GPU is one synchronisation per
sampled row per step, and Day 46 measured the result: `sample` was 86% to 89% of
the engine's whole Python loop. Nothing here got cleverer. The tokens just stopped
coming home one at a time. See `nanoserve.output` for what that is worth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# The logit value that means "masked": softmax sends it to exactly 0 probability.
_FILTER = float("-inf")


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature. `< 1` sharpens, `> 1` flattens.

    A plain divide, matching HF's `TemperatureLogitsWarper`. `temperature == 0`
    is the greedy limit and is handled in `sample` (dividing by zero is not), so
    callers of this function should pass a positive temperature.
    """
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits, mask the rest to -inf. `k <= 0` is a no-op.

    `k` is clamped to the vocab size, so an oversized k keeps everything. The
    threshold is the k-th largest logit; anything strictly below it is dropped,
    exactly as HF's `TopKLogitsWarper` does it.
    """
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    # The smallest of the top-k logits along the last dim, kept as a column so it
    # broadcasts back against every position.
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, _FILTER)


def top_p_filter(
    logits: torch.Tensor, p: float, min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """Keep the smallest nucleus of most likely tokens reaching mass `p`.

    `p >= 1.0` keeps everything. Otherwise this mirrors HF's `TopPLogitsWarper`:
    sort ascending, take the cumulative softmax mass, and remove every token whose
    cumulative mass (counted from the least likely upward) sits at or below
    `1 - p`. At least `min_tokens_to_keep` of the most likely tokens always
    survive, so the nucleus is never empty even when `p` is tiny.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Ascending order, so "remove the unlikely tail" is the low-cumulative end.
    sorted_remove = cumulative_probs <= (1 - p)
    # The most likely tokens are at the high end; never drop the last few.
    sorted_remove[..., -min_tokens_to_keep:] = False
    # Scatter the per-rank mask back to the original token positions.
    remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, _FILTER)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Draw the next token id from a single `[vocab]` logits vector.

    The knobs compose in HF's order: temperature, then top-k, then top-p, then a
    softmax and one `torch.multinomial` draw. `temperature == 0` is the greedy
    limit and short-circuits to the argmax (the same token Week 2's `greedy_token`
    picks), so greedy is just the zero-temperature corner of this one function and
    never touches the RNG. The defaults (`temperature=1, top_k=0, top_p=1`) are
    plain softmax sampling over the full vocabulary.

    `generator` threads an explicit `torch.Generator` through the draw so a run is
    reproducible from a seed; omit it to use the global RNG.
    """
    if temperature == 0:
        return int(logits.argmax(dim=-1))
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = logits.softmax(dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


# --- one tensor, many callers: Week 11, Day 40 ---------------------------------


@dataclass(frozen=True)
class SamplingParams:
    """What one request asked the sampler for. Pure data, no tensors, no RNG.

    temperature: `0.0` means greedy and is the default, so every `Request` built
                 before today keeps the behaviour it had. Anything positive
                 sharpens (`< 1`) or flattens (`> 1`) before the filters run.
    top_k:       keep the k most likely tokens; `0` is off.
    top_p:       keep the smallest nucleus reaching mass p; `1.0` is off.
    seed:        this request's own RNG seed, or None to share the engine's.

    Frozen for the same reason `PaddedBatch` is: a request may not change what it
    asked for halfway through its own generation, or two halves of one answer come
    from two different distributions and nothing downstream can tell.

    It carries no `torch.Generator`, deliberately. A generator is mutable state
    that advances with every draw, and hanging it off the request would put RNG
    state inside the scheduler, which is the one module in this engine that owns
    no tensors. The state lives in `BatchedSampler` instead, keyed by request id,
    which also gives it somewhere to be released from when the request finishes.
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 means off), got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    @property
    def is_greedy(self) -> bool:
        """Greedy is a branch, not a limit. See `BatchedSampler.sample_batch`."""
        return self.temperature == 0.0

    @property
    def filter_key(self) -> tuple[int, float]:
        """What two rows must agree on to be filtered in one call.

        Temperature does not appear here: it is a per-row divide and vectorises
        against a `[rows, 1]` column whatever the values are. `top_k` and `top_p`
        do, because their thresholds are computed *inside* the filter from a
        scalar, so rows with different k or p have to be different calls.
        """
        return (self.top_k, self.top_p)


# The default every request gets, and the behaviour every day before this one had.
GREEDY = SamplingParams()


class BatchedSampler:
    """Turn one `[rows, vocab]` logits tensor into one token per row. Day 40.

    Under continuous batching the rows of a single forward belong to different
    callers, and after today they no longer agree on how to sample. One row wants
    the argmax, the next wants `top_p=0.9` at `temperature=1.2` with its own seed,
    and they are columns of the same tensor. Three rules make that work.

    **Greedy is a branch, not `temperature=0`.** Dividing logits by zero gives
    `inf`, whose softmax is `nan`, and a draw over `nan` is an exception or a lie.
    Greedy rows are pulled out and answered with one batched `argmax`, which is
    also the *only* way a greedy row can be bit-identical to what it would get
    alone. And they consume no randomness, so adding a greedy neighbour to a batch
    cannot move anybody else's draw.

    **Sampled rows are grouped by `filter_key`, not run one at a time.** Every row
    in a group shares `top_k` and `top_p`, so the two filters run once for the
    whole group, and temperature is a per-row `[m, 1]` divide inside it. A server
    sees a handful of distinct filter settings across a batch, not one per
    request, so this is a few kernel launches per step rather than one per row.
    vLLM vectorises the whole thing with per-row k and p tensors; the grouping
    here is the same idea with less machinery and the same answers.

    **The draw itself cannot be batched, because a seed is per request.** One
    `torch.multinomial` over `[m, vocab]` advances one generator once for the
    whole block, which makes every row's token a function of who else was in the
    block. That is exactly the coupling a seed exists to remove, so a seeded row
    draws from its own generator, alone, one row at a time. It costs one
    `multinomial` over a `[vocab]` vector per sampled row per step, against a
    forward pass measured in tens of milliseconds.

    The generators are kept here, in a dict keyed by request id, and `release`
    drops one when its request finishes. Both halves matter: a generator rebuilt
    each step would redraw from the same state forever, and a generator never
    dropped is a per-request leak in a process meant to run for weeks.
    """

    def __init__(self, seed: int | None = None):
        """`seed` seeds the *shared* generator, used by requests that gave none.

        A request without a seed is not promised reproducibility, and it does not
        get it: it draws from this one generator, so its tokens depend on how many
        other unseeded rows drew before it in the same step. Seeding the engine
        makes a whole run repeatable when the batch composition is also repeatable,
        which is what the benchmarks want and what a live server never has.
        """
        self._shared = torch.Generator()
        if seed is not None:
            self._shared.manual_seed(seed)
        self._generators: dict[str, torch.Generator] = {}

    @property
    def num_generators(self) -> int:
        """Live per-request generators. Should track the sampled requests in flight."""
        return len(self._generators)

    def release(self, request_id: str) -> None:
        """Drop a finished request's generator. A no-op for one that never had any."""
        self._generators.pop(request_id, None)

    def _generator_for(self, request_id: str, params: SamplingParams) -> torch.Generator:
        """This request's generator, created on its first draw and kept after it.

        Created here rather than at admission because a greedy request never needs
        one, and because this is the only place that knows a draw is about to
        happen. Kept because the state has to advance: seeding per step would draw
        the same token from the same distribution every step.
        """
        if params.seed is None:
            return self._shared
        generator = self._generators.get(request_id)
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(params.seed)
            self._generators[request_id] = generator
        return generator


    def sample_batch(
        self,
        logits: torch.Tensor,
        rows: Sequence[tuple[str, SamplingParams]],
    ) -> list[int]:
        """One token per row of `[rows, vocab]` logits, as Python ints.

        `rows` pairs each row with the request id and params that own it, as one
        sequence rather than two parallel lists: a length mismatch between ids and
        params would hand one caller another caller's distribution, and that is a
        bug no shape check catches.

        Since Day 47 this is `sample_batch_device` plus exactly one `.tolist()`,
        rather than a second implementation. That matters for two reasons. The
        obvious one is that the two paths cannot drift, so a request gets the same
        token whichever entry point the engine calls. The less obvious one is the
        *count*: one journey home per step rather than one per sampled row, which
        is the whole of Day 47 and is measured in `nanoserve.output`.

        Callers who do not need Python ints this instant should call
        `sample_batch_device` and keep the tensor. On a GPU this `.tolist()` is a
        synchronisation, and a synchronisation is where the host stops running
        ahead of the device.
        """
        return self.sample_batch_device(logits, rows).tolist()

    def sample_batch_device(
        self,
        logits: torch.Tensor,
        rows: Sequence[tuple[str, SamplingParams]],
    ) -> torch.Tensor:
        """The same draw as `sample_batch`, left on the device as `[rows]` int64.

        Day 46 profiled a decode step and found `sample` holding 86% to 89% of the
        Python loop at every batch size. Sampling is not expensive; *returning* it
        was. `sample_batch` produced a `list[int]`, and every one of those ints was
        a separate `int(tensor)`: a separate journey back across the bus, each one
        a point where the host stops and waits for every kernel the step has
        queued. The arithmetic below is unchanged. What changed is that it ends in
        a tensor.

        Three things make that possible without altering a single drawn token:

        **The all-greedy batch is one kernel.** Every `Request` defaults to
        `GREEDY`, so this is the case the engine actually runs, and it used to
        gather the greedy rows into a `[m, vocab]` copy before taking the argmax.
        An argmax is per row and independent of its neighbours, so the gather was
        never buying anything: `logits.argmax(dim=-1)` is the same answer with no
        copy, no dict, no Python loop and no readback.

        **A mixed batch writes into one buffer instead of a list.** The greedy rows
        and each filter group land in the right slots via `index_copy_`, which is a
        device-side scatter. The index tensors are built from Python lists and
        copied host to device, which is a *launch*, not a synchronisation: the host
        hands the copy to the driver and keeps going.

        **The per-row draw survives, because a seed is per request.** Day 40's rule
        has not changed: one `multinomial` over a `[vocab]` vector per sampled row,
        from that request's own generator, in row order, so the RNG is consumed in
        exactly the sequence `sample_batch` consumed it in. The results are
        concatenated and scattered rather than converted one at a time. That is the
        difference between N readbacks and none.

        Returns a `[rows]` int64 tensor on the logits device, empty when `rows` is.
        """
        if logits.shape[0] != len(rows):
            raise ValueError(
                f"the logits have {logits.shape[0]} rows but {len(rows)} requests "
                "were given: a row is a request, so they must line up"
            )
        device = logits.device
        if not rows:
            # A step that only prefilled, or one whose whole batch was released.
            return torch.empty(0, dtype=torch.long, device=device)

        greedy = [i for i, (_, p) in enumerate(rows) if p.is_greedy]
        if len(greedy) == len(rows):
            # The engine's default path: one argmax over the tensor as it stands.
            return logits.argmax(dim=-1)

        tokens = torch.empty(len(rows), dtype=torch.long, device=device)
        if greedy:
            # Argmax over every row and keep the greedy ones. Cheaper than gathering
            # `[m, vocab]` floats first, and identical: an argmax does not look
            # sideways.
            index = torch.tensor(greedy, dtype=torch.long, device=device)
            tokens.index_copy_(0, index, logits.argmax(dim=-1).index_select(0, index))

        groups: dict[tuple[int, float], list[int]] = {}
        for i, (_, params) in enumerate(rows):
            if not params.is_greedy:
                groups.setdefault(params.filter_key, []).append(i)

        for (top_k, top_p), members in groups.items():
            block = logits[members]
            temperatures = torch.tensor(
                [rows[i][1].temperature for i in members],
                dtype=block.dtype,
                device=block.device,
            ).unsqueeze(1)
            block = block / temperatures
            block = top_k_filter(block, top_k)
            block = top_p_filter(block, top_p)
            probs = block.softmax(dim=-1)
            drawn = torch.cat(
                [
                    torch.multinomial(
                        probs[offset],
                        num_samples=1,
                        generator=self._generator_for(*rows[i]),
                    )
                    for offset, i in enumerate(members)
                ]
            )
            tokens.index_copy_(0, torch.tensor(members, dtype=torch.long, device=device), drawn)
        return tokens
