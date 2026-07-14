"""The real Triton paged-attention kernel. Week 6, Day 23.

Day 22 wrote the fused paged attention as a grid of `tlsim` programs: one program
owns one query position, walks the history a tile of keys at a time, reads each
tile's K/V through the block table with a masked load, and folds it into a
flash-attention online softmax (running max, running denominator, running weighted-V
sum). That loop is pinned to the Day-18 oracle. This module is that same loop
written for hardware: a `@triton.jit` kernel whose tiles stream from HBM into SRAM
and whose programs run in parallel instead of in a Python `for`.

Two things change on the way down, and both are visible below.

*The grid grows an axis.* On the CPU one program looped its query heads, because a
loop is a loop. On a GPU the heads are free parallelism, so program `(i, h)` owns
query position `i` of query head `h`: one query, one head, one independent softmax,
nothing shared. The accumulators shrink from `[n_q]` and `[n_q, d]` to a scalar and
one `[BLOCK_D]` row, which is what lets them live in registers.

*Shapes must be powers of two.* `tl.arange` takes only power-of-two lengths, so the
head dimension is padded up to `BLOCK_D` and the key tile up to `BLOCK_N`, with a
mask discarding the lanes past the real extent. That is the same mask discipline the
CPU model used for the ragged tail of the history, applied now to the channel axis
too, and it is why a head_dim of 48 or a tile of 13 is legal here at all.

Everything else is a transcription. The layering is deliberate: every integer the
kernel turns into a pointer (the slot stride, `past`, the grid, the padded tile
sizes) is computed by ordinary Python in `check_paged_inputs` / `launch_grid` /
`next_power_of_2`, which are tested on any box, GPU or not. Addressing bugs live in
that arithmetic, and arithmetic you can single step is arithmetic you can fix. The
jitted body is held to `paged_attention_reference` by tests gated on a real device,
the way the end-to-end model tests are gated on ./weights.

Triton ships with the Linux GPU torch wheel and is simply absent from a CPU-only
one, so the import is guarded and `paged_attention` degrades to the Day-22 CPU model
rather than exploding. The fallback is the model, not a fast path: it is correct and
slow, and it exists so the engine runs everywhere while the kernel runs where it
can. This is the shape vLLM and SGLang ship their paged-attention kernels in; the
version here is the one I can read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paged_attention import paged_attention_kernel

try:  # Triton rides along with the Linux GPU torch wheel; a CPU wheel has no module.
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised by whichever box you are on
    triton = None
    tl = None


def has_triton() -> bool:
    """True when the `triton` package imported, i.e. a jitted kernel can be compiled.

    Says nothing about whether a GPU exists. Compiling needs the package; launching
    needs a device. `select_backend` wants both, and asks for both separately.
    """
    return triton is not None


def next_power_of_2(n: int) -> int:
    """The smallest power of two that is at least `n`, the only length `tl.arange` takes.

    Triton's block shapes are compile-time powers of two, so a head_dim of 48 or a
    key tile of 13 cannot be used as-is. The kernel takes the rounded-up extent and
    masks the surplus lanes off, which is exactly what the ragged tail of a history
    already required. Rounding up is the only safe direction: rounding down would
    silently drop channels (or keys) from the softmax.
    """
    if n < 1:
        raise ValueError(f"block extents are positive; got {n}")
    return 1 << (n - 1).bit_length()


def launch_grid(seq_q: int, n_q: int) -> tuple[int, int]:
    """The launch grid: one program per (query position, query head).

    Program `(i, h)` computes a whole row of the attention output on its own. It
    reads shared K/V and writes only its own `[head_dim]` slice, so no two programs
    touch the same output element and the grid can be the full product with no
    reduction between programs. That independence is the reason a paged attention
    parallelizes at all, and it is why the online softmax has to be per-program.
    """
    return (seq_q, n_q)


@dataclass(frozen=True)
class PagedGeometry:
    """Every integer the kernel turns into a pointer, derived once on the host.

    n_q, n_kv: query heads and (compact) KV heads; `n_q == n_kv * n_rep` under GQA.
    seq_q:     new queries this step (the prompt length on prefill, 1 on a decode).
    seq_total: the whole history including this step's tokens.
    past:      `seq_total - seq_q`, tokens already cached. Query i's causal extent is
               `past + i + 1`, so a wrong `past` mis-sizes every program's loop.
    channels:  `n_kv * head_dim`, the stride from one pool slot to the next in the
               flat pool. This is the number the block table's slot gets multiplied
               by; get it wrong and the kernel reads a neighbouring token's K.
    """

    n_q: int
    seq_q: int
    head_dim: int
    n_kv: int
    num_slots: int
    channels: int
    seq_total: int
    past: int


def check_paged_inputs(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_rep: int,
) -> PagedGeometry:
    """Validate the paged-attention inputs and derive the kernel's geometry.

    Same arguments as `paged_attention_reference`. Returns a `PagedGeometry`; raises
    `ValueError` on anything the kernel would otherwise turn into a wrong pointer.

    The checks are the ones a kernel cannot make for itself. A jitted body has no
    shapes, only pointers and integers: it will happily let query head 7 index past
    the end of a slot if `n_rep` is wrong, and the result is a finite, plausible
    number computed from the next token's key. Every invariant that keeps the
    pointer arithmetic inside the buffer is stated once, here, on the host.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError(
            "paged attention handles one sequence; q must be [1, n_q, seq_q, d] "
            "(per-sequence batching arrives in Phase 3)"
        )
    if k_pool.ndim != 3 or k_pool.shape != v_pool.shape:
        raise ValueError(
            "k_pool and v_pool must have the same shape [num_slots, n_kv, d]: a token's "
            f"key and value share a slot; got {tuple(k_pool.shape)} and {tuple(v_pool.shape)}"
        )
    if slot_mapping.ndim != 1:
        raise ValueError(f"slot_mapping must be 1-D [seq_total]; got {tuple(slot_mapping.shape)}")

    _, n_q, seq_q, head_dim = q.shape
    num_slots, n_kv, pool_dim = k_pool.shape
    if head_dim != pool_dim:
        raise ValueError(f"head_dim disagrees with the pool: q has {head_dim}, pool has {pool_dim}")
    if n_rep < 1 or n_kv * n_rep != n_q:
        raise ValueError(
            f"n_rep must tile the query heads: n_q ({n_q}) != n_kv ({n_kv}) * n_rep ({n_rep}); "
            "the kernel maps query head h to KV head h // n_rep and cannot check the range"
        )
    seq_total = int(slot_mapping.shape[0])
    if seq_total < seq_q:
        raise ValueError(
            f"the history is shorter than the queries: seq_total ({seq_total}) < seq_q ({seq_q}). "
            "slot_mapping covers the whole sequence, including this step's new tokens"
        )
    return PagedGeometry(
        n_q=n_q,
        seq_q=seq_q,
        head_dim=head_dim,
        n_kv=n_kv,
        num_slots=num_slots,
        channels=n_kv * head_dim,
        seq_total=seq_total,
        past=seq_total - seq_q,
    )


def select_backend(device: torch.device) -> str:
    """Which paged attention `device` can actually run: "triton" or "tlsim".

    A jitted kernel cannot address host memory, so a CPU tensor is always the CPU
    model no matter what is installed; a CUDA tensor is the kernel exactly when the
    package is importable. The dispatcher never names a backend it cannot launch,
    which is the whole reason this is a function and not an assumption.
    """
    return "triton" if device.type == "cuda" and has_triton() else "tlsim"


if triton is not None:  # pragma: no cover - compiled and run only on a GPU box

    @triton.jit
    def _paged_attention_fwd(
        q_ptr,  # [n_q, seq_q, head_dim] contiguous, this step's rotated queries
        k_ptr,  # [num_slots * channels] the layer's flat physical K pool
        v_ptr,  # [num_slots * channels] the layer's flat physical V pool
        slot_ptr,  # [seq_total] logical position -> physical slot (the block table)
        out_ptr,  # [n_q, seq_q, head_dim] contiguous
        scale,  # softmax scale, usually head_dim ** -0.5
        past,  # seq_total - seq_q: how many tokens precede this step's queries
        seq_q,  # new queries this step, the stride between heads in q/out
        stride_slot,  # channels = n_kv * head_dim: elements from one slot to the next
        HEAD_DIM: tl.constexpr,  # the true head dimension
        N_REP: tl.constexpr,  # GQA repeat: query heads per KV head
        BLOCK_D: tl.constexpr,  # HEAD_DIM padded up to a power of two
        BLOCK_N: tl.constexpr,  # keys folded per step, the SRAM tile
    ):
        """One program: query position `i` of query head `h`, streamed over the history.

        The Day-22 loop, unchanged in structure. The program loads its single query
        row into registers, then walks the causally visible history `BLOCK_N` keys at
        a time. Each tile is read through the block table (`slot * stride_slot` is
        where that token's K/V live in the pool, wherever the allocator put them) and
        folded into an online softmax, so the `[kv_len, head_dim]` history is never
        materialized. Peak state is one tile plus a `[BLOCK_D]` accumulator.
        """
        i = tl.program_id(0)  # which new query
        h = tl.program_id(1)  # which query head
        kv_head = h // N_REP  # GQA: this query head's compact KV head
        kv_len = past + i + 1  # causal: query i may see keys 0..past+i, no further

        # The channel ramp, padded to a power of two. Lanes past HEAD_DIM are masked
        # everywhere: they load 0.0, contribute 0 to the dot product, and are never
        # stored. This is what makes a head_dim of 48 legal.
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        q_off = h * seq_q * HEAD_DIM + i * HEAD_DIM + offs_d
        q = tl.load(q_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

        # Online-softmax state, held as 1-element tensors so the loop-carried types
        # never change between iterations (Triton requires that; a Python float that
        # becomes a tensor on the first pass will not compile). These are registers.
        m = tl.full([1], float("-inf"), dtype=tl.float32)  # running max
        denom = tl.zeros([1], dtype=tl.float32)  # running softmax denominator
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # running weighted-V sum

        for b in range(0, tl.cdiv(kv_len, BLOCK_N)):
            offs_n = b * BLOCK_N + tl.arange(0, BLOCK_N)  # this tile's key positions
            valid = offs_n < kv_len  # guards the ragged tail of the history
            # The paged read: each key's physical slot, then that slot's K/V channels
            # for *this* KV head. Masked-off lanes read slot 0 and never fault.
            slots = tl.load(slot_ptr + offs_n, mask=valid, other=0)
            kv_off = slots[:, None] * stride_slot + kv_head * HEAD_DIM + offs_d[None, :]
            tile_mask = valid[:, None] & mask_d[None, :]
            k = tl.load(k_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)

            s = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_N] scores
            # A masked load returned zeros, and a query scores a real, finite weight
            # against a zero key. Force the phantom lanes to -inf so exp gives exactly
            # zero: out of bounds and out of the softmax are two different jobs.
            s = tl.where(valid, s, float("-inf"))

            # Fold the tile in: renormalize the state to the new running max, then add.
            # Drop the `alpha` rescale and every tile after the first is normalized
            # against the wrong max.
            m_new = tl.maximum(m, tl.max(s, axis=0))
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)  # [BLOCK_N], unnormalized tile weights
            denom = denom * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m = m_new

        out = acc / denom  # the deferred normalization: one divide, at the very end
        tl.store(out_ptr + q_off, out.to(out_ptr.dtype.element_ty), mask=mask_d)


def paged_attention_triton(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = 64,
) -> torch.Tensor:
    """Launch the Triton paged-attention kernel. Same contract as the reference.

    q, k_pool, v_pool, slot_mapping, n_rep, scale: exactly as in
    `paged_attention_reference`, but every tensor must live on the same CUDA device.
    block:   keys folded per step, the SRAM tile. Rounded up to a power of two for
             `tl.arange`; a pure performance knob, since the mask makes any extent
             return the same attention.

    Returns [1, n_q, seq_q, d], matching the reference to about 1e-4. Streaming the
    softmax reassociates the exponent sums, so it is close, not bit-identical: the
    accuracy trade every flash-attention kernel makes, and here in bf16/fp16 the
    accumulators still run in fp32.

    Raises `ValueError` on host tensors and `RuntimeError` when Triton is missing,
    rather than letting a launch crash somewhere unreadable. `paged_attention` picks
    this path only when both hold; call it directly when you mean to demand the GPU.
    """
    geom = check_paged_inputs(q, k_pool, v_pool, slot_mapping, n_rep)
    if q.device.type != "cuda":
        raise ValueError(
            f"the Triton kernel reads device memory; q is on {q.device}. "
            "Use `paged_attention` for a dispatch that falls back to the CPU model"
        )
    if not has_triton():
        raise RuntimeError("the triton package is not installed; `paged_attention` falls back")
    if k_pool.device != q.device or slot_mapping.device != q.device:
        raise ValueError("q, the pools, and slot_mapping must live on the same device")

    if scale is None:
        scale = geom.head_dim**-0.5

    # Flatten to the 1-D buffers the kernel addresses: a token's K/V is `channels`
    # contiguous elements at `slot * channels`. `.contiguous()` is what makes that
    # arithmetic true, so it is not optional even when it is usually a no-op.
    q_rows = q[0].contiguous()  # [n_q, seq_q, head_dim]
    k_flat = k_pool.contiguous().reshape(-1)
    v_flat = v_pool.contiguous().reshape(-1)
    slots = slot_mapping.to(torch.int64).contiguous()
    out = torch.empty_like(q_rows)

    _paged_attention_fwd[launch_grid(geom.seq_q, geom.n_q)](
        q_rows,
        k_flat,
        v_flat,
        slots,
        out,
        scale,
        geom.past,
        geom.seq_q,
        geom.channels,
        HEAD_DIM=geom.head_dim,
        N_REP=n_rep,
        BLOCK_D=next_power_of_2(geom.head_dim),
        BLOCK_N=next_power_of_2(block),
        num_warps=4,
    )
    return out[None]  # [1, n_q, seq_q, d], the shape o_proj expects


def paged_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_rep: int,
    scale: float | None = None,
    block: int = 64,
) -> torch.Tensor:
    """Paged attention, on whichever backend this device can actually run.

    The public entry point. On a CUDA tensor with Triton installed it launches the
    kernel above; anywhere else it runs the Day-22 `paged_attention_kernel`, the
    tlsim model of the same loop. Both read K/V through the block table and never
    assemble the contiguous history, and both agree with `paged_attention_reference`
    to a few ulps, so the backend is a speed decision and never a numerics one.

    The fallback is honest about what it is. It is the loop, on the CPU, running a
    masked gather per tile; it is correct and it is slow, and it exists so the engine
    runs on a laptop while the kernel runs on the card.
    """
    if select_backend(q.device) == "triton":
        return paged_attention_triton(q, k_pool, v_pool, slot_mapping, n_rep, scale, block)
    return paged_attention_kernel(q, k_pool, v_pool, slot_mapping, n_rep, scale, block)
