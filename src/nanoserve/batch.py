"""Padded batching: many ragged prompts, one rectangular forward. Week 7.

Phase 3 starts here. Everything up to Day 26 ran one sequence: one prompt in, one
token per step out, a cache with one block table. That is the correct shape for
learning the math and it is the wrong shape for a server, because a GPU running a
1B model on a single sequence is almost entirely idle. The matmuls are
memory-bound: the weights have to be read out of HBM every step regardless, so
running 8 sequences through the same read costs barely more time than running 1.
Batching is not a micro-optimization, it is the difference between using the card
and paying for it.

The obstacle is that prompts are ragged and tensors are rectangles. Sequences of
5, 2 and 3 tokens have to become one `[3, 5]` tensor, and the 5 empty slots get
filled with a pad token that means nothing. Three consequences, and this module
exists to make all three explicit rather than accidental:

  1. **The pads must not be attended to.** A pad key is a real row of K/V computed
     from a meaningless token id; if a real query scores against it, the softmax
     spends probability mass on nothing. Under *right* padding the causal mask
     already hides them (pads come after every real token, so no query may look
     that far ahead). Under *left* padding it does not: the pads sit at the low
     indices every query is allowed to see. So left padding needs a key mask.
  2. **The positions should skip the pads, though not for the obvious reason.**
     RoPE rotates by absolute position, so a left-padded row whose first real
     token sits at index 3 looks like it is being read at the wrong positions. It
     is not: a RoPE score depends only on the *difference* between the query and
     key positions, so a constant offset applied to a whole row cancels inside
     every dot product. Measured on this repo's test model the logits are
     bit-identical under a shift of 3 and 2e-6 off under a shift of 5000, which is
     float error in the cos/sin table. The positions are still built from the mask
     here, for reasons that live outside this one forward pass: a decode step
     addresses a block table by absolute position, so prefill and decode must
     agree on what position a token holds; keeping the offset out means a long
     padded batch does not push short rows toward the edge of the context window;
     and a *non-uniform* position error (a gap, not a shift) is not invisible at
     all. Free invariance is worth knowing about and not worth relying on.
  3. **The padding is pure waste.** Every pad slot costs a full row of attention
     and MLP arithmetic for a token that does not exist. `padding_waste` reports
     the fraction, because that number is the honest argument for everything that
     comes after static batching: ragged/varlen packing, and the continuous
     batching of Week 8 that stops a batch from being sized by its longest member.

Left padding is the default here, and the reason is narrow and practical: a decode
step reads each row's next-token logits from `logits[:, -1]`. With left padding
that column *is* every row's last real token, so the batch stays one tensor with
no per-row gather. Right padding is supported because it is what a prefill wants
when the whole logit grid is being compared token for token, and having both makes
the "is the mask load-bearing?" question testable rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PaddedBatch:
    """N ragged prompts stacked into one rectangle, plus what makes it honest.

    input_ids:      [batch, max_len] token ids with `pad_id` in the empty slots.
    attention_mask: [batch, max_len] bool, True where a token is real. This is a
                    *key* mask: it says which columns may be attended to, and it
                    is combined with (not a replacement for) the causal mask.
    position_ids:   [batch, max_len] RoPE positions, restarting at 0 at each row's
                    first real token so padding never shifts a sequence.
    lengths:        [batch] real token count per row.
    pad_side:       "left" or "right", recorded because `last_token_logits` and
                    any later decode step need to know where a row ends.
    pad_id:         the filler token id. Any in-vocab id works, since the mask
                    means the model's opinion of it is never read.

    Frozen because a batch is a description of one forward pass. Growing it by one
    decode step means building the next one, not mutating this.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    lengths: torch.Tensor
    pad_side: str
    pad_id: int

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def max_length(self) -> int:
        """Columns in the rectangle, i.e. the length of the longest prompt."""
        return self.input_ids.shape[1]

    @property
    def num_pad_tokens(self) -> int:
        """Slots in the rectangle that hold nothing."""
        return int(self.batch_size * self.max_length - self.lengths.sum().item())

    @property
    def padding_waste(self) -> float:
        """Fraction of the forward spent on tokens that do not exist.

        The cost of a rectangle: one 500-token prompt batched with seven 20-token
        prompts is 86% padding, and every one of those slots pays a full block of
        attention and MLP. This is why static batching alone is not the answer, and
        why a scheduler that groups by length (and later, one that batches at the
        iteration level) is worth building.
        """
        total = self.batch_size * self.max_length
        return self.num_pad_tokens / total if total else 0.0

    def last_index(self) -> torch.Tensor:
        """Column holding each row's final real token: [batch].

        Left padding puts every row's end at the last column, which is exactly why
        it is the default for decode. Right padding ends each row at `length - 1`.
        """
        if self.pad_side == "left":
            return torch.full(
                (self.batch_size,),
                self.max_length - 1,
                dtype=torch.long,
                device=self.input_ids.device,
            )
        return self.lengths - 1


def pad_prompts(
    prompts: list[list[int]],
    pad_id: int = 0,
    side: str = "left",
    device=None,
) -> PaddedBatch:
    """Stack ragged prompts into a `PaddedBatch`, mask and positions included.

    prompts: one list of token ids per sequence, each non-empty.
    pad_id:  filler id for the empty slots (never attended to, so its value is
             irrelevant to the result; pick something in-vocab so the embedding
             lookup does not fault).
    side:    "left" (default, see the module docstring) or "right".
    device:  where to build the tensors; defaults to CPU like the rest of the repo.

    The positions are derived from the mask rather than from the prompt lengths
    directly: `cumsum(mask) - 1` counts real tokens seen so far, which is the
    position of the current one, and clamping at 0 parks the pad slots at position
    0 where they are harmless. That one expression is correct for both sides and
    keeps the mask as the single source of truth about what is real. It also
    cannot introduce a *gap*, which is the one position error RoPE's relative
    scoring does not forgive (see the module docstring).
    """
    if not prompts:
        raise ValueError("pad_prompts needs at least one prompt")
    if any(len(p) == 0 for p in prompts):
        raise ValueError("cannot batch an empty prompt (no token to attend to)")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    lengths = [len(p) for p in prompts]
    max_len = max(lengths)
    rows, masks = [], []
    for prompt in prompts:
        pad = [pad_id] * (max_len - len(prompt))
        real = [True] * len(prompt)
        blank = [False] * (max_len - len(prompt))
        if side == "left":
            rows.append(pad + list(prompt))
            masks.append(blank + real)
        else:
            rows.append(list(prompt) + pad)
            masks.append(real + blank)

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    # Real tokens seen so far, minus one: the position of the token in this slot.
    # Pad slots land on -1 (left) or the row's last position (right); clamping the
    # first case to 0 keeps every index a legal RoPE position.
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return PaddedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        pad_side=side,
        pad_id=pad_id,
    )


def last_token_logits(logits: torch.Tensor, batch: PaddedBatch) -> torch.Tensor:
    """Pick each row's final real position out of a `[batch, seq, vocab]` grid.

    The next-token distribution of a padded batch is not `logits[:, -1]` in
    general: a right-padded row ends earlier, and reading the last column there
    returns the model's opinion of a pad token. `PaddedBatch.last_index` knows
    where each row ends, and this gathers there. Left padding makes the gather a
    no-op by construction, which is the point of preferring it.

    Returns [batch, vocab].
    """
    idx = batch.last_index().to(logits.device)
    return logits[torch.arange(logits.shape[0], device=logits.device), idx]
