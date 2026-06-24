"""The transformer: stack of blocks, final norm, LM head, forward to logits. Week 2.

Day 8, the start of Week 2. Week 1 ended with one decoder block verified end to
end against HuggingFace. The whole model is that block stacked `num_hidden_layers`
times between an embedding table and a final norm, so today is mostly a `for`
loop and a name-it-correctly exercise, not new math:

    logits = lm_head( norm( block_15( ... block_0( embed(input_ids) ) ... ) ) )

Three things are worth saying out loud because each is a place the forward pass
can run and still be wrong:

  1. One RoPE table for the whole stack. The cos/sin tables depend only on the
     positions, not on the layer, so they are built once from `position_ids` and
     handed to every block. Rebuilding them per layer would be wasted work; using
     a different convention than the per-block test used would be a bug.
  2. The final norm is real and easy to forget. There is a `norm.weight` after
     the last block and before the LM head. Drop it and the logits are off by a
     whole RMSNorm, which looks like "close but never quite matches HF".
  3. The LM head is tied to the embedding. `lm_head.weight` is the same tensor as
     `embed_tokens.weight` (the loader wired this alias on Day 3), so the output
     projection is literally the input embedding matrix reused. Nothing to do
     here except trust the loader and not accidentally make a second copy.

There is no KV cache yet. `forward` recomputes the whole prefix every call, which
is O(n^2) and exactly what a cache will later fix; for Week 2 the only goal is
numeric agreement with the reference, so the slow, obvious path is the right one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RotaryEmbedding, rms_norm, transformer_block
from .loader import EMBED, LM_HEAD, Weights


class LlamaModel:
    """Llama-3.2-1B forward pass built from nanoserve.layers and a loaded Weights.

    Construct with the parsed `ModelConfig` and the `Weights` bag from the loader;
    the model owns nothing but its config, its weights, and the position-driven
    RoPE table builder. It never trains, so there are no `nn.Module`s and no
    parameters to register; `forward` just threads tensors through plain functions.
    """

    def __init__(self, config: ModelConfig, weights: Weights):
        self.config = config
        self.weights = weights
        # One rotary table builder for the whole stack; positions, not layers,
        # decide the angles, so this is shared across all blocks.
        self.rotary = RotaryEmbedding(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full prefill forward pass and return logits over the vocab.

        input_ids:    [batch, seq] token ids (long).
        position_ids: [batch, seq] positions; defaults to 0..seq-1, which is the
                      right thing for a contiguous prompt. Passed through to RoPE
                      so the scattered positions of a later decode step will work
                      unchanged.

        Returns logits [batch, seq, vocab_size]. The next-token distribution is
        `logits[:, -1]`; the full sequence of logits is returned (not just the
        last position) so this can be compared to HF `model(input_ids).logits`
        token for token. Matches HF `LlamaForCausalLM` logits to ~1e-4.
        """
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]

        # Token embedding: a row lookup into the [vocab, hidden] table.
        x = F.embedding(input_ids, self.weights[EMBED])

        # Build the RoPE tables once for these positions, then reuse for every block.
        cos, sin = self.rotary.cos_sin(position_ids)

        # The stack. Each block is the Day-7 pre-norm decoder layer, verified on
        # its own; here they just compose, block i's output feeding block i+1.
        for i in range(self.config.num_hidden_layers):
            x = transformer_block(x, self.weights.layer(i), cos, sin, self.config)

        # Final RMSNorm before the head (the one that is easy to forget), then the
        # tied output projection: logits = norm(x) @ embed_tokens.T.
        x = rms_norm(x, self.weights["norm.weight"], self.config.rms_norm_eps)
        return F.linear(x, self.weights[LM_HEAD])

    @torch.no_grad()
    def greedy_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Argmax next token for each sequence in the batch: [batch].

        The one-step greedy choice (no sampling, no cache). Repeating this with
        the token appended is greedy decode; Week 2 keeps it this naive
        (recompute the prefix each step) and chases HF token for token before any
        cache exists to make it fast.
        """
        return self.forward(input_ids)[:, -1].argmax(dim=-1)

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy decode: append the argmax, recompute, repeat. Returns prompt+gen.

        input_ids:      [1, seq] the prompt. One sequence only; real batching of
                        ragged, independently-stopping sequences is Phase 3, so
                        this guards against a batch dim greater than 1 rather than
                        pretending to handle it.
        max_new_tokens: how many tokens to generate at most.
        eos_id:         if given, stop after emitting this token (it is kept in the
                        output, matching what HF `generate` returns).

        Returns [1, seq + generated], the prompt with the continuation appended.

        This is the whole of greedy decode, and it is deliberately the slow path:
        there is no KV cache yet, so every step re-runs `forward` on the entire
        growing prefix (O(n^2) over the run). Week 3 adds the contiguous cache
        that turns each step into one new-token forward; Week 2 only needs the
        tokens to come out identical to the reference, so the obvious loop wins.

        `greedy_token` already does one step's argmax via a full forward, so this
        is just that step, appended, in a loop. `position_ids` defaults inside
        `forward` to `0..len-1`, which stays correct as the sequence grows because
        the whole prefix is passed every time.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                "greedy_generate decodes a single sequence; batch must be 1 "
                "(ragged multi-sequence batching arrives in Phase 3)"
            )
        ids = input_ids
        for _ in range(max_new_tokens):
            nxt = self.greedy_token(ids)  # [1], the argmax over the last position
            ids = torch.cat([ids, nxt[:, None]], dim=1)  # append, never in place
            if eos_id is not None and nxt.item() == eos_id:
                break
        return ids
