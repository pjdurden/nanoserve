"""Ids to text, one token at a time, without lying. Week 11, Day 39.

Every day until now decoded once. `Engine.generate` returns a finished list of
ids and `server.py` calls `tokenizer.decode` on all of it, which is correct and
which streaming cannot do, because streaming's whole product is text before the
end exists. So the decode has to become incremental, and the obvious incremental
decode is wrong.

**A token is a run of bytes, not a run of characters.** Llama-3's tokenizer is
byte-level BPE: the vocabulary is built over the 256 byte values, and a merge
never checks whether the run it just created ends on a UTF-8 boundary. So a
4-byte emoji is routinely two tokens, and `decode([first])` is asked to render
half a character. It answers with U+FFFD, the replacement character, because
that is what a decoder is supposed to do with a truncated sequence. Do that per
token and the stream prints two black diamonds where the model wrote one emoji,
while the non-streaming endpoint prints the emoji, from the same ids. The bug is
entirely in the incremental path, so no amount of unary testing finds it.

The fix is to never decode a token alone. Keep the ids, decode a small *window*
of them twice (once without the newest token, once with) and emit the difference:

    pre   = decode(ids[prefix_offset:read_offset])
    whole = decode(ids[prefix_offset:])
    delta = whole[len(pre):]

The window costs two decodes per token instead of one, over a handful of ids,
against a forward pass that is tens of milliseconds. It is free. What it buys is
that the decoder always sees the bytes on both sides of every boundary it is
asked about, which is the only way it can know whether it is looking at a
character or at the front of one.

**And then the hold-back.** A window that ends mid-character still decodes to a
trailing U+FFFD, so the diff would emit a replacement character that the finished
text does not contain. There is no repair for that: the bytes are already on the
socket, and a stream cannot recall a character it printed. So a delta ending in
U+FFFD is not emitted at all, the offsets do not move, and the next token
re-decodes the same window one byte longer. The latency this costs is bounded by
3 tokens, because UTF-8 is at most 4 bytes long and a token carries at least one.

The invariant, which is the only thing a caller can build on: **what has been
emitted, concatenated, is always a prefix of the whole-list decode, and equals it
once `flush` has run.** Late, never wrong. `flush` is the other half of that: a
generation cut off by `max_tokens` in the middle of a character has held-back
bytes that really happened, and hiding them would make the streamed text differ
from the unary answer for the same ids. So the end of the stream emits whatever
`decode` makes of them, replacement character and all, which is exactly what
`/v1/completions` would have printed.

vLLM's `detokenize_incrementally` is this, with the same two offsets and the same
U+FFFD test; SGLang keeps the same shape. The one known false hold is a model
that genuinely emits U+FFFD as content, which waits for the next token and is
released by `flush` at the latest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

#: The character a UTF-8 decoder substitutes for bytes it cannot finish reading.
#: Its presence at the end of a window means "ask me again with one more token".
REPLACEMENT = "�"

#: Prompt tokens kept in front of the first generated one. Byte-level BPE only
#: needs the bytes of the character being finished, but a tokenizer that decides
#: a leading space from context needs a little more, and the cost of extra window
#: is one slightly longer decode of a string nobody reads.
DEFAULT_CONTEXT_TOKENS = 4


class Tokenizer(Protocol):
    """All this needs, which is why the tests can use 256 bytes as a vocabulary."""

    def decode(self, token_ids: Sequence[int]) -> str: ...


class IncrementalDetokenizer:
    """One request's growing id list, rendered as a growing string.

    tokenizer:        anything with `decode`.
    prompt_token_ids: context, never emitted. A completions stream shows the
                      generation, but the decoder wants the bytes immediately
                      before it, so the prompt's tail sits in the window and its
                      text is subtracted out by the diff.
    context_tokens:   how much of that tail to keep at the start.

    Stateful and single-request: one of these per stream, alive as long as the
    stream is. It holds the ids, so it is O(tokens) in memory and O(1) work per
    token, which is the right shape for something called once per decode step.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        prompt_token_ids: Iterable[int] = (),
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    ) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = list(prompt_token_ids)
        # Where the decode window starts, and how far into the ids the emitted
        # text has already reached. They advance together, one chunk behind each
        # other, so the window is always "the last thing I showed, plus what is
        # new" and never grows without bound.
        self._read_offset = len(self.token_ids)
        self._prefix_offset = max(self._read_offset - context_tokens, 0)
        self._text_parts: list[str] = []

    @property
    def text(self) -> str:
        """Everything emitted so far. The concatenation of every delta returned."""
        return "".join(self._text_parts)

    @property
    def num_held(self) -> int:
        """Tokens decoded into no text yet, because they end mid-character."""
        return len(self.token_ids) - self._read_offset

    def append(self, token_id: int) -> str:
        """Add one token and return the text that became readable because of it.

        Usually one token's worth. Empty while a multi-byte character is still
        arriving, then the whole character at once on the token that completes it.
        """
        self.token_ids.append(token_id)
        pre = self.tokenizer.decode(self.token_ids[self._prefix_offset : self._read_offset])
        whole = self.tokenizer.decode(self.token_ids[self._prefix_offset :])
        if len(whole) <= len(pre) or whole.endswith(REPLACEMENT):
            # Either nothing new rendered, or what rendered is the front half of a
            # character. Leave the offsets alone: the next token re-decodes this
            # same window one token longer, which is how the held bytes come back.
            return ""
        delta = whole[len(pre) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(self.token_ids)
        self._text_parts.append(delta)
        return delta

    def extend(self, token_ids: Iterable[int]) -> str:
        """Append several and return their combined delta. For catch-up, not steps."""
        return "".join(self.append(token_id) for token_id in token_ids)

    def flush(self) -> str:
        """Emit whatever is still held back, because the stream is over.

        Called once, at the end. A generation that ran out of budget in the middle
        of a character leaves real bytes held; `decode` renders them as a
        replacement character, and printing that is right, because it is exactly
        what the non-streaming endpoint prints for the same ids. Idempotent, since
        a stream can end from more than one place.
        """
        if self.num_held == 0:
            return ""
        pre = self.tokenizer.decode(self.token_ids[self._prefix_offset : self._read_offset])
        whole = self.tokenizer.decode(self.token_ids[self._prefix_offset :])
        delta = whole[len(pre) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(self.token_ids)
        if delta:
            self._text_parts.append(delta)
        return delta
