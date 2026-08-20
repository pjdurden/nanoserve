"""Day 39 tests: a stream is not a list of decoded tokens.

The unary server decodes once, at the end, over the whole output. Streaming has
no "end" to wait for: the caller wants text at every token, so something has to
turn a growing list of ids into a growing string. The obvious thing,
`tokenizer.decode([new_id])` per token, is wrong, and this file is mostly about
how wrong and why.

Llama-3's tokenizer is byte-level BPE. A token is a run of *bytes*, not
characters, and nothing makes those runs line up with UTF-8 boundaries. A 4-byte
emoji is routinely two tokens, so decoding either half alone hands back U+FFFD,
the replacement character, and the stream prints two black diamonds where the
model wrote one emoji. The whole-list decode of the same ids is correct, which is
the tell: the bug lives entirely in the incremental path, so a unary test suite
cannot see it.

`ByteTokenizer` here maps id 0..255 to that byte and decodes with
`errors="replace"`, which *is* the mechanism, not a mock of it. Everything these
tests prove about it is re-proved against the real Llama-3 tokenizer under
`requires_weights`.

The invariant the whole file is built on: **whatever the caller has been shown,
concatenated, equals the whole-list decode of the ids shown so far.** Streaming
may be late, never wrong. Being late is bounded and cheap; being wrong is
unfixable, because the bytes were already written to a socket.
"""

from __future__ import annotations

import random

import pytest

from nanoserve.detokenizer import IncrementalDetokenizer
from tests.reference import requires_weights

# --- a real byte-level tokenizer, small enough to reason about ------------------


class ByteTokenizer:
    """Id i is byte i. Decode is UTF-8 with replacement, which is the whole point.

    This is not a simplification of Llama-3's tokenizer, it is the pathological
    corner of it made total: every multi-byte character is split across tokens, so
    a test does not have to go hunting for an id pair that happens to break.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


TOK = ByteTokenizer()
REPLACEMENT = "�"


def _stream(text: str, tokenizer=TOK, **kw) -> tuple[list[str], str]:
    """Feed `text`'s ids one at a time. Returns (per-token deltas, final text)."""
    detok = IncrementalDetokenizer(tokenizer, **kw)
    deltas = [detok.append(i) for i in tokenizer.encode(text)]
    deltas.append(detok.flush())
    return deltas, detok.text


# --- the invariant --------------------------------------------------------------


def test_ascii_streams_a_character_at_a_time():
    """The easy case, and the one a naive implementation also passes."""
    deltas, text = _stream("hello")
    assert text == "hello"
    assert [d for d in deltas if d] == ["h", "e", "l", "l", "o"]


def test_text_is_the_concatenation_of_the_deltas():
    """The caller sees the deltas; `text` is what they must add up to."""
    detok = IncrementalDetokenizer(TOK)
    seen = "".join(detok.append(i) for i in TOK.encode("a café story"))
    seen += detok.flush()
    assert seen == detok.text == "a café story"


@pytest.mark.parametrize(
    "text",
    ["\U0001f600", "café", "日本語", "naïve", "a \U0001f600 b \U0001f389 c"],
)
def test_incremental_equals_whole_decode(text):
    """The invariant, on characters that do not fit in one byte."""
    _, got = _stream(text)
    assert got == TOK.decode(TOK.encode(text))


def test_prefix_property_holds_at_every_step():
    """Not just at the end: what was shown is always a prefix of the truth.

    A stream that reached the right answer by emitting a wrong character and a
    correcting one would pass the final-equality test and still have written
    nonsense to a socket that cannot be un-written.
    """
    text = "he\U0001f600llo 日本"
    ids = TOK.encode(text)
    detok = IncrementalDetokenizer(TOK)
    for n, token_id in enumerate(ids, start=1):
        detok.append(token_id)
        assert TOK.decode(ids[:n]).startswith(detok.text) or detok.text == TOK.decode(ids[:n])
        assert text.startswith(detok.text)


# --- the bug this exists to fix -------------------------------------------------


def test_naive_per_token_decode_is_wrong_on_the_same_input():
    """Document the thing being fixed, so the fix is not mistaken for ceremony."""
    ids = TOK.encode("\U0001f600")
    naive = "".join(TOK.decode([i]) for i in ids)
    assert naive == REPLACEMENT * 4
    assert naive != TOK.decode(ids)
    _, streamed = _stream("\U0001f600")
    assert streamed == "\U0001f600"


def test_a_split_character_is_held_back_then_emitted_whole():
    """Nothing at all until the last byte, then the character in one delta."""
    detok = IncrementalDetokenizer(TOK)
    ids = TOK.encode("\U0001f600")
    assert [detok.append(i) for i in ids] == ["", "", "", "\U0001f600"]
    assert detok.flush() == ""


def test_holdback_is_bounded_by_three_tokens():
    """UTF-8 is at most 4 bytes and a token carries at least one, so at most 3 wait.

    This is the latency the correctness costs, and it is worth knowing that it is
    a constant rather than something that can grow with the output.
    """
    text = "a \U0001f600 日 é b \U0001f1ef\U0001f1f5"
    detok = IncrementalDetokenizer(TOK)
    run = worst = 0
    for token_id in TOK.encode(text):
        if detok.append(token_id):
            run = 0
        else:
            run += 1
            worst = max(worst, run)
    assert worst == 3
    assert detok.text + detok.flush() == text


# --- truncation, which is where flush earns its keep ----------------------------


def test_flush_emits_the_partial_character_a_budget_cut_left_behind():
    """`max_tokens` can end a generation mid-character. The bytes still happened.

    Holding them back forever would make the stream's text differ from the unary
    answer for the same ids, so the last word goes to `decode` of everything:
    a replacement character, which is exactly what the non-streaming server
    would have printed.
    """
    ids = TOK.encode("hi \U0001f600")[:-2]  # cut two bytes out of the emoji
    detok = IncrementalDetokenizer(TOK)
    for token_id in ids:
        detok.append(token_id)
    assert detok.text == "hi "
    assert detok.flush() == REPLACEMENT
    assert detok.text == TOK.decode(ids)


def test_flush_is_idempotent_and_empty_when_nothing_is_held():
    detok = IncrementalDetokenizer(TOK)
    for token_id in TOK.encode("done"):
        detok.append(token_id)
    assert detok.flush() == ""
    assert detok.flush() == ""
    assert detok.text == "done"


def test_nothing_appended_is_empty_text():
    detok = IncrementalDetokenizer(TOK)
    assert detok.text == ""
    assert detok.flush() == ""


# --- the prompt is context, not output ------------------------------------------


def test_prompt_tokens_are_context_and_are_never_emitted():
    """A completions stream shows the generation. The prompt is only there to
    give the decoder the bytes immediately before it."""
    prompt = TOK.encode("The capital is")
    detok = IncrementalDetokenizer(TOK, prompt_token_ids=prompt)
    got = "".join(detok.append(i) for i in TOK.encode(" Paris")) + detok.flush()
    assert got == " Paris"
    assert detok.text == " Paris"


def test_a_prompt_cut_mid_character_does_not_corrupt_the_first_delta():
    """The context window can land inside a multi-byte prompt character.

    The window then decodes with a leading replacement char, in both halves of
    the diff, so it cancels. If it did not, every stream after an emoji-final
    prompt would open with a stray U+FFFD.
    """
    prompt = TOK.encode("hi \U0001f600")
    detok = IncrementalDetokenizer(TOK, prompt_token_ids=prompt, context_tokens=2)
    got = "".join(detok.append(i) for i in TOK.encode("!")) + detok.flush()
    assert got == "!"


def test_extend_matches_appending_one_at_a_time():
    ids = TOK.encode("a \U0001f389 b")
    one = IncrementalDetokenizer(TOK)
    for token_id in ids:
        one.append(token_id)
    many = IncrementalDetokenizer(TOK)
    assert many.extend(ids) == one.text
    assert many.text == one.text


# --- fuzz -----------------------------------------------------------------------


def test_incremental_matches_whole_decode_on_random_unicode():
    """The invariant over 300 random strings, which is where a window bug shows."""
    alphabet = list("abc ,.\n") + [
        "\U0001f600", "\U0001f389", "\U0001f1ef\U0001f1f5", "日", "é",
        "ñ", "한", "→", "\U0001d54f",
    ]
    rng = random.Random(1)
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 20)))
        _, got = _stream(text)
        assert got == TOK.decode(TOK.encode(text)), text


def test_the_naive_path_would_have_failed_most_of_that_corpus():
    """Quantify it: this is not a rare corner, it is most unicode text."""
    alphabet = list("abc ,.\n") + ["\U0001f600", "\U0001f389", "日", "é"]
    rng = random.Random(1)
    wrong = 0
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 20)))
        ids = TOK.encode(text)
        if "".join(TOK.decode([i]) for i in ids) != TOK.decode(ids):
            wrong += 1
    assert wrong > 200


# --- the real tokenizer ---------------------------------------------------------


@requires_weights
def test_real_llama_tokenizer_splits_an_emoji_across_two_tokens():
    """The premise, checked against the thing the server actually loads."""
    from transformers import AutoTokenizer

    from tests.reference import WEIGHTS_DIR

    tok = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
    ids = tok.encode("\U0001f600", add_special_tokens=False)
    assert len(ids) == 2
    assert [tok.decode([i]) for i in ids] == [REPLACEMENT, REPLACEMENT]
    assert tok.decode(ids) == "\U0001f600"


@requires_weights
@pytest.mark.parametrize(
    "text",
    [
        "Paris is the capital of France.",
        "café au lait",
        "\U0001f600 hello \U0001f389",
        "日本語のテキスト",
        "  leading and trailing  ",
    ],
)
def test_real_llama_tokenizer_streams_exactly(text):
    from transformers import AutoTokenizer

    from tests.reference import WEIGHTS_DIR

    tok = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
    ids = tok.encode(text, add_special_tokens=False)
    detok = IncrementalDetokenizer(tok)
    got = "".join(detok.append(i) for i in ids) + detok.flush()
    assert got == tok.decode(ids)
