"""SentenceSegmenter tests — the hardest pure logic in the repo.

Pins the contract the voice path depends on: sentences emit the moment their
punctuation closes, <think> blocks never reach speech (including partial
markers split across tokens), and Hindi Danda is a sentence boundary.
"""

from snap.pc.sentence_stream import (
    _SHAPES_CLOSE,
    _SHAPES_OPEN,
    SentenceSegmenter,
    _partial_marker_len,
)


def test_no_boundary_no_emit() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("just words without punctuation") == []
    assert seg.flush() == "just words without punctuation"


def test_empty_token_ignored() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("") == []


def test_sentence_emits_on_boundary_close() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("Hello there.") == []  # no trailing space yet — boundary not closed
    assert seg.feed(" Hi.") == ["Hello there."]
    assert seg.flush() == "Hi."


def test_multiple_sentences_one_feed() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("A. B. C.") == ["A.", "B."]
    assert seg.flush() == "C."


def test_question_and_exclamation_boundaries() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("Wait! What? Hmm.") == ["Wait!", "What?"]
    assert seg.flush() == "Hmm."


def test_think_block_never_spoken() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("<think>internal reasoning. hidden.</think>") == []
    assert seg.feed(" Visible answer.") == []
    assert seg.flush() == "Visible answer."


def test_think_block_across_tokens() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("<think") == []  # partial marker held back, not spoken
    assert seg.feed(">secret stuff.</think>Hello.") == []
    assert seg.flush() == "Hello."


def test_thinking_variant_marker() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("<thinking>abc</thinking>Hi!") == []
    assert seg.flush() == "Hi!"


def test_unterminated_think_block_yields_none() -> None:
    seg = SentenceSegmenter()
    seg.feed("<think>never closed")
    assert seg.flush() is None


def test_hindi_danda_boundary() -> None:
    seg = SentenceSegmenter()
    assert seg.feed("नमस्ते।") == []
    assert seg.feed(" आप कैसे हैं?") == ["नमस्ते।"]
    assert seg.flush() == "आप कैसे हैं?"


def test_long_punctuation_free_text_flushes_at_clause() -> None:
    seg = SentenceSegmenter()
    out = seg.feed("one two three four, " * 30)
    assert len(out) >= 1  # flushed before the buffer grew unbounded
    assert seg.flush() is None


def test_partial_marker_len() -> None:
    assert _partial_marker_len("plain text", _SHAPES_OPEN) == 0
    assert _partial_marker_len("wait <thi", _SHAPES_OPEN) == 4
    assert _partial_marker_len("done </th", _SHAPES_CLOSE) == 4
