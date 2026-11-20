"""Turns streamed agent tokens into speakable sentences, so that TTS can start before the agent finishes."""

import re

BOUNDARY = re.compile(r"(?<=[.!?;:,])\s+|\n+")
HTML_TAG = re.compile(r"<[^>]+>")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
LIST_MARKER = re.compile(r"^\s*[-+*]\s+", re.MULTILINE)
MD_SYMBOLS = re.compile(r"[*_`#|~>•·▪]+|-{2,}")
WHITESPACE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
WORD_CHAR = re.compile(r"\w")


def to_speakable(text: str) -> str:
    """Remove markdown and HTML, so that TTS can speak ``text`` naturally.

    Returns ``""`` if no speakable text remains (for example, a table separator ``|---|``).
    TTS engines reject such text, so callers skip empty results.
    """
    text = HTML_TAG.sub(" ", text)
    text = MD_LINK.sub(r"\1", text)
    text = LIST_MARKER.sub("", text)
    text = MD_SYMBOLS.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text if WORD_CHAR.search(text) else ""


class SentenceSegmenter:
    """Collects streamed tokens and returns complete, speakable sentences.

    A piece ends at sentence or clause punctuation (``. ! ? ; : ,``) followed
    by whitespace, or at a newline. The segmenter merges a piece shorter than
    ``min_chars`` with the next piece. Thus abbreviations ("Dr. Smith") and
    short list items do not become small, separate TTS requests.

    Example:
        >>> segmenter = SentenceSegmenter()
        >>> segmenter.push("Hello there, how are")
        []
        >>> segmenter.push(" you today? I'm fine")
        ['Hello there, how are you today?']
        >>> segmenter.flush()
        ["I'm fine"]
    """

    def __init__(self, min_chars: int = 20) -> None:
        self.min_chars = min_chars
        self._buffer = ""
        self._carry = ""

    def push(self, token: str) -> list[str]:
        """Add a token and return the sentences it completed (often none)."""
        self._buffer += token
        *complete, self._buffer = BOUNDARY.split(self._buffer)
        sentences = []
        for piece in complete:
            self._carry = f"{self._carry} {piece}".strip()
            if len(to_speakable(self._carry)) >= self.min_chars:
                sentences.extend(self._take_carry())
        return sentences

    def flush(self) -> list[str]:
        """Return the remaining speakable text after the stream ends."""
        self._carry = f"{self._carry} {self._buffer}".strip()
        self._buffer = ""
        return self._take_carry()

    def _take_carry(self) -> list[str]:
        speakable, self._carry = to_speakable(self._carry), ""
        return [speakable] if speakable else []
