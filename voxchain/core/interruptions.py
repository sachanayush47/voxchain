"""Decides from the words of the user whether speech over the agent is a real interruption."""

import re
import unicodedata

STOP_WORDS = frozenset(
    {"stop", "wait", "no", "nope", "hold", "pause", "cancel", "enough", "quiet", "halt", "wrong", "incorrect"}
)
"""Words that interrupt the agent even alone, for example "stop" or "no, wait"."""

STOP_PHRASES = frozenset({"hold on", "hang on", "one moment", "one second", "excuse me", "shut up"})
"""Short phrases that interrupt the agent although they are shorter than the minimum."""

# Chinese, Japanese and Korean text has no spaces, so each of these characters counts as one word.
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def normalize(text: str) -> str:
    """Return ``text`` in lowercase, without punctuation and with single spaces."""
    kept = "".join(c if c.isalnum() or c.isspace() or c == "'" else " " for c in text.lower())
    return " ".join(kept.split())


def count_words(text: str) -> int:
    """Count the spoken words in ``text``. Each Chinese, Japanese or Korean character counts as one word."""
    cjk = len(CJK.findall(text))
    other = [word for word in CJK.sub(" ", normalize(unicodedata.normalize("NFKC", text))).split() if word]
    return cjk + len(other)


class InterruptionPolicy:
    """Decides whether a transcript of speech over the agent is a real interruption.

    A transcript interrupts if it has at least ``min_words`` words, if it is a
    stop phrase ("hold on"), or if all its words are stop words ("stop",
    "no wait"). Shorter speech, for example "uh-huh", "okay" or noise that the
    STT misheard, does not interrupt.

    STT providers often split one sentence into several transcripts. Thus the
    policy also adds up the transcripts that did not interrupt, for a maximum of
    ``window`` seconds, and decides on the combined text.
    """

    def __init__(
        self,
        *,
        min_words: int = 3,
        stop_words: frozenset[str] = STOP_WORDS,
        stop_phrases: frozenset[str] = STOP_PHRASES,
        window: float = 2.0,
    ) -> None:
        if min_words < 1:
            raise ValueError(f"The minimum number of words must be 1 or more, got {min_words}")
        self.min_words = min_words
        self.stop_words = stop_words
        self.stop_phrases = stop_phrases
        self.window = window
        self._pending: list[str] = []
        self._pending_at = 0.0

    def confirms(self, text: str) -> bool:
        """Whether ``text`` alone is a real interruption."""
        words = normalize(text).split()
        if not words:
            return False
        return (
            count_words(text) >= self.min_words
            or " ".join(words) in self.stop_phrases
            or all(word in self.stop_words for word in words)
        )

    def hear(self, text: str, now: float) -> str | None:
        """Add a final transcript of speech over the agent.

        Return all the text heard since the last decision if it is a real
        interruption. Return ``None`` if the agent continues.
        """
        if self._pending and now - self._pending_at > self.window:
            self._pending.clear()
        self._pending.append(text)
        self._pending_at = now
        combined = " ".join(self._pending)
        if not self.confirms(combined):
            return None
        self._pending.clear()
        return combined

    def reset(self) -> None:
        """Forget the transcripts that did not interrupt."""
        self._pending.clear()
