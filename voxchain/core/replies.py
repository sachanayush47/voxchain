"""Follows the replies of the agent from the user turn to the end of their audio, and what a barge-in must undo."""

import asyncio
from collections import deque
from dataclasses import dataclass, field

# A typical TTS speech rate. The estimate of heard text uses it until all the audio of a reply is known.
SPEECH_CHARS_PER_SECOND = 15.0


@dataclass(eq=False)
class Reply:
    """The reply of the agent to one committed user turn."""

    text: str
    resumable: bool = True
    task: asyncio.Task | None = None
    first_heard_at: float | None = None
    spoken: list[str] = field(default_factory=list)
    audio_seconds: float = 0.0
    complete: bool = False
    tool_calls: int = 0

    def heard_text(self, now: float) -> str:
        """Estimate the text that the user heard until ``now``, cut at a word boundary."""
        text = " ".join(self.spoken)
        if self.first_heard_at is None or not text:
            return ""
        seconds = max(0.0, now - self.first_heard_at)
        if self.complete and self.audio_seconds > 0:
            chars = len(text) * min(1.0, seconds / self.audio_seconds)
        else:
            chars = seconds * SPEECH_CHARS_PER_SECOND
        if chars >= len(text):
            return text
        return text[: int(chars)].rsplit(" ", 1)[0].strip() if " " in text[: int(chars)] else ""


@dataclass(frozen=True)
class Interruption:
    """What a barge-in undoes.

    ``cancelled`` are the replies to cancel. ``cancel_synthesis`` tells if TTS
    must discard work. ``resumed`` are the replies whose user text goes back to
    the user, as ``resume_text``. ``heard`` is the reply that the user heard
    in part, and ``heard_text`` is the text that they heard.
    """

    cancelled: list[Reply]
    cancel_synthesis: bool
    resume_text: str
    resumed: list[Reply] = field(default_factory=list)
    heard: Reply | None = None
    heard_text: str = ""


class ReplyTracker:
    """Follows each reply through generation, synthesis and playback.

    A reply is *active* from its user turn until TTS synthesized all its audio.
    If the user heard a reply for no longer than ``resume_window`` seconds before
    a barge-in, the end-of-turn decision was too early. Then the tracker gives
    back the user text of the reply, so that it merges with the next words of
    the user. If ``resume_window`` is ``None``, the tracker never merges.
    """

    def __init__(self, resume_window: float | None) -> None:
        self.resume_window = resume_window
        self._replies: list[Reply] = []
        self._speaking: deque[Reply] = deque()
        self._last_heard: Reply | None = None

    @property
    def active(self) -> bool:
        """Whether a reply is still in generation or synthesis."""
        return bool(self._replies)

    @property
    def latest(self) -> Reply | None:
        """The most recent active reply. The next reply must wait for it."""
        return self._replies[-1] if self._replies else None

    @property
    def tool_running(self) -> bool:
        """Whether an active reply waits for a tool call."""
        return any(reply.tool_calls for reply in self._replies)

    def is_active(self, reply: Reply) -> bool:
        """Whether ``reply`` is still in generation or synthesis."""
        return reply in self._replies

    def start(self, text: str, resumable: bool = True) -> Reply:
        """Follow a reply to a new committed user turn.

        A barge-in never gives back the text of a reply that is not ``resumable``.
        """
        reply = Reply(text, resumable)
        self._replies.append(reply)
        return reply

    def spoke(self, reply: Reply) -> None:
        """A reply sent its first sentence to TTS. TTS synthesizes replies in this order."""
        self._speaking.append(reply)

    def finished_generating(self, reply: Reply) -> None:
        """The agent turn of a reply ended. A reply that never spoke is done. A reply that spoke waits for TTS."""
        if reply not in self._speaking and reply in self._replies:
            self._replies.remove(reply)

    def audio_received(self, seconds: float) -> Reply | None:
        """TTS sent ``seconds`` of audio. Return the reply in synthesis, which owns the audio."""
        if not self._speaking:
            return None
        self._speaking[0].audio_seconds += seconds
        return self._speaking[0]

    def audio_played(self, reply: Reply | None, now: float) -> None:
        """Audio of ``reply`` got to the client at ``now``. Playback can come after TTS completed the reply."""
        if reply is not None and reply.first_heard_at is None:
            reply.first_heard_at = now
            self._last_heard = reply

    def synthesized(self) -> None:
        """TTS completed the audio of the oldest reply in synthesis."""
        if self._speaking:
            done = self._speaking.popleft()
            done.complete = True
            if done in self._replies:
                self._replies.remove(done)

    def interrupt(self, now: float) -> Interruption:
        """The user interrupted: forget all replies and tell what to undo."""
        candidates = self._replies if self._last_heard in self._replies else [self._last_heard, *self._replies]
        window = self.resume_window
        resumed = [
            reply
            for reply in candidates
            if reply is not None
            and reply.text
            and reply.resumable
            and window is not None
            and (reply.first_heard_at is None or now - reply.first_heard_at <= window)
        ]
        heard = self._last_heard if self._last_heard is not None and self._last_heard not in resumed else None
        interruption = Interruption(
            cancelled=list(self._replies),
            cancel_synthesis=self.active,
            resume_text=" ".join(reply.text for reply in resumed),
            resumed=resumed,
            heard=heard,
            heard_text=heard.heard_text(now) if heard is not None else "",
        )
        self._replies.clear()
        self._speaking.clear()
        self._last_heard = None
        return interruption
