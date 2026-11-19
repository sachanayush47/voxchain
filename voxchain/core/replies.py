"""Tracking the agent's replies from the user's turn to the last of their audio, and what barge-in should undo."""

import asyncio
from collections import deque
from dataclasses import dataclass


@dataclass(eq=False)
class Reply:
    """The agent's reply to one committed user turn."""

    text: str
    task: asyncio.Task | None = None
    first_heard_at: float | None = None


@dataclass(frozen=True)
class Interruption:
    """What a barge-in undoes: replies to cancel, whether TTS must drop work, and user text to merge back."""

    cancelled: list[Reply]
    cancel_synthesis: bool
    resume_text: str


class ReplyTracker:
    """Follows each reply through generation, synthesis and playback.

    A reply is *active* from its user turn until TTS has synthesized all of its
    audio. On barge-in, a reply the user heard for no longer than
    ``resume_window`` seconds means the end-of-turn call was premature: its user
    text is handed back to be merged with what they say next. ``None`` never merges.
    """

    def __init__(self, resume_window: float | None) -> None:
        self.resume_window = resume_window
        self._replies: list[Reply] = []
        self._speaking: deque[Reply] = deque()
        self._last_heard: Reply | None = None

    @property
    def active(self) -> bool:
        """Whether any reply is still being generated or synthesized."""
        return bool(self._replies)

    @property
    def latest(self) -> Reply | None:
        """The most recent active reply, which the next one must wait for."""
        return self._replies[-1] if self._replies else None

    def start(self, text: str) -> Reply:
        """Track a reply to a newly committed user turn."""
        reply = Reply(text)
        self._replies.append(reply)
        return reply

    def spoke(self, reply: Reply) -> None:
        """A reply sent its first sentence to TTS. Replies are synthesized in this order."""
        self._speaking.append(reply)

    def finished_generating(self, reply: Reply) -> None:
        """A reply's agent turn ended. One that never spoke is done; one that did waits for TTS."""
        if reply not in self._speaking and reply in self._replies:
            self._replies.remove(reply)

    def audio_played(self, now: float) -> None:
        """Audio of the reply now being synthesized reached the client."""
        if self._speaking and self._speaking[0].first_heard_at is None:
            self._speaking[0].first_heard_at = now
            self._last_heard = self._speaking[0]

    def synthesized(self) -> None:
        """TTS finished the oldest speaking reply's audio."""
        if self._speaking:
            done = self._speaking.popleft()
            if done in self._replies:
                self._replies.remove(done)

    def interrupt(self, now: float) -> Interruption:
        """The user barged in: forget every reply and say what to undo."""
        candidates = self._replies if self._last_heard in self._replies else [self._last_heard, *self._replies]
        window = self.resume_window
        resumed = [
            reply.text
            for reply in candidates
            if reply is not None
            and window is not None
            and (reply.first_heard_at is None or now - reply.first_heard_at <= window)
        ]
        interruption = Interruption(
            cancelled=list(self._replies), cancel_synthesis=self.active, resume_text=" ".join(resumed)
        )
        self._replies.clear()
        self._speaking.clear()
        self._last_heard = None
        return interruption
