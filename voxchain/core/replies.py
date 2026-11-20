"""Tracks the agent's replies from the user turn to the end of their audio, and what a barge-in must undo."""

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
    """What a barge-in undoes: the replies to cancel, whether TTS must discard work, and the user text to restore."""

    cancelled: list[Reply]
    cancel_synthesis: bool
    resume_text: str


class ReplyTracker:
    """Follows each reply through generation, synthesis and playback.

    A reply is *active* from its user turn until TTS completes the synthesis of
    all its audio. At a barge-in, a reply that the user heard for ``resume_window``
    seconds or less shows that the end-of-turn decision was too early. The tracker
    then returns the user text of that reply, and the pipeline merges it with the
    next words of the user. If ``resume_window`` is ``None``, the tracker never merges.
    """

    def __init__(self, resume_window: float | None) -> None:
        self.resume_window = resume_window
        self._replies: list[Reply] = []
        self._speaking: deque[Reply] = deque()
        self._last_heard: Reply | None = None

    @property
    def active(self) -> bool:
        """Whether any reply is still in generation or in synthesis."""
        return bool(self._replies)

    @property
    def latest(self) -> Reply | None:
        """The most recent active reply. The next reply must wait for it."""
        return self._replies[-1] if self._replies else None

    def start(self, text: str) -> Reply:
        """Track a reply to a new committed user turn."""
        reply = Reply(text)
        self._replies.append(reply)
        return reply

    def spoke(self, reply: Reply) -> None:
        """A reply sent its first sentence to TTS. TTS synthesizes the replies in this order."""
        self._speaking.append(reply)

    def finished_generating(self, reply: Reply) -> None:
        """The agent turn of a reply ended. If the reply did not speak, it is done. If it spoke, it waits for TTS."""
        if reply not in self._speaking and reply in self._replies:
            self._replies.remove(reply)

    def audio_played(self, now: float) -> None:
        """Audio of the reply that TTS synthesizes now reached the client."""
        if self._speaking and self._speaking[0].first_heard_at is None:
            self._speaking[0].first_heard_at = now
            self._last_heard = self._speaking[0]

    def synthesized(self) -> None:
        """TTS finished the audio of the oldest reply in the speech queue."""
        if self._speaking:
            done = self._speaking.popleft()
            if done in self._replies:
                self._replies.remove(done)

    def interrupt(self, now: float) -> Interruption:
        """The user interrupted: forget all replies and return what to undo."""
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
