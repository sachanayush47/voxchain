"""The events that a pipeline sends to the client: audio to play, when to stop, and the state of the conversation."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayAudio:
    """Play this audio after all audio already in the queue. The data is raw PCM16 at the output sample rate."""

    data: bytes


@dataclass(frozen=True)
class StopPlayback:
    """The user interrupted the agent. Stop playback now and discard all audio in the queue."""


@dataclass(frozen=True)
class ReplyFinished:
    """The pipeline sent all the audio of the current reply."""


@dataclass(frozen=True)
class StreamClosed:
    """The pipeline lost its session permanently and will stop."""

    reason: str


Output = PlayAudio | StopPlayback | ReplyFinished | StreamClosed


class OutputChannel:
    """Delivers output events to the client from its own task, and calculates when each audio chunk plays.

    The channel assumes that the client plays chunks one after the other in real
    time. If the client has no audio left, it starts a chunk immediately when the
    chunk arrives. ``on_played(tag, start, end)`` receives the start and end time
    of each chunk, measured on ``clock``. Delivery occurs in :meth:`run`, apart
    from the code that adds events. Thus a callback that blocks during playback
    never delays that code. If a callback raises, the channel logs the error and continues.
    """

    def __init__(
        self,
        callback: Callable[[Output], Awaitable[None]],
        *,
        sample_rate: int,
        clock: Callable[[], float],
        on_played: Callable[[object, float, float], None] | None = None,
    ) -> None:
        self.callback = callback
        self.clock = clock
        self.on_played = on_played
        self._bytes_per_second = sample_rate * 2
        self._queue: asyncio.Queue[tuple[Output, object]] = asyncio.Queue()
        self._playing_until = 0.0

    def play(self, data: bytes, tag: object = None) -> None:
        """Queue audio. The channel sends ``tag`` to ``on_played``."""
        self._queue.put_nowait((PlayAudio(data), tag))

    def reply_finished(self) -> None:
        """Queue a ``ReplyFinished`` after the audio that is already in the queue."""
        self._queue.put_nowait((ReplyFinished(), None))

    def playing(self, now: float) -> bool:
        """Whether the client still plays audio that the channel sent to it."""
        return now < self._playing_until

    def has_queued_audio(self) -> bool:
        """Whether events wait in the queue for delivery to the client."""
        return not self._queue.empty()

    def stop(self, now: float) -> None:
        """Handle a barge-in: discard the queue and tell the client to stop playback."""
        _drain(self._queue)
        self._playing_until = now
        self._queue.put_nowait((StopPlayback(), None))

    async def close(self, reason: str) -> None:
        """Discard the queue and deliver ``StreamClosed`` immediately."""
        _drain(self._queue)
        await self._deliver(StreamClosed(reason=reason))

    async def run(self) -> None:
        """Send the events in the queue to the client in order. This never returns."""
        while True:
            event, tag = await self._queue.get()
            if isinstance(event, PlayAudio):
                start = max(self.clock(), self._playing_until)
                self._playing_until = start + len(event.data) / self._bytes_per_second
                if self.on_played is not None:
                    self.on_played(tag, start, self._playing_until)
            await self._deliver(event)

    async def _deliver(self, event: Output) -> None:
        try:
            await self.callback(event)
        except Exception:
            logger.exception(f"OutputChannel: Callback failed on {type(event).__name__}")


def _drain(queue: asyncio.Queue) -> None:
    while not queue.empty():
        queue.get_nowait()
