"""What a pipeline tells the client: audio to play, when to stop, and how the conversation is going."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayAudio:
    """Play this audio after anything already queued. Raw PCM16 at the provider's output sample rate."""

    data: bytes


@dataclass(frozen=True)
class StopPlayback:
    """The user barged in: stop playing now and drop anything queued."""


@dataclass(frozen=True)
class ReplyFinished:
    """The current reply's audio has all been sent."""


@dataclass(frozen=True)
class StreamClosed:
    """The session was lost for good; the pipeline is stopping."""

    reason: str


Output = PlayAudio | StopPlayback | ReplyFinished | StreamClosed


class OutputChannel:
    """Delivers output events to the client from its own task, and models when each audio chunk plays.

    The client is assumed to play chunks back to back in real time, starting
    each one as soon as it arrives if it had run dry. ``on_played(tag, start,
    end)`` hears when each chunk plays, as times on ``clock``. Delivery runs in
    :meth:`run`, separate from whoever queues events, so a callback that blocks
    on playback never delays them; a callback that raises is logged and skipped.
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
        """Queue audio; ``tag`` is handed back to ``on_played``."""
        self._queue.put_nowait((PlayAudio(data), tag))

    def reply_finished(self) -> None:
        """Queue a ``ReplyFinished`` after the audio already queued."""
        self._queue.put_nowait((ReplyFinished(), None))

    def playing(self, now: float) -> bool:
        """Whether the client is still playing audio already handed to it."""
        return now < self._playing_until

    def has_queued_audio(self) -> bool:
        """Whether anything is waiting to be handed to the client."""
        return not self._queue.empty()

    def stop(self, now: float) -> None:
        """Barge-in: drop everything queued and tell the client to stop playing."""
        _drain(self._queue)
        self._playing_until = now
        self._queue.put_nowait((StopPlayback(), None))

    async def close(self, reason: str) -> None:
        """Drop everything queued and deliver ``StreamClosed`` straight away."""
        _drain(self._queue)
        await self._deliver(StreamClosed(reason=reason))

    async def run(self) -> None:
        """Hand queued events to the client, in order, forever."""
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
