"""The interface every text-to-speech provider implements."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voxkit.tts.events import TTSEvent


class TTSProvider(ABC):
    """Streaming text-to-speech: sentences in, PCM16 audio out.

    The pipeline calls :meth:`speak` once per sentence and :meth:`flush` at the
    end of a reply; one :class:`~voxkit.tts.events.Flushed` follows each flush,
    after that reply's audio. After :meth:`cancel`, nothing requested before it
    is yielded, not even a pending ``Flushed``. Providers reconnect when the stream drops and
    yield :class:`~voxkit.session.SessionClosed` only when they can't.
    """

    sample_rate: int

    @abstractmethod
    async def connect(self) -> None:
        """Open the stream, raising if the provider rejects it."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Queue a sentence for synthesis."""

    @abstractmethod
    async def flush(self) -> None:
        """Mark the end of a reply, so buffered text is synthesized now."""

    @abstractmethod
    async def cancel(self) -> None:
        """Barge-in: abandon everything requested so far."""

    @abstractmethod
    def receive_events(self) -> AsyncIterator[TTSEvent]:
        """Yield audio and flush markers for the life of the conversation, across reconnects."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the stream. Safe to call more than once."""
