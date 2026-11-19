"""The interface every speech-to-text provider implements."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voxchain.stt.events import STTEvent


class STTProvider(ABC):
    """Streaming speech-to-text: audio in, voice activity and transcripts out.

    Providers report voice activity themselves (from the vendor, or a local VAD),
    reconnect when the stream drops, and yield
    :class:`~voxchain.session.SessionClosed` only when they can't.
    """

    sample_rate: int

    @abstractmethod
    async def connect(self) -> None:
        """Open the stream, raising if the provider rejects it."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 audio at ``sample_rate``. Dropped while the stream is reconnecting."""

    @abstractmethod
    def receive_events(self) -> AsyncIterator[STTEvent]:
        """Yield events for the life of the conversation, across reconnects."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the stream. Safe to call more than once."""
