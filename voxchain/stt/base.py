"""The interface that all speech-to-text providers implement."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voxchain.stt.events import STTEvent


class STTProvider(ABC):
    """Real-time speech-to-text: audio goes in, and voice activity and transcripts come out.

    Providers report voice activity themselves, from the vendor or from a local
    VAD. They reconnect when the stream drops. They yield
    :class:`~voxchain.session.SessionClosed` only if they cannot reconnect.
    """

    sample_rate: int

    @abstractmethod
    async def connect(self) -> None:
        """Open the stream. Raise an error if the vendor rejects it."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 audio at ``sample_rate``. The provider discards audio during a reconnect."""

    @abstractmethod
    def receive_events(self) -> AsyncIterator[STTEvent]:
        """Yield events for the life of the conversation, across reconnects."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the stream. It is safe to call this more than once."""
