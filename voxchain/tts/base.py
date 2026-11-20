"""The interface that all text-to-speech providers implement."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from voxchain.tts.events import TTSEvent


class TTSProvider(ABC):
    """Real-time text-to-speech: sentences go in, and PCM16 audio comes out.

    The pipeline calls :meth:`speak` once for each sentence and :meth:`flush` at
    the end of a reply. One :class:`~voxchain.tts.events.Flushed` follows each
    flush, after the audio of that reply. After :meth:`cancel`, the provider
    yields nothing that the pipeline requested before the cancel, not even a
    ``Flushed`` that is still due. Providers reconnect when the stream drops.
    They yield :class:`~voxchain.session.SessionClosed` only if they cannot reconnect.
    """

    sample_rate: int

    @abstractmethod
    async def connect(self) -> None:
        """Open the stream. Raise an error if the vendor rejects it."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Queue a sentence for synthesis."""

    @abstractmethod
    async def flush(self) -> None:
        """Mark the end of a reply, so that the provider synthesizes the buffered text now."""

    @abstractmethod
    async def cancel(self) -> None:
        """Handle a barge-in: discard all requests up to now."""

    @abstractmethod
    def receive_events(self) -> AsyncIterator[TTSEvent]:
        """Yield audio and flush markers for the life of the conversation, across reconnects."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the stream. It is safe to call this more than once."""
