"""The interface every speech-to-speech provider implements."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from voxchain.realtime.events import RealtimeEvent


@dataclass(frozen=True)
class ToolSpec:
    """A tool that the model can call: a name, a description and a JSON Schema for its arguments."""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)


class RealtimeProvider(ABC):
    """A model that hears, reasons, calls tools and speaks in one session.

    The provider owns the session. If the connection fails, the provider
    reconnects and restores the conversation. :meth:`receive_events` yields
    :class:`~voxchain.realtime.events.SessionClosed` only if the provider cannot do this.
    """

    input_sample_rate: int
    output_sample_rate: int
    interrupts_on_speech: bool = True

    @abstractmethod
    def set_tools(self, tools: list[ToolSpec]) -> None:
        """Set the tools that the model can call. Call this method before :meth:`connect`."""

    @abstractmethod
    async def connect(self) -> None:
        """Open and configure a session. Raise an error if the vendor rejects the session."""

    @abstractmethod
    def receive_events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield events until the conversation ends, across reconnects."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 audio at ``input_sample_rate``. The provider discards the audio while it reconnects."""

    @abstractmethod
    async def send_function_result(self, call_id: str, output: str) -> None:
        """Send the result of a tool to the model."""

    @abstractmethod
    async def request_response(self) -> None:
        """Ask the model to respond to the latest tool results, unless a later response will see them anyway."""

    @abstractmethod
    async def truncate(self, item_id: str, audio_end_ms: int) -> None:
        """Truncate a spoken item to the milliseconds that the user heard."""

    @abstractmethod
    async def delete_item(self, item_id: str) -> None:
        """Remove a spoken item when the user heard no part of it."""

    @abstractmethod
    async def disconnect(self) -> None:
        """End the session. You can call this method more than once."""
