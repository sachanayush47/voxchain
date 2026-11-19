"""The interface every speech-to-speech provider implements."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from voxkit.realtime.events import RealtimeEvent


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call: a name, a description and a JSON Schema for its arguments."""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)


class RealtimeProvider(ABC):
    """A model that hears, reasons, calls tools and speaks in one session.

    Providers own the session: they reconnect when it drops and restore the
    conversation so far. :meth:`receive_events` yields
    :class:`~voxkit.realtime.events.SessionClosed` only when they can't.
    """

    input_sample_rate: int
    output_sample_rate: int
    interrupts_on_speech: bool = True

    @abstractmethod
    def set_tools(self, tools: list[ToolSpec]) -> None:
        """Declare the tools the model may call. Call before :meth:`connect`."""

    @abstractmethod
    async def connect(self) -> None:
        """Open and configure a session, raising if the provider rejects it."""

    @abstractmethod
    def receive_events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield events for the life of the conversation, across reconnects."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 audio at ``input_sample_rate``. Dropped while the session is reconnecting."""

    @abstractmethod
    async def send_function_result(self, call_id: str, output: str) -> None:
        """Hand a tool's result back to the model."""

    @abstractmethod
    async def request_response(self) -> None:
        """Have the model respond to the results just sent, unless a response that sees them is already coming."""

    @abstractmethod
    async def truncate(self, item_id: str, audio_end_ms: int) -> None:
        """Cut a spoken item down to the milliseconds the user heard."""

    @abstractmethod
    async def delete_item(self, item_id: str) -> None:
        """Remove a spoken item the user heard none of."""

    @abstractmethod
    async def disconnect(self) -> None:
        """End the session. Safe to call more than once."""
