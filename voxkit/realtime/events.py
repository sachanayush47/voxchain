"""Events a speech-to-speech provider yields to the pipeline."""

from dataclasses import dataclass

from voxkit.events import SessionClosed, SpeechStarted, SpeechStopped


@dataclass(frozen=True)
class UserTranscript:
    """What the user said in one turn."""

    text: str


@dataclass(frozen=True)
class ResponseStarted:
    """The model started a response."""

    response_id: str


@dataclass(frozen=True)
class AudioDelta:
    """A chunk of the model's speech: PCM16 at the provider's ``output_sample_rate``."""

    response_id: str
    item_id: str
    data: bytes


@dataclass(frozen=True)
class AgentTranscript:
    """What the model said in one spoken item."""

    response_id: str
    item_id: str
    text: str


@dataclass(frozen=True)
class FunctionCall:
    """The model wants a tool run. Yielded just before its response's ``ResponseDone``."""

    response_id: str
    call_id: str
    name: str
    arguments: str  # JSON object string


@dataclass(frozen=True)
class ResponseDone:
    """A response ended: ``completed``, ``cancelled``, ``failed`` or ``incomplete``."""

    response_id: str
    status: str


RealtimeEvent = (
    SpeechStarted
    | SpeechStopped
    | UserTranscript
    | ResponseStarted
    | AudioDelta
    | AgentTranscript
    | FunctionCall
    | ResponseDone
    | SessionClosed
)
