"""Events that a speech-to-text provider yields to the pipeline."""

from dataclasses import dataclass

from voxchain.events import SessionClosed, SpeechStarted, SpeechStopped


@dataclass(frozen=True)
class PartialTranscript:
    """An interim transcript that can still change."""

    text: str


@dataclass(frozen=True)
class FinalTranscript:
    """A complete transcript that is safe to send to the agent."""

    text: str


STTEvent = SpeechStarted | SpeechStopped | PartialTranscript | FinalTranscript | SessionClosed
