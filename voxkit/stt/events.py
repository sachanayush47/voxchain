"""Events a speech-to-text provider yields to the pipeline."""

from dataclasses import dataclass

from voxkit.events import SessionClosed, SpeechStarted, SpeechStopped


@dataclass(frozen=True)
class PartialTranscript:
    """An interim transcript that may still change."""

    text: str


@dataclass(frozen=True)
class FinalTranscript:
    """A finished transcript, safe to hand to the agent."""

    text: str


STTEvent = SpeechStarted | SpeechStopped | PartialTranscript | FinalTranscript | SessionClosed
