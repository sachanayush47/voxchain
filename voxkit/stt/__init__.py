"""Speech-to-text providers and the interface they implement."""

from voxkit.session import SessionClosed
from voxkit.stt.base import STTProvider
from voxkit.stt.events import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped, STTEvent
from voxkit.stt.openai import OpenAISTTProvider
from voxkit.stt.sarvam import SarvamSTTProvider

__all__ = [
    "FinalTranscript",
    "OpenAISTTProvider",
    "PartialTranscript",
    "STTEvent",
    "STTProvider",
    "SarvamSTTProvider",
    "SessionClosed",
    "SpeechStarted",
    "SpeechStopped",
]
