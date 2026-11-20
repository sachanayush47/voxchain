"""Speech-to-text providers and the interface that they implement."""

from voxchain.session import SessionClosed
from voxchain.stt.base import STTProvider
from voxchain.stt.events import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped, STTEvent
from voxchain.stt.openai import OpenAISTTProvider
from voxchain.stt.sarvam import SarvamSTTProvider

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
