"""Text-to-speech providers and the interface they implement."""

from voxchain.session import SessionClosed
from voxchain.tts.base import TTSProvider
from voxchain.tts.events import AudioChunk, Flushed, TTSEvent
from voxchain.tts.openai import VERBATIM_INSTRUCTIONS, OpenAITTSProvider
from voxchain.tts.sarvam import SarvamTTSProvider

__all__ = [
    "VERBATIM_INSTRUCTIONS",
    "AudioChunk",
    "Flushed",
    "OpenAITTSProvider",
    "SarvamTTSProvider",
    "SessionClosed",
    "TTSEvent",
    "TTSProvider",
]
