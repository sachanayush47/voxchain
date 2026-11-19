"""Text-to-speech providers and the interface they implement."""

from voxkit.session import SessionClosed
from voxkit.tts.base import TTSProvider
from voxkit.tts.events import AudioChunk, Flushed, TTSEvent
from voxkit.tts.openai import VERBATIM_INSTRUCTIONS, OpenAITTSProvider
from voxkit.tts.sarvam import SarvamTTSProvider

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
