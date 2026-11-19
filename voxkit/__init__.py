"""voxkit: a thin, event-driven library for building real-time voice agents on LangChain/LangGraph.

Two ways to build an agent:

- :class:`~voxkit.core.pipeline.VoxkitPipeline` chains streaming speech-to-text, any
  LangGraph agent and streaming text-to-speech.
- :class:`~voxkit.realtime.pipeline.RealtimePipeline` runs one speech-to-speech model
  that hears, reasons, calls LangChain tools and speaks.

Both send the client the same :mod:`voxkit.output` events.
"""

from voxkit.core import VoxkitPipeline
from voxkit.output import Output, PlayAudio, ReplyFinished, StopPlayback, StreamClosed
from voxkit.realtime import OpenAIRealtimeProvider, RealtimePipeline, RealtimeProvider
from voxkit.stt import OpenAISTTProvider, SarvamSTTProvider, STTProvider
from voxkit.tts import OpenAITTSProvider, SarvamTTSProvider, TTSProvider
from voxkit.turn import EndOfTurnDetector, PipecatSmartTurnDetector

__version__ = "0.0.1"

__all__ = [
    "EndOfTurnDetector",
    "OpenAIRealtimeProvider",
    "OpenAISTTProvider",
    "OpenAITTSProvider",
    "Output",
    "PipecatSmartTurnDetector",
    "PlayAudio",
    "RealtimePipeline",
    "RealtimeProvider",
    "ReplyFinished",
    "STTProvider",
    "SarvamSTTProvider",
    "SarvamTTSProvider",
    "StopPlayback",
    "StreamClosed",
    "TTSProvider",
    "VoxkitPipeline",
]
