"""voxchain: a thin, event-driven library for building real-time voice agents on LangChain/LangGraph.

Two ways to build an agent:

- :class:`~voxchain.core.pipeline.VoxchainPipeline` chains streaming speech-to-text, any
  LangGraph agent and streaming text-to-speech.
- :class:`~voxchain.realtime.pipeline.RealtimePipeline` runs one speech-to-speech model
  that hears, reasons, calls LangChain tools and speaks.

Both send the client the same :mod:`voxchain.output` events.
"""

from voxchain.core import VoxchainPipeline
from voxchain.output import Output, PlayAudio, ReplyFinished, StopPlayback, StreamClosed
from voxchain.realtime import OpenAIRealtimeProvider, RealtimePipeline, RealtimeProvider
from voxchain.stt import OpenAISTTProvider, SarvamSTTProvider, STTProvider
from voxchain.tts import OpenAITTSProvider, SarvamTTSProvider, TTSProvider
from voxchain.turn import EndOfTurnDetector, PipecatSmartTurnDetector

__version__ = "0.1.0"

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
    "VoxchainPipeline",
]
