"""voxchain: a thin, event-driven library for real-time voice agents on LangChain/LangGraph.

There are two ways to build an agent:

- :class:`~voxchain.core.pipeline.VoxchainPipeline` connects streaming speech-to-text, a
  LangGraph agent and streaming text-to-speech in a chain.
- :class:`~voxchain.realtime.pipeline.RealtimePipeline` runs one speech-to-speech model.
  The model hears, reasons, calls LangChain tools and speaks.

The two pipelines send the same :mod:`voxchain.output` events to the client.
"""

from voxchain.core import VoxchainPipeline
from voxchain.output import Output, PlayAudio, ReplyFinished, StopPlayback, StreamClosed
from voxchain.realtime import OpenAIRealtimeProvider, RealtimePipeline, RealtimeProvider
from voxchain.stt import OpenAISTTProvider, SarvamSTTProvider, STTProvider
from voxchain.tts import OpenAITTSProvider, SarvamTTSProvider, TTSProvider
from voxchain.turn import EndOfTurnDetector, PipecatSmartTurnDetector

__version__ = "0.2.0"

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
