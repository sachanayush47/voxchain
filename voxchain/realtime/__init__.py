"""Speech-to-speech voice agents: one realtime model that hears, reasons, calls tools and speaks."""

from voxchain.realtime.base import RealtimeProvider, ToolSpec
from voxchain.realtime.events import (
    AgentTranscript,
    AudioDelta,
    FunctionCall,
    RealtimeEvent,
    ResponseDone,
    ResponseStarted,
    SessionClosed,
    SpeechStarted,
    SpeechStopped,
    UserTranscript,
)
from voxchain.realtime.history import ConversationHistory, Turn
from voxchain.realtime.openai import OpenAIRealtimeProvider
from voxchain.realtime.pipeline import RealtimePipeline
from voxchain.realtime.playback import AudioAction, Cut, PlaybackTracker, ResponseOutcome
from voxchain.realtime.response_gate import ResponseGate
from voxchain.realtime.tools import NOT_RUN_OUTPUT, ToolRunner, tool_spec

__all__ = [
    "NOT_RUN_OUTPUT",
    "AgentTranscript",
    "AudioAction",
    "AudioDelta",
    "ConversationHistory",
    "Cut",
    "FunctionCall",
    "OpenAIRealtimeProvider",
    "PlaybackTracker",
    "RealtimeEvent",
    "RealtimePipeline",
    "RealtimeProvider",
    "ResponseDone",
    "ResponseGate",
    "ResponseOutcome",
    "ResponseStarted",
    "SessionClosed",
    "SpeechStarted",
    "SpeechStopped",
    "ToolRunner",
    "ToolSpec",
    "Turn",
    "UserTranscript",
    "tool_spec",
]
