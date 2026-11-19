"""Speech-to-speech voice agents: one realtime model that hears, reasons, calls tools and speaks."""

from voxkit.realtime.base import RealtimeProvider, ToolSpec
from voxkit.realtime.events import (
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
from voxkit.realtime.history import ConversationHistory, Turn
from voxkit.realtime.openai import OpenAIRealtimeProvider
from voxkit.realtime.pipeline import RealtimePipeline
from voxkit.realtime.playback import AudioAction, Cut, PlaybackTracker, ResponseOutcome
from voxkit.realtime.response_gate import ResponseGate
from voxkit.realtime.tools import NOT_RUN_OUTPUT, ToolRunner, tool_spec

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
