"""Collects the report of a session from the calls of a pipeline."""

import copy
import dataclasses
import json
import time
from collections.abc import Callable

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage

from voxchain.events import SpeechStarted, SpeechStopped
from voxchain.report.models import (
    AgentStartedSpeaking,
    AgentStoppedSpeaking,
    AudioRecording,
    BargeIn,
    ErrorOccurred,
    ErrorSource,
    FinalTranscriptReceived,
    PartialTranscriptReceived,
    ReportEvent,
    SessionEnded,
    SessionReport,
    ToolCall,
    ToolFinished,
    ToolStarted,
    TranscriptIgnored,
    Turn,
    TurnCommitted,
    TurnKind,
    Usage,
    UserSpeechStarted,
    UserSpeechStopped,
)
from voxchain.stt import FinalTranscript, PartialTranscript


class AudioTrack:
    """PCM16 mono audio on the timeline of the session. Silence fills the gaps."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.data = bytearray()

    def write(self, at: float, pcm: bytes) -> None:
        """Put ``pcm`` at ``at`` seconds. It replaces audio that is already there."""
        position = max(0, round(at * self.sample_rate)) * 2
        if position > len(self.data):
            self.data += bytes(position - len(self.data))
        self.data[position : position + len(pcm)] = pcm

    def append(self, at: float, pcm: bytes) -> None:
        """Add ``pcm`` after the audio in the track. The first audio goes to ``at`` seconds."""
        if self.data:
            self.data += pcm
        else:
            self.write(at, pcm)

    def cut(self, at: float) -> None:
        """Discard the audio after ``at`` seconds."""
        del self.data[max(0, round(at * self.sample_rate)) * 2 :]


class SessionRecorder:
    """Collects the turns, events, usage and audio of one session. It does no I/O.

    The pipeline calls it at each step. ``clock`` is the clock of the pipeline.
    The recorder keeps all the audio of the session in memory: approximately
    2 MB for each minute of each track at 16 kHz.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        input_sample_rate: int,
        agent_sample_rate: int,
        thread_id: str = "",
        providers: dict[str, str] | None = None,
        settings: dict[str, object] | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.clock = clock
        self.wall_clock = wall_clock
        self.thread_id = thread_id
        self.providers = providers or {}
        self.settings = settings or {}
        self.input_sample_rate = input_sample_rate
        self.agent_sample_rate = agent_sample_rate
        self.turns: list[Turn] = []
        self.events: list[ReportEvent] = []
        self.usage = Usage()
        self.close_reason: str | None = None
        self.start()

    def start(self) -> None:
        """Start the timeline of the session now, and discard what the recorder collected."""
        self._start = self.clock()
        self._started_at = self.wall_clock()
        self._ended: float | None = None
        self._raw = AudioTrack(self.input_sample_rate)
        self._filtered = AudioTrack(self.input_sample_rate)
        self._agent = AudioTrack(self.agent_sample_rate)
        self._replies: dict[object, Turn] = {}
        self._tools: dict[str, ToolCall] = {}
        self._arguments: dict[str, str] = {}  # The streamed argument text of each tool call
        self._chunk_ids: dict[object, str] = {}  # Later chunks of a tool call give only its index
        self._speech_started_at: float | None = None
        self._speech_stopped_at: float | None = None
        self._transcript_at: float | None = None

    def at(self, now: float | None = None) -> float:
        """Return ``now`` (default: the clock) in seconds from the start of the session."""
        return (self.clock() if now is None else now) - self._start

    def input_audio(self, raw: bytes, filtered: bytes) -> None:
        """Record a chunk of input audio before and after the audio filters."""
        self.usage.stt_audio_seconds += len(filtered) / (2 * self.input_sample_rate)
        at = self.at()
        self._raw.append(at, raw)
        if filtered:
            self._filtered.append(at, filtered)

    def stt_event(self, event: object) -> None:
        """Record an event of the STT provider. The end of the session comes from :meth:`closed`."""
        at = self.at()
        match event:
            case SpeechStarted():
                self.events.append(UserSpeechStarted(at=at))
                if self._speech_started_at is None:
                    self._speech_started_at = at
            case SpeechStopped():
                self.events.append(UserSpeechStopped(at=at))
                self._speech_stopped_at = at
            case PartialTranscript(text=text):
                self.events.append(PartialTranscriptReceived(at=at, text=text))
            case FinalTranscript(text=text):
                self.events.append(FinalTranscriptReceived(at=at, text=text))
                self._transcript_at = at

    def transcript_ignored(self, text: str) -> None:
        """Record a transcript that the pipeline discarded. Its speech does not count for the next turn."""
        self.events.append(TranscriptIgnored(at=self.at(), text=text))
        self._speech_started_at = self._speech_stopped_at = self._transcript_at = None

    def turn_committed(self, reply: object, text: str, kind: TurnKind) -> None:
        """Start a turn for ``reply``, the reply of the pipeline to ``text``."""
        at = self.at()
        turn = Turn(kind=kind, user_text=text if kind != TurnKind.GREETING else "", committed_at=at)
        if kind == TurnKind.USER:
            turn.user_started_at, turn.user_stopped_at = self._speech_started_at, self._speech_stopped_at
            turn.transcript_at = self._transcript_at
            self.events.append(TurnCommitted(at=at, text=text))
        self._speech_started_at = self._speech_stopped_at = self._transcript_at = None
        self.turns.append(turn)
        self._replies[reply] = turn

    def agent_started(self, reply: object) -> None:
        """The agent started to work on ``reply``."""
        if turn := self._replies.get(reply):
            turn.agent_started_at = self.at()

    def agent_text(self, reply: object) -> None:
        """The agent streamed text for ``reply``. Only the first text counts."""
        if (turn := self._replies.get(reply)) and turn.first_token_at is None:
            turn.first_token_at = self.at()

    def agent_message(self, reply: object, message: BaseMessage) -> None:
        """Record the tool calls, their results and the token usage in a message that the agent streamed."""
        turn = self._replies.get(reply)
        if turn is None:
            return
        at = self.at()
        if isinstance(message, AIMessageChunk) and message.tool_call_chunks:
            for chunk in message.tool_call_chunks:
                if call_id := chunk.get("id"):
                    self._chunk_ids[chunk.get("index")] = call_id
                    self._start_tool(turn, call_id, chunk.get("name") or "", at)
                call_id = call_id or self._chunk_ids.get(chunk.get("index"))
                if call_id in self._tools and chunk.get("args"):
                    self._arguments[call_id] = text = self._arguments.get(call_id, "") + chunk["args"]
                    try:
                        self._tools[call_id].arguments = json.loads(text)
                    except ValueError:
                        self._tools[call_id].arguments = text
        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call.get("id"):
                    self._start_tool(turn, call["id"], call.get("name") or "", at)
                    self._tools[call["id"]].arguments = call.get("args")
        if isinstance(message, AIMessage):
            if usage := message.usage_metadata:
                turn.input_tokens += usage.get("input_tokens", 0)
                turn.output_tokens += usage.get("output_tokens", 0)
                self.usage.llm_input_tokens += usage.get("input_tokens", 0)
                self.usage.llm_output_tokens += usage.get("output_tokens", 0)
        elif isinstance(message, ToolMessage):
            tool = self._tools.get(message.tool_call_id)
            if tool is not None and tool.ended_at is None:
                tool.ended_at, tool.result, tool.error = at, message.text, message.status == "error"
                tool.name = tool.name or message.name or ""
                self.events.append(ToolFinished(at=at, name=tool.name, call_id=tool.call_id, error=tool.error))

    def _start_tool(self, turn: Turn, call_id: str, name: str, at: float) -> None:
        if call_id not in self._tools:
            self._tools[call_id] = tool = ToolCall(name=name, call_id=call_id, started_at=at)
            turn.tool_calls.append(tool)
            self.events.append(ToolStarted(at=at, name=name, call_id=call_id))

    def error(self, source: ErrorSource, error: BaseException, reply: object | None = None) -> None:
        """Record an error that the pipeline survived. An error of the agent also goes to the turn of ``reply``."""
        message = f"{type(error).__name__}: {error}"
        self.events.append(ErrorOccurred(at=self.at(), source=source, message=message))
        if turn := self._replies.get(reply):
            turn.error = message

    def sentence(self, reply: object, text: str) -> None:
        """The pipeline sent a sentence of ``reply`` to TTS."""
        self.usage.tts_characters += len(text)
        if turn := self._replies.get(reply):
            if turn.first_sentence_at is None:
                turn.first_sentence_at = self.at()
            turn.reply_text = f"{turn.reply_text} {text}".strip()

    def tts_audio(self, reply: object) -> None:
        """TTS sent audio for ``reply``. Only the first audio counts."""
        if (turn := self._replies.get(reply)) and turn.first_audio_at is None:
            turn.first_audio_at = self.at()

    def played(self, reply: object, start: float, end: float, pcm: bytes) -> None:
        """Audio of ``reply`` plays from ``start`` to ``end`` (times on the clock)."""
        at = self.at(start)
        if turn := self._replies.get(reply):
            if turn.playback_started_at is None:
                turn.playback_started_at = at
                self.events.append(AgentStartedSpeaking(at=at))
            turn.playback_ended_at = max(turn.playback_ended_at or 0.0, self.at(end))
        self._agent.write(at, pcm)

    def _stop_playback(self, at: float) -> None:
        for turn in self.turns:
            if turn.playback_ended_at is not None and turn.playback_ended_at > at:
                turn.playback_ended_at = max(at, turn.playback_started_at or at)

    def barge_in(self, now: float, heard: object | None, heard_text: str, cancelled: list) -> None:
        """The user interrupted. ``heard`` is the reply that the user heard in part, and ``cancelled`` the others."""
        at = self.at(now)
        self.events.append(BargeIn(at=at, heard_text=heard_text))
        if (turn := self._replies.get(heard)) and heard_text != turn.reply_text:
            turn.interrupted, turn.heard_text = True, heard_text
        for reply in cancelled:
            if (turn := self._replies.get(reply)) and reply is not heard:
                turn.interrupted, turn.heard_text = True, ""
        self._stop_playback(at)
        self._agent.cut(at)

    def closed(self, reason: str) -> None:
        """The session ended because of ``reason``. Only the first reason counts."""
        if self._ended is None:
            self._ended = self.at()
            self.close_reason = reason
            self.events.append(SessionEnded(at=self._ended, reason=reason))
            self._stop_playback(self._ended)

    def report(self) -> SessionReport:
        """Return a copy of the report of the session up to now.

        The recorder adds an ``AgentStoppedSpeaking`` event for each reply whose audio stopped before now.
        """
        now = self._ended if self._ended is not None else self.at()
        ends = [turn.playback_ended_at for turn in self.turns if turn.playback_ended_at is not None]
        stops = [AgentStoppedSpeaking(at=at) for at in ends if at <= now]
        # At the same time: the cause (a transcript, a barge-in), then the stop, then the next reply and the end
        order = {AgentStoppedSpeaking: 1, AgentStartedSpeaking: 2, SessionEnded: 3}
        events = sorted([*self.events, *stops], key=lambda event: (event.at, order.get(type(event), 0)))
        audio = AudioRecording(
            raw=bytes(self._raw.data),
            filtered=bytes(self._filtered.data),
            agent=bytes(self._agent.data),
            input_sample_rate=self.input_sample_rate,
            agent_sample_rate=self.agent_sample_rate,
        )
        return SessionReport(
            started_at=self._started_at,
            duration=now,
            close_reason=self.close_reason,
            thread_id=self.thread_id,
            providers=self.providers,
            settings=self.settings,
            turns=copy.deepcopy(self.turns),
            events=events,
            usage=dataclasses.replace(self.usage),
            audio=audio,
        )
