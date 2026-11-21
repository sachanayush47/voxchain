"""The report of one voice session: the transcript, a timeline of events, the latency of each turn, usage and audio."""

import base64
import io
import math
import wave
from array import array
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum

from voxchain.audio import PCM16Resampler

WATERFALL_SECONDS_PER_CHAR = 0.05
"""The time that one character of a bar in :meth:`SessionReport.waterfall` shows."""


class TurnKind(StrEnum):
    """What started a turn."""

    USER = "user"
    GREETING = "greeting"
    RECOVERY = "recovery"


class ErrorSource(StrEnum):
    """The part of the pipeline where an error occurred."""

    AGENT = "agent"
    MEMORY = "memory"
    END_OF_TURN = "end_of_turn"


@dataclass(frozen=True)
class UserSpeechStarted:
    """The STT provider reported the start of user speech."""

    at: float


@dataclass(frozen=True)
class UserSpeechStopped:
    """The STT provider reported the end of user speech."""

    at: float


@dataclass(frozen=True)
class PartialTranscriptReceived:
    """The STT provider sent an interim transcript."""

    at: float
    text: str


@dataclass(frozen=True)
class FinalTranscriptReceived:
    """The STT provider sent a final transcript."""

    at: float
    text: str


@dataclass(frozen=True)
class TranscriptIgnored:
    """The pipeline discarded a transcript: it was too short to interrupt the agent, or came during the greeting."""

    at: float
    text: str


@dataclass(frozen=True)
class TurnCommitted:
    """The pipeline sent a user turn to the agent."""

    at: float
    text: str


@dataclass(frozen=True)
class ToolStarted:
    """The agent called a tool."""

    at: float
    name: str
    call_id: str


@dataclass(frozen=True)
class ToolFinished:
    """A tool gave its result to the agent. ``error`` is ``True`` if the tool failed."""

    at: float
    name: str
    call_id: str
    error: bool


@dataclass(frozen=True)
class AgentStartedSpeaking:
    """The first audio of a reply started to play."""

    at: float


@dataclass(frozen=True)
class AgentStoppedSpeaking:
    """The audio of a reply stopped: it played to the end, or a barge-in cut it."""

    at: float


@dataclass(frozen=True)
class ErrorOccurred:
    """A part of the pipeline failed. The pipeline continued."""

    at: float
    source: ErrorSource
    message: str


@dataclass(frozen=True)
class BargeIn:
    """The user interrupted the agent. ``heard_text`` is the text of the reply that the user heard."""

    at: float
    heard_text: str


@dataclass(frozen=True)
class SessionEnded:
    """The session ended because of ``reason``."""

    at: float
    reason: str


ReportEvent = (
    UserSpeechStarted
    | UserSpeechStopped
    | PartialTranscriptReceived
    | FinalTranscriptReceived
    | TranscriptIgnored
    | TurnCommitted
    | ToolStarted
    | ToolFinished
    | AgentStartedSpeaking
    | AgentStoppedSpeaking
    | BargeIn
    | ErrorOccurred
    | SessionEnded
)
"""One event of the timeline. ``at`` is its time in seconds from the start of the session."""


def rounded(value: object) -> object:
    """Round each float in ``value`` (also inside lists and dicts) to milliseconds."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


@dataclass
class ToolCall:
    """One tool call of the agent. The times are seconds from the start of the session.

    ``arguments`` are the arguments from the agent: a dict, or the text if it is not
    valid JSON. ``result`` is the text that the tool gave back, and ``error`` is
    ``True`` if the tool failed.
    """

    name: str
    call_id: str
    started_at: float
    ended_at: float | None = None
    arguments: dict | str | None = None
    result: str | None = None
    error: bool = False


@dataclass
class Latency:
    """The latency of a turn in stages. Each stage starts where the stage before it ends.

    A stage is ``None`` if the turn did not get to it, and ``tools`` is ``None``
    if the agent called no tool before its first text.

    - ``transcription``: the end of speech until the final transcript.
    - ``end_of_turn``: the wait of the end-of-turn detector.
    - ``waiting``: the wait for the previous reply or a memory correction.
    - ``tools``: the tool calls before the first text of the agent.
    - ``llm``: the rest of the time until the first text of the agent.
    - ``sentence``: the wait until the first sentence is complete.
    - ``tts``: the first sentence until the first TTS audio.
    - ``playback``: the first TTS audio until it starts to play.
    """

    transcription: float | None = None
    end_of_turn: float | None = None
    waiting: float | None = None
    tools: float | None = None
    llm: float | None = None
    sentence: float | None = None
    tts: float | None = None
    playback: float | None = None

    def stages(self) -> list[tuple[str, float]]:
        """Return the stages that occurred, in order, as ``(name, seconds)``."""
        stages = [(stage.name, getattr(self, stage.name)) for stage in fields(self)]
        return [(name, seconds) for name, seconds in stages if seconds is not None]


@dataclass
class Turn:
    """One user turn and the reply of the agent.

    The times are seconds from the start of the session, or ``None`` if the
    step did not occur. ``heard_text`` is the part of the reply that the user
    heard before a barge-in, and is ``None`` if the user heard all of it.
    ``error`` is the error of the agent if the reply failed.
    """

    kind: TurnKind
    user_text: str
    committed_at: float
    user_started_at: float | None = None
    user_stopped_at: float | None = None
    transcript_at: float | None = None
    agent_started_at: float | None = None
    first_token_at: float | None = None
    first_sentence_at: float | None = None
    first_audio_at: float | None = None
    playback_started_at: float | None = None
    playback_ended_at: float | None = None
    reply_text: str = ""
    interrupted: bool = False
    heard_text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def start(self) -> float:
        """The time from which the latency counts: the end of user speech if it is known, otherwise the commit."""
        known = [at for at in (self.user_stopped_at, self.transcript_at) if at is not None]
        return min(known) if known else self.committed_at

    @property
    def e2e_latency(self) -> float | None:
        """Seconds from the end of user speech to the start of the reply audio. A greeting has no latency."""
        if self.kind == TurnKind.GREETING or self.playback_started_at is None:
            return None
        return self.playback_started_at - self.start

    @property
    def latency(self) -> Latency:
        """The latency in stages. The stages add up to :attr:`e2e_latency`."""
        latency = Latency()
        if self.kind == TurnKind.GREETING:
            return latency
        points = [
            ("transcription", self.transcript_at),
            ("end_of_turn", self.committed_at),
            ("waiting", self.agent_started_at),
            ("llm", self.first_token_at),
            ("sentence", self.first_sentence_at),
            ("tts", self.first_audio_at),
            ("playback", self.playback_started_at),
        ]
        last = self.start
        for name, at in points:
            if at is None:
                break
            seconds = max(0.0, at - last)
            if name == "llm":
                tools = sum(
                    min(call.ended_at if call.ended_at is not None else at, at) - max(call.started_at, last)
                    for call in self.tool_calls
                    if call.started_at < at
                )
                if tools := min(max(0.0, tools), seconds):
                    latency.tools = tools
                seconds -= tools
            setattr(latency, name, seconds)
            last = max(last, at)
        return latency

    @property
    def reply_seconds(self) -> float | None:
        """How long the audio of the reply played, or ``None`` if it did not play."""
        if self.playback_started_at is None or self.playback_ended_at is None:
            return None
        return self.playback_ended_at - self.playback_started_at

    def to_dict(self) -> dict:
        """Return the turn as a dict, with its latency stages and its end-to-end latency."""
        return {**asdict(self), "latency": asdict(self.latency), "e2e_latency": self.e2e_latency}


@dataclass(frozen=True)
class LatencyStats:
    """Statistics of one latency over the user turns, in seconds."""

    count: int
    mean: float
    p50: float
    p95: float
    max: float

    @classmethod
    def of(cls, values: list[float]) -> "LatencyStats | None":
        """Return the statistics of ``values``, or ``None`` if there are no values. Percentiles use the nearest rank."""
        if not values:
            return None
        ordered = sorted(values)

        def rank(quantile: float) -> float:
            return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]

        return cls(len(ordered), sum(ordered) / len(ordered), rank(0.5), rank(0.95), ordered[-1])


@dataclass(frozen=True)
class Summary:
    """The main numbers of a session.

    ``latency`` has the statistics of ``e2e`` and of each latency stage over the
    user turns. Speech times add up the speech of the user (from the voice
    activity of the STT provider) and the reply audio that played.
    """

    user_turns: int
    barge_ins: int
    ignored_transcripts: int
    tool_calls: int
    errors: int
    user_speech_seconds: float
    agent_speech_seconds: float
    longest_reply_seconds: float
    latency: dict[str, LatencyStats]


@dataclass
class Usage:
    """What the session used of each provider."""

    stt_audio_seconds: float = 0.0
    tts_characters: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


def wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Return PCM16 audio as the bytes of a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm)
    return buffer.getvalue()


def base64_wav(pcm: bytes, sample_rate: int) -> str:
    """Return PCM16 mono audio as a WAV file in base64."""
    return base64.b64encode(wav(pcm, sample_rate)).decode("ascii")


@dataclass(frozen=True)
class AudioRecording:
    """The audio of the session as PCM16 mono, on one timeline from the start of the session.

    ``raw`` is the input before the audio filters, and ``filtered`` is the input
    that STT heard. Both are at ``input_sample_rate``. ``agent`` is the reply audio
    that played, cut at each barge-in, at ``agent_sample_rate``.
    """

    raw: bytes = field(repr=False)
    filtered: bytes = field(repr=False)
    agent: bytes = field(repr=False)
    input_sample_rate: int
    agent_sample_rate: int

    @property
    def raw_seconds(self) -> float:
        """The length of the raw input track in seconds."""
        return len(self.raw) / (2 * self.input_sample_rate)

    @property
    def filtered_seconds(self) -> float:
        """The length of the filtered input track in seconds."""
        return len(self.filtered) / (2 * self.input_sample_rate)

    @property
    def agent_seconds(self) -> float:
        """The length of the agent track in seconds."""
        return len(self.agent) / (2 * self.agent_sample_rate)

    def to_dict(self) -> dict:
        """Return the sample rates, the length of each track, and each track as a base64 WAV file."""
        return {
            "input_sample_rate": self.input_sample_rate,
            "agent_sample_rate": self.agent_sample_rate,
            "raw_seconds": self.raw_seconds,
            "filtered_seconds": self.filtered_seconds,
            "agent_seconds": self.agent_seconds,
            "raw": base64_wav(self.raw, self.input_sample_rate),
            "filtered": base64_wav(self.filtered, self.input_sample_rate),
            "agent": base64_wav(self.agent, self.agent_sample_rate),
        }

    def stereo_wav(self) -> bytes:
        """Return a stereo WAV file: the filtered input on the left channel and the agent on the right."""
        agent = PCM16Resampler(self.agent_sample_rate, self.input_sample_rate).process(self.agent)
        user = self.filtered
        size = max(len(user), len(agent))
        frames = array("h", bytes(size * 2))
        frames[0::2] = array("h", user.ljust(size, b"\0"))
        frames[1::2] = array("h", agent.ljust(size, b"\0"))
        return wav(frames.tobytes(), self.input_sample_rate, channels=2)


@dataclass
class SessionReport:
    """Everything about one session: the turns, the events, the usage and the audio.

    ``started_at`` is a Unix timestamp. All other times are seconds from the
    start of the session. ``close_reason`` is ``None`` while the session runs.
    """

    started_at: float
    duration: float
    close_reason: str | None
    thread_id: str
    providers: dict[str, str]
    settings: dict[str, object]
    turns: list[Turn]
    events: list[ReportEvent]
    usage: Usage
    audio: AudioRecording

    def to_dict(self) -> dict:
        """Return the report as a dict for ``json.dumps``, with times rounded to milliseconds.

        Each event has its class name as ``type``. ``audio`` gives the sample rates,
        the length of each track, and each track as a WAV file in base64. A 10-minute
        call gives approximately 90 MB of base64 audio.
        """
        return rounded(
            {
                "started_at": self.started_at,
                "duration": self.duration,
                "close_reason": self.close_reason,
                "thread_id": self.thread_id,
                "providers": self.providers,
                "settings": self.settings,
                "summary": asdict(self.summary),
                "turns": [turn.to_dict() for turn in self.turns],
                "events": [{"type": type(event).__name__, **asdict(event)} for event in self.events],
                "usage": asdict(self.usage),
                "audio": self.audio.to_dict(),
            }
        )

    @property
    def summary(self) -> Summary:
        """Return the main numbers of the session, calculated from its turns and events."""
        user_turns = [turn for turn in self.turns if turn.kind == TurnKind.USER]
        samples: dict[str, list[float]] = {"e2e": [t.e2e_latency for t in user_turns if t.e2e_latency is not None]}
        for turn in user_turns:
            for name, seconds in turn.latency.stages():
                samples.setdefault(name, []).append(seconds)
        user_speech, speaking_since = 0.0, None
        for event in self.events:
            if isinstance(event, UserSpeechStarted) and speaking_since is None:
                speaking_since = event.at
            elif isinstance(event, UserSpeechStopped) and speaking_since is not None:
                user_speech, speaking_since = user_speech + event.at - speaking_since, None
        replies = [turn.reply_seconds for turn in self.turns if turn.reply_seconds is not None]

        def count(kind: type) -> int:
            return sum(isinstance(event, kind) for event in self.events)

        return Summary(
            user_turns=len(user_turns),
            barge_ins=count(BargeIn),
            ignored_transcripts=count(TranscriptIgnored),
            tool_calls=sum(len(turn.tool_calls) for turn in self.turns),
            errors=count(ErrorOccurred),
            user_speech_seconds=user_speech,
            agent_speech_seconds=sum(replies),
            longest_reply_seconds=max(replies, default=0.0),
            latency={name: stats for name, values in samples.items() if (stats := LatencyStats.of(values))},
        )

    def transcript(self) -> str:
        """Return the conversation as text, one line for each message, with the time of each.

        A reply shows the text that the agent generated, with a note if the user did not hear all of it.
        """
        lines = []
        for turn in self.turns:
            if turn.kind == TurnKind.USER:
                lines.append(f"[{turn.start:7.2f}s] User: {turn.user_text}")
            if turn.reply_text:
                at = turn.playback_started_at if turn.playback_started_at is not None else turn.committed_at
                note = ""
                if turn.playback_started_at is None:
                    note = "  (not played)"
                elif turn.interrupted:
                    note = f"  (interrupted, heard: {turn.heard_text!r})"
                lines.append(f"[{at:7.2f}s] Agent: {turn.reply_text}{note}")
        return "\n".join(lines)

    def waterfall(self) -> str:
        """Return the latency stages of each turn as text bars, each one after the stage before it."""
        lines = []
        for number, turn in enumerate(self.turns, 1):
            stages = turn.latency.stages()
            if not stages:
                continue
            total = f"{turn.e2e_latency:.2f}s" if turn.e2e_latency is not None else "no audio"
            lines.append(f"Turn {number}: {turn.user_text[:60]!r}  e2e {total}")
            offset = 0
            for name, seconds in stages:
                width = round(seconds / WATERFALL_SECONDS_PER_CHAR)
                bar = " " * offset + "█" * max(1, width) if seconds else ""
                lines.append(f"  {name:<13} {seconds:6.3f}s {bar}")
                offset += width
        return "\n".join(lines)
