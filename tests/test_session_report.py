"""SessionReport: the latency stages of a turn add up to its latency, and the report exports to JSON, text and WAV."""

import base64
import io
import json
import wave
from dataclasses import asdict

import pytest

from voxchain.report import (
    AudioRecording,
    BargeIn,
    ErrorOccurred,
    ErrorSource,
    LatencyStats,
    SessionReport,
    ToolCall,
    TranscriptIgnored,
    Turn,
    TurnCommitted,
    TurnKind,
    Usage,
    UserSpeechStarted,
    UserSpeechStopped,
)


def turn(**kwargs):
    times = {
        "user_stopped_at": 10.0,
        "transcript_at": 10.2,
        "committed_at": 10.5,
        "agent_started_at": 10.5,
        "first_token_at": 11.5,
        "first_sentence_at": 11.6,
        "first_audio_at": 11.9,
        "playback_started_at": 12.0,
    }
    return Turn(kind=TurnKind.USER, user_text="Book a table", **{**times, **kwargs})


def audio(**kwargs):
    values = {"raw": bytes(32000), "filtered": bytes(32000), "agent": bytes(24000), "agent_sample_rate": 24000}
    return AudioRecording(input_sample_rate=16000, **{**values, **kwargs})


def report(*turns, audio=None, events=None):
    audio = audio or AudioRecording(raw=b"", filtered=b"", agent=b"", input_sample_rate=16000, agent_sample_rate=24000)
    return SessionReport(
        started_at=0.0,
        duration=20.0,
        close_reason="Audio stream ended",
        thread_id="call-7",
        providers={"stt": "FakeSTT"},
        settings={"interrupt_min_words": 3},
        turns=list(turns),
        events=events if events is not None else [TurnCommitted(at=10.5, text="Book a table")],
        usage=Usage(tts_characters=12),
        audio=audio,
    )


def test_the_stages_of_a_turn_follow_one_another_and_add_up_to_the_latency():
    latency = turn().latency
    assert asdict(latency) == pytest.approx(
        {
            "transcription": 0.2,
            "end_of_turn": 0.3,
            "waiting": 0.0,
            "tools": None,
            "llm": 1.0,
            "sentence": 0.1,
            "tts": 0.3,
            "playback": 0.1,
        }
    )
    assert sum(seconds for _name, seconds in latency.stages()) == pytest.approx(turn().e2e_latency)


def test_tool_calls_before_the_first_text_are_a_stage_of_their_own():
    latency = turn(tool_calls=[ToolCall(name="book", call_id="1", started_at=10.7, ended_at=11.3)]).latency
    assert latency.tools == pytest.approx(0.6)
    assert latency.llm == pytest.approx(0.4)


def test_a_transcript_before_the_end_of_speech_counts_from_the_transcript():
    late_stop = turn(user_stopped_at=10.4)
    assert late_stop.start == 10.2
    assert sum(seconds for _name, seconds in late_stop.latency.stages()) == pytest.approx(late_stop.e2e_latency)


def test_a_turn_without_audio_has_stages_until_its_last_step_and_no_latency():
    cancelled = turn(first_audio_at=None, playback_started_at=None)
    assert cancelled.latency.stages()[-1][0] == "sentence"
    assert cancelled.latency.tts is None
    assert cancelled.e2e_latency is None


def test_a_greeting_has_no_latency():
    greeting = Turn(kind=TurnKind.GREETING, user_text="", committed_at=0.0, playback_started_at=0.4)
    assert greeting.latency.stages() == []
    assert greeting.e2e_latency is None


def test_the_json_has_the_latency_and_events_with_their_type_and_times_in_milliseconds():
    data = json.loads(json.dumps(report(turn(user_stopped_at=10.000000001)).to_dict()))
    assert data["turns"][0]["user_stopped_at"] == 10.0
    assert data["turns"][0]["latency"]["llm"] == 1.0
    assert data["turns"][0]["e2e_latency"] == 2.0
    assert data["events"][0] == {"type": "TurnCommitted", "at": 10.5, "text": "Book a table"}


def test_the_dict_gives_each_audio_track_as_a_base64_wav_file_with_its_length():
    data = json.loads(json.dumps(report(turn(), audio=audio(agent=b"\1\0" * 12000)).to_dict()))["audio"]
    assert {key: data[key] for key in ("input_sample_rate", "agent_sample_rate", "raw_seconds", "agent_seconds")} == {
        "input_sample_rate": 16000,
        "agent_sample_rate": 24000,
        "raw_seconds": 1.0,
        "agent_seconds": 0.5,
    }
    with wave.open(io.BytesIO(base64.b64decode(data["agent"]))) as file:
        assert (file.getframerate(), file.getnframes()) == (24000, 12000)
        assert file.readframes(2) == b"\1\0\1\0"
    with wave.open(io.BytesIO(base64.b64decode(data["filtered"]))) as file:
        assert (file.getframerate(), file.getnframes()) == (16000, 16000)


def test_the_transcript_shows_what_the_user_heard_of_an_interrupted_reply():
    text = report(turn(reply_text="Sure. For how many?", interrupted=True, heard_text="Sure.")).transcript()
    assert text.splitlines() == [
        "[  10.00s] User: Book a table",
        "[  12.00s] Agent: Sure. For how many?  (interrupted, heard: 'Sure.')",
    ]


def test_the_transcript_marks_a_reply_that_never_played():
    text = report(turn(reply_text="Sure.", first_audio_at=None, playback_started_at=None)).transcript()
    assert text.splitlines()[-1] == "[  10.50s] Agent: Sure.  (not played)"


def test_the_waterfall_starts_each_bar_where_the_bar_before_it_ends():
    lines = report(turn()).waterfall().splitlines()
    assert lines[0] == "Turn 1: 'Book a table'  e2e 2.00s"
    transcription, end_of_turn = lines[1], lines[2]
    assert transcription.endswith(" " + "█" * 4)
    assert end_of_turn.endswith(" " * 4 + "█" * 6)


def test_the_stereo_wav_has_the_user_on_the_left_and_the_agent_on_the_right():
    user = (1000).to_bytes(2, "little", signed=True) * 4
    agent = (-1000).to_bytes(2, "little", signed=True) * 2
    recording = audio(raw=user, filtered=user, agent=agent, agent_sample_rate=16000)
    with wave.open(io.BytesIO(recording.stereo_wav())) as file:
        assert (file.getnchannels(), file.getframerate(), file.getnframes()) == (2, 16000, 4)
        frames = file.readframes(4)
    samples = [int.from_bytes(frames[i : i + 2], "little", signed=True) for i in range(0, len(frames), 2)]
    assert samples == [1000, -1000, 1000, -1000, 1000, 0, 1000, 0]


def test_the_reply_seconds_are_the_time_that_its_audio_played():
    assert turn(playback_ended_at=14.5).reply_seconds == pytest.approx(2.5)
    assert turn().reply_seconds is None


def test_latency_statistics_use_the_nearest_rank():
    stats = LatencyStats.of([1.0, 2.0, 3.0, 4.0, 10.0])
    assert (stats.count, stats.mean, stats.p50, stats.p95, stats.max) == (5, 4.0, 3.0, 10.0, 10.0)
    assert LatencyStats.of([]) is None


def test_the_summary_counts_the_session_and_gives_latency_statistics_over_the_user_turns():
    greeting = Turn(
        kind=TurnKind.GREETING, user_text="", committed_at=0.0, playback_started_at=0.0, playback_ended_at=3.0
    )
    fast = turn(playback_ended_at=13.0, tool_calls=[ToolCall(name="book", call_id="1", started_at=10.7, ended_at=11.3)])
    slow = turn(playback_started_at=13.0, playback_ended_at=18.0)
    events = [
        UserSpeechStarted(at=8.0),
        UserSpeechStopped(at=10.0),
        TranscriptIgnored(at=12.0, text="Okay."),
        BargeIn(at=12.5, heard_text="Sure."),
        ErrorOccurred(at=13.0, source=ErrorSource.MEMORY, message="ValueError: x"),
        UserSpeechStarted(at=14.0),
    ]
    summary = report(greeting, fast, slow, events=events).summary
    assert (summary.user_turns, summary.barge_ins, summary.ignored_transcripts) == (2, 1, 1)
    assert (summary.tool_calls, summary.errors) == (1, 1)
    assert summary.user_speech_seconds == 2.0  # The speech that did not stop does not count
    assert summary.agent_speech_seconds == pytest.approx(3.0 + 1.0 + 5.0)
    assert summary.longest_reply_seconds == pytest.approx(5.0)
    assert summary.latency["e2e"].count == 2
    assert summary.latency["e2e"].max == pytest.approx(3.0)
    assert summary.latency["tools"].count == 1


def test_the_dict_has_the_thread_id_the_settings_and_the_summary():
    data = report(turn()).to_dict()
    assert data["thread_id"] == "call-7"
    assert data["settings"] == {"interrupt_min_words": 3}
    assert data["summary"]["user_turns"] == 1
    assert data["summary"]["latency"]["e2e"]["mean"] == 2.0
