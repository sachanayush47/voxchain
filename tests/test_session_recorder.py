"""SessionRecorder: it collects turns, tool calls, usage, barge-ins and audio from the calls of a pipeline."""

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from voxchain.report import ErrorSource, SessionRecorder, TurnKind
from voxchain.report.recorder import AudioTrack
from voxchain.stt import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def recorder(**kwargs):
    clock = Clock()
    return clock, SessionRecorder(clock=clock, input_sample_rate=16000, agent_sample_rate=16000, **kwargs)


def test_times_count_from_the_start_of_the_session():
    clock, rec = recorder()
    clock.now = 101.0
    rec.stt_event(SpeechStopped())
    rec.stt_event(FinalTranscript("hi"))
    clock.now = 101.5
    rec.turn_committed("reply", "hi", TurnKind.USER)
    turn = rec.report().turns[0]
    assert (turn.user_stopped_at, turn.transcript_at, turn.committed_at) == (1.0, 1.0, 1.5)


def test_the_end_of_speech_of_a_previous_turn_does_not_count_for_the_next_turn():
    _clock, rec = recorder()
    rec.stt_event(SpeechStopped())
    rec.turn_committed("first", "hi", TurnKind.USER)
    rec.turn_committed("second", "(Ask again)", TurnKind.RECOVERY)
    assert rec.report().turns[1].user_stopped_at is None


def test_tool_calls_and_token_usage_come_from_the_agent_messages():
    clock, rec = recorder()
    rec.turn_committed("reply", "book it", TurnKind.USER)
    clock.now = 101.0
    rec.agent_message("reply", AIMessage(content="", tool_calls=[{"name": "book", "args": {}, "id": "c1"}]))
    clock.now = 102.0
    rec.agent_message("reply", ToolMessage(content="Booked", tool_call_id="c1"))
    usage = {"input_tokens": 50, "output_tokens": 7, "total_tokens": 57}
    rec.agent_message("reply", AIMessage(content="Done.", usage_metadata=usage))
    report = rec.report()
    call = report.turns[0].tool_calls[0]
    assert (call.name, call.started_at, call.ended_at) == ("book", 1.0, 2.0)
    assert (report.usage.llm_input_tokens, report.usage.llm_output_tokens) == (50, 7)
    assert [type(event).__name__ for event in report.events] == ["TurnCommitted", "ToolStarted", "ToolFinished"]


def test_a_reply_that_the_user_heard_completely_does_not_count_as_interrupted():
    _clock, rec = recorder()
    rec.turn_committed("old", "hi", TurnKind.USER)
    rec.sentence("old", "Hello.")
    rec.turn_committed("new", "and", TurnKind.USER)
    rec.barge_in(100.0, heard="old", heard_text="Hello.", cancelled=["new"])
    old, new = rec.report().turns
    assert not old.interrupted
    assert (new.interrupted, new.heard_text) == (True, "")


def test_only_the_first_close_reason_counts():
    clock, rec = recorder()
    clock.now = 105.0
    rec.closed("STT: Lost")
    clock.now = 106.0
    rec.closed("Stopped")
    report = rec.report()
    assert (report.close_reason, report.duration) == ("STT: Lost", 5.0)
    assert [(type(event).__name__, event.reason) for event in report.events] == [("SessionEnded", "STT: Lost")]


def test_a_report_does_not_change_after_the_recorder_collects_more():
    _clock, rec = recorder()
    rec.turn_committed("reply", "hi", TurnKind.USER)
    report = rec.report()
    rec.sentence("reply", "Hello.")
    assert report.turns[0].reply_text == ""
    assert report.usage.tts_characters == 0


def test_the_seconds_of_audio_sent_to_stt_count_the_filtered_audio():
    _clock, rec = recorder()
    rec.input_audio(b"\1\0" * 10, b"\1\0" * 5)
    assert rec.report().usage.stt_audio_seconds == 5 / 16000


def test_the_input_track_starts_at_the_time_of_its_first_audio():
    clock, rec = recorder()
    clock.now = 100.5
    rec.input_audio(b"\1\0" * 4, b"")
    clock.now = 103.0  # A late chunk follows the audio before it, not the clock
    rec.input_audio(b"\1\0" * 4, b"\1\0" * 4)
    audio = rec.report().audio
    assert len(audio.raw) == 8000 * 2 + 16
    assert len(audio.filtered) == 48000 * 2 + 8


def test_a_track_fills_gaps_with_silence_and_a_cut_discards_the_audio_that_did_not_play():
    track = AudioTrack(sample_rate=10)
    track.write(0.5, b"\1\0" * 10)
    assert track.data == bytearray(10) + b"\1\0" * 10
    track.cut(1.0)
    assert len(track.data) == 20


def test_stt_events_become_typed_events_of_the_timeline():
    clock, rec = recorder()
    clock.now = 101.0
    rec.stt_event(SpeechStarted())
    rec.stt_event(PartialTranscript("hel"))
    rec.stt_event(SpeechStopped())
    rec.stt_event(FinalTranscript("hello"))
    events = rec.report().events
    assert [type(event).__name__ for event in events] == [
        "UserSpeechStarted",
        "PartialTranscriptReceived",
        "UserSpeechStopped",
        "FinalTranscriptReceived",
    ]
    assert (events[3].at, events[3].text) == (1.0, "hello")


def test_a_turn_starts_with_the_first_speech_after_the_previous_turn_and_ignored_speech_does_not_count():
    clock, rec = recorder()
    clock.now = 101.0
    rec.stt_event(SpeechStarted())
    rec.transcript_ignored("Okay.")
    clock.now = 102.0
    rec.stt_event(SpeechStarted())
    clock.now = 102.5
    rec.stt_event(SpeechStarted())  # A second segment of the same turn
    rec.turn_committed("reply", "book it", TurnKind.USER)
    assert rec.report().turns[0].user_started_at == 2.0


def test_a_reply_stops_at_the_end_of_its_audio_or_at_a_barge_in():
    clock, rec = recorder()
    rec.turn_committed("a", "hi", TurnKind.USER)
    rec.played("a", 101.0, 102.0, b"")
    rec.played("a", 102.0, 103.0, b"")
    rec.turn_committed("b", "and", TurnKind.USER)
    rec.played("b", 104.0, 108.0, b"")
    clock.now = 105.0
    assert [turn.playback_ended_at for turn in rec.report().turns] == [3.0, 8.0]
    stops = [event.at for event in rec.report().events if type(event).__name__ == "AgentStoppedSpeaking"]
    assert stops == [3.0]  # The second reply plays until 8 s, so it did not stop yet
    clock.now = 105.5
    rec.barge_in(105.5, heard="b", heard_text="", cancelled=[])
    report = rec.report()
    assert [turn.playback_ended_at for turn in report.turns] == [3.0, 5.5]
    assert [
        (type(e).__name__, e.at) for e in report.events if "Speaking" in type(e).__name__ or "Barge" in type(e).__name__
    ] == [
        ("AgentStartedSpeaking", 1.0),
        ("AgentStoppedSpeaking", 3.0),
        ("AgentStartedSpeaking", 4.0),
        ("BargeIn", 5.5),
        ("AgentStoppedSpeaking", 5.5),
    ]


def test_the_end_of_the_session_stops_the_reply_that_plays():
    clock, rec = recorder()
    rec.turn_committed("a", "hi", TurnKind.USER)
    rec.played("a", 100.0, 110.0, b"")
    clock.now = 104.0
    rec.closed("Stopped")
    assert rec.report().turns[0].playback_ended_at == 4.0


def test_streamed_tool_calls_get_their_arguments_and_result():
    _clock, rec = recorder()
    rec.turn_committed("reply", "book a table", TurnKind.USER)
    first = {"name": "book", "args": '{"people": ', "id": "c1", "index": 0}
    rec.agent_message("reply", AIMessageChunk(content="", tool_call_chunks=[first]))
    rec.agent_message("reply", AIMessageChunk(content="", tool_call_chunks=[{"args": "4}", "id": None, "index": 0}]))
    rec.agent_message("reply", ToolMessage(content="Booked for 4", tool_call_id="c1"))
    call = rec.report().turns[0].tool_calls[0]
    assert (call.name, call.arguments, call.result, call.error) == ("book", {"people": 4}, "Booked for 4", False)


def test_a_failed_tool_call_is_marked_as_an_error():
    _clock, rec = recorder()
    rec.turn_committed("reply", "book", TurnKind.USER)
    rec.agent_message("reply", AIMessage(content="", tool_calls=[{"name": "book", "args": {"day": "Mon"}, "id": "c1"}]))
    rec.agent_message("reply", ToolMessage(content="Error: closed", tool_call_id="c1", status="error"))
    report = rec.report()
    call = report.turns[0].tool_calls[0]
    assert (call.arguments, call.error) == ({"day": "Mon"}, True)
    assert report.events[-1].error is True


def test_an_error_goes_to_the_timeline_and_to_the_turn_of_its_reply():
    _clock, rec = recorder()
    rec.turn_committed("reply", "hi", TurnKind.USER)
    rec.error(ErrorSource.AGENT, ValueError("Bad model"), "reply")
    report = rec.report()
    assert report.turns[0].error == "ValueError: Bad model"
    assert (report.events[-1].source, report.events[-1].message) == (ErrorSource.AGENT, "ValueError: Bad model")


def test_at_the_same_time_a_reply_stops_before_the_next_reply_starts_and_before_the_session_ends():
    clock, rec = recorder()
    rec.turn_committed("a", "hi", TurnKind.USER)
    rec.played("a", 100.0, 102.0, b"")
    rec.turn_committed("b", "and", TurnKind.USER)
    rec.played("b", 102.0, 103.0, b"")
    clock.now = 103.0
    rec.closed("Audio stream ended")
    names = [type(event).__name__ for event in rec.report().events if event.at >= 2.0]
    assert names == ["AgentStoppedSpeaking", "AgentStartedSpeaking", "AgentStoppedSpeaking", "SessionEnded"]
