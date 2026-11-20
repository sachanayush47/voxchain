"""OpenAIRealtimeProvider.handle_event: it turns server events into pipeline events and replayable history."""

import base64

from voxchain.realtime import (
    AgentTranscript,
    AudioDelta,
    FunctionCall,
    OpenAIRealtimeProvider,
    ResponseDone,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    UserTranscript,
)


def _provider():
    return OpenAIRealtimeProvider(api_key="test")


def _feed(provider, *events):
    out = []
    for event in events:
        out += provider.handle_event(event)
    return out


def _audio(ms: int) -> str:
    return base64.b64encode(b"\0" * (48 * ms)).decode()


def _assistant_turn(item_id="a1", response_id="r1", text="Hello there, how can I help?", audio_ms=1000):
    return [
        {"type": "response.created", "response": {"id": response_id}},
        {
            "type": "response.output_audio.delta",
            "response_id": response_id,
            "item_id": item_id,
            "delta": _audio(audio_ms),
        },
        {
            "type": "response.output_audio_transcript.done",
            "response_id": response_id,
            "item_id": item_id,
            "transcript": text,
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_audio", "transcript": text}],
            },
        },
        {"type": "response.done", "response": {"id": response_id, "status": "completed", "output": []}},
    ]


def test_a_turn_maps_onto_pipeline_events():
    p = _provider()
    events = _feed(
        p,
        {"type": "input_audio_buffer.speech_started", "item_id": "u1"},
        {"type": "input_audio_buffer.speech_stopped", "item_id": "u1"},
        {"type": "input_audio_buffer.committed", "item_id": "u1"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "u1", "transcript": "Hi"},
        *_assistant_turn(),
    )
    assert events == [
        SpeechStarted(),
        SpeechStopped(),
        UserTranscript(text="Hi"),
        ResponseStarted(response_id="r1"),
        AudioDelta(response_id="r1", item_id="a1", data=b"\0" * 48000),
        AgentTranscript(response_id="r1", item_id="a1", text="Hello there, how can I help?"),
        ResponseDone(response_id="r1", status="completed"),
    ]


def test_function_calls_come_from_response_done_and_precede_it():
    p = _provider()
    done = {
        "type": "response.done",
        "response": {
            "id": "r1",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "get_time",
                    "arguments": "{}",
                    "status": "completed",
                },
                {"type": "function_call", "call_id": "c2", "name": "lookup", "arguments": '{"q": "x"}'},
                {
                    "type": "function_call",
                    "call_id": "c3",
                    "name": "lookup",
                    "arguments": '{"q',
                    "status": "incomplete",
                },
            ],
        },
    }
    assert _feed(p, done) == [
        FunctionCall(response_id="r1", call_id="c1", name="get_time", arguments="{}"),
        FunctionCall(response_id="r1", call_id="c2", name="lookup", arguments='{"q": "x"}'),
        ResponseDone(response_id="r1", status="completed"),
    ]


def test_a_late_user_transcript_still_replays_before_the_reply_it_prompted():
    p = _provider()
    _feed(
        p,
        {"type": "input_audio_buffer.committed", "item_id": "u1"},
        *_assistant_turn(text="Sure."),
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "u1",
            "transcript": "Can you help?",
        },
    )
    assert [item["role"] for item in p.history_items()] == ["user", "assistant"]
    assert p.history_items()[0]["content"] == [{"type": "input_text", "text": "Can you help?"}]


async def test_a_truncated_reply_replays_only_what_the_user_heard():
    p = _provider()
    _feed(p, *_assistant_turn(text="one two three four five six seven eight nine ten", audio_ms=2000))
    await p.truncate("a1", 600)
    assert p.history_items()[-1]["content"] == [{"type": "output_text", "text": "one two three"}]


async def test_a_deleted_reply_is_not_replayed():
    p = _provider()
    _feed(p, *_assistant_turn())
    await p.delete_item("a1")
    assert p.history_items() == []


def test_a_user_turn_without_a_transcript_is_skipped_on_replay():
    p = _provider()
    _feed(
        p,
        {"type": "input_audio_buffer.committed", "item_id": "u1"},
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "u1",
            "error": {"message": "Transcription unavailable"},
        },
        *_assistant_turn(),
    )
    assert [item["role"] for item in p.history_items()] == ["assistant"]


def test_tool_calls_replay_in_order_with_their_results_after_them():
    p = _provider()
    call = {"id": "f1", "type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"}
    _feed(p, {"type": "response.output_item.done", "item": call})
    assert p.history_items() == [{"type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"}]


def test_a_bad_event_is_ignored_not_fatal():
    assert _provider().handle_event({"type": "rate_limits.updated", "rate_limits": []}) == []
