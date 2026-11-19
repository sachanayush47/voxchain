"""OpenAISTTProvider: local VAD drives turns, only speech is streamed, transcripts come out in speech order."""

import asyncio
import base64

import pytest

from voxkit.stt import FinalTranscript, OpenAISTTProvider, PartialTranscript, SpeechStarted, SpeechStopped

WINDOW = 512 * 2


class ScriptedVAD:
    sample_rate = 16000
    window_bytes = WINDOW

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def speech_probability(self, window):
        return self.probabilities.pop(0) if self.probabilities else 0.0


def _stt(server, probabilities=(), **kwargs):
    return OpenAISTTProvider(api_key="key", vad=ScriptedVAD(probabilities), connect=server.connect, **kwargs)


def test_the_session_transcribes_24khz_pcm_with_server_turn_detection_off(openai_server):
    config = _stt(openai_server, languages=["en"], keywords=["AC-42"]).session_config()
    assert config == {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-live-transcribe", "languages": ["en"], "keywords": ["AC-42"]},
                "turn_detection": None,
            }
        },
    }


async def test_connecting_asks_for_a_transcription_session(openai_server):
    await _stt(openai_server).connect()
    assert openai_server.sockets[0].url == "wss://api.openai.com/v1/realtime?intent=transcription"


async def test_only_speech_is_streamed_and_each_segment_is_committed(openai_server):
    # 2 silent windows, 7 speech windows, 16 silent windows (a 500 ms hangover is 16 windows of 32 ms).
    stt = _stt(openai_server, [0.0] * 2 + [0.9] * 7 + [0.0] * 16, vad_prefix_ms=32)
    await stt.connect()
    await stt.send_audio(b"\0" * WINDOW * 25)
    sent = openai_server.sockets[0].sent[1:]  # After session.update
    assert [m["type"] for m in sent][-1] == "input_audio_buffer.commit"
    streamed = sum(len(base64.b64decode(m["audio"])) for m in sent if m["type"] == "input_audio_buffer.append")
    # Pre-roll (8 windows) + 7 speech + 16 hangover - the windows before the pre-roll; upsampled by 1.5
    assert abs(streamed - (25 - 1) * WINDOW * 1.5) <= 4


async def test_voice_activity_from_the_vad_and_transcripts_from_openai_share_one_stream(openai_server):
    stt = _stt(openai_server, [0.9] * 7 + [0.0] * 16)
    await stt.connect()
    events = []
    reader = asyncio.create_task(_collect(stt, events))
    await stt.send_audio(b"\0" * WINDOW * 23)
    openai_server.sockets[0].push(type="input_audio_buffer.committed", item_id="i1")
    openai_server.sockets[0].push(
        type="conversation.item.input_audio_transcription.completed", item_id="i1", transcript="hello"
    )
    await asyncio.sleep(0.01)
    assert events == [SpeechStarted(), SpeechStopped(), FinalTranscript("hello")]
    await stt.disconnect()
    await reader


def test_transcripts_finished_out_of_order_come_out_in_speech_order(openai_server):
    stt = _stt(openai_server)
    done = "conversation.item.input_audio_transcription.completed"
    events = []
    for event in [
        {"type": "input_audio_buffer.committed", "item_id": "a"},
        {"type": "input_audio_buffer.committed", "item_id": "b"},
        {"type": done, "item_id": "b", "transcript": "second"},
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "x",
            "error": {"message": "Unknown failure"},
        },
        {"type": done, "item_id": "a", "transcript": "first"},
    ]:
        events += stt.handle_event(event)
    assert events == [FinalTranscript("first"), FinalTranscript("second")]


def test_partials_accumulate_when_asked_for(openai_server):
    stt = _stt(openai_server, include_partials=True)
    delta = {"type": "conversation.item.input_audio_transcription.delta", "item_id": "a"}
    assert stt.handle_event({**delta, "delta": "hel"}) == [PartialTranscript("hel")]
    assert stt.handle_event({**delta, "delta": "lo"}) == [PartialTranscript("hello")]
    assert _stt(openai_server).handle_event({**delta, "delta": "hel"}) == []


def test_settings_openai_would_reject_fail_at_construction(openai_server):
    with pytest.raises(ValueError):
        _stt(openai_server, keywords=["bad<keyword"])
    with pytest.raises(ValueError):
        _stt(openai_server, sample_rate=24000)


async def _collect(provider, events):
    async for event in provider.receive_events():
        events.append(event)
