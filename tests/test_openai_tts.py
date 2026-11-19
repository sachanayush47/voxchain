"""OpenAITTSProvider: one response at a time, batched sentences, reliable flush markers, clean cancels."""

import asyncio
import base64
import logging

from voxchain.tts import AudioChunk, Flushed, OpenAITTSProvider


def _tts(server, **kwargs):
    return OpenAITTSProvider(api_key="key", connect=server.connect, **kwargs)


async def _start(server, **kwargs):
    tts = _tts(server, **kwargs)
    await tts.connect()
    events = []

    async def collect():
        async for event in tts.receive_events():
            events.append(event)

    return tts, events, asyncio.create_task(collect()), server.sockets[0]


def _text(message):
    return message["response"]["input"][0]["content"][0]["text"]


def _play(socket, response_id, *, audio=b"\x01\x02", transcript=None, status="completed"):
    socket.push(type="response.created", response={"id": response_id})
    socket.push(type="response.output_audio.delta", response_id=response_id, delta=base64.b64encode(audio).decode())
    if transcript is not None:
        socket.push(type="response.output_audio_transcript.done", response_id=response_id, transcript=transcript)
    socket.push(type="response.done", response={"id": response_id, "status": status})


async def test_each_response_reads_its_text_out_of_band(openai_server):
    tts, _, reader, socket = await _start(openai_server)
    await tts.speak("Hello there.")
    create = socket.of_type("response.create")[0]
    assert create["response"]["conversation"] == "none"
    assert create["response"]["output_modalities"] == ["audio"]
    assert _text(create) == "Hello there."
    await tts.disconnect()
    await reader


async def test_sentences_that_arrive_mid_response_are_batched_into_the_next(openai_server, settle):
    tts, events, reader, socket = await _start(openai_server)
    await tts.speak("One.")
    await tts.speak("Two.")
    await tts.speak("Three.")
    await tts.flush()
    _play(socket, "r1")
    await settle()
    assert [_text(m) for m in socket.of_type("response.create")] == ["One.", "Two. Three."]
    _play(socket, "r2")
    await settle()
    assert events == [AudioChunk(b"\x01\x02"), AudioChunk(b"\x01\x02"), Flushed()]
    await tts.disconnect()
    await reader


async def test_a_flush_after_the_audio_already_finished_still_yields_flushed(openai_server, settle):
    tts, events, reader, socket = await _start(openai_server)
    await tts.speak("Quick.")
    _play(socket, "r1")
    await settle()
    await tts.flush()  # The agent was slower than the voice
    await settle()
    assert events == [AudioChunk(b"\x01\x02"), Flushed()]
    await tts.disconnect()
    await reader


async def test_cancel_drops_queued_text_and_the_running_responses_audio(openai_server, settle):
    tts, events, reader, socket = await _start(openai_server)
    await tts.speak("A long reply.")
    socket.push(type="response.created", response={"id": "r1"})
    await settle()
    await tts.speak("More.")
    await tts.flush()
    await tts.cancel()
    assert socket.of_type("response.cancel") == [{"type": "response.cancel", "response_id": "r1"}]
    socket.push(type="response.output_audio.delta", response_id="r1", delta=base64.b64encode(b"stale").decode())
    socket.push(type="response.done", response={"id": "r1", "status": "cancelled"})
    await settle()
    assert events == []
    assert len(socket.of_type("response.create")) == 1  # "More." was dropped with the cancel
    await tts.disconnect()
    await reader


async def test_a_rejected_request_moves_on_to_the_next_text(openai_server, settle):
    tts, _, reader, socket = await _start(openai_server)
    await tts.speak("First.")
    rejected = socket.of_type("response.create")[0]["event_id"]
    await tts.speak("Second.")
    socket.push(type="error", error={"code": "invalid_value", "message": "Rejected", "event_id": rejected})
    await settle()
    assert [_text(m) for m in socket.of_type("response.create")] == ["First.", "Second."]
    await tts.disconnect()
    await reader


async def test_a_paraphrase_is_logged(openai_server, settle, caplog):
    tts, _, reader, socket = await _start(openai_server)
    await tts.speak("Your appointment is at five.")
    with caplog.at_level(logging.WARNING, logger="voxchain.tts.openai"):
        _play(socket, "r1", transcript="Sure! Your appointment is at five.")
        await settle()
    assert "instead of" in caplog.text
    await tts.disconnect()
    await reader
