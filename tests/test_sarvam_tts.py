"""SarvamTTSProvider: config, flush markers, and cancel by socket replacement with no stale audio."""

import asyncio
import base64

from voxchain.session import SessionClosed
from voxchain.tts import AudioChunk, Flushed, SarvamTTSProvider


def _tts(server, **kwargs):
    return SarvamTTSProvider(api_key="key", speaker="priya", connect=server.connect, **kwargs)


def _audio(data: bytes) -> dict:
    return {"type": "audio", "data": {"content_type": "audio/pcm", "audio": base64.b64encode(data).decode()}}


async def _collect(provider, events):
    async for event in provider.receive_events():
        events.append(event)


async def test_connecting_configures_raw_pcm_and_leaves_unset_options_to_sarvam(plain_server):
    tts = _tts(plain_server, pace=1.2)
    await tts.connect()
    socket = plain_server.sockets[0]
    assert "send_completion_event=true" in socket.url
    assert socket.sent == [
        {
            "type": "config",
            "data": {
                "model": "bulbul:v3",
                "target_language_code": "en-IN",
                "speaker": "priya",
                "output_audio_codec": "linear16",
                "speech_sample_rate": 24000,
                "pace": 1.2,
            },
        }
    ]


async def test_each_flush_yields_one_flushed_after_its_audio(plain_server, settle):
    tts = _tts(plain_server)
    await tts.connect()
    events = []
    reader = asyncio.create_task(_collect(tts, events))
    await tts.speak("Hello.")
    await tts.flush()
    socket = plain_server.sockets[0]
    assert [m["type"] for m in socket.sent[1:]] == ["text", "flush"]
    socket.push(**_audio(b"\x01\x02"))
    socket.push(type="event", data={"event_type": "final"})
    socket.push(type="event", data={"event_type": "final"})  # The provider did not ask for this marker and ignores it
    await settle()
    assert events == [AudioChunk(b"\x01\x02"), Flushed()]
    await tts.disconnect()
    await reader


async def test_cancel_replaces_the_socket_and_drops_what_the_old_one_still_sends(plain_server, settle):
    tts = _tts(plain_server)
    await tts.connect()
    events = []
    reader = asyncio.create_task(_collect(tts, events))
    await tts.speak("A long reply.")
    await tts.flush()
    old = plain_server.sockets[0]
    old.push(**_audio(b"stale"))  # The old socket buffered this audio before the cancel arrived
    await tts.cancel()
    await settle()
    await tts.speak("Next.")
    await tts.flush()
    new = plain_server.sockets[1]
    new.push(**_audio(b"fresh"))
    new.push(type="event", data={"event_type": "final"})
    await settle()
    assert events == [AudioChunk(b"fresh"), Flushed()]
    assert [m["type"] for m in new.sent] == ["config", "text", "flush"]
    await tts.disconnect()
    await reader


async def test_cancels_are_planned_replacements_that_never_count_toward_giving_up(plain_server, settle):
    tts = _tts(plain_server)
    await tts.connect()
    events = []
    reader = asyncio.create_task(_collect(tts, events))
    for _ in range(7):  # Five failed attempts end a session, so seven cancels are more than that limit
        await tts.cancel()
        await settle()
    assert len(plain_server.sockets) == 8
    assert events == []  # Specifically, the provider sends no SessionClosed
    await tts.disconnect()
    await reader


async def test_a_credits_error_ends_the_session_without_reconnect_attempts(plain_server):
    tts = _tts(plain_server)
    await tts.connect()
    events = []
    reader = asyncio.create_task(_collect(tts, events))
    error = {"code": 402, "message": "Credits exhausted. Add credits in the API Dashboard."}
    plain_server.sockets[0].push(type="error", data=error)
    await asyncio.wait_for(reader, 1)
    assert events == [
        SessionClosed(reason="Session rejected: (402) Credits exhausted. Add credits in the API Dashboard.")
    ]
    assert len(plain_server.sockets) == 1


async def test_an_error_that_a_reconnect_can_fix_keeps_the_session(plain_server, settle):
    tts = _tts(plain_server)
    await tts.connect()
    events = []
    reader = asyncio.create_task(_collect(tts, events))
    plain_server.sockets[0].push(type="error", data={"code": 500, "message": "Internal error"})
    plain_server.sockets[0].push(**_audio(b"\x01\x00"))
    await settle()
    assert events == [AudioChunk(b"\x01\x00")]
    await tts.disconnect()
    await reader
