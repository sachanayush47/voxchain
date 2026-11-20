"""SarvamSTTProvider: connection parameters, audio frames, and Sarvam messages mapped to STT events."""

import asyncio
import base64
import urllib.parse

import pytest

from voxchain.session import SessionClosed
from voxchain.stt import FinalTranscript, SarvamSTTProvider, SpeechStarted, SpeechStopped


def _stt(server, **kwargs):
    return SarvamSTTProvider(api_key="key", connect=server.connect, **kwargs)


def _query(socket):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(socket.url).query))


async def test_connecting_sends_the_settings_and_always_asks_for_voice_activity(plain_server):
    stt = _stt(plain_server, language_code="hi-IN", extra_params={"interrupt_min_speech_frames": 4})
    await stt.connect()
    socket = plain_server.sockets[0]
    assert socket.url.startswith("wss://api.sarvam.ai/speech-to-text/ws?")
    assert socket.headers == {"Api-Subscription-Key": "key"}
    assert _query(socket) == {
        "language-code": "hi-IN",
        "model": "saaras:v3",
        "mode": "transcribe",
        "sample_rate": "16000",
        "input_audio_codec": "pcm_s16le",
        "high_vad_sensitivity": "true",
        "vad_signals": "true",
        "interrupt_min_speech_frames": "4",
    }


async def test_audio_goes_out_base64_in_sarvams_audio_message(plain_server):
    stt = _stt(plain_server, sample_rate=8000)
    await stt.connect()
    await stt.send_audio(b"\x01\x02")
    assert plain_server.sockets[0].sent == [
        {"audio": {"data": base64.b64encode(b"\x01\x02").decode(), "encoding": "audio/wav", "sample_rate": 8000}}
    ]


def test_messages_map_onto_stt_events(plain_server):
    stt = _stt(plain_server)
    assert stt.handle_message({"type": "events", "data": {"signal_type": "START_SPEECH"}}) == [SpeechStarted()]
    assert stt.handle_message({"type": "events", "data": {"signal_type": "END_SPEECH"}}) == [SpeechStopped()]
    assert stt.handle_message({"type": "data", "data": {"transcript": "namaste"}}) == [FinalTranscript("namaste")]
    assert stt.handle_message({"type": "data", "data": {"transcript": "  "}}) == []
    assert stt.handle_message({"type": "error", "data": {"code": "x", "error": "bad"}}) == []


async def test_a_dropped_socket_is_reopened_and_events_keep_flowing(plain_server, settle):
    stt = _stt(plain_server)
    await stt.connect()
    events = []
    reader = asyncio.create_task(_collect(stt, events))
    plain_server.sockets[0].drop()
    await settle()
    plain_server.sockets[1].push(type="data", data={"transcript": "still here"})
    await settle()
    assert events == [FinalTranscript("still here")]
    await stt.disconnect()
    await reader


async def test_giving_up_on_reconnecting_yields_session_closed(plain_server):
    stt = _stt(plain_server)
    await stt.connect()
    events = []
    reader = asyncio.create_task(_collect(stt, events))
    plain_server.refuse = True
    plain_server.sockets[0].drop()
    await asyncio.wait_for(reader, 1)
    assert isinstance(events[-1], SessionClosed)


def test_an_unsupported_sample_rate_is_rejected(plain_server):
    with pytest.raises(ValueError):
        _stt(plain_server, sample_rate=44100)


async def _collect(provider, events):
    async for event in provider.receive_events():
        events.append(event)
