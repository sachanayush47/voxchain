"""OpenAI Realtime API wire vocabulary, shared by the OpenAI STT, TTS and speech-to-speech providers."""

import asyncio
import json
import time
import urllib.parse
from collections.abc import Callable
from enum import StrEnum

URL = "wss://api.openai.com/v1/realtime"
SAMPLE_RATE = 24000  # The API takes and returns 24 kHz PCM16
BYTES_PER_MS = SAMPLE_RATE * 2 / 1000
# OpenAI ends a session at 60 minutes; an idle one is replaced a little before that.
MAX_SESSION_AGE_S = 55 * 60
HANDSHAKE_TIMEOUT_S = 10.0
# Expected when racing the server (a cancel after it finished, a create while one is starting).
EXPECTED_ERROR_CODES = frozenset({"conversation_already_has_active_response", "response_cancel_not_active"})


class ServerEvent(StrEnum):
    """Event types OpenAI sends that voxkit acts on."""

    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA = "conversation.item.input_audio_transcription.delta"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
    RESPONSE_CREATED = "response.created"
    RESPONSE_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_DONE = "response.done"
    ERROR = "error"


class ClientEvent(StrEnum):
    """Event types voxkit sends to OpenAI."""

    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"


class ItemType(StrEnum):
    """Conversation item types voxkit builds or inspects."""

    MESSAGE = "message"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"


async def open_session(connect: Callable, *, api_key: str, session: dict, **query: str):
    """Connect and configure a session, raising if OpenAI rejects the config. Returns the open socket.

    ``query`` becomes the URL's query string: ``model=...`` for a conversation,
    ``intent="transcription"`` for a transcription session.
    """
    ws = await connect(
        f"{URL}?{urllib.parse.urlencode(query)}",
        additional_headers={"Authorization": f"Bearer {api_key}"},
        max_size=None,
    )
    try:
        await _expect(ws, ServerEvent.SESSION_CREATED)
        await ws.send(json.dumps({"type": ClientEvent.SESSION_UPDATE, "session": session}))
        # A rejected config must surface now; mid-call is too late to fall back.
        await _expect(ws, ServerEvent.SESSION_UPDATED)
    except BaseException:
        await ws.close()
        raise
    return ws


async def _expect(ws, event_type: ServerEvent) -> None:
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.01, deadline - time.monotonic()))
        event = json.loads(raw)
        if event.get("type") == event_type:
            return
        if event.get("type") == ServerEvent.ERROR:
            error = event.get("error", {})
            raise ConnectionError(f"OpenAI Realtime rejected the session: {error.get('message')} ({error.get('code')})")
