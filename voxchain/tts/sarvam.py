"""Sarvam AI text-to-speech on a websocket stream."""

import base64
import json
import logging
from collections.abc import AsyncIterator, Callable

import websockets

from voxchain.protocols import compact
from voxchain.protocols.sarvam import TTS_URL, TTSClientMessage, TTSEventType, TTSServerMessage, open_socket
from voxchain.session import ProviderSession
from voxchain.tts.base import TTSProvider
from voxchain.tts.events import AudioChunk, Flushed, TTSEvent

logger = logging.getLogger(__name__)


class SarvamTTSProvider(TTSProvider):
    """Sarvam ``bulbul`` voices as a stream of raw PCM16.

    Sarvam cannot cancel synthesis on a socket, so :meth:`cancel` replaces the
    socket. The provider discards all data that the old socket still delivers.
    The provider does not send options that are ``None``, so Sarvam's defaults
    apply. Some options work on only one model: ``pitch`` and ``loudness`` on
    ``bulbul:v2``, and ``temperature`` and ``dict_id`` on ``bulbul:v3``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        speaker: str,
        language_code: str = "en-IN",
        model: str = "bulbul:v3",
        sample_rate: int = 24000,
        pace: float | None = None,
        temperature: float | None = None,
        pitch: float | None = None,
        loudness: float | None = None,
        enable_preprocessing: bool | None = None,
        dict_id: str | None = None,
        min_buffer_size: int | None = None,
        max_chunk_length: int | None = None,
        connect: Callable = websockets.connect,
    ) -> None:
        if sample_rate not in (8000, 16000, 22050, 24000):
            raise ValueError(f"Sarvam synthesizes at 8000, 16000, 22050 or 24000 Hz, got {sample_rate}")
        self.api_key = api_key
        self.speaker = speaker
        self.language_code = language_code
        self.model = model
        self.sample_rate = sample_rate
        self.options = {
            "pace": pace,
            "temperature": temperature,
            "pitch": pitch,
            "loudness": loudness,
            "enable_preprocessing": enable_preprocessing,
            "dict_id": dict_id,
            "min_buffer_size": min_buffer_size,
            "max_chunk_length": max_chunk_length,
        }
        self._connect = connect

        self.session = ProviderSession(name="Sarvam TTS", open=self._open, handle=self.handle_message)
        self._pending_flushes = 0

    def config(self) -> dict:
        """The ``config`` message that the provider sends first on each socket."""
        data = {
            "model": self.model,
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "output_audio_codec": "linear16",  # Raw PCM, because voxchain sends raw PCM to the client
            "speech_sample_rate": self.sample_rate,
            **compact(self.options),
        }
        return {"type": TTSClientMessage.CONFIG, "data": data}

    async def connect(self) -> None:
        """Open a synthesis socket and configure it."""
        await self.session.connect()

    async def _open(self):
        ws = await open_socket(
            self._connect, TTS_URL, api_key=self.api_key, model=self.model, send_completion_event=True
        )
        try:
            await ws.send(json.dumps(self.config()))
        except BaseException:
            await ws.close()
            raise
        self._pending_flushes = 0  # Flushes on the old socket never complete on this one
        logger.info(f"Sarvam TTS connected | model={self.model} | speaker={self.speaker}")
        return ws

    async def speak(self, text: str) -> None:
        """Queue a sentence for synthesis."""
        await self.session.socket.send_reliably({"type": TTSClientMessage.TEXT, "data": {"text": text}})

    async def flush(self) -> None:
        """Synthesize all text up to now. A ``Flushed`` follows its audio."""
        if await self.session.socket.send_reliably({"type": TTSClientMessage.FLUSH}):
            self._pending_flushes += 1

    async def cancel(self) -> None:
        """Replace the socket to discard all requests up to now."""
        await self.session.replace()

    def receive_events(self) -> AsyncIterator[TTSEvent]:
        """Yield audio and flush markers for the life of the conversation, across reconnects."""
        return self.session.events()

    async def disconnect(self) -> None:
        """Close the socket. It is safe to call this more than once."""
        await self.session.close()

    def handle_message(self, message: dict) -> list[TTSEvent]:
        """Translate one Sarvam message."""
        data = message.get("data") or {}
        match message.get("type"):
            case TTSServerMessage.AUDIO:
                return [AudioChunk(base64.b64decode(data["audio"]))]
            case TTSServerMessage.EVENT if data.get("event_type") == TTSEventType.FINAL:
                if self._pending_flushes:
                    self._pending_flushes -= 1
                    return [Flushed()]
            case TTSServerMessage.ERROR:
                logger.error(f"Sarvam TTS error ({data.get('code')}): {data.get('message')}")
            case other:
                logger.debug(f"Sarvam TTS: Ignored {other!r}")
        return []
