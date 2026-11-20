"""Sarvam AI speech-to-text on a websocket stream."""

import base64
import logging
from collections.abc import AsyncIterator, Callable

import websockets

from voxchain.protocols.sarvam import (
    STT_URL,
    STTClientMessage,
    STTServerMessage,
    STTSignal,
    open_socket,
    read_error,
    rejection,
)
from voxchain.session import ProviderSession, SessionRejected
from voxchain.stt.base import STTProvider
from voxchain.stt.events import FinalTranscript, SpeechStarted, SpeechStopped, STTEvent

logger = logging.getLogger(__name__)


class SarvamSTTProvider(STTProvider):
    """Sarvam ``saaras``/``saarika`` transcription, with the voice activity signals of Sarvam for barge-in.

    ``extra_params`` sends other Sarvam connection parameters without change,
    for example a VAD setting such as ``{"interrupt_min_speech_frames": 4}``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saaras:v3",
        mode: str | None = "transcribe",
        language_code: str = "unknown",
        sample_rate: int = 16000,
        input_audio_codec: str = "pcm_s16le",
        high_vad_sensitivity: bool | None = True,
        flush_signal: bool | None = None,
        extra_params: dict[str, str | int | float | bool] | None = None,
        connect: Callable = websockets.connect,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(f"The Sarvam STT stream takes 8000 or 16000 Hz audio, got {sample_rate}")
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.input_audio_codec = input_audio_codec
        self.high_vad_sensitivity = high_vad_sensitivity
        self.flush_signal = flush_signal
        self.extra_params = extra_params or {}
        self._connect = connect

        self.session = ProviderSession(
            name="Sarvam STT", open=self._open, handle=self.handle_message, rejection=rejection
        )

    def query(self) -> dict:
        """The connection parameters. Sarvam's defaults apply to ``None`` values."""
        return {
            "language-code": self.language_code,
            "model": self.model,
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "input_audio_codec": self.input_audio_codec,
            "high_vad_sensitivity": self.high_vad_sensitivity,
            "vad_signals": True,  # The pipeline needs SpeechStarted for barge-in
            "flush_signal": self.flush_signal,
            **self.extra_params,
        }

    async def connect(self) -> None:
        """Open the transcription socket."""
        await self.session.connect()

    async def _open(self):
        ws = await open_socket(self._connect, STT_URL, api_key=self.api_key, **self.query())
        logger.info(f"Sarvam STT connected | model={self.model} | language={self.language_code}")
        return ws

    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 audio at ``sample_rate``. The provider discards audio during a reconnect."""
        audio = {
            "data": base64.b64encode(pcm).decode("ascii"),
            "encoding": "audio/wav",
            "sample_rate": self.sample_rate,
        }
        await self.session.socket.send({"audio": audio})

    async def flush(self) -> None:
        """Make Sarvam finalize the buffered audio now, and not wait for silence. This needs ``flush_signal=True``."""
        await self.session.socket.send({"type": STTClientMessage.FLUSH})

    def receive_events(self) -> AsyncIterator[STTEvent]:
        """Yield events for the life of the conversation, across reconnects."""
        return self.session.events()

    async def disconnect(self) -> None:
        """Close the socket. It is safe to call this more than once."""
        await self.session.close()

    def handle_message(self, message: dict) -> list[STTEvent]:
        """Translate one Sarvam message."""
        data = message.get("data") or {}
        match message.get("type"):
            case STTServerMessage.DATA:
                transcript = data.get("transcript", "")
                logger.debug(f"Sarvam STT: Transcript {transcript!r}")
                return [FinalTranscript(transcript)] if transcript.strip() else []
            case STTServerMessage.EVENTS:
                match data.get("signal_type"):
                    case STTSignal.START_SPEECH:
                        return [SpeechStarted()]
                    case STTSignal.END_SPEECH:
                        return [SpeechStopped()]
                logger.debug(f"Sarvam STT: Ignored signal {data.get('signal_type')!r}")
            case STTServerMessage.ERROR:
                fatal, text = read_error(data)
                if fatal:
                    raise SessionRejected(text)
                logger.error(f"Sarvam STT error: {text}")
            case other:
                logger.debug(f"Sarvam STT: Ignored {other!r}")
        return []
