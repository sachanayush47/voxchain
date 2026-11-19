"""OpenAI realtime transcription (``gpt-live-transcribe``) with local voice activity detection."""

import base64
import logging
import re
from collections import deque
from collections.abc import AsyncIterator, Callable

import websockets

from voxkit.audio import PCM16Resampler
from voxkit.protocols import compact
from voxkit.protocols.openai_realtime import SAMPLE_RATE, ClientEvent, ServerEvent, open_session
from voxkit.session import ProviderSession
from voxkit.stt.base import STTProvider
from voxkit.stt.events import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped, STTEvent
from voxkit.vad.gate import SegmentAudio, SegmentEnded, SegmentStarted, VoiceActivityModel, VoiceGate

logger = logging.getLogger(__name__)

# OpenAI rejects the whole session config if any keyword contains these.
INVALID_KEYWORD = re.compile(r"[<>\r\n]")


class OpenAISTTProvider(STTProvider):
    """Transcribes speech on an OpenAI transcription session, cutting turns with a local VAD.

    OpenAI's current transcription models run with server-side turn detection
    off, so a local :class:`~voxkit.vad.gate.VoiceGate` decides when the user
    starts and stops talking. It reports that for barge-in, streams only speech
    to OpenAI (silence is never sent or billed), and commits each segment so
    OpenAI finalizes its transcript. Transcripts are yielded in speech order even
    when OpenAI finishes them out of order.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-live-transcribe",
        languages: list[str] | None = None,
        keywords: list[str] | None = None,
        prompt: str | None = None,
        delay: str | None = None,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        vad_start_ms: int = 200,
        vad_stop_ms: int = 500,
        vad_prefix_ms: int = 300,
        include_partials: bool = False,
        vad: VoiceActivityModel | None = None,
        connect: Callable = websockets.connect,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000 (what the VAD runs at), got {sample_rate}")
        for keyword in keywords or []:
            if INVALID_KEYWORD.search(keyword):
                raise ValueError(f"Keywords may not contain '<', '>' or line breaks: {keyword!r}")
        self.api_key = api_key
        self.model = model
        self.languages = languages
        self.keywords = keywords
        self.prompt = prompt
        self.delay = delay
        self.sample_rate = sample_rate
        self.include_partials = include_partials
        self._connect = connect

        if vad is None:
            from voxkit.vad.silero import SileroVAD

            vad = SileroVAD(sample_rate)
        self.gate = VoiceGate(
            vad, threshold=vad_threshold, start_ms=vad_start_ms, stop_ms=vad_stop_ms, prefix_ms=vad_prefix_ms
        )
        self.session = ProviderSession(name="OpenAI STT", open=self._open, handle=self.handle_event)
        self._resampler = PCM16Resampler(sample_rate, SAMPLE_RATE)
        self._order = TranscriptOrder()
        self._partials: dict[str, str] = {}

    def session_config(self) -> dict:
        """The ``session.update`` payload: 24 kHz PCM in, server-side turn detection off."""
        transcription = {
            "model": self.model,
            "languages": self.languages,
            "keywords": self.keywords,
            "prompt": self.prompt,
            "delay": self.delay,
        }
        audio_input = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "transcription": compact(transcription),
            "turn_detection": None,
        }
        return {"type": "transcription", "audio": {"input": audio_input}}

    async def connect(self) -> None:
        """Open and configure the transcription session, raising if OpenAI rejects it."""
        await self.session.connect()

    async def _open(self):
        ws = await open_session(
            self._connect, api_key=self.api_key, session=self.session_config(), intent="transcription"
        )
        self._order = TranscriptOrder()  # Segments from an old session never complete on this one
        logger.info(f"OpenAI STT connected | model={self.model}")
        return ws

    async def send_audio(self, pcm: bytes) -> None:
        """Run the VAD on PCM16 audio at ``sample_rate`` and stream any speech to OpenAI."""
        for segment in self.gate.push(pcm):
            match segment:
                case SegmentStarted(audio=audio):
                    self.session.emit(SpeechStarted())
                    self._resampler = PCM16Resampler(self.sample_rate, SAMPLE_RATE)
                    await self._append(audio)
                case SegmentAudio(audio=audio):
                    await self._append(audio)
                case SegmentEnded(audio=audio):
                    await self._append(audio)
                    self.session.emit(SpeechStopped())
                    await self.session.socket.send({"type": ClientEvent.INPUT_AUDIO_BUFFER_COMMIT})

    async def _append(self, audio: bytes) -> None:
        if data := self._resampler.process(audio):
            encoded = base64.b64encode(data).decode("ascii")
            await self.session.socket.send({"type": ClientEvent.INPUT_AUDIO_BUFFER_APPEND, "audio": encoded})

    def receive_events(self) -> AsyncIterator[STTEvent]:
        """Yield voice activity from the local VAD and transcripts from OpenAI, across reconnects."""
        return self.session.events()

    async def disconnect(self) -> None:
        """Close the session. Safe to call more than once."""
        await self.session.close()

    def handle_event(self, event: dict) -> list[STTEvent]:
        """Translate one server event into transcripts, in speech order."""
        match event.get("type"):
            case ServerEvent.INPUT_AUDIO_BUFFER_COMMITTED:
                return self._order.committed(event["item_id"])
            case ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA:
                if not self.include_partials or not event.get("delta"):
                    return []
                text = self._partials.get(event["item_id"], "") + event["delta"]
                self._partials[event["item_id"]] = text
                return [PartialTranscript(text)]
            case ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                self._partials.pop(event["item_id"], None)
                logger.debug(f"OpenAI STT: Transcript {event.get('transcript')!r}")
                return self._order.finished(event["item_id"], event.get("transcript", ""))
            case ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED:
                self._partials.pop(event["item_id"], None)
                logger.warning(f"OpenAI STT: Transcription failed: {event.get('error', {}).get('message')}")
                return self._order.finished(event["item_id"], "")
            case ServerEvent.ERROR:
                error = event.get("error", {})
                logger.error(f"OpenAI STT error ({error.get('code')}): {error.get('message')}")
            case other:
                logger.debug(f"OpenAI STT: Ignoring {other}")
        return []


class TranscriptOrder:
    """Releases transcripts in commit order, since OpenAI may finish later segments first."""

    def __init__(self) -> None:
        self._committed: deque[str] = deque()
        self._finished: dict[str, str] = {}

    def committed(self, item_id: str) -> list[STTEvent]:
        """A segment was committed; release whatever is now in order."""
        self._committed.append(item_id)
        return self._release()

    def finished(self, item_id: str, text: str) -> list[STTEvent]:
        """A segment's transcript arrived (empty if it failed); release whatever is now in order."""
        self._finished[item_id] = text
        return self._release()

    def _release(self) -> list[STTEvent]:
        # A transcript for a segment never seen committed (e.g. across a reconnect) can't be ordered; release it.
        ready = [item_id for item_id in self._finished if item_id not in self._committed]
        while self._committed and self._committed[0] in self._finished:
            ready.append(self._committed.popleft())
        texts = [self._finished.pop(item_id) for item_id in ready]
        return [FinalTranscript(text) for text in texts if text.strip()]
