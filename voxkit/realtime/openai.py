"""OpenAI Realtime speech-to-speech over a websocket (GA protocol)."""

import base64
import itertools
import json
import logging
import time
from collections.abc import AsyncIterator, Callable

import websockets

from voxkit.protocols import compact
from voxkit.protocols.openai_realtime import (
    BYTES_PER_MS,
    EXPECTED_ERROR_CODES,
    MAX_SESSION_AGE_S,
    SAMPLE_RATE,
    ClientEvent,
    ItemType,
    ServerEvent,
    open_session,
)
from voxkit.realtime.base import RealtimeProvider, ToolSpec
from voxkit.realtime.events import (
    AgentTranscript,
    AudioDelta,
    FunctionCall,
    RealtimeEvent,
    ResponseDone,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    UserTranscript,
)
from voxkit.realtime.history import ConversationHistory, Turn
from voxkit.realtime.response_gate import ResponseGate
from voxkit.session import ProviderSession

logger = logging.getLogger(__name__)


class OpenAIRealtimeProvider(RealtimeProvider):
    """A ``gpt-realtime-2.1`` conversation: OpenAI's turn detection, transcription, speech and tool calls.

    The conversation is replayed as text into a fresh session after a drop or
    before the 60-minute limit (see :class:`~voxkit.realtime.history.ConversationHistory`).
    User turns replay from their transcripts, so keep ``transcription_model`` set.
    """

    input_sample_rate = SAMPLE_RATE
    output_sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        api_key: str,
        instructions: str = "",
        model: str = "gpt-realtime-2.1",
        voice: str = "marin",
        speed: float | None = None,
        turn_detection: str = "semantic_vad",
        eagerness: str | None = None,
        vad_threshold: float | None = None,
        silence_duration_ms: int | None = None,
        prefix_padding_ms: int | None = None,
        idle_timeout_ms: int | None = None,
        interrupt_response: bool = True,
        noise_reduction: str | None = None,
        transcription_model: str | None = "gpt-transcribe",
        transcription_languages: list[str] | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        truncation: str | float | None = None,
        connect: Callable = websockets.connect,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if turn_detection not in ("semantic_vad", "server_vad"):
            raise ValueError(f"Turn detection must be 'semantic_vad' or 'server_vad', got {turn_detection!r}")
        if isinstance(truncation, float) and not 0.0 <= truncation <= 1.0:
            raise ValueError(f"A truncation retention ratio must be between 0 and 1, got {truncation}")
        self.api_key = api_key
        self.instructions = instructions
        self.model = model
        self.voice = voice
        self.speed = speed
        self.turn_detection = turn_detection
        self.eagerness = eagerness
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.prefix_padding_ms = prefix_padding_ms
        self.idle_timeout_ms = idle_timeout_ms
        self.interrupts_on_speech = interrupt_response
        self.noise_reduction = noise_reduction
        self.transcription_model = transcription_model
        self.transcription_languages = transcription_languages
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.truncation = truncation
        self._connect = connect
        self._clock = clock

        self.history = ConversationHistory()
        self.responses = ResponseGate()
        self.session = ProviderSession(
            name="OpenAI Realtime",
            open=self._open,
            handle=self.handle_event,
            after_connect=self._maybe_create_response,  # A response owed before a drop is still owed
            after_message=self._after_message,
        )
        self._tools: list[ToolSpec] = []
        self._connected_at = 0.0
        self._event_ids = itertools.count()
        self._user_speaking = False

    def set_tools(self, tools: list[ToolSpec]) -> None:
        """Declare the tools the model may call. Call before :meth:`connect`."""
        self._tools = list(tools)

    def session_config(self) -> dict:
        """The ``session.update`` payload. Unset options are left out so OpenAI's defaults apply."""
        turn_detection = {
            "type": self.turn_detection,
            "create_response": True,
            "interrupt_response": self.interrupts_on_speech,
        }
        if self.turn_detection == "semantic_vad":
            turn_detection["eagerness"] = self.eagerness
        else:
            turn_detection |= {
                "threshold": self.vad_threshold,
                "silence_duration_ms": self.silence_duration_ms,
                "prefix_padding_ms": self.prefix_padding_ms,
                "idle_timeout_ms": self.idle_timeout_ms,
            }
        audio_input = {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}, "turn_detection": compact(turn_detection)}
        if self.noise_reduction:
            audio_input["noise_reduction"] = {"type": self.noise_reduction}
        if self.transcription_model:
            transcription = {"model": self.transcription_model, "languages": self.transcription_languages}
            audio_input["transcription"] = compact(transcription)

        config = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self.instructions,
            "audio": {
                "input": audio_input,
                "output": compact(
                    {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}, "voice": self.voice, "speed": self.speed}
                ),
            },
            "max_output_tokens": self.max_output_tokens,
        }
        if self._tools:
            config["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self._tools
            ]
            config["tool_choice"] = "auto"
        if self.reasoning_effort:
            config["reasoning"] = {"effort": self.reasoning_effort}
        if isinstance(self.truncation, float):
            config["truncation"] = {"type": "retention_ratio", "retention_ratio": self.truncation}
        elif self.truncation:
            config["truncation"] = self.truncation
        return compact(config)

    def history_items(self) -> list[dict]:
        """The conversation so far as ``conversation.item.create`` items for a fresh session."""
        return [_replay_item(turn) for turn in self.history.replay()]

    async def connect(self) -> None:
        """Open a session, configure it and replay the conversation, raising if OpenAI rejects any of it."""
        await self.session.connect()

    def receive_events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield events for the life of the conversation, across reconnects."""
        return self.session.events()

    async def disconnect(self) -> None:
        """End the session. Safe to call more than once."""
        await self.session.close()

    async def _open(self):
        if lost := self.responses.active:
            # The in-flight response died with the old socket; the pipeline still needs its ResponseDone.
            self.responses.response_done(lost)
            self.session.emit(ResponseDone(response_id=lost, status="failed"))
        started = self._clock()
        ws = await open_session(self._connect, api_key=self.api_key, session=self.session_config(), model=self.model)
        history = self.history_items()
        try:
            for item in history:
                await ws.send(json.dumps({"type": ClientEvent.CONVERSATION_ITEM_CREATE, "item": item}))
        except BaseException:
            await ws.close()
            raise
        self._connected_at = self._clock()
        self._user_speaking = False
        self.responses.reset()
        logger.info(
            f"OpenAI Realtime connected in {(self._connected_at - started) * 1000:.0f}ms "
            f"| model={self.model} | restored {len(history)} items"
        )
        return ws

    async def _after_message(self) -> None:
        await self._maybe_create_response()
        idle = self.responses.idle and self.history.settled and not self._user_speaking
        if idle and self._clock() - self._connected_at > MAX_SESSION_AGE_S:
            logger.info("OpenAI Realtime: Replacing the session before its 60-minute limit")
            await self.session.replace()

    def handle_event(self, event: dict) -> list[RealtimeEvent]:
        """Translate one server event, updating the conversation history and response state."""
        event_type = event.get("type", "")
        match event_type:
            case ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                self._user_speaking = True
                self.responses.user_started_speaking()
                return [SpeechStarted()]

            case ServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
                self._user_speaking = False
                return [SpeechStopped()]

            case ServerEvent.INPUT_AUDIO_BUFFER_COMMITTED:
                self.history.user_turn(event["item_id"], awaiting_transcript=bool(self.transcription_model))
                return []

            case ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                text = event.get("transcript", "")
                self.history.user_transcript(event["item_id"], text)
                return [UserTranscript(text=text)] if text else []

            case ServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED:
                self.history.user_transcript(event["item_id"], None)
                logger.warning(f"OpenAI Realtime: User transcription failed: {event.get('error', {}).get('message')}")
                return []

            case ServerEvent.RESPONSE_CREATED:
                response_id = event["response"]["id"]
                self.responses.response_created(response_id)
                return [ResponseStarted(response_id=response_id)]

            case ServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA:
                data = base64.b64decode(event["delta"])
                self.history.assistant_audio(event["item_id"], len(data) / BYTES_PER_MS)
                return [AudioDelta(response_id=event["response_id"], item_id=event["item_id"], data=data)]

            case ServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE:
                text = event.get("transcript", "")
                return [AgentTranscript(response_id=event["response_id"], item_id=event["item_id"], text=text)]

            case ServerEvent.RESPONSE_OUTPUT_ITEM_DONE:
                self._record_output_item(event["item"])
                return []

            case ServerEvent.RESPONSE_DONE:
                return self._on_response_done(event["response"])

            case ServerEvent.ERROR:
                error = event.get("error", {})
                self.responses.refused(error.get("event_id"))
                message = f"OpenAI Realtime error ({error.get('code')}): {error.get('message')}"
                if error.get("code") in EXPECTED_ERROR_CODES:
                    logger.debug(message)
                else:
                    logger.error(message)
                return []

            case _:
                logger.debug(f"OpenAI Realtime: Ignoring {event_type}")
                return []

    def _record_output_item(self, item: dict) -> None:
        if item.get("type") == ItemType.MESSAGE and item.get("role") == "assistant":
            parts = [part.get("transcript") or part.get("text") or "" for part in item.get("content", [])]
            self.history.assistant_text(item["id"], " ".join(p for p in parts if p))
        elif item.get("type") == ItemType.FUNCTION_CALL:
            arguments = item.get("arguments") or "{}"
            self.history.function_call(item["id"], item.get("call_id", ""), item.get("name", ""), arguments)

    def _on_response_done(self, response: dict) -> list[RealtimeEvent]:
        response_id, status = response.get("id", ""), response.get("status", "")
        self.responses.response_done(response_id)
        if status in ("failed", "incomplete"):
            logger.warning(f"OpenAI Realtime: Response {status}: {response.get('status_details')}")

        events: list[RealtimeEvent] = []
        for item in response.get("output") or []:
            # Status is optional on function calls; only an incomplete one lacks usable arguments.
            if item.get("type") == ItemType.FUNCTION_CALL and item.get("status") != "incomplete":
                self.history.open_call(item["call_id"])
                arguments = item.get("arguments") or "{}"
                events.append(FunctionCall(response_id, item["call_id"], item.get("name", ""), arguments))
        events.append(ResponseDone(response_id=response_id, status=status))
        return events

    async def send_audio(self, pcm: bytes) -> None:
        """Send PCM16 audio at 24 kHz. Dropped while the session is reconnecting."""
        await self.session.socket.send(
            {"type": ClientEvent.INPUT_AUDIO_BUFFER_APPEND, "audio": base64.b64encode(pcm).decode("ascii")}
        )

    async def send_function_result(self, call_id: str, output: str) -> None:
        """Add a tool's result to the conversation, waiting out a reconnect if one is under way."""
        item = {"type": ItemType.FUNCTION_CALL_OUTPUT, "call_id": call_id, "output": output}
        if await self.session.socket.send_reliably({"type": ClientEvent.CONVERSATION_ITEM_CREATE, "item": item}):
            self.history.function_result(call_id, output)
        else:
            self.history.close_call(call_id)

    async def request_response(self) -> None:
        """Have the model respond to the results just sent, unless a response created after now will see them."""
        self.responses.request()
        await self._maybe_create_response()

    async def _maybe_create_response(self) -> None:
        if not self.session.socket.ready or not self.responses.should_create():
            return
        event_id = f"voxkit_{next(self._event_ids)}"
        self.responses.creating(event_id)
        if not await self.session.socket.send({"type": ClientEvent.RESPONSE_CREATE, "event_id": event_id}):
            self.responses.create_failed()

    async def truncate(self, item_id: str, audio_end_ms: int) -> None:
        """Cut a spoken item down to the milliseconds the user heard."""
        self.history.heard(item_id, audio_end_ms)
        await self.session.socket.send(
            {
                "type": ClientEvent.CONVERSATION_ITEM_TRUNCATE,
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            }
        )

    async def delete_item(self, item_id: str) -> None:
        """Remove a spoken item the user heard none of."""
        self.history.remove(item_id)
        await self.session.socket.send({"type": ClientEvent.CONVERSATION_ITEM_DELETE, "item_id": item_id})


def _replay_item(turn: Turn) -> dict:
    match turn.kind:
        case "user":
            return {"type": ItemType.MESSAGE, "role": "user", "content": [{"type": "input_text", "text": turn.text}]}
        case "assistant":
            content = [{"type": "output_text", "text": turn.heard_text()}]
            return {"type": ItemType.MESSAGE, "role": "assistant", "content": content}
        case "function_call":
            return {
                "type": ItemType.FUNCTION_CALL,
                "call_id": turn.call_id,
                "name": turn.name,
                "arguments": turn.arguments,
            }
        case "function_call_output":
            return {"type": ItemType.FUNCTION_CALL_OUTPUT, "call_id": turn.call_id, "output": turn.output}
