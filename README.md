# voxkit

A thin, event-driven Python library for building real-time voice agents on **LangChain / LangGraph**.

voxkit handles the hard parts of a voice conversation (streaming audio in and out, turn-taking, barge-in, reconnects) and leaves the "brain" to you. There are two ways to build an agent:

- **`VoxkitPipeline`** chains streaming speech-to-text, any LangGraph agent, and streaming text-to-speech.
- **`RealtimePipeline`** runs one speech-to-speech model that hears, reasons, calls your LangChain tools and speaks.

Both send your client the same small set of events, so playback code works with either.

**Design goals:**
- **Thin.** No transport bus, no call/room management. Providers report voice activity themselves.
- **LangChain/LangGraph-first.** Bring any compiled graph (e.g. from `langchain.agents.create_agent`) or LangChain tools.
- **Explicit events.** Providers and pipelines talk in small typed events, so turns, interrupts and lost sessions are easy to follow.

> **Status:** pre-alpha (`0.0.1`); the API will change. Providers: Sarvam AI (STT, TTS) and OpenAI (STT, TTS, speech-to-speech).

## Installation

```bash
pip install voxkit
```

| Extra | Adds | Needed for |
|---|---|---|
| `vad` | `onnxruntime`, `huggingface-hub`, `numpy` | `OpenAISTTProvider`, which runs Silero voice detection locally |
| `smart-turn` | `onnxruntime`, `transformers`, `huggingface-hub`, `numpy` | [End-of-turn check](#end-of-turn-check) (`PipecatSmartTurnDetector`) |
| `all` | both | everything |

Requires Python 3.13+.

## Quick start

```python
import asyncio
import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from voxkit import Output, PlayAudio, SarvamSTTProvider, SarvamTTSProvider, StopPlayback, VoxkitPipeline

stt = SarvamSTTProvider(api_key=os.environ["SARVAM_API_KEY"], language_code="en-IN", sample_rate=16000)
tts = SarvamTTSProvider(api_key=os.environ["SARVAM_API_KEY"], speaker="priya", language_code="en-IN")
agent = create_agent(model=ChatGroq(model="llama-3.3-70b-versatile"), tools=[])


async def play(event: Output) -> None:
    match event:
        case PlayAudio(data=data):
            speaker.write(data)          # raw PCM16 at tts.sample_rate
        case StopPlayback():
            speaker.clear()              # the user barged in


async def main() -> None:
    pipeline = VoxkitPipeline(stt, tts, agent, play)
    await pipeline.run(microphone_audio_stream())   # PCM16 at stt.sample_rate


asyncio.run(main())
```

[`main.py`](main.py) is a complete, runnable version using `sounddevice` for the microphone and speakers. `VOXKIT_PROVIDER=openai` switches it to OpenAI STT/TTS, and `VOXKIT_PROVIDER=openai-realtime` to speech-to-speech.

### What your callback receives

| Event | Meaning |
|---|---|
| `PlayAudio(data)` | Play this raw PCM16 audio after anything already queued. |
| `StopPlayback()` | The user barged in: stop now and drop anything queued. |
| `ReplyFinished()` | The current reply's audio has all been sent. |
| `StreamClosed(reason)` | A provider's session was lost for good; the pipeline is stopping. |

The callback runs in its own task, so it can block on playback without delaying barge-in.

## Providers

Every provider takes keyword arguments; the signature is the full reference. The settings people usually change:

**`SarvamSTTProvider`** uses Sarvam's streaming `saaras` / `saarika` models, with Sarvam's voice activity signals driving barge-in.
- `model` (`"saaras:v3"`), `mode` (`"transcribe"`, `"translate"`, …), `language_code` (`"unknown"` auto-detects).
- `sample_rate`: 8000 or 16000.
- `extra_params` passes any other Sarvam parameter through as-is, e.g. `{"interrupt_min_speech_frames": 4}` to make barge-in less twitchy.
- `await stt.flush()` finalizes buffered audio without waiting for silence (needs `flush_signal=True`).

**`SarvamTTSProvider`** uses Sarvam's `bulbul` voices, streamed as raw PCM16.
- `speaker` (required; must belong to the model), `language_code`, `model` (`"bulbul:v3"`).
- `sample_rate`: 8000, 16000, 22050 or 24000.
- `pace`, plus model-specific options: `temperature` and `dict_id` on v3, `pitch` and `loudness` on v2. Options left unset use Sarvam's defaults.
- Sarvam can't cancel synthesis on a socket, so on barge-in the provider replaces the socket and drops whatever the old one was still sending.

**`OpenAISTTProvider`** uses OpenAI's realtime transcription (`gpt-live-transcribe`). Needs the `vad` extra.
- OpenAI's current transcription models have no server-side turn detection, so a local Silero VAD decides when the user starts and stops talking. Only speech is streamed, so silence isn't billed.
- `languages`, `keywords` (terms to recognize), `prompt`.
- `vad_threshold`, `vad_start_ms`, `vad_stop_ms` tune turn detection; raise `vad_start_ms` if coughs trigger barge-in.
- Transcripts always arrive in speech order.

**`OpenAITTSProvider`** reads text aloud with `gpt-realtime-2.1-mini`, OpenAI's named replacement for its retired TTS models (`tts-1`, `gpt-4o-mini-tts` shut down 2027-01-06).
- `voice` (`"marin"` or `"cedar"` recommended), `instructions` (tone and pacing), `speed`.
- A realtime model has no "speak this text" call. It is told to read the text verbatim, and with `verify_transcript=True` (the default) a paraphrase is logged as a warning.

All providers reconnect on their own when their connection drops (5 attempts). `StreamClosed` reaches your callback only when that fails.

## How the cascaded pipeline works

```
audio in ──▶ STTProvider ──▶ VoxkitPipeline ──▶ LangGraph agent
                                  │                    │ streamed tokens
                                  └── sentences ──▶ TTSProvider ──▶ your callback
```

1. Audio goes to the STT provider (and the optional end-of-turn detector).
2. A final transcript starts an agent turn. The reply streams to TTS **sentence by sentence**, so speech starts before the agent finishes. Only the assistant's own text is spoken (never tool output), with markdown stripped. Replies run one at a time, in order.
3. **Barge-in:** when the user starts talking over a reply, the reply is cancelled, TTS drops its pending work, and your callback gets `StopPlayback`. If they interrupt within `resume_window` seconds (default 2) of first hearing the reply, the end-of-turn call was probably premature. Their previous turn is then merged with what they say next ("I want a ticket to" + "Bangkok"). Later interruptions ("stop", "ok thanks") drop it. `interrupt=False` lets replies finish.

`VoxkitPipeline(stt, tts, agent, callback, *, thread_id="default", interrupt=True, resume_window=2.0, end_of_turn=None, end_of_turn_timeout=2.0)`. `thread_id` keys the agent's checkpointed memory; use one per conversation.

### End-of-turn check

Voice detection ends a turn on silence alone, so a pause mid-sentence ("book it for... um...") gets answered too early. [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) is a small local audio model that judges from intonation whether the user has finished:

```python
from voxkit import PipecatSmartTurnDetector

pipeline = VoxkitPipeline(stt, tts, agent, play, end_of_turn=PipecatSmartTurnDetector(sample_rate=16000))
```

A transcript that sounds unfinished is held for up to `end_of_turn_timeout` seconds of silence while more speech is appended and re-scored. It needs `pip install "voxkit[smart-turn]"`. Use one detector per pipeline.

## Speech-to-speech (OpenAI Realtime)

`RealtimePipeline` runs a single `gpt-realtime-2.1` session that hears the user, reasons, calls your tools and speaks. This gives the lowest latency and the most natural turn-taking. The model is the brain: you give it instructions and LangChain tools rather than a graph.

```python
from langchain_core.tools import tool

from voxkit import OpenAIRealtimeProvider, RealtimePipeline


@tool
def book_appointment(pet_name: str, date: str) -> str:
    """Book a vet appointment for a pet on a date (YYYY-MM-DD)."""
    return f"Booked {pet_name} on {date}."


provider = OpenAIRealtimeProvider(
    api_key=os.environ["OPENAI_API_KEY"],
    instructions="# Role and Objective\nYou are a friendly vet clinic receptionist. Keep answers short.",
)
pipeline = RealtimePipeline(provider, play, tools=[book_appointment], input_sample_rate=16000)
await pipeline.run(microphone_audio_stream())
```

What it handles:
- **Turn-taking:** OpenAI's `semantic_vad` decides when the user has finished.
- **Barge-in:** when the user starts talking, playback stops and the conversation is cut to what they actually heard. Partly heard replies are truncated, unheard ones deleted, so the model never believes it said something the user missed.
- **Tools:** calls run concurrently and their results go back to the model. If a tool raises, the model gets the error. A call from a reply the user interrupted is not run, which matters for bookings.
- **Reconnects:** after a drop, or before OpenAI's 60-minute session limit, the conversation is replayed into a fresh session.
- **Transcripts:** pass `on_transcript=async def(role, text)` for `"user"` and `"agent"` transcripts.

Common settings: `model` (`"gpt-realtime-2.1"` or the cheaper `-mini`), `voice`, `turn_detection` (`"semantic_vad"` / `"server_vad"`), `interrupt_response`, `noise_reduction` (`"near_field"` / `"far_field"`), `reasoning_effort` (OpenAI suggests `"low"`), and `truncation` (e.g. `0.8` to keep 80% of the context when trimming). `PipecatSmartTurnDetector` doesn't apply here.

With either pipeline, use headphones or echo-cancelled audio capture. Otherwise the agent can hear itself and interrupt its own reply.

## Logging

voxkit logs routine activity at `DEBUG` (transcripts, replies, voice activity) and only real problems at `WARNING`/`ERROR`. To watch a conversation:

```python
logging.getLogger("voxkit").setLevel(logging.DEBUG)
```

## Adding a provider

Implement the interface in `voxkit/stt/base.py`, `voxkit/tts/base.py` or `voxkit/realtime/base.py`. Compose the shared pieces rather than inheriting behaviour:
- `ProviderSession` (`voxkit/session.py`) gives you the socket, reconnects and the event stream. You supply `open` (connect and configure) and `handle` (one message to events).
- `VoiceGate` (`voxkit/vad/gate.py`) if the vendor doesn't report voice activity.
- For speech-to-speech, `ResponseGate` and `ConversationHistory`.

Put the vendor's wire vocabulary (URLs, message types as `StrEnum`s, handshake) in `voxkit/protocols/<vendor>.py`. The existing providers are compact references. The pipelines only depend on the interfaces.

## Documentation

API reference, generated from the docstrings: **[sachanayush47.github.io/voxkit](https://sachanayush47.github.io/voxkit/)**. Build it locally with `uv sync --extra docs && uv run mkdocs serve`.

## Development

```bash
uv sync                     # dependencies and dev tooling
uv run pytest               # test suite: no network or credentials needed
uv run ruff check --fix .   # lint
uv run ruff format .        # format
uv run pre-commit install   # once per clone: lint and format on commit
```

Contributor conventions (coding standards, design principles, testing) are in [`AGENTS.md`](AGENTS.md).

## License

TBD.
