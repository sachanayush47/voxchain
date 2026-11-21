# voxchain

[![PyPI](https://img.shields.io/pypi/v/voxchain)](https://pypi.org/project/voxchain/)
[![Python](https://img.shields.io/pypi/pyversions/voxchain)](https://pypi.org/project/voxchain/)
[![License](https://img.shields.io/pypi/l/voxchain)](https://github.com/sachanayush47/voxchain/blob/master/LICENSE)

A thin, event-driven Python library to build real-time voice agents on **LangChain / LangGraph**.

voxchain does the difficult parts of a voice conversation: audio streams in and out, turn-taking, barge-in and reconnects. You supply the "brain". You can build an agent in two ways:

- **`VoxchainPipeline`** connects streaming speech-to-text, any LangGraph agent and streaming text-to-speech.
- **`RealtimePipeline`** runs one speech-to-speech model. The model hears, reasons, calls your LangChain tools and speaks.

Both pipelines send the same small set of events to your client, so your playback code works with each of them.

**Design goals:**
- **Thin.** No transport bus, no call or room management. Providers report voice activity themselves.
- **LangChain/LangGraph first.** Use any compiled graph (for example, from `langchain.agents.create_agent`) or LangChain tools.
- **Explicit events.** Providers and pipelines send small typed events, so it is easy to follow turns, barge-ins and lost sessions.

> **Status:** pre-alpha (`0.1.0`). The API will change. Providers: Sarvam AI (STT, TTS) and OpenAI (STT, TTS, speech-to-speech).

## Installation

```bash
pip install voxchain
```

| Extra | Adds | Necessary for |
|---|---|---|
| `vad` | `onnxruntime`, `huggingface-hub`, `numpy` | `OpenAISTTProvider`, which runs Silero voice detection locally |
| `smart-turn` | `onnxruntime`, `transformers`, `huggingface-hub`, `numpy` | [End-of-turn check](#end-of-turn-check) (`PipecatSmartTurnDetector`) |
| `noise` | `pyrnnoise` | [RNNoise filter](#audio-filters) (`RNNoiseFilter`) |
| `hush` | `huggingface-hub`, `numpy` | [Hush filter](#audio-filters) (`HushFilter`) |
| `all` | all of the above | all features |

You must have Python 3.13 or later.

## Quick start

The agent can use any LangChain chat model. This example uses Groq (`pip install langchain-groq`).

```python
import asyncio
import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from voxchain import Output, PlayAudio, SarvamSTTProvider, SarvamTTSProvider, StopPlayback, VoxchainPipeline

stt = SarvamSTTProvider(api_key=os.environ["SARVAM_API_KEY"], language_code="en-IN", sample_rate=16000)
tts = SarvamTTSProvider(api_key=os.environ["SARVAM_API_KEY"], speaker="priya", language_code="en-IN")
agent = create_agent(model=ChatGroq(model="llama-3.3-70b-versatile"), tools=[])


async def play(event: Output) -> None:
    match event:
        case PlayAudio(data=data):
            speaker.write(data)  # Raw PCM16 at tts.sample_rate
        case StopPlayback():
            speaker.clear()  # The user interrupts the reply


async def main() -> None:
    pipeline = VoxchainPipeline(stt, tts, agent, play)
    await pipeline.run(microphone_audio_stream())  # PCM16 at stt.sample_rate


asyncio.run(main())
```

The [`examples/`](https://github.com/sachanayush47/voxchain/tree/master/examples) folder has complete versions that you can run, with your microphone and speakers. There is one example for each pipeline and provider, one example with all the features (`sarvam_full.py`), and STT-only and TTS-only examples. For example, run `uv run -m examples.sarvam` or `uv run -m examples.realtime_openai`.

### What your callback receives

| Event | Meaning |
|---|---|
| `PlayAudio(data)` | Play this raw PCM16 audio after all audio that is already in the queue. |
| `StopPlayback()` | The user interrupted the agent. Stop playback now and discard all queued audio. |
| `ReplyFinished()` | The pipeline sent all the audio of the current reply. |
| `StreamClosed(reason)` | A provider lost its session permanently. The pipeline stops. |

The callback runs in its own task. Thus it can block on playback, and barge-in does not wait for it.

## Providers

Each provider takes keyword arguments. The signature is the full reference. These are the settings that people usually change:

**`SarvamSTTProvider`** uses the Sarvam streaming models `saaras` and `saarika`. The Sarvam voice activity signals control barge-in.
- `model` (`"saaras:v3"`), `mode` (for example `"transcribe"` or `"translate"`), `language_code` (`"unknown"` detects the language automatically).
- `sample_rate`: 8000 or 16000.
- `extra_params` sends all other Sarvam parameters without change. For example, `{"interrupt_min_speech_frames": 4}` makes barge-in less sensitive.
- `await stt.flush()` finalizes the buffered audio immediately. It does not wait for silence. It needs `flush_signal=True`.

**`SarvamTTSProvider`** uses the Sarvam `bulbul` voices and streams them as raw PCM16.
- `speaker` (required, and it must belong to the model), `language_code`, `model` (`"bulbul:v3"`).
- `sample_rate`: 8000, 16000, 22050 or 24000.
- `pace`, and options for each model: `temperature` and `dict_id` on v3, `pitch` and `loudness` on v2. If you do not set an option, Sarvam uses its default.
- Sarvam cannot cancel synthesis on a socket. Thus, at a barge-in, the provider replaces the socket and discards all audio that the old socket still sends.

**`OpenAISTTProvider`** uses OpenAI realtime transcription (`gpt-live-transcribe`). It needs the `vad` extra.
- The current OpenAI transcription models have no server-side turn detection. Thus a local Silero VAD decides when speech starts and stops. The provider streams only speech, so OpenAI does not bill you for silence.
- `languages`, `keywords` (terms to recognize), `prompt`.
- `vad_threshold` (default 0.7), `vad_start_ms` and `vad_stop_ms` tune turn detection. Increase `vad_start_ms` if coughs cause a barge-in.
- `vad_min_volume_db` (default -50 dBFS) ignores quiet audio, for example a TV in the room. Set `None` to turn it off.
- `noise_reduction` (`"near_field"` or `"far_field"`) makes OpenAI remove noise before it transcribes.
- Transcripts always arrive in the order of speech.

**`OpenAITTSProvider`** reads text aloud with `gpt-realtime-2.1-mini`. OpenAI names this model as the replacement for its retired TTS models (`tts-1` and `gpt-4o-mini-tts` will shut down on 2027-01-06).
- `voice` (OpenAI recommends `"marin"` or `"cedar"`), `instructions` (tone and pace), `speed`.
- A realtime model has no "speak this text" call. Thus the provider tells the model to read the text word for word. With `verify_transcript=True` (the default), the provider logs a paraphrase as a warning.

All providers reconnect automatically when they lose their connection (5 attempts). Your callback receives `StreamClosed` only if all attempts fail. If the vendor rejects the session for a cause that a reconnect cannot fix (a bad API key, no credits, a bad parameter), the provider stops immediately and does not try again.

## How the cascaded pipeline works

```
audio in ──▶ STTProvider ──▶ VoxchainPipeline ──▶ LangGraph agent
                                  │                    │ streamed tokens
                                  └── sentences ──▶ TTSProvider ──▶ your callback
```

1. Audio goes to the STT provider and to the optional end-of-turn detector.
2. A final transcript starts an agent turn. The reply streams to TTS **sentence by sentence**, so speech starts before the agent completes the reply. The pipeline speaks only the text of the assistant (never tool output) and removes markdown from it. Replies run one at a time, in order.
3. **Barge-in:** see below.

`thread_id` is the key for the checkpointed memory of the agent. Use one `thread_id` for each conversation. `greeting` is a sentence that the agent speaks first, for example `greeting="Hi, this is the clinic. How can I help?"`.

### Barge-in

The pipeline decides from the **words** of the user, not from voice activity, whether the user interrupts the agent. Thus noise, music and "uh-huh" do not stop the agent:

- While the user hears the agent, the user interrupts only with `interrupt_min_words` words (default 3) or a stop word, for example "stop", "wait" or "no". The pipeline discards shorter transcripts.
- The pipeline adds up short transcripts that arrive within 2 seconds, because STT often splits one sentence.
- Speech that starts before the user hears the reply always interrupts. If the user interrupts less than `resume_window` seconds (default 2) after they first hear a reply, the end-of-turn decision was probably too early. The pipeline then merges their previous turn with what they say next ("I want a ticket to" + "Bangkok").
- The user cannot interrupt while a tool call runs, so a booking does not happen two times. The pipeline answers their words after the tool call.
- The user cannot interrupt the `greeting`. The pipeline ignores speech while the greeting plays.
- After a barge-in, the agent's memory keeps only the part of the reply that the user heard.
- If a barge-in gives no transcript (for example, a cough), the agent asks the user to say it again after `recovery_delay` seconds (default 1.5). Set `recovery_prompt=None` to turn this off.

Set `interrupt_min_words=0` to interrupt on voice activity, or `interrupt=False` to let replies complete. A partial transcript (for example from `OpenAISTTProvider(include_partials=True)`) interrupts as soon as it has enough words. Sarvam STT sends only final transcripts, so a barge-in comes after a short pause in the speech of the user.

### Audio filters

Words decide barge-in, but background sound can still give transcripts. A TV or a person across the room can then interrupt the agent. Audio filters remove that sound before any provider hears it. They work with both pipelines and with each provider. The pipeline runs them one after another, in the order of the list:

```python
from voxchain.filters import HushFilter, LoudnessGate

pipeline = VoxchainPipeline(
    stt, tts, agent, play, audio_filters=[LoudnessGate(), HushFilter(library_path="libweya_nc.dylib")]
)
```

| Filter | It removes | It needs |
|---|---|---|
| `HushFilter` | Background noise and other voices (a TV, a person across the room) | `pip install "voxchain[hush]"` and the native `weya_nc` library of the [Hush project](https://github.com/pulp-vision/Hush) |
| `RNNoiseFilter` | Steady background noise (fans, traffic, hum) | `pip install "voxchain[noise]"` |
| `LoudnessGate` | Audio much quieter than the voice of the user, for example a far voice | Nothing |

Use one denoiser (`HushFilter` or `RNNoiseFilter`), and put `LoudnessGate` before it. To write your own filter, implement `AudioFilter`: one method, `process(pcm) -> bytes`. Use headphones, or audio capture with echo cancellation. If you do not, the agent can hear itself.

### Session report

After a call, `pipeline.report()` gives the transcript, a timeline of events, the latency of each turn divided into stages, tool calls with their arguments and results, errors, a summary (barge-ins, talk time, latency percentiles), the usage of each provider, and the audio: the input before and after the filters, and the agent audio that played:

```python
pipeline = VoxchainPipeline(stt, tts, agent, play)
try:
    await pipeline.run(microphone())
finally:
    report = pipeline.report()
    print(report.transcript())
    print(report.waterfall())  # The latency stages of each turn as text bars
    Path("call.wav").write_bytes(report.audio.stereo_wav())  # The user on the left, the agent on the right
```

The report and its parts are dataclasses, so you use typed attributes (`turn.latency.llm`, `turn.heard_text`) and not JSON keys. `report.to_dict()` gives a dict for `json.dumps`. See the [session report guide](https://sachanayush47.github.io/voxchain/guides/session-report/).

### End-of-turn check

Voice detection ends a turn on silence alone. Thus, if the user pauses in the middle of a sentence ("book it for... um..."), the agent answers too early. [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) is a small local audio model. It uses intonation to decide if the user turn is complete:

```python
from voxchain import PipecatSmartTurnDetector

pipeline = VoxchainPipeline(stt, tts, agent, play, end_of_turn=PipecatSmartTurnDetector(sample_rate=16000))
```

If a transcript sounds incomplete, the pipeline holds it for a maximum of `end_of_turn_timeout` seconds of silence. During that time, the pipeline adds new speech to it and scores it again. The detector needs `pip install "voxchain[smart-turn]"`. Use one detector for each pipeline.

## Speech-to-speech (OpenAI Realtime)

`RealtimePipeline` runs a single `gpt-realtime-2.1` session. In that session, the model hears the user, reasons, calls your tools and speaks. This gives the lowest latency and the most natural turn-taking. The model is the brain: you give it instructions and LangChain tools, not a graph.

```python
import os

from langchain_core.tools import tool

from voxchain import OpenAIRealtimeProvider, RealtimePipeline


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

What the pipeline does:
- **Turn-taking:** the OpenAI `semantic_vad` decides when the user turn is complete.
- **Barge-in:** when the user starts to speak, playback stops. The pipeline cuts the conversation to what the user actually heard. It truncates a response that the user heard in part and deletes the responses that the user did not hear. Thus the model never thinks that it said something that the user did not hear.
- **Tools:** tool calls run concurrently, and their results go back to the model. If a tool raises an exception, the model receives the error. The pipeline does not run a tool call from a response that the user interrupted. This is important for bookings.
- **Reconnects:** after a connection loss, or before the 60-minute session limit of OpenAI, the provider replays the conversation into a new session.
- **Transcripts:** pass `on_transcript=async def(role, text)` to receive the `"user"` and `"agent"` transcripts.

Common settings: `model` (`"gpt-realtime-2.1"` or the cheaper `-mini`), `voice`, `turn_detection` (`"semantic_vad"` / `"server_vad"`), `interrupt_response`, `noise_reduction` (`"near_field"` / `"far_field"`), `reasoning_effort` (OpenAI recommends `"low"`), and `truncation` (for example, `0.8` keeps 80% of the context when OpenAI trims it). `PipecatSmartTurnDetector` does not apply to this pipeline.

`RealtimePipeline` also takes `audio_filters` (see [Audio filters](#audio-filters)). The server of the provider decides barge-in for this pipeline, from voice activity. Thus a filter is the main protection against noise. `OpenAIRealtimeProvider` also has `noise_reduction` and `eagerness` (for example `"low"`).

## Logs

voxchain logs routine activity at `DEBUG` (transcripts, replies, voice activity). It logs only real problems at `WARNING` or `ERROR`. To see the log of a conversation, use this code:

```python
logging.getLogger("voxchain").setLevel(logging.DEBUG)
```

## How to add a provider

Implement the interface in `voxchain/stt/base.py`, `voxchain/tts/base.py` or `voxchain/realtime/base.py`. Compose the shared pieces. Do not inherit behaviour from them:
- `ProviderSession` (`voxchain/session.py`) gives you the socket, the reconnects and the event stream. You supply `open` (connect and configure) and `handle` (change one message into events).
- `VoiceGate` (`voxchain/vad/gate.py`), if the vendor does not report voice activity.
- `ResponseGate` and `ConversationHistory`, for speech-to-speech.

Put the wire vocabulary of the vendor (URLs, message types as `StrEnum`s, handshake) in `voxchain/protocols/<vendor>.py`. The current providers are compact references. The pipelines depend only on the interfaces.

## Documentation

The docs site has guides for each pipeline and provider, the architecture, and an API reference from the docstrings: **[sachanayush47.github.io/voxchain](https://sachanayush47.github.io/voxchain/)**. To build it locally, run `uv sync --extra docs && uv run mkdocs serve`.

## Development

```bash
uv sync                     # Install the dependencies and dev tools
uv run pytest               # Run the tests (no network or credentials necessary)
uv run ruff check --fix .   # Lint
uv run ruff format .        # Format
uv run pre-commit install   # Do this once for each clone: lint and format at each commit
```

To release, increase `version` in `pyproject.toml` and `__version__` in `voxchain/__init__.py`. Then publish a GitHub release with the tag `v<version>`. The release workflow tests the package, builds it and uploads it to PyPI through trusted publishing.

The contributor conventions (coding standards, design principles, tests) are in [`AGENTS.md`](https://github.com/sachanayush47/voxchain/blob/master/AGENTS.md).

## License

MIT. See [`LICENSE`](https://github.com/sachanayush47/voxchain/blob/master/LICENSE).
