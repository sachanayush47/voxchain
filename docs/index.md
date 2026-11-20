# voxchain

A thin, event-driven Python library to build real-time voice agents on **LangChain / LangGraph**.

voxchain does the difficult parts of a voice conversation: audio streams in and out, turn-taking, barge-in and reconnects. You supply the "brain". You can build an agent in two ways:

- [`VoxchainPipeline`][voxchain.core.pipeline.VoxchainPipeline] connects streaming speech-to-text, any LangGraph agent and streaming text-to-speech. See [Cascaded pipeline](guides/cascaded.md).
- [`RealtimePipeline`][voxchain.realtime.pipeline.RealtimePipeline] runs one speech-to-speech model. The model hears, reasons, calls your LangChain tools and speaks. See [Speech-to-speech](guides/speech-to-speech.md).

Both pipelines send the same [output events][voxchain.output] to your client, so your playback code works with each of them.

!!! warning "Status"
    Pre-alpha (`0.1.0`). The API will change. Providers: Sarvam AI (STT, TTS) and OpenAI (STT, TTS, speech-to-speech).

## Installation

```bash
pip install voxchain
```

| Extra | Adds | Necessary for |
|---|---|---|
| `vad` | `onnxruntime`, `huggingface-hub`, `numpy` | [`OpenAISTTProvider`][voxchain.stt.openai.OpenAISTTProvider], which runs Silero voice detection locally |
| `smart-turn` | `onnxruntime`, `transformers`, `huggingface-hub`, `numpy` | The [end-of-turn check](guides/cascaded.md#end-of-turn-check) ([`PipecatSmartTurnDetector`][voxchain.turn.smart_turn.PipecatSmartTurnDetector]) |
| `noise` | `pyrnnoise` | The [noise filter](guides/cascaded.md#background-noise) ([`RNNoiseFilter`][voxchain.filters.RNNoiseFilter]) |
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

The [`examples/`](https://github.com/sachanayush47/voxchain/tree/master/examples) folder has complete versions that you can run, with your microphone and speakers. There is one example for each pipeline and provider, and STT-only and TTS-only examples. For example, run `uv run -m examples.sarvam` or `uv run -m examples.realtime_openai`.

## What your callback receives

| Event | Meaning |
|---|---|
| [`PlayAudio(data)`][voxchain.output.PlayAudio] | Play this raw PCM16 audio after all audio that is already in the queue. |
| [`StopPlayback()`][voxchain.output.StopPlayback] | The user interrupted the agent. Stop playback now and discard all queued audio. |
| [`ReplyFinished()`][voxchain.output.ReplyFinished] | The pipeline sent all the audio of the current reply. |
| [`StreamClosed(reason)`][voxchain.output.StreamClosed] | A provider lost its session permanently. The pipeline stops. |

The callback runs in its own task. Thus it can block on playback, and barge-in does not wait for it.

With each pipeline, use headphones or audio capture with echo cancellation. If you do not, the agent can hear itself and interrupt its own reply.

## Logs

voxchain logs routine activity at `DEBUG` (transcripts, replies, voice activity). It logs only real problems at `WARNING` or `ERROR`. To see the log of a conversation, use this code:

```python
logging.getLogger("voxchain").setLevel(logging.DEBUG)
```

## Next steps

- [Providers](guides/providers.md): the settings of each STT, TTS and speech-to-speech provider.
- [Architecture](architecture.md): how turns, barge-in and reconnects work.
- [Contributing](contributing.md): how to add a provider, run the tests and release.
- The API reference: each class and function, from the docstrings.
