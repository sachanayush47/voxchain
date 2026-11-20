# voxchain

A thin, event-driven Python library to build real-time voice agents on **LangChain / LangGraph**.

voxchain does the difficult parts of a voice conversation: audio streams in and out, turn-taking, barge-in and reconnects. You supply the "brain":

- [`VoxchainPipeline`][voxchain.core.pipeline.VoxchainPipeline] connects streaming speech-to-text, any LangGraph agent and streaming text-to-speech.
- [`RealtimePipeline`][voxchain.realtime.pipeline.RealtimePipeline] runs one speech-to-speech model. The model hears, reasons, calls your LangChain tools and speaks.

Both pipelines send the same [output events][voxchain.output] to your client: `PlayAudio`, `StopPlayback`, `ReplyFinished`, `StreamClosed`.

!!! warning "Status"
    Pre-alpha (`0.1.0`). The API will change. Providers: Sarvam AI (STT, TTS) and OpenAI (STT, TTS, speech-to-speech).

## Installation

```bash
pip install voxchain               # Core
pip install "voxchain[vad]"        # OpenAISTTProvider (local Silero voice detection)
pip install "voxchain[all]"        # Also the Smart Turn end-of-turn check
```

You must have Python 3.13 or later.

## Quick start

```python
from voxchain import Output, PlayAudio, SarvamSTTProvider, SarvamTTSProvider, StopPlayback, VoxchainPipeline

stt = SarvamSTTProvider(api_key=SARVAM_API_KEY, language_code="en-IN", sample_rate=16000)
tts = SarvamTTSProvider(api_key=SARVAM_API_KEY, speaker="priya", language_code="en-IN")


async def play(event: Output) -> None:
    match event:
        case PlayAudio(data=data):
            speaker.write(data)  # Raw PCM16 at tts.sample_rate
        case StopPlayback():
            speaker.clear()  # The user interrupts the reply


pipeline = VoxchainPipeline(stt, tts, agent, play)  # Agent: any compiled LangGraph graph
await pipeline.run(microphone_audio_stream())  # PCM16 at stt.sample_rate
```

See the [README](https://github.com/sachanayush47/voxchain#readme) for the provider settings and the speech-to-speech example. See [Architecture](architecture.md) for how turns and barge-in work.
