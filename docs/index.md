# voxchain

A thin, event-driven Python library for building real-time voice agents on **LangChain / LangGraph**.

voxchain handles the hard parts of a voice conversation (streaming audio in and out, turn-taking, barge-in, reconnects) and leaves the "brain" to you:

- [`VoxchainPipeline`][voxchain.core.pipeline.VoxchainPipeline] chains streaming speech-to-text, any LangGraph agent, and streaming text-to-speech.
- [`RealtimePipeline`][voxchain.realtime.pipeline.RealtimePipeline] runs one speech-to-speech model that hears, reasons, calls your LangChain tools and speaks.

Both send your client the same [output events][voxchain.output]: `PlayAudio`, `StopPlayback`, `ReplyFinished`, `StreamClosed`.

!!! warning "Status"
    Pre-alpha (`0.1.0`); the API will change. Providers: Sarvam AI (STT, TTS) and OpenAI (STT, TTS, speech-to-speech).

## Installation

```bash
pip install voxchain               # core
pip install "voxchain[vad]"        # OpenAISTTProvider (local Silero voice detection)
pip install "voxchain[all]"        # plus the Smart Turn end-of-turn check
```

Requires Python 3.13+.

## Quick start

```python
from voxchain import Output, PlayAudio, SarvamSTTProvider, SarvamTTSProvider, StopPlayback, VoxchainPipeline

stt = SarvamSTTProvider(api_key=SARVAM_API_KEY, language_code="en-IN", sample_rate=16000)
tts = SarvamTTSProvider(api_key=SARVAM_API_KEY, speaker="priya", language_code="en-IN")


async def play(event: Output) -> None:
    match event:
        case PlayAudio(data=data):
            speaker.write(data)      # raw PCM16 at tts.sample_rate
        case StopPlayback():
            speaker.clear()          # the user barged in


pipeline = VoxchainPipeline(stt, tts, agent, play)   # agent: any compiled LangGraph graph
await pipeline.run(microphone_audio_stream())     # PCM16 at stt.sample_rate
```

See the [README](https://github.com/sachanayush47/voxchain#readme) for provider settings and the speech-to-speech example, and [Architecture](architecture.md) for how turns and barge-in work.
