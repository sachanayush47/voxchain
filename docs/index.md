# voxkit

A thin, event-driven Python library for building real-time voice agents on **LangChain / LangGraph**.

voxkit handles the hard parts of a voice conversation (streaming audio in and out, turn-taking, barge-in, reconnects) and leaves the "brain" to you:

- [`VoxkitPipeline`][voxkit.core.pipeline.VoxkitPipeline] chains streaming speech-to-text, any LangGraph agent, and streaming text-to-speech.
- [`RealtimePipeline`][voxkit.realtime.pipeline.RealtimePipeline] runs one speech-to-speech model that hears, reasons, calls your LangChain tools and speaks.

Both send your client the same [output events][voxkit.output]: `PlayAudio`, `StopPlayback`, `ReplyFinished`, `StreamClosed`.

!!! warning "Status"
    Pre-alpha (`0.0.1`); the API will change. Providers: Sarvam AI (STT, TTS) and OpenAI (STT, TTS, speech-to-speech).

## Installation

```bash
pip install voxkit            # core
pip install "voxkit[vad]"     # OpenAISTTProvider (local Silero voice detection)
pip install "voxkit[all]"     # plus the Smart Turn end-of-turn check
```

Requires Python 3.13+.

## Quick start

```python
from voxkit import Output, PlayAudio, SarvamSTTProvider, SarvamTTSProvider, StopPlayback, VoxkitPipeline

stt = SarvamSTTProvider(api_key=SARVAM_API_KEY, language_code="en-IN", sample_rate=16000)
tts = SarvamTTSProvider(api_key=SARVAM_API_KEY, speaker="priya", language_code="en-IN")


async def play(event: Output) -> None:
    match event:
        case PlayAudio(data=data):
            speaker.write(data)      # raw PCM16 at tts.sample_rate
        case StopPlayback():
            speaker.clear()          # the user barged in


pipeline = VoxkitPipeline(stt, tts, agent, play)   # agent: any compiled LangGraph graph
await pipeline.run(microphone_audio_stream())     # PCM16 at stt.sample_rate
```

See the [README](https://github.com/sachanayush47/voxkit#readme) for provider settings and the speech-to-speech example, and [Architecture](architecture.md) for how turns and barge-in work.
