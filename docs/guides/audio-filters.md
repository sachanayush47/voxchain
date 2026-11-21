# Audio filters

Audio filters clean the microphone audio before any provider hears it. They work with both pipelines and with each STT, TTS and speech-to-speech provider.

Words decide barge-in in the [cascaded pipeline](cascaded.md#barge-in). But background sound can still give transcripts, and then the agent hears words that the user did not say. A filter removes that sound before STT.

## Use filters

Give the filters as a list. The pipeline runs them one after another, in the order of the list:

```python
from voxchain.filters import HushFilter, LoudnessGate

pipeline = VoxchainPipeline(
    stt,
    tts,
    agent,
    play,
    audio_filters=[LoudnessGate(), HushFilter(library_path="libweya_nc.dylib")],
)
```

`RealtimePipeline` takes the same `audio_filters` argument. The filters receive audio at `input_sample_rate`.

```
microphone ──▶ filter 1 ──▶ filter 2 ──▶ … ──▶ provider (STT or speech-to-speech)
```

[`FilterChain`][voxchain.filters.base.FilterChain] runs the list:

- The output of each filter is the input of the next filter.
- A filter can keep audio in a buffer and return nothing. Then the chain stops for that chunk.
- A `FilterChain` is also a filter, so you can put a chain in a list.

## The filters

| Filter | It removes | It needs |
|---|---|---|
| [`HushFilter`][voxchain.filters.hush.HushFilter] | Background noise and other voices, for example a TV or a person across the room | `pip install "voxchain[hush]"` and the native `weya_nc` library |
| [`RNNoiseFilter`][voxchain.filters.rnnoise.RNNoiseFilter] | Steady background noise, for example fans, traffic or hum | `pip install "voxchain[noise]"` |
| [`LoudnessGate`][voxchain.filters.loudness.LoudnessGate] | Audio much quieter than the voice of the user, for example a far voice | Nothing |

### HushFilter

[Hush](https://huggingface.co/weya-ai/hush) is an open-source (Apache 2.0) model that keeps only the main speaker. It is a version of DeepFilterNet3 that is trained to remove competing voices. It runs on CPU in real time and adds approximately 20 ms of delay.

Hush runs in a native library, `weya_nc`, from the [Hush project](https://github.com/pulp-vision/Hush). voxchain does not include this library:

1. Get the library for your platform (`libweya_nc.so`, `libweya_nc.dylib` or `weya_nc.dll`). Build it from `native/weya_nc_build` with `cargo build --release`, or use a prebuilt file from the project.
2. Give its path as `library_path`.

The filter downloads the model from Hugging Face the first time. To use a local copy, give `model_path`.

In a test with real speech and a streaming STT provider:

- Hush made a far voice 14 dB quieter than the user 23 dB quieter again. STT did not hear it. The `LoudnessGate` alone cannot remove a voice this loud.
- The voice of the user became only 1 dB quieter, and five short barge-in phrases ("Stop.", "Hold on.", "No, stop.") stayed complete.
- Hush used approximately 2 ms of CPU for each 100 ms of audio.
- A far voice only 8 dB quieter than the user stayed. Very loud background voices are a known limit of the model.
- After Hush removed a far voice, it made the next short word of the user weaker. A single "Stop." then became too short for the voice detection of the STT provider. With `LoudnessGate` before Hush, the word stayed.

### RNNoiseFilter

RNNoise is a small neural network for steady noise. It removes music and other voices less well. It adds approximately 10 ms of delay.

### LoudnessGate

Sound becomes approximately 6 dB quieter each time the distance doubles. Thus the voice of the user near the microphone is louder than other voices in the room. The gate learns the level of the speech of the user. Then it silences audio that is more than `margin_db` (default 12 dB) below that level.

- The gate measures the level of the last `window_ms` (default 200 ms) of audio. Thus single loud syllables of a far voice do not open it.
- In a test with real speech, the gate removed a voice 18 dB quieter than the user (approximately 3 m against 40 cm). The words of the user stayed complete. A voice only 14 dB quieter passed, because it was inside the margin.
- The gate needs no model and almost no CPU.
- It replaces quiet audio with silence of the same length, so the timing does not change.
- It cannot remove a voice that is as loud as the user. For that, use `HushFilter`.
- A small `margin_db` can cut quiet sounds of the user. A large `margin_db` lets more far voices through.
- The level rises slowly at the start of speech, so the gate can cut the first tens of milliseconds of a word. If your STT loses first sounds, set `lookahead_ms=100`. This delays all audio by 100 ms.

## Which filters to use

- Use one denoiser: `HushFilter` or `RNNoiseFilter`. Hush also removes steady noise, so RNNoise after Hush adds little. Two denoisers add delay and can make speech less clear.
- Put `LoudnessGate` before `HushFilter`. Then Hush does not adapt to a far voice that the gate already removed. In a test, this order kept a short "Stop." of the user, and the other orders lost it.
- Test the filters with your STT provider and with real speech. A filter changes the audio, and a short word such as "stop" can then fall below the voice detection of the provider.
- Some STT models are less accurate with denoised audio, because the vendor trained them on noisy audio. Measure with your provider before you turn on a denoiser.
- Use headphones, or audio capture with echo cancellation. A filter does not remove the voice of the agent from the speakers.

Some providers also have their own settings against noise. For example, `OpenAISTTProvider` has `vad_min_volume_db`, Sarvam STT has `start_speech_volume_threshold` in `extra_params`, and `OpenAIRealtimeProvider` has `noise_reduction`. These settings work only with that provider.

## Write a filter

Implement [`AudioFilter`][voxchain.filters.base.AudioFilter]. It has one method:

```python
class MyFilter:
    def process(self, pcm: bytes) -> bytes:
        """Receive PCM16 audio and return PCM16 audio."""
        ...
```

The output can be shorter or longer than the input, because a filter can keep audio in a buffer. Return `b""` while the buffer fills.
