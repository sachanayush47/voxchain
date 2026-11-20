# Providers

Each provider takes keyword arguments. The signature in the API reference is the full list. This page shows the settings that people usually change.

| Provider | Kind | Vendor model |
|---|---|---|
| [`SarvamSTTProvider`][voxchain.stt.sarvam.SarvamSTTProvider] | STT | `saaras`, `saarika` |
| [`SarvamTTSProvider`][voxchain.tts.sarvam.SarvamTTSProvider] | TTS | `bulbul` |
| [`OpenAISTTProvider`][voxchain.stt.openai.OpenAISTTProvider] | STT | `gpt-live-transcribe` |
| [`OpenAITTSProvider`][voxchain.tts.openai.OpenAITTSProvider] | TTS | `gpt-realtime-2.1-mini` |
| [`OpenAIRealtimeProvider`][voxchain.realtime.openai.OpenAIRealtimeProvider] | Speech-to-speech | `gpt-realtime-2.1` |

All providers reconnect automatically when they lose their connection (5 attempts). Your callback receives `StreamClosed` only if all attempts fail. If the vendor rejects the session for a cause that a reconnect cannot fix (a bad API key, no credits, a bad parameter), the provider stops immediately and does not try again.

## SarvamSTTProvider

Uses the Sarvam streaming models `saaras` and `saarika`. The Sarvam voice activity signals control barge-in.

- `model` (`"saaras:v3"`), `mode` (for example `"transcribe"` or `"translate"`), `language_code` (`"unknown"` detects the language automatically).
- `sample_rate`: 8000 or 16000.
- `extra_params` sends all other Sarvam parameters without change. For example, `{"interrupt_min_speech_frames": 4}` makes barge-in less sensitive.
- `await stt.flush()` finalizes the buffered audio immediately. It does not wait for silence. It needs `flush_signal=True`.

## SarvamTTSProvider

Uses the Sarvam `bulbul` voices and streams them as raw PCM16.

- `speaker` (required, and it must belong to the model), `language_code`, `model` (`"bulbul:v3"`).
- `sample_rate`: 8000, 16000, 22050 or 24000.
- `pace`, and options for each model: `temperature` and `dict_id` on v3, `pitch` and `loudness` on v2. If you do not set an option, Sarvam uses its default.
- Sarvam cannot cancel synthesis on a socket. Thus, at a barge-in, the provider replaces the socket and discards all audio that the old socket still sends.

## OpenAISTTProvider

Uses OpenAI realtime transcription (`gpt-live-transcribe`). It needs the `vad` extra (`pip install "voxchain[vad]"`).

- The current OpenAI transcription models have no server-side turn detection. Thus a local Silero VAD decides when speech starts and stops. The provider streams only speech, so OpenAI does not bill you for silence.
- `languages`, `keywords` (terms to recognize), `prompt`.
- `vad_threshold`, `vad_start_ms` and `vad_stop_ms` tune turn detection. Increase `vad_start_ms` if coughs cause a barge-in.
- Transcripts always arrive in the order of speech.

## OpenAITTSProvider

Reads text aloud with `gpt-realtime-2.1-mini`. OpenAI names this model as the replacement for its retired TTS models (`tts-1` and `gpt-4o-mini-tts` will shut down on 2027-01-06).

- `voice` (OpenAI recommends `"marin"` or `"cedar"`), `instructions` (tone and pace), `speed`.
- A realtime model has no "speak this text" call. Thus the provider tells the model to read the text word for word. With `verify_transcript=True` (the default), the provider logs a paraphrase as a warning.

## OpenAIRealtimeProvider

See [Speech-to-speech](speech-to-speech.md#settings).
