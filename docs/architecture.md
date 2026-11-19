# Architecture

voxchain is built from small pieces with one job each. The pipelines and providers wire them to I/O; the pieces themselves are pure and unit-tested on their own.

## Cascaded pipeline

```
audio in ──▶ STTProvider ──▶ VoxchainPipeline ──▶ LangGraph agent
                                  │                    │ streamed tokens
                                  └── sentences ──▶ TTSProvider ──▶ your callback
```

1. **Audio in** goes to the [`STTProvider`][voxchain.stt.base.STTProvider] and to the optional end-of-turn detector.
2. **STT events** ([`SpeechStarted`][voxchain.stt.events.SpeechStarted], [`SpeechStopped`][voxchain.stt.events.SpeechStopped], [`FinalTranscript`][voxchain.stt.events.FinalTranscript]) feed the [`EndOfTurnGate`][voxchain.core.turn_gate.EndOfTurnGate], which decides when the user's turn is complete.
3. **A completed turn** starts an agent reply. It streams to the [`TTSProvider`][voxchain.tts.base.TTSProvider] sentence by sentence ([`SentenceSegmenter`][voxchain.core.text.SentenceSegmenter]), ending with `flush()`. Replies run one at a time, in order.
4. **TTS events** ([`AudioChunk`][voxchain.tts.events.AudioChunk], [`Flushed`][voxchain.tts.events.Flushed]) become `PlayAudio` and `ReplyFinished` for the client. They are delivered from their own task, so a slow callback never delays barge-in.
5. **Barge-in:** on `SpeechStarted` while a reply is active or still playing, [`ReplyTracker`][voxchain.core.replies.ReplyTracker] says what to undo. The reply tasks are cancelled, queued audio is dropped, the client gets `StopPlayback`, and TTS is told to `cancel()`. A reply the user heard for no longer than `resume_window` hands its user text back to the gate, to be merged with what they say next.

Because voxchain bundles no pipeline-level VAD, *when* `SpeechStarted` fires is the STT provider's call. Tune barge-in on the provider: Sarvam's `extra_params` (e.g. `interrupt_min_speech_frames`), or `OpenAISTTProvider`'s `vad_threshold` / `vad_start_ms`.

## Speech-to-speech pipeline

```
audio in ──▶ RealtimeProvider (hears, reasons, speaks) ──▶ RealtimePipeline ──▶ your callback
                      ▲                                          │
                      └──────── tool results ◀── LangChain tools ◀┘
```

1. **Audio in** is resampled to the provider's `input_sample_rate` and sent straight to it. Turn detection and response creation happen on the vendor's side.
2. **Playback** runs in its own task. [`PlaybackTracker`][voxchain.realtime.playback.PlaybackTracker] notes when each chunk reached the client, so it knows how much of every item was heard, even across playback gaps.
3. **Barge-in:** on `SpeechStarted` the active response is discarded and its queued audio dropped, and the client gets `StopPlayback`. The item being heard is truncated to the milliseconds heard; items the user heard none of are deleted.
4. **Tools:** [`ToolRunner`][voxchain.realtime.tools.ToolRunner] runs a response's calls concurrently once it ends. Results go back through `send_function_result`, then `request_response`. [`ResponseGate`][voxchain.realtime.response_gate.ResponseGate] skips that request if a later response already sees the results. Calls from an interrupted response are answered with an error and not run.
5. **Session continuity:** the provider replays its [`ConversationHistory`][voxchain.realtime.history.ConversationHistory] into a fresh session after a drop or before the 60-minute limit.

## Shared pieces

- [`ProviderSession`][voxchain.session.ProviderSession] is each websocket provider's plumbing. The provider passes in `open` and `handle` (plus optional hooks) and gets one event stream back. Underneath it:
    - [`ReconnectingSession`][voxchain.session.ReconnectingSession] reconnects on drops (5 attempts) and treats planned replacements, like a cancel or the 60-minute swap, as free.
    - [`LiveSocket`][voxchain.session.LiveSocket] holds the current socket. Its sends either drop while reconnecting (audio) or wait for the new socket (tool results, text to speak).
- [`OutputChannel`][voxchain.output.OutputChannel] delivers both pipelines' client events from their own task. It models when each chunk plays and tells the pipeline's tracker.
- [`VoiceGate`][voxchain.vad.gate.VoiceGate] turns a local VAD's probabilities into speech segments, for providers whose vendor doesn't report voice activity.
- `voxchain/protocols/` holds each vendor's wire vocabulary: URLs, message types as `StrEnum`s, and handshakes.

## Connection drops and logging

Every provider reconnects on its own; you lose at most the audio in flight. [`SessionClosed`][voxchain.session.SessionClosed] is emitted only when reconnecting fails, and the pipeline then stops and sends `StreamClosed`.

voxchain logs routine activity at `DEBUG` and only real problems at `WARNING`/`ERROR`:

```python
logging.getLogger("voxchain").setLevel(logging.DEBUG)
```
