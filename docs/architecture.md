# Architecture

voxchain consists of small pieces, and each piece has one job. The pipelines and providers connect the pieces to I/O. The pieces are pure, and each piece has its own unit tests.

## Cascaded pipeline

```
audio in ──▶ STTProvider ──▶ VoxchainPipeline ──▶ LangGraph agent
                                  │                    │ streamed tokens
                                  └── sentences ──▶ TTSProvider ──▶ your callback
```

1. **Audio in** goes to the [`STTProvider`][voxchain.stt.base.STTProvider] and to the optional end-of-turn detector.
2. **STT events** ([`SpeechStarted`][voxchain.stt.events.SpeechStarted], [`SpeechStopped`][voxchain.stt.events.SpeechStopped], [`FinalTranscript`][voxchain.stt.events.FinalTranscript]) go to the [`EndOfTurnGate`][voxchain.core.turn_gate.EndOfTurnGate]. The gate decides when the user turn is complete.
3. **A complete turn** starts an agent reply. The reply streams to the [`TTSProvider`][voxchain.tts.base.TTSProvider] sentence by sentence ([`SentenceSegmenter`][voxchain.core.text.SentenceSegmenter]) and ends with `flush()`. Replies run one at a time, in order.
4. **TTS events** ([`AudioChunk`][voxchain.tts.events.AudioChunk], [`Flushed`][voxchain.tts.events.Flushed]) become `PlayAudio` and `ReplyFinished` for the client. A separate task delivers them, so a slow callback never delays barge-in.
5. **Barge-in:** `SpeechStarted` can occur while a reply is active or still plays. Then [`ReplyTracker`][voxchain.core.replies.ReplyTracker] tells the pipeline what to undo. The pipeline cancels the reply tasks and discards the queued audio. The client receives `StopPlayback`, and the pipeline tells TTS to `cancel()`. If the user heard a reply for no longer than `resume_window`, the pipeline returns the user text of that reply to the gate. The gate merges it with what the user says next.

voxchain has no pipeline-level VAD, so the STT provider decides *when* `SpeechStarted` occurs. Tune barge-in on the provider: the Sarvam `extra_params` (for example `interrupt_min_speech_frames`), or the `vad_threshold` / `vad_start_ms` of `OpenAISTTProvider`.

## Speech-to-speech pipeline

```
audio in ──▶ RealtimeProvider (hears, reasons, speaks) ──▶ RealtimePipeline ──▶ your callback
                      ▲                                          │
                      └──────── tool results ◀── LangChain tools ◀┘
```

1. **Audio in:** the pipeline resamples the audio to the `input_sample_rate` of the provider and sends it directly to the provider. The vendor does the turn detection and creates the responses.
2. **Playback** runs in its own task. [`PlaybackTracker`][voxchain.realtime.playback.PlaybackTracker] records when each chunk got to the client. Thus it knows how much of each item the user heard, also across gaps in playback.
3. **Barge-in:** on `SpeechStarted`, the pipeline discards the active response and its queued audio, and the client receives `StopPlayback`. The pipeline truncates the current item to the milliseconds that the user heard. It deletes the items that the user did not hear at all.
4. **Tools:** when a response ends, [`ToolRunner`][voxchain.realtime.tools.ToolRunner] runs its tool calls concurrently. The results go back through `send_function_result`, then `request_response`. [`ResponseGate`][voxchain.realtime.response_gate.ResponseGate] skips that request if a later response already sees the results. The pipeline does not run the calls from an interrupted response. It answers them with an error.
5. **Session continuity:** after a connection loss, or before the 60-minute limit, the provider replays its [`ConversationHistory`][voxchain.realtime.history.ConversationHistory] into a new session.

## Shared pieces

- [`ProviderSession`][voxchain.session.ProviderSession] is the plumbing of each websocket provider. The provider gives it `open` and `handle` (and optional hooks) and gets one event stream back. Below it:
    - [`ReconnectingSession`][voxchain.session.ReconnectingSession] reconnects after a connection loss (5 attempts). Planned replacements, such as a cancel or the 60-minute swap, do not use an attempt. A [`SessionRejected`][voxchain.session.SessionRejected] error stops it immediately.
    - [`LiveSocket`][voxchain.session.LiveSocket] holds the current socket. During a reconnect, its sends either discard the data (audio) or wait for the new socket (tool results, text to speak).
- [`OutputChannel`][voxchain.output.OutputChannel] delivers the client events of both pipelines from their own task. It calculates when each chunk plays and tells the tracker of the pipeline.
- [`VoiceGate`][voxchain.vad.gate.VoiceGate] changes the probabilities of a local VAD into speech segments. It is for providers whose vendor does not report voice activity.
- `voxchain/protocols/` contains the wire vocabulary of each vendor: URLs, message types as `StrEnum`s, and handshakes.

## Connection loss and logs

Each provider reconnects automatically. You lose only the audio that is in transit at that time. The provider emits [`SessionClosed`][voxchain.session.SessionClosed] only if the reconnect fails. The pipeline then stops and sends `StreamClosed`.

Some errors cannot change on a reconnect: HTTP 401, 402 or 403 when the provider connects, and the errors that the vendor marks as fatal (for Sarvam: no credits, or a bad parameter). For these errors, the provider does not try again. It emits `SessionClosed` immediately, with the reason of the vendor.

voxchain logs routine activity at `DEBUG`. It logs only real problems at `WARNING` or `ERROR`:

```python
logging.getLogger("voxchain").setLevel(logging.DEBUG)
```
