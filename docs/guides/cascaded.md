# Cascaded pipeline

[`VoxchainPipeline`][voxchain.core.pipeline.VoxchainPipeline] connects three parts: an STT provider, your LangGraph agent and a TTS provider. Use it when you want full control of the agent: any chat model, middleware, multi-step graphs and checkpointed memory.

```
audio in ──▶ STTProvider ──▶ VoxchainPipeline ──▶ LangGraph agent
                                  │                    │ streamed tokens
                                  └── sentences ──▶ TTSProvider ──▶ your callback
```

## How it works

1. Audio goes to the STT provider and to the optional end-of-turn detector.
2. A final transcript starts an agent turn. The reply streams to TTS **sentence by sentence**, so speech starts before the agent completes the reply. The pipeline speaks only the text of the assistant (never tool output) and removes markdown from it. Replies run one at a time, in order.
3. **Barge-in:** if the user really interrupts a reply, the pipeline cancels the reply. TTS discards its pending work, and your callback receives `StopPlayback`. See [Barge-in](#barge-in).

See [Architecture](../architecture.md#cascaded-pipeline) for the full event flow.

## Settings

```python
VoxchainPipeline(
    stt,
    tts,
    agent,
    callback,
    *,
    thread_id="default",
    greeting=None,
    interrupt=True,
    interrupt_min_words=3,
    resume_window=2.0,
    recovery_prompt=RECOVERY_PROMPT,
    recovery_delay=1.5,
    end_of_turn=None,
    end_of_turn_timeout=2.0,
    audio_filter=None,
)
```

- `agent`: any compiled LangGraph graph, for example from `langchain.agents.create_agent`. Give it a checkpointer (for example `InMemorySaver()`), or it forgets each turn.
- `thread_id`: the key for the checkpointed memory of the agent. Use one `thread_id` for each conversation.
- `greeting`: a sentence that the agent speaks first. The user cannot interrupt it, and the agent's memory starts with it.
- `interrupt`: set `False` to let replies complete. The user then cannot interrupt the agent.
- `interrupt_min_words`, `resume_window`, `recovery_prompt` and `recovery_delay`: see [Barge-in](#barge-in).
- `end_of_turn` and `end_of_turn_timeout`: see [End-of-turn check](#end-of-turn-check).
- `audio_filter`: cleans the microphone audio before STT. See [Background noise](#background-noise).

## Barge-in

The pipeline decides from the **words** of the user, not from voice activity, whether the user interrupts. Voice activity also comes from noise, music, a TV or a cough. Words come only from speech.

| The user says, while they hear the agent | Result |
|---|---|
| Nothing that STT transcribes (noise, music) | The agent continues. |
| Fewer than `interrupt_min_words` words (default 3), for example "uh-huh" or "okay" | The pipeline discards the transcript. The agent continues. |
| A stop word or phrase, for example "stop", "no wait" or "hold on" | Barge-in. The words become the next user turn. |
| `interrupt_min_words` words or more | Barge-in. The words become the next user turn. |

[`InterruptionPolicy`][voxchain.core.interruptions.InterruptionPolicy] makes this decision. More rules:

- STT often splits one sentence into several transcripts. The policy adds up short transcripts that arrive within 2 seconds.
- A partial transcript interrupts as soon as it has enough words, for example from `OpenAISTTProvider(include_partials=True)`. Sarvam STT sends only final transcripts, so a barge-in comes after a short pause in the speech of the user.
- Speech that starts before the user hears the reply always interrupts, with any number of words.
- The user cannot interrupt while a tool call runs, so the tool does not run two times. The pipeline answers the words of the user after the tool call.
- The user cannot interrupt the greeting. The pipeline ignores speech while the greeting plays.
- Set `interrupt_min_words=0` to interrupt on voice activity, as in voxchain 0.1.0.

### Resume

If the user interrupts less than `resume_window` seconds (default 2) after they first hear a reply, the end-of-turn decision was probably too early. The pipeline merges their previous turn with what they say next ("I want a ticket to" + "Bangkok"). Later barge-ins discard the previous turn. Set `resume_window=None` to never merge.

### Memory after a barge-in

After a barge-in, the agent's memory keeps only the part of the reply that the user heard. The pipeline estimates this part from the audio that played. If the turns merge, the memory keeps only the merged turn. Thus the agent never thinks that the user heard something that they did not hear. This needs a checkpointer on the agent.

### Recovery

With voice activity barge-in, a cough can stop the agent and give no transcript. Then the agent waits in silence. After `recovery_delay` seconds (default 1.5) without a transcript, the agent receives `recovery_prompt` and asks the user to say it again. This occurs only one time in a row. Set `recovery_prompt=None` to turn it off.

## Background noise

Words decide barge-in, but noise can still cause wrong transcripts. These settings help:

- `audio_filter=RNNoiseFilter()` removes steady noise (fans, traffic, hum) from the microphone audio before STT. It needs `pip install "voxchain[noise]"`. It removes music and other voices less well.
- `OpenAISTTProvider` ignores audio quieter than `vad_min_volume_db` (default -50 dBFS). For Sarvam STT, set `start_speech_volume_threshold` (dB) in `extra_params`.
- Use headphones, or audio capture with echo cancellation. If you do not, the agent can hear itself.

```python
from voxchain.filters import RNNoiseFilter

pipeline = VoxchainPipeline(stt, tts, agent, play, audio_filter=RNNoiseFilter(sample_rate=16000))
```

## End-of-turn check

Voice detection ends a turn on silence alone. Thus, if the user pauses in the middle of a sentence ("book it for... um..."), the agent answers too early. [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) is a small local audio model. It uses intonation to decide if the user turn is complete:

```python
from voxchain import PipecatSmartTurnDetector

pipeline = VoxchainPipeline(stt, tts, agent, play, end_of_turn=PipecatSmartTurnDetector(sample_rate=16000))
```

If a transcript sounds incomplete, the pipeline holds it for a maximum of `end_of_turn_timeout` seconds of silence. During that time, the pipeline adds new speech to it and scores it again.

- The detector needs `pip install "voxchain[smart-turn]"`.
- Use one detector for each pipeline.
- To write your own detector, implement [`EndOfTurnDetector`][voxchain.turn.base.EndOfTurnDetector].
