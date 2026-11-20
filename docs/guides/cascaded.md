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
3. **Barge-in:** if the user starts to speak during a reply, the pipeline cancels the reply. TTS discards its pending work, and your callback receives `StopPlayback`.

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
    interrupt=True,
    resume_window=2.0,
    end_of_turn=None,
    end_of_turn_timeout=2.0,
)
```

- `agent`: any compiled LangGraph graph, for example from `langchain.agents.create_agent`.
- `thread_id`: the key for the checkpointed memory of the agent. Use one `thread_id` for each conversation.
- `interrupt`: set `False` to let replies complete. The user then cannot interrupt the agent.
- `resume_window`: see [Barge-in and resume](#barge-in-and-resume). Set `None` to never merge turns.
- `end_of_turn` and `end_of_turn_timeout`: see [End-of-turn check](#end-of-turn-check).

## Barge-in and resume

The user can interrupt less than `resume_window` seconds (default 2) after they first hear the reply. In that case, the end-of-turn decision was probably too early. The pipeline then merges their previous turn with what they say next ("I want a ticket to" + "Bangkok"). Later barge-ins ("stop", "ok thanks") discard the previous turn.

voxchain has no pipeline-level VAD, so the STT provider decides when the user starts to speak. Tune barge-in on the provider: the Sarvam `extra_params` (for example `interrupt_min_speech_frames`), or the `vad_threshold` and `vad_start_ms` of `OpenAISTTProvider`. See [Providers](providers.md).

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
