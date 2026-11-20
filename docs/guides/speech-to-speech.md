# Speech-to-speech

[`RealtimePipeline`][voxchain.realtime.pipeline.RealtimePipeline] runs a single `gpt-realtime-2.1` session. In that session, the model hears the user, reasons, calls your tools and speaks. This gives the lowest latency and the most natural turn-taking.

The model is the brain: you give it instructions and LangChain tools, not a graph. LangChain only defines and runs the tools. If you need LangGraph features such as middleware, human-in-the-loop or a checkpointer, use the [cascaded pipeline](cascaded.md).

```
audio in ──▶ RealtimeProvider (hears, reasons, speaks) ──▶ RealtimePipeline ──▶ your callback
                      ▲                                          │
                      └──────── tool results ◀── LangChain tools ◀┘
```

## Example

```python
import os

from langchain_core.tools import tool

from voxchain import OpenAIRealtimeProvider, RealtimePipeline


@tool
def book_appointment(pet_name: str, date: str) -> str:
    """Book a vet appointment for a pet on a date (YYYY-MM-DD)."""
    return f"Booked {pet_name} on {date}."


provider = OpenAIRealtimeProvider(
    api_key=os.environ["OPENAI_API_KEY"],
    instructions="# Role and Objective\nYou are a friendly vet clinic receptionist. Keep answers short.",
)
pipeline = RealtimePipeline(provider, play, tools=[book_appointment], input_sample_rate=16000)
await pipeline.run(microphone_audio_stream())
```

The `play` callback is the same as in the [quick start](../index.md#quick-start).

## What the pipeline does

- **Turn-taking:** the OpenAI `semantic_vad` decides when the user turn is complete.
- **Barge-in:** when the user starts to speak, playback stops. The pipeline cuts the conversation to what the user actually heard. It truncates a response that the user heard in part and deletes the responses that the user did not hear. Thus the model never thinks that it said something that the user did not hear.
- **Tools:** tool calls run concurrently, and their results go back to the model. If a tool raises an exception, the model receives the error. The pipeline does not run a tool call from a response that the user interrupted. This is important for bookings.
- **Reconnects:** after a connection loss, or before the 60-minute session limit of OpenAI, the provider replays the conversation into a new session.
- **Transcripts:** pass `on_transcript=async def(role, text)` to receive the `"user"` and `"agent"` transcripts.

## How tool calls work

1. The provider declares your tools to the model in the session configuration.
2. The model asks for a tool call in a response. The pipeline collects the calls until the response ends.
3. [`ToolRunner`][voxchain.realtime.tools.ToolRunner] runs the calls concurrently with `ainvoke`. An error becomes a JSON error for the model, not an exception.
4. The provider sends each result as a `function_call_output` item. Then it asks the model for a new response.

[`ResponseGate`][voxchain.realtime.response_gate.ResponseGate] makes sure that the model never answers twice or speaks over the user.

## Settings

The settings that people usually change on [`OpenAIRealtimeProvider`][voxchain.realtime.openai.OpenAIRealtimeProvider]:

- `model`: `"gpt-realtime-2.1"` or the cheaper `"gpt-realtime-2.1-mini"`.
- `voice`, `instructions`.
- `turn_detection`: `"semantic_vad"` or `"server_vad"`.
- `interrupt_response`: set `False` to let the model complete its response when the user speaks.
- `noise_reduction`: `"near_field"` or `"far_field"`.
- `reasoning_effort`: OpenAI recommends `"low"`.
- `truncation`: for example, `0.8` keeps 80% of the context when OpenAI trims it.

`PipecatSmartTurnDetector` does not apply to this pipeline.

## Background noise

The OpenAI server decides barge-in for this pipeline, from voice activity. Thus the word-based barge-in of the [cascaded pipeline](cascaded.md#barge-in) does not apply. These settings help against noise:

- `noise_reduction="near_field"` (headphones or a close microphone) or `"far_field"` (a laptop microphone).
- `eagerness="low"` with `semantic_vad`, so that the model waits longer before it decides that the user spoke.
- `audio_filter=RNNoiseFilter(sample_rate=16000)` on `RealtimePipeline` removes steady noise before OpenAI hears it. It needs `pip install "voxchain[noise]"`.
