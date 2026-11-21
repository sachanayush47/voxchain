# Session report

After a call, [`VoxchainPipeline.report()`][voxchain.core.pipeline.VoxchainPipeline.report] gives a [`SessionReport`][voxchain.report.models.SessionReport]. It contains:

- The **turns**: what the user said, what the agent replied, what the user heard of an interrupted reply, the tool calls and the token usage.
- The **latency of each turn**, divided into stages that add up to the total.
- A **timeline** of the events of the session.
- The **usage** of each provider.
- The **audio**: the input before and after the audio filters, and the agent audio that played.

The pipeline measures all of this from the provider interfaces. Thus the report works with each STT, TTS and agent.

## Get the report

```python
pipeline = VoxchainPipeline(stt, tts, agent, play)
try:
    await pipeline.run(microphone())
finally:
    report = pipeline.report()
    print(report.transcript())
    print(report.waterfall())
    Path("report.json").write_text(json.dumps(report.to_dict(), indent=2))
    Path("call.wav").write_bytes(report.audio.stereo_wav())
```

You can call `report()` during the call too. It gives a copy of the session until that time. The `sarvam_full` example saves the report and the audio of each call in `sessions/`.

All times are seconds from the start of the session. `started_at` is the Unix time of the start. `close_reason` tells why the session ended:

| `close_reason` | Cause |
|---|---|
| `Audio stream ended` | The input audio stream ended. |
| `STT: <reason>` or `TTS: <reason>` | A provider lost its session and could not reconnect. |
| `Stopped` | The task stopped or failed, for example after Ctrl+C. |

## Typed models

The report and all its parts are dataclasses. Use their attributes, not JSON keys, so your editor and type checker know each field:

```python
for turn in report.turns:
    if turn.interrupted:
        print(turn.user_text, "->", turn.heard_text)
    print(turn.latency.llm, turn.latency.tts, turn.e2e_latency)
```

`report.to_dict()` gives the report as a dict for `json.dumps`, with times rounded to milliseconds. Each event in it has its class name as `type`, for example `{"type": "TurnCommitted", "at": 3.0, "text": "What time is it?"}`.

## Full example

This is the complete `to_dict()` of a short call:

1. The agent says the greeting.
2. The user asks "What time is it?". The agent calls a tool and answers.
3. During the answer, the user says "Okay." The pipeline ignores it, because it is too short to interrupt the agent.
4. The user says "Can you keep it shorter?". This interrupts the agent, and the agent replies to the new turn.

```json
{
  "started_at": 1795271579.652,
  "duration": 13.25,
  "close_reason": "Audio stream ended",
  "thread_id": "call-42",
  "providers": {
    "stt": "SarvamSTTProvider",
    "tts": "SarvamTTSProvider",
    "agent": "LangGraph"
  },
  "settings": {
    "greeting": "Hi! I am your voice assistant. How can I help you today?",
    "interrupt": true,
    "interrupt_min_words": 3,
    "resume_window": 2.0,
    "recovery": true,
    "recovery_delay": 1.5,
    "end_of_turn": null,
    "end_of_turn_timeout": 2.0,
    "audio_filters": [
      "LoudnessGate"
    ]
  },
  "summary": {
    "user_turns": 2,
    "barge_ins": 1,
    "ignored_transcripts": 1,
    "tool_calls": 1,
    "errors": 0,
    "user_speech_seconds": 2.4,
    "agent_speech_seconds": 8.68,
    "longest_reply_seconds": 3.88,
    "latency": {
      "e2e": {"count": 2, "mean": 1.125, "p50": 0.96, "p95": 1.29, "max": 1.29},
      "transcription": {"count": 2, "mean": 0.2, "p50": 0.18, "p95": 0.22, "max": 0.22},
      "end_of_turn": {"count": 2, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
      "waiting": {"count": 2, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
      "tools": {"count": 1, "mean": 0.4, "p50": 0.4, "p95": 0.4, "max": 0.4},
      "llm": {"count": 2, "mean": 0.425, "p50": 0.35, "p95": 0.5, "max": 0.5},
      "sentence": {"count": 2, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
      "tts": {"count": 2, "mean": 0.3, "p50": 0.28, "p95": 0.32, "max": 0.32},
      "playback": {"count": 2, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    }
  },
  "turns": [
    {
      "kind": "greeting",
      "user_text": "",
      "committed_at": 0.0,
      "user_started_at": null,
      "user_stopped_at": null,
      "transcript_at": null,
      "agent_started_at": null,
      "first_token_at": null,
      "first_sentence_at": 0.0,
      "first_audio_at": 0.0,
      "playback_started_at": 0.0,
      "playback_ended_at": 3.2,
      "reply_text": "Hi! I am your voice assistant. How can I help you today?",
      "interrupted": false,
      "heard_text": null,
      "tool_calls": [],
      "input_tokens": 0,
      "output_tokens": 0,
      "error": null,
      "latency": {
        "transcription": null,
        "end_of_turn": null,
        "waiting": null,
        "tools": null,
        "llm": null,
        "sentence": null,
        "tts": null,
        "playback": null
      },
      "e2e_latency": null
    },
    {
      "kind": "user",
      "user_text": "What time is it?",
      "committed_at": 5.52,
      "user_started_at": 4.1,
      "user_stopped_at": 5.3,
      "transcript_at": 5.52,
      "agent_started_at": 5.52,
      "first_token_at": 6.27,
      "first_sentence_at": 6.27,
      "first_audio_at": 6.59,
      "playback_started_at": 6.59,
      "playback_ended_at": 10.47,
      "reply_text": "It is 3:40 in the afternoon. Is there anything else I can help you with today?",
      "interrupted": true,
      "heard_text": "It is 3:40 in the afternoon. Is there anything else I can",
      "tool_calls": [
        {
          "name": "get_current_time",
          "call_id": "call_1",
          "started_at": 5.52,
          "ended_at": 5.92,
          "arguments": {
            "timezone": "Asia/Kolkata"
          },
          "result": "Saturday, 21 November 2026, 03:40 PM",
          "error": false
        }
      ],
      "input_tokens": 472,
      "output_tokens": 21,
      "error": null,
      "latency": {
        "transcription": 0.22,
        "end_of_turn": 0.0,
        "waiting": 0.0,
        "tools": 0.4,
        "llm": 0.35,
        "sentence": 0.0,
        "tts": 0.32,
        "playback": 0.0
      },
      "e2e_latency": 1.29
    },
    {
      "kind": "user",
      "user_text": "Can you keep it shorter?",
      "committed_at": 10.47,
      "user_started_at": 9.39,
      "user_stopped_at": 10.29,
      "transcript_at": 10.47,
      "agent_started_at": 10.47,
      "first_token_at": 10.97,
      "first_sentence_at": 10.97,
      "first_audio_at": 11.25,
      "playback_started_at": 11.25,
      "playback_ended_at": 12.85,
      "reply_text": "Sure, I will keep it short.",
      "interrupted": false,
      "heard_text": null,
      "tool_calls": [],
      "input_tokens": 532,
      "output_tokens": 21,
      "error": null,
      "latency": {
        "transcription": 0.18,
        "end_of_turn": 0.0,
        "waiting": 0.0,
        "tools": null,
        "llm": 0.5,
        "sentence": 0.0,
        "tts": 0.28,
        "playback": 0.0
      },
      "e2e_latency": 0.96
    }
  ],
  "events": [
    {"type": "AgentStartedSpeaking", "at": 0.0},
    {"type": "AgentStoppedSpeaking", "at": 3.2},
    {"type": "UserSpeechStarted", "at": 4.1},
    {"type": "PartialTranscriptReceived", "at": 4.6, "text": "what time"},
    {"type": "UserSpeechStopped", "at": 5.3},
    {"type": "FinalTranscriptReceived", "at": 5.52, "text": "What time is it?"},
    {"type": "TurnCommitted", "at": 5.52, "text": "What time is it?"},
    {"type": "ToolStarted", "at": 5.52, "name": "get_current_time", "call_id": "call_1"},
    {"type": "ToolFinished", "at": 5.92, "name": "get_current_time", "call_id": "call_1", "error": false},
    {"type": "AgentStartedSpeaking", "at": 6.59},
    {"type": "UserSpeechStarted", "at": 7.49},
    {"type": "UserSpeechStopped", "at": 7.79},
    {"type": "FinalTranscriptReceived", "at": 7.79, "text": "Okay."},
    {"type": "TranscriptIgnored", "at": 7.79, "text": "Okay."},
    {"type": "UserSpeechStarted", "at": 9.39},
    {"type": "PartialTranscriptReceived", "at": 9.79, "text": "can you"},
    {"type": "UserSpeechStopped", "at": 10.29},
    {"type": "FinalTranscriptReceived", "at": 10.47, "text": "Can you keep it shorter?"},
    {"type": "BargeIn", "at": 10.47, "heard_text": "It is 3:40 in the afternoon. Is there anything else I can"},
    {"type": "TurnCommitted", "at": 10.47, "text": "Can you keep it shorter?"},
    {"type": "AgentStoppedSpeaking", "at": 10.47},
    {"type": "AgentStartedSpeaking", "at": 11.25},
    {"type": "AgentStoppedSpeaking", "at": 12.85},
    {"type": "SessionEnded", "at": 13.25, "reason": "Audio stream ended"}
  ],
  "usage": {
    "stt_audio_seconds": 13.2,
    "tts_characters": 160,
    "llm_input_tokens": 1004,
    "llm_output_tokens": 42
  },
  "audio": {
    "input_sample_rate": 16000,
    "agent_sample_rate": 24000,
    "raw_seconds": 13.2,
    "filtered_seconds": 13.2,
    "agent_seconds": 12.85,
    "raw": "UklGRiRyBgBXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQByBgAAAAAA...",
    "filtered": "UklGRiRyBgBXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQByBgAAAAAA...",
    "agent": "UklGRoRpCQBXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YWBpCQAAAAAA..."
  }
}
```

The three base64 strings are shortened here.

Some things to see in it:

- `summary` gives the main numbers. The user spoke for 2.4 seconds, and the agent audio played for 8.68 seconds.
- `settings` are the settings of the pipeline for this call.
- The greeting is a turn of kind `greeting`, with no latency.
- The second turn has `interrupted: true`. Its `heard_text` is the part of the reply that played before the barge-in, and its `playback_ended_at` is the time of the barge-in.
- The tool call has the `arguments` from the agent and the `result` of the tool. Its time is the `tools` stage of the latency.
- The third turn called no tool, so its `latency.tools` is `null`.
- "Okay." shows as `FinalTranscriptReceived` and then `TranscriptIgnored`, with no `TurnCommitted`.
- `audio` gives the length of each track, and each track as a WAV file in base64.

## Turns

Each [`Turn`][voxchain.report.models.Turn] is one committed user turn and the reply of the agent. `kind` is a [`TurnKind`][voxchain.report.models.TurnKind]: `user`, `greeting` or `recovery` (the agent asks the user to say it again after a barge-in without words).

- `user_started_at` and `user_stopped_at` are the start of the first speech and the end of the last speech of the user turn.
- `reply_text` is the text that the agent sent to TTS.
- `playback_started_at` and `playback_ended_at` are the start and the end of the reply audio. A barge-in or the end of the session cuts the end. `reply_seconds` is the time between them.
- If the user interrupted the reply, `interrupted` is `True`, and `heard_text` is the part that the user heard. This is the same text that the agent memory keeps.
- `tool_calls` give the name, the call ID, the start and end time, the `arguments` from the agent, the `result` of the tool, and `error` if the tool failed. Streamed arguments that are not valid JSON stay as text.
- `input_tokens` and `output_tokens` come from the `usage_metadata` of the LangChain messages. Some chat models send usage only if you turn it on.
- `error` is the error of the agent if the reply failed, for example `"RuntimeError: Model down"`.

## Latency

[`Turn.latency`][voxchain.report.models.Turn.latency] is a [`Latency`][voxchain.report.models.Latency] model. It divides the time from the end of user speech to the first reply audio into stages. Each stage starts where the stage before it ends, so the stages add up to `e2e_latency`:

| Stage | From | To |
|---|---|---|
| `transcription` | The end of user speech | The final transcript |
| `end_of_turn` | The final transcript | The commit of the turn (the end-of-turn detector waits) |
| `waiting` | The commit | The start of the agent (it waits for the previous reply or a memory correction) |
| `tools` | Tool calls before the first text of the agent | |
| `llm` | The rest of the time until the first text of the agent | |
| `sentence` | The first text | The first complete sentence |
| `tts` | The first sentence to TTS | The first TTS audio |
| `playback` | The first TTS audio | The start of playback |

A stage is `None` if the turn did not get to it, for example `tts` of a reply that the user interrupted before its audio. `tools` is `None` if the agent called no tool. `latency.stages()` gives the stages that occurred, in order.

`waterfall()` shows the stages as text. Each bar starts where the bar before it ends:

```
Turn 2: 'What time is it?'  e2e 1.29s
  transcription  0.220s ████
  end_of_turn    0.000s
  waiting        0.000s
  tools          0.400s     ████████
  llm            0.350s             ███████
  sentence       0.000s
  tts            0.320s                    ██████
  playback       0.000s
```

The end of user speech comes from the voice activity of the STT provider. If a provider sends the final transcript before the end of speech, the latency counts from the transcript. A greeting has no latency.

## Timeline

`events` is a list of event dataclasses. Each event has its time `at`:

| Event | Fields |
|---|---|
| [`UserSpeechStarted`][voxchain.report.models.UserSpeechStarted] | |
| [`UserSpeechStopped`][voxchain.report.models.UserSpeechStopped] | |
| [`PartialTranscriptReceived`][voxchain.report.models.PartialTranscriptReceived] | `text` |
| [`FinalTranscriptReceived`][voxchain.report.models.FinalTranscriptReceived] | `text` |
| [`TranscriptIgnored`][voxchain.report.models.TranscriptIgnored] | `text`: too short to interrupt the agent, or during the greeting |
| [`TurnCommitted`][voxchain.report.models.TurnCommitted] | `text`: the user turn that the agent received |
| [`ToolStarted`][voxchain.report.models.ToolStarted] | `name`, `call_id` |
| [`ToolFinished`][voxchain.report.models.ToolFinished] | `name`, `call_id`, `error` |
| [`AgentStartedSpeaking`][voxchain.report.models.AgentStartedSpeaking] | |
| [`AgentStoppedSpeaking`][voxchain.report.models.AgentStoppedSpeaking] | The reply audio played to the end, or a barge-in cut it |
| [`BargeIn`][voxchain.report.models.BargeIn] | `heard_text`: the text of the reply that the user heard |
| [`ErrorOccurred`][voxchain.report.models.ErrorOccurred] | `source` (`agent`, `memory` or `end_of_turn`), `message`. The pipeline continued after the error |
| [`SessionEnded`][voxchain.report.models.SessionEnded] | `reason`: the same as `close_reason` |

Use `match` on the class:

```python
for event in report.events:
    match event:
        case BargeIn(at=at, heard_text=heard):
            print(f"{at:.2f}s: the user interrupted after {heard!r}")
        case ToolStarted(at=at, name=name):
            print(f"{at:.2f}s: tool {name}")
```

Events at the same time come in this order: the cause (for example a transcript or a barge-in), then `AgentStoppedSpeaking`, then `AgentStartedSpeaking` of the next reply, then `SessionEnded`.

## Summary

[`report.summary`][voxchain.report.models.SessionReport.summary] gives the main numbers of the session as a [`Summary`][voxchain.report.models.Summary]:

| Field | Content |
|---|---|
| `user_turns` | The number of user turns |
| `barge_ins` | The number of barge-ins |
| `ignored_transcripts` | The number of transcripts that the pipeline discarded |
| `tool_calls` | The number of tool calls |
| `errors` | The number of `ErrorOccurred` events |
| `user_speech_seconds` | The time that the user spoke, from the voice activity of the STT provider |
| `agent_speech_seconds` | The time that the reply audio played |
| `longest_reply_seconds` | The longest time that one reply played |
| `latency` | A [`LatencyStats`][voxchain.report.models.LatencyStats] (`count`, `mean`, `p50`, `p95`, `max`) for `e2e` and for each latency stage, over the user turns |

The percentiles use the nearest rank. With few turns, `p95` is the same as `max`.

## Settings

`thread_id` is the thread of the agent memory. Use it to find the conversation in your LangGraph checkpointer. `settings` are the settings of the pipeline: `greeting`, `interrupt`, `interrupt_min_words`, `resume_window`, `recovery`, `recovery_delay`, `end_of_turn` (the class name of the detector), `end_of_turn_timeout` and `audio_filters` (the class names, in order). Use them to compare calls with different settings.

## Usage

[`Usage`][voxchain.report.models.Usage] gives the seconds of audio sent to STT, the characters sent to TTS, and the input and output tokens of the agent. Use these numbers with the prices of your providers to calculate the cost.

## Audio

The pipeline always keeps the audio of the session. [`AudioRecording`][voxchain.report.models.AudioRecording] has three PCM16 mono tracks on the timeline of the session:

| Track | Content |
|---|---|
| `raw` | The input audio before the audio filters |
| `filtered` | The input audio after the filters: what STT heard |
| `agent` | The reply audio that played, cut at each barge-in |

Compare `raw` and `filtered` to hear what your [audio filters](audio-filters.md) removed. `stereo_wav()` gives one WAV file with the user (`filtered`) on the left channel and the agent on the right. [`wav()`][voxchain.report.models.wav] makes a WAV file from one track.

In `to_dict()`, `audio` gives the sample rates, the length of each track in seconds (`raw_seconds`, `filtered_seconds`, `agent_seconds`), and each track as a WAV file in base64 (`raw`, `filtered`, `agent`). Decode a track to get a WAV file:

```python
data = report.to_dict()
Path("raw.wav").write_bytes(base64.b64decode(data["audio"]["raw"]))
```

A browser can play a track directly as `data:audio/wav;base64,<track>`. Base64 makes the audio one third larger. A 10-minute call gives approximately 90 MB of JSON, so store the audio in files or object storage if your calls are long.

The pipeline keeps the audio in memory. Each track at 16 kHz uses approximately 2 MB for each minute.

The agent track shows when the pipeline sent each chunk to the client, on the assumption that the client plays the audio in real time. If your client buffers audio, the real playback can start later.
