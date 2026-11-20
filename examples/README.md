# Examples

Each example is a script that you can run. Each one uses your microphone and speakers through `sounddevice`. Use headphones, or the agent can hear itself.

## Setup

1. Install the dev dependencies: `uv sync`.
2. Put the API keys in a `.env` file at the repo root:

   ```
   SARVAM_API_KEY=...
   OPENAI_API_KEY=...
   GROQ_API_KEY=...
   ```

   Each example needs only the keys in the table below.

## The examples

| Example | What it shows | Keys |
|---|---|---|
| `sarvam.py` | `VoxchainPipeline`: Sarvam STT, a Groq agent with a time tool, and Sarvam TTS. The Smart Turn model checks the end of each turn. | `SARVAM_API_KEY`, `GROQ_API_KEY` |
| `openai.py` | `VoxchainPipeline`: OpenAI STT (with local Silero VAD), a Groq agent and OpenAI TTS. | `OPENAI_API_KEY`, `GROQ_API_KEY` |
| `realtime_openai.py` | `RealtimePipeline`: one OpenAI Realtime model that hears, calls a LangChain tool and speaks. It prints the transcripts. | `OPENAI_API_KEY` |
| `transcribe.py` | STT only: it prints voice activity and transcripts. Use `--provider sarvam` or `--provider openai`. | The key of the provider |
| `speak.py` | TTS only: type a line of text and hear it. Use `--provider sarvam` or `--provider openai`. | The key of the provider |

Run an example from the repo root:

```bash
uv run examples/sarvam.py
uv run examples/transcribe.py --provider openai
```

Ask "What time is it?" to see the agent call its tool. Speak while the agent speaks to see barge-in. Press Ctrl+C to stop.

`common.py` holds the shared parts: the microphone, the speaker, the `.env` file, the logs and the Groq agent.
