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
| `sarvam_full.py` | All the features together: Sarvam STT and TTS, a Groq agent, `LoudnessGate` with Hush (or RNNoise without `HUSH_LIBRARY_PATH`), Smart Turn, the barge-in settings and all the output events. | `SARVAM_API_KEY`, `GROQ_API_KEY` |
| `openai.py` | `VoxchainPipeline`: OpenAI STT (with local Silero VAD), a Groq agent and OpenAI TTS. | `OPENAI_API_KEY`, `GROQ_API_KEY` |
| `realtime_openai.py` | `RealtimePipeline`: one OpenAI Realtime model that hears, calls a LangChain tool and speaks. It prints the transcripts. | `OPENAI_API_KEY` |
| `transcribe.py` | STT only: it prints voice activity and transcripts. Use `--provider sarvam` or `--provider openai`. | The key of the provider |
| `speak.py` | TTS only: type a line of text and hear it. Use `--provider sarvam` or `--provider openai`. | The key of the provider |

Run an example as a module from the repo root. Do not run the file directly (`uv run examples/openai.py`): then Python finds `examples/openai.py` before the real `openai` package.

```bash
uv run -m examples.sarvam
uv run -m examples.sarvam_full
uv run -m examples.transcribe --provider openai
```

Add `--filter` to `sarvam.py`, `openai.py` or `realtime_openai.py` to clean the microphone audio. Repeat it to chain filters in that order:

```bash
uv run -m examples.openai --filter loudness --filter hush
```

| Filter | What it removes |
|---|---|
| `rnnoise` | Steady background noise (fans, traffic, hum) |
| `hush` | Background noise and other voices. Set `HUSH_LIBRARY_PATH` to the `weya_nc` library (see `HushFilter`) |
| `loudness` | Audio much quieter than your voice, for example a person across the room |

Ask "What time is it?" to see the agent call its tool. Speak while the agent speaks to see barge-in. Press Ctrl+C to stop.

`common.py` holds the shared parts: the microphone, the speaker, the `.env` file, the logs and the Groq agent.
