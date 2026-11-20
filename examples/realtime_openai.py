"""Talk to one OpenAI Realtime speech-to-speech model that can call a LangChain tool.

Run: uv run examples/realtime_openai.py   (needs OPENAI_API_KEY)
"""

import asyncio
import os

from common import MIC_RATE, Speaker, get_current_time, microphone, require, setup

from voxchain import OpenAIRealtimeProvider, RealtimePipeline

INSTRUCTIONS = """# Role and Objective
You are a friendly voice assistant in a live phone-style conversation.

# Personality and Tone
Warm, calm and concise. Speak naturally, in one to three short sentences per turn.

# Tools
When the user asks for the time or date, call get_current_time before answering."""


async def show_transcript(role: str, text: str) -> None:
    """Print each transcript of the user and the agent."""
    print(f"{role:>5}: {text}")


async def main() -> None:
    """Connect the providers and talk until Ctrl+C."""
    setup()
    require("OPENAI_API_KEY")
    provider = OpenAIRealtimeProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        instructions=INSTRUCTIONS,
        noise_reduction="far_field",
        transcription_languages=["en"],
    )
    with Speaker(provider.output_sample_rate) as speaker:
        pipeline = RealtimePipeline(
            provider,
            speaker.play,
            tools=[get_current_time],
            input_sample_rate=MIC_RATE,
            on_transcript=show_transcript,
        )
        print("Speak into your microphone. Press Ctrl+C to stop.")
        await pipeline.run(microphone())


if __name__ == "__main__":
    asyncio.run(main())
