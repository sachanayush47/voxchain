"""Talk to a Groq agent through OpenAI STT and OpenAI TTS.

Run: uv run examples/openai.py   (needs OPENAI_API_KEY and GROQ_API_KEY)
"""

import asyncio
import os

from common import MIC_RATE, Speaker, groq_agent, microphone, require, setup

from voxchain import OpenAISTTProvider, OpenAITTSProvider, VoxchainPipeline


async def main() -> None:
    """Connect the providers and talk until Ctrl+C."""
    setup()
    require("OPENAI_API_KEY")
    stt = OpenAISTTProvider(api_key=os.environ["OPENAI_API_KEY"], languages=["en"], sample_rate=MIC_RATE)
    tts = OpenAITTSProvider(
        api_key=os.environ["OPENAI_API_KEY"], instructions="Speak in a friendly, natural, conversational tone."
    )
    with Speaker(tts.sample_rate) as speaker:
        pipeline = VoxchainPipeline(stt, tts, groq_agent(), speaker.play)
        print("Speak into your microphone. Press Ctrl+C to stop.")
        await pipeline.run(microphone())


if __name__ == "__main__":
    asyncio.run(main())
