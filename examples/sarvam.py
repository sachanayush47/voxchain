"""Talk to a Groq agent through Sarvam STT and Sarvam TTS, with the Smart Turn end-of-turn check.

Run: uv run -m examples.sarvam [--filter NAME ...]   (needs SARVAM_API_KEY and GROQ_API_KEY)
"""

import asyncio
import os

from examples.common import GREETING, MIC_RATE, Speaker, audio_filters, groq_agent, microphone, options, require, setup
from voxchain import PipecatSmartTurnDetector, SarvamSTTProvider, SarvamTTSProvider, VoxchainPipeline


async def main() -> None:
    """Connect the providers and talk until Ctrl+C."""
    args = options(__doc__.splitlines()[0])
    setup()
    require("SARVAM_API_KEY")
    stt = SarvamSTTProvider(api_key=os.environ["SARVAM_API_KEY"], language_code="en-IN", sample_rate=MIC_RATE)
    tts = SarvamTTSProvider(api_key=os.environ["SARVAM_API_KEY"], speaker="priya", language_code="en-IN")
    with Speaker(tts.sample_rate) as speaker:
        pipeline = VoxchainPipeline(
            stt,
            tts,
            groq_agent(),
            speaker.play,
            greeting=GREETING,
            end_of_turn=PipecatSmartTurnDetector(sample_rate=MIC_RATE),
            audio_filters=audio_filters(args.filter),
        )
        print("Speak into your microphone. Press Ctrl+C to stop.")
        await pipeline.run(microphone())


if __name__ == "__main__":
    asyncio.run(main())
