"""Type text and hear it, with TTS only. An empty line stops the example.

Run: uv run -m examples.speak --provider sarvam   (or --provider openai)
"""

import argparse
import asyncio
import logging
import os

from examples.common import Speaker, require, setup
from voxchain import OpenAITTSProvider, PlayAudio, SarvamTTSProvider
from voxchain.session import SessionClosed
from voxchain.tts import AudioChunk, Flushed


def build(provider: str):
    """Return the TTS provider that the command line selects."""
    if provider == "openai":
        require("OPENAI_API_KEY")
        return OpenAITTSProvider(api_key=os.environ["OPENAI_API_KEY"])
    require("SARVAM_API_KEY")
    return SarvamTTSProvider(api_key=os.environ["SARVAM_API_KEY"], speaker="priya", language_code="en-IN")


async def main() -> None:
    """Read the command line, then speak each line of text that you type."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=["sarvam", "openai"], default="sarvam")
    args = parser.parse_args()
    setup(logging.INFO)
    tts = build(args.provider)
    await tts.connect()
    flushed: asyncio.Queue[None] = asyncio.Queue()

    with Speaker(tts.sample_rate) as speaker:

        async def read() -> None:
            # The provider reconnects from this reader, so it must run while the example sends text.
            async for event in tts.receive_events():
                match event:
                    case AudioChunk(data=data):
                        await speaker.play(PlayAudio(data))
                    case Flushed():
                        flushed.put_nowait(None)
                    case SessionClosed(reason=reason):
                        print(f"[session closed: {reason}]")
                        return

        reader = asyncio.create_task(read())
        try:
            while text := (await asyncio.to_thread(input, "Text to speak: ")).strip():
                await tts.speak(text)
                await tts.flush()
                await flushed.get()
        finally:
            reader.cancel()
            await tts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
