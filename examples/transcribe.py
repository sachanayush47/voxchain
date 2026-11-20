"""Print live transcripts and voice activity from the microphone, with STT only.

Run: uv run -m examples.transcribe --provider sarvam   (or --provider openai)
"""

import argparse
import asyncio
import logging
import os

from examples.common import MIC_RATE, microphone, require, setup
from voxchain import OpenAISTTProvider, SarvamSTTProvider
from voxchain.session import SessionClosed
from voxchain.stt import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped


def build(provider: str):
    """Return the STT provider that the command line selects."""
    if provider == "openai":
        require("OPENAI_API_KEY")
        return OpenAISTTProvider(
            api_key=os.environ["OPENAI_API_KEY"], languages=["en"], sample_rate=MIC_RATE, include_partials=True
        )
    require("SARVAM_API_KEY")
    return SarvamSTTProvider(api_key=os.environ["SARVAM_API_KEY"], language_code="en-IN", sample_rate=MIC_RATE)


async def main() -> None:
    """Read the command line, then print events until Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=["sarvam", "openai"], default="sarvam")
    args = parser.parse_args()
    setup(logging.INFO)
    stt = build(args.provider)
    await stt.connect()

    async def send() -> None:
        async for chunk in microphone():
            await stt.send_audio(chunk)

    sender = asyncio.create_task(send())
    print("Speak into your microphone. Press Ctrl+C to stop.")
    try:
        async for event in stt.receive_events():
            match event:
                case SpeechStarted():
                    print("[speech started]")
                case SpeechStopped():
                    print("[speech stopped]")
                case PartialTranscript(text=text):
                    print(f"  ... {text}")
                case FinalTranscript(text=text):
                    print(f"  >>> {text}")
                case SessionClosed(reason=reason):
                    print(f"[session closed: {reason}]")
                    break
    finally:
        sender.cancel()
        await stt.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
