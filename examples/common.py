"""Shared parts of the examples: the environment, logs, the microphone, the speaker and a small agent."""

import argparse
import asyncio
import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from voxchain import Output, PlayAudio, StopPlayback
from voxchain.filters import AudioFilter

MIC_RATE = 16000
MIC_CHUNK = MIC_RATE // 10  # 100 ms

SYSTEM_PROMPT = (
    "You are in a realtime voice conversation with a human. Your replies are spoken aloud, "
    "so answer in a few short, plain sentences. Never use markdown, lists, tables or emojis."
)

GREETING = "Hi! I am your voice assistant. How can I help you today?"

logger = logging.getLogger("examples")


def options(description: str) -> argparse.Namespace:
    """Read the command line options that all conversation examples share."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--denoise", action="store_true", help="Remove background noise with RNNoise before STT")
    return parser.parse_args()


def noise_filter(enabled: bool) -> AudioFilter | None:
    """Return an RNNoise filter for the microphone audio if ``enabled``, or ``None``."""
    if not enabled:
        return None
    from voxchain.filters import RNNoiseFilter

    return RNNoiseFilter(sample_rate=MIC_RATE)


def setup(level: int = logging.DEBUG) -> None:
    """Load the API keys from the .env file at the repo root, and show the voxchain logs."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("voxchain").setLevel(level)


def require(*names: str) -> None:
    """Stop with a clear message if an environment variable is missing."""
    if missing := [name for name in names if not os.getenv(name)]:
        raise SystemExit(f"Set {', '.join(missing)} in the .env file at the repo root")


@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def groq_agent(model: str = "openai/gpt-oss-120b"):
    """Return a LangGraph agent on Groq that can tell the time, with a memory of the conversation."""
    require("GROQ_API_KEY")
    return create_agent(
        model=ChatGroq(model=model),
        tools=[get_current_time],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


async def microphone() -> AsyncIterator[bytes]:
    """Yield 100 ms chunks of PCM16 audio at 16 kHz from the default microphone."""
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, _frames, _time, status):
        if status:
            logger.warning(f"Microphone: {status}")
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16", blocksize=MIC_CHUNK, callback=callback):
        while True:
            yield await queue.get()


class Speaker:
    """Plays the output events of a pipeline on the default speaker."""

    def __init__(self, sample_rate: int) -> None:
        self.stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")

    async def play(self, event: Output) -> None:
        """The pipeline callback: play audio, and discard queued audio at a barge-in."""
        match event:
            case PlayAudio(data=data):
                await asyncio.to_thread(self.stream.write, data)
            case StopPlayback():
                await asyncio.to_thread(self.stream.abort)
                await asyncio.to_thread(self.stream.start)

    def __enter__(self) -> "Speaker":
        self.stream.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stream.stop()
        self.stream.close()
