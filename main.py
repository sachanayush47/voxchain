import asyncio
import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime

import sounddevice as sd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from voxchain import (
    OpenAIRealtimeProvider,
    OpenAISTTProvider,
    OpenAITTSProvider,
    Output,
    PipecatSmartTurnDetector,
    PlayAudio,
    RealtimePipeline,
    SarvamSTTProvider,
    SarvamTTSProvider,
    StopPlayback,
    VoxchainPipeline,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("voxchain").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE // 10  # 100 ms

# VOXCHAIN_PROVIDER selects the stack (default "sarvam"):
#   "sarvam"          Sarvam STT + Groq agent + Sarvam TTS
#   "openai"          OpenAI STT + Groq agent + OpenAI TTS (needs OPENAI_API_KEY)
#   "openai-realtime" One OpenAI Realtime speech-to-speech model with tools (needs OPENAI_API_KEY)
PROVIDER = os.getenv("VOXCHAIN_PROVIDER", "sarvam")

SYSTEM_PROMPT = (
    "You are in a realtime voice conversation with a human. Your replies are spoken aloud, "
    "so answer in a few short, plain sentences. Never use markdown, lists, tables or emojis."
)

REALTIME_INSTRUCTIONS = """# Role and Objective
You are a friendly voice assistant in a live phone-style conversation.

# Personality and Tone
Warm, calm and concise. Speak naturally, in one to three short sentences per turn.

# Tools
When the user asks for the time or date, call get_current_time before answering."""


@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


async def microphone_stream() -> AsyncIterator[bytes]:
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, _frames, _time, status):
        if status:
            logger.warning(f"Microphone: {status}")
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK_SIZE, callback=callback):
        while True:
            yield await queue.get()


def build_pipeline(play):
    """Return the pipeline for PROVIDER and the sample rate of its output audio."""
    if PROVIDER == "openai-realtime":
        provider = OpenAIRealtimeProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            instructions=REALTIME_INSTRUCTIONS,
            noise_reduction="far_field",
            transcription_languages=["en"],
        )
        pipeline = RealtimePipeline(provider, play, tools=[get_current_time], input_sample_rate=SAMPLE_RATE)
        return pipeline, provider.output_sample_rate

    if PROVIDER == "openai":
        stt = OpenAISTTProvider(api_key=os.environ["OPENAI_API_KEY"], languages=["en"], sample_rate=SAMPLE_RATE)
        tts = OpenAITTSProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            instructions="Speak in a friendly, natural, conversational tone.",
        )
    else:
        stt = SarvamSTTProvider(api_key=os.environ["SARVAM_API_KEY"], language_code="en-IN", sample_rate=SAMPLE_RATE)
        tts = SarvamTTSProvider(api_key=os.environ["SARVAM_API_KEY"], speaker="priya", language_code="en-IN")

    agent = create_agent(
        model=ChatGroq(model="openai/gpt-oss-120b"), tools=[get_current_time], system_prompt=SYSTEM_PROMPT
    )
    pipeline = VoxchainPipeline(
        stt,
        tts,
        agent,
        play,
        end_of_turn=PipecatSmartTurnDetector(sample_rate=SAMPLE_RATE),
    )
    return pipeline, tts.sample_rate


async def main() -> None:
    speaker: sd.RawOutputStream | None = None

    async def play(event: Output) -> None:
        match event:
            case PlayAudio(data=data):
                await asyncio.to_thread(speaker.write, data)
            case StopPlayback():
                await asyncio.to_thread(speaker.abort)
                await asyncio.to_thread(speaker.start)

    pipeline, output_rate = build_pipeline(play)
    speaker = sd.RawOutputStream(samplerate=output_rate, channels=1, dtype="int16")
    speaker.start()
    logger.info("Speak into your microphone (Ctrl+C to stop)...")
    try:
        await pipeline.run(microphone_stream())
    finally:
        speaker.stop()
        speaker.close()


if __name__ == "__main__":
    asyncio.run(main())
