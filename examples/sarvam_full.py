"""Talk to a Groq agent through Sarvam, with all the voxchain features: filters, Smart Turn, barge-in and the report.

Run: uv run -m examples.sarvam_full   (needs SARVAM_API_KEY and GROQ_API_KEY. Set HUSH_LIBRARY_PATH to use Hush)
When the call ends, the example prints the transcript and the latency of each turn. It saves the report and
the audio in sessions/<time>/ at the repo root.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from examples.common import GREETING, MIC_RATE, Speaker, groq_agent, microphone, require, setup
from voxchain import (
    Output,
    PipecatSmartTurnDetector,
    ReplyFinished,
    SarvamSTTProvider,
    SarvamTTSProvider,
    StopPlayback,
    StreamClosed,
    VoxchainPipeline,
)
from voxchain.filters import AudioFilter, HushFilter, LoudnessGate, RNNoiseFilter
from voxchain.report import SessionReport, wav

SESSIONS = Path(__file__).resolve().parent.parent / "sessions"


def audio_filters() -> list[AudioFilter]:
    """Return a LoudnessGate and one denoiser: Hush if its library is set, otherwise RNNoise."""
    # The gate silences far voices before Hush hears them. Then Hush does not weaken the next word of the user.
    gate = LoudnessGate(sample_rate=MIC_RATE, margin_db=12.0)
    if library := os.getenv("HUSH_LIBRARY_PATH"):
        return [gate, HushFilter(library_path=library, sample_rate=MIC_RATE)]
    print("HUSH_LIBRARY_PATH is not set. RNNoise removes steady noise, but not other voices.")
    return [gate, RNNoiseFilter(sample_rate=MIC_RATE)]


def save_report(report: SessionReport) -> None:
    """Print the transcript and the latency of each turn. Save the report and the audio in a new folder."""
    folder = SESSIONS / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    audio = report.audio
    (folder / "conversation.wav").write_bytes(audio.stereo_wav())  # The user on the left, the agent on the right
    (folder / "raw.wav").write_bytes(wav(audio.raw, audio.input_sample_rate))
    (folder / "filtered.wav").write_bytes(wav(audio.filtered, audio.input_sample_rate))
    print(f"\n{report.transcript()}\n\n{report.waterfall()}\n\nSaved the report and the audio in {folder}")


async def main() -> None:
    """Connect the providers and talk until Ctrl+C."""
    setup()
    require("SARVAM_API_KEY")
    stt = SarvamSTTProvider(api_key=os.environ["SARVAM_API_KEY"], language_code="en-IN", sample_rate=MIC_RATE)
    tts = SarvamTTSProvider(api_key=os.environ["SARVAM_API_KEY"], speaker="priya", language_code="en-IN")

    with Speaker(tts.sample_rate) as speaker:

        async def on_output(event: Output) -> None:
            match event:
                case StopPlayback():
                    print("[Barge-in: the agent stops]")
                case ReplyFinished():
                    print("[The agent sent all of its reply]")
                case StreamClosed(reason=reason):
                    print(f"[The pipeline stops: {reason}]")
            await speaker.play(event)

        pipeline = VoxchainPipeline(
            stt,
            tts,
            groq_agent(),
            on_output,
            thread_id="sarvam-full",  # The key of the agent memory. Use one for each conversation
            greeting=GREETING,  # The user cannot interrupt the greeting
            audio_filters=audio_filters(),
            # Smart Turn holds a transcript that sounds incomplete, for a maximum of 2 seconds
            end_of_turn=PipecatSmartTurnDetector(sample_rate=MIC_RATE, threshold=0.5),
            end_of_turn_timeout=2.0,
            # While the agent speaks, only 3 words or a stop word ("stop", "wait") interrupt it
            interrupt_min_words=3,
            # If the user interrupts within 2 seconds, the previous turn and the next words merge
            resume_window=2.0,
            # If a barge-in gives no words, the agent asks the user to say it again
            recovery_delay=1.5,
        )
        print("Speak into your microphone. Press Ctrl+C to stop.")
        try:
            await pipeline.run(microphone())
        finally:
            save_report(pipeline.report())


if __name__ == "__main__":
    asyncio.run(main())
