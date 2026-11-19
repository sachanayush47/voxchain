"""Runs a cascaded voice agent: speech-to-text, a LangGraph agent, then text-to-speech."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from langgraph.graph.state import CompiledStateGraph

from voxchain.core.agent import stream_agent_text
from voxchain.core.replies import Reply, ReplyTracker
from voxchain.core.text import SentenceSegmenter
from voxchain.core.turn_gate import EndOfTurnGate
from voxchain.output import Output, OutputChannel
from voxchain.session import SessionClosed
from voxchain.stt import FinalTranscript, SpeechStarted, SpeechStopped, STTProvider
from voxchain.tasks import run_until_first_exit
from voxchain.tts import AudioChunk, Flushed, TTSProvider
from voxchain.turn import EndOfTurnDetector

logger = logging.getLogger(__name__)


class VoxchainPipeline:
    """Turns user speech into agent replies spoken back, with turn-taking and barge-in.

    Each committed user turn runs the agent; its reply streams to TTS sentence
    by sentence, and replies run one at a time, in order. ``callback`` receives
    :mod:`voxchain.output` events from its own task, so a callback that blocks on
    playback never delays barge-in.

    With ``interrupt`` on, the user talking over a reply cancels it and stops
    playback. If they barge in within ``resume_window`` seconds of first hearing
    it, their previous turn is merged with what they say next, because the
    end-of-turn call was most likely premature (``None`` never merges). An
    ``end_of_turn`` detector holds transcripts that sound unfinished for up to
    ``end_of_turn_timeout`` seconds. ``thread_id`` keys the agent's checkpointed
    memory. A pipeline runs once.
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        agent: CompiledStateGraph,
        callback: Callable[[Output], Awaitable[None]],
        *,
        thread_id: str = "default",
        interrupt: bool = True,
        resume_window: float | None = 2.0,
        end_of_turn: EndOfTurnDetector | None = None,
        end_of_turn_timeout: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if end_of_turn_timeout < 0:
            raise ValueError(f"End-of-turn timeout must be >= 0, got {end_of_turn_timeout}")
        if resume_window is not None and resume_window < 0:
            raise ValueError(f"Resume window must be >= 0 or None, got {resume_window}")
        self.stt = stt
        self.tts = tts
        self.agent = agent
        self.thread_id = thread_id
        self.interrupt = interrupt
        self.clock = clock

        self.replies = ReplyTracker(resume_window)
        self.output = OutputChannel(
            callback,
            sample_rate=tts.sample_rate,
            clock=clock,
            on_played=lambda _tag, start, _end: self.replies.audio_played(start),
        )
        self._gate = EndOfTurnGate(end_of_turn, end_of_turn_timeout, self._start_reply)
        self._started = False

    async def run(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Run until a provider's session is lost, the audio stream ends, or a task fails; then shut down."""
        if self._started:
            raise RuntimeError("A VoxchainPipeline runs once; create a new one")
        self._started = True

        await self.stt.connect()
        await self.tts.connect()
        loops = {
            "send_audio": self._send_audio(audio_stream),
            "handle_stt": self._handle_stt(),
            "handle_tts": self._handle_tts(),
            "play": self.output.run(),
        }
        try:
            await run_until_first_exit("VoxchainPipeline", loops)
        finally:
            self._gate.close()
            reply_tasks = [reply.task for reply in self.replies.interrupt(self.clock()).cancelled if reply.task]
            for task in reply_tasks:
                task.cancel()
            await asyncio.gather(*reply_tasks, return_exceptions=True)
            await self.stt.disconnect()
            await self.tts.disconnect()

    async def _send_audio(self, audio_stream: AsyncIterator[bytes]) -> None:
        async for chunk in audio_stream:
            self._gate.push_audio(chunk)
            await self.stt.send_audio(chunk)
        logger.info("VoxchainPipeline: Audio stream ended")

    async def _handle_stt(self) -> None:
        async for event in self.stt.receive_events():
            match event:
                case SpeechStarted():
                    await self._on_speech_started()
                case SpeechStopped():
                    self._gate.on_speech_end()
                case FinalTranscript(text=text) if text.strip():
                    self._gate.on_transcript(text.strip())
                case SessionClosed(reason=reason):
                    await self._session_lost("STT", reason)
                    return

    async def _handle_tts(self) -> None:
        async for event in self.tts.receive_events():
            match event:
                case AudioChunk(data=data):
                    self.output.play(data)
                case Flushed():
                    self.replies.synthesized()
                    self.output.reply_finished()
                case SessionClosed(reason=reason):
                    await self._session_lost("TTS", reason)
                    return

    async def _on_speech_started(self) -> None:
        now = self.clock()
        if self.interrupt and (self.replies.active or self.output.playing(now)):
            interruption = self.replies.interrupt(now)
            logger.debug(f"VoxchainPipeline: Barge-in (resuming {interruption.resume_text!r})")
            for reply in interruption.cancelled:
                if reply.task:
                    reply.task.cancel()
            # Stop playback before any network call, so nothing stale reaches the client.
            self.output.stop(now)
            if interruption.cancel_synthesis:
                await self.tts.cancel()
            if interruption.resume_text:
                self._gate.restore(interruption.resume_text)
        self._gate.on_speech_start()

    def _start_reply(self, text: str) -> None:
        logger.debug(f"VoxchainPipeline: User turn {text!r}")
        previous = self.replies.latest
        reply = self.replies.start(text)
        reply.task = asyncio.create_task(self._run_reply(reply, previous.task if previous else None))
        reply.task.add_done_callback(lambda _: self.replies.finished_generating(reply))

    async def _run_reply(self, reply: Reply, previous: asyncio.Task | None) -> None:
        # Cancellation (barge-in) skips the closing flush, so a cancelled reply never ends in a late Flushed.
        if previous is not None:
            await asyncio.wait({previous})
        segmenter = SentenceSegmenter()
        spoken: list[str] = []
        try:
            async for chunk in stream_agent_text(self.agent, reply.text, self.thread_id):
                for sentence in segmenter.push(chunk):
                    await self._speak(reply, sentence, spoken)
            for sentence in segmenter.flush():
                await self._speak(reply, sentence, spoken)
        except asyncio.CancelledError:
            logger.debug(f"VoxchainPipeline: Reply cancelled after {' '.join(spoken)!r}")
            raise
        except Exception:
            logger.exception("VoxchainPipeline: Agent turn failed")
        logger.debug(f"VoxchainPipeline: Agent replied {' '.join(spoken)!r}")
        if spoken:
            await self.tts.flush()

    async def _speak(self, reply: Reply, sentence: str, spoken: list[str]) -> None:
        if not spoken:
            self.replies.spoke(reply)
        await self.tts.speak(sentence)
        spoken.append(sentence)

    async def _session_lost(self, provider: str, reason: str) -> None:
        logger.error(f"VoxchainPipeline: {provider} session closed ({reason}), stopping")
        await self.output.close(f"{provider}: {reason}")
