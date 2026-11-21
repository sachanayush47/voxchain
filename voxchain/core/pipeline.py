"""Runs a cascaded voice agent: speech-to-text, a LangGraph agent, then text-to-speech."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from langgraph.graph.state import CompiledStateGraph

from voxchain.core.agent import correct_turn, remember_reply, stream_agent_text
from voxchain.core.interruptions import InterruptionPolicy
from voxchain.core.replies import Interruption, Reply, ReplyTracker
from voxchain.core.text import SentenceSegmenter
from voxchain.core.turn_gate import EndOfTurnGate
from voxchain.filters import AudioFilter, FilterChain
from voxchain.output import Output, OutputChannel
from voxchain.session import SessionClosed
from voxchain.stt import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped, STTProvider
from voxchain.tasks import run_until_first_exit
from voxchain.tts import AudioChunk, Flushed, TTSProvider
from voxchain.turn import EndOfTurnDetector

logger = logging.getLogger(__name__)

RECOVERY_PROMPT = (
    "(The user spoke while you talked, but the speech was not recognized. Your last reply stopped, "
    "and the user heard only part of it. Briefly ask the user to say it again.)"
)
"""The agent input after a barge-in that gave no transcript, for example a cough or noise."""


class VoxchainPipeline:
    """Turns user speech into spoken agent replies, and controls turns and barge-in.

    Each committed user turn runs the agent. The reply streams to TTS one
    sentence at a time. Replies run one at a time, in order. ``callback``
    receives :mod:`voxchain.output` events from its own task. Thus a callback
    that blocks on playback never delays a barge-in.

    Barge-in (if ``interrupt`` is on):

    - While the user hears the agent, the user interrupts only with
      ``interrupt_min_words`` words or a stop word such as "stop" (see
      :class:`~voxchain.core.interruptions.InterruptionPolicy`). The pipeline
      discards shorter transcripts, so noise, music and "uh-huh" do not stop
      the agent. Set ``interrupt_min_words=0`` to interrupt on voice activity.
    - Speech that starts before the user hears the reply always interrupts,
      and the turns merge.
    - The user cannot interrupt while a tool call runs, or while the
      ``greeting`` plays. The pipeline ignores speech during the greeting.
    - If the user interrupts within ``resume_window`` seconds after they first
      hear a reply, the pipeline merges their previous turn with their next
      words. ``None`` never merges.
    - After a barge-in, the agent's memory keeps only the part of the reply
      that the user heard.
    - If a barge-in gives no transcript within ``recovery_delay`` seconds after
      the speech ends, the agent receives ``recovery_prompt`` and asks the user
      to say it again. ``None`` turns this off.

    An ``end_of_turn`` detector holds transcripts that sound incomplete for a
    maximum of ``end_of_turn_timeout`` seconds. ``audio_filters`` clean the
    input audio one after another before STT (see :mod:`voxchain.filters`).
    ``thread_id`` is the key for the
    agent's checkpointed memory. A pipeline runs once.
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        agent: CompiledStateGraph,
        callback: Callable[[Output], Awaitable[None]],
        *,
        thread_id: str = "default",
        greeting: str | None = None,
        interrupt: bool = True,
        interrupt_min_words: int = 3,
        resume_window: float | None = 2.0,
        recovery_prompt: str | None = RECOVERY_PROMPT,
        recovery_delay: float = 1.5,
        end_of_turn: EndOfTurnDetector | None = None,
        end_of_turn_timeout: float = 2.0,
        audio_filters: Sequence[AudioFilter] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if end_of_turn_timeout < 0:
            raise ValueError(f"End-of-turn timeout must be >= 0, got {end_of_turn_timeout}")
        if resume_window is not None and resume_window < 0:
            raise ValueError(f"Resume window must be >= 0 or None, got {resume_window}")
        if interrupt_min_words < 0:
            raise ValueError(f"The minimum number of words must be >= 0, got {interrupt_min_words}")
        self.stt = stt
        self.tts = tts
        self.agent = agent
        self.thread_id = thread_id
        self.greeting = greeting
        self.interrupt = interrupt
        self.policy = InterruptionPolicy(min_words=interrupt_min_words) if interrupt_min_words else None
        self.recovery_prompt = recovery_prompt
        self.recovery_delay = recovery_delay
        self.audio_filter = FilterChain(audio_filters)
        self.clock = clock

        self.replies = ReplyTracker(resume_window)
        self.output = OutputChannel(
            callback,
            sample_rate=tts.sample_rate,
            clock=clock,
            on_played=lambda reply, start, _end: self.replies.audio_played(reply, start),
        )
        self._gate = EndOfTurnGate(end_of_turn, end_of_turn_timeout, self._start_reply)
        self._greeting_reply: Reply | None = None
        self._onset_audible = True
        self._awaiting_words = False
        self._recovered = False
        self._recovery: asyncio.Task | None = None
        self._memory_sync: asyncio.Task | None = None
        self._started = False

    async def run(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Run until a provider loses its session, the audio stream ends or a task fails. Then stop."""
        if self._started:
            raise RuntimeError("A VoxchainPipeline runs once; create a new one")
        self._started = True

        await self.stt.connect()
        await self.tts.connect()
        if self.greeting:
            self._start_greeting(self.greeting)
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
            if self._recovery is not None:
                self._recovery.cancel()
            tasks = [reply.task for reply in self.replies.interrupt(self.clock()).cancelled if reply.task]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, *[t for t in [self._memory_sync] if t], return_exceptions=True)
            await self.stt.disconnect()
            await self.tts.disconnect()

    async def _send_audio(self, audio_stream: AsyncIterator[bytes]) -> None:
        async for chunk in audio_stream:
            if not (chunk := self.audio_filter.process(chunk)):
                continue
            self._gate.push_audio(chunk)
            await self.stt.send_audio(chunk)
        logger.info("VoxchainPipeline: Audio stream ended")

    async def _handle_stt(self) -> None:
        async for event in self.stt.receive_events():
            match event:
                case SpeechStarted():
                    await self._on_speech_started()
                case SpeechStopped():
                    self._on_speech_stopped()
                case PartialTranscript(text=text) if text.strip():
                    await self._on_partial(text.strip())
                case FinalTranscript(text=text) if text.strip():
                    await self._on_transcript(text.strip())
                case SessionClosed(reason=reason):
                    await self._session_lost("STT", reason)
                    return

    async def _handle_tts(self) -> None:
        async for event in self.tts.receive_events():
            match event:
                case AudioChunk(data=data):
                    reply = self.replies.audio_received(len(data) / (2 * self.tts.sample_rate))
                    self.output.play(data, tag=reply)
                case Flushed():
                    self.replies.synthesized()
                    self.output.reply_finished()
                case SessionClosed(reason=reason):
                    await self._session_lost("TTS", reason)
                    return

    def _greeting_protected(self, now: float) -> bool:
        greeting = self._greeting_reply
        if greeting is not None and (self.replies.is_active(greeting) or self.output.playing(now)):
            return True
        self._greeting_reply = None
        return False

    def _can_interrupt(self, now: float) -> bool:
        busy = self.replies.active or self.output.playing(now)
        return self.interrupt and busy and not self.replies.tool_running

    async def _on_speech_started(self) -> None:
        now = self.clock()
        if self._greeting_protected(now):
            return
        self._onset_audible = self.output.playing(now)
        self._cancel_recovery()
        if self.policy is None and self._can_interrupt(now):
            await self._barge_in(now)
        self._gate.on_speech_start()

    def _on_speech_stopped(self) -> None:
        if self._greeting_protected(self.clock()):
            return
        self._gate.on_speech_end()
        if self._awaiting_words and self.recovery_prompt and not self._recovered:
            self._cancel_recovery()
            self._recovery = asyncio.create_task(self._recover())

    async def _on_partial(self, text: str) -> None:
        now = self.clock()
        if self.policy is None or self._greeting_protected(now):
            return
        if self._onset_audible and self.output.playing(now) and self._can_interrupt(now) and self.policy.confirms(text):
            await self._barge_in(now)

    async def _on_transcript(self, text: str) -> None:
        now = self.clock()
        if self._greeting_protected(now):
            logger.debug(f"VoxchainPipeline: Ignored {text!r} during the greeting")
            return
        self._awaiting_words = False
        self._recovered = False
        self._cancel_recovery()
        if self.policy is not None and self.interrupt:
            if self._onset_audible and self.output.playing(now):
                heard = self.policy.hear(text, now)
                if heard is None:
                    logger.debug(f"VoxchainPipeline: Ignored {text!r}, too short to interrupt the agent")
                    return
                text = heard
                if self._can_interrupt(now):
                    await self._barge_in(now)
            elif self._can_interrupt(now):
                await self._barge_in(now)  # The user started before they heard the reply, so the turns merge
        self._gate.on_transcript(text)

    async def _barge_in(self, now: float) -> None:
        interruption = self.replies.interrupt(now)
        logger.debug(
            f"VoxchainPipeline: Barge-in (heard {interruption.heard_text!r}, text to resume: "
            f"{interruption.resume_text!r})"
        )
        for reply in interruption.cancelled:
            if reply.task:
                reply.task.cancel()
        # Stop playback before any network call, so that no stale audio gets to the client.
        self.output.stop(now)
        if interruption.cancel_synthesis:
            await self.tts.cancel()
        if interruption.resume_text:
            self._gate.restore(interruption.resume_text)
        else:
            self._awaiting_words = True
        if self.policy is not None:
            self.policy.reset()
        self._memory_sync = asyncio.create_task(self._correct_memory(interruption, self._memory_sync))

    async def _correct_memory(self, interruption: Interruption, previous: asyncio.Task | None) -> None:
        # The cancelled agent turns must end first, so that no checkpoint write comes after the correction.
        tasks = [reply.task for reply in interruption.cancelled if reply.task]
        await asyncio.gather(*tasks, *[t for t in [previous] if t], return_exceptions=True)
        # Oldest first: the heard reply, then the replies that the user did not hear. Correct the newest first.
        heard = [interruption.heard] if interruption.heard else []
        turns = list(dict.fromkeys([*heard, *interruption.resumed, *interruption.cancelled]))
        for reply in reversed(turns):
            if reply in interruption.resumed:
                heard = None
            elif reply is interruption.heard:
                heard = interruption.heard_text
            else:
                heard = ""
            try:
                await correct_turn(self.agent, self.thread_id, reply.text, heard)
            except Exception:
                logger.exception("VoxchainPipeline: Could not correct the agent memory after a barge-in")

    async def _recover(self) -> None:
        await asyncio.sleep(self.recovery_delay)
        if self._awaiting_words and not self.replies.active:
            logger.debug("VoxchainPipeline: Barge-in without a transcript, the agent asks the user to say it again")
            self._awaiting_words = False
            self._recovered = True
            self._start_reply(self.recovery_prompt, resumable=False)

    def _cancel_recovery(self) -> None:
        if self._recovery is not None and not self._recovery.done():
            self._recovery.cancel()

    def _start_greeting(self, text: str) -> None:
        reply = self.replies.start("", resumable=False)
        self._greeting_reply = reply
        reply.task = asyncio.create_task(self._speak_greeting(reply, text))
        reply.task.add_done_callback(lambda _: self.replies.finished_generating(reply))

    async def _speak_greeting(self, reply: Reply, text: str) -> None:
        await self._speak(reply, text)
        await self.tts.flush()
        try:
            await remember_reply(self.agent, self.thread_id, text)
        except Exception:
            logger.exception("VoxchainPipeline: Could not add the greeting to the agent memory")

    def _start_reply(self, text: str, resumable: bool = True) -> None:
        logger.debug(f"VoxchainPipeline: User turn {text!r}")
        previous = self.replies.latest
        reply = self.replies.start(text, resumable=resumable)
        reply.task = asyncio.create_task(self._run_reply(reply, previous.task if previous else None))
        reply.task.add_done_callback(lambda _: self.replies.finished_generating(reply))

    async def _run_reply(self, reply: Reply, previous: asyncio.Task | None) -> None:
        # Cancellation (barge-in) skips the final flush, so a cancelled reply never ends with a late Flushed.
        waits = {task for task in (previous, self._memory_sync) if task is not None}
        if waits:
            await asyncio.wait(waits)
        segmenter = SentenceSegmenter()
        try:
            async for chunk in stream_agent_text(
                self.agent, reply.text, self.thread_id, on_tools=lambda n: setattr(reply, "tool_calls", n)
            ):
                for sentence in segmenter.push(chunk):
                    await self._speak(reply, sentence)
            for sentence in segmenter.flush():
                await self._speak(reply, sentence)
        except asyncio.CancelledError:
            logger.debug(f"VoxchainPipeline: Reply cancelled after {' '.join(reply.spoken)!r}")
            raise
        except Exception:
            logger.exception("VoxchainPipeline: Agent turn failed")
        finally:
            reply.tool_calls = 0
        logger.debug(f"VoxchainPipeline: Agent replied {' '.join(reply.spoken)!r}")
        if reply.spoken:
            await self.tts.flush()

    async def _speak(self, reply: Reply, sentence: str) -> None:
        if not reply.spoken:
            self.replies.spoke(reply)
        await self.tts.speak(sentence)
        reply.spoken.append(sentence)

    async def _session_lost(self, provider: str, reason: str) -> None:
        logger.error(f"VoxchainPipeline: {provider} session closed ({reason}), the pipeline stops")
        await self.output.close(f"{provider}: {reason}")
