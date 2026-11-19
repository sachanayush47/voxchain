"""Runs a speech-to-speech voice agent: microphone audio in, the model's speech out, tools in between."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

from langchain_core.tools import BaseTool

from voxchain.audio import PCM16Resampler
from voxchain.output import Output, OutputChannel
from voxchain.realtime.base import RealtimeProvider
from voxchain.realtime.events import (
    AgentTranscript,
    AudioDelta,
    FunctionCall,
    ResponseDone,
    ResponseStarted,
    SessionClosed,
    SpeechStarted,
    UserTranscript,
)
from voxchain.realtime.playback import AudioAction, PlaybackTracker
from voxchain.realtime.tools import NOT_RUN_OUTPUT, ToolRunner
from voxchain.tasks import run_until_first_exit

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Feeds audio to a :class:`~voxchain.realtime.base.RealtimeProvider`, plays its speech and runs its tool calls.

    ``callback`` receives :mod:`voxchain.output` events, the same ones
    :class:`~voxchain.core.pipeline.VoxchainPipeline` sends, from its own task, so a
    callback that blocks on playback never delays barge-in. On barge-in the conversation is
    cut to what the user heard (see :class:`~voxchain.realtime.playback.PlaybackTracker`),
    and calls from the interrupted response are not run. A pipeline runs once.
    """

    def __init__(
        self,
        provider: RealtimeProvider,
        callback: Callable[[Output], Awaitable[None]],
        *,
        tools: Sequence[BaseTool] = (),
        input_sample_rate: int = 16000,
        on_transcript: Callable[[str, str], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.on_transcript = on_transcript
        self.clock = clock
        self.tools = ToolRunner(tools)

        self._resampler = PCM16Resampler(input_sample_rate, provider.input_sample_rate)
        self._bytes_per_ms = provider.output_sample_rate * 2 / 1000
        self._playback = PlaybackTracker()
        self.output = OutputChannel(
            callback, sample_rate=provider.output_sample_rate, clock=clock, on_played=self._playback.played
        )
        self._calls: dict[str, list[FunctionCall]] = {}
        self._tool_tasks: set[asyncio.Task] = set()
        self._started = False

    async def run(self, audio_stream: AsyncIterator[bytes]) -> None:
        """Run until the session is lost, the audio stream ends, or a task fails; then shut down."""
        if self._started:
            raise RuntimeError("A RealtimePipeline runs once; create a new one")
        self._started = True

        self.provider.set_tools(self.tools.specs())
        await self.provider.connect()
        loops = {
            "send_audio": self._send_audio(audio_stream),
            "handle_events": self._handle_events(),
            "play": self.output.run(),
        }
        try:
            await run_until_first_exit("RealtimePipeline", loops)
        finally:
            for task in self._tool_tasks:
                task.cancel()
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
            await self.provider.disconnect()

    async def _send_audio(self, audio_stream: AsyncIterator[bytes]) -> None:
        async for chunk in audio_stream:
            if pcm := self._resampler.process(chunk):
                await self.provider.send_audio(pcm)
        logger.info("RealtimePipeline: Audio stream ended")

    async def _handle_events(self) -> None:
        async for event in self.provider.receive_events():
            match event:
                case SpeechStarted() if self.provider.interrupts_on_speech:
                    await self._barge_in()
                case ResponseStarted(response_id=response_id):
                    self._playback.response_started(response_id)
                case AudioDelta():
                    await self._on_audio(event)
                case UserTranscript(text=text):
                    await self._transcript("user", text)
                case AgentTranscript(response_id=response_id, text=text):
                    if not self._playback.is_discarded(response_id):
                        await self._transcript("agent", text)
                case FunctionCall(response_id=response_id):
                    self._calls.setdefault(response_id, []).append(event)
                case ResponseDone():
                    self._on_response_done(event)
                case SessionClosed(reason=reason):
                    logger.error(f"RealtimePipeline: Session closed ({reason})")
                    await self.output.close(reason)
                    return

    async def _on_audio(self, event: AudioDelta) -> None:
        duration_ms = len(event.data) / self._bytes_per_ms
        match self._playback.accept(event.response_id, event.item_id, duration_ms):
            case AudioAction.PLAY:
                self.output.play(event.data, tag=event.item_id)
            case AudioAction.DELETE:
                await self.provider.delete_item(event.item_id)
            case AudioAction.DROP:
                pass

    async def _barge_in(self) -> None:
        now = self.clock()
        cuts = self._playback.barge_in(now)
        # Stop playback before any network call, so the heard estimate can't drift.
        if cuts or self.output.has_queued_audio():
            self.output.stop(now)
            logger.debug(f"RealtimePipeline: Barge-in, cutting {cuts}")
        for cut in cuts:
            if cut.heard_ms > 0:
                await self.provider.truncate(cut.item_id, cut.heard_ms)
            else:
                await self.provider.delete_item(cut.item_id)

    def _on_response_done(self, event: ResponseDone) -> None:
        outcome = self._playback.response_done(event.response_id)
        if outcome.spoke and not outcome.interrupted:
            self.output.reply_finished()
        if calls := self._calls.pop(event.response_id, []):
            task = asyncio.create_task(self._answer_calls(calls, outcome.interrupted))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def _answer_calls(self, calls: list[FunctionCall], interrupted: bool) -> None:
        outputs = [NOT_RUN_OUTPUT] * len(calls) if interrupted else await self.tools.run_all(calls)
        for call, output in zip(calls, outputs, strict=True):
            await self.provider.send_function_result(call.call_id, output)
        if not interrupted:
            await self.provider.request_response()

    async def _transcript(self, role: str, text: str) -> None:
        logger.debug(f"RealtimePipeline: {role}: {text!r}")
        if self.on_transcript and text.strip():
            try:
                await self.on_transcript(role, text)
            except Exception:
                logger.exception("RealtimePipeline: On_transcript failed")
