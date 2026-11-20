"""Decides when a user turn is complete and ready for the agent."""

import asyncio
import logging
from collections.abc import Callable

from voxchain.turn import EndOfTurnDetector

logger = logging.getLogger(__name__)


class EndOfTurnGate:
    """Holds finalized transcripts until the gate decides that the user turn is complete.

    Without a detector, the gate commits each transcript immediately, merged
    with any restored text. With a detector, the gate runs this state machine:

    - **A transcript arrives** -> add it to the held text and ask the
      detector. Complete -> commit. Incomplete -> wait ``timeout`` seconds.
    - **Speech starts** -> pause the wait, because the user continues. If
      the gate holds no text, tell the detector that a new turn started.
    - **Speech ends without a transcript** (for example, noise) -> continue the wait.
    - **The wait ends** -> commit all held text.

    If the pipeline cancels the reply to a committed turn before TTS speaks
    it, the pipeline gives the turn text to :meth:`restore`. The text then
    merges with the next words of the user, and the gate does not lose it.

    A detector error counts as "complete", so the user always gets an answer.
    """

    def __init__(
        self,
        detector: EndOfTurnDetector | None,
        timeout: float,
        commit: Callable[[str], None],
    ) -> None:
        """Create the gate."""
        self.detector = detector
        self.timeout = timeout
        self._commit = commit
        self._held = ""
        self._task: asyncio.Task | None = None

    def push_audio(self, chunk: bytes) -> None:
        """Send an input audio chunk to the detector, if there is one."""
        if self.detector is not None:
            self.detector.push_audio(chunk)

    def on_speech_start(self) -> None:
        """Handle the start of user speech."""
        if self.detector is not None and not self._held:
            self.detector.start_turn()
        self._cancel()

    def on_speech_end(self) -> None:
        """Handle the end of user speech."""
        if self._held and (self._task is None or self._task.done()):
            self._start(check=False)

    def on_transcript(self, text: str) -> None:
        """Handle a finalized transcript."""
        self._held = f"{self._held} {text}".strip()
        if self.detector is None:
            self._cancel()
            text, self._held = self._held, ""
            self._commit(text)
            return
        self._start(check=True)

    def restore(self, text: str) -> None:
        """Restore the text of a committed turn if the pipeline cancelled its reply before TTS spoke it.

        The gate puts the text before any held text. It commits the text together
        with the next words of the user, or after the timeout if the user says nothing.
        """
        self._held = f"{text} {self._held}".strip()

    def close(self) -> None:
        """Cancel any pending decision and discard the held text."""
        self._cancel()
        self._held = ""

    def _cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def _start(self, check: bool) -> None:
        self._cancel()
        self._task = asyncio.create_task(self._resolve(check))

    async def _resolve(self, check: bool) -> None:
        complete = False
        if check and self.detector is not None:
            try:
                complete = await self.detector.is_end_of_turn(self._held)
            except Exception:
                logger.exception("EndOfTurnGate: Detector failed, the turn counts as complete")
                complete = True
        if not complete:
            if check:
                logger.debug("EndOfTurnGate: Held %r because it sounds incomplete", self._held)
            logger.debug("EndOfTurnGate: The gate waits %.1fs for the user to continue", self.timeout)
            await asyncio.sleep(self.timeout)

        # There is no await between the text read and the commit, so cancellation cannot split them.
        text, self._held = self._held, ""
        if text:
            self._commit(text)
