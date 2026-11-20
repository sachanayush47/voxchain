"""The :class:`EndOfTurnDetector` interface for end-of-turn checks.

STT providers use only silence to decide when speech *stops*. Thus, if a user
pauses in the middle of a thought ("I'd like to book for... um..."), the STT
provider finalizes the half sentence and the agent answers it. An end-of-turn
detector gives a second opinion. Before :class:`~voxchain.core.pipeline.VoxchainPipeline`
sends a finalized transcript to the agent, it asks the detector if the user
really sounds finished.
"""

from abc import ABC, abstractmethod


class EndOfTurnDetector(ABC):
    """Decides if the user turn is complete.

    Give a detector to :class:`~voxchain.core.pipeline.VoxchainPipeline` as
    ``end_of_turn``. The pipeline then does these steps:

    - It sends each input audio chunk to :meth:`push_audio`. These are the
      same bytes that it sends to the STT provider.
    - It calls :meth:`start_turn` when the user starts a new turn.
    - It awaits :meth:`is_end_of_turn` for each finalized transcript.

    If the detector decides that a turn is incomplete, the pipeline holds the
    transcript. It waits a maximum of ``end_of_turn_timeout`` seconds for the
    user to continue. It adds the next transcript to the held text and asks
    the detector again.
    """

    @abstractmethod
    def push_audio(self, chunk: bytes) -> None:
        """Receive one chunk of the raw input audio.

        The pipeline calls this method for each chunk, in order, on the event
        loop. Thus, keep the work small. Add the chunk to a buffer, and do not
        run inference here.
        """

    @abstractmethod
    def start_turn(self) -> None:
        """Mark the start of a new user turn.

        The pipeline calls this method on :class:`~voxchain.stt.events.SpeechStarted`
        if it holds no transcript. That is, the user starts a new turn and does
        not continue an incomplete turn. Audio from earlier turns must not
        change the next decision.
        """

    @abstractmethod
    async def is_end_of_turn(self, transcript: str) -> bool:
        """Decide if the user finished their speech. ``True`` means that the agent must reply now.

        The pipeline calls this method one time for each finalized transcript,
        directly on the path that sets the reply latency. Thus, keep it fast.
        If it raises an exception, the pipeline logs the error and treats the
        turn as complete. Thus a broken detector never stops the
        agent. The agent always answers the user.
        """
