"""Events that all kinds of provider share."""

from dataclasses import dataclass

from voxchain.session import SessionClosed


@dataclass(frozen=True)
class SpeechStarted:
    """The speech of the user started. The pipeline uses this event for barge-in."""


@dataclass(frozen=True)
class SpeechStopped:
    """The speech of the user stopped."""


__all__ = ["SessionClosed", "SpeechStarted", "SpeechStopped"]
