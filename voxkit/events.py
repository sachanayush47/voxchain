"""Events shared by every kind of provider."""

from dataclasses import dataclass

from voxkit.session import SessionClosed


@dataclass(frozen=True)
class SpeechStarted:
    """The user started talking. Used for barge-in."""


@dataclass(frozen=True)
class SpeechStopped:
    """The user stopped talking."""


__all__ = ["SessionClosed", "SpeechStarted", "SpeechStopped"]
