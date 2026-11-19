"""End-of-turn detection: deciding whether the user has actually finished speaking."""

from voxchain.turn.base import EndOfTurnDetector
from voxchain.turn.smart_turn import SMART_TURN_FILENAME, SMART_TURN_REPO, PipecatSmartTurnDetector

__all__ = ["SMART_TURN_FILENAME", "SMART_TURN_REPO", "EndOfTurnDetector", "PipecatSmartTurnDetector"]
