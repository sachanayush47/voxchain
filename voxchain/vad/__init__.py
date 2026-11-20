"""Local voice activity detection, for providers whose backend does not report voice activity."""

from voxchain.vad.gate import SegmentAudio, SegmentEnded, SegmentStarted, VoiceActivityModel, VoiceGate
from voxchain.vad.silero import SileroVAD

__all__ = ["SegmentAudio", "SegmentEnded", "SegmentStarted", "SileroVAD", "VoiceActivityModel", "VoiceGate"]
