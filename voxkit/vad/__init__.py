"""Local voice activity detection, for providers whose backend doesn't report voice activity itself."""

from voxkit.vad.gate import SegmentAudio, SegmentEnded, SegmentStarted, VoiceActivityModel, VoiceGate
from voxkit.vad.silero import SileroVAD

__all__ = ["SegmentAudio", "SegmentEnded", "SegmentStarted", "SileroVAD", "VoiceActivityModel", "VoiceGate"]
