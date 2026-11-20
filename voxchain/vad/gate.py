"""Cuts a PCM16 stream into speech segments with a local model for voice activity."""

from collections import deque
from dataclasses import dataclass
from typing import Protocol

# Silero recommends this hysteresis: a window is silence only if it scores well below the speech threshold.
NEGATIVE_THRESHOLD_GAP = 0.15


class VoiceActivityModel(Protocol):
    """Anything that scores fixed-size PCM16 windows for speech, such as :class:`~voxchain.vad.silero.SileroVAD`."""

    sample_rate: int
    window_bytes: int

    def speech_probability(self, window: bytes) -> float:
        """The probability (0.0 to 1.0) that one window contains speech."""
        ...


@dataclass(frozen=True)
class SegmentStarted:
    """Speech started. ``audio`` is the pre-roll and the speech that confirmed the start."""

    audio: bytes


@dataclass(frozen=True)
class SegmentAudio:
    """More audio inside the current segment."""

    audio: bytes


@dataclass(frozen=True)
class SegmentEnded:
    """Speech ended. ``audio`` is the segment's last window."""

    audio: bytes


Segment = SegmentStarted | SegmentAudio | SegmentEnded


class VoiceGate:
    """Puts audio through a VAD model and reports speech segments, so that only speech must go further.

    A segment opens after ``start_ms`` of continuous speech and closes after
    ``stop_ms`` of silence. The segment includes ``prefix_ms`` of audio from
    before the speech starts, so that it does not clip the first word.
    """

    def __init__(
        self,
        vad: VoiceActivityModel,
        *,
        threshold: float = 0.5,
        start_ms: int = 200,
        stop_ms: int = 500,
        prefix_ms: int = 300,
    ) -> None:
        if not NEGATIVE_THRESHOLD_GAP < threshold < 1.0:
            raise ValueError(f"Threshold must be between {NEGATIVE_THRESHOLD_GAP} and 1.0, got {threshold}")
        self.vad = vad
        self.threshold = threshold
        window_ms = vad.window_bytes / 2 * 1000 / vad.sample_rate
        self._start_windows = max(1, round(start_ms / window_ms))
        self._stop_windows = max(1, round(stop_ms / window_ms))
        self._preroll: deque[bytes] = deque(maxlen=max(1, round((prefix_ms + start_ms) / window_ms)))
        self._pending = b""
        self._run = 0
        self.speaking = False

    def push(self, chunk: bytes) -> list[Segment]:
        """Score the audio in ``chunk`` and report the changes to the segment."""
        self._pending += chunk
        segments: list[Segment] = []
        size = self.vad.window_bytes
        while len(self._pending) >= size:
            window, self._pending = self._pending[:size], self._pending[size:]
            if segment := self._window(window, self.vad.speech_probability(window)):
                segments.append(segment)
        return segments

    def _window(self, window: bytes, probability: float) -> Segment | None:
        if not self.speaking:
            self._preroll.append(window)
            self._run = self._run + 1 if probability >= self.threshold else 0
            if self._run < self._start_windows:
                return None
            self.speaking, self._run = True, 0
            audio = b"".join(self._preroll)
            self._preroll.clear()
            return SegmentStarted(audio)

        if probability < self.threshold - NEGATIVE_THRESHOLD_GAP:
            self._run += 1
        elif probability >= self.threshold:
            self._run = 0
        if self._run < self._stop_windows:
            return SegmentAudio(window)
        self.speaking, self._run = False, 0
        return SegmentEnded(window)
