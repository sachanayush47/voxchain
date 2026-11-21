"""A filter that silences audio that is much quieter than the speech of the user."""

import math
from collections import deque

from voxchain.audio import SILENT_DB, rms_db

FRAME_MS = 10
# The reference rises quickly to the level of near speech and falls slowly. Thus steady background
# sound that passes the gate cannot pull the reference down to its own level.
REFERENCE_RISE = 0.02
REFERENCE_FALL = 0.002


class LoudnessGate:
    """Silences audio that is much quieter than the speech of the user, for example a voice across the room.

    Sound becomes approximately 6 dB quieter each time the distance doubles. Thus
    the voice of the user near the microphone is louder than other voices in the
    room. The gate measures the level of the last ``window_ms`` of audio. Single
    loud syllables of a far voice thus do not open it. The gate learns the level
    of the speech that it lets through (``reference_db``), and silences audio more
    than ``margin_db`` below that level. Before it learns a level, and as a lower
    limit, it uses ``min_db`` (dBFS). After loud audio, it stays open for
    ``hold_ms``, so that it does not cut the ends of words.

    The level rises slowly at the start of speech, so the gate can cut the first
    tens of milliseconds of a word. ``lookahead_ms`` delays the audio by that time
    and opens the gate before the speech starts. The gate needs no model. It
    replaces audio with silence of the same length, so the timing does not change.
    It cannot remove a voice that is as loud as the user.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        margin_db: float = 12.0,
        min_db: float = -50.0,
        window_ms: int = 200,
        hold_ms: int = 200,
        lookahead_ms: int = 0,
    ) -> None:
        if margin_db <= 0:
            raise ValueError(f"Margin must be more than 0 dB, got {margin_db}")
        self.margin_db = margin_db
        self.min_db = min_db
        self.reference_db: float | None = None
        self._frame_bytes = sample_rate * FRAME_MS // 1000 * 2
        self._powers: deque[float] = deque(maxlen=max(1, window_ms // FRAME_MS))
        self._hold_frames = hold_ms // FRAME_MS
        self._lookahead_frames = lookahead_ms // FRAME_MS
        self._delayed: deque[list] = deque()  # [frame, open] pairs inside the lookahead
        self._hold = 0
        self._pending = b""

    @property
    def threshold_db(self) -> float:
        """The level (dBFS) below which the gate silences audio."""
        if self.reference_db is None:
            return self.min_db
        return max(self.min_db, self.reference_db - self.margin_db)

    def process(self, pcm: bytes) -> bytes:
        """Return ``pcm`` with silence in place of the quiet audio. A part of a frame waits for the next chunk."""
        data = self._pending + pcm
        end = len(data) - len(data) % self._frame_bytes
        self._pending = data[end:]
        out = bytearray()
        for start in range(0, end, self._frame_bytes):
            frame = data[start : start + self._frame_bytes]
            self._delayed.append([frame, self._is_open(frame)])
            while len(self._delayed) > self._lookahead_frames:
                frame, is_open = self._delayed.popleft()
                out += frame if is_open else bytes(len(frame))
        return bytes(out)

    def _is_open(self, frame: bytes) -> bool:
        self._powers.append(10 ** (rms_db(frame) / 10))
        power = sum(self._powers) / len(self._powers)
        level = 10 * math.log10(power) if power > 0 else SILENT_DB
        if level >= self.threshold_db:
            if self.reference_db is None:
                self.reference_db = level
            else:
                rate = REFERENCE_RISE if level > self.reference_db else REFERENCE_FALL
                self.reference_db += rate * (level - self.reference_db)
            self._hold = self._hold_frames
            for entry in self._delayed:
                entry[1] = True
            return True
        if self._hold:
            self._hold -= 1
            return True
        return False
