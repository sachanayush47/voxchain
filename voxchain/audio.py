"""Small audio helpers that providers share."""

import math
import sys
from array import array

SILENT_DB = -100.0


def rms_db(pcm: bytes) -> float:
    """Return the RMS level of PCM16 audio in dBFS. Full scale is 0, and silence is ``SILENT_DB``."""
    samples = array("h", pcm)
    if not samples:
        return SILENT_DB
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768
    return max(SILENT_DB, 20 * math.log10(rms)) if rms > 0 else SILENT_DB


class PCM16Resampler:
    """A resampler for streams of 16-bit little-endian mono PCM that uses linear interpolation.

    The resampler keeps the last sample and the fractional read position between
    calls. Thus chunk boundaries do not click, and the output is the same if audio
    arrives in one piece or in many. Linear interpolation is sufficient to
    upsample speech for a recognizer (for example, from 16 kHz to 24 kHz). It does
    not apply a low-pass filter, so a large decrease of the sample rate can alias.

    Example:
        >>> resampler = PCM16Resampler(16000, 24000)
        >>> out = resampler.process(chunk)  # Call once for each chunk that arrives
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        """Create a resampler."""
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError(f"Sample rates must be positive, got {source_rate} and {target_rate}")
        # Positions use units of 1/target_rate input samples, so they stay exact.
        self._step = source_rate
        self._scale = target_rate
        self._pos = 0
        self._prev: int | None = None
        self._odd = b""

    def process(self, chunk: bytes) -> bytes:
        """Resample one chunk."""
        data = self._odd + chunk
        cut = len(data) - len(data) % 2
        data, self._odd = data[:cut], data[cut:]
        if self._step == self._scale:
            return data

        samples = array("h")
        samples.frombytes(data)
        if sys.byteorder == "big":
            samples.byteswap()
        if self._prev is not None:
            samples.insert(0, self._prev)
        if not samples:
            return b""

        out = array("h")
        pos, step, scale = self._pos, self._step, self._scale
        end = (len(samples) - 1) * scale
        while pos < end:
            i, frac = divmod(pos, scale)
            a = samples[i]
            out.append(a + (samples[i + 1] - a) * frac // scale)
            pos += step
        self._pos = pos - end
        self._prev = samples[-1]

        if sys.byteorder == "big":
            out.byteswap()
        return out.tobytes()
