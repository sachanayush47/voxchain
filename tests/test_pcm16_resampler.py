"""PCM16Resampler: chunk boundaries never change the output."""

import math
from array import array

from voxchain.audio import PCM16Resampler


def _sine(n: int, rate: int) -> bytes:
    return array("h", (int(10000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n))).tobytes()


def test_chunked_input_resamples_exactly_like_one_piece():
    audio = _sine(16000, 16000)
    chunked = PCM16Resampler(16000, 24000)
    out, i = b"", 0
    for size in [1, 333, 1000, 7, 5000] * 10:
        out += chunked.process(audio[i : i + size])
        i += size
    out += chunked.process(audio[i:])
    assert out == PCM16Resampler(16000, 24000).process(audio)


def test_upsampling_keeps_the_duration():
    out = PCM16Resampler(16000, 24000).process(_sine(16000, 16000))
    assert abs(len(out) // 2 - 24000) <= 1


def test_equal_rates_pass_audio_through_untouched():
    audio = _sine(1000, 24000)
    assert PCM16Resampler(24000, 24000).process(audio) == audio
