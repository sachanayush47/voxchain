"""RNNoiseFilter: it removes steady background noise from 16 kHz PCM16 audio."""

import array
import random

import pytest

pytest.importorskip("pyrnnoise")

from voxchain.filters import RNNoiseFilter
from voxchain.vad.gate import rms_db


def test_steady_noise_becomes_much_quieter():
    rng = random.Random(0)
    audio_filter = RNNoiseFilter(sample_rate=16000)
    out = b""
    for _ in range(20):
        noise = array.array("h", (int(rng.gauss(0, 2000)) for _ in range(1600))).tobytes()
        out += audio_filter.process(noise)
    assert len(out) > 1600 * 2 * 18  # Almost all audio comes back, after a short buffer
    assert rms_db(out[-3200:]) < rms_db(noise) - 30
