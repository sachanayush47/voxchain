"""LoudnessGate: it learns the level of near speech and silences much quieter audio, such as a far voice."""

import math
import struct

import pytest

from voxchain.audio import rms_db
from voxchain.filters import LoudnessGate

RATE = 16000


def tone(db, ms, hz=220):
    """A sine wave at a RMS level of ``db`` dBFS."""
    amplitude = 32768 * 10 ** (db / 20) * math.sqrt(2)
    n = RATE * ms // 1000
    return struct.pack(f"<{n}h", *(round(amplitude * math.sin(2 * math.pi * hz * i / RATE)) for i in range(n)))


def is_silent(pcm):
    return pcm == bytes(len(pcm))


def test_audio_louder_than_min_db_passes_before_the_gate_learns_a_level():
    audio = tone(-40, 100)
    assert LoudnessGate().process(audio) == audio


def test_audio_quieter_than_min_db_becomes_silence_of_the_same_length():
    out = LoudnessGate().process(tone(-60, 100))
    assert len(out) == RATE // 10 * 2
    assert is_silent(out)


def test_after_near_speech_a_far_voice_18_db_quieter_becomes_silence():
    gate = LoudnessGate()
    gate.process(tone(-26, 2000) + bytes(16000))
    assert is_silent(gate.process(tone(-44, 1000)))


def test_short_loud_syllables_of_a_far_voice_do_not_open_the_gate():
    gate = LoudnessGate()
    gate.process(tone(-26, 2000) + bytes(16000))
    far_voice = (tone(-46, 180) + tone(-32, 20)) * 5
    assert is_silent(gate.process(far_voice))


def test_a_voice_inside_the_margin_passes():
    gate = LoudnessGate()
    gate.process(tone(-26, 2000))
    audio = tone(-34, 500)
    assert gate.process(audio) == audio


def test_the_gate_stays_open_for_hold_ms_after_loud_audio_so_word_ends_stay():
    gate = LoudnessGate(window_ms=10, hold_ms=100)
    gate.process(tone(-26, 2000))
    out = gate.process(tone(-46, 300))
    assert not is_silent(out[: RATE // 10 * 2])  # The first 100 ms pass
    assert is_silent(out[RATE // 10 * 2 :])


def test_lookahead_keeps_the_start_of_speech_after_a_pause():
    def speech_frames_kept(lookahead_ms):
        gate = LoudnessGate(lookahead_ms=lookahead_ms)
        gate.process(tone(-26, 2000) + bytes(32000))  # The gate learns the level of the user. Then a pause
        out = gate.process(tone(-26, 500) + bytes(32000))
        return sum(out[start : start + 320] != bytes(320) for start in range(0, len(out), 320))

    assert speech_frames_kept(0) < 50  # The level rises slowly, so the first frames of the 50 are silent
    assert speech_frames_kept(100) == 50


def test_steady_background_sound_cannot_pull_the_learned_level_down_to_its_own_level():
    gate = LoudnessGate()
    for _ in range(10):
        gate.process(tone(-45, 1400))  # A fan, all the time
        gate.process(tone(-25, 600))  # The user speaks 30% of the time
    assert gate.threshold_db > -45


def test_chunks_that_are_not_whole_frames_give_the_same_audio_as_one_chunk():
    audio = tone(-30, 200)
    gate = LoudnessGate()
    out = b"".join(gate.process(audio[start : start + 333]) for start in range(0, len(audio), 333))
    assert out == audio[: len(out)]
    assert len(audio) - len(out) < 320  # Less than one 10 ms frame waits


def test_the_learned_level_follows_the_speech_that_passes():
    gate = LoudnessGate()
    gate.process(tone(-25, 3000))
    assert gate.reference_db == pytest.approx(rms_db(tone(-25, 10)), abs=1)
    assert gate.threshold_db == pytest.approx(-37, abs=1)


def test_the_margin_must_be_positive():
    with pytest.raises(ValueError, match="Margin"):
        LoudnessGate(margin_db=0)
