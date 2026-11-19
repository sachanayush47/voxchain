"""VoiceGate: VAD probabilities become speech segments with pre-roll, debounced on both edges."""

import pytest

from voxkit.vad import SegmentAudio, SegmentEnded, SegmentStarted, VoiceGate


class ScriptedVAD:
    sample_rate = 16000
    window_bytes = 512 * 2  # 32 ms windows

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def speech_probability(self, window):
        return self.probabilities.pop(0)


def _windows(*labels):
    return b"".join(bytes([label]) * ScriptedVAD.window_bytes for label in labels)


def _gate(probabilities, **kwargs):
    return VoiceGate(ScriptedVAD(probabilities), start_ms=64, stop_ms=96, prefix_ms=32, **kwargs)


def test_speech_opens_a_segment_only_after_the_start_debounce_and_carries_its_pre_roll():
    gate = _gate([0.1, 0.9, 0.9, 0.9])
    segments = gate.push(_windows(1, 2, 3, 4))
    assert [type(s) for s in segments] == [SegmentStarted, SegmentAudio]
    assert segments[0].audio == _windows(1, 2, 3)  # Pre-roll window plus the two that confirmed speech


def test_a_cough_shorter_than_the_start_debounce_opens_nothing():
    assert _gate([0.9, 0.1, 0.9, 0.1]).push(_windows(1, 2, 3, 4)) == []


def test_a_segment_closes_after_enough_silence_and_survives_a_short_pause():
    gate = _gate([0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    kinds = [type(s) for s in gate.push(_windows(*range(8)))]
    assert kinds == [SegmentStarted, SegmentAudio, SegmentAudio, SegmentAudio, SegmentAudio, SegmentAudio, SegmentEnded]
    assert not gate.speaking


def test_borderline_probabilities_keep_the_current_state():
    gate = _gate([0.9, 0.9, 0.4, 0.4, 0.4, 0.4])  # Between the silence (0.35) and speech (0.5) thresholds
    kinds = [type(s) for s in gate.push(_windows(*range(6)))]
    assert SegmentEnded not in kinds


def test_audio_split_across_pushes_is_scored_in_whole_windows():
    gate = _gate([0.9, 0.9])
    audio = _windows(1, 2)
    assert gate.push(audio[:700]) == []
    assert [type(s) for s in gate.push(audio[700:])] == [SegmentStarted]


def test_an_unusable_threshold_is_rejected():
    with pytest.raises(ValueError):
        _gate([], threshold=0.1)
