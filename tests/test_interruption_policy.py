"""InterruptionPolicy: enough words or a stop word interrupt the agent, and noise or backchannels do not."""

import pytest

from voxchain.core.interruptions import InterruptionPolicy, count_words


@pytest.mark.parametrize("text", ["what about tomorrow", "Stop!", "no, wait", "Hold on.", "NO"])
def test_real_interruptions_are_confirmed(text):
    assert InterruptionPolicy(min_words=3).confirms(text)


@pytest.mark.parametrize("text", ["uh-huh", "okay", "yeah sure", "", "...", "hmm right"])
def test_backchannels_and_noise_are_not(text):
    assert not InterruptionPolicy(min_words=3).confirms(text)


def test_words_are_counted_without_punctuation_and_each_cjk_character_counts():
    assert count_words("Hello, world!") == 2
    assert count_words("你好吗") == 3
    assert count_words("ok 你好") == 3


def test_short_transcripts_add_up_within_the_window():
    policy = InterruptionPolicy(min_words=3, window=2.0)
    assert policy.hear("I said", now=0.0) is None
    assert policy.hear("Paris", now=1.0) == "I said Paris"


def test_an_old_short_transcript_does_not_count_after_the_window():
    policy = InterruptionPolicy(min_words=3, window=2.0)
    assert policy.hear("I said", now=0.0) is None
    assert policy.hear("Paris", now=5.0) is None


def test_reset_forgets_pending_words():
    policy = InterruptionPolicy(min_words=3)
    policy.hear("I said", now=0.0)
    policy.reset()
    assert policy.hear("Paris", now=0.5) is None


def test_a_minimum_below_one_is_rejected():
    with pytest.raises(ValueError):
        InterruptionPolicy(min_words=0)
