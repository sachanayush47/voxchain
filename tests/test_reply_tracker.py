"""ReplyTracker: which replies are live, and what a barge-in cancels or hands back."""

from voxkit.core import ReplyTracker


def test_a_reply_is_active_until_its_audio_is_synthesized():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("hi")
    tracker.spoke(reply)
    tracker.finished_generating(reply)
    assert tracker.active
    tracker.synthesized()
    assert not tracker.active


def test_a_reply_that_never_spoke_ends_with_its_agent_turn():
    tracker = ReplyTracker(resume_window=2.0)
    tracker.finished_generating(tracker.start("hi"))
    assert not tracker.active


def test_barging_in_on_an_unheard_reply_hands_the_user_text_back():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("book a ticket to")
    tracker.spoke(reply)
    interruption = tracker.interrupt(now=10.0)
    assert interruption.cancelled == [reply]
    assert interruption.cancel_synthesis
    assert interruption.resume_text == "book a ticket to"
    assert not tracker.active


def test_barging_in_long_after_hearing_the_reply_is_a_real_interruption():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("what's the weather")
    tracker.spoke(reply)
    tracker.audio_played(now=1.0)
    assert tracker.interrupt(now=5.0).resume_text == ""


def test_a_reply_already_synthesized_can_still_be_resumed_while_it_plays():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("I want to")
    tracker.spoke(reply)
    tracker.audio_played(now=1.0)
    tracker.synthesized()
    tracker.finished_generating(reply)
    interruption = tracker.interrupt(now=1.5)
    assert interruption.resume_text == "I want to"
    assert not interruption.cancel_synthesis  # Nothing left for TTS to drop


def test_no_resume_window_never_merges():
    tracker = ReplyTracker(resume_window=None)
    tracker.spoke(tracker.start("hi"))
    assert tracker.interrupt(now=0.0).resume_text == ""


def test_replies_queue_behind_the_latest_one():
    tracker = ReplyTracker(resume_window=2.0)
    first = tracker.start("one")
    assert tracker.latest is first
    second = tracker.start("two")
    assert tracker.latest is second
