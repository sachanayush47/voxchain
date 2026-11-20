"""ReplyTracker: which replies are live, and what a barge-in cancels or returns for a resume."""

from voxchain.core import ReplyTracker
from voxchain.core.replies import Reply


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
    tracker.audio_played(reply, now=1.0)
    assert tracker.interrupt(now=5.0).resume_text == ""


def test_a_reply_already_synthesized_can_still_be_resumed_while_it_plays():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("I want to")
    tracker.spoke(reply)
    tracker.audio_played(reply, now=1.0)
    tracker.synthesized()
    tracker.finished_generating(reply)
    interruption = tracker.interrupt(now=1.5)
    assert interruption.resume_text == "I want to"
    assert not interruption.cancel_synthesis  # TTS has nothing left to discard


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


def test_the_heard_text_of_a_complete_reply_follows_its_audio():
    reply = Reply("hi", spoken=["The weather today is sunny and warm."], audio_seconds=4.0, complete=True)
    reply.first_heard_at = 0.0
    assert reply.heard_text(now=2.0) == "The weather today"
    assert reply.heard_text(now=9.0) == "The weather today is sunny and warm."


def test_the_heard_text_of_a_reply_still_in_synthesis_uses_a_typical_speech_rate():
    reply = Reply("hi", spoken=["The weather today is sunny and warm."])
    reply.first_heard_at = 0.0
    assert reply.heard_text(now=1.0) == "The weather"  # 15 characters a second, cut at a word


def test_a_barge_in_reports_the_reply_the_user_heard_and_what_they_heard():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("what's the weather")
    tracker.spoke(reply)
    reply.spoken.append("It is sunny and warm today in the whole city.")
    tracker.audio_played(reply, now=0.0)
    interruption = tracker.interrupt(now=1.0 + 2.0)
    assert interruption.heard is reply
    assert interruption.heard_text.startswith("It is sunny")
    assert interruption.resumed == []


def test_a_reply_that_is_not_resumable_never_gives_its_text_back():
    tracker = ReplyTracker(resume_window=2.0)
    tracker.spoke(tracker.start("(Ask the user to repeat)", resumable=False))
    assert tracker.interrupt(now=0.0).resume_text == ""


def test_a_reply_waiting_for_a_tool_call_blocks_interruptions():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("book a table")
    assert not tracker.tool_running
    reply.tool_calls = 1
    assert tracker.tool_running


def test_audio_that_plays_after_synthesis_ended_still_counts_for_its_reply():
    tracker = ReplyTracker(resume_window=2.0)
    reply = tracker.start("hi")
    tracker.spoke(reply)
    owner = tracker.audio_received(3.0)
    tracker.synthesized()  # TTS is faster than playback
    tracker.audio_played(owner, now=1.0)
    assert reply.first_heard_at == 1.0
    assert tracker.interrupt(now=5.0).heard is reply
