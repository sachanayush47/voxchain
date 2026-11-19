"""PlaybackTracker: what the user heard of each item, and what to cut on barge-in."""

from voxkit.realtime import AudioAction, Cut, PlaybackTracker


def _playing(tracker, response_id, item_id, ms, start):
    assert tracker.accept(response_id, item_id, ms) == AudioAction.PLAY
    tracker.played(item_id, start, start + ms / 1000)


def test_barge_in_truncates_the_item_to_what_was_heard():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    _playing(tracker, "r1", "a1", 1000, start=0.0)
    assert tracker.barge_in(now=0.4) == [Cut("a1", 400)]


def test_heard_time_adds_up_across_an_items_chunks():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    _playing(tracker, "r1", "a1", 500, start=0.0)
    _playing(tracker, "r1", "a1", 500, start=0.5)
    assert tracker.barge_in(now=0.7) == [Cut("a1", 700)]


def test_a_playback_gap_does_not_count_as_heard():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    _playing(tracker, "r1", "a1", 200, start=0.0)
    _playing(tracker, "r1", "a1", 200, start=1.0)
    assert tracker.barge_in(now=1.1) == [Cut("a1", 300)]


def test_an_item_that_never_reached_the_client_is_deleted():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    assert tracker.accept("r1", "a1", 300) == AudioAction.PLAY  # Queued, not yet played
    assert tracker.barge_in(now=0.0) == [Cut("a1", 0)]


def test_a_fully_heard_finished_reply_is_not_cut():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    _playing(tracker, "r1", "a1", 200, start=0.0)
    tracker.response_done("r1")
    assert tracker.barge_in(now=5.0) == []


def test_late_audio_of_a_cut_item_is_dropped_and_new_items_of_the_cancelled_reply_deleted():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    _playing(tracker, "r1", "a1", 200, start=0.0)
    tracker.barge_in(now=0.1)
    assert tracker.accept("r1", "a1", 200) == AudioAction.DROP
    assert tracker.accept("r1", "a2", 200) == AudioAction.DELETE
    assert tracker.accept("r1", "a2", 200) == AudioAction.DROP


def test_a_response_reports_whether_it_spoke_and_whether_it_was_interrupted():
    tracker = PlaybackTracker()
    tracker.response_started("r1")
    assert tracker.response_done("r1").spoke is False
    tracker.response_started("r2")
    _playing(tracker, "r2", "a1", 200, start=0.0)
    tracker.barge_in(now=0.1)
    outcome = tracker.response_done("r2")
    assert outcome.spoke and outcome.interrupted
    assert not tracker.is_discarded("r2")  # Forgotten once finished
