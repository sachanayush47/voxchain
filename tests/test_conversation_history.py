"""ConversationHistory: turns stay in conversation order, and replay only what the user heard."""

from voxchain.realtime import ConversationHistory


def test_turns_replay_in_order_even_when_the_user_transcript_arrives_late():
    history = ConversationHistory()
    history.user_turn("u1", awaiting_transcript=True)
    history.assistant_text("a1", "Sure.")
    history.user_transcript("u1", "Can you help?")
    assert [(t.kind, t.text) for t in history.replay()] == [("user", "Can you help?"), ("assistant", "Sure.")]


def test_an_interrupted_reply_replays_only_the_heard_part():
    history = ConversationHistory()
    history.assistant_audio("a1", 2000)
    history.assistant_text("a1", "one two three four five six seven eight nine ten")
    history.heard("a1", 600)
    assert history.replay()[0].heard_text() == "one two three"


def test_removed_and_empty_turns_are_not_replayed():
    history = ConversationHistory()
    history.user_turn("u1", awaiting_transcript=True)
    history.user_transcript("u1", None)
    history.assistant_text("a1", "Hello")
    history.remove("a1")
    assert history.replay() == []


def test_a_tool_result_replays_after_its_call():
    history = ConversationHistory()
    history.function_call("f1", "c1", "get_time", "{}")
    history.function_result("c1", '"noon"')
    assert [t.kind for t in history.replay()] == ["function_call", "function_call_output"]


def test_history_is_unsettled_while_a_transcript_or_tool_result_is_outstanding():
    history = ConversationHistory()
    assert history.settled
    history.user_turn("u1", awaiting_transcript=True)
    assert not history.settled
    history.user_transcript("u1", "hi")
    history.open_call("c1")
    assert not history.settled
    history.function_result("c1", "{}")
    assert history.settled


def test_an_undeliverable_call_stops_being_owed():
    history = ConversationHistory()
    history.open_call("c1")
    history.close_call("c1")
    assert history.settled
