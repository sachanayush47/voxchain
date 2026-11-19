"""ResponseGate: a requested response is created once, never over the user, never as a duplicate."""

from voxchain.realtime import ResponseGate


def test_a_request_with_nothing_running_creates_immediately():
    gate = ResponseGate()
    gate.request()
    assert gate.should_create()


def test_a_request_waits_while_a_response_runs():
    gate = ResponseGate()
    gate.response_created("r1")
    gate.request()
    assert not gate.should_create()
    gate.response_done("r1")
    assert gate.should_create()


def test_a_response_created_after_the_request_satisfies_it():
    gate = ResponseGate()
    gate.request()
    gate.response_created("r2")
    gate.response_done("r2")
    assert not gate.wanted
    assert not gate.should_create()


def test_the_users_turn_holds_the_request_until_its_automatic_response_starts():
    gate = ResponseGate()
    gate.user_started_speaking()
    gate.request()
    assert not gate.should_create()
    gate.response_created("r_auto")  # Created after the request, so it already sees the results
    assert not gate.wanted


def test_a_response_that_started_before_the_request_does_not_satisfy_it():
    gate = ResponseGate()
    gate.response_created("r1")
    gate.request()
    gate.response_done("r1")
    assert gate.should_create()


def test_only_one_create_is_in_flight():
    gate = ResponseGate()
    gate.request()
    gate.creating("e1")
    assert not gate.should_create()


def test_a_refused_create_waits_for_the_running_response_to_end():
    gate = ResponseGate()
    gate.request()
    gate.creating("e1")
    gate.refused("e1")
    assert not gate.should_create()
    gate.response_done("r_unannounced")
    assert gate.should_create()


def test_an_error_for_some_other_event_is_not_a_refusal():
    gate = ResponseGate()
    gate.request()
    gate.creating("e1")
    gate.refused("e9")
    assert not gate.should_create()  # Still in flight
    gate.create_failed()
    assert gate.should_create()


def test_a_fresh_session_keeps_an_owed_response():
    gate = ResponseGate()
    gate.response_created("r1")
    gate.request()
    gate.reset()
    assert gate.should_create()


def test_idle_means_nothing_running_in_flight_or_owed():
    gate = ResponseGate()
    assert gate.idle
    gate.request()
    assert not gate.idle
