"""ReconnectingSession: it recovers lost connections, and endless failure or a rejection ends in SessionLost."""

import pytest

from voxchain.session import ReconnectingSession, SessionLost, SessionRejected


async def _no_sleep(_):
    pass


async def test_events_continue_across_a_drop():
    sessions = [["a", "b", ConnectionError("Drop")], ["c"]]
    reopened = []

    async def read():
        for item in sessions.pop(0) if sessions else []:
            if isinstance(item, Exception):
                raise item
            yield item
        if not sessions:
            session.close()

    async def reopen():
        reopened.append(True)

    session = ReconnectingSession(name="test", read=read, reopen=reopen, sleep=_no_sleep)
    assert [event async for event in session.events()] == ["a", "b", "c"]
    assert reopened == [True]


async def test_a_server_that_drops_every_session_without_traffic_ends_in_session_lost():
    reopens = []

    async def read():
        raise ConnectionError("Drop")
        yield  # pragma: no cover

    async def reopen():
        reopens.append(True)

    session = ReconnectingSession(name="test", read=read, reopen=reopen, max_attempts=3, sleep=_no_sleep)
    with pytest.raises(SessionLost):
        async for _ in session.events():
            pass
    assert len(reopens) == 3


async def test_a_deliberate_close_ends_quietly():
    async def read():
        session.close()
        raise ConnectionError("Closed under us")
        yield  # pragma: no cover

    async def reopen():
        raise AssertionError("Must not reconnect after close")

    session = ReconnectingSession(name="test", read=read, reopen=reopen, sleep=_no_sleep)
    assert [event async for event in session.events()] == []


async def test_a_rejected_session_ends_in_session_lost_without_another_attempt():
    async def read():
        raise SessionRejected("(402) Credits exhausted")
        yield  # pragma: no cover

    async def reopen():
        raise AssertionError("Must not reconnect after a rejection")

    session = ReconnectingSession(name="test", read=read, reopen=reopen, sleep=_no_sleep)
    with pytest.raises(SessionLost, match="Credits exhausted"):
        async for _ in session.events():
            pass


async def test_a_rejected_reconnect_ends_in_session_lost_after_one_attempt():
    reopens = []

    async def read():
        raise ConnectionError("Drop")
        yield  # pragma: no cover

    async def reopen():
        reopens.append(True)
        raise SessionRejected("HTTP 403")

    session = ReconnectingSession(name="test", read=read, reopen=reopen, sleep=_no_sleep)
    with pytest.raises(SessionLost, match="HTTP 403"):
        async for _ in session.events():
            pass
    assert reopens == [True]
