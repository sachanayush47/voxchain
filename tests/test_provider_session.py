"""ProviderSession: one event stream across reconnects, with local events merged in and replaced sockets silent."""

import asyncio

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

from voxchain.session import ProviderSession, SessionClosed, SessionRejected


def _session(server, **hooks):
    opened = []

    async def open():
        socket = await server.connect("wss://example")
        opened.append(socket)
        return socket

    def handle(message):
        return [message["say"]] if "say" in message else []

    return ProviderSession(name="test", open=open, handle=handle, **hooks), opened


async def _collect(session, events):
    async for event in session.events():
        events.append(event)


async def test_messages_become_events_and_local_events_share_the_stream(plain_server, settle):
    session, opened = _session(plain_server)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    session.emit("from the provider")
    opened[0].push(say="from the server")
    await settle()
    assert events == ["from the provider", "from the server"]
    await session.close()
    await asyncio.wait_for(reader, 1)  # A deliberate close ends the stream quietly


async def test_a_dropped_socket_is_reopened_and_after_connect_runs_each_time(plain_server, settle):
    connects = []

    async def after_connect():
        connects.append(True)

    session, opened = _session(plain_server, after_connect=after_connect)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    opened[0].drop()
    await settle()
    opened[1].push(say="back")
    await settle()
    assert events == ["back"]
    assert len(connects) == 2
    await session.close()
    await reader


async def test_nothing_from_a_replaced_socket_gets_through(plain_server, settle):
    session, opened = _session(plain_server)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    opened[0].push(say="stale")  # The old socket buffered this message
    await session.replace()
    await settle()
    opened[1].push(say="fresh")
    await settle()
    assert events == ["fresh"]
    await session.close()
    await reader


async def test_a_handler_bug_skips_the_message_instead_of_dropping_the_session(plain_server, settle):
    def handle(message):
        if message.get("bad"):
            raise KeyError("Oops")
        return [message["say"]]

    async def open():
        return await plain_server.connect("wss://example")

    session = ProviderSession(name="test", open=open, handle=handle)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    plain_server.sockets[0].push(bad=True)
    plain_server.sockets[0].push(say="next")
    await settle()
    assert events == ["next"]
    assert len(plain_server.sockets) == 1
    await session.close()
    await reader


async def test_after_message_runs_after_each_message(plain_server, settle):
    seen = []

    async def after_message():
        seen.append("after")

    session, opened = _session(plain_server, after_message=after_message)
    await session.connect()
    reader = asyncio.create_task(_collect(session, []))
    opened[0].push(say="a")
    opened[0].push(note="no events")
    await settle()
    assert seen == ["after", "after"]
    await session.close()
    await reader


async def test_losing_the_session_for_good_ends_the_stream_with_session_closed(plain_server):
    session, opened = _session(plain_server)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    plain_server.refuse = True
    opened[0].drop()
    await asyncio.wait_for(reader, 1)
    assert isinstance(events[-1], SessionClosed)


async def test_an_http_402_on_reconnect_ends_the_stream_at_once(plain_server):
    session, opened = _session(plain_server)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    plain_server.refuse_with = websockets.InvalidStatus(Response(402, "Payment Required", Headers()))
    opened[0].drop()
    await asyncio.wait_for(reader, 1)
    assert events == [SessionClosed(reason="Session rejected: HTTP 402")]
    assert len(opened) == 1


async def test_a_close_that_the_rejection_hook_flags_ends_the_stream_without_a_reconnect(plain_server):
    session, opened = _session(plain_server, rejection=lambda e: "Bad parameter" if "4000" in repr(e) else None)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    opened[0].close_with(4000, "Unsupported model")
    await asyncio.wait_for(reader, 1)
    assert events == [SessionClosed(reason="Session rejected: Bad parameter")]
    assert len(opened) == 1


async def test_a_handler_that_raises_session_rejected_ends_the_stream(plain_server):
    def handle(message):
        raise SessionRejected(message["error"])

    async def open():
        return await plain_server.connect("wss://example")

    session = ProviderSession(name="test", open=open, handle=handle)
    await session.connect()
    events = []
    reader = asyncio.create_task(_collect(session, events))
    plain_server.sockets[0].push(error="No credits")
    await asyncio.wait_for(reader, 1)
    assert events == [SessionClosed(reason="Session rejected: No credits")]
    assert len(plain_server.sockets) == 1
