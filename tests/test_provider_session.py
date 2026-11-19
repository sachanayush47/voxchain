"""ProviderSession: one event stream across reconnects, local events merged in, replaced sockets silenced."""

import asyncio

from voxchain.session import ProviderSession, SessionClosed


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
    opened[0].push(say="stale")  # Buffered on the old socket
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
