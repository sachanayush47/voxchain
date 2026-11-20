"""OpenAIRealtimeProvider sessions: handshake, reconnect with history, rotation, and response requests."""

import asyncio

import pytest

from voxchain.realtime import OpenAIRealtimeProvider, ResponseDone, SessionClosed


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def server(openai_server):
    return openai_server


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make(server, clock):
    def make(**kwargs):
        return OpenAIRealtimeProvider(api_key="test", connect=server.connect, clock=clock, **kwargs)

    return make


def _reader(provider):
    events = []

    async def read():
        async for event in provider.receive_events():
            events.append(event)

    return events, asyncio.create_task(read())


def _converse(socket):
    socket.push(type="input_audio_buffer.committed", item_id="u1")
    socket.push(type="conversation.item.input_audio_transcription.completed", item_id="u1", transcript="Hi")
    socket.push(type="response.created", response={"id": "r1"})
    item = {"id": "a1", "type": "message", "role": "assistant", "content": [{"transcript": "Hello!"}]}
    socket.push(type="response.output_item.done", item=item)
    socket.push(type="response.done", response={"id": "r1", "status": "completed", "output": []})


async def test_a_rejected_config_fails_at_connect(server, make, clock, settle):
    server.reject = True
    with pytest.raises(ConnectionError, match="Bad voice"):
        await make().connect()
    assert server.sockets[0].closed


async def test_a_dropped_session_reconnects_and_replays_the_conversation(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    _converse(server.sockets[0])
    await settle()
    server.sockets[0].drop()
    await settle()

    replayed = [m["item"] for m in server.sockets[1].of_type("conversation.item.create")]
    assert [(i["role"], i["content"][0]["text"]) for i in replayed] == [("user", "Hi"), ("assistant", "Hello!")]
    await provider.disconnect()
    await task


async def test_a_response_lost_in_a_drop_still_gets_its_response_done(server, make, clock, settle):
    provider = make()
    await provider.connect()
    events, task = _reader(provider)
    server.sockets[0].push(type="response.created", response={"id": "r1"})
    await settle()
    server.sockets[0].drop()
    await settle()
    assert ResponseDone(response_id="r1", status="failed") in events
    await provider.disconnect()
    await task


async def test_a_failed_reconnect_keeps_the_history_for_the_next_attempt(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    _converse(server.sockets[0])
    await settle()
    server.rejections = 2
    server.sockets[0].drop()
    await settle()

    last = server.sockets[-1]
    assert len(last.of_type("conversation.item.create")) == 2
    assert all(s.closed for s in server.sockets[1:-1])
    await provider.disconnect()
    await task


async def test_reconnecting_gives_up_with_session_closed(server, make, clock, settle):
    provider = make()
    await provider.connect()
    events, task = _reader(provider)
    server.refuse = True
    server.sockets[0].drop()
    await asyncio.wait_for(task, 1)
    assert isinstance(events[-1], SessionClosed)


async def test_a_tool_result_sent_mid_reconnect_lands_after_the_replayed_call(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    call = {"id": "f1", "type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"}
    server.sockets[0].push(type="response.created", response={"id": "r1"})
    server.sockets[0].push(type="response.output_item.done", item=call)
    server.sockets[0].push(type="response.done", response={"id": "r1", "status": "completed", "output": [call]})
    await settle()
    server.gate = asyncio.Event()
    server.sockets[0].drop()
    await settle()
    sending = asyncio.create_task(provider.send_function_result("c1", '"12:00"'))
    await settle()
    assert not sending.done()  # It waits for a session and does not discard the result
    server.gate.set()
    await asyncio.wait_for(sending, 1)

    creates = [m["item"]["type"] for m in server.sockets[-1].of_type("conversation.item.create")]
    assert creates == ["function_call", "function_call_output"]
    await provider.disconnect()
    await task


async def test_an_idle_session_is_replaced_before_the_time_limit_but_not_mid_tool_call(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    clock.now = 56 * 60  # After the 55-minute point for session replacement
    call = {"type": "function_call", "call_id": "c1", "name": "get_time", "arguments": "{}"}
    server.sockets[0].push(type="response.done", response={"id": "r1", "status": "completed", "output": [call]})
    await settle()
    assert len(server.sockets) == 1  # The provider still owes a tool result

    await provider.send_function_result("c1", "{}")
    server.sockets[0].push(type="rate_limits.updated")
    await settle()
    assert len(server.sockets) == 2
    await provider.disconnect()
    await task


async def test_requested_responses_wait_for_the_active_one(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    socket = server.sockets[0]
    socket.push(type="response.created", response={"id": "r1"})
    await settle()
    await provider.request_response()
    assert socket.of_type("response.create") == []
    socket.push(type="response.done", response={"id": "r1", "status": "completed", "output": []})
    await settle()
    assert len(socket.of_type("response.create")) == 1
    await provider.disconnect()
    await task


async def test_no_extra_response_when_a_later_one_already_sees_the_results(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    socket = server.sockets[0]
    socket.push(type="input_audio_buffer.speech_started", item_id="u1")
    await settle()
    await provider.request_response()  # The user speaks, so the provider waits
    socket.push(type="input_audio_buffer.speech_stopped", item_id="u1")
    socket.push(type="response.created", response={"id": "r2"})  # The automatic response to the user turn
    socket.push(type="response.done", response={"id": "r2", "status": "completed", "output": []})
    await settle()
    assert socket.of_type("response.create") == []
    await provider.disconnect()
    await task


async def test_tool_results_still_get_spoken_when_the_user_spoke_while_the_tool_ran(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    socket = server.sockets[0]
    # The user says "hello?" during a slow tool. Its automatic response starts and ends before the result exists.
    socket.push(type="response.created", response={"id": "r2"})
    socket.push(type="response.done", response={"id": "r2", "status": "completed", "output": []})
    await settle()
    await provider.request_response()
    await settle()
    assert len(socket.of_type("response.create")) == 1
    await provider.disconnect()
    await task


async def test_a_refused_create_is_retried_when_the_blocking_response_ends(server, make, clock, settle):
    provider = make()
    await provider.connect()
    _events, task = _reader(provider)
    socket = server.sockets[0]
    await provider.request_response()
    create = socket.of_type("response.create")[0]
    socket.push(
        type="error", error={"code": "conversation_already_has_active_response", "event_id": create["event_id"]}
    )
    socket.push(type="response.created", response={"id": "r_other"})
    await settle()
    # The response r_other started after the request, so it sees the results. No second create follows.
    socket.push(type="response.done", response={"id": "r_other", "status": "completed", "output": []})
    await settle()
    assert len(socket.of_type("response.create")) == 1
    await provider.disconnect()
    await task
