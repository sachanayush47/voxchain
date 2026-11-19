"""Shared test doubles: a fake websocket server that providers connect to instead of the network."""

import asyncio
import json

import pytest
import websockets

import voxkit.session

CLOSE = object()


class FakeSocket:
    def __init__(self, server, url, headers):
        self.server = server
        self.url = url
        self.headers = headers
        self.sent: list[dict] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed = False
        if server.openai:
            self.push(type="session.created")

    def push(self, **message):
        self.incoming.put_nowait(json.dumps(message))

    def drop(self):
        self.incoming.put_nowait(websockets.ConnectionClosedError(None, None))

    async def send(self, raw):
        if self.closed:
            raise websockets.ConnectionClosedError(None, None)
        message = json.loads(raw)
        self.sent.append(message)
        if self.server.openai and message.get("type") == "session.update":
            if self.server.rejections:
                self.server.rejections -= 1
                self.push(type="error", error={"code": "invalid_value", "message": "Bad voice"})
            elif self.server.reject:
                self.push(type="error", error={"code": "invalid_value", "message": "Bad voice"})
            else:
                self.push(type="session.updated")

    async def recv(self):
        item = await self.incoming.get()
        if isinstance(item, Exception):
            raise item
        return item

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is CLOSE:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(CLOSE)

    def of_type(self, message_type):
        return [m for m in self.sent if m.get("type") == message_type]


class FakeServer:
    """Hands out FakeSockets. ``openai=True`` adds OpenAI's session.created/session.updated handshake."""

    def __init__(self, *, openai: bool):
        self.openai = openai
        self.sockets: list[FakeSocket] = []
        self.reject = False
        self.refuse = False
        self.rejections = 0  # Reject only this many handshakes, then accept
        self.gate: asyncio.Event | None = None  # When set, connects wait for it

    async def connect(self, url, additional_headers=None, **kwargs):
        if self.gate is not None:
            await self.gate.wait()
        if self.refuse:
            raise OSError("Network down")
        socket = FakeSocket(self, url, additional_headers or {})
        self.sockets.append(socket)
        return socket


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.fixture
def settle():
    return _settle


@pytest.fixture
def no_reconnect_delay(monkeypatch):
    monkeypatch.setattr(voxkit.session, "RECONNECT_DELAY_S", 0)


@pytest.fixture
def openai_server(no_reconnect_delay):
    return FakeServer(openai=True)


@pytest.fixture
def plain_server(no_reconnect_delay):
    return FakeServer(openai=False)
