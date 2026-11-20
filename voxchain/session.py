"""Helpers that keep the session of a provider alive when the connection drops."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import websockets

logger = logging.getLogger(__name__)

# Without a limit, a provider that rejects every connection gets new attempts for the full call.
# The first reconnect attempt has no delay, so one short drop costs the user almost nothing.
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_S = 0.5


class SessionLost(Exception):
    """All attempts to open a new session failed."""


@dataclass(frozen=True)
class SessionClosed:
    """The session of a provider stopped, and all attempts to open a new one failed. No events follow."""

    reason: str


class LiveSocket:
    """The current websocket of a session, with JSON sends that continue to work when the socket changes.

    Each send goes to the current socket. When there is no socket (during a
    reconnect), :meth:`send` discards the message and :meth:`send_reliably` waits.
    """

    def __init__(self) -> None:
        self.ws = None
        self.closed = False
        self._ready = asyncio.Event()

    @property
    def ready(self) -> bool:
        """Whether a socket is present, so that sends go out."""
        return self._ready.is_set() and self.ws is not None

    def attach(self, ws) -> None:
        """Make ``ws`` the socket that sends go to."""
        self.ws = ws
        self._ready.set()

    async def detach(self) -> None:
        """Close and forget the current socket.

        Until the next :meth:`attach`, :meth:`send_reliably` waits and :meth:`send` discards its message.
        """
        self._ready.clear()
        ws, self.ws = self.ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"LiveSocket: The socket did not close cleanly: {e!r}")

    async def close(self) -> None:
        """Detach permanently, and release all calls that wait in :meth:`send_reliably`."""
        self.closed = True
        await self.detach()
        self._ready.set()

    async def wait_ready(self) -> None:
        """Wait until there is a socket, or until the socket closes permanently."""
        await self._ready.wait()

    async def send(self, payload: dict) -> bool:
        """Send now if there is a socket, or discard the payload if not. The result shows if it went out."""
        ws = self.ws
        if not self._ready.is_set() or ws is None:
            return False
        try:
            await ws.send(json.dumps(payload))
            return True
        except websockets.ConnectionClosed:
            return False

    async def send_reliably(self, payload: dict) -> bool:
        """Send, and wait for a reconnect to complete if necessary. Fail only after the socket closes permanently."""
        while not self.closed:
            await self._ready.wait()
            ws = self.ws
            if self.closed or ws is None:
                return False
            try:
                await ws.send(json.dumps(payload))
                return True
            except websockets.ConnectionClosed:
                if self.ws is ws:
                    self._ready.clear()  # Wait for the reader to open the replacement socket
        return False


class ReconnectingSession[T]:
    """Reads events from a session, and replaces the session when it drops or when the provider asks for a new one.

    ``read`` yields the events of the current session. If ``read`` raises, the
    session dropped. If ``read`` returns normally, the session closed. In both
    cases, ``reopen`` builds the next session. :meth:`events` raises
    :class:`SessionLost` after ``max_attempts`` consecutive replacements produce
    no traffic. A replacement that the provider asks for with
    :meth:`request_replacement` is planned, so it never counts as a failure.
    """

    def __init__(
        self,
        *,
        name: str,
        read: Callable[[], AsyncIterator[T]],
        reopen: Callable[[], Awaitable[None]],
        max_attempts: int = MAX_RECONNECT_ATTEMPTS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.name = name
        self.read = read
        self.reopen = reopen
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.closed = False
        self._replacement_requested = False

    async def events(self) -> AsyncIterator[T]:
        """Yield events for as long as a session stays alive."""
        attempts = 0
        while not self.closed:
            try:
                async for event in self.read():
                    attempts = 0  # The session produces traffic again
                    yield event
            except Exception as e:
                if not self.closed:
                    logger.warning(f"{self.name}: Session lost: {e!r}")
            if self.closed:
                return

            if self._replacement_requested:
                self._replacement_requested = False
                attempts = 0
            else:
                attempts += 1
            if attempts > self.max_attempts:
                self.closed = True
                raise SessionLost(f"No working session after {self.max_attempts} attempts")
            if attempts > 1:
                await self.sleep(RECONNECT_DELAY_S)
            try:
                await self.reopen()
                logger.info(f"{self.name}: Session restored")
            except Exception as e:
                # The next read fails immediately because there is no session, and that counts as one more attempt.
                logger.warning(f"{self.name}: Reconnect attempt {attempts} failed: {e!r}")

    def request_replacement(self) -> None:
        """Make the next end of ``read`` reopen the session immediately. This replacement is not a failure."""
        self._replacement_requested = True

    def close(self) -> None:
        """Record that the session ended deliberately, so that a later drop does not cause a reconnect."""
        self.closed = True


END_OF_STREAM = object()  # The session ended deliberately


class ProviderSession[E]:
    """The parts of a websocket provider that are not vendor logic: the socket, reconnects and one event stream.

    The provider supplies the vendor-specific parts:

    - ``open`` connects and configures a socket. It raises an error if the vendor rejects it.
    - ``handle`` turns one decoded message into events.
    - ``after_connect`` runs once after each socket is live.
    - ``after_message`` runs after every message, for example to start the next request.

    Events that the provider makes itself, for example from a local VAD, go in
    through :meth:`emit` and use the same stream. :meth:`replace` deliberately
    swaps the socket. After that, no message from the old socket gets through.
    """

    def __init__(
        self,
        *,
        name: str,
        open: Callable[[], Awaitable],
        handle: Callable[[dict], list[E]],
        after_connect: Callable[[], Awaitable[None]] | None = None,
        after_message: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self.socket = LiveSocket()
        self._open = open
        self._handle = handle
        self._after_connect = after_connect
        self._after_message = after_message
        self._reconnecting = ReconnectingSession(name=name, read=self._read, reopen=self._reopen)
        self._events: asyncio.Queue = asyncio.Queue()
        self._generation = 0

    async def connect(self) -> None:
        """Open a socket and make it the socket that sends go to."""
        self.socket.attach(await self._open())
        if self._after_connect is not None:
            await self._after_connect()

    def emit(self, event: E) -> None:
        """Add an event that the provider made itself to the stream."""
        self._events.put_nowait(event)

    async def events(self) -> AsyncIterator[E | SessionClosed]:
        """Yield events for the life of the conversation. After a permanent session loss, ``SessionClosed`` is last."""
        reader = asyncio.create_task(self._forward())
        try:
            while (event := await self._events.get()) is not END_OF_STREAM:
                yield event
                if isinstance(event, SessionClosed):
                    return
        finally:
            reader.cancel()

    async def replace(self) -> None:
        """Swap the socket for a new one now, for example to discard work in progress. This is never a failure."""
        self._generation += 1
        self._reconnecting.request_replacement()
        await self.socket.detach()

    async def close(self) -> None:
        """End the session permanently. It is safe to call this more than once."""
        self._reconnecting.close()
        await self.socket.close()

    async def _forward(self) -> None:
        try:
            async for event in self._reconnecting.events():
                self._events.put_nowait(event)
        except SessionLost as e:
            self._events.put_nowait(SessionClosed(reason=str(e)))
        else:
            self._events.put_nowait(END_OF_STREAM)

    async def _reopen(self) -> None:
        await self.socket.detach()
        await self.connect()

    async def _read(self) -> AsyncIterator[E]:
        ws, generation = self.socket.ws, self._generation
        if ws is None:
            raise ConnectionError("No open socket")
        async for raw in ws:
            # A replaced socket can still deliver frames that it buffered before it closed.
            if generation != self._generation:
                return
            try:
                events = self._handle(json.loads(raw))
            except Exception:
                logger.exception(f"{self.name}: Failed to handle a message")
                continue
            for event in events:
                yield event
            if self._after_message is not None:
                await self._after_message()
