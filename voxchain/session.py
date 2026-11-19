"""Keeping a provider's streaming session alive across drops."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import websockets

logger = logging.getLogger(__name__)

# Without a cap, a provider that rejects every connection is re-dialled for the whole call.
# The first retry skips the delay, so one transient drop costs the user almost nothing.
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_S = 0.5


class SessionLost(Exception):
    """No working session could be re-established."""


@dataclass(frozen=True)
class SessionClosed:
    """A provider's session died and could not be re-established. Nothing follows."""

    reason: str


class LiveSocket:
    """The current websocket of a session, with JSON sends that tolerate it being replaced.

    Sends go out on whichever socket is attached. While none is (a reconnect is
    under way), :meth:`send` drops the message and :meth:`send_reliably` waits.
    """

    def __init__(self) -> None:
        self.ws = None
        self.closed = False
        self._ready = asyncio.Event()

    @property
    def ready(self) -> bool:
        """Whether a socket is attached and sends will go out."""
        return self._ready.is_set() and self.ws is not None

    def attach(self, ws) -> None:
        """Make ``ws`` the socket sends go to."""
        self.ws = ws
        self._ready.set()

    async def detach(self) -> None:
        """Close and forget the current socket. Sends wait or drop until the next :meth:`attach`."""
        self._ready.clear()
        ws, self.ws = self.ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"LiveSocket: Error closing socket: {e!r}")

    async def close(self) -> None:
        """Detach for good, releasing anything waiting in :meth:`send_reliably`."""
        self.closed = True
        await self.detach()
        self._ready.set()

    async def wait_ready(self) -> None:
        """Wait until a socket is attached (or the socket is closed for good)."""
        await self._ready.wait()

    async def send(self, payload: dict) -> bool:
        """Send now if a socket is attached; otherwise drop. Returns whether it went out."""
        ws = self.ws
        if not self._ready.is_set() or ws is None:
            return False
        try:
            await ws.send(json.dumps(payload))
            return True
        except websockets.ConnectionClosed:
            return False

    async def send_reliably(self, payload: dict) -> bool:
        """Send, waiting out a reconnect if one is under way. Fails only once the socket is closed for good."""
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
                    self._ready.clear()  # Wait for the reader to bring up the replacement
        return False


class ReconnectingSession[T]:
    """Reads events from a session and replaces the session whenever it drops or asks to be replaced.

    ``read`` yields the current session's events. Raising means the session
    dropped; returning normally means it closed. Either way ``reopen`` builds the
    next one. :meth:`events` raises :class:`SessionLost` after ``max_attempts``
    replacements in a row produce no traffic. A replacement the provider asks for
    with :meth:`request_replacement` is planned, so it never counts as a failure.
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
        """Yield events for as long as a session can be kept alive."""
        attempts = 0
        while not self.closed:
            try:
                async for event in self.read():
                    attempts = 0  # The session is producing traffic again
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
                # The next read fails fast on the missing session and counts another attempt.
                logger.warning(f"{self.name}: Reconnect attempt {attempts} failed: {e!r}")

    def request_replacement(self) -> None:
        """Have the next end of ``read`` reopen the session at once, without counting it as a failure."""
        self._replacement_requested = True

    def close(self) -> None:
        """Mark the session as deliberately ended, so a drop isn't treated as one to recover from."""
        self.closed = True


END_OF_STREAM = object()  # The session ended deliberately


class ProviderSession[E]:
    """Everything a websocket provider needs around its vendor logic: the socket, reconnects and one event stream.

    The provider supplies the vendor-specific parts: ``open`` connects and
    configures a socket (raising if the vendor rejects it), ``handle`` turns one
    decoded message into events, ``after_connect`` runs once each socket is live,
    and ``after_message`` runs after every message (e.g. to start the next request).
    Events the provider raises itself (from a local VAD, say) go in through
    :meth:`emit` and share the one stream. :meth:`replace` swaps the socket on
    purpose; nothing the old socket still delivers gets through.
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
        """Open a socket and make it the one sends go to."""
        self.socket.attach(await self._open())
        if self._after_connect is not None:
            await self._after_connect()

    def emit(self, event: E) -> None:
        """Add an event the provider produced itself to the stream."""
        self._events.put_nowait(event)

    async def events(self) -> AsyncIterator[E | SessionClosed]:
        """Yield events for the life of the conversation, ending with ``SessionClosed`` if the session is lost."""
        reader = asyncio.create_task(self._forward())
        try:
            while (event := await self._events.get()) is not END_OF_STREAM:
                yield event
                if isinstance(event, SessionClosed):
                    return
        finally:
            reader.cancel()

    async def replace(self) -> None:
        """Swap the socket for a fresh one now, e.g. to abandon in-flight work. Never counts as a failure."""
        self._generation += 1
        self._reconnecting.request_replacement()
        await self.socket.detach()

    async def close(self) -> None:
        """End the session for good. Safe to call more than once."""
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
            # A replaced socket can still hand over frames it buffered before closing.
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
