"""Deciding when a speech-to-speech provider may ask the model for a response."""


class ResponseGate:
    """Tracks responses so a requested one is created only when it is still needed and safe to send.

    The model must never answer a turn twice or talk over the user, so a
    request is held while a response is running, while the user's turn is
    about to get its own automatic response, or after the server refused one
    until the running response ends. A request is dropped as soon as any
    response created after it starts, because that response already sees what
    the request was for.
    """

    def __init__(self) -> None:
        self.active: str | None = None
        self._pending_create: str | None = None
        self._refused = False
        self._user_turn_pending = False
        self._created = 0
        self._wanted_after: int | None = None

    @property
    def wanted(self) -> bool:
        """Whether a requested response is still owed."""
        return self._wanted_after is not None

    @property
    def idle(self) -> bool:
        """No response running, being created or owed."""
        return self.active is None and self._pending_create is None and not self.wanted

    def request(self) -> None:
        """Ask for a response that sees everything sent so far."""
        self._wanted_after = self._created

    def should_create(self) -> bool:
        """Whether to send a response request now."""
        return (
            self.wanted
            and self.active is None
            and self._pending_create is None
            and not self._refused
            and not self._user_turn_pending  # Their turn gets an automatic response that sees the same context
        )

    def creating(self, event_id: str) -> None:
        """A response request with this client event ID was just sent."""
        self._pending_create = event_id

    def create_failed(self) -> None:
        """The response request never reached the server."""
        self._pending_create = None

    def refused(self, event_id: str | None) -> None:
        """The server rejected a request, typically because a response it hadn't announced yet is running."""
        if event_id is not None and event_id == self._pending_create:
            self._pending_create = None
            self._refused = True

    def user_started_speaking(self) -> None:
        """The user's turn began; the server will create its response."""
        self._user_turn_pending = True

    def response_created(self, response_id: str) -> None:
        """A response started, whoever asked for it."""
        self.active = response_id
        self._pending_create = None
        self._user_turn_pending = False
        self._created += 1
        if self._wanted_after is not None and self._created > self._wanted_after:
            self._wanted_after = None

    def response_done(self, response_id: str) -> None:
        """A response ended."""
        if response_id == self.active:
            self.active = None
        self._refused = False

    def reset(self) -> None:
        """Forget per-session state for a fresh session. A response still owed stays owed."""
        self.active = None
        self._pending_create = None
        self._refused = False
        self._user_turn_pending = False
