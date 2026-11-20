"""Decides when a speech-to-speech provider can ask the model for a response."""


class ResponseGate:
    """Tracks responses, so that the provider creates a requested response only if it is necessary and safe.

    The model must not answer a turn two times or speak over the user. Thus the
    gate holds a request in these conditions:

    - A response runs.
    - The user turn will get its own automatic response.
    - The server refused a request, until the active response ends.

    The gate discards a request when a response that was created after the
    request starts. That response already sees what the request was for.
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
        """Whether the provider still owes a requested response."""
        return self._wanted_after is not None

    @property
    def idle(self) -> bool:
        """Whether there is no active, pending or owed response."""
        return self.active is None and self._pending_create is None and not self.wanted

    def request(self) -> None:
        """Ask for a response that sees all that the provider sent until now."""
        self._wanted_after = self._created

    def should_create(self) -> bool:
        """Whether to send a response request now."""
        return (
            self.wanted
            and self.active is None
            and self._pending_create is None
            and not self._refused
            and not self._user_turn_pending  # The user turn gets an automatic response that sees the same context
        )

    def creating(self, event_id: str) -> None:
        """The provider sent a response request with this client event ID."""
        self._pending_create = event_id

    def create_failed(self) -> None:
        """The response request did not get to the server."""
        self._pending_create = None

    def refused(self, event_id: str | None) -> None:
        """The server rejected a request, usually because a response runs that the server did not announce yet."""
        if event_id is not None and event_id == self._pending_create:
            self._pending_create = None
            self._refused = True

    def user_started_speaking(self) -> None:
        """The user turn started. The server will create its response."""
        self._user_turn_pending = True

    def response_created(self, response_id: str) -> None:
        """A response started. It does not matter who asked for it."""
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
        """Forget the session state before a new session. A response that is still owed stays owed."""
        self.active = None
        self._pending_create = None
        self._refused = False
        self._user_turn_pending = False
