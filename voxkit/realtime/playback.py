"""Estimating what the user heard of the model's speech, and what to cut when they interrupt."""

from dataclasses import dataclass, field
from enum import Enum


class AudioAction(Enum):
    """What to do with an arriving chunk of the model's speech."""

    PLAY = "play"
    DROP = "drop"  # Late audio of an item already cut
    DELETE = "delete"  # First audio of an item from an interrupted reply: the user heard none of it


@dataclass(frozen=True)
class Cut:
    """An interrupted item: truncate it to ``heard_ms``, or delete it when that is 0."""

    item_id: str
    heard_ms: int


@dataclass(frozen=True)
class ResponseOutcome:
    """How a response ended, from the listener's point of view."""

    interrupted: bool
    spoke: bool


@dataclass
class ItemPlayback:
    """One spoken item: how much audio arrived, and when each chunk played."""

    response_id: str
    received_ms: float = 0.0
    chunks: list[tuple[float, float]] = field(default_factory=list)  # (start, end) on the playback clock
    done: bool = False
    handled: bool = False

    def heard_ms(self, now: float) -> float:
        """Milliseconds of this item the user has heard by ``now``."""
        return sum(min(max(now - start, 0.0), end - start) for start, end in self.chunks) * 1000


class PlaybackTracker:
    """Follows each spoken item from arrival to playback, and decides what a barge-in must cut.

    When each chunk plays comes from :class:`~voxkit.output.OutputChannel`; times
    are seconds on its clock.
    """

    def __init__(self) -> None:
        self._items: dict[str, ItemPlayback] = {}
        self._active: str | None = None
        self._discarded: set[str] = set()
        self._spoken: set[str] = set()

    def is_discarded(self, response_id: str) -> bool:
        """Whether a response was interrupted and its remaining output should be ignored."""
        return response_id in self._discarded

    def response_started(self, response_id: str) -> None:
        """A response started generating."""
        self._active = response_id

    def accept(self, response_id: str, item_id: str, duration_ms: float) -> AudioAction:
        """Decide what to do with a chunk that just arrived, and count it if it will play."""
        item = self._items.get(item_id)
        if response_id in self._discarded:
            if item is None:
                self._items[item_id] = ItemPlayback(response_id, handled=True)
                return AudioAction.DELETE
            return AudioAction.DROP
        if item is None:
            item = self._items[item_id] = ItemPlayback(response_id)
        if item.handled:
            return AudioAction.DROP
        item.received_ms += duration_ms
        self._spoken.add(response_id)
        return AudioAction.PLAY

    def played(self, item_id: str, start: float, end: float) -> None:
        """A chunk of ``item_id`` plays from ``start`` to ``end``."""
        if item := self._items.get(item_id):
            item.chunks.append((start, end))

    def barge_in(self, now: float) -> list[Cut]:
        """The user started talking: discard the reply being generated and say which items to cut."""
        if self._active:
            self._discarded.add(self._active)
        cuts = []
        for item_id, item in list(self._items.items()):
            heard = item.heard_ms(now)
            if item.handled or (item.done and heard >= item.received_ms - 1):
                if item.done:
                    del self._items[item_id]  # Fully played; nothing left to track
                continue
            item.handled = True
            if not item.done:
                self._discarded.add(item.response_id)
            cuts.append(Cut(item_id, int(heard)))
        return cuts

    def response_done(self, response_id: str) -> ResponseOutcome:
        """A response ended. Its items stop being tracked once nothing more can happen to them."""
        if response_id == self._active:
            self._active = None
        interrupted = response_id in self._discarded
        self._discarded.discard(response_id)
        for item_id, item in list(self._items.items()):
            if item.response_id == response_id:
                item.done = True
                if item.handled:
                    del self._items[item_id]
        spoke = response_id in self._spoken
        self._spoken.discard(response_id)
        return ResponseOutcome(interrupted=interrupted, spoke=spoke)
