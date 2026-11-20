"""The conversation that a speech-to-speech provider replays into a new session."""

from dataclasses import dataclass
from typing import Literal

TurnKind = Literal["user", "assistant", "function_call", "function_call_output"]


@dataclass
class Turn:
    """One replayable conversation item, in vendor-neutral form."""

    kind: TurnKind
    item_id: str | None = None
    text: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = "{}"
    output: str = ""
    audio_ms: float = 0.0
    heard_ms: float | None = None

    def heard_text(self) -> str:
        """The text, cut in proportion to the part of the audio that the user heard."""
        words = self.text.split()
        if self.heard_ms is not None and self.audio_ms:
            words = words[: round(len(words) * min(1.0, self.heard_ms / self.audio_ms))]
        return " ".join(words)


class ConversationHistory:
    """Keeps turns in conversation order when their parts arrive, and knows when nothing is pending.

    A user turn gets its position in the order when the server commits its
    audio. The turn gets its text when its transcript arrives. Thus a late
    transcript still replays before the assistant turn that it caused.
    """

    def __init__(self) -> None:
        self._turns: list[Turn] = []
        self._by_item: dict[str, Turn] = {}
        self._awaiting_transcripts: set[str] = set()
        self._open_calls: set[str] = set()

    @property
    def settled(self) -> bool:
        """Whether no transcript is pending and no tool result is owed, so a session replacement loses nothing."""
        return not self._awaiting_transcripts and not self._open_calls

    def _turn(self, item_id: str, kind: TurnKind) -> Turn:
        if item_id not in self._by_item:
            self._by_item[item_id] = Turn(kind=kind, item_id=item_id)
            self._turns.append(self._by_item[item_id])
        return self._by_item[item_id]

    def user_turn(self, item_id: str, *, awaiting_transcript: bool) -> None:
        """The server committed the audio of a user turn."""
        self._turn(item_id, "user")
        if awaiting_transcript:
            self._awaiting_transcripts.add(item_id)

    def user_transcript(self, item_id: str, text: str | None) -> None:
        """The transcript of a user turn arrived, or ``None`` if the transcription failed."""
        self._awaiting_transcripts.discard(item_id)
        if text:
            self._turn(item_id, "user").text = text

    def assistant_audio(self, item_id: str, ms: float) -> None:
        """Audio for an assistant item arrived."""
        self._turn(item_id, "assistant").audio_ms += ms

    def assistant_text(self, item_id: str, text: str) -> None:
        """An assistant item finished, with its full transcript."""
        self._turn(item_id, "assistant").text = text

    def function_call(self, item_id: str, call_id: str, name: str, arguments: str) -> None:
        """The model made a tool call."""
        turn = self._turn(item_id, "function_call")
        turn.call_id, turn.name, turn.arguments = call_id, name, arguments

    def open_call(self, call_id: str) -> None:
        """A tool call now owes a result."""
        self._open_calls.add(call_id)

    def function_result(self, call_id: str, output: str) -> None:
        """The provider sent a tool result. The history records it only after the send.

        Thus a replay never puts the result before its call.
        """
        self._open_calls.discard(call_id)
        self._turns.append(Turn(kind="function_call_output", call_id=call_id, output=output))

    def close_call(self, call_id: str) -> None:
        """Close a call without a result, for example when the provider cannot send it."""
        self._open_calls.discard(call_id)

    def heard(self, item_id: str, ms: float) -> None:
        """The user heard only the first ``ms`` of an assistant item."""
        if turn := self._by_item.get(item_id):
            turn.heard_ms = ms

    def remove(self, item_id: str) -> None:
        """Remove an item, for example an assistant item when the user heard no part of it."""
        if turn := self._by_item.pop(item_id, None):
            self._turns.remove(turn)

    def replay(self) -> list[Turn]:
        """The turns to restore, in order. The list skips empty user and assistant turns."""
        return [turn for turn in self._turns if _worth_replaying(turn)]


def _worth_replaying(turn: Turn) -> bool:
    if turn.kind == "user":
        return bool(turn.text)  # A turn whose transcript never arrived has nothing to restore
    if turn.kind == "assistant":
        return bool(turn.heard_text())
    return True
