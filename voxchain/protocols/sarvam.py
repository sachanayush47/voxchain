"""The vocabulary of the Sarvam AI websocket API, which the Sarvam STT and TTS providers share."""

import urllib.parse
from collections.abc import Callable
from enum import StrEnum

from voxchain.protocols import compact

STT_URL = "wss://api.sarvam.ai/speech-to-text/ws"
TTS_URL = "wss://api.sarvam.ai/text-to-speech/ws"


class STTServerMessage(StrEnum):
    """Message types that the speech-to-text socket sends."""

    DATA = "data"
    EVENTS = "events"
    ERROR = "error"


class STTSignal(StrEnum):
    """Types of voice-activity signal in an ``events`` message."""

    START_SPEECH = "START_SPEECH"
    END_SPEECH = "END_SPEECH"


class STTClientMessage(StrEnum):
    """Message types that voxchain sends to the speech-to-text socket. Audio messages have no type."""

    FLUSH = "flush"


class TTSClientMessage(StrEnum):
    """Message types that voxchain sends to the text-to-speech socket."""

    CONFIG = "config"
    TEXT = "text"
    FLUSH = "flush"


class TTSServerMessage(StrEnum):
    """Message types that the text-to-speech socket sends."""

    AUDIO = "audio"
    EVENT = "event"
    ERROR = "error"


class TTSEventType(StrEnum):
    """Event types in an ``event`` message."""

    FINAL = "final"


async def open_socket(connect: Callable, url: str, *, api_key: str, **query: str | int | float | bool | None):
    """Connect to a Sarvam socket. Sarvam's defaults apply to ``None`` query values."""
    # The Sarvam query parser accepts only lowercase booleans.
    params = {
        key: ("true" if value else "false") if isinstance(value, bool) else str(value)
        for key, value in compact(query).items()
    }
    return await connect(
        f"{url}?{urllib.parse.urlencode(params)}",
        additional_headers={"Api-Subscription-Key": api_key},
        max_size=None,
    )
