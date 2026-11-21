"""The interface of an audio filter, and a chain that runs filters one after another."""

from collections.abc import Sequence
from typing import Protocol


class AudioFilter(Protocol):
    """Anything that changes PCM16 input audio chunk by chunk, for example noise suppression.

    The output can be shorter or longer than the input, because a filter can buffer audio.
    """

    def process(self, pcm: bytes) -> bytes:
        """Return the filtered audio for ``pcm``."""
        ...


class FilterChain:
    """Runs audio filters one after another. The output of each filter is the input of the next filter.

    The chain stops for a chunk when a filter returns no audio, for example while
    the filter fills its buffer. A chain is also an :class:`AudioFilter`. With no
    filters, it returns the audio without a change.
    """

    def __init__(self, filters: Sequence[AudioFilter] = ()) -> None:
        self.filters = list(filters)

    def process(self, pcm: bytes) -> bytes:
        """Return ``pcm`` after all the filters."""
        for audio_filter in self.filters:
            if not pcm:
                break
            pcm = audio_filter.process(pcm)
        return pcm
