"""Events that a text-to-speech provider yields to the pipeline."""

from dataclasses import dataclass

from voxchain.session import SessionClosed


@dataclass(frozen=True)
class AudioChunk:
    """Synthesized speech as raw PCM16 at the ``sample_rate`` of the provider."""

    data: bytes


@dataclass(frozen=True)
class Flushed:
    """The provider synthesized all the text before one :meth:`~voxchain.tts.base.TTSProvider.flush`."""


TTSEvent = AudioChunk | Flushed | SessionClosed
