"""Events a text-to-speech provider yields to the pipeline."""

from dataclasses import dataclass

from voxchain.session import SessionClosed


@dataclass(frozen=True)
class AudioChunk:
    """Synthesized speech: raw PCM16 at the provider's ``sample_rate``."""

    data: bytes


@dataclass(frozen=True)
class Flushed:
    """Everything spoken before one :meth:`~voxchain.tts.base.TTSProvider.flush` has been synthesized."""


TTSEvent = AudioChunk | Flushed | SessionClosed
