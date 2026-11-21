"""A filter that removes steady background noise with RNNoise."""


class RNNoiseFilter:
    """Removes background noise with RNNoise, a small open-source neural network that runs locally on CPU.

    It removes steady noise (fans, traffic, hum) well, and music and other voices
    less well. It adds approximately 10 ms of delay and uses approximately 7 ms of
    CPU for each 100 ms of audio. It needs the ``noise`` extra:
    ``pip install "voxchain[noise]"``.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        try:
            import numpy as np
            from pyrnnoise import RNNoise
        except ImportError as e:
            raise ImportError('RNNoiseFilter requires the noise extra: pip install "voxchain[noise]"') from e
        self.sample_rate = sample_rate
        self._np = np
        self._denoiser = RNNoise(sample_rate=sample_rate)

    def process(self, pcm: bytes) -> bytes:
        """Return the audio of ``pcm`` without background noise.

        The first chunks are shorter, because RNNoise fills its buffer first.
        """
        np = self._np
        frames = [frame for _probability, frame in self._denoiser.denoise_chunk(np.frombuffer(pcm, dtype=np.int16))]
        if not frames:
            return b""
        return np.concatenate(frames, axis=1).astype(np.int16).tobytes()
