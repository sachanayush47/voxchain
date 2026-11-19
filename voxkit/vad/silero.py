"""Local voice activity detection with the Silero VAD model.

`Silero VAD <https://github.com/snakers4/silero-vad>`_ is a small (~2 MB,
MIT-licensed) neural voice activity detector that runs on CPU via ONNX Runtime
in well under a millisecond per 32 ms window.

voxkit has no pipeline-level VAD: STT providers report voice activity
themselves. :class:`SileroVAD` exists for providers whose backend can't, such
as :class:`~voxkit.stt.openai.OpenAISTTProvider`, which runs it on the input
audio to emit ``SPEECH_START``/``SPEECH_END`` and to decide when to commit a
turn.

Requires the ``vad`` extra: ``pip install "voxkit[vad]"``.
"""

from typing import Literal

SILERO_VAD_REPO = "onnx-community/silero-vad"
"""Hugging Face repo the model is downloaded from."""

SILERO_VAD_FILENAME = "onnx/model.onnx"
"""Model file within :data:`SILERO_VAD_REPO` (the full-precision Silero VAD v5 build)."""

SILERO_VAD_REVISION = "e71cae966052b992a7eca6b17738916ce0eca4ec"
"""Pinned commit of :data:`SILERO_VAD_REPO`, so a changed upload can't silently change behaviour."""

SileroSampleRate = Literal[8000, 16000]
"""Sample rates the Silero model supports."""


class SileroVAD:
    """Scores fixed-size windows of 16-bit PCM audio for speech probability.

    Stateful: the model carries context from one window to the next, so feed
    one stream's windows in order and call :meth:`reset` between unrelated
    streams.

    Example:
        >>> vad = SileroVAD(sample_rate=16000)
        >>> probability = vad.speech_probability(window)  # Window is vad.window_bytes long
    """

    def __init__(self, sample_rate: SileroSampleRate = 16000, model_path: str | None = None) -> None:
        """Load the model, downloading it from Hugging Face on first use."""
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sample_rate}")
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            msg = 'SileroVAD requires the vad extra: pip install "voxkit[vad]"'
            raise ImportError(msg) from e

        self._np = np
        self.sample_rate = sample_rate
        self.window_samples = 512 if sample_rate == 16000 else 256
        """Samples per window that :meth:`speech_probability` expects (32 ms at either rate)."""
        self.window_bytes = self.window_samples * 2
        """Bytes per window: :attr:`window_samples` 16-bit samples."""
        # Silero v5 expects each window prefixed with the tail of the previous one.
        self._context_samples = 64 if sample_rate == 16000 else 32

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        path = model_path or hf_hub_download(SILERO_VAD_REPO, SILERO_VAD_FILENAME, revision=SILERO_VAD_REVISION)
        self._session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget the model's carried-over state, e.g. before scoring an unrelated stream."""
        np = self._np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_samples), dtype=np.float32)

    def speech_probability(self, window: bytes) -> float:
        """Score one window of audio."""
        if len(window) != self.window_bytes:
            raise ValueError(f"Window must be {self.window_bytes} bytes, got {len(window)}")
        np = self._np
        samples = np.frombuffer(window, dtype="<i2").astype(np.float32).reshape(1, -1) / 32768.0
        audio = np.concatenate([self._context, samples], axis=1)
        output, self._state = self._session.run(None, {"input": audio, "state": self._state, "sr": self._sr})
        self._context = audio[:, -self._context_samples :]
        return float(output[0][0])
