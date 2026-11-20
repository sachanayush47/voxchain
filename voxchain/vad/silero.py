"""Local voice activity detection with the Silero VAD model.

`Silero VAD <https://github.com/snakers4/silero-vad>`_ is a small neural detector
for voice activity (approximately 2 MB, MIT license). It runs on CPU with ONNX
Runtime in much less than a millisecond for each 32 ms window.

voxchain has no pipeline-level VAD: STT providers report voice activity
themselves. :class:`SileroVAD` is for providers whose backend cannot do this,
such as :class:`~voxchain.stt.openai.OpenAISTTProvider`. That provider runs it
on the input audio to send ``SPEECH_START``/``SPEECH_END`` and to decide when
to commit a turn.

This module requires the ``vad`` extra: ``pip install "voxchain[vad]"``.
"""

from typing import Literal

SILERO_VAD_REPO = "onnx-community/silero-vad"
"""The Hugging Face repo from which voxchain downloads the model."""

SILERO_VAD_FILENAME = "onnx/model.onnx"
"""The model file in :data:`SILERO_VAD_REPO` (the full-precision build of Silero VAD v5)."""

SILERO_VAD_REVISION = "e71cae966052b992a7eca6b17738916ce0eca4ec"
"""The pinned commit of :data:`SILERO_VAD_REPO`, so that a new upload cannot change the behaviour without notice."""

SileroSampleRate = Literal[8000, 16000]
"""The sample rates that the Silero model supports."""


class SileroVAD:
    """Scores fixed-size windows of 16-bit PCM audio for speech probability.

    The model keeps state: it carries context from one window to the next.
    Thus, give it the windows of one stream in order. Call :meth:`reset`
    between unrelated streams.

    Example:
        >>> vad = SileroVAD(sample_rate=16000)
        >>> probability = vad.speech_probability(window)  # Window is vad.window_bytes long
    """

    def __init__(self, sample_rate: SileroSampleRate = 16000, model_path: str | None = None) -> None:
        """Load the model, and download it from Hugging Face at first use."""
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sample_rate}")
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            msg = 'SileroVAD requires the vad extra: pip install "voxchain[vad]"'
            raise ImportError(msg) from e

        self._np = np
        self.sample_rate = sample_rate
        self.window_samples = 512 if sample_rate == 16000 else 256
        """The number of samples per window that :meth:`speech_probability` expects (32 ms at each rate)."""
        self.window_bytes = self.window_samples * 2
        """Bytes per window: :attr:`window_samples` 16-bit samples."""
        # Silero v5 expects the tail of the previous window before each window.
        self._context_samples = 64 if sample_rate == 16000 else 32

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        path = model_path or hf_hub_download(SILERO_VAD_REPO, SILERO_VAD_FILENAME, revision=SILERO_VAD_REVISION)
        self._session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget the state that the model carried over, for example before you score an unrelated stream."""
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
