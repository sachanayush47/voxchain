"""An :class:`~voxchain.turn.base.EndOfTurnDetector` that uses the Smart Turn v3 model from pipecat.

`Smart Turn <https://github.com/pipecat-ai/smart-turn>`_ is a small audio
classifier (approximately 8 MB). It examines the end of an utterance, for
example the intonation, filler words and incomplete words. Then it predicts if
the speaker finished. It runs locally on CPU with ONNX Runtime in tens of
milliseconds, and it needs no API key.

This module requires the ``smart-turn`` extra: ``pip install "voxchain[smart-turn]"``.
"""

import asyncio
import logging
from typing import Literal

from voxchain.turn.base import EndOfTurnDetector

logger = logging.getLogger(__name__)

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
"""The Hugging Face repo from which voxchain downloads the model."""

SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"
"""The model file in :data:`SMART_TURN_REPO` (the v3.2 build, optimized for CPU)."""


MODEL_SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
PRE_ROLL_SECONDS = 1.0


class PipecatSmartTurnDetector(EndOfTurnDetector):
    """Predicts the end of turn from the user's audio with Smart Turn v3.

    The detector keeps the input audio from the start of the user turn in a
    buffer. For each finalized transcript, it scores the last 8 seconds of this
    audio. A score equal to or more than ``threshold`` means that the turn is
    complete. Increase ``threshold`` to wait more often for the user to
    continue. Decrease it to reply sooner.

    Input audio must be mono 16-bit little-endian PCM (``pcm_s16le``). This is
    the default format of :class:`~voxchain.stt.sarvam.SarvamSTTProvider`. The
    detector upsamples 8 kHz audio to the 16 kHz that the model expects.

    At first use, the detector downloads the model from Hugging Face and
    caches it. The constructor loads the model synchronously, so create the
    detector one time at startup.

    Example:
        >>> detector = PipecatSmartTurnDetector(sample_rate=16000)
        >>> pipeline = VoxchainPipeline(stt, tts, agent, play, end_of_turn=detector)
    """

    def __init__(
        self,
        sample_rate: Literal[8000, 16000] = 16000,
        threshold: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        """Load the Smart Turn model."""
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sample_rate}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor
        except ImportError as e:
            msg = 'PipecatSmartTurnDetector requires the smart-turn extra: pip install "voxchain[smart-turn]"'
            raise ImportError(msg) from e

        self.sample_rate = sample_rate
        self.threshold = threshold

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = model_path or hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILENAME)
        self._session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)

        self._bytes_per_second = sample_rate * 2
        self._audio = bytearray()

    def push_audio(self, chunk: bytes) -> None:
        """Add ``chunk`` to the turn buffer, and keep only the last 8 seconds."""
        self._audio += chunk
        overflow = len(self._audio) - WINDOW_SECONDS * self._bytes_per_second
        if overflow > 0:
            del self._audio[: overflow + overflow % 2]

    def start_turn(self) -> None:
        """Discard the buffered audio from before this turn, but keep a short pre-roll."""
        keep = int(PRE_ROLL_SECONDS * self._bytes_per_second)
        keep -= keep % 2
        if len(self._audio) > keep:
            del self._audio[:-keep]

    async def is_end_of_turn(self, transcript: str) -> bool:
        """Score the buffered turn audio with Smart Turn."""
        audio = bytes(self._audio[: len(self._audio) - len(self._audio) % 2])
        if not audio:
            return True
        probability = await asyncio.to_thread(self.predict, audio)
        logger.debug("PipecatSmartTurnDetector: %r -> completion probability %.2f", transcript, probability)
        return probability >= self.threshold

    def predict(self, audio: bytes) -> float:
        """Return Smart Turn's completion probability for ``audio``.

        This method blocks because it runs ONNX inference. :meth:`is_end_of_turn`
        calls it outside the event loop. It does the same steps as
        ``predict_endpoint``, the reference function from pipecat.
        """
        import numpy as np

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if self.sample_rate != MODEL_SAMPLE_RATE:
            positions = np.arange(0, len(samples), self.sample_rate / MODEL_SAMPLE_RATE)
            samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
        samples = samples[-WINDOW_SECONDS * MODEL_SAMPLE_RATE :]

        features = self._feature_extractor(
            samples,
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SECONDS * MODEL_SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        return float(self._session.run(None, {"input_features": features})[0].flatten()[0])
