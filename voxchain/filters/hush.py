"""A filter that keeps only the main speaker with Hush, an open-source speech enhancement model."""

import ctypes
import logging
import weakref
from collections.abc import Callable

logger = logging.getLogger(__name__)

HUSH_REPO = "weya-ai/hush"
HUSH_FILENAME = "onnx/advanced_dfnet16k_model_best_onnx.tar.gz"
HUSH_REVISION = "a55d932cbf6344d284ac985f21e7f6e5bc4d38a5"  # A pinned model cannot change without a voxchain release
FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
# The model weighs approximately 8 MB, so all the filters with the same library and model share one copy.
LOADED_MODELS: dict[tuple[str, str], tuple[object, int]] = {}


def declare_api(lib) -> None:
    """Declare the C signatures of the ``weya_nc`` functions that :class:`HushFilter` calls."""
    lib.weya_nc_model_load_from_path.argtypes = [ctypes.c_char_p]
    lib.weya_nc_model_load_from_path.restype = ctypes.c_void_p
    lib.weya_nc_session_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float]
    lib.weya_nc_session_create.restype = ctypes.c_void_p
    lib.weya_nc_session_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_session_free.restype = None
    lib.weya_nc_get_frame_length.argtypes = [ctypes.c_void_p]
    lib.weya_nc_get_frame_length.restype = ctypes.c_size_t
    lib.weya_nc_process_frame.argtypes = [ctypes.c_void_p, FLOAT_POINTER, FLOAT_POINTER]
    lib.weya_nc_process_frame.restype = ctypes.c_float


class HushFilter:
    """Keeps only the main speaker. It removes background noise and other voices, for example a TV or a far voice.

    `Hush <https://huggingface.co/weya-ai/hush>`_ (Apache 2.0) is a version of
    DeepFilterNet3 that is trained to remove competing speakers. It runs on CPU in
    real time and adds approximately 20 ms of delay. Very loud background voices can
    stay. After it removes a background voice, it can make the next short word of the
    user weaker, so put a :class:`~voxchain.filters.loudness.LoudnessGate` before it.

    It needs the ``hush`` extra (``pip install "voxchain[hush]"``) and the
    native ``weya_nc`` library of the `Hush project
    <https://github.com/pulp-vision/Hush>`_ at ``library_path``. If ``model_path``
    is ``None``, the filter downloads the model from Hugging Face.
    ``attenuation_limit_db`` is the maximum reduction of the noise. 100 dB removes it.
    """

    def __init__(
        self,
        *,
        library_path: str,
        model_path: str | None = None,
        sample_rate: int = 16000,
        attenuation_limit_db: float = 100.0,
        load_library: Callable[[str], object] = ctypes.CDLL,
    ) -> None:
        try:
            import numpy as np
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError('HushFilter requires the hush extra: pip install "voxchain[hush]"') from e

        self._np = np
        path = model_path or hf_hub_download(HUSH_REPO, HUSH_FILENAME, revision=HUSH_REVISION)
        if (library_path, path) not in LOADED_MODELS:
            lib = load_library(library_path)
            declare_api(lib)
            model = lib.weya_nc_model_load_from_path(path.encode())
            if not model:
                raise RuntimeError(f"Hush: The library could not load the model at {path}")
            LOADED_MODELS[library_path, path] = (lib, model)

        self._lib, model = LOADED_MODELS[library_path, path]
        self._session = self._lib.weya_nc_session_create(model, sample_rate, attenuation_limit_db)
        if not self._session:
            raise RuntimeError(f"Hush: The library could not start a session at {sample_rate} Hz")
        weakref.finalize(self, self._lib.weya_nc_session_free, self._session)
        self._frame_bytes = self._lib.weya_nc_get_frame_length(self._session) * 2
        self._output = np.zeros(self._frame_bytes // 2, dtype=np.float32)
        self._pending = b""
        logger.info(f"Hush loaded | sample_rate={sample_rate} | frame={self._frame_bytes // 2} samples")

    def process(self, pcm: bytes) -> bytes:
        """Return the audio of ``pcm`` with only the main speaker. A part of a frame waits for the next chunk."""
        np = self._np
        data = self._pending + pcm
        end = len(data) - len(data) % self._frame_bytes
        self._pending = data[end:]
        out = bytearray()
        for start in range(0, end, self._frame_bytes):
            frame = np.frombuffer(data[start : start + self._frame_bytes], dtype=np.int16).astype(np.float32) / 32768
            self._lib.weya_nc_process_frame(
                self._session, frame.ctypes.data_as(FLOAT_POINTER), self._output.ctypes.data_as(FLOAT_POINTER)
            )
            out += (np.clip(self._output, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return bytes(out)
