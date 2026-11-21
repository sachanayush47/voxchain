"""HushFilter: it sends whole frames to the native weya_nc library and shares one model between filters."""

import gc

import pytest

pytest.importorskip("numpy")
pytest.importorskip("huggingface_hub")

from voxchain.filters import HushFilter

FRAME = 160


class FakeLibrary:
    """The weya_nc C API: each function is a plain function, so that ctypes can set its argtypes."""

    def __init__(self, model=1):
        self.models_loaded = []
        self.sessions = []
        self.freed = []

        def load(path):
            self.models_loaded.append(path)
            return model

        def create(model, sample_rate, attenuation):
            self.sessions.append((model, sample_rate, attenuation))
            return len(self.sessions) + 100

        def process(session, input_frame, output_frame):
            for i in range(FRAME):
                output_frame[i] = input_frame[i] / 2  # Half the volume, so that the test can see the output
            return 0.0

        self.weya_nc_model_load_from_path = load
        self.weya_nc_session_create = create
        self.weya_nc_session_free = lambda session: self.freed.append(session)
        self.weya_nc_get_frame_length = lambda session: FRAME
        self.weya_nc_process_frame = process


def make_filter(library, name, **kwargs):
    return HushFilter(library_path=name, model_path="model.tar.gz", load_library=lambda _path: library, **kwargs)


def test_the_filter_processes_whole_frames_and_keeps_the_rest_for_the_next_chunk():
    library = FakeLibrary()
    hush = make_filter(library, "lib-frames")
    loud = (16000).to_bytes(2, "little", signed=True)
    assert hush.process(loud * 100) == b""  # Less than one frame
    out = hush.process(loud * 100)
    assert len(out) == FRAME * 2
    assert int.from_bytes(out[:2], "little", signed=True) == pytest.approx(8000, abs=1)


def test_filters_with_the_same_library_and_model_share_one_loaded_model():
    library = FakeLibrary()
    make_filter(library, "lib-shared")
    make_filter(library, "lib-shared", sample_rate=8000, attenuation_limit_db=30.0)
    assert library.models_loaded == [b"model.tar.gz"]
    assert library.sessions == [(1, 16000, 100.0), (1, 8000, 30.0)]


def test_a_model_that_does_not_load_raises_an_error():
    with pytest.raises(RuntimeError, match="could not load the model"):
        make_filter(FakeLibrary(model=None), "lib-broken")


def test_the_session_is_freed_when_the_filter_is_garbage():
    library = FakeLibrary()
    hush = make_filter(library, "lib-free")
    del hush
    gc.collect()
    assert library.freed == [101]
