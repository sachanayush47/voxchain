"""FilterChain: audio filters run one after another, and a filter that returns no audio stops the chain."""

from voxchain.filters import FilterChain


class Recorder:
    def __init__(self, name, log, keep=True):
        self.name, self.log, self.keep = name, log, keep

    def process(self, pcm):
        self.log.append((self.name, pcm))
        return pcm + self.name.encode() if self.keep else b""


def test_each_filter_receives_the_output_of_the_filter_before_it():
    log = []
    chain = FilterChain([Recorder("a", log), Recorder("b", log)])
    assert chain.process(b"x") == b"xab"
    assert log == [("a", b"x"), ("b", b"xa")]


def test_a_filter_that_returns_no_audio_stops_the_chain_for_that_chunk():
    log = []
    chain = FilterChain([Recorder("a", log, keep=False), Recorder("b", log)])
    assert chain.process(b"x") == b""
    assert log == [("a", b"x")]


def test_a_chain_with_no_filters_returns_the_audio_without_a_change():
    assert FilterChain().process(b"audio") == b"audio"


def test_a_chain_can_be_a_filter_in_another_chain():
    log = []
    inner = FilterChain([Recorder("a", log)])
    assert FilterChain([inner, Recorder("b", log)]).process(b"x") == b"xab"
