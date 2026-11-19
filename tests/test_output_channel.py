"""OutputChannel: client delivery from its own task, with a model of when each chunk plays."""

import asyncio

from voxkit.output import OutputChannel, PlayAudio, ReplyFinished, StopPlayback, StreamClosed

SECOND = 48000  # Bytes of 24 kHz PCM16


class Harness:
    def __init__(self):
        self.now = 0.0
        self.received = []
        self.played = []
        self.channel = OutputChannel(
            self.callback, sample_rate=24000, clock=lambda: self.now, on_played=lambda *args: self.played.append(args)
        )

    async def callback(self, event):
        self.received.append(event)


async def _run(h):
    task = asyncio.create_task(h.channel.run())
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()


async def test_chunks_play_back_to_back_when_they_arrive_faster_than_real_time():
    h = Harness()
    h.channel.play(b"\0" * SECOND, tag="a")
    h.channel.play(b"\0" * SECOND, tag="b")
    h.channel.reply_finished()
    await _run(h)
    assert h.played == [("a", 0.0, 1.0), ("b", 1.0, 2.0)]
    assert [type(e) for e in h.received] == [PlayAudio, PlayAudio, ReplyFinished]
    assert h.channel.playing(1.5) and not h.channel.playing(2.0)


async def test_a_chunk_after_the_client_ran_dry_starts_when_it_arrives():
    h = Harness()
    h.channel.play(b"\0" * SECOND, tag="a")
    await _run(h)
    h.now = 3.0
    h.channel.play(b"\0" * SECOND, tag="b")
    await _run(h)
    assert h.played[-1] == ("b", 3.0, 4.0)


async def test_stop_drops_queued_audio_and_resets_the_clock():
    h = Harness()
    h.channel.play(b"\0" * SECOND)
    h.channel.play(b"\0" * SECOND)
    h.channel.stop(now=0.0)
    await _run(h)
    assert h.received == [StopPlayback()]
    assert not h.channel.playing(0.0)


async def test_close_skips_the_queue_and_delivers_stream_closed_now():
    h = Harness()
    h.channel.play(b"\0" * SECOND)
    await h.channel.close("Gone")
    assert h.received == [StreamClosed("Gone")]


async def test_a_failing_callback_is_logged_and_delivery_goes_on():
    h = Harness()
    calls = []

    async def flaky(event):
        calls.append(event)
        if len(calls) == 1:
            raise RuntimeError("Speaker unplugged")

    h.channel.callback = flaky
    h.channel.play(b"\0" * 10)
    h.channel.play(b"\0" * 10)
    await _run(h)
    assert len(calls) == 2
