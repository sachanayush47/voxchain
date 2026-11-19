"""RealtimePipeline: playback, barge-in cuts, tool calls and lifecycle, against a scripted provider."""

import asyncio
import json

from langchain_core.tools import tool

from voxkit.output import PlayAudio
from voxkit.realtime import (
    AgentTranscript,
    AudioDelta,
    FunctionCall,
    RealtimePipeline,
    RealtimeProvider,
    ResponseDone,
    ResponseStarted,
    SessionClosed,
    SpeechStarted,
    UserTranscript,
)

MS = 48  # Bytes per millisecond of 24 kHz PCM16


class ScriptedProvider(RealtimeProvider):
    input_sample_rate = 24000
    output_sample_rate = 24000

    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue()
        self.calls: list[tuple] = []
        self.tools = []
        self.audio_in = 0

    def set_tools(self, tools):
        self.tools = tools

    async def connect(self):
        self.calls.append(("connect",))

    async def send_audio(self, pcm):
        self.audio_in += len(pcm)

    async def send_function_result(self, call_id, output):
        self.calls.append(("result", call_id, output))

    async def request_response(self):
        self.calls.append(("request_response",))

    async def truncate(self, item_id, audio_end_ms):
        self.calls.append(("truncate", item_id, audio_end_ms))

    async def delete_item(self, item_id):
        self.calls.append(("delete", item_id))

    async def receive_events(self):
        while True:
            event = await self.events.get()
            yield event
            if isinstance(event, SessionClosed):
                return

    async def disconnect(self):
        self.calls.append(("disconnect",))

    def push(self, *events):
        for event in events:
            self.events.put_nowait(event)

    def made(self, kind):
        return [c for c in self.calls if c[0] == kind]


class Harness:
    def __init__(self, *, tools=(), block_audio=False):
        self.provider = ScriptedProvider()
        self.received = []
        self.now = 0.0
        self.unblock = asyncio.Event()
        self.block_audio = block_audio
        self.transcripts = []

        async def callback(event):
            self.received.append(type(event).__name__)
            if isinstance(event, PlayAudio) and self.block_audio:
                await self.unblock.wait()  # A client whose playback buffer is full

        async def on_transcript(role, text):
            self.transcripts.append((role, text))

        self.pipeline = RealtimePipeline(
            self.provider,
            callback,
            tools=tools,
            input_sample_rate=16000,
            on_transcript=on_transcript,
            clock=lambda: self.now,
        )
        self.mic_done = asyncio.Event()

    async def start(self):
        async def mic():
            for _ in range(3):
                yield b"\0" * 320
            await self.mic_done.wait()

        self.task = asyncio.create_task(self.pipeline.run(mic()))
        await settle()

    async def stop(self):
        self.provider.push(SessionClosed(reason="Test over"))
        await asyncio.wait_for(self.task, 1)


async def settle():
    for _ in range(20):
        await asyncio.sleep(0)


async def until(condition):
    # Sync tools run in a worker thread, so they need real time, not just event-loop turns.
    for _ in range(100):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Condition never became true")


def audio(response_id, item_id, ms):
    return AudioDelta(response_id=response_id, item_id=item_id, data=b"\0" * (MS * ms))


async def test_a_reply_plays_and_ends_with_end_of_turn():
    h = Harness()
    await h.start()
    h.provider.push(
        UserTranscript(text="Hi"),
        ResponseStarted(response_id="r1"),
        audio("r1", "a1", 100),
        audio("r1", "a1", 100),
        AgentTranscript(response_id="r1", item_id="a1", text="Hello!"),
        ResponseDone(response_id="r1", status="completed"),
    )
    await settle()
    assert h.received == ["PlayAudio", "PlayAudio", "ReplyFinished"]
    assert h.transcripts == [("user", "Hi"), ("agent", "Hello!")]
    # 16 kHz mic audio arrives resampled to 24 kHz; the resampler holds back one sample for the next chunk.
    assert abs(h.provider.audio_in - 3 * 480) <= 2
    await h.stop()


async def test_barge_in_is_not_delayed_by_a_callback_blocked_on_playback():
    h = Harness(block_audio=True)
    await h.start()
    h.provider.push(ResponseStarted(response_id="r1"), *[audio("r1", "a1", 500) for _ in range(6)])
    await settle()
    h.now = 0.3
    h.provider.push(SpeechStarted())
    await settle()
    assert h.provider.made("truncate") == [("truncate", "a1", 300)]
    h.unblock.set()
    await settle()
    assert h.received == ["PlayAudio", "StopPlayback"]  # The five queued chunks were dropped
    await h.stop()


async def test_heard_time_survives_a_playback_gap():
    h = Harness()
    await h.start()
    h.provider.push(ResponseStarted(response_id="r1"), audio("r1", "a1", 200))
    await settle()
    h.now = 1.0  # The client ran dry for 800 ms before the next chunk arrived
    h.provider.push(audio("r1", "a1", 200))
    await settle()
    h.now = 1.1
    h.provider.push(SpeechStarted())
    await settle()
    assert h.provider.made("truncate") == [("truncate", "a1", 300)]
    await h.stop()


async def test_an_item_fully_heard_while_its_reply_is_still_generating_is_truncated_not_deleted():
    h = Harness()
    await h.start()
    h.provider.push(ResponseStarted(response_id="r1"), audio("r1", "a1", 200))
    await settle()
    h.now = 5.0
    h.provider.push(SpeechStarted())
    await settle()
    assert h.provider.made("truncate") == [("truncate", "a1", 200)]
    assert h.provider.made("delete") == []
    await h.stop()


async def test_unheard_items_of_the_cancelled_reply_are_deleted_and_their_late_audio_dropped():
    h = Harness(block_audio=True)
    await h.start()
    h.provider.push(ResponseStarted(response_id="r1"), audio("r1", "a1", 100), audio("r1", "a2", 100))
    await settle()
    h.provider.push(SpeechStarted(), audio("r1", "a1", 100), audio("r1", "a3", 100))
    await settle()
    h.provider.push(ResponseDone(response_id="r1", status="cancelled"))
    h.unblock.set()
    await settle()
    assert h.provider.made("delete") == [("delete", "a1"), ("delete", "a2"), ("delete", "a3")]
    assert "ReplyFinished" not in h.received
    assert h.received.count("PlayAudio") == 1
    await h.stop()


async def test_speech_after_the_reply_finished_playing_cuts_nothing():
    h = Harness()
    await h.start()
    h.provider.push(
        ResponseStarted(response_id="r1"), audio("r1", "a1", 100), ResponseDone(response_id="r1", status="completed")
    )
    await settle()
    h.now = 2.0
    h.provider.push(SpeechStarted())
    await settle()
    assert h.provider.made("truncate") == h.provider.made("delete") == []
    assert "StopPlayback" not in h.received
    await h.stop()


@tool
def get_weather(city: str) -> str:
    """Weather for a city."""
    return f"Sunny in {city}"


@tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("DB down")


async def test_tools_run_and_their_results_go_back_before_a_response_is_requested():
    h = Harness(tools=[get_weather, broken])
    await h.start()
    assert [t.name for t in h.provider.tools] == ["get_weather", "broken"]
    h.provider.push(
        ResponseStarted(response_id="r1"),
        FunctionCall(response_id="r1", call_id="c1", name="get_weather", arguments='{"city": "Delhi"}'),
        FunctionCall(response_id="r1", call_id="c2", name="broken", arguments="{}"),
        FunctionCall(response_id="r1", call_id="c3", name="missing", arguments="{}"),
        ResponseDone(response_id="r1", status="completed"),
    )
    await until(lambda: h.provider.made("request_response"))
    results = {c[1]: c[2] for c in h.provider.made("result")}
    assert results["c1"] == "Sunny in Delhi"
    assert json.loads(results["c2"]) == {"error": "RuntimeError: DB down"}
    assert "Unknown tool" in json.loads(results["c3"])["error"]
    assert h.provider.calls[-1] == ("request_response",)
    await h.stop()


async def test_calls_from_an_interrupted_reply_are_not_run():
    ran = []

    @tool
    def book(slot: str) -> str:
        """Book a slot."""
        ran.append(slot)
        return "booked"

    h = Harness(tools=[book])
    await h.start()
    h.provider.push(ResponseStarted(response_id="r1"), SpeechStarted())
    h.provider.push(
        FunctionCall(response_id="r1", call_id="c1", name="book", arguments='{"slot": "5pm"}'),
        ResponseDone(response_id="r1", status="cancelled"),
    )
    await settle()
    assert ran == []
    assert "interrupted" in json.loads(h.provider.made("result")[0][2])["error"]
    assert h.provider.made("request_response") == []
    await h.stop()


async def test_the_pipeline_stops_when_the_session_is_lost_for_good():
    h = Harness()
    await h.start()
    await h.stop()
    assert h.received[-1] == "StreamClosed"
    assert ("disconnect",) in h.provider.calls


async def test_the_pipeline_stops_when_the_audio_stream_ends():
    h = Harness()
    await h.start()
    h.mic_done.set()
    await asyncio.wait_for(h.task, 1)
    assert ("disconnect",) in h.provider.calls


async def test_a_failing_callback_does_not_stop_playback():
    h = Harness()

    async def flaky(event):
        h.received.append(type(event).__name__)
        if len(h.received) == 1:
            raise RuntimeError("Speaker unplugged")

    h.pipeline.output.callback = flaky
    await h.start()
    h.provider.push(ResponseStarted(response_id="r1"), audio("r1", "a1", 50), audio("r1", "a1", 50))
    await settle()
    assert h.received == ["PlayAudio", "PlayAudio"]
    await h.stop()
