"""VoxchainPipeline: transcripts become spoken replies, and barge-in cancels, stops and merges correctly."""

import asyncio

from langchain_core.messages import AIMessage

from voxchain import VoxchainPipeline
from voxchain.output import PlayAudio
from voxchain.session import SessionClosed
from voxchain.stt import FinalTranscript, SpeechStarted, SpeechStopped, STTProvider
from voxchain.tts import AudioChunk, Flushed, TTSProvider


class ScriptedSTT(STTProvider):
    sample_rate = 16000

    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue()
        self.audio = 0
        self.disconnected = False

    async def connect(self):
        pass

    async def send_audio(self, pcm):
        self.audio += len(pcm)

    async def receive_events(self):
        while True:
            event = await self.events.get()
            yield event
            if isinstance(event, SessionClosed):
                return

    async def disconnect(self):
        self.disconnected = True


class ScriptedTTS(TTSProvider):
    sample_rate = 24000

    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue()
        self.calls: list[tuple] = []

    async def connect(self):
        pass

    async def speak(self, text):
        self.calls.append(("speak", text))

    async def flush(self):
        self.calls.append(("flush",))

    async def cancel(self):
        self.calls.append(("cancel",))

    async def receive_events(self):
        while True:
            yield await self.events.get()

    async def disconnect(self):
        self.calls.append(("disconnect",))


class ScriptedAgent:
    """Replies with canned text; ``hold`` keeps a reply generating until released."""

    def __init__(self, reply="Hello there, nice to meet you. How can I help you today?"):
        self.reply = reply
        self.heard: list[str] = []
        self.hold: asyncio.Event | None = None

    async def astream(self, payload, config, stream_mode):
        self.heard.append(payload["messages"][0][1])
        for word in self.reply.split(" "):
            if self.hold is not None:
                await self.hold.wait()
            yield AIMessage(content=word + " "), {}


class Harness:
    def __init__(self, *, block_audio=False, **kwargs):
        self.stt, self.tts, self.agent = ScriptedSTT(), ScriptedTTS(), ScriptedAgent()
        self.received: list[str] = []
        self.now = 0.0
        self.unblock = asyncio.Event()
        self.block_audio = block_audio
        self.mic_done = asyncio.Event()

        async def callback(event):
            self.received.append(type(event).__name__)
            if isinstance(event, PlayAudio) and self.block_audio:
                await self.unblock.wait()

        self.pipeline = VoxchainPipeline(self.stt, self.tts, self.agent, callback, clock=lambda: self.now, **kwargs)

    async def start(self):
        async def mic():
            yield b"\0" * 320
            await self.mic_done.wait()

        self.task = asyncio.create_task(self.pipeline.run(mic()))
        await settle()

    async def say(self, text):
        self.stt.events.put_nowait(SpeechStarted())
        self.stt.events.put_nowait(SpeechStopped())
        self.stt.events.put_nowait(FinalTranscript(text))
        await settle()

    def spoken(self):
        return [c[1] for c in self.tts.calls if c[0] == "speak"]

    async def stop(self):
        self.stt.events.put_nowait(SessionClosed(reason="Test over"))
        await asyncio.wait_for(self.task, 1)


async def settle():
    for _ in range(30):
        await asyncio.sleep(0)


async def test_a_transcript_becomes_a_reply_spoken_sentence_by_sentence():
    h = Harness()
    await h.start()
    await h.say("hi")
    assert h.agent.heard == ["hi"]
    assert h.spoken() == ["Hello there, nice to meet you.", "How can I help you today?"]
    assert h.tts.calls[-1] == ("flush",)
    h.tts.events.put_nowait(AudioChunk(b"\0" * 480))
    h.tts.events.put_nowait(Flushed())
    await settle()
    assert h.received == ["PlayAudio", "ReplyFinished"]
    assert h.stt.audio == 320
    await h.stop()


async def test_barge_in_mid_generation_cancels_the_reply_and_tts_without_a_flush():
    h = Harness()
    h.agent.hold = asyncio.Event()
    await h.start()
    await h.say("tell me a story")
    h.now = 10.0
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    assert ("cancel",) in h.tts.calls
    assert ("flush",) not in h.tts.calls
    assert h.received == ["StopPlayback"]
    await h.stop()


async def test_barging_in_before_hearing_the_reply_merges_the_turns():
    h = Harness()
    h.agent.hold = asyncio.Event()
    await h.start()
    await h.say("I want a ticket to")
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    h.agent.hold = None
    h.stt.events.put_nowait(FinalTranscript("Bangkok"))
    await settle()
    assert h.agent.heard == ["I want a ticket to", "I want a ticket to Bangkok"]
    await h.stop()


async def test_barge_in_is_not_delayed_by_a_callback_blocked_on_playback():
    h = Harness(block_audio=True)
    await h.start()
    await h.say("hi")
    for _ in range(5):
        h.tts.events.put_nowait(AudioChunk(b"\0" * 48000))  # 1 s each
    await settle()
    h.now = 0.5
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    assert ("cancel",) in h.tts.calls  # Handled while the callback is still stuck on the first chunk
    h.unblock.set()
    await settle()
    assert h.received == ["PlayAudio", "StopPlayback"]
    await h.stop()


async def test_speech_while_the_last_reply_is_still_playing_stops_playback_only():
    h = Harness()
    await h.start()
    await h.say("hi")
    h.tts.events.put_nowait(AudioChunk(b"\0" * 48000 * 3))  # 3 s of audio
    h.tts.events.put_nowait(Flushed())
    await settle()
    h.now = 1.0
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    assert h.received[-1] == "StopPlayback"
    assert ("cancel",) not in h.tts.calls  # Synthesis was already finished
    await h.stop()


async def test_with_interrupt_off_speech_never_cancels_a_reply():
    h = Harness(interrupt=False)
    h.agent.hold = asyncio.Event()
    await h.start()
    await h.say("hi")
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    assert ("cancel",) not in h.tts.calls
    assert "StopPlayback" not in h.received
    await h.stop()


async def test_a_lost_session_stops_the_pipeline_and_closes_both_providers():
    h = Harness()
    await h.start()
    await h.stop()
    assert h.received == ["StreamClosed"]
    assert h.stt.disconnected and ("disconnect",) in h.tts.calls


async def test_the_pipeline_stops_when_the_audio_stream_ends():
    h = Harness()
    await h.start()
    h.mic_done.set()
    await asyncio.wait_for(h.task, 1)
    assert h.stt.disconnected
