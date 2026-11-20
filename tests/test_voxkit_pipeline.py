"""VoxchainPipeline: transcripts become spoken replies, and barge-in happens only for real interruptions."""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from voxchain import VoxchainPipeline
from voxchain.core.pipeline import RECOVERY_PROMPT
from voxchain.output import PlayAudio
from voxchain.session import SessionClosed
from voxchain.stt import FinalTranscript, PartialTranscript, SpeechStarted, SpeechStopped, STTProvider
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
    """Replies with fixed text. If ``hold`` is an event, each word waits until the test sets it."""

    def __init__(self, reply="Hello there, nice to meet you. How can I help you today?"):
        self.reply = reply
        self.heard: list[str] = []
        self.hold: asyncio.Event | None = None
        self.tool_done: asyncio.Event | None = None

    async def astream(self, payload, config, stream_mode):
        self.heard.append(payload["messages"][0][1])
        if self.tool_done is not None:
            yield AIMessage(content="", tool_calls=[{"name": "book", "args": {}, "id": "call-1"}]), {}
            await self.tool_done.wait()
            yield ToolMessage(content="Booked", tool_call_id="call-1"), {}
        for word in self.reply.split(" "):
            if self.hold is not None:
                await self.hold.wait()
            yield AIMessage(content=word + " "), {}


class Harness:
    def __init__(self, *, block_audio=False, agent=None, **kwargs):
        self.stt, self.tts, self.agent = ScriptedSTT(), ScriptedTTS(), agent or ScriptedAgent()
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

    async def play(self, seconds, *, finished=True):
        """TTS sends ``seconds`` of audio for the current reply, and the client starts to play it."""
        self.tts.events.put_nowait(AudioChunk(b"\0" * int(48000 * seconds)))
        if finished:
            self.tts.events.put_nowait(Flushed())
        await settle()

    async def speak_over(self, text):
        """The user speaks over the agent, and the STT sends one final transcript."""
        self.stt.events.put_nowait(SpeechStarted())
        await settle()
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


async def test_with_voice_activity_barge_in_speech_mid_generation_cancels_the_reply_without_a_flush():
    h = Harness(interrupt_min_words=0)
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
    h = Harness(block_audio=True, interrupt_min_words=0)
    await h.start()
    await h.say("hi")
    for _ in range(5):
        h.tts.events.put_nowait(AudioChunk(b"\0" * 48000))  # 1 s each
    await settle()
    h.now = 0.5
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    assert ("cancel",) in h.tts.calls  # The cancel happens while the callback is still blocked on the first chunk
    h.unblock.set()
    await settle()
    assert h.received == ["PlayAudio", "StopPlayback"]
    await h.stop()


async def test_with_voice_activity_barge_in_speech_over_the_last_reply_stops_playback_only():
    h = Harness(interrupt_min_words=0)
    await h.start()
    await h.say("hi")
    h.tts.events.put_nowait(AudioChunk(b"\0" * 48000 * 3))  # 3 s of audio
    h.tts.events.put_nowait(Flushed())
    await settle()
    h.now = 1.0
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    assert h.received[-1] == "StopPlayback"
    assert ("cancel",) not in h.tts.calls  # Synthesis already finished
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


async def test_speech_over_the_agent_without_words_does_not_stop_it():
    h = Harness()
    await h.start()
    await h.say("hi")
    await h.play(3)
    h.now = 1.0
    h.stt.events.put_nowait(SpeechStarted())  # Noise or music that the VAD reports as speech
    await settle()
    assert "StopPlayback" not in h.received
    await h.stop()


async def test_a_short_transcript_over_the_agent_is_discarded_and_the_agent_continues():
    h = Harness()
    h.agent.hold = asyncio.Event()
    await h.start()
    await h.say("hi")
    h.agent.hold.set()
    await h.play(3, finished=False)
    h.now = 1.0
    await h.speak_over("uh-huh")
    assert "StopPlayback" not in h.received
    assert ("cancel",) not in h.tts.calls
    assert h.agent.heard == ["hi"]
    await h.stop()


async def test_enough_words_over_the_agent_interrupt_it_and_become_the_next_turn():
    h = Harness()
    await h.start()
    await h.say("hi")
    await h.play(3)
    h.now = 2.5
    await h.speak_over("what about tomorrow")
    assert h.received[-1] == "StopPlayback"
    assert h.agent.heard == ["hi", "what about tomorrow"]
    await h.stop()


async def test_a_stop_word_interrupts_although_it_is_short():
    h = Harness()
    await h.start()
    await h.say("hi")
    await h.play(3)
    h.now = 2.5
    await h.speak_over("Stop!")
    assert h.received[-1] == "StopPlayback"
    await h.stop()


async def test_short_transcripts_close_together_add_up_to_an_interruption():
    h = Harness()
    await h.start()
    await h.say("hi")
    await h.play(5)
    h.now = 2.5
    await h.speak_over("I said")
    assert "StopPlayback" not in h.received
    h.now = 3.0
    await h.speak_over("Paris")
    assert h.received[-1] == "StopPlayback"
    assert h.agent.heard[-1] == "I said Paris"
    await h.stop()


async def test_a_partial_transcript_with_enough_words_interrupts_before_the_final_one():
    h = Harness()
    await h.start()
    await h.say("hi")
    await h.play(3)
    h.now = 2.5
    h.stt.events.put_nowait(SpeechStarted())
    await settle()
    h.stt.events.put_nowait(PartialTranscript("no no I meant"))
    await settle()
    assert h.received[-1] == "StopPlayback"
    await h.stop()


async def test_the_user_cannot_interrupt_while_a_tool_call_runs_but_is_answered_after_it():
    h = Harness()
    h.agent.tool_done = asyncio.Event()
    await h.start()
    await h.say("book a table")
    h.now = 1.0
    await h.speak_over("and add a cake please")
    assert ("cancel",) not in h.tts.calls
    assert h.agent.heard == ["book a table"]
    h.agent.tool_done.set()
    await settle()
    h.agent.tool_done = None
    await h.play(1)
    await settle()
    assert h.agent.heard == ["book a table", "and add a cake please"]
    await h.stop()


async def test_the_greeting_plays_first_and_the_user_cannot_interrupt_it():
    h = Harness(greeting="Hello, this is the clinic. How can I help?")
    await h.start()
    assert h.spoken() == ["Hello, this is the clinic. How can I help?"]
    await h.speak_over("hello hello is anyone there")
    assert "StopPlayback" not in h.received
    assert h.agent.heard == []  # Speech during the greeting is ignored
    await h.play(3)
    h.now = 4.0  # The greeting audio has ended
    await h.say("I need an appointment")
    assert h.agent.heard == ["I need an appointment"]
    await h.stop()


async def test_a_barge_in_without_a_transcript_makes_the_agent_ask_again():
    h = Harness(interrupt_min_words=0, recovery_delay=0.01)
    await h.start()
    await h.say("hi")
    await h.play(5)
    h.now = 4.0
    h.stt.events.put_nowait(SpeechStarted())  # A cough cuts the agent off
    h.stt.events.put_nowait(SpeechStopped())
    await settle()
    await asyncio.sleep(0.05)
    assert h.agent.heard == ["hi", RECOVERY_PROMPT]
    await h.stop()


async def test_a_transcript_after_the_barge_in_cancels_the_recovery():
    h = Harness(interrupt_min_words=0, recovery_delay=0.01)
    await h.start()
    await h.say("hi")
    await h.play(5)
    h.now = 4.0
    await h.speak_over("what is the price")
    await asyncio.sleep(0.05)
    assert h.agent.heard == ["hi", "what is the price"]
    await h.stop()


async def test_the_audio_filter_cleans_audio_before_stt():
    class HalfFilter:
        def process(self, pcm):
            return pcm[: len(pcm) // 2]

    h = Harness(audio_filter=HalfFilter())
    await h.start()
    assert h.stt.audio == 160
    await h.stop()


async def test_after_a_barge_in_the_agent_memory_holds_only_what_the_user_heard():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    model = GenericFakeChatModel(
        messages=iter([AIMessage("The weather today is sunny and warm everywhere."), AIMessage("Tomorrow it rains.")])
    )
    agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
    h = Harness(agent=agent)
    await h.start()
    await h.say("weather?")
    await asyncio.sleep(0.05)
    await h.play(4)
    h.now = 2.5  # 2.5 s of the 4 s of audio played, after the resume window
    await h.speak_over("and what about tomorrow")
    await asyncio.sleep(0.05)
    messages = (await agent.aget_state({"configurable": {"thread_id": "default"}})).values["messages"]
    assert [(type(m).__name__, m.text) for m in messages] == [
        ("HumanMessage", "weather?"),
        ("AIMessage", "The weather today is sunny"),
        ("HumanMessage", "and what about tomorrow"),
        ("AIMessage", "Tomorrow it rains."),
    ]
    await h.stop()


async def test_a_merged_turn_replaces_the_early_turn_in_the_agent_memory():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    model = GenericFakeChatModel(messages=iter([AIMessage("Where do you want to go?"), AIMessage("Booked Bangkok.")]))
    agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
    h = Harness(agent=agent)
    await h.start()
    await h.say("I want a ticket to")
    await asyncio.sleep(0.05)
    await h.play(2)
    h.now = 0.5  # Within the resume window: the end-of-turn decision was too early
    await h.speak_over("Bangkok in first class")
    await asyncio.sleep(0.05)
    messages = (await agent.aget_state({"configurable": {"thread_id": "default"}})).values["messages"]
    assert [(type(m).__name__, m.text) for m in messages] == [
        ("HumanMessage", "I want a ticket to Bangkok in first class"),
        ("AIMessage", "Booked Bangkok."),
    ]
    await h.stop()
