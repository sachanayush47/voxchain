"""Agent memory after a barge-in: it keeps only what the user heard, and an unheard turn goes away."""

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from voxchain.core.agent import correct_turn, remember_reply, stream_agent_text

THREAD = "t"
CONFIG = {"configurable": {"thread_id": THREAD}}


def _agent(*replies):
    model = GenericFakeChatModel(messages=iter([AIMessage(text) for text in replies]))
    return create_agent(model=model, tools=[], checkpointer=InMemorySaver())


async def _turn(agent, text):
    return "".join([chunk async for chunk in stream_agent_text(agent, text, THREAD)])


async def _memory(agent):
    return [(type(m).__name__, m.text) for m in (await agent.aget_state(CONFIG)).values["messages"]]


async def test_the_reply_in_memory_becomes_the_part_the_user_heard():
    agent = _agent("The weather today is sunny and warm.")
    await _turn(agent, "weather?")
    await correct_turn(agent, THREAD, "weather?", "The weather today")
    assert await _memory(agent) == [("HumanMessage", "weather?"), ("AIMessage", "The weather today")]


async def test_a_turn_the_user_did_not_hear_is_removed_so_it_can_merge():
    agent = _agent("Sure.", "Where to?")
    await _turn(agent, "hello")
    await _turn(agent, "I want a ticket to")
    await correct_turn(agent, THREAD, "I want a ticket to", None)
    assert await _memory(agent) == [("HumanMessage", "hello"), ("AIMessage", "Sure.")]


async def test_a_reply_heard_not_at_all_is_removed_but_the_user_turn_stays():
    agent = _agent("A long answer.")
    await _turn(agent, "question")
    await correct_turn(agent, THREAD, "question", "")
    assert await _memory(agent) == [("HumanMessage", "question")]


async def test_a_different_last_turn_is_left_alone():
    agent = _agent("Answer.")
    await _turn(agent, "question")
    await correct_turn(agent, THREAD, "another question", "")
    assert len(await _memory(agent)) == 2


async def test_the_greeting_goes_into_memory_before_the_first_turn():
    agent = _agent("Sure.")
    await remember_reply(agent, THREAD, "Hello, how can I help?")
    await _turn(agent, "hi")
    messages = (await agent.aget_state(CONFIG)).values["messages"]
    assert isinstance(messages[0], AIMessage) and messages[0].text == "Hello, how can I help?"
    assert isinstance(messages[1], HumanMessage)


async def test_an_agent_without_memory_is_skipped():
    agent = create_agent(model=GenericFakeChatModel(messages=iter([AIMessage("x")])), tools=[])
    await correct_turn(agent, THREAD, "question", "")
    await remember_reply(agent, THREAD, "Hello")
