"""Streams the spoken reply of a LangGraph agent, and keeps its memory equal to what the user heard."""

import logging
from collections.abc import AsyncIterator, Callable

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


async def stream_agent_text(
    agent: CompiledStateGraph,
    text: str,
    thread_id: str,
    on_tools: Callable[[int], None] | None = None,
) -> AsyncIterator[str]:
    """Run one agent turn and yield the reply text as it streams. The function never yields tool output.

    ``thread_id`` is the key for the agent's checkpointed memory across turns.
    ``on_tools`` receives the number of tool calls that wait for a result, each time that number changes.
    """
    pending: set[str] = set()
    async for message, _metadata in agent.astream(
        {"messages": [("user", text)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        before = len(pending)
        if isinstance(message, AIMessage):
            pending.update(call["id"] for call in message.tool_calls if call.get("id"))
            pending.update(chunk["id"] for chunk in getattr(message, "tool_call_chunks", []) if chunk.get("id"))
            if chunk := message.text:
                yield chunk
        elif isinstance(message, ToolMessage):
            pending.discard(message.tool_call_id)
        if on_tools is not None and len(pending) != before:
            on_tools(len(pending))


def has_memory(agent: CompiledStateGraph) -> bool:
    """Whether the agent keeps a checkpointed memory that voxchain can correct."""
    return getattr(agent, "checkpointer", None) is not None and hasattr(agent, "aupdate_state")


async def remember_reply(agent: CompiledStateGraph, thread_id: str, text: str) -> None:
    """Add ``text`` to the agent's memory as an AI message, for example a greeting that the pipeline spoke."""
    if has_memory(agent):
        await agent.aupdate_state({"configurable": {"thread_id": thread_id}}, {"messages": [AIMessage(text)]})


async def correct_turn(agent: CompiledStateGraph, thread_id: str, user_text: str, heard: str | None) -> None:
    """Make the agent's memory of one turn agree with what the user heard after a barge-in.

    The turn starts at the last user message equal to ``user_text``. If ``heard``
    is ``None``, the user did not hear the reply and says the turn again, so the
    function removes the whole turn. If not, the function replaces the spoken AI
    text of the turn with ``heard``. Tool calls and their results stay, so the
    agent does not do a tool call again.
    """
    if not has_memory(agent) or not user_text:
        return
    config = {"configurable": {"thread_id": thread_id}}
    messages = (await agent.aget_state(config)).values.get("messages", [])
    start = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None)
    if start is None or messages[start].text.strip() != user_text.strip():
        return
    if heard is None:
        changes = [RemoveMessage(id=message.id) for message in messages[start:]]
    else:
        spoken = [
            message for message in messages[start + 1 :] if isinstance(message, AIMessage) and not message.tool_calls
        ]
        changes = [RemoveMessage(id=message.id) for message in spoken]
        if heard:
            changes.append(AIMessage(heard))
    if changes:
        await agent.aupdate_state(config, {"messages": changes})
        logger.debug(f"Agent memory: Turn {user_text!r} now ends with {heard!r}")
