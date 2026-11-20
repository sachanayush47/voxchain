"""Streams the spoken reply of a LangGraph agent."""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph


async def stream_agent_text(agent: CompiledStateGraph, text: str, thread_id: str) -> AsyncIterator[str]:
    """Run one agent turn and yield the reply text as it streams. The function never yields tool output.

    ``thread_id`` is the key for the agent's checkpointed memory across turns.
    """
    async for message, _metadata in agent.astream(
        {"messages": [("user", text)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        if isinstance(message, AIMessage) and (chunk := message.text):
            yield chunk
