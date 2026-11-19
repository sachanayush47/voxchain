"""Running LangChain tools on behalf of a speech-to-speech model."""

import asyncio
import json
import logging
from collections.abc import Sequence

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from voxchain.realtime.base import ToolSpec
from voxchain.realtime.events import FunctionCall

logger = logging.getLogger(__name__)

NOT_RUN_OUTPUT = json.dumps({"error": "Not run: the user interrupted before this tool call was carried out."})


def tool_spec(tool: BaseTool) -> ToolSpec:
    """Describe a LangChain tool for a provider."""
    function = convert_to_openai_tool(tool)["function"]
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    return ToolSpec(name=function["name"], description=function.get("description", ""), parameters=parameters)


class ToolRunner:
    """Runs the model's tool calls and renders each result, or error, as a string for the model."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Tool names must be unique")

    def specs(self) -> list[ToolSpec]:
        """The tools, described for a provider."""
        return [tool_spec(tool) for tool in self.tools.values()]

    async def run_all(self, calls: list[FunctionCall]) -> list[str]:
        """Run calls concurrently; results come back in call order."""
        return list(await asyncio.gather(*(self.run(call) for call in calls)))

    async def run(self, call: FunctionCall) -> str:
        """Run one call. Failures come back as a JSON error the model can react to, never as an exception."""
        tool = self.tools.get(call.name)
        if tool is None:
            logger.warning(f"ToolRunner: Model called unknown tool {call.name!r}")
            return json.dumps({"error": f"Unknown tool {call.name!r}."})
        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError:
            logger.warning(f"ToolRunner: Invalid arguments for {call.name}: {call.arguments!r}")
            return json.dumps({"error": "Arguments were not valid JSON."})

        logger.debug(f"ToolRunner: Calling {call.name}({call.arguments})")
        try:
            result = await tool.ainvoke({"type": "tool_call", "id": call.call_id, "name": call.name, "args": args})
        except Exception as e:
            logger.exception(f"ToolRunner: Tool {call.name} failed")
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        content = result.content if isinstance(result, ToolMessage) else result
        return content if isinstance(content, str) else json.dumps(content, default=str)
