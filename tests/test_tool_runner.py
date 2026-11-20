"""ToolRunner: it returns tool results and failures as strings that the model can use."""

import json

import pytest
from langchain_core.tools import tool

from voxchain.realtime import FunctionCall, ToolRunner, tool_spec


@tool
def lookup(query: str) -> dict:
    """Look something up."""
    return {"answer": query.upper()}


@tool
def broken() -> str:
    """Always fails."""
    raise RuntimeError("DB down")


def _call(name, arguments="{}", call_id="c1"):
    return FunctionCall(response_id="r1", call_id=call_id, name=name, arguments=arguments)


def test_a_tool_is_described_with_its_json_schema():
    spec = tool_spec(lookup)
    assert (spec.name, spec.description) == ("lookup", "Look something up.")
    assert spec.parameters["properties"] == {"query": {"type": "string"}}


async def test_a_structured_result_comes_back_as_json():
    assert json.loads(await ToolRunner([lookup]).run(_call("lookup", '{"query": "hi"}'))) == {"answer": "HI"}


async def test_failures_become_errors_the_model_can_read():
    runner = ToolRunner([broken])
    assert json.loads(await runner.run(_call("broken"))) == {"error": "RuntimeError: DB down"}
    assert "Unknown tool" in json.loads(await runner.run(_call("missing")))["error"]
    assert "not valid JSON" in json.loads(await runner.run(_call("broken", "{oops")))["error"]


async def test_results_come_back_in_call_order():
    calls = [_call("lookup", '{"query": "a"}', "c1"), _call("lookup", '{"query": "b"}', "c2")]
    results = await ToolRunner([lookup]).run_all(calls)
    assert [json.loads(r)["answer"] for r in results] == ["A", "B"]


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError):
        ToolRunner([lookup, lookup])
