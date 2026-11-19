"""OpenAIRealtimeProvider.session_config: what each option sends, and what it leaves to OpenAI."""

import pytest

from voxchain.realtime import OpenAIRealtimeProvider, ToolSpec


def _config(**kwargs):
    return OpenAIRealtimeProvider(api_key="test", **kwargs).session_config()


def test_defaults_ask_for_semantic_turns_with_barge_in_and_transcripts():
    config = _config(instructions="Be brief.")
    audio_input = config["audio"]["input"]
    assert config["type"] == "realtime"
    assert config["output_modalities"] == ["audio"]
    assert config["instructions"] == "Be brief."
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["turn_detection"] == {
        "type": "semantic_vad",
        "create_response": True,
        "interrupt_response": True,
    }
    assert audio_input["transcription"] == {"model": "gpt-transcribe"}
    assert config["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"}


def test_unset_options_are_left_to_openai():
    config = _config()
    for key in ("tools", "tool_choice", "reasoning", "truncation", "max_output_tokens"):
        assert key not in config
    assert "noise_reduction" not in config["audio"]["input"]
    assert "speed" not in config["audio"]["output"]


def test_server_vad_sends_its_tuning_and_semantic_vad_does_not():
    server = _config(turn_detection="server_vad", vad_threshold=0.7, silence_duration_ms=400, idle_timeout_ms=8000)
    assert server["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "create_response": True,
        "interrupt_response": True,
        "threshold": 0.7,
        "silence_duration_ms": 400,
        "idle_timeout_ms": 8000,
    }
    semantic = _config(eagerness="low", vad_threshold=0.7)
    assert semantic["audio"]["input"]["turn_detection"]["eagerness"] == "low"
    assert "threshold" not in semantic["audio"]["input"]["turn_detection"]


def test_tools_are_sent_in_the_flat_realtime_shape():
    provider = OpenAIRealtimeProvider(api_key="test")
    provider.set_tools([ToolSpec("get_time", "Current time.", {"type": "object", "properties": {}})])
    config = provider.session_config()
    assert config["tools"] == [
        {
            "type": "function",
            "name": "get_time",
            "description": "Current time.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert config["tool_choice"] == "auto"


def test_a_numeric_truncation_becomes_a_retention_ratio():
    assert _config(truncation=0.8)["truncation"] == {"type": "retention_ratio", "retention_ratio": 0.8}
    assert _config(truncation="auto")["truncation"] == "auto"


def test_options_that_would_fail_mid_call_fail_at_construction():
    with pytest.raises(ValueError):
        OpenAIRealtimeProvider(api_key="test", turn_detection="push_to_talk")
    with pytest.raises(ValueError):
        OpenAIRealtimeProvider(api_key="test", truncation=1.5)


def test_interrupt_response_false_turns_off_barge_in_end_to_end():
    provider = OpenAIRealtimeProvider(api_key="test", interrupt_response=False)
    assert provider.interrupts_on_speech is False
    assert provider.session_config()["audio"]["input"]["turn_detection"]["interrupt_response"] is False
