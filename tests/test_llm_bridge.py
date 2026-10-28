from __future__ import annotations

import json

import httpx
import pytest

from among_us_ai.llm_bridge import (
    VOTE_TEMPLATE,
    AnthropicProvider,
    FieldSpec,
    GeminiProvider,
    LLMBridge,
    LLMConfigurationError,
    LLMError,
    LLMRequest,
    LLMResponseFormatError,
    LLMTransientError,
    OpenAIProvider,
    RetryPolicy,
    ScriptedProvider,
    create_provider,
    extract_json_object,
    validate_fields,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_extract_json_variants() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Sure!\n```json\n{"vote": "Red"}\n```') == {"vote": "Red"}
    assert extract_json_object('prefix {"msg": "a } brace", "n": {"k": 2}} suffix') == {"msg": "a } brace", "n": {"k": 2}}
    with pytest.raises(LLMResponseFormatError):
        extract_json_object("no json here")


def test_validate_fields() -> None:
    specs = (FieldSpec("vote", str), FieldSpec("confidence", float), FieldSpec("note", str, required=False))
    assert validate_fields({"vote": "skip", "confidence": 1}, specs) == {"vote": "skip", "confidence": 1.0, "note": None}
    with pytest.raises(LLMResponseFormatError):
        validate_fields({"vote": 3, "confidence": 0.5}, specs)
    with pytest.raises(LLMResponseFormatError):
        validate_fields({"confidence": 0.5}, specs)


def test_anthropic_provider_request_shape() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "claude-test", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": '{"ok": true}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    provider = AnthropicProvider("claude-test", api_key="k", client=_client(handler))
    response = provider.complete(LLMRequest.single("sys", "hi", reasoning_budget=2048, max_tokens=500))
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["body"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert seen["body"]["max_tokens"] > 2048 and "temperature" not in seen["body"]
    assert response.text == '{"ok": true}' and response.usage.output_tokens == 5


def test_openai_provider_request_shape() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "model": "gpt-test",
            "choices": [{"message": {"content": '{"x": 1}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        })

    provider = OpenAIProvider("gpt-test", api_key="sk", base_url="https://example.test/v1", client=_client(handler))
    response = provider.complete(LLMRequest.single("sys", "hi"))
    assert seen["auth"] == "Bearer sk"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert response.text == '{"x": 1}'


def test_gemini_provider_request_shape() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": '{"y": 2}'}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
        })

    provider = GeminiProvider("gemini-test", api_key="g", client=_client(handler))
    response = provider.complete(LLMRequest.single("sys", "hi", reasoning_budget=1000))
    assert seen["url"].endswith("/models/gemini-test:generateContent")
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert seen["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1000}
    assert response.text == '{"y": 2}'


def test_transient_errors_are_retried_then_fallback() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="overloaded", headers={"retry-after": "0"})

    primary = AnthropicProvider("m", api_key="k", client=_client(flaky))
    backup = ScriptedProvider(['{"ok": 1}'])
    sleeps: list[float] = []
    bridge = LLMBridge({"claude": primary, "backup": backup}, active="claude", fallback_order=["backup"],
                       retry=RetryPolicy(max_attempts=3), sleep=sleeps.append)
    response = bridge.complete(LLMRequest.single("s", "p"))
    assert calls["n"] == 3 and len(sleeps) == 2
    assert response.provider == "scripted"
    assert bridge.stats.fallbacks == 1


def test_auth_error_is_not_retried() -> None:
    provider = OpenAIProvider("m", api_key="bad", client=_client(lambda r: httpx.Response(401)))
    with pytest.raises(LLMConfigurationError):
        provider.complete(LLMRequest.single("s", "p"))
    with pytest.raises(LLMConfigurationError):
        GeminiProvider("m", api_key="").complete(LLMRequest.single("s", "p"))


def test_hot_swap_and_registry() -> None:
    a = ScriptedProvider(['{"from": "a"}'])
    b = ScriptedProvider(['{"from": "b"}'])
    bridge = LLMBridge({"a": a})
    assert bridge.complete_json(LLMRequest.single("s", "p")).data == {"from": "a"}
    bridge.register("b", b)
    bridge.swap("b")
    assert bridge.active == "b"
    assert bridge.complete_json(LLMRequest.single("s", "p")).data == {"from": "b"}
    bridge.unregister("b")
    assert bridge.active == "a"
    with pytest.raises(LLMConfigurationError):
        bridge.swap("missing")


def test_json_repair_round_trip() -> None:
    provider = ScriptedProvider(["I think Red.", '{"vote": "Red", "confidence": 0.9, "reasoning": "vented"}'])
    bridge = LLMBridge({"s": provider})
    result = bridge.run_template(VOTE_TEMPLATE, {
        "name": "Blue", "color": "blue", "role": "crewmate", "alive": "Red, Blue", "impostors_alive": 1,
        "suspicion": "-", "recommendation": "Red", "transcript": "-", "targets": "Red",
    })
    assert result.data["vote"] == "Red"
    assert len(provider.requests) == 2
    assert "invalid" in provider.requests[1].messages[-1].content


def test_all_failures_raise() -> None:
    def broken(_: LLMRequest) -> str:
        raise LLMTransientError("down")

    bridge = LLMBridge({"x": ScriptedProvider(broken)}, retry=RetryPolicy(max_attempts=1))
    with pytest.raises(LLMError):
        bridge.complete(LLMRequest.single("s", "p"))


def test_create_provider_factory() -> None:
    assert isinstance(create_provider("claude", api_key="k"), AnthropicProvider)
    assert isinstance(create_provider("OpenAI", api_key="k"), OpenAIProvider)
    with pytest.raises(LLMConfigurationError):
        create_provider("llama")


def test_template_missing_variable() -> None:
    with pytest.raises(LLMConfigurationError):
        VOTE_TEMPLATE.render({"name": "x"})
