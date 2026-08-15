"""Request-side normalization and upstream failure classification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from opencode_proxy.compat import annotate_empty_completion, is_empty_completion
from opencode_proxy.proxy import (
    classify_upstream_status,
    finish_reason_label,
    retry_after_seconds,
)
from opencode_proxy.request_compat import (
    EMPTY_TOOL_RESULT_PLACEHOLDER,
    normalize_request,
)


def test_null_assistant_content_becomes_an_empty_string() -> None:
    body: dict[str, Any] = {
        "messages": [
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "kept"},
        ],
    }

    stats = normalize_request(body, thinking_transport=None)

    assert body["messages"][0]["content"] == ""
    assert body["messages"][1]["content"] == "kept"
    assert stats.null_assistant_content == 1


def test_reasoning_is_replayed_on_tool_call_turns_and_dropped_elsewhere() -> None:
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "kept for the tool round trip",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls"}}],
            },
            {
                "role": "assistant",
                "content": "plain answer",
                "reasoning_content": "ignored by the API",
                "reasoning": "also ignored",
            },
        ],
    }

    stats = normalize_request(body, thinking_transport=None)

    assert body["messages"][0]["reasoning_content"] == "kept for the tool round trip"
    assert "reasoning_content" not in body["messages"][1]
    assert "reasoning" not in body["messages"][1]
    assert stats.dropped_reasoning == 2


def test_reasoning_moves_into_the_field_the_api_reads() -> None:
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning": "why I called the tool",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls"}}],
            },
        ],
    }

    stats = normalize_request(body, thinking_transport=None)

    message = body["messages"][0]
    assert message["reasoning_content"] == "why I called the tool"
    assert "reasoning" not in message
    assert stats.moved_reasoning == 1


def test_empty_tool_results_get_placeholder_content() -> None:
    body: dict[str, Any] = {
        "messages": [
            {"role": "tool", "tool_call_id": "call_1", "content": ""},
            {"role": "tool", "tool_call_id": "call_2", "content": None},
            {"role": "tool", "tool_call_id": "call_3", "content": "output"},
        ],
    }

    stats = normalize_request(body, thinking_transport=None)

    assert body["messages"][0]["content"] == EMPTY_TOOL_RESULT_PLACEHOLDER
    assert body["messages"][1]["content"] == EMPTY_TOOL_RESULT_PLACEHOLDER
    assert body["messages"][2]["content"] == "output"
    assert stats.empty_tool_results == 2


def test_disabled_reasoning_effort_becomes_the_thinking_field() -> None:
    body: dict[str, Any] = {"model": "deepseek-v4", "reasoning_effort": "off", "messages": []}

    stats = normalize_request(body, thinking_transport="api")

    assert "reasoning_effort" not in body
    assert body["thinking"] == {"type": "disabled"}
    assert stats.thinking_disabled == 1


def test_accepted_reasoning_effort_is_forwarded_untouched() -> None:
    body: dict[str, Any] = {"model": "deepseek-v4", "reasoning_effort": "max", "messages": []}

    stats = normalize_request(body, thinking_transport="api")

    assert body["reasoning_effort"] == "max"
    assert "thinking" not in body
    assert not stats.changed


def test_chat_template_transport_uses_the_vllm_argument() -> None:
    body: dict[str, Any] = {"model": "deepseek-v4-flash", "reasoning_effort": "off", "messages": []}

    stats = normalize_request(body, thinking_transport="chat_template_kwargs")

    # vLLM ignores the API's top-level field and reads the template argument.
    assert body["chat_template_kwargs"] == {"thinking": False}
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert stats.thinking_disabled == 1


def test_chat_template_transport_translates_an_enabled_effort() -> None:
    body: dict[str, Any] = {"model": "deepseek-v4-flash", "reasoning_effort": "max", "messages": []}

    stats = normalize_request(body, thinking_transport="chat_template_kwargs")

    # The template argument is a boolean, so a level cannot survive; forwarding
    # the effort field would leave it silently ignored.
    assert body["chat_template_kwargs"] == {"thinking": True}
    assert "reasoning_effort" not in body
    assert stats.thinking_enabled == 1


def test_chat_template_transport_keeps_the_callers_other_kwargs() -> None:
    body: dict[str, Any] = {
        "model": "deepseek-v4-flash",
        "reasoning_effort": "off",
        "chat_template_kwargs": {"add_generation_prompt": True},
        "messages": [],
    }

    normalize_request(body, thinking_transport="chat_template_kwargs")

    assert body["chat_template_kwargs"] == {"add_generation_prompt": True, "thinking": False}


def test_thinking_mapping_is_deepseek_only() -> None:
    body: dict[str, Any] = {"model": "qwen", "reasoning_effort": "off", "messages": []}

    normalize_request(body, thinking_transport=None)

    assert body["reasoning_effort"] == "off"
    assert "thinking" not in body


def test_normalization_leaves_a_clean_request_alone() -> None:
    body: dict[str, Any] = {
        "model": "deepseek-v4",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    original = {"role": "user", "content": "hi"}

    stats = normalize_request(body, thinking_transport="api")

    assert not stats.changed
    assert body["messages"][0] == original


def test_multipart_assistant_content_is_not_flattened() -> None:
    body: dict[str, Any] = {
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "kept"}]},
        ],
    }

    stats = normalize_request(body, thinking_transport=None)

    assert body["messages"][0]["content"] == [{"type": "text", "text": "kept"}]
    assert not stats.changed


def test_is_empty_completion_only_flags_a_stop_with_no_output() -> None:
    def completion(message: dict[str, object], finish_reason: str | None) -> dict[str, object]:
        return {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}

    assert is_empty_completion(completion({"role": "assistant", "content": ""}, "stop"))
    assert is_empty_completion(completion({"role": "assistant", "content": "   "}, "stop"))
    assert is_empty_completion(completion({"role": "assistant", "content": None}, None))
    # Truncated by the budget: truthfully reported, and a retry burns it again.
    assert not is_empty_completion(completion({"role": "assistant", "content": ""}, "length"))
    assert not is_empty_completion(completion({"role": "assistant", "content": "hi"}, "stop"))
    assert not is_empty_completion(
        completion(
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
            "tool_calls",
        ),
    )
    assert not is_empty_completion({"choices": []})


def test_annotate_empty_completion_replaces_the_message_content() -> None:
    body: dict[str, Any] = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}]
    }

    assert annotate_empty_completion(body, "[proxy: nothing came back]")
    assert body["choices"][0]["message"]["content"] == "[proxy: nothing came back]"
    assert not annotate_empty_completion(body, "")


def test_retry_after_accepts_both_header_forms() -> None:
    now = datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)

    assert retry_after_seconds("3") == 3.0
    assert retry_after_seconds("Sat, 15 Aug 2026 12:00:05 GMT", now=now) == 5.0
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("") is None
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds("-5") is None
    # A distant date cannot park the caller for longer than the clamp.
    assert retry_after_seconds("Sat, 15 Aug 2026 18:00:00 GMT", now=now) == 30.0
    assert retry_after_seconds("600") == 30.0


def test_classify_upstream_status_separates_operator_actions() -> None:
    assert classify_upstream_status(401, "") == "auth"
    assert classify_upstream_status(403, "") == "auth"
    assert classify_upstream_status(429, "rate limit reached") == "rate_limit"
    assert classify_upstream_status(429, "Insufficient Balance") == "quota"
    assert (
        classify_upstream_status(400, "This model's maximum context length is 1000000 tokens")
        == "context_window_exceeded"
    )
    assert classify_upstream_status(400, "unknown field") == "invalid_request"
    assert classify_upstream_status(503, "engine restarting") == "server"
    assert classify_upstream_status(418, "") == "http_4xx"
    assert classify_upstream_status(600, "") == "http_other"


def test_finish_reason_label_is_bounded() -> None:
    assert finish_reason_label("stop") == "stop"
    assert finish_reason_label("tool_calls") == "tool_calls"
    assert finish_reason_label("insufficient_system_resource") == "other"
    assert finish_reason_label(None) == "absent"
