"""Streaming conversion from repaired OpenAI SSE events to Ollama NDJSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opencode_proxy.ollama_models import (
    OllamaChatResponse,
    OllamaFunction,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaToolCall,
)
from opencode_proxy.proxy import _rewrite_sse_stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx
    from fastapi import Request

    from opencode_proxy.settings import Settings


@dataclass
class _ToolAccumulator:
    names: dict[tuple[int, int], str] = field(default_factory=dict)
    arguments: dict[tuple[int, int], str] = field(default_factory=dict)

    def add(self, choice_index: int, call_index: int, function: dict[str, Any]) -> None:
        key = (choice_index, call_index)
        name = function.get("name")
        if isinstance(name, str) and name:
            self.names[key] = name
        fragment = function.get("arguments")
        if isinstance(fragment, str):
            self.arguments[key] = self.arguments.get(key, "") + fragment

    def pop(self, choice_index: int) -> list[OllamaToolCall]:
        keys = sorted(
            key for key in self.names.keys() | self.arguments.keys() if key[0] == choice_index
        )
        calls = [
            OllamaToolCall(
                function=OllamaFunction(
                    name=self.names.get(key, ""),
                    arguments=_json_object(self.arguments.get(key, "")),
                )
            )
            for key in keys
            if self.names.get(key)
        ]
        for key in keys:
            self.names.pop(key, None)
            self.arguments.pop(key, None)
        return calls


async def stream_chat_to_ollama(
    request: Request,
    response: httpx.Response,
    settings: Settings,
    model: str,
) -> AsyncIterator[bytes]:
    tools = _ToolAccumulator()
    usage: dict[str, Any] = {}
    done = False
    async for raw_frame in _rewrite_sse_stream(request, response, settings):
        for event in _json_events(raw_frame):
            if event == "[DONE]":
                if not done:
                    for choice_index in sorted({key[0] for key in tools.names | tools.arguments}):
                        calls = tools.pop(choice_index)
                        if calls:
                            yield _chat_line(
                                model,
                                OllamaMessage(role="assistant", tool_calls=calls),
                                True,
                                "stop",
                                usage,
                            )
                    yield _chat_line(
                        model,
                        OllamaMessage(role="assistant", content=""),
                        True,
                        "stop",
                        usage,
                    )
                return
            if not isinstance(event, dict):
                continue
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            for raw_choice in choices:
                if not isinstance(raw_choice, dict):
                    continue
                raw_choice_index = raw_choice.get("index")
                choice_index = raw_choice_index if isinstance(raw_choice_index, int) else 0
                delta = raw_choice.get("delta")
                delta = delta if isinstance(delta, dict) else {}
                for raw_call in delta.get("tool_calls", []):
                    if not isinstance(raw_call, dict):
                        continue
                    call_index = raw_call.get("index")
                    call_index = call_index if isinstance(call_index, int) else 0
                    function = raw_call.get("function")
                    if isinstance(function, dict):
                        tools.add(choice_index, call_index, function)

                finish_reason = raw_choice.get("finish_reason")
                message = _message_from_delta(delta)
                if finish_reason in {"tool_calls", "stop", "length", "content_filter"}:
                    calls = tools.pop(choice_index)
                    if calls:
                        message.tool_calls = calls
                    yield _chat_line(model, message, True, _ollama_finish(finish_reason), usage)
                    done = True
                elif _message_has_content(message):
                    yield _chat_line(model, message, False, None, {})
    if not done:
        calls = tools.pop(0)
        if calls:
            yield _chat_line(
                model, OllamaMessage(role="assistant", tool_calls=calls), True, "stop", usage
            )
        yield _chat_line(model, OllamaMessage(role="assistant", content=""), True, "stop", usage)


async def stream_generate_to_ollama(
    request: Request,
    response: httpx.Response,
    settings: Settings,
    model: str,
) -> AsyncIterator[bytes]:
    done = False
    async for raw_frame in _rewrite_sse_stream(request, response, settings):
        for event in _json_events(raw_frame):
            if event == "[DONE]":
                if not done:
                    yield _generate_line(model, "", None, True, "stop", {})
                return
            if not isinstance(event, dict):
                continue
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            delta = delta if isinstance(delta, dict) else {}
            finish_reason = choice.get("finish_reason")
            content = _string_or_empty(delta.get("content"))
            thinking = delta.get("reasoning_content")
            thinking = thinking if isinstance(thinking, str) else None
            is_done = isinstance(finish_reason, str) and finish_reason in {
                "tool_calls",
                "stop",
                "length",
                "content_filter",
            }
            if is_done:
                done = True
            if content or thinking or is_done:
                yield _generate_line(
                    model,
                    content,
                    thinking,
                    is_done,
                    _ollama_finish(finish_reason) if is_done else None,
                    {},
                )
    if not done:
        yield _generate_line(model, "", None, True, "stop", {})


def _json_events(frame: bytes) -> list[dict[str, Any] | str]:
    events: list[dict[str, Any] | str] = []
    for line in frame.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if payload == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _message_from_delta(delta: dict[str, Any]) -> OllamaMessage:
    raw_role = delta.get("role")
    role = "assistant"
    if isinstance(raw_role, str):
        role = raw_role
    content = delta.get("content") if isinstance(delta.get("content"), str) else None
    thinking = delta.get("reasoning_content") or delta.get("reasoning")
    thinking = thinking if isinstance(thinking, str) else None
    return OllamaMessage(role=role, content=content, thinking=thinking)


def _message_has_content(message: OllamaMessage) -> bool:
    return (
        message.content is not None or message.thinking is not None or message.role != "assistant"
    )


def _chat_line(
    model: str,
    message: OllamaMessage,
    done: bool,
    reason: str | None,
    usage: dict[str, Any],
) -> bytes:
    return _ndjson(
        OllamaChatResponse(
            model=model,
            created_at=_now_iso(),
            message=message,
            done=done,
            done_reason=reason,
            prompt_eval_count=_int_or_none(usage.get("prompt_tokens")),
            eval_count=_int_or_none(usage.get("completion_tokens")),
        )
    )


def _generate_line(
    model: str,
    content: str,
    thinking: str | None,
    done: bool,
    reason: str | None,
    usage: dict[str, Any],
) -> bytes:
    return _ndjson(
        OllamaGenerateResponse(
            model=model,
            created_at=_now_iso(),
            response=content,
            thinking=thinking,
            done=done,
            done_reason=reason,
            prompt_eval_count=_int_or_none(usage.get("prompt_tokens")),
            eval_count=_int_or_none(usage.get("completion_tokens")),
        )
    )


def _ndjson(value: OllamaChatResponse | OllamaGenerateResponse) -> bytes:
    return (
        json.dumps(
            value.model_dump(exclude_none=True), separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
    )


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ollama_finish(reason: object) -> str:
    return "length" if reason == "length" else "stop"


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
