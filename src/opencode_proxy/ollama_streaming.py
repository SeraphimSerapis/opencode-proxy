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

    from opencode_proxy.proxy import ToolRepairContext
    from opencode_proxy.settings import Settings


@dataclass
class _ToolAccumulator:
    max_argument_chars: int
    names: dict[tuple[int, int], str] = field(default_factory=dict)
    arguments: dict[tuple[int, int], str] = field(default_factory=dict)
    oversized: set[tuple[int, int]] = field(default_factory=set)

    def add(self, choice_index: int, call_index: int, function: dict[str, Any]) -> None:
        key = (choice_index, call_index)
        name = function.get("name")
        if isinstance(name, str) and name:
            self.names[key] = name
        fragment = function.get("arguments")
        if isinstance(fragment, str):
            current = self.arguments.get(key, "")
            remaining = self.max_argument_chars - len(current)
            if remaining <= 0:
                self.oversized.add(key)
            else:
                self.arguments[key] = current + fragment[:remaining]
                if len(fragment) > remaining:
                    self.oversized.add(key)

    def pop(self, choice_index: int) -> list[OllamaToolCall]:
        keys = sorted(
            key for key in self.names.keys() | self.arguments.keys() if key[0] == choice_index
        )
        calls = [
            OllamaToolCall(
                function=OllamaFunction(
                    name=self.names.get(key, ""),
                    arguments=(
                        {} if key in self.oversized else _json_object(self.arguments.get(key, ""))
                    ),
                )
            )
            for key in keys
            if self.names.get(key)
        ]
        for key in keys:
            self.names.pop(key, None)
            self.arguments.pop(key, None)
            self.oversized.discard(key)
        return calls


async def stream_chat_to_ollama(
    request: Request,
    response: httpx.Response,
    settings: Settings,
    model: str,
    repair_context: ToolRepairContext,
) -> AsyncIterator[bytes]:
    tools = _ToolAccumulator(max_argument_chars=settings.max_tool_argument_chars)
    usage: dict[str, Any] = {}
    terminal: dict[int, tuple[OllamaMessage, str]] = {}
    async for raw_frame in _rewrite_sse_stream(request, response, settings, repair_context):
        for event in _json_events(raw_frame):
            if event == "[DONE]":
                async for line in _terminal_chat_lines(terminal, tools, model, usage):
                    yield line
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
                    if message.content is None:
                        message.content = ""
                    calls = tools.pop(choice_index)
                    if calls:
                        message.tool_calls = calls
                    terminal[choice_index] = (message, _ollama_finish(finish_reason))
                elif _message_has_content(message):
                    yield _chat_line(model, message, False, None, {})
    async for line in _terminal_chat_lines(terminal, tools, model, usage):
        yield line


async def stream_generate_to_ollama(
    request: Request,
    response: httpx.Response,
    settings: Settings,
    model: str,
    repair_context: ToolRepairContext,
) -> AsyncIterator[bytes]:
    terminal: tuple[str, str | None, str] | None = None
    usage: dict[str, Any] = {}
    stream_complete = False
    async for raw_frame in _rewrite_sse_stream(request, response, settings, repair_context):
        for event in _json_events(raw_frame):
            if event == "[DONE]":
                stream_complete = True
                break
            if not isinstance(event, dict):
                continue
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
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
                terminal = (content, thinking, _ollama_finish(finish_reason))
            elif content or thinking:
                yield _generate_line(
                    model,
                    content,
                    thinking,
                    False,
                    None,
                    {},
                )
        if stream_complete:
            break
    if terminal is None:
        terminal = ("", None, "stop")
    yield _generate_line(model, terminal[0], terminal[1], True, terminal[2], usage)


async def _terminal_chat_lines(
    terminal: dict[int, tuple[OllamaMessage, str]],
    tools: _ToolAccumulator,
    model: str,
    usage: dict[str, Any],
) -> AsyncIterator[bytes]:
    for choice_index in sorted({*terminal, *(key[0] for key in tools.names | tools.arguments)}):
        message, reason = terminal.get(
            choice_index, (OllamaMessage(role="assistant", content=""), "stop")
        )
        calls = tools.pop(choice_index)
        if calls:
            message.tool_calls = calls
        yield _chat_line(model, message, True, reason, usage)
    if not terminal and not tools.names and not tools.arguments:
        yield _chat_line(model, OllamaMessage(role="assistant", content=""), True, "stop", usage)


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
