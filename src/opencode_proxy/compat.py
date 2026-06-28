"""Tool-call compatibility transforms for OpenAI-compatible chat responses."""

from __future__ import annotations

import html
import json
import re
import uuid
from collections.abc import Iterable
from typing import Any, NotRequired, TypedDict, cast

JsonObject = dict[str, Any]

FULLWIDTH_BAR = "\uff5c"
DSML_OPEN = f"<{FULLWIDTH_BAR}DSML{FULLWIDTH_BAR}tool_calls>"
DSML_CLOSE = f"</{FULLWIDTH_BAR}DSML{FULLWIDTH_BAR}tool_calls>"
DSML_INVOKE_OPEN = f"<{FULLWIDTH_BAR}DSML{FULLWIDTH_BAR}invoke"
DSML_INVOKE_CLOSE = f"</{FULLWIDTH_BAR}DSML{FULLWIDTH_BAR}invoke>"
DSML_PARAMETER_OPEN = f"<{FULLWIDTH_BAR}DSML{FULLWIDTH_BAR}parameter"
DSML_PARAMETER_CLOSE = f"</{FULLWIDTH_BAR}DSML{FULLWIDTH_BAR}parameter>"

RAW_TOOL_START_MARKERS = (
    DSML_OPEN,
    "<|DSML|tool_calls>",
    "<DSML>tool_calls>",
    "<tool_calls>",
    "<tool_call>",
)


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: FunctionCall


class DeltaToolCallFunction(TypedDict, total=False):
    name: str
    arguments: str


class DeltaToolCall(TypedDict, total=False):
    index: int
    id: str
    type: str
    function: DeltaToolCallFunction


class ChatCompletionDelta(TypedDict, total=False):
    role: str
    content: str | None
    reasoning: str
    reasoning_content: str
    tool_calls: list[DeltaToolCall]


class ChatCompletionChoice(TypedDict, total=False):
    index: int
    message: JsonObject
    delta: ChatCompletionDelta
    finish_reason: str | None


class ChatCompletionChunk(TypedDict):
    id: str
    object: str
    model: str
    choices: list[ChatCompletionChoice]
    created: NotRequired[int]


def normalize_raw_tool_markup(text: str) -> str:
    """Convert known raw tool-call variants into one canonical DSML-ish shape."""

    normalized = text
    normalized = normalized.replace("<|DSML|tool_calls>", DSML_OPEN)
    normalized = normalized.replace("</|DSML|tool_calls>", DSML_CLOSE)
    normalized = normalized.replace("<|DSML|invoke", DSML_INVOKE_OPEN)
    normalized = normalized.replace("</|DSML|invoke>", DSML_INVOKE_CLOSE)
    normalized = normalized.replace("<|DSML|parameter", DSML_PARAMETER_OPEN)
    normalized = normalized.replace("</|DSML|parameter>", DSML_PARAMETER_CLOSE)

    normalized = normalized.replace("<DSML>tool_calls>", DSML_OPEN, 1)
    normalized = re.sub(r"</DSML[:\s]+tool_calls\s*>", DSML_CLOSE, normalized)
    normalized = re.sub(r"<DSML[:\s]+invoke\s+", f"{DSML_INVOKE_OPEN} ", normalized)
    normalized = re.sub(r"</DSML[:\s]+invoke\s*>", DSML_INVOKE_CLOSE, normalized)
    normalized = re.sub(r"<DSML[:\s]+parameter\s+", f"{DSML_PARAMETER_OPEN} ", normalized)
    normalized = re.sub(r"</DSML[:\s]+parameter\s*>", DSML_PARAMETER_CLOSE, normalized)

    if "<tool_calls>" in normalized and "</tool_calls>" in normalized:
        normalized = normalized.replace("<tool_calls>", DSML_OPEN, 1)
        normalized = normalized.replace("</tool_calls>", DSML_CLOSE, 1)
        normalized = re.sub(r"<invoke\s+", f"{DSML_INVOKE_OPEN} ", normalized)
        normalized = re.sub(r"</invoke\s*>", DSML_INVOKE_CLOSE, normalized)
        normalized = re.sub(r"<parameter\s+", f"{DSML_PARAMETER_OPEN} ", normalized)
        normalized = re.sub(r"</parameter\s*>", DSML_PARAMETER_CLOSE, normalized)

    return normalized


def has_complete_raw_tool_block(text: str) -> bool:
    normalized = normalize_raw_tool_markup(text)
    return (DSML_OPEN in normalized and DSML_CLOSE in normalized) or (
        "<tool_call>" in normalized and "</tool_call>" in normalized
    )


def has_raw_tool_prefix(text: str) -> bool:
    """Return true when the text tail may contain a split raw tool-call marker."""

    if not text:
        return False

    normalized = normalize_raw_tool_markup(text)
    tail = normalized[-200:]
    if any(marker in tail for marker in RAW_TOOL_START_MARKERS):
        return True

    for marker in RAW_TOOL_START_MARKERS:
        max_prefix = min(len(marker) - 1, len(tail))
        for size in range(max_prefix, 3, -1):
            if tail.endswith(marker[:size]):
                return True

    return False


def find_raw_tool_start(text: str) -> int | None:
    normalized = normalize_raw_tool_markup(text)
    indexes = [idx for marker in RAW_TOOL_START_MARKERS if (idx := normalized.find(marker)) != -1]
    if not indexes:
        return None
    return min(indexes)


def normalize_argument_value(value: object) -> str:
    if value is None:
        return "{}"

    if isinstance(value, str):
        unescaped = html.unescape(value).strip()
        if not unescaped:
            return "{}"
        try:
            parsed = json.loads(unescaped)
        except json.JSONDecodeError:
            return unescaped
        if isinstance(parsed, dict | list):
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return unescaped

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_tool_call(name: str, arguments: object, call_id: str | None = None) -> ToolCall:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": html.unescape(name).strip(),
            "arguments": normalize_argument_value(arguments),
        },
    }


def parse_raw_tool_calls(text: str) -> list[ToolCall]:
    normalized = normalize_raw_tool_markup(text)
    parsed = [*parse_dsml_tool_calls(normalized), *parse_qwen_xml_tool_calls(normalized)]
    return _dedupe_tool_calls(parsed)


def parse_dsml_tool_calls(text: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    block_pattern = re.compile(
        re.escape(DSML_OPEN) + r"(?P<body>.*?)" + re.escape(DSML_CLOSE),
        re.DOTALL,
    )
    for match in block_pattern.finditer(text):
        block = match.group("body")
        results.extend(_parse_name_parameter_blocks(block))
        results.extend(_parse_dsml_invoke_blocks(block))
    return results


def parse_qwen_xml_tool_calls(text: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for match in re.finditer(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", text, re.DOTALL):
        block = match.group("body")
        name_matches = _parse_name_parameter_blocks(block)
        if name_matches:
            results.extend(name_matches)
            continue

        function_match = re.search(
            r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>",
            block,
            re.DOTALL,
        )
        if function_match is None:
            continue

        params: JsonObject = {}
        for param in re.finditer(
            r"<parameter=(?P<name>[^>]+)>(?P<value>.*?)</parameter>",
            function_match.group("body"),
            re.DOTALL,
        ):
            params[html.unescape(param.group("name")).strip()] = html.unescape(
                param.group("value"),
            ).strip()
        results.append(make_tool_call(function_match.group("name"), params))
    return results


def convert_chat_completion_response(body: JsonObject) -> tuple[JsonObject, bool]:
    """Convert non-streaming OpenAI-compatible chat completion JSON in place."""

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return body, False

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return body, False

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return body, False

    existing_tool_calls = message.get("tool_calls")
    if existing_tool_calls:
        return body, False

    content = message.get("content")
    if not isinstance(content, str) or not has_complete_raw_tool_block(content):
        return body, False

    tool_calls = parse_raw_tool_calls(content)
    if not tool_calls:
        return body, False

    message["tool_calls"] = tool_calls
    message["content"] = None
    first_choice["finish_reason"] = "tool_calls"
    return body, True


def collect_delta_text(delta: JsonObject) -> str:
    parts: list[str] = []
    for field in ("content", "reasoning_content", "reasoning"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
    return "".join(parts)


def strip_empty_tool_calls(delta: JsonObject) -> JsonObject:
    tool_calls = delta.get("tool_calls")
    if tool_calls == []:
        cleaned = dict(delta)
        cleaned.pop("tool_calls", None)
        return cleaned
    return delta


def build_tool_call_chunks(
    tool_calls: Iterable[ToolCall],
    *,
    chunk_id: str,
    model: str,
    argument_chunk_size: int,
) -> list[ChatCompletionChunk]:
    chunks: list[ChatCompletionChunk] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call["function"]
        chunks.append(
            _make_chunk(
                chunk_id=chunk_id,
                model=model,
                delta={
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": function["name"],
                                "arguments": "",
                            },
                        },
                    ],
                },
                finish_reason=None,
            ),
        )

        arguments = function["arguments"]
        for start in range(0, len(arguments), argument_chunk_size):
            chunks.append(
                _make_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    delta={
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {
                                    "arguments": arguments[start : start + argument_chunk_size],
                                },
                            },
                        ],
                    },
                    finish_reason=None,
                ),
            )

    chunks.append(
        _make_chunk(
            chunk_id=chunk_id,
            model=model,
            delta={},
            finish_reason="tool_calls",
        ),
    )
    return chunks


def _parse_name_parameter_blocks(block: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for match in re.finditer(
        r"<name>\s*(?P<name>.*?)\s*</name>.*?<parameters>\s*(?P<args>.*?)\s*</parameters>",
        block,
        re.DOTALL,
    ):
        results.append(make_tool_call(match.group("name"), match.group("args")))
    return results


def _parse_dsml_invoke_blocks(block: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    invoke_pattern = re.compile(
        re.escape(DSML_INVOKE_OPEN) + r"""\s+name=(?P<quote>["'])(?P<name>.*?)(?P=quote)\s*>""",
        re.DOTALL,
    )
    for invoke in invoke_pattern.finditer(block):
        remaining = block[invoke.end() :]
        end = remaining.find(DSML_INVOKE_CLOSE)
        if end == -1:
            continue

        params: JsonObject = {}
        for param in re.finditer(
            re.escape(DSML_PARAMETER_OPEN)
            + r"""\s+name=(?P<quote>["'])(?P<name>.*?)(?P=quote)[^>]*>"""
            + r"(?P<value>.*?)"
            + re.escape(DSML_PARAMETER_CLOSE),
            remaining[:end],
            re.DOTALL,
        ):
            params[html.unescape(param.group("name")).strip()] = html.unescape(
                param.group("value"),
            ).strip()
        results.append(make_tool_call(invoke.group("name"), params))
    return results


def _dedupe_tool_calls(tool_calls: list[ToolCall]) -> list[ToolCall]:
    deduped: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for tool_call in tool_calls:
        key = (tool_call["function"]["name"], tool_call["function"]["arguments"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tool_call)
    return deduped


def _make_chunk(
    *,
    chunk_id: str,
    model: str,
    delta: ChatCompletionDelta,
    finish_reason: str | None,
) -> ChatCompletionChunk:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": cast(ChatCompletionDelta, delta),
                "finish_reason": finish_reason,
            },
        ],
    }
