"""Tool-call compatibility transforms for OpenAI-compatible chat responses."""

from __future__ import annotations

import html
import json
import re
import uuid
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    "<DSML: tool_calls>",
    "<tool_calls>",
    "<tool_call>",
)

RAW_TOOL_BLOCK_PATTERNS = (
    (
        re.compile(re.escape(DSML_OPEN), re.DOTALL),
        re.compile(re.escape(DSML_CLOSE), re.DOTALL),
    ),
    (
        re.compile(r"<\|DSML\|tool_calls\s*>", re.DOTALL),
        re.compile(r"</\|DSML\|tool_calls\s*>", re.DOTALL),
    ),
    (
        re.compile(r"<DSML>\s*tool_calls\s*>", re.DOTALL),
        re.compile(r"</DSML[:\s]+tool_calls\s*>", re.DOTALL),
    ),
    (
        re.compile(r"<DSML[:\s]+tool_calls\s*>", re.DOTALL),
        re.compile(r"</DSML[:\s]+tool_calls\s*>", re.DOTALL),
    ),
    (
        re.compile(r"<tool_calls\s*>", re.DOTALL),
        re.compile(r"</tool_calls\s*>", re.DOTALL),
    ),
    (
        re.compile(r"<tool_call\b[^>]*>", re.DOTALL),
        re.compile(r"</tool_call\s*>", re.DOTALL),
    ),
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
    return find_complete_raw_tool_block_span(text) is not None


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
    indexes = [
        match.start()
        for start_pattern, _ in RAW_TOOL_BLOCK_PATTERNS
        for match in start_pattern.finditer(text)
    ]
    if not indexes:
        return None
    return min(indexes)


def find_complete_raw_tool_block_span(text: str) -> tuple[int, int] | None:
    spans: list[tuple[int, int]] = []
    for start_pattern, close_pattern in RAW_TOOL_BLOCK_PATTERNS:
        for start_match in start_pattern.finditer(text):
            close_match = close_pattern.search(text, start_match.end())
            if close_match is not None:
                spans.append((start_match.start(), close_match.end()))
                break

    if not spans:
        return None
    return min(spans, key=lambda span: (span[0], span[1]))


def extract_raw_tool_call_segments(
    text: str,
    *,
    max_raw_tool_block_chars: int | None = None,
) -> tuple[list[ToolCall], str, bool]:
    """Return parsed tool calls and text with parsed raw tool blocks removed."""

    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    cursor = 0
    changed = False

    while cursor < len(text):
        span = find_complete_raw_tool_block_span(text[cursor:])
        if span is None:
            text_parts.append(text[cursor:])
            break

        start = cursor + span[0]
        end = cursor + span[1]
        block = text[start:end]
        if max_raw_tool_block_chars is not None and len(block) > max_raw_tool_block_chars:
            text_parts.append(text[cursor:end])
            cursor = end
            continue

        parsed = parse_raw_tool_calls(block)
        if parsed:
            text_parts.append(text[cursor:start])
            tool_calls.extend(parsed)
            changed = True
        else:
            text_parts.append(text[cursor:end])
        cursor = end

    return _dedupe_tool_calls(tool_calls), "".join(text_parts), changed


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

        json_matches = _parse_json_tool_call_block(block)
        if json_matches:
            results.extend(json_matches)
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


def convert_chat_completion_response(
    body: JsonObject,
    *,
    tool_call_scan_fields: Iterable[str] = ("content", "reasoning", "reasoning_content"),
    max_raw_tool_block_chars: int | None = None,
    max_tool_calls: int | None = None,
    max_tool_argument_chars: int | None = None,
) -> tuple[JsonObject, bool]:
    """Convert non-streaming OpenAI-compatible chat completion JSON in place."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return body, False

    changed = False
    scan_fields = tuple(tool_call_scan_fields)
    for choice in choices:
        if isinstance(choice, dict) and _convert_chat_completion_choice(
            choice,
            tool_call_scan_fields=scan_fields,
            max_raw_tool_block_chars=max_raw_tool_block_chars,
            max_tool_calls=max_tool_calls,
            max_tool_argument_chars=max_tool_argument_chars,
        ):
            changed = True

    return body, changed


def _convert_chat_completion_choice(
    choice: JsonObject,
    *,
    tool_call_scan_fields: Iterable[str],
    max_raw_tool_block_chars: int | None,
    max_tool_calls: int | None,
    max_tool_argument_chars: int | None,
) -> bool:
    message = choice.get("message")
    if not isinstance(message, dict):
        return False

    existing_tool_calls = message.get("tool_calls")
    if existing_tool_calls:
        return False

    for field in tool_call_scan_fields:
        value = message.get(field)
        if not isinstance(value, str) or not has_complete_raw_tool_block(value):
            continue

        tool_calls, remaining_text, changed = extract_raw_tool_call_segments(
            value,
            max_raw_tool_block_chars=max_raw_tool_block_chars,
        )
        if not changed or not tool_calls:
            continue
        if not tool_calls_within_limits(
            tool_calls,
            max_tool_calls=max_tool_calls,
            max_tool_argument_chars=max_tool_argument_chars,
        ):
            continue

        message["tool_calls"] = tool_calls
        message[field] = remaining_text if remaining_text.strip() else None
        if "content" not in message:
            message["content"] = None
        choice["finish_reason"] = "tool_calls"
        return True

    return False


def tool_calls_within_limits(
    tool_calls: Iterable[ToolCall],
    *,
    max_tool_calls: int | None = None,
    max_tool_argument_chars: int | None = None,
) -> bool:
    tool_call_list = list(tool_calls)
    if max_tool_calls is not None and len(tool_call_list) > max_tool_calls:
        return False
    if max_tool_argument_chars is not None:
        for tool_call in tool_call_list:
            if len(tool_call["function"]["arguments"]) > max_tool_argument_chars:
                return False
    return True


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
    choice_index: int = 0,
    include_finish: bool = True,
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
                choice_index=choice_index,
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
                    choice_index=choice_index,
                ),
            )

    if include_finish:
        chunks.append(
            _make_chunk(
                chunk_id=chunk_id,
                model=model,
                delta={},
                finish_reason="tool_calls",
                choice_index=choice_index,
            ),
        )
    return chunks


def make_finish_chunk(
    *,
    chunk_id: str,
    model: str,
    finish_reason: str,
    choice_index: int = 0,
) -> ChatCompletionChunk:
    return _make_chunk(
        chunk_id=chunk_id,
        model=model,
        delta={},
        finish_reason=finish_reason,
        choice_index=choice_index,
    )


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


def _parse_json_tool_call_block(block: str) -> list[ToolCall]:
    raw = html.unescape(block).strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    objects: list[JsonObject]
    if isinstance(parsed, dict):
        objects = [parsed]
    elif isinstance(parsed, list):
        objects = [item for item in parsed if isinstance(item, dict)]
    else:
        return []

    results: list[ToolCall] = []
    for item in objects:
        name = item.get("name") or item.get("function")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = item.get("arguments", item.get("parameters", {}))
        results.append(make_tool_call(name, arguments))
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
    choice_index: int = 0,
) -> ChatCompletionChunk:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": choice_index,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ],
    }
