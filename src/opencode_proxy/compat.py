"""Tool-call compatibility transforms for OpenAI-compatible chat responses."""

from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass, field
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
ORPHAN_DSML_INVOKE_START = DSML_INVOKE_OPEN

DSML_DEGRADED_OPEN = r"<DSML(?:>\s*|:\s*|\s+)tool_calls\s*>"
DSML_DEGRADED_CLOSE = r"</DSML(?:>\s*|:\s*|\s+)tool_calls\s*>"

# Common complete openers recognized by ``has_raw_tool_prefix``. Complete block
# detection is governed by ``RAW_TOOL_BLOCK_PATTERNS`` below.
RAW_TOOL_START_MARKERS = (
    DSML_OPEN,
    "<|DSML|tool_calls>",
    "<DSML>tool_calls>",
    "<DSML: tool_calls>",
    "<DSML:tool_calls>",
    "<DSML tool_calls>",
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
        re.compile(DSML_DEGRADED_OPEN, re.DOTALL),
        re.compile(DSML_DEGRADED_CLOSE, re.DOTALL),
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


@dataclass
class RepairStats:
    raw_repairs: list[tuple[str, str]] = field(default_factory=list)
    orphan_accepted: int = 0
    orphan_rejected: dict[str, int] = field(default_factory=dict)
    synthesized_ids: int = 0

    def reject_orphan(self, reason: str) -> None:
        self.orphan_rejected[reason] = self.orphan_rejected.get(reason, 0) + 1


def normalize_raw_tool_markup(text: str) -> str:
    """Convert known raw tool-call variants into one canonical DSML-ish shape."""

    normalized = text
    normalized = re.sub(r"<\|DSML\|tool_calls\s*>", DSML_OPEN, normalized)
    normalized = re.sub(r"</\|DSML\|tool_calls\s*>", DSML_CLOSE, normalized)
    normalized = normalized.replace("<|DSML|invoke", DSML_INVOKE_OPEN)
    normalized = normalized.replace("</|DSML|invoke>", DSML_INVOKE_CLOSE)
    normalized = normalized.replace("<|DSML|parameter", DSML_PARAMETER_OPEN)
    normalized = normalized.replace("</|DSML|parameter>", DSML_PARAMETER_CLOSE)

    normalized = re.sub(DSML_DEGRADED_OPEN, DSML_OPEN, normalized, count=1)
    normalized = re.sub(DSML_DEGRADED_CLOSE, DSML_CLOSE, normalized)
    normalized = re.sub(r"<DSML[:\s]+invoke\s+", f"{DSML_INVOKE_OPEN} ", normalized)
    normalized = re.sub(r"</DSML[:\s]+invoke\s*>", DSML_INVOKE_CLOSE, normalized)
    normalized = re.sub(r"<DSML[:\s]+parameter\s+", f"{DSML_PARAMETER_OPEN} ", normalized)
    normalized = re.sub(r"</DSML[:\s]+parameter\s*>", DSML_PARAMETER_CLOSE, normalized)

    if re.search(r"<tool_calls\s*>", normalized) and re.search(r"</tool_calls\s*>", normalized):
        normalized = re.sub(r"<tool_calls\s*>", DSML_OPEN, normalized, count=1)
        normalized = re.sub(r"</tool_calls\s*>", DSML_CLOSE, normalized, count=1)
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

    return tool_calls, "".join(text_parts), changed


def find_orphan_dsml_invoke_start(text: str) -> int | None:
    """Find a canonical V4 invoke opener before any non-whitespace text."""
    cursor = 0
    while True:
        start = text.find(ORPHAN_DSML_INVOKE_START, cursor)
        if start < 0:
            return None
        if not text[:start].strip():
            return start
        cursor = start + 1


def find_complete_orphan_dsml_invoke_span(text: str) -> tuple[int, int] | None:
    start = find_orphan_dsml_invoke_start(text)
    while start is not None:
        end = text.find(DSML_INVOKE_CLOSE, start + len(ORPHAN_DSML_INVOKE_START))
        if end >= 0:
            return start, end + len(DSML_INVOKE_CLOSE)
        next_offset = start + len(ORPHAN_DSML_INVOKE_START)
        nested = find_orphan_dsml_invoke_start(text[next_offset:])
        start = None if nested is None else next_offset + nested
    return None


def extract_orphan_dsml_invokes(
    text: str,
    *,
    declared_tool_names: frozenset[str],
    max_raw_tool_block_chars: int | None = None,
    max_tool_calls: int | None = None,
    max_tool_argument_chars: int | None = None,
    stats: RepairStats | None = None,
) -> tuple[list[ToolCall], str, bool]:
    """Recover canonical V4 invokes missing only their outer tool-call wrapper.

    Rejected candidates remain byte-for-byte identical in ``remaining_text``.
    ASCII/degraded DSML is deliberately excluded because this fallback targets
    one concrete DeepSeek V4/vLLM regression, not every DSML-looking dialect.
    """
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    cursor = 0
    changed = False

    while cursor < len(text):
        span = find_complete_orphan_dsml_invoke_span(text[cursor:])
        if span is None:
            text_parts.append(text[cursor:])
            break
        start = cursor + span[0]
        end = cursor + span[1]
        block = text[start:end]
        text_parts.append(text[cursor:start])

        rejection: str | None = None
        parsed = _parse_single_orphan_dsml_invoke(block)
        if max_raw_tool_block_chars is not None and len(block) > max_raw_tool_block_chars:
            rejection = "oversized_block"
        elif parsed is None:
            rejection = "malformed"
        elif parsed["function"]["name"] not in declared_tool_names:
            rejection = "undeclared_tool"
        elif not tool_calls_within_limits(
            [*tool_calls, parsed],
            max_tool_calls=max_tool_calls,
            max_tool_argument_chars=max_tool_argument_chars,
        ):
            rejection = "limits"

        if rejection is not None:
            text_parts.append(block)
            if stats is not None:
                stats.reject_orphan(rejection)
        else:
            assert parsed is not None
            tool_calls.append(parsed)
            changed = True
            if stats is not None:
                stats.orphan_accepted += 1
        cursor = end

    return tool_calls, "".join(text_parts), changed


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
    return [*parse_dsml_tool_calls(normalized), *parse_qwen_xml_tool_calls(normalized)]


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
    for match in re.finditer(
        r"<tool_call\b[^>]*>\s*(?P<body>.*?)\s*</tool_call\s*>",
        text,
        re.DOTALL,
    ):
        block = match.group("body")
        name_matches = _parse_name_parameter_blocks(block)
        if name_matches:
            results.extend(name_matches)
            continue

        laguna_matches = _parse_laguna_tool_call_blocks(block)
        if laguna_matches:
            results.extend(laguna_matches)
            continue

        json_matches = _parse_json_tool_call_block(block)
        if json_matches:
            results.extend(json_matches)
            continue

        for function_match in re.finditer(
            r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>",
            block,
            re.DOTALL,
        ):
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


def is_empty_completion(body: JsonObject) -> bool:
    """True when a completed turn carries nothing the caller can act on.

    A model that closes with ``stop`` but produced no content and no tool call
    has failed, not answered: agent clients render nothing and execute nothing,
    which is indistinguishable from a hang. DeepSeek's own client treats exactly
    this shape as a retryable error rather than a successful empty message.

    A turn cut short by ``length`` is excluded. That one is truthfully reported
    and retrying it unchanged only burns the same budget again.
    """
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            return False
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and finish_reason != "stop":
            return False
        message = choice.get("message")
        if not isinstance(message, dict):
            return False
        if message.get("tool_calls"):
            return False
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return False
        if isinstance(content, list) and content:
            return False
    return True


def annotate_empty_completion(body: JsonObject, notice: str) -> bool:
    """Replace empty assistant content with ``notice`` so the turn is visibly dead.

    Mirrors the streamed empty-turn annotation for the buffered transport.
    """
    if not notice:
        return False
    choices = body.get("choices")
    if not isinstance(choices, list):
        return False

    annotated = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        message["content"] = notice
        annotated = True
    return annotated


def convert_chat_completion_response(
    body: JsonObject,
    *,
    tool_call_scan_fields: Iterable[str] = ("content", "reasoning", "reasoning_content"),
    max_raw_tool_block_chars: int | None = None,
    max_tool_calls: int | None = None,
    max_tool_argument_chars: int | None = None,
    recover_orphan_invokes: bool = False,
    declared_tool_names: frozenset[str] = frozenset(),
    stats: RepairStats | None = None,
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
            recover_orphan_invokes=recover_orphan_invokes,
            declared_tool_names=declared_tool_names,
            stats=stats,
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
    recover_orphan_invokes: bool,
    declared_tool_names: frozenset[str],
    stats: RepairStats | None,
) -> bool:
    message = choice.get("message")
    if not isinstance(message, dict):
        return False

    existing_tool_calls = message.get("tool_calls")
    if isinstance(existing_tool_calls, list) and existing_tool_calls:
        synthesized = normalize_native_tool_call_ids(existing_tool_calls)
        if stats is not None:
            stats.synthesized_ids += synthesized
        return synthesized > 0

    for field_name in tool_call_scan_fields:
        value = message.get(field_name)
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
        message[field_name] = remaining_text if remaining_text.strip() else None
        if "content" not in message:
            message["content"] = None
        choice["finish_reason"] = "tool_calls"
        if stats is not None:
            stats.raw_repairs.append((raw_tool_format(value), field_name))
        return True

    content = message.get("content")
    if (
        recover_orphan_invokes
        and declared_tool_names
        and isinstance(content, str)
        and find_orphan_dsml_invoke_start(content) is not None
    ):
        tool_calls, remaining_text, changed = extract_orphan_dsml_invokes(
            content,
            declared_tool_names=declared_tool_names,
            max_raw_tool_block_chars=max_raw_tool_block_chars,
            max_tool_calls=max_tool_calls,
            max_tool_argument_chars=max_tool_argument_chars,
            stats=stats,
        )
        if changed and tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = remaining_text if remaining_text.strip() else None
            choice["finish_reason"] = "tool_calls"
            if stats is not None:
                stats.raw_repairs.append(("deepseek_v4_orphan", "content"))
            return True

    return False


def normalize_native_tool_call_ids(tool_calls: list[object]) -> int:
    """Add IDs only to otherwise valid OpenAI function tool calls."""
    synthesized = 0
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        existing_id = tool_call.get("id")
        if isinstance(existing_id, str) and existing_id:
            continue
        function = tool_call.get("function")
        if (
            tool_call.get("type", "function") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
        ):
            continue
        tool_call["id"] = f"call_{uuid.uuid4().hex[:24]}"
        synthesized += 1
    return synthesized


def raw_tool_format(text: str) -> str:
    if DSML_OPEN in text or "<|DSML|tool_calls" in text or "<DSML" in text:
        return "dsml"
    if "<tool_call" in text:
        return "qwen_xml"
    return "unknown"


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


def complete_truncated_json(text: str) -> str | None:
    """Return the suffix that turns truncated JSON into a valid document.

    Upstreams sometimes close a turn with ``finish_reason: "tool_calls"`` while
    the streamed ``arguments`` are cut off mid-value, which leaves the client
    holding a tool call it cannot execute. Deltas already sent cannot be
    retracted, so a repair has to be expressible as an *append*.

    Returns ``""`` when ``text`` already parses, the completing suffix when one
    exists, and ``None`` when the truncation cannot be repaired by appending
    (a dangling ``,`` or an empty fragment). The result is always verified with
    ``json.loads`` before it is returned, so a non-``None`` return is a promise
    that ``text + suffix`` parses.
    """
    if not text.strip():
        return None

    try:
        json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        pass
    else:
        return ""

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()

    suffix = ""
    if in_string:
        # A trailing backslash is a half-written escape; completing it as an
        # escaped backslash is the only single-character reading that keeps the
        # value a string.
        if escaped:
            suffix += "\\"
        suffix += '"'

    tail = (text + suffix).rstrip()
    if tail.endswith(":"):
        suffix += "null"
    elif tail.endswith(","):
        # Removing the comma would mean rewriting bytes the client already has.
        return None

    for opener in reversed(stack):
        suffix += "}" if opener == "{" else "]"

    try:
        json.loads(text + suffix)
    except (json.JSONDecodeError, RecursionError):
        return None
    return suffix


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
    tool_index_offset: int = 0,
    include_finish: bool = True,
) -> list[ChatCompletionChunk]:
    chunks: list[ChatCompletionChunk] = []
    for local_index, tool_call in enumerate(tool_calls):
        index = tool_index_offset + local_index
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


def make_content_chunk(
    *,
    chunk_id: str,
    model: str,
    content: str,
    choice_index: int = 0,
) -> ChatCompletionChunk:
    return _make_chunk(
        chunk_id=chunk_id,
        model=model,
        delta={"content": content},
        finish_reason=None,
        choice_index=choice_index,
    )


def make_tool_argument_repair_chunk(
    *,
    chunk_id: str,
    model: str,
    tool_index: int,
    suffix: str,
    choice_index: int = 0,
) -> ChatCompletionChunk:
    """Build the delta that completes truncated streamed tool ``arguments``."""
    return _make_chunk(
        chunk_id=chunk_id,
        model=model,
        delta={
            "tool_calls": [
                {
                    "index": tool_index,
                    "function": {"arguments": suffix},
                },
            ],
        },
        finish_reason=None,
        choice_index=choice_index,
    )


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


def _parse_laguna_tool_call_blocks(block: str) -> list[ToolCall]:
    """Parse Poolside / Laguna S 2.1 tool calls.

    Format: ``func_name<arg_key>k</arg_key><arg_value>v</arg_value>``
    where each ``<arg_value>`` is either a raw string or a JSON-encoded
    non-string value (boolean, number, array, object) produced by
    Jinja's ``tojson`` filter.
    """
    if "<arg_key>" not in block:
        return []

    name_part, _, args_part = block.partition("<arg_key>")
    func_name = html.unescape(name_part).strip()
    if not func_name:
        return []

    args_str = "<arg_key>" + args_part
    params: JsonObject = {}
    cursor = 0
    for param_match in re.finditer(
        r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>",
        args_str,
        re.DOTALL,
    ):
        if args_str[cursor : param_match.start()].strip():
            return []
        key = html.unescape(param_match.group("key")).strip()
        if not key or key in params:
            return []
        raw_val = html.unescape(param_match.group("value")).strip()

        if (
            raw_val.startswith(("{", "[", "-"))
            or raw_val in ("true", "false", "null")
            or (raw_val and raw_val[0].isdigit())
        ):
            try:
                parsed_val: object = json.loads(raw_val)
            except (json.JSONDecodeError, TypeError):
                parsed_val = raw_val
        else:
            parsed_val = raw_val

        params[key] = parsed_val
        cursor = param_match.end()

    if not params or args_str[cursor:].strip():
        return []
    return [make_tool_call(func_name, params)]


def _parse_dsml_invoke_blocks(block: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    invoke_pattern = re.compile(
        re.escape(DSML_INVOKE_OPEN)
        + r"""\s+name\s*=\s*(?P<quote>["'])(?P<name>.*?)(?P=quote)\s*>""",
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
            + r"""\s+name\s*=\s*(?P<quote>["'])(?P<name>.*?)(?P=quote)"""
            + r"""(?:\s+string\s*=\s*(?P<str_quote>["'])(?P<string>true|false)"""
            + r"""(?P=str_quote))?[^>]*>"""
            + r"(?P<value>.*?)"
            + re.escape(DSML_PARAMETER_CLOSE),
            remaining[:end],
            re.DOTALL,
        ):
            value = html.unescape(param.group("value")).strip()
            if param.group("string") == "false":
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
            else:
                parsed_value = value
            params[html.unescape(param.group("name")).strip()] = parsed_value
        results.append(make_tool_call(invoke.group("name"), params))
    return results


def _parse_single_orphan_dsml_invoke(block: str) -> ToolCall | None:
    invoke_pattern = re.compile(
        r"\A"
        + re.escape(DSML_INVOKE_OPEN)
        + r"""\s+name\s*=\s*(?P<quote>["'])(?P<name>.*?)(?P=quote)\s*>"""
        + r"(?P<body>.*)"
        + re.escape(DSML_INVOKE_CLOSE)
        + r"\Z",
        re.DOTALL,
    )
    invoke = invoke_pattern.fullmatch(block)
    if invoke is None:
        return None

    name = html.unescape(invoke.group("name")).strip()
    if not name:
        return None

    parameter_pattern = re.compile(
        re.escape(DSML_PARAMETER_OPEN)
        + r"""\s+name\s*=\s*(?P<quote>["'])(?P<name>.*?)(?P=quote)"""
        + r"""(?:\s+string\s*=\s*(?P<str_quote>["'])(?P<string>true|false)"""
        + r"""(?P=str_quote))?[^>]*>"""
        + r"(?P<value>.*?)"
        + re.escape(DSML_PARAMETER_CLOSE),
        re.DOTALL,
    )
    params: JsonObject = {}
    cursor = 0
    body = invoke.group("body")
    for parameter in parameter_pattern.finditer(body):
        if body[cursor : parameter.start()].strip():
            return None
        parameter_name = html.unescape(parameter.group("name")).strip()
        if not parameter_name or parameter_name in params:
            return None
        value = html.unescape(parameter.group("value")).strip()
        if parameter.group("string") == "false":
            try:
                parsed_value: object = json.loads(value)
            except json.JSONDecodeError:
                return None
        else:
            parsed_value = value
        params[parameter_name] = parsed_value
        cursor = parameter.end()
    if body[cursor:].strip():
        return None
    return make_tool_call(name, params)


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
