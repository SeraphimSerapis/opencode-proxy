"""Request-side normalization for OpenAI-compatible chat completion bodies.

The rules here follow DeepSeek's own reference client (``deepseek-harness``,
``packages/llm/llm-deepseek``), which documents several message shapes the
DeepSeek API rejects outright. Because a rejected shape usually lives durably in
the caller's session log, one bad message keeps failing every later turn of that
session, so the cheapest place to repair it is on the way upstream.

Everything here is pure: no FastAPI, no network, no settings object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencode_proxy.compat import JsonObject

LOG = logging.getLogger(__name__)

# DeepSeek rejects a tool message with empty content, so an empty tool result
# needs some placeholder text. This is the literal the reference client sends.
EMPTY_TOOL_RESULT_PLACEHOLDER = "(no output)"

REASONING_FIELDS = ("reasoning_content", "reasoning")

# The only two efforts DeepSeek accepts on the wire. "off" is expressed as
# ``thinking: {"type": "disabled"}`` instead and never crosses as an effort.
WIRE_REASONING_EFFORTS = frozenset({"high", "max"})
DISABLED_REASONING_EFFORTS = frozenset({"off", "none", "disabled"})


@dataclass
class RequestNormalizationStats:
    """Counts of request repairs, for logging and metrics."""

    null_assistant_content: int = 0
    dropped_reasoning: int = 0
    moved_reasoning: int = 0
    empty_tool_results: int = 0
    thinking_disabled: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.null_assistant_content
            or self.dropped_reasoning
            or self.moved_reasoning
            or self.empty_tool_results
            or self.thinking_disabled
        )

    def as_labels(self) -> dict[str, int]:
        """Non-zero counts keyed by metric label."""
        counts = {
            "null_assistant_content": self.null_assistant_content,
            "dropped_reasoning": self.dropped_reasoning,
            "moved_reasoning": self.moved_reasoning,
            "empty_tool_results": self.empty_tool_results,
            "thinking_disabled": self.thinking_disabled,
        }
        return {label: count for label, count in counts.items() if count}


def normalize_messages(body: JsonObject, stats: RequestNormalizationStats) -> None:
    """Repair message shapes that a DeepSeek-compatible upstream rejects.

    Three rules, all from the reference client:

    * An assistant message must carry string ``content``, never ``null``. A
      reasoning-only turn (V4 Flash answers short prompts entirely in the
      reasoning channel) otherwise serializes as ``content: null`` with no tool
      calls, which the API rejects with "content or tool_calls must be set".
    * ``reasoning_content`` must be replayed on assistant turns that carried
      tool calls, and is ignored on every other turn, so it is dropped there to
      save the tokens.
    * A tool message needs non-empty content.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            _normalize_assistant_message(message, stats)
        elif role == "tool":
            _normalize_tool_message(message, stats)


def _normalize_assistant_message(message: JsonObject, stats: RequestNormalizationStats) -> None:
    content = message.get("content")
    if content is None:
        message["content"] = ""
        stats.null_assistant_content += 1

    tool_calls = message.get("tool_calls")
    has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
    if not has_tool_calls:
        for name in REASONING_FIELDS:
            if name in message:
                message.pop(name)
                stats.dropped_reasoning += 1
        return

    # Tool-call turn: the API requires the reasoning back, under its own field
    # name. Callers that only kept ``reasoning`` still satisfy that if it moves.
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and not message.get("reasoning_content"):
        message["reasoning_content"] = reasoning
        stats.moved_reasoning += 1
    if "reasoning" in message:
        message.pop("reasoning")


def _normalize_tool_message(message: JsonObject, stats: RequestNormalizationStats) -> None:
    content = message.get("content")
    if content is None or content == "" or content == []:
        message["content"] = EMPTY_TOOL_RESULT_PLACEHOLDER
        stats.empty_tool_results += 1


def normalize_reasoning_effort(body: JsonObject, stats: RequestNormalizationStats) -> None:
    """Express "no thinking" the way the DeepSeek wire format does.

    ``reasoning_effort`` accepts only ``high`` and ``max``; disabling thinking is
    ``thinking: {"type": "disabled"}`` with no effort field at all. Enabled is the
    provider default, so an accepted effort is forwarded on its own.

    DeepSeek-profile models only: ``thinking`` is not an OpenAI field.
    """
    effort = body.get("reasoning_effort")
    if not isinstance(effort, str):
        return

    normalized = effort.strip().lower()
    if normalized in DISABLED_REASONING_EFFORTS:
        body.pop("reasoning_effort", None)
        body["thinking"] = {"type": "disabled"}
        stats.thinking_disabled += 1
        return
    if normalized not in WIRE_REASONING_EFFORTS:
        LOG.warning(
            "forwarding unsupported reasoning_effort %r; DeepSeek accepts only %s",
            effort,
            ", ".join(sorted(WIRE_REASONING_EFFORTS)),
        )


def normalize_request(body: JsonObject, *, deepseek_profile: bool) -> RequestNormalizationStats:
    """Apply every request repair that applies to this body.

    Message hygiene is safe for any OpenAI-compatible upstream; the thinking
    mapping is DeepSeek-specific and runs only for a model configured with the
    ``deepseek_v4`` compatibility profile.
    """
    stats = RequestNormalizationStats()
    normalize_messages(body, stats)
    if deepseek_profile:
        normalize_reasoning_effort(body, stats)
    if stats.changed:
        LOG.info("normalized outgoing chat request: %s", stats.as_labels())
    return stats
