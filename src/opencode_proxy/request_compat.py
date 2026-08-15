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

# DeepSeek's current API accepts these levels. "off" is expressed as
# ``thinking: {"type": "disabled"}`` instead and never crosses as an effort.
WIRE_REASONING_EFFORTS = frozenset({"low", "high", "max"})
REASONING_EFFORT_ALIASES = {"medium": "high", "xhigh": "high"}

# How an upstream wants thinking expressed. The DeepSeek API reads a top-level
# `thinking` object; vLLM ignores that and reads a chat-template argument
# instead, so the same profile has to be able to target either. These live here
# rather than in settings.py because settings imports config_file, which needs
# them too.
THINKING_TRANSPORTS = frozenset({"api", "chat_template_kwargs"})
DEFAULT_THINKING_TRANSPORT = "api"
DISABLED_REASONING_EFFORTS = frozenset({"off", "none", "disabled"})


@dataclass
class RequestNormalizationStats:
    """Counts of request repairs, for logging and metrics."""

    null_assistant_content: int = 0
    dropped_reasoning: int = 0
    moved_reasoning: int = 0
    empty_tool_results: int = 0
    developer_roles: int = 0
    max_completion_tokens: int = 0
    reasoning_effort_aliases: int = 0
    thinking_disabled: int = 0
    thinking_enabled: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.null_assistant_content
            or self.dropped_reasoning
            or self.moved_reasoning
            or self.empty_tool_results
            or self.developer_roles
            or self.max_completion_tokens
            or self.reasoning_effort_aliases
            or self.thinking_disabled
            or self.thinking_enabled
        )

    def as_labels(self) -> dict[str, int]:
        """Non-zero counts keyed by metric label."""
        counts = {
            "null_assistant_content": self.null_assistant_content,
            "dropped_reasoning": self.dropped_reasoning,
            "moved_reasoning": self.moved_reasoning,
            "empty_tool_results": self.empty_tool_results,
            "developer_roles": self.developer_roles,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort_aliases": self.reasoning_effort_aliases,
            "thinking_disabled": self.thinking_disabled,
            "thinking_enabled": self.thinking_enabled,
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
        if role == "developer":
            # DeepSeek's current API follows the older OpenAI role set and
            # rejects ``developer``. Preserve the instruction as a system turn.
            message["role"] = "system"
            stats.developer_roles += 1
        elif role == "assistant":
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


def normalize_reasoning_effort(
    body: JsonObject,
    stats: RequestNormalizationStats,
    *,
    transport: str,
) -> None:
    """Express thinking the way *this* upstream reads it.

    ``reasoning_effort`` accepts ``low``, ``high``, and ``max`` on the current
    DeepSeek wire; common client aliases ``medium`` and ``xhigh`` map to
    ``high``.
    "off" is a different field entirely. Which field depends on who is serving:

    * ``api`` -- the DeepSeek API's top-level ``thinking`` object. An enabled
      toggle may accompany an accepted effort level; a disabled toggle cannot.
    * ``chat_template_kwargs`` -- vLLM, which ignores the top-level field and
      reads ``chat_template_kwargs.thinking`` instead. That argument is a
      boolean, so it cannot carry an effort *level*; the effort field is
      translated and dropped rather than forwarded to be silently ignored.

    DeepSeek-profile models only: neither field is an OpenAI one.
    """
    raw_thinking = body.get("thinking")
    explicit_enabled = _thinking_enabled(raw_thinking)
    effort = body.get("reasoning_effort")

    # Some clients use the vendor field directly while others use the
    # OpenAI-compatible effort field. vLLM's boolean template argument cannot
    # carry a level, so it consumes either form. The vendor API explicitly
    # supports an enabled thinking toggle together with an effort level.
    if explicit_enabled is not None:
        if transport == "chat_template_kwargs":
            body.pop("reasoning_effort", None)
            _set_chat_template_thinking(body, enabled=explicit_enabled)
            body.pop("thinking", None)
            if explicit_enabled:
                stats.thinking_enabled += 1
            else:
                stats.thinking_disabled += 1
            return

        canonical_thinking = {
            "type": "enabled" if explicit_enabled else "disabled",
        }
        if raw_thinking != canonical_thinking:
            body["thinking"] = canonical_thinking
            if explicit_enabled:
                stats.thinking_enabled += 1
            else:
                stats.thinking_disabled += 1

        if not explicit_enabled:
            if "reasoning_effort" in body:
                LOG.warning(
                    "request disables thinking but also sets reasoning_effort; using thinking"
                )
                body.pop("reasoning_effort", None)
            return

        if effort is None:
            return

    if explicit_enabled is None and "thinking" in body and effort is None:
        # Preserve an unknown vendor value for the upstream to reject with its
        # normal validation message. We only rewrite values we understand.
        LOG.warning(
            "unsupported thinking value type=%s; forwarding unchanged",
            type(raw_thinking).__name__,
        )

    if not isinstance(effort, str):
        return

    normalized = effort.strip().lower()
    canonical_effort = REASONING_EFFORT_ALIASES.get(normalized, normalized)
    normalized = canonical_effort
    disabled = normalized in DISABLED_REASONING_EFFORTS
    if not disabled and normalized not in WIRE_REASONING_EFFORTS:
        LOG.warning(
            "unsupported reasoning_effort %r; DeepSeek accepts only %s",
            effort,
            ", ".join(sorted(WIRE_REASONING_EFFORTS)),
        )
        if transport != "chat_template_kwargs":
            return
    elif canonical_effort != effort:
        body["reasoning_effort"] = canonical_effort
        stats.reasoning_effort_aliases += 1

    if transport == "chat_template_kwargs":
        _set_chat_template_thinking(body, enabled=not disabled)
        body.pop("reasoning_effort", None)
        if disabled:
            stats.thinking_disabled += 1
        else:
            stats.thinking_enabled += 1
        return

    if disabled:
        if explicit_enabled:
            LOG.warning("request enables thinking but disables reasoning effort; using thinking")
            body.pop("reasoning_effort", None)
            return
        body.pop("reasoning_effort", None)
        body["thinking"] = {"type": "disabled"}
        stats.thinking_disabled += 1


def _set_chat_template_thinking(body: JsonObject, *, enabled: bool) -> None:
    """Set the template argument without disturbing the caller's other kwargs."""
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    body["chat_template_kwargs"] = {**kwargs, "thinking": enabled}


def _thinking_enabled(value: object) -> bool | None:
    """Return the boolean represented by a supported thinking value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        thinking_type = value.get("type")
        if isinstance(thinking_type, str):
            normalized = thinking_type.strip().lower()
            if normalized in {"enabled", "on", "true"}:
                return True
            if normalized in {"disabled", "off", "false"}:
                return False
    return None


def normalize_request(
    body: JsonObject,
    *,
    thinking_transport: str | None,
) -> RequestNormalizationStats:
    """Apply every request repair that applies to this body.

    The message hygiene and thinking mapping are DeepSeek-specific. HTTP routes
    do not call this helper for models without a ``deepseek_v4`` profile.
    ``thinking_transport=None`` only suppresses the thinking-field mapping for
    direct helper callers.
    """
    stats = RequestNormalizationStats()
    normalize_messages(body, stats)
    _normalize_completion_limit(body, stats)
    if thinking_transport is not None:
        normalize_reasoning_effort(body, stats, transport=thinking_transport)
    if stats.changed:
        LOG.info("normalized outgoing chat request: %s", stats.as_labels())
    return stats


def _normalize_completion_limit(body: JsonObject, stats: RequestNormalizationStats) -> None:
    """Use DeepSeek's legacy ``max_tokens`` field for the newer alias."""
    if "max_completion_tokens" not in body:
        return
    value = body.pop("max_completion_tokens")
    if "max_tokens" not in body and value is not None:
        body["max_tokens"] = value
    stats.max_completion_tokens += 1
