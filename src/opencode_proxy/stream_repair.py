"""Streaming tool-call repair as a pure, dict-in/dict-out state machine.

Consumes ``chat.completion.chunk`` events shaped as plain JSON objects and
returns plain JSON objects. The module deliberately imports neither FastAPI nor
httpx nor ``Settings``, so the FastAPI SSE frontend and the LiteLLM callback
plugin drive exactly the same repair logic and both can be tested without a
network.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from opencode_proxy.compat import (
    DEFAULT_TOOL_REPAIR_CONTEXT,
    DSML_OPEN,
    JsonObject,
    RepairStats,
    ToolRepairContext,
    build_tool_call_chunks,
    complete_truncated_json,
    extract_orphan_dsml_invokes,
    find_complete_orphan_dsml_invoke_span,
    find_complete_raw_tool_block_span,
    find_orphan_dsml_invoke_start,
    find_raw_tool_start,
    make_content_chunk,
    make_finish_chunk,
    make_tool_argument_repair_chunk,
    parse_raw_tool_calls,
    strip_empty_tool_calls,
    tool_calls_within_limits,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from opencode_proxy.metrics import ProxyMetrics
    from opencode_proxy.settings import Settings

LOG = logging.getLogger(__name__)

REASONING_SCAN_FIELDS = ("reasoning", "reasoning_content")

# Terminators an OpenAI-compatible client knows how to handle. Anything else
# (content_filter, vLLM's insufficient_system_resource, future additions) ended
# the turn abnormally and is worth surfacing rather than passing off as a stop.
KNOWN_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "function_call"})

# Emitted instead of the token-budget notice when the upstream itself reported
# an abnormal terminator, so the annotation never misattributes the cause.
ABNORMAL_FINISH_NOTICE = (
    "[proxy: the upstream ended this turn with finish_reason={reason} and produced "
    "no output or tool call.]"
)


@dataclass(frozen=True)
class StreamRepairConfig:
    """Knobs the streaming repair reads, decoupled from ``Settings``."""

    scan_fields: tuple[str, ...] = ("content", "reasoning", "reasoning_content")
    stream_guard_chars: int = 192
    tool_argument_chunk_size: int = 64
    max_raw_tool_block_chars: int = 131_072
    max_tool_calls: int = 32
    max_tool_argument_chars: int = 262_144
    empty_turn_notice: str = ""

    @classmethod
    def from_settings(cls, settings: Settings) -> StreamRepairConfig:
        return cls(
            scan_fields=settings.parsed_tool_call_scan_fields,
            stream_guard_chars=settings.stream_guard_chars,
            tool_argument_chunk_size=settings.tool_argument_chunk_size,
            max_raw_tool_block_chars=settings.max_raw_tool_block_chars,
            max_tool_calls=settings.max_tool_calls,
            max_tool_argument_chars=settings.max_tool_argument_chars,
            empty_turn_notice=settings.empty_turn_notice,
        )


@dataclass
class StreamChoiceState:
    field_buffers: dict[str, str] = field(default_factory=dict)
    field_event_metadata: dict[str, JsonObject] = field(default_factory=dict)
    pending_raw_fields: set[str] = field(default_factory=set)
    raw_tool_calls_emitted: bool = False
    finish_sent: bool = False
    next_tool_call_index: int = 0
    # Streamed ``arguments`` fragments per upstream tool-call index, kept so a
    # truncated JSON payload can be completed before the turn closes.
    tool_call_arguments: dict[int, str] = field(default_factory=dict)
    unrepairable_tool_call_indexes: set[int] = field(default_factory=set)
    invalid_tool_call_index_seen: bool = False
    native_tool_call_ids: dict[int, str] = field(default_factory=dict)
    emitted_content: bool = False
    emitted_tool_calls: bool = False


def choice_index(choice: JsonObject) -> int:
    index = choice.get("index")
    return index if type(index) is int else 0


def ordered_scan_fields(
    scanned_text: Mapping[str, str],
    scan_fields: tuple[str, ...],
) -> list[str]:
    reasoning_first = [
        field_name
        for field_name in scan_fields
        if field_name != "content" and field_name in scanned_text
    ]
    if "content" in scanned_text:
        reasoning_first.append("content")
    return reasoning_first


def finish_reason_for_state(finish_reason: object, state: StreamChoiceState) -> str:
    if state.raw_tool_calls_emitted:
        return "tool_calls"
    if isinstance(finish_reason, str):
        return finish_reason
    if state.emitted_tool_calls:
        return "tool_calls"
    return "stop"


def single_choice_event(event: JsonObject, choice: JsonObject) -> JsonObject:
    return {**event, "choices": [choice]}


def choice_delta_event(
    event: JsonObject,
    choice: JsonObject,
    delta: JsonObject,
    *,
    finish_reason: str | None,
) -> JsonObject:
    return {
        **event,
        "choices": [
            {
                **choice,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ],
    }


def has_stream_metadata(choice: JsonObject) -> bool:
    # Null-valued extras carry no information; vLLM sends logprobs and stop_reason
    # on every choice, which would otherwise force an empty delta event per chunk.
    choice_envelope = {"index", "delta", "finish_reason"}
    return any(value is not None for key, value in choice.items() if key not in choice_envelope)


def raw_format(text: str) -> str:
    if DSML_OPEN in text or "<|DSML|tool_calls" in text or "<DSML" in text:
        return "dsml"
    if "<tool_call" in text:
        return "qwen_xml"
    return "unknown"


def earliest_span(
    first: tuple[int, int] | None,
    second: tuple[int, int] | None,
) -> tuple[int, int] | None:
    spans = [span for span in (first, second) if span is not None]
    if not spans:
        return None
    return min(spans, key=lambda span: (span[0], span[1]))


def earliest_index(first: int | None, second: int | None) -> int | None:
    candidates = [index for index in (first, second) if index is not None]
    if not candidates:
        return None
    return min(candidates)


def finish_reason_label(reason: object) -> str:
    """Bounded metric label for an upstream-controlled ``finish_reason``."""
    if isinstance(reason, str) and reason in KNOWN_FINISH_REASONS:
        return reason
    if reason is None:
        return "absent"
    return "other"


def note_finish_reason(
    reason: object,
    *,
    chunk_id: str,
    model: str,
    choice_index: int,
    transport: str,
    metrics: ProxyMetrics | None,
) -> None:
    if metrics is not None:
        metrics.finish_reasons.labels(
            reason=finish_reason_label(reason),
            transport=transport,
        ).inc()
    if isinstance(reason, str) and reason not in KNOWN_FINISH_REASONS:
        LOG.warning(
            "upstream closed a turn with an abnormal finish_reason=%r; "
            "chunk_id=%s model=%s choice=%d",
            reason,
            chunk_id,
            model,
            choice_index,
        )


def record_repair_stats(
    metrics: ProxyMetrics | None,
    stats: RepairStats,
    *,
    transport: str,
) -> None:
    if metrics is None:
        return
    for repair_format, field_name in stats.raw_repairs:
        metrics.raw_tool_repair.labels(format=repair_format, field=field_name).inc()
    if stats.orphan_accepted:
        metrics.orphan_recovery.labels(outcome="accepted", reason="valid").inc(
            stats.orphan_accepted
        )
    for reason, count in stats.orphan_rejected.items():
        metrics.orphan_recovery.labels(outcome="rejected", reason=reason).inc(count)
    if stats.synthesized_ids:
        metrics.synthesized_tool_call_ids.labels(transport=transport).inc(stats.synthesized_ids)


def normalize_stream_tool_call_ids(
    tool_calls: object,
    state: StreamChoiceState,
) -> tuple[object, int]:
    if not isinstance(tool_calls, list):
        return tool_calls, 0

    normalized: list[object] = []
    changed = False
    synthesized = 0
    for raw_tool_call in tool_calls:
        if not isinstance(raw_tool_call, dict):
            normalized.append(raw_tool_call)
            continue
        index = raw_tool_call.get("index")
        function = raw_tool_call.get("function")
        if type(index) is not int or index < 0 or not isinstance(function, dict):
            normalized.append(raw_tool_call)
            continue

        existing_id = raw_tool_call.get("id")
        if isinstance(existing_id, str) and existing_id:
            state.native_tool_call_ids[index] = existing_id
            normalized.append(raw_tool_call)
            continue

        call_id = state.native_tool_call_ids.get(index)
        if call_id is None:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            state.native_tool_call_ids[index] = call_id
            synthesized += 1
        normalized.append({**raw_tool_call, "id": call_id})
        changed = True
    return (normalized if changed else tool_calls), synthesized


def rewrite_stream_choice(
    event: JsonObject,
    choice: JsonObject,
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    config: StreamRepairConfig,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
    metrics: ProxyMetrics | None = None,
) -> list[JsonObject]:
    delta = choice.get("delta")
    finish_reason = choice.get("finish_reason")
    index = choice_index(choice)
    outputs: list[JsonObject] = []

    if state.finish_sent:
        LOG.warning(
            "ignoring SSE choice data after terminal chunk; chunk_id=%s model=%s choice=%d",
            chunk_id,
            model,
            index,
        )
        return outputs

    if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
        state.emitted_content = True

    if not isinstance(delta, dict):
        outputs.extend(
            flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
            ),
        )
        passthrough_choice = dict(choice)
        if finish_reason is not None:
            passthrough_choice["finish_reason"] = finish_reason_for_state(finish_reason, state)
            note_finish_reason(
                finish_reason,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
                transport="streaming",
                metrics=metrics,
            )
            outputs.extend(
                repair_tool_call_arguments(
                    state,
                    chunk_id=chunk_id,
                    model=model,
                    choice_index=index,
                    metrics=metrics,
                ),
            )
            outputs.extend(
                annotate_empty_turn(
                    state,
                    chunk_id=chunk_id,
                    model=model,
                    choice_index=index,
                    notice=config.empty_turn_notice,
                    upstream_completed=True,
                    finish_reason=finish_reason,
                    metrics=metrics,
                ),
            )
            state.finish_sent = True
        outputs.append({**event, "choices": [passthrough_choice]})
        return outputs

    if delta.get("tool_calls"):
        outputs.extend(
            flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
            ),
        )
        normalized_tool_calls, synthesized = normalize_stream_tool_call_ids(
            delta["tool_calls"],
            state,
        )
        if synthesized and metrics is not None:
            metrics.synthesized_tool_call_ids.labels(transport="streaming").inc(synthesized)
        if normalized_tool_calls is not delta["tool_calls"]:
            delta = {**delta, "tool_calls": normalized_tool_calls}
            choice = {**choice, "delta": delta}
        state.emitted_tool_calls = True
        record_tool_call_arguments(
            state,
            delta["tool_calls"],
            max_tool_calls=config.max_tool_calls,
            max_argument_chars=config.max_tool_argument_chars,
        )
        if finish_reason is None:
            outputs.append(single_choice_event(event, choice))
            return outputs

        # The same event carries the last argument fragment and the finish, so
        # it has to be split: fragments, then the repair, then the terminator.
        open_choice = {**choice, "finish_reason": None}
        outputs.append({**event, "choices": [open_choice]})
        note_finish_reason(
            finish_reason,
            chunk_id=chunk_id,
            model=model,
            choice_index=index,
            transport="streaming",
            metrics=metrics,
        )
        outputs.extend(
            repair_tool_call_arguments(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
                metrics=metrics,
            ),
        )
        outputs.extend(
            annotate_empty_turn(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
                notice=config.empty_turn_notice,
                upstream_completed=True,
                finish_reason=finish_reason,
                metrics=metrics,
            ),
        )
        outputs.append(
            finish_payload(
                event,
                state,
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
            ),
        )
        state.finish_sent = True
        return outputs

    scan_fields = config.scan_fields
    scan_field_set = set(scan_fields)
    scanned_text = {
        key: value
        for key, value in delta.items()
        if key in scan_field_set and isinstance(value, str) and value
    }
    other_delta = strip_empty_tool_calls(
        {key: value for key, value in delta.items() if key not in scan_field_set}
    )

    emitted_any_delta = False

    def _flush_reasoning_before_content() -> None:
        """Release held reasoning tails so thinking cannot trail the answer.

        Fields mid raw tool-call block keep their buffer: flushing there would
        leak a half-parsed block as visible text.
        """
        nonlocal emitted_any_delta
        flushed = flush_choice_buffers(
            state,
            chunk_id=chunk_id,
            model=model,
            choice_index=index,
            only_fields=tuple(
                name for name in REASONING_SCAN_FIELDS if name not in state.pending_raw_fields
            ),
        )
        if flushed:
            outputs.extend(flushed)
            emitted_any_delta = True

    if other_delta or has_stream_metadata(choice):
        if other_delta.get("content"):
            _flush_reasoning_before_content()
        outputs.append(choice_delta_event(event, choice, other_delta, finish_reason=None))
        emitted_any_delta = True

    for field_name in ordered_scan_fields(scanned_text, scan_fields):
        if field_name == "content":
            _flush_reasoning_before_content()
        outputs.extend(
            process_stream_field_text(
                state,
                event=event,
                field_name=field_name,
                text=scanned_text[field_name],
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
                config=config,
                repair_context=repair_context,
                metrics=metrics,
            ),
        )
        emitted_any_delta = True

    if finish_reason is not None:
        note_finish_reason(
            finish_reason,
            chunk_id=chunk_id,
            model=model,
            choice_index=index,
            transport="streaming",
            metrics=metrics,
        )
        outputs.extend(
            flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
            ),
        )
        outputs.extend(
            repair_tool_call_arguments(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
                metrics=metrics,
            ),
        )
        outputs.extend(
            annotate_empty_turn(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
                notice=config.empty_turn_notice,
                upstream_completed=True,
                finish_reason=finish_reason,
                metrics=metrics,
            ),
        )
        outputs.append(
            finish_payload(
                event,
                state,
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model=model,
                choice_index=index,
            ),
        )
        state.finish_sent = True
        emitted_any_delta = True

    if not emitted_any_delta and (not delta or delta.get("tool_calls") == []):
        outputs.append(choice_delta_event(event, choice, {}, finish_reason=None))

    return outputs


def finish_payload(
    event: JsonObject,
    state: StreamChoiceState,
    *,
    finish_reason: object,
    chunk_id: str,
    model: str,
    choice_index: int,
) -> JsonObject:
    """Build a terminal chunk, carrying over any extra upstream event fields."""
    payload = cast(
        "JsonObject",
        make_finish_chunk(
            chunk_id=chunk_id,
            model=model,
            finish_reason=finish_reason_for_state(finish_reason, state),
            choice_index=choice_index,
        ),
    )
    payload.update(
        {
            key: value
            for key, value in event.items()
            if key not in {"choices", "id", "object", "model"}
        }
    )
    return payload


def annotate_empty_turn(
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    choice_index: int,
    notice: str,
    upstream_completed: bool,
    finish_reason: object = None,
    metrics: ProxyMetrics | None = None,
) -> list[JsonObject]:
    """Give the client something to act on when a turn produced nothing.

    A reasoning model can spend its whole ``max_tokens`` budget thinking and
    close the turn with no content and no tool calls. The stream is well formed,
    but the agent driving it has nothing to render and nothing to execute, which
    is indistinguishable from a hang. Always log it; optionally emit a short
    proxy annotation so the turn is visibly a dead end rather than a silent one.
    """
    if state.emitted_content or state.emitted_tool_calls or state.raw_tool_calls_emitted:
        return []

    if metrics is not None:
        metrics.empty_turns.inc()
    LOG.warning(
        "upstream turn produced no content and no tool calls; "
        "chunk_id=%s model=%s choice=%d upstream_finished=%s",
        chunk_id,
        model,
        choice_index,
        state.finish_sent or upstream_completed,
    )
    # Only explain the turn when the upstream closed it itself. If the proxy is
    # synthesising the terminator after a failure or an idle timeout, the turn
    # is empty because the stream broke, and blaming the token budget would be
    # wrong; the synthetic finish_reason already says it was truncated.
    if not notice or not (state.finish_sent or upstream_completed):
        return []
    # An abnormal terminator (content_filter, insufficient_system_resource) is
    # the upstream stopping the turn itself, so blaming the token budget would
    # misattribute it. Say what actually happened instead.
    if isinstance(finish_reason, str) and finish_reason not in KNOWN_FINISH_REASONS:
        notice = ABNORMAL_FINISH_NOTICE.format(reason=finish_reason)
    state.emitted_content = True
    return [
        cast(
            "JsonObject",
            make_content_chunk(
                chunk_id=chunk_id,
                model=model,
                content=notice,
                choice_index=choice_index,
            ),
        ),
    ]


def record_tool_call_arguments(
    state: StreamChoiceState,
    tool_calls: object,
    *,
    max_tool_calls: int,
    max_argument_chars: int,
) -> None:
    """Accumulate streamed ``arguments`` fragments per tool-call index."""
    if not isinstance(tool_calls, list):
        return
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        index = tool_call.get("index")
        if type(index) is not int or index < 0:
            # An invalid index makes it unsafe to attach a repair to a
            # particular call, especially when several calls are interleaved.
            state.invalid_tool_call_index_seen = True
            continue
        if index in state.unrepairable_tool_call_indexes:
            continue

        current = state.tool_call_arguments.get(index)
        if current is None:
            if len(state.tool_call_arguments) + len(state.unrepairable_tool_call_indexes) >= (
                max_tool_calls
            ):
                continue
            current = ""

        if len(current) + len(arguments) > max_argument_chars:
            state.tool_call_arguments.pop(index, None)
            state.unrepairable_tool_call_indexes.add(index)
            continue
        state.tool_call_arguments[index] = current + arguments


def repair_tool_call_arguments(
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    choice_index: int,
    metrics: ProxyMetrics | None = None,
) -> list[JsonObject]:
    """Complete any tool ``arguments`` the upstream truncated mid-value.

    An upstream can close a turn with ``finish_reason: "tool_calls"`` while the
    streamed JSON is cut off, handing the client a call it cannot execute and
    that fails validation if the turn is ever replayed. Earlier deltas are
    already on the wire, so the repair is appended as one final fragment.
    """
    repairs: list[JsonObject] = []
    if state.invalid_tool_call_index_seen:
        if metrics is not None:
            metrics.tool_argument_repair.labels(outcome="invalid_index").inc()
        LOG.error(
            "upstream tool call has an invalid index; skipping argument repair; "
            "chunk_id=%s model=%s",
            chunk_id,
            model,
        )
    for tool_index in sorted(state.unrepairable_tool_call_indexes):
        if metrics is not None:
            metrics.tool_argument_repair.labels(outcome="limit_exceeded").inc()
        LOG.error(
            "upstream tool call arguments exceeded repair limit; "
            "chunk_id=%s model=%s tool_index=%d",
            chunk_id,
            model,
            tool_index,
        )
    for tool_index, arguments in sorted(state.tool_call_arguments.items()):
        suffix = complete_truncated_json(arguments)
        if suffix is None:
            if metrics is not None:
                metrics.tool_argument_repair.labels(outcome="unrepairable").inc()
            LOG.error(
                "upstream tool call has unrepairable arguments; "
                "chunk_id=%s model=%s tool_index=%d arguments=%r",
                chunk_id,
                model,
                tool_index,
                arguments[:200],
            )
            continue
        if not suffix:
            continue
        if metrics is not None:
            metrics.tool_argument_repair.labels(outcome="completed").inc()
        LOG.warning(
            "completing truncated tool-call arguments; "
            "chunk_id=%s model=%s tool_index=%d suffix=%r",
            chunk_id,
            model,
            tool_index,
            suffix,
        )
        repairs.append(
            cast(
                "JsonObject",
                make_tool_argument_repair_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    tool_index=tool_index,
                    suffix=suffix,
                    choice_index=choice_index,
                ),
            ),
        )
    state.tool_call_arguments.clear()
    state.unrepairable_tool_call_indexes.clear()
    state.invalid_tool_call_index_seen = False
    return repairs


def process_stream_field_text(
    state: StreamChoiceState,
    *,
    event: JsonObject,
    field_name: str,
    text: str,
    chunk_id: str,
    model: str,
    choice_index: int,
    config: StreamRepairConfig,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
    metrics: ProxyMetrics | None = None,
) -> list[JsonObject]:
    recover_orphan_invokes = repair_context.recover_orphan_invokes
    declared_tool_names = repair_context.declared_tool_names

    outputs: list[JsonObject] = []
    state.field_event_metadata[field_name] = {
        key: value
        for key, value in event.items()
        if key not in {"choices", "id", "object", "model"}
    }
    previous_buffer = state.field_buffers.get(field_name, "")
    state.field_buffers[field_name] = previous_buffer + text

    if field_name in state.pending_raw_fields and "</" not in state.field_buffers[field_name]:
        if len(state.field_buffers[field_name]) > config.max_raw_tool_block_chars:
            LOG.warning(
                "incomplete raw tool-call block exceeded max size; passing through as text",
            )
            outputs.append(
                field_chunk(
                    chunk_id,
                    model,
                    field_name,
                    state.field_buffers[field_name],
                    choice_index,
                    event_metadata=state.field_event_metadata.get(field_name),
                )
            )
            state.field_buffers[field_name] = ""
            state.pending_raw_fields.discard(field_name)
        return outputs

    while state.field_buffers[field_name]:
        buffer = state.field_buffers[field_name]
        wrapped_span = find_complete_raw_tool_block_span(buffer)
        orphan_span = (
            find_complete_orphan_dsml_invoke_span(buffer)
            if field_name == "content" and recover_orphan_invokes
            else None
        )
        span = earliest_span(wrapped_span, orphan_span)
        if span is not None:
            state.pending_raw_fields.discard(field_name)
            start, end = span
            prefix = buffer[:start]
            block = buffer[start:end]
            suffix = buffer[end:]
            is_orphan = orphan_span == span and (
                wrapped_span is None or orphan_span[0] <= wrapped_span[0]
            )

            if len(block) > config.max_raw_tool_block_chars:
                if is_orphan and metrics is not None:
                    metrics.orphan_recovery.labels(
                        outcome="rejected", reason="oversized_block"
                    ).inc()
                LOG.warning(
                    "raw tool-call block exceeded max size; passing through as text",
                )
                outputs.append(
                    field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        prefix + block,
                        choice_index,
                        event_metadata=state.field_event_metadata.get(field_name),
                    )
                )
                state.field_buffers[field_name] = suffix
                continue

            orphan_stats = RepairStats()
            if is_orphan:
                tool_calls, rejected_text, changed = extract_orphan_dsml_invokes(
                    block,
                    declared_tool_names=declared_tool_names,
                    max_raw_tool_block_chars=config.max_raw_tool_block_chars,
                    max_tool_calls=config.max_tool_calls,
                    max_tool_argument_chars=config.max_tool_argument_chars,
                    stats=orphan_stats,
                )
                record_repair_stats(metrics, orphan_stats, transport="streaming")
                if not changed:
                    tool_calls = []
                    block = rejected_text
            else:
                tool_calls = parse_raw_tool_calls(block)
            if not tool_calls:
                LOG.info("raw tool-call block could not be parsed; passing through as text")
                outputs.append(
                    field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        prefix + block,
                        choice_index,
                        event_metadata=state.field_event_metadata.get(field_name),
                    )
                )
                state.field_buffers[field_name] = suffix
                continue

            if not tool_calls_within_limits(
                tool_calls,
                max_tool_calls=config.max_tool_calls,
                max_tool_argument_chars=config.max_tool_argument_chars,
            ):
                LOG.warning(
                    "raw tool-call block exceeded tool-call limits; passing through as text"
                )
                outputs.append(
                    field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        prefix + block,
                        choice_index,
                        event_metadata=state.field_event_metadata.get(field_name),
                    )
                )
                state.field_buffers[field_name] = suffix
                continue

            if prefix:
                outputs.append(
                    field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        prefix,
                        choice_index,
                        event_metadata=state.field_event_metadata.get(field_name),
                    ),
                )
            LOG.info(
                "converted %d raw tool call(s) in streaming chat completion",
                len(tool_calls),
            )
            if metrics is not None:
                repair_format = "deepseek_v4_orphan" if is_orphan else raw_format(block)
                metrics.raw_tool_repair.labels(format=repair_format, field=field_name).inc()
            for tool_chunk in build_tool_call_chunks(
                tool_calls,
                chunk_id=chunk_id,
                model=model,
                argument_chunk_size=config.tool_argument_chunk_size,
                choice_index=choice_index,
                tool_index_offset=state.next_tool_call_index,
                include_finish=False,
            ):
                outputs.append(
                    {
                        **state.field_event_metadata.get(field_name, {}),
                        **cast("JsonObject", tool_chunk),
                    }
                )
            state.next_tool_call_index += len(tool_calls)
            state.raw_tool_calls_emitted = True
            state.field_buffers[field_name] = suffix
            continue

        wrapped_start = find_raw_tool_start(buffer)
        orphan_start = (
            find_orphan_dsml_invoke_start(buffer)
            if field_name == "content" and recover_orphan_invokes
            else None
        )
        raw_start = earliest_index(wrapped_start, orphan_start)
        if raw_start is not None:
            if len(buffer) - raw_start > config.max_raw_tool_block_chars:
                LOG.warning(
                    "incomplete raw tool-call block exceeded max size; passing through as text",
                )
                outputs.append(
                    field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        buffer,
                        choice_index,
                        event_metadata=state.field_event_metadata.get(field_name),
                    )
                )
                state.field_buffers[field_name] = ""
                state.pending_raw_fields.discard(field_name)
                break
            if raw_start > 0:
                outputs.append(
                    field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        buffer[:raw_start],
                        choice_index,
                        event_metadata=state.field_event_metadata.get(field_name),
                    ),
                )
                state.field_buffers[field_name] = buffer[raw_start:]
            state.pending_raw_fields.add(field_name)
            break

        flush_size = len(buffer) - config.stream_guard_chars
        if flush_size > 0:
            outputs.append(
                field_chunk(
                    chunk_id,
                    model,
                    field_name,
                    buffer[:flush_size],
                    choice_index,
                    event_metadata=state.field_event_metadata.get(field_name),
                ),
            )
            state.field_buffers[field_name] = buffer[flush_size:]
        break

    return outputs


def flush_choice_buffers(
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    choice_index: int,
    only_fields: tuple[str, ...] | None = None,
) -> list[JsonObject]:
    outputs: list[JsonObject] = []
    if only_fields is None:
        preferred = [*REASONING_SCAN_FIELDS, "content"]
        field_names = [name for name in preferred if name in state.field_buffers]
        field_names.extend(name for name in state.field_buffers if name not in field_names)
    else:
        field_names = [name for name in only_fields if name in state.field_buffers]

    for field_name in field_names:
        buffered_text = state.field_buffers.get(field_name) or ""
        if buffered_text:
            outputs.append(
                field_chunk(
                    chunk_id,
                    model,
                    field_name,
                    buffered_text,
                    choice_index,
                    event_metadata=state.field_event_metadata.get(field_name),
                )
            )
            state.field_buffers[field_name] = ""
            state.pending_raw_fields.discard(field_name)
    return outputs


def field_chunk(
    chunk_id: str,
    model: str,
    field_name: str,
    text: str,
    choice_index: int,
    *,
    event_metadata: Mapping[str, Any] | None = None,
) -> JsonObject:
    return {
        **(event_metadata or {}),
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": choice_index,
                "delta": {field_name: text},
                "finish_reason": None,
            },
        ],
    }


def iter_finish_payloads(
    choice_states: Mapping[int, StreamChoiceState],
    *,
    chunk_id: str,
    model: str,
    fallback_finish_reason: str | None = None,
    upstream_completed: bool = False,
    empty_turn_notice: str = "",
    metrics: ProxyMetrics | None = None,
) -> Iterator[JsonObject]:
    """Close every open choice: flush buffers, repair, annotate, terminate.

    Pure counterpart of the SSE-level finish step; yields plain chunk payloads
    (the ``data: [DONE]`` sentinel is the caller's job).
    """
    state_items = sorted(choice_states.items())
    if not state_items and upstream_completed and metrics is not None:
        metrics.empty_turns.inc()
    if not state_items and (
        fallback_finish_reason is not None or (upstream_completed and empty_turn_notice)
    ):
        state_items = [(0, StreamChoiceState())]

    for choice_idx, state in state_items:
        if not state.finish_sent:
            yield from flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_idx,
            )
            yield from repair_tool_call_arguments(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_idx,
                metrics=metrics,
            )
            yield from annotate_empty_turn(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_idx,
                notice=empty_turn_notice,
                upstream_completed=upstream_completed,
                metrics=metrics,
            )
        elif (
            state.tool_call_arguments
            or state.unrepairable_tool_call_indexes
            or state.invalid_tool_call_index_seen
        ):
            # A terminal chunk has already reached the client. There is no safe
            # place to append a repair after it, so discard only this impossible
            # leftover state rather than emitting an out-of-order delta.
            LOG.error(
                "tool-call repair state remained after terminal chunk; "
                "chunk_id=%s model=%s choice=%d",
                chunk_id,
                model,
                choice_idx,
            )
            state.tool_call_arguments.clear()
            state.unrepairable_tool_call_indexes.clear()
            state.invalid_tool_call_index_seen = False
        if not state.finish_sent:
            # A stream that ends without a finish_reason leaves agent clients
            # waiting on a turn the model already completed; always close the
            # choice out.
            synthesized_reason = finish_reason_for_state(fallback_finish_reason, state)
            note_finish_reason(
                synthesized_reason,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_idx,
                transport="synthesized",
                metrics=metrics,
            )
            yield cast(
                "JsonObject",
                make_finish_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    finish_reason=synthesized_reason,
                    choice_index=choice_idx,
                ),
            )
            state.finish_sent = True
