"""HTTP proxy routes and SSE response rewriting."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from opencode_proxy.capture import StreamCapture, open_capture
from opencode_proxy.compat import (
    JsonObject,
    RepairStats,
    build_tool_call_chunks,
    complete_truncated_json,
    convert_chat_completion_response,
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
from opencode_proxy.concurrency import UpstreamConcurrencyLimiter
from opencode_proxy.metrics import ProxyMetrics
from opencode_proxy.routing import (
    UpstreamTarget,
    default_upstream_target,
    resolve_upstream_target,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from opencode_proxy.settings import Settings

LOG = logging.getLogger(__name__)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

DECODED_BODY_HEADERS = {
    "content-encoding",
    "content-md5",
    "digest",
    "etag",
}

SSE_KEEPALIVE_COMMENT = b": keepalive\n\n"

# Upstream statuses worth one more attempt: overload and gateway failures, where
# the model most likely never ran.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
REASONING_SCAN_FIELDS = ("reasoning", "reasoning_content")


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


@dataclass(frozen=True)
class SseFrame:
    data: str | None
    raw_lines: tuple[str, ...]


@dataclass(frozen=True)
class ToolRepairContext:
    recover_orphan_invokes: bool = False
    declared_tool_names: frozenset[str] = frozenset()


DEFAULT_TOOL_REPAIR_CONTEXT = ToolRepairContext()


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.api_route(
        CHAT_COMPLETIONS_PATH,
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def chat_completions(request: Request) -> Response:
        return await proxy_chat_completions(request, settings)

    @router.api_route(
        MODELS_PATH,
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def v1_models(request: Request) -> Response:
        return await proxy_models(request, settings)

    @router.api_route(
        "/models",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def models(request: Request) -> Response:
        return await proxy_models(request, settings)

    @router.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def catch_all(request: Request, path: str) -> Response:
        return await proxy_passthrough(request, settings, f"/{path}")

    return router


async def proxy_chat_completions(request: Request, settings: Settings) -> Response:
    body = await request.body()
    parsed_body = _parse_json_object(body)
    target = default_upstream_target(settings)
    repair_context = ToolRepairContext()
    if parsed_body is not None:
        if settings.sanitize_tools:
            _sanitize_tools(parsed_body)
        _drop_request_fields(parsed_body, settings.parsed_request_drop_fields)
        _apply_model_alias(parsed_body, settings.parsed_model_aliases)
        target = resolve_upstream_target(settings, parsed_body)
        apply_target_model(parsed_body, target)
        repair_context = _tool_repair_context(parsed_body, settings)

    stream = bool(parsed_body.get("stream")) if parsed_body is not None else False
    overload = await _acquire_upstream_slot(request)
    if overload is not None:
        return overload
    if stream:
        return await _proxy_streaming_chat_completion(
            request, settings, parsed_body, body, target, repair_context
        )
    return await _proxy_buffered_chat_completion(
        request, settings, parsed_body, body, target, repair_context
    )


async def proxy_passthrough(request: Request, settings: Settings, path: str) -> Response:
    headers = _forward_request_headers(request, settings=settings, stream=False)
    client = _upstream_client(request)
    try:
        upstream_request = client.build_request(
            request.method,
            _upstream_url(settings, path, request.url.query),
            headers=headers,
            content=request.stream(),
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return _proxy_error(exc)

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_forward_response_headers(upstream_response.headers, body_decoded=False),
        media_type=upstream_response.headers.get("content-type"),
        background=BackgroundTask(upstream_response.aclose),
    )


async def proxy_models(request: Request, settings: Settings) -> Response:
    body = await request.body()
    headers = _forward_request_headers(request, settings=settings, stream=False)
    client = _upstream_client(request)
    try:
        upstream_response = await client.request(
            request.method,
            _upstream_url(settings, MODELS_PATH, request.url.query),
            headers=headers,
            content=body,
        )
    except httpx.HTTPError as exc:
        return _proxy_error(exc)

    response_headers = _forward_response_headers(upstream_response.headers)
    content_type = upstream_response.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=content_type or None,
        )

    try:
        response_body = upstream_response.json()
    except json.JSONDecodeError:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=content_type,
        )

    if isinstance(response_body, dict):
        if not _add_model_aliases(response_body, settings):
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "message": "model alias conflicts with upstream model list",
                        "type": "alias_conflict",
                    },
                },
                headers=response_headers,
            )
        return JSONResponse(
            content=response_body,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    return JSONResponse(
        content=response_body,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def _proxy_buffered_chat_completion(
    request: Request,
    settings: Settings,
    parsed_body: JsonObject | None,
    raw_body: bytes,
    target: UpstreamTarget,
    repair_context: ToolRepairContext,
) -> Response:
    headers = _forward_request_headers(request, settings=settings, stream=False, target=target)
    client = _upstream_client(request)
    try:
        try:
            upstream_response = await send_upstream_with_retries(
                client,
                lambda: client.build_request(
                    request.method,
                    _upstream_url(
                        settings, CHAT_COMPLETIONS_PATH, request.url.query, target=target
                    ),
                    headers=headers,
                    **_body_kwargs(parsed_body, raw_body),
                ),
                settings=settings,
                stream=False,
                metrics=_request_metrics(request),
            )
        except httpx.HTTPError as exc:
            return _proxy_error(exc)

        response_headers = _forward_response_headers(upstream_response.headers)
        content_type = upstream_response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                headers=response_headers,
                media_type=content_type or None,
            )

        try:
            response_body = upstream_response.json()
        except json.JSONDecodeError:
            return Response(
                content=upstream_response.content,
                status_code=upstream_response.status_code,
                headers=response_headers,
                media_type=content_type,
            )

        if isinstance(response_body, dict):
            stats = RepairStats()
            converted, changed = convert_chat_completion_response(
                response_body,
                tool_call_scan_fields=settings.parsed_tool_call_scan_fields,
                max_raw_tool_block_chars=settings.max_raw_tool_block_chars,
                max_tool_calls=settings.max_tool_calls,
                max_tool_argument_chars=settings.max_tool_argument_chars,
                recover_orphan_invokes=repair_context.recover_orphan_invokes,
                declared_tool_names=repair_context.declared_tool_names,
                stats=stats,
            )
            _record_repair_stats(_request_metrics(request), stats, transport="buffered")
            if changed:
                LOG.info("converted raw tool call in non-streaming chat completion")
            return JSONResponse(
                content=converted,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        return JSONResponse(
            content=response_body,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )
    finally:
        await _release_upstream_slot(request)


async def _proxy_streaming_chat_completion(
    request: Request,
    settings: Settings,
    parsed_body: JsonObject | None,
    raw_body: bytes,
    target: UpstreamTarget,
    repair_context: ToolRepairContext,
) -> Response:
    headers = _forward_request_headers(request, settings=settings, stream=True, target=target)

    client = _upstream_client(request)
    try:
        upstream_response = await send_upstream_with_retries(
            client,
            lambda: client.build_request(
                request.method,
                _upstream_url(settings, CHAT_COMPLETIONS_PATH, request.url.query, target=target),
                headers=headers,
                **_body_kwargs(parsed_body, raw_body),
            ),
            settings=settings,
            stream=True,
            metrics=_request_metrics(request),
        )
    except httpx.HTTPError as exc:
        await _release_upstream_slot(request)
        return _proxy_error(exc)
    except Exception:
        await _release_upstream_slot(request)
        raise

    response_headers = _forward_response_headers(upstream_response.headers)
    if upstream_response.status_code >= 400:
        content = await upstream_response.aread()
        await upstream_response.aclose()
        await _release_upstream_slot(request)
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type"),
        )

    content_type = upstream_response.headers.get("content-type", "")
    if "text/event-stream" not in content_type.lower():
        content = await upstream_response.aread()
        await upstream_response.aclose()
        await _release_upstream_slot(request)
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=content_type or None,
        )

    capture = open_capture(
        settings.capture_stream_dir,
        max_bytes=settings.capture_stream_max_bytes,
        model=str(parsed_body.get("model")) if parsed_body is not None else None,
        upstream_url=_upstream_url(settings, CHAT_COMPLETIONS_PATH, "", target=target),
    )
    if capture is not None:
        LOG.info("capturing streamed turn to %s", capture.path)
        if settings.capture_stream_include_request:
            capture.request_body(parsed_body if parsed_body is not None else _safe_text(raw_body))

    generator = _rewrite_sse_stream(
        request, upstream_response, settings, repair_context, capture=capture
    )
    background = BackgroundTask(_aclose_upstream_and_release_slot, upstream_response, request)
    return StreamingResponse(
        generator,
        status_code=upstream_response.status_code,
        headers=apply_stream_response_headers(response_headers),
        media_type="text/event-stream",
        background=background,
    )


def apply_stream_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Mark an SSE response as unbuffered so intermediaries relay it verbatim."""
    _set_header(headers, "Cache-Control", "no-cache")
    _set_header(headers, "X-Accel-Buffering", "no")
    return headers


async def send_upstream_with_retries(
    client: httpx.AsyncClient,
    build_request: Callable[[], httpx.Request],
    *,
    settings: Settings,
    stream: bool,
    metrics: ProxyMetrics | None = None,
) -> httpx.Response:
    """Send an upstream request, retrying transport and overload failures.

    Retries only happen here, before any response bytes have reached the client,
    so a stream that has already started is never restarted. The request is
    rebuilt each attempt because httpx consumes the outgoing body.
    """
    attempt = 0
    while True:
        try:
            response = await client.send(build_request(), stream=stream)
        except httpx.HTTPError as exc:
            if attempt >= settings.upstream_max_retries or not isinstance(
                exc, httpx.TransportError
            ):
                raise
            attempt += 1
            if metrics is not None:
                metrics.upstream_retries.labels(reason="transport").inc()
            await _retry_backoff(attempt, reason=type(exc).__name__)
            continue

        if response.status_code not in RETRYABLE_STATUS or attempt >= settings.upstream_max_retries:
            return response

        await response.aread()
        await response.aclose()
        attempt += 1
        if metrics is not None:
            metrics.upstream_retries.labels(reason=f"http_{response.status_code}").inc()
        await _retry_backoff(attempt, reason=f"HTTP {response.status_code}")


async def _retry_backoff(attempt: int, *, reason: str) -> None:
    jitter = random.uniform(0, 0.25)  # noqa: S311 - retry jitter, not security
    delay = min(0.5 * (2**attempt), 8.0) + jitter
    LOG.warning("retrying upstream request (attempt %d) after %s in %.1fs", attempt, reason, delay)
    await asyncio.sleep(delay)


async def _rewrite_sse_stream(
    request: Request,
    upstream_response: httpx.Response,
    settings: Settings,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
    *,
    capture: StreamCapture | None = None,
) -> AsyncIterator[bytes]:
    """Rewrite an SSE stream, recording both sides when capture is enabled.

    Capture wraps rather than threads through the rewrite so that every byte
    leaving this proxy is recorded at exactly one point, including the ones
    synthesized during error and truncation handling.
    """
    if capture is None:
        async for chunk in _rewrite_sse_stream_inner(
            request, upstream_response, settings, repair_context
        ):
            yield chunk
        return

    reason = "completed"
    try:
        async for chunk in _rewrite_sse_stream_inner(
            request, upstream_response, settings, repair_context, capture=capture
        ):
            capture.client_bytes(chunk)
            yield chunk
    except BaseException as exc:  # recorded, then re-raised unchanged
        reason = type(exc).__name__
        raise
    finally:
        capture.close(reason=reason)


async def _rewrite_sse_stream_inner(
    request: Request,
    upstream_response: httpx.Response,
    settings: Settings,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
    *,
    capture: StreamCapture | None = None,
) -> AsyncIterator[bytes]:
    """Rewrite an SSE chat-completion stream into OpenAI ``tool_calls`` deltas."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = "unknown"
    choice_states: dict[int, StreamChoiceState] = {}

    # Only an explicit upstream [DONE] proves a turn completed normally.
    fallback_finish_reason = "length"

    try:
        async for frame in _iter_sse_frames_with_idle_guard(
            upstream_response,
            settings.upstream_stream_idle_timeout,
            settings.sse_keepalive_interval,
            metrics=_request_metrics(request),
        ):
            if await request.is_disconnected():
                LOG.info("client disconnected; stopping upstream SSE rewrite")
                if capture is not None:
                    capture.note("client disconnected")
                return

            if frame is None:
                yield SSE_KEEPALIVE_COMMENT
                continue

            if capture is not None:
                capture.upstream_frame(frame.raw_lines)

            if frame.data is None:
                yield _encode_sse_raw_frame(frame.raw_lines)
                continue

            event = _parse_sse_data(frame.data)
            if event == "[DONE]":
                async for done_payload in _finish_sse_stream(
                    choice_states,
                    chunk_id=chunk_id,
                    model=model,
                    upstream_completed=True,
                    empty_turn_notice=settings.empty_turn_notice,
                    metrics=_request_metrics(request),
                ):
                    yield done_payload
                return

            if not isinstance(event, dict):
                yield _encode_sse_raw_frame(frame.raw_lines)
                continue

            chunk_id = str(event.get("id") or chunk_id)
            model = str(event.get("model") or model)
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                yield _encode_sse_json(event)
                continue

            if not all(isinstance(choice, dict) for choice in choices):
                yield _encode_sse_json(event)
                continue

            for choice in choices:
                choice_index = _choice_index(choice)
                state = choice_states.setdefault(choice_index, StreamChoiceState())
                for payload in _rewrite_stream_choice(
                    event,
                    choice,
                    state,
                    chunk_id=chunk_id,
                    model=model,
                    settings=settings,
                    repair_context=repair_context,
                    metrics=_request_metrics(request),
                ):
                    yield _encode_sse_json(payload)
        if capture is not None:
            capture.upstream_eof()
    except asyncio.CancelledError:
        LOG.info("SSE rewrite cancelled")
        raise
    except httpx.TransportError as exc:
        if capture is not None:
            capture.note("upstream transport error", detail=type(exc).__name__)
        _, error_type = classify_upstream_error(exc)
        LOG.warning(
            "upstream SSE stream interrupted type=%s chunk_id=%s model=%s",
            error_type,
            chunk_id,
            model,
        )
        # The response is already 200/SSE, so it cannot become an HTTP error.
        # The default ``length`` finish truthfully marks the partial turn truncated.
    except Exception as exc:
        if capture is not None:
            capture.note("rewrite failed", detail=type(exc).__name__)
        # Once SSE headers are out the status is committed, so re-raising cannot
        # produce an HTTP error - it only truncates the body and strands the
        # client on an unterminated turn. Close the turn and log loudly instead.
        LOG.exception(
            "SSE rewrite failed; terminating the turn chunk_id=%s model=%s",
            chunk_id,
            model,
        )

    async for done_payload in _finish_sse_stream(
        choice_states,
        chunk_id=chunk_id,
        model=model,
        fallback_finish_reason=fallback_finish_reason,
        empty_turn_notice=settings.empty_turn_notice,
        metrics=_request_metrics(request),
    ):
        yield done_payload


async def _iter_sse_frames_with_idle_guard(
    upstream_response: httpx.Response,
    idle_timeout: float,
    keepalive_interval: float = 0.0,
    *,
    metrics: ProxyMetrics | None = None,
) -> AsyncIterator[SseFrame | None]:
    """Yield upstream SSE frames, plus ``None`` whenever the upstream is quiet.

    A ``None`` is a keepalive tick: the caller turns it into an SSE comment so
    intermediaries do not drop an idle connection and the client keeps seeing
    signs of life during a long reasoning pause.

    Iteration also ends once the upstream has been silent for ``idle_timeout``.
    An upstream that stops sending without closing the connection would
    otherwise strand the caller forever, because the read timeout is usually
    disabled to allow slow local models. Ending iteration lets the caller flush
    its buffers and terminate the client stream normally.
    """
    frames = _iter_sse_frames(upstream_response).__aiter__()
    pending: asyncio.Task[SseFrame] | None = None
    silence = 0.0
    try:
        while True:
            if pending is None:
                # Held across timeouts so a keepalive tick never cancels a
                # partially received frame.
                pending = asyncio.ensure_future(anext(frames))

            wait_seconds = _next_wait_interval(idle_timeout, keepalive_interval, silence)
            done, _ = await asyncio.wait({pending}, timeout=wait_seconds)
            if not done:
                silence += wait_seconds or 0.0
                if idle_timeout > 0 and silence >= idle_timeout:
                    LOG.warning(
                        "upstream sent no SSE frame for %.1fs; terminating the client stream",
                        silence,
                    )
                    if metrics is not None:
                        metrics.stream_idle_terminations.inc()
                    return
                yield None
                continue

            finished, pending = pending, None
            try:
                frame = finished.result()
            except StopAsyncIteration:
                return
            silence = 0.0
            yield frame
    finally:
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                await pending


def _next_wait_interval(
    idle_timeout: float,
    keepalive_interval: float,
    silence: float,
) -> float | None:
    """Seconds to wait for the next frame, or ``None`` to wait indefinitely."""
    remaining_idle = idle_timeout - silence if idle_timeout > 0 else None
    if keepalive_interval > 0 and remaining_idle is not None:
        return min(keepalive_interval, remaining_idle)
    if keepalive_interval > 0:
        return keepalive_interval
    return remaining_idle


async def _iter_sse_frames(upstream_response: httpx.Response) -> AsyncIterator[SseFrame]:
    raw_lines: list[str] = []
    data_lines: list[str] = []

    async for line in upstream_response.aiter_lines():
        if line == "":
            if raw_lines:
                yield SseFrame(
                    data="\n".join(data_lines) if data_lines else None, raw_lines=tuple(raw_lines)
                )
                raw_lines = []
                data_lines = []
            continue

        raw_lines.append(line)
        field_name, _, raw_value = line.partition(":")
        if field_name != "data":
            continue

        value = raw_value[1:] if raw_value.startswith(" ") else raw_value
        data_lines.append(value)

    if raw_lines:
        yield SseFrame(
            data="\n".join(data_lines) if data_lines else None, raw_lines=tuple(raw_lines)
        )


async def _finish_sse_stream(
    choice_states: Mapping[int, StreamChoiceState],
    *,
    chunk_id: str,
    model: str,
    fallback_finish_reason: str | None = None,
    upstream_completed: bool = False,
    empty_turn_notice: str = "",
    metrics: ProxyMetrics | None = None,
) -> AsyncIterator[bytes]:
    state_items = sorted(choice_states.items())
    if not state_items and upstream_completed and metrics is not None:
        metrics.empty_turns.inc()
    if not state_items and (
        fallback_finish_reason is not None or (upstream_completed and empty_turn_notice)
    ):
        state_items = [(0, StreamChoiceState())]

    for choice_index, state in state_items:
        if not state.finish_sent:
            for payload in _flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
            ):
                yield _encode_sse_json(payload)
            for payload in _repair_tool_call_arguments(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                metrics=metrics,
            ):
                yield _encode_sse_json(payload)
            for payload in _annotate_empty_turn(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                notice=empty_turn_notice,
                upstream_completed=upstream_completed,
                metrics=metrics,
            ):
                yield _encode_sse_json(payload)
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
                choice_index,
            )
            state.tool_call_arguments.clear()
            state.unrepairable_tool_call_indexes.clear()
            state.invalid_tool_call_index_seen = False
        if not state.finish_sent:
            # A stream that ends without a finish_reason leaves agent clients
            # waiting on a turn the model already completed; always close the
            # choice out.
            yield _encode_sse_json(
                make_finish_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    finish_reason=_finish_reason_for_state(fallback_finish_reason, state),
                    choice_index=choice_index,
                ),
            )
            state.finish_sent = True
    yield b"data: [DONE]\n\n"


def _rewrite_stream_choice(
    event: JsonObject,
    choice: JsonObject,
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    settings: Settings,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
    metrics: ProxyMetrics | None = None,
) -> list[JsonObject]:
    delta = choice.get("delta")
    finish_reason = choice.get("finish_reason")
    choice_index = _choice_index(choice)
    outputs: list[JsonObject] = []

    if state.finish_sent:
        LOG.warning(
            "ignoring SSE choice data after terminal chunk; chunk_id=%s model=%s choice=%d",
            chunk_id,
            model,
            choice_index,
        )
        return outputs

    if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
        state.emitted_content = True

    if not isinstance(delta, dict):
        outputs.extend(
            _flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
            ),
        )
        passthrough_choice = dict(choice)
        if finish_reason is not None:
            passthrough_choice["finish_reason"] = _finish_reason_for_state(
                finish_reason,
                state,
            )
            outputs.extend(
                _repair_tool_call_arguments(
                    state,
                    chunk_id=chunk_id,
                    model=model,
                    choice_index=choice_index,
                    metrics=metrics,
                ),
            )
            outputs.extend(
                _annotate_empty_turn(
                    state,
                    chunk_id=chunk_id,
                    model=model,
                    choice_index=choice_index,
                    notice=settings.empty_turn_notice,
                    upstream_completed=True,
                    metrics=metrics,
                ),
            )
            state.finish_sent = True
        outputs.append({**event, "choices": [passthrough_choice]})
        return outputs

    if delta.get("tool_calls"):
        outputs.extend(
            _flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
            ),
        )
        normalized_tool_calls, synthesized = _normalize_stream_tool_call_ids(
            delta["tool_calls"],
            state,
        )
        if synthesized and metrics is not None:
            metrics.synthesized_tool_call_ids.labels(transport="streaming").inc(synthesized)
        if normalized_tool_calls is not delta["tool_calls"]:
            delta = {**delta, "tool_calls": normalized_tool_calls}
            choice = {**choice, "delta": delta}
        state.emitted_tool_calls = True
        _record_tool_call_arguments(
            state,
            delta["tool_calls"],
            max_tool_calls=settings.max_tool_calls,
            max_argument_chars=settings.max_tool_argument_chars,
        )
        if finish_reason is None:
            outputs.append(_single_choice_event(event, choice))
            return outputs

        # The same event carries the last argument fragment and the finish, so
        # it has to be split: fragments, then the repair, then the terminator.
        open_choice = {**choice, "finish_reason": None}
        outputs.append({**event, "choices": [open_choice]})
        outputs.extend(
            _repair_tool_call_arguments(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                metrics=metrics,
            ),
        )
        outputs.extend(
            _annotate_empty_turn(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                notice=settings.empty_turn_notice,
                upstream_completed=True,
                metrics=metrics,
            ),
        )
        outputs.append(
            _finish_payload(
                event,
                state,
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
            ),
        )
        state.finish_sent = True
        return outputs

    scan_fields = settings.parsed_tool_call_scan_fields
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
        flushed = _flush_choice_buffers(
            state,
            chunk_id=chunk_id,
            model=model,
            choice_index=choice_index,
            only_fields=tuple(
                name for name in REASONING_SCAN_FIELDS if name not in state.pending_raw_fields
            ),
        )
        if flushed:
            outputs.extend(flushed)
            emitted_any_delta = True

    if other_delta or _has_stream_metadata(choice):
        if other_delta.get("content"):
            _flush_reasoning_before_content()
        outputs.append(_choice_delta_event(event, choice, other_delta, finish_reason=None))
        emitted_any_delta = True

    for field_name in _ordered_scan_fields(scanned_text, scan_fields):
        if field_name == "content":
            _flush_reasoning_before_content()
        outputs.extend(
            _process_stream_field_text(
                state,
                event=event,
                field_name=field_name,
                text=scanned_text[field_name],
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                settings=settings,
                repair_context=repair_context,
                metrics=metrics,
            ),
        )
        emitted_any_delta = True

    if finish_reason is not None:
        outputs.extend(
            _flush_choice_buffers(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
            ),
        )
        outputs.extend(
            _repair_tool_call_arguments(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                metrics=metrics,
            ),
        )
        outputs.extend(
            _annotate_empty_turn(
                state,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                notice=settings.empty_turn_notice,
                upstream_completed=True,
                metrics=metrics,
            ),
        )
        outputs.append(
            _finish_payload(
                event,
                state,
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
            ),
        )
        state.finish_sent = True
        emitted_any_delta = True

    if not emitted_any_delta and (not delta or delta.get("tool_calls") == []):
        outputs.append(_choice_delta_event(event, choice, {}, finish_reason=None))

    return outputs


def _finish_payload(
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
            finish_reason=_finish_reason_for_state(finish_reason, state),
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


def _annotate_empty_turn(
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    choice_index: int,
    notice: str,
    upstream_completed: bool,
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


def _record_tool_call_arguments(
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


def _repair_tool_call_arguments(
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


def _process_stream_field_text(
    state: StreamChoiceState,
    *,
    event: JsonObject,
    field_name: str,
    text: str,
    chunk_id: str,
    model: str,
    choice_index: int,
    settings: Settings,
    repair_context: ToolRepairContext,
    metrics: ProxyMetrics | None,
) -> list[JsonObject]:
    outputs: list[JsonObject] = []
    state.field_event_metadata[field_name] = {
        key: value
        for key, value in event.items()
        if key not in {"choices", "id", "object", "model"}
    }
    previous_buffer = state.field_buffers.get(field_name, "")
    state.field_buffers[field_name] = previous_buffer + text

    if field_name in state.pending_raw_fields and "</" not in state.field_buffers[field_name]:
        if len(state.field_buffers[field_name]) > settings.max_raw_tool_block_chars:
            LOG.warning(
                "incomplete raw tool-call block exceeded max size; passing through as text",
            )
            outputs.append(
                _field_chunk(
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
            if field_name == "content" and repair_context.recover_orphan_invokes
            else None
        )
        span = _earliest_span(wrapped_span, orphan_span)
        if span is not None:
            state.pending_raw_fields.discard(field_name)
            start, end = span
            prefix = buffer[:start]
            block = buffer[start:end]
            suffix = buffer[end:]
            is_orphan = orphan_span == span and (
                wrapped_span is None or orphan_span[0] <= wrapped_span[0]
            )

            if len(block) > settings.max_raw_tool_block_chars:
                if is_orphan and metrics is not None:
                    metrics.orphan_recovery.labels(
                        outcome="rejected", reason="oversized_block"
                    ).inc()
                LOG.warning(
                    "raw tool-call block exceeded max size; passing through as text",
                )
                outputs.append(
                    _field_chunk(
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
                    declared_tool_names=repair_context.declared_tool_names,
                    max_raw_tool_block_chars=settings.max_raw_tool_block_chars,
                    max_tool_calls=settings.max_tool_calls,
                    max_tool_argument_chars=settings.max_tool_argument_chars,
                    stats=orphan_stats,
                )
                _record_repair_stats(metrics, orphan_stats, transport="streaming")
                if not changed:
                    tool_calls = []
                    block = rejected_text
            else:
                tool_calls = parse_raw_tool_calls(block)
            if not tool_calls:
                LOG.info("raw tool-call block could not be parsed; passing through as text")
                outputs.append(
                    _field_chunk(
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
                max_tool_calls=settings.max_tool_calls,
                max_tool_argument_chars=settings.max_tool_argument_chars,
            ):
                LOG.warning(
                    "raw tool-call block exceeded tool-call limits; passing through as text"
                )
                outputs.append(
                    _field_chunk(
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
                    _field_chunk(
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
                repair_format = "deepseek_v4_orphan" if is_orphan else _raw_format(block)
                metrics.raw_tool_repair.labels(format=repair_format, field=field_name).inc()
            for tool_chunk in build_tool_call_chunks(
                tool_calls,
                chunk_id=chunk_id,
                model=model,
                argument_chunk_size=settings.tool_argument_chunk_size,
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
            if field_name == "content" and repair_context.recover_orphan_invokes
            else None
        )
        raw_start = _earliest_index(wrapped_start, orphan_start)
        if raw_start is not None:
            if len(buffer) - raw_start > settings.max_raw_tool_block_chars:
                LOG.warning(
                    "incomplete raw tool-call block exceeded max size; passing through as text",
                )
                outputs.append(
                    _field_chunk(
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
                    _field_chunk(
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

        flush_size = len(buffer) - settings.stream_guard_chars
        if flush_size > 0:
            outputs.append(
                _field_chunk(
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


def _flush_choice_buffers(
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
                _field_chunk(
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


def _field_chunk(
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


def _parse_json_object(body: bytes) -> JsonObject | None:
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _tool_repair_context(body: JsonObject, settings: Settings) -> ToolRepairContext:
    model = body.get("model")
    profile = settings.parsed_model_compatibility.get(model) if isinstance(model, str) else None
    tools = body.get("tools")
    declared_names: frozenset[str] = frozenset()
    if isinstance(tools, list):
        names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                name = function["name"]
                if name:
                    names.add(name)
        declared_names = frozenset(names)
    return ToolRepairContext(
        recover_orphan_invokes=bool(
            profile
            and profile.profile == "deepseek_v4"
            and profile.recover_orphan_invokes
            and declared_names
            and body.get("tool_choice") != "none"
        ),
        declared_tool_names=declared_names,
    )


def _normalize_stream_tool_call_ids(
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


def _request_metrics(request: Request) -> ProxyMetrics | None:
    metrics = getattr(request.app.state, "metrics", None)
    return metrics if isinstance(metrics, ProxyMetrics) else None


def _record_repair_stats(
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


def _earliest_span(
    first: tuple[int, int] | None,
    second: tuple[int, int] | None,
) -> tuple[int, int] | None:
    spans = [span for span in (first, second) if span is not None]
    return min(spans, key=lambda span: (span[0], span[1])) if spans else None


def _earliest_index(first: int | None, second: int | None) -> int | None:
    indexes = [index for index in (first, second) if index is not None]
    return min(indexes) if indexes else None


def _raw_format(text: str) -> str:
    if "<tool_call" in text:
        return "qwen_xml"
    if "DSML" in text:
        return "dsml"
    return "unknown"


def _body_kwargs(parsed_body: JsonObject | None, raw_body: bytes) -> dict[str, Any]:
    if parsed_body is not None:
        return {"json": parsed_body}
    return {"content": raw_body}


def apply_target_model(body: JsonObject, target: UpstreamTarget) -> None:
    """Swap in the routed model so the alternate upstream sees a name it serves."""
    if target.model and body.get("model") != target.model:
        LOG.info(
            "rewriting model %r to %r for the %s route",
            body.get("model"),
            target.model,
            target.modality,
        )
        body["model"] = target.model


def _apply_model_alias(body: JsonObject, aliases: Mapping[str, str]) -> None:
    model = body.get("model")
    if isinstance(model, str) and model in aliases:
        target = aliases[model]
        LOG.info("rewriting model alias %r to upstream model %r", model, target)
        body["model"] = target


def _add_model_aliases(body: JsonObject, settings: Settings) -> bool:
    aliases = settings.parsed_model_aliases
    data = body.get("data")
    if not isinstance(data, list) or not aliases:
        return True

    model_entries = [entry for entry in data if isinstance(entry, dict)]
    entries_by_id = {
        entry["id"]: entry for entry in model_entries if isinstance(entry.get("id"), str)
    }

    for alias, target in aliases.items():
        if alias in entries_by_id:
            if alias != target:
                LOG.warning(
                    "model alias %r conflicts with an upstream model id for target %r",
                    alias,
                    target,
                )
                if settings.alias_conflict_policy == "error":
                    return False
                if settings.alias_conflict_policy == "skip":
                    continue
                data[:] = [
                    entry
                    for entry in data
                    if not (isinstance(entry, dict) and entry.get("id") == alias)
                ]
                entries_by_id.pop(alias, None)
            else:
                continue

        target_entry = entries_by_id.get(target)
        if target_entry is not None:
            alias_entry = dict(target_entry)
            alias_entry["id"] = alias
        else:
            alias_entry = {"id": alias, "object": "model", "owned_by": "opencode-proxy"}

        data.append(alias_entry)
        entries_by_id[alias] = alias_entry
    return True


def _drop_request_fields(body: JsonObject, field_names: tuple[str, ...]) -> None:
    for field_name in field_names:
        if field_name in body:
            LOG.info("dropping request field %r before forwarding upstream", field_name)
            body.pop(field_name, None)


def _sanitize_tools(body: JsonObject) -> None:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return

    function_tools = [
        tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"
    ]
    if function_tools:
        if len(function_tools) != len(tools):
            LOG.info(
                "dropping %d non-function tool(s) before forwarding upstream",
                len(tools) - len(function_tools),
            )
        body["tools"] = function_tools
    else:
        LOG.info("dropping tools field because it contains no function tools")
        body.pop("tools", None)


def _choice_index(choice: JsonObject) -> int:
    index = choice.get("index")
    return index if type(index) is int else 0


def _ordered_scan_fields(
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


def _finish_reason_for_state(finish_reason: object, state: StreamChoiceState) -> str:
    if state.raw_tool_calls_emitted:
        return "tool_calls"
    if isinstance(finish_reason, str):
        return finish_reason
    if state.emitted_tool_calls:
        return "tool_calls"
    return "stop"


def _single_choice_event(event: JsonObject, choice: JsonObject) -> JsonObject:
    return {**event, "choices": [choice]}


def _choice_delta_event(
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


def _has_stream_metadata(choice: JsonObject) -> bool:
    # Null-valued extras carry no information; vLLM sends logprobs and stop_reason
    # on every choice, which would otherwise force an empty delta event per chunk.
    choice_envelope = {"index", "delta", "finish_reason"}
    return any(value is not None for key, value in choice.items() if key not in choice_envelope)


def _forward_request_headers(
    request: Request,
    *,
    settings: Settings,
    stream: bool,
    target: UpstreamTarget | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lower_key = key.lower()
        if lower_key in HOP_BY_HOP_HEADERS or lower_key == "host":
            continue
        if stream and lower_key == "accept-encoding":
            continue
        headers[key] = value

    if stream:
        headers["accept"] = "text/event-stream"
        headers["accept-encoding"] = "identity"

    client_host = request.client.host if request.client else None
    if client_host and "x-forwarded-for" not in {key.lower() for key in headers}:
        headers["x-forwarded-for"] = client_host

    if settings.upstream_api_key and not any(key.lower() == "authorization" for key in headers):
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    for key, value in settings.parsed_custom_headers.items():
        lower_key = key.lower()
        if lower_key in HOP_BY_HOP_HEADERS:
            continue
        if stream and lower_key == "accept-encoding":
            continue
        _set_header(headers, key, value)

    if target is not None and target.modality:
        # A routed request goes to a different host, so the caller credential for
        # the primary upstream must not follow it there.
        if target.api_key:
            _set_header(headers, "Authorization", f"Bearer {target.api_key}")
        for key, value in target.extra_headers:
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            _set_header(headers, key, value)

    return headers


def _set_header(headers: dict[str, str], key: str, value: str) -> None:
    for existing_key in list(headers):
        if existing_key.lower() == key.lower():
            headers.pop(existing_key)
    headers[key] = value


def _forward_response_headers(
    headers: httpx.Headers,
    *,
    body_decoded: bool = True,
) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    excluded_headers = HOP_BY_HOP_HEADERS | (DECODED_BODY_HEADERS if body_decoded else set())
    for key, value in headers.items():
        if key.lower() in excluded_headers:
            continue
        forwarded[key] = value
    return forwarded


def _upstream_url(
    settings: Settings,
    path: str,
    query: str,
    *,
    target: UpstreamTarget | None = None,
) -> str:
    base_url = target.base_url if target is not None else settings.upstream_base_url
    normalized_path = "/" + quote(path.lstrip("/"), safe="/:")
    url = f"{base_url}{normalized_path}"
    if query:
        return f"{url}?{query}"
    return url


def _upstream_timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=None
        if settings.upstream_connect_timeout == 0
        else settings.upstream_connect_timeout,
        read=None if settings.upstream_read_timeout == 0 else settings.upstream_read_timeout,
        write=None if settings.upstream_write_timeout == 0 else settings.upstream_write_timeout,
        pool=None if settings.upstream_pool_timeout == 0 else settings.upstream_pool_timeout,
    )


def create_upstream_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_upstream_timeout(settings))


def _upstream_client(request: Request) -> httpx.AsyncClient:
    client = request.app.state.upstream_client
    if not isinstance(client, httpx.AsyncClient):
        msg = "upstream HTTP client is not initialized"
        raise RuntimeError(msg)
    return client


def _upstream_limiter(request: Request) -> UpstreamConcurrencyLimiter | None:
    limiter = getattr(request.app.state, "upstream_limiter", None)
    return limiter if isinstance(limiter, UpstreamConcurrencyLimiter) else None


async def _acquire_upstream_slot(request: Request) -> JSONResponse | None:
    limiter = _upstream_limiter(request)
    if limiter is None:
        return None
    if await limiter.try_acquire():
        return None
    LOG.warning(
        "rejecting request; upstream concurrency limit reached (%d)",
        limiter.limit,
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "upstream concurrency limit reached",
                "type": "proxy_overload",
            },
        },
        headers={"Retry-After": "1"},
    )


async def _release_upstream_slot(request: Request) -> None:
    limiter = _upstream_limiter(request)
    if limiter is not None:
        await limiter.release()


async def _aclose_upstream_and_release_slot(
    upstream_response: httpx.Response,
    request: Request,
) -> None:
    try:
        await upstream_response.aclose()
    finally:
        await _release_upstream_slot(request)


def _parse_sse_data(payload: str) -> JsonObject | str:
    payload = payload.strip()
    if payload == "[DONE]":
        return "[DONE]"

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload

    return parsed if isinstance(parsed, dict) else payload


def _encode_sse_raw_frame(raw_lines: tuple[str, ...]) -> bytes:
    return ("\n".join(raw_lines) + "\n\n").encode()


def _encode_sse_json(payload: Mapping[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def classify_upstream_error(exc: BaseException) -> tuple[str, str]:
    """Map an upstream transport failure to a safe ``(message, type)`` pair.

    The returned strings never include hostnames or raw exception text so they
    are safe to return to clients while still distinguishing common failure modes.
    """
    if isinstance(exc, httpx.ConnectTimeout):
        return "upstream connect timeout", "connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "upstream read timeout", "read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "upstream write timeout", "write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "upstream pool timeout", "pool_timeout"
    if isinstance(exc, httpx.TimeoutException):
        return "upstream request timeout", "timeout"
    if isinstance(exc, httpx.ConnectError):
        detail = str(exc).lower()
        if "refused" in detail:
            return "upstream connection refused", "connection_refused"
        if _looks_like_dns_failure(detail):
            return "upstream DNS resolution failed", "dns_error"
        return "upstream connection failed", "connect_error"
    if isinstance(exc, httpx.ProxyError):
        return "upstream proxy error", "proxy_error"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "upstream protocol error", "protocol_error"
    if isinstance(exc, httpx.NetworkError):
        return "upstream network error", "network_error"
    return "upstream request failed", "proxy_error"


def _looks_like_dns_failure(detail: str) -> bool:
    markers = (
        "name or service not known",
        "nodename nor servname",
        "getaddrinfo failed",
        "name resolution",
        "temporary failure in name resolution",
        "nodename not known",
    )
    return any(marker in detail for marker in markers)


def _proxy_error(exc: httpx.HTTPError) -> JSONResponse:
    message, error_type = classify_upstream_error(exc)
    LOG.warning("upstream request failed type=%s: %s", error_type, exc)
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": message,
                "type": error_type,
            },
        },
    )
