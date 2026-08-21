"""HTTP proxy routes and SSE response rewriting."""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import json
import logging
import math
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from opencode_proxy.capture import StreamCapture, open_capture
from opencode_proxy.compat import (
    DEFAULT_TOOL_REPAIR_CONTEXT,
    JsonObject,
    RepairStats,
    ToolRepairContext,
    annotate_empty_completion,
    convert_chat_completion_response,
    is_empty_completion,
)
from opencode_proxy.concurrency import UpstreamConcurrencyLimiter
from opencode_proxy.metrics import ProxyMetrics
from opencode_proxy.request_compat import RequestNormalizationStats, normalize_request
from opencode_proxy.routing import (
    UpstreamTarget,
    default_upstream_target,
    resolve_upstream_target,
)
from opencode_proxy.stream_repair import (
    StreamChoiceState,
    StreamRepairConfig,
    choice_index,
    iter_finish_payloads,
    note_finish_reason,
    record_repair_stats,
    rewrite_stream_choice,
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

# Opt-in, per request: a client that sets this to ``chunk`` gets keepalives as
# empty-delta chat.completion chunks instead of SSE comments. Comments are
# discarded by SSE parsers (the OpenAI SDK drops any line starting with ``:``),
# so they keep intermediaries from dropping the connection but are invisible to
# application code. A client that watches for forward progress to extend its own
# deadline therefore sees total silence through a multi-minute prefill. Stays
# opt-in and per request because injecting synthesized frames is a payload
# mutation: callers that do not ask for it get byte-identical output.
KEEPALIVE_MODE_HEADER = "x-opencode-proxy-keepalive"
KEEPALIVE_MODE_CHUNK = "chunk"

# Upstream statuses worth one more attempt: overload and gateway failures, where
# the model most likely never ran.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# A server may ask for a longer wait than our own backoff would pick. Honour it,
# but never let one header park a caller's request for minutes.
MAX_RETRY_AFTER_SECONDS = 30.0

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
PRIMARY_MODEL_ALIAS = "primary"

# Markers that identify a specific class of upstream rejection. Matched against
# the error body because OpenAI-compatible servers overload the status codes.
QUOTA_MARKERS = ("quota", "insufficient balance", "insufficient_quota", "credit")
CONTEXT_WINDOW_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "too many tokens",
    "reduce the length",
)


@dataclass(frozen=True)
class SseFrame:
    data: str | None
    raw_lines: tuple[str, ...]


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
        alias_error = await _apply_model_alias(request, parsed_body, settings)
        if alias_error is not None:
            return alias_error
        target = resolve_upstream_target(settings, parsed_body)
        apply_target_model(parsed_body, target)
        repair_context = _tool_repair_context(parsed_body, settings)
        if settings.normalize_requests:
            thinking_transport = _thinking_transport(parsed_body, settings)
            # The message rules are DeepSeek-specific wire repairs. Keep the
            # proxy transparent for other providers, even when normalization is
            # enabled globally.
            if thinking_transport is not None:
                stats = normalize_request(
                    parsed_body,
                    thinking_transport=thinking_transport,
                )
                _record_request_normalizations(_request_metrics(request), stats)

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
    metrics = _request_metrics(request)
    try:
        deepseek_profile = (
            parsed_body is not None and _thinking_transport(parsed_body, settings) is not None
        )
        empty_retries_left = settings.empty_response_retries if deepseek_profile else 0
        while True:
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
                    metrics=metrics,
                )
            except httpx.HTTPError as exc:
                return _proxy_error(exc)

            response_headers = _forward_response_headers(upstream_response.headers)
            content_type = upstream_response.headers.get("content-type", "")
            if upstream_response.status_code >= 400:
                _note_upstream_error(upstream_response, metrics=metrics)

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

            if not isinstance(response_body, dict):
                return JSONResponse(
                    content=response_body,
                    status_code=upstream_response.status_code,
                    headers=response_headers,
                )

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
            record_repair_stats(metrics, stats, transport="buffered")
            if changed:
                LOG.info("converted raw tool call in non-streaming chat completion")
            _record_usage(converted.get("usage"), metrics)
            _note_finish_reasons(converted, metrics=metrics, transport="buffered")

            # Emptiness is judged after repair: a raw tool-call block only
            # becomes a tool call once converted.
            if is_empty_completion(converted):
                if empty_retries_left > 0:
                    empty_retries_left -= 1
                    if metrics is not None:
                        metrics.upstream_retries.labels(reason="empty_response").inc()
                    LOG.warning(
                        "upstream returned a completed turn with no output; retrying "
                        "(%d attempt(s) left)",
                        empty_retries_left,
                    )
                    continue
                if metrics is not None:
                    metrics.empty_turns.inc()
                LOG.warning("upstream returned a completed turn with no output; giving up")
                annotate_empty_completion(converted, settings.empty_turn_notice)

            return JSONResponse(
                content=converted,
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
        _note_upstream_error(upstream_response, metrics=_request_metrics(request))
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
        request,
        upstream_response,
        settings,
        repair_context,
        capture=capture,
        requested_model=str(parsed_body.get("model")) if parsed_body is not None else None,
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

        retry_after = retry_after_seconds(response.headers.get("retry-after"))
        await response.aread()
        await response.aclose()
        attempt += 1
        if metrics is not None:
            metrics.upstream_retries.labels(reason=f"http_{response.status_code}").inc()
        await _retry_backoff(
            attempt,
            reason=f"HTTP {response.status_code}",
            retry_after=retry_after,
        )


def retry_after_seconds(header_value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header into seconds, clamped to a sane bound.

    Accepts both header forms (delay seconds and HTTP-date). A malformed,
    non-finite, negative, or absurdly distant value is ignored so a hostile or broken
    upstream cannot park a caller's request.
    """
    if header_value is None:
        return None
    value = header_value.strip()
    if not value:
        return None

    try:
        delay = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delay = (when - (now or datetime.now(UTC))).total_seconds()

    if not math.isfinite(delay) or delay <= 0:
        return None
    return min(delay, MAX_RETRY_AFTER_SECONDS)


async def _retry_backoff(attempt: int, *, reason: str, retry_after: float | None = None) -> None:
    if retry_after is not None:
        LOG.warning(
            "retrying upstream request (attempt %d) after %s in %.1fs (Retry-After)",
            attempt,
            reason,
            retry_after,
        )
        await asyncio.sleep(retry_after)
        return

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
    requested_model: str | None = None,
    config: StreamRepairConfig | None = None,
) -> AsyncIterator[bytes]:
    """Rewrite an SSE stream, recording both sides when capture is enabled.

    Capture wraps rather than threads through the rewrite so that every byte
    leaving this proxy is recorded at exactly one point, including the ones
    synthesized during error and truncation handling.
    """
    if capture is None:
        async for chunk in _rewrite_sse_stream_inner(
            request,
            upstream_response,
            settings,
            repair_context,
            requested_model=requested_model,
            config=config,
        ):
            yield chunk
        return

    reason = "completed"
    try:
        async for chunk in _rewrite_sse_stream_inner(
            request,
            upstream_response,
            settings,
            repair_context,
            capture=capture,
            requested_model=requested_model,
            config=config,
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
    requested_model: str | None = None,
    config: StreamRepairConfig | None = None,
) -> AsyncIterator[bytes]:
    """Rewrite an SSE chat-completion stream into OpenAI ``tool_calls`` deltas."""
    if config is None:
        config = StreamRepairConfig.from_settings(settings)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = "unknown"
    keepalive_as_chunk = (
        request.headers.get(KEEPALIVE_MODE_HEADER, "").strip().lower() == KEEPALIVE_MODE_CHUNK
    )
    choice_states: dict[int, StreamChoiceState] = {}
    # Usage rides either on the finish chunk or on a trailing usage-only chunk,
    # and some upstreams send both. Keep the last one and count it once.
    last_usage: object = None

    # Only an explicit upstream [DONE] proves a turn completed normally.
    fallback_finish_reason = "length"

    try:
        async for frame in _iter_sse_frames_with_idle_guard(
            upstream_response,
            settings.upstream_stream_idle_timeout,
            settings.sse_keepalive_interval,
            first_frame_timeout=settings.upstream_stream_first_frame_timeout,
            metrics=_request_metrics(request),
        ):
            if await request.is_disconnected():
                LOG.info("client disconnected; stopping upstream SSE rewrite")
                if capture is not None:
                    capture.note("client disconnected")
                return

            if frame is None:
                if keepalive_as_chunk:
                    yield _encode_sse_json(_keepalive_chunk(chunk_id, requested_model or model))
                else:
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
            if isinstance(event.get("usage"), dict):
                last_usage = event["usage"]
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                yield _encode_sse_json(event)
                continue

            if not all(isinstance(choice, dict) for choice in choices):
                yield _encode_sse_json(event)
                continue

            for choice in choices:
                choice_idx = choice_index(choice)
                state = choice_states.setdefault(choice_idx, StreamChoiceState())
                for payload in rewrite_stream_choice(
                    event,
                    choice,
                    state,
                    chunk_id=chunk_id,
                    model=model,
                    config=config,
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
    finally:
        _record_usage(last_usage, _request_metrics(request))

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
    first_frame_timeout: float | None = None,
    metrics: ProxyMetrics | None = None,
) -> AsyncIterator[SseFrame | None]:
    """Yield upstream SSE frames, plus ``None`` whenever the upstream is quiet.

    A ``None`` is a keepalive tick: the caller turns it into an SSE comment so
    intermediaries do not drop an idle connection and the client keeps seeing
    signs of life during a long reasoning pause.

    Iteration also ends once the upstream has gone quiet for too long. An
    upstream that stops sending without closing the connection would otherwise
    strand the caller forever, because the read timeout is usually disabled to
    allow slow local models. Ending iteration lets the caller flush its buffers
    and terminate the client stream normally.

    The wait before the *first* frame is a different measurement from the waits
    between later frames, so they get separate budgets. Nothing arrives during
    prefill, which on a long prompt legitimately takes minutes; once tokens are
    flowing, a gap of more than a few seconds means the turn has stalled. One
    flat budget has to be loose enough for the former, which makes it far too
    loose for the latter. ``first_frame_timeout`` defaults to ``idle_timeout``.
    """
    frames = _iter_sse_frames(upstream_response).__aiter__()
    pending: asyncio.Task[SseFrame] | None = None
    silence = 0.0
    seen_frame = False
    try:
        while True:
            if pending is None:
                # Held across timeouts so a keepalive tick never cancels a
                # partially received frame.
                pending = asyncio.ensure_future(anext(frames))

            budget = idle_timeout
            if not seen_frame and first_frame_timeout is not None:
                budget = first_frame_timeout

            wait_seconds = _next_wait_interval(budget, keepalive_interval, silence)
            done, _ = await asyncio.wait({pending}, timeout=wait_seconds)
            if not done:
                silence += wait_seconds or 0.0
                if budget > 0 and silence >= budget:
                    phase = "mid_stream" if seen_frame else "first_frame"
                    LOG.warning(
                        "upstream sent no %s SSE frame for %.1fs; terminating the client stream",
                        "further" if seen_frame else "first",
                        silence,
                    )
                    if metrics is not None:
                        metrics.stream_idle_terminations.labels(phase=phase).inc()
                    return
                yield None
                continue

            finished, pending = pending, None
            try:
                frame = finished.result()
            except StopAsyncIteration:
                return
            silence = 0.0
            seen_frame = True
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
    for payload in iter_finish_payloads(
        choice_states,
        chunk_id=chunk_id,
        model=model,
        fallback_finish_reason=fallback_finish_reason,
        upstream_completed=upstream_completed,
        empty_turn_notice=empty_turn_notice,
        metrics=metrics,
    ):
        yield _encode_sse_json(payload)
    yield b"data: [DONE]\n\n"


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


def _thinking_transport(body: JsonObject, settings: Settings) -> str | None:
    """How this request's model wants thinking expressed, or ``None`` for no mapping."""
    model = body.get("model")
    if not isinstance(model, str):
        return None
    profile = settings.parsed_model_compatibility.get(model)
    if profile is None or profile.profile != "deepseek_v4":
        return None
    return profile.thinking_transport


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


def _request_metrics(request: Request) -> ProxyMetrics | None:
    metrics = getattr(request.app.state, "metrics", None)
    return metrics if isinstance(metrics, ProxyMetrics) else None


def _record_request_normalizations(
    metrics: ProxyMetrics | None,
    stats: RequestNormalizationStats,
) -> None:
    """Record bounded request-repair labels for any ingress adapter."""
    if metrics is None:
        return
    labels = stats.as_labels()
    for kind, count in labels.items():
        metrics.request_normalizations.labels(kind=kind).inc(count)


def _record_usage(usage: object, metrics: ProxyMetrics | None) -> None:
    """Count upstream-reported tokens disjointly.

    DeepSeek's ``prompt_tokens`` *includes* cache hits
    (``prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens``), so
    the cached share is subtracted out before ``input`` is counted. Summing the
    kinds then reproduces the billed prompt without double counting.
    """
    if metrics is None or not isinstance(usage, dict):
        return

    prompt_tokens = _non_negative_int(usage.get("prompt_tokens"))
    completion_tokens = _non_negative_int(usage.get("completion_tokens"))
    prompt_details = usage.get("prompt_tokens_details")
    cache_read = _non_negative_int(
        prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None,
    )
    if cache_read is None:
        cache_read = _non_negative_int(usage.get("prompt_cache_hit_tokens"))
    completion_details = usage.get("completion_tokens_details")
    reasoning = _non_negative_int(
        completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    )

    if prompt_tokens is not None:
        metrics.usage_tokens.labels(kind="input").inc(max(prompt_tokens - (cache_read or 0), 0))
    if cache_read is not None:
        metrics.usage_tokens.labels(kind="cache_read").inc(cache_read)
    if completion_tokens is not None:
        metrics.usage_tokens.labels(kind="output").inc(completion_tokens)
    if reasoning is not None:
        metrics.usage_tokens.labels(kind="reasoning").inc(reasoning)


def _non_negative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _note_finish_reasons(body: JsonObject, *, metrics: ProxyMetrics | None, transport: str) -> None:
    """Record the terminator of every choice in a buffered completion."""
    choices = body.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        note_finish_reason(
            choice.get("finish_reason"),
            chunk_id=str(body.get("id") or ""),
            model=str(body.get("model") or ""),
            choice_index=choice_index(choice),
            transport=transport,
            metrics=metrics,
        )


def classify_upstream_status(status_code: int, body_text: str) -> str:
    """Name the failure class behind an upstream error status.

    OpenAI-compatible servers overload their status codes, so the body decides
    between the cases that call for different operator action: an exhausted
    balance is not a rate limit, and an over-long prompt is not a bad request.
    """
    detail = body_text.lower()
    if status_code in {401, 403}:
        return "auth"
    if status_code == 429:
        if any(marker in detail for marker in QUOTA_MARKERS):
            return "quota"
        return "rate_limit"
    if status_code == 400:
        if any(marker in detail for marker in CONTEXT_WINDOW_MARKERS):
            return "context_window_exceeded"
        return "invalid_request"
    if 500 <= status_code < 600:
        return "server"
    if 400 <= status_code < 500:
        return "http_4xx"
    return "http_other"


def _note_upstream_error(response: httpx.Response, *, metrics: ProxyMetrics | None) -> None:
    """Classify, count, and log an upstream error without leaking its body."""
    error_type = classify_upstream_status(response.status_code, _safe_text(response.content))
    if metrics is not None:
        metrics.upstream_errors.labels(type=error_type).inc()
    request_id = response.headers.get("x-request-id") or response.headers.get(
        "x-deepseek-request-id"
    )
    LOG.warning(
        "upstream chat completion failed status=%d type=%s request_id=%s retry_after=%s",
        response.status_code,
        error_type,
        request_id or "-",
        response.headers.get("retry-after") or "-",
    )


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


async def _apply_model_alias(
    request: Request,
    body: JsonObject,
    settings: Settings,
) -> Response | None:
    model = body.get("model")
    if not isinstance(model, str):
        return None

    aliases = settings.parsed_model_aliases
    if model in aliases:
        target = aliases[model]
    elif model == PRIMARY_MODEL_ALIAS:
        discovered = await _discover_primary_model(request, settings)
        if isinstance(discovered, Response):
            return discovered
        target = discovered
    else:
        return None

    if target != model:
        LOG.info("rewriting model alias %r to upstream model %r", model, target)
        body["model"] = target
    return None


async def _discover_primary_model(request: Request, settings: Settings) -> str | Response:
    """Resolve ``primary`` to the first model advertised by the upstream."""
    try:
        response = await _upstream_client(request).get(
            _upstream_url(settings, MODELS_PATH, ""),
            headers=_forward_request_headers(request, settings=settings, stream=False),
            timeout=httpx.Timeout(settings.upstream_ready_timeout),
        )
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        LOG.warning("failed to discover the upstream primary model: %s", exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "could not discover the upstream primary model",
                    "type": "primary_model_discovery_failed",
                }
            },
        )

    model_ids = _model_ids(body)
    if not model_ids:
        LOG.warning("upstream model discovery returned no model ids")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "upstream model discovery returned no models",
                    "type": "primary_model_discovery_failed",
                }
            },
        )
    return PRIMARY_MODEL_ALIAS if PRIMARY_MODEL_ALIAS in model_ids else model_ids[0]


def _model_ids(body: object) -> list[str]:
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return []
    return [
        model_id
        for entry in body["data"]
        if isinstance(entry, dict) and isinstance((model_id := entry.get("id")), str)
    ]


def _add_model_aliases(body: JsonObject, settings: Settings) -> bool:
    aliases = dict(settings.parsed_model_aliases)
    data = body.get("data")
    if not isinstance(data, list):
        return True

    model_entries = [entry for entry in data if isinstance(entry, dict)]
    entries_by_id = {
        entry["id"]: entry for entry in model_entries if isinstance(entry.get("id"), str)
    }
    if PRIMARY_MODEL_ALIAS not in entries_by_id and PRIMARY_MODEL_ALIAS not in aliases:
        model_ids = _model_ids(body)
        if model_ids:
            aliases[PRIMARY_MODEL_ALIAS] = model_ids[0]

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
        # Proxy control header, not part of the upstream contract.
        if lower_key == KEEPALIVE_MODE_HEADER:
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
        metrics = _request_metrics(request)
        if metrics is not None:
            metrics.upstream_active.set(limiter.active)
        return None
    LOG.warning(
        "rejecting request; upstream concurrency limit reached (%d)",
        limiter.limit,
    )
    metrics = _request_metrics(request)
    if metrics is not None:
        metrics.upstream_overloads.inc()
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
        metrics = _request_metrics(request)
        if metrics is not None:
            metrics.upstream_active.set(limiter.active)


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


def _keepalive_chunk(chunk_id: str, model: str) -> JsonObject:
    """Build an empty-delta chunk used as a visible keepalive.

    Carries the same synthesized ``chunk_id`` as the rest of the rewritten
    stream, so a client that keys on the id sees one continuous turn. The empty
    ``delta`` adds no content: its only job is to be a frame the client's SSE
    parser surfaces, unlike a comment.
    """
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            },
        ],
    }


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
