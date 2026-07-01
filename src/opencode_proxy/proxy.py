"""HTTP proxy routes and SSE response rewriting."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from opencode_proxy.compat import (
    JsonObject,
    build_tool_call_chunks,
    convert_chat_completion_response,
    find_complete_raw_tool_block_span,
    find_raw_tool_start,
    make_finish_chunk,
    parse_raw_tool_calls,
    strip_empty_tool_calls,
    tool_calls_within_limits,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

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

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"


@dataclass
class StreamChoiceState:
    field_buffers: dict[str, str] = field(default_factory=dict)
    raw_tool_calls_emitted: bool = False
    finish_sent: bool = False


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
    if parsed_body is not None:
        if settings.sanitize_tools:
            _sanitize_tools(parsed_body)
        _drop_request_fields(parsed_body, settings.parsed_request_drop_fields)
        _apply_model_alias(parsed_body, settings.parsed_model_aliases)

    stream = bool(parsed_body.get("stream")) if parsed_body is not None else False
    if stream:
        return await _proxy_streaming_chat_completion(request, settings, parsed_body, body)
    return await _proxy_buffered_chat_completion(request, settings, parsed_body, body)


async def proxy_passthrough(request: Request, settings: Settings, path: str) -> Response:
    body = await request.body()
    headers = _forward_request_headers(request, settings=settings, stream=False)
    try:
        async with httpx.AsyncClient(timeout=_upstream_timeout(settings)) as client:
            upstream_response = await client.request(
                request.method,
                _upstream_url(settings, path, request.url.query),
                headers=headers,
                content=body,
            )
    except httpx.HTTPError as exc:
        return _proxy_error(exc)

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_forward_response_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


async def proxy_models(request: Request, settings: Settings) -> Response:
    body = await request.body()
    headers = _forward_request_headers(request, settings=settings, stream=False)
    try:
        async with httpx.AsyncClient(timeout=_upstream_timeout(settings)) as client:
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
) -> Response:
    headers = _forward_request_headers(request, settings=settings, stream=False)
    try:
        async with httpx.AsyncClient(timeout=_upstream_timeout(settings)) as client:
            upstream_response = await client.request(
                request.method,
                _upstream_url(settings, CHAT_COMPLETIONS_PATH, request.url.query),
                headers=headers,
                **_body_kwargs(parsed_body, raw_body),
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
        converted, changed = convert_chat_completion_response(
            response_body,
            tool_call_scan_fields=settings.parsed_tool_call_scan_fields,
            max_raw_tool_block_chars=settings.max_raw_tool_block_chars,
            max_tool_calls=settings.max_tool_calls,
            max_tool_argument_chars=settings.max_tool_argument_chars,
        )
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


async def _proxy_streaming_chat_completion(
    request: Request,
    settings: Settings,
    parsed_body: JsonObject | None,
    raw_body: bytes,
) -> Response:
    headers = _forward_request_headers(request, settings=settings, stream=True)

    client = httpx.AsyncClient(timeout=_upstream_timeout(settings))
    try:
        upstream_request = client.build_request(
            request.method,
            _upstream_url(settings, CHAT_COMPLETIONS_PATH, request.url.query),
            headers=headers,
            **_body_kwargs(parsed_body, raw_body),
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        return _proxy_error(exc)

    response_headers = _forward_response_headers(upstream_response.headers)
    if upstream_response.status_code >= 400:
        content = await upstream_response.aread()
        await upstream_response.aclose()
        await client.aclose()
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
        await client.aclose()
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=content_type or None,
        )

    generator = _rewrite_sse_stream(request, upstream_response, settings)
    background = BackgroundTask(_close_streaming_resources, upstream_response, client)
    return StreamingResponse(
        generator,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type="text/event-stream",
        background=background,
    )


async def _rewrite_sse_stream(
    request: Request,
    upstream_response: httpx.Response,
    settings: Settings,
) -> AsyncIterator[bytes]:
    """Rewrite an SSE chat-completion stream into OpenAI ``tool_calls`` deltas."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = "unknown"
    choice_states: dict[int, StreamChoiceState] = {}

    try:
        async for frame in _iter_sse_frames(upstream_response):
            if await request.is_disconnected():
                LOG.info("client disconnected; stopping upstream SSE rewrite")
                return

            if frame.data is None:
                yield _encode_sse_raw_frame(frame.raw_lines)
                continue

            event = _parse_sse_data(frame.data)
            if event == "[DONE]":
                async for done_payload in _finish_sse_stream(
                    choice_states,
                    chunk_id=chunk_id,
                    model=model,
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
                ):
                    yield _encode_sse_json(payload)
    except asyncio.CancelledError:
        LOG.info("SSE rewrite cancelled")
        raise

    async for done_payload in _finish_sse_stream(
        choice_states,
        chunk_id=chunk_id,
        model=model,
    ):
        yield done_payload


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
) -> AsyncIterator[bytes]:
    for choice_index, state in sorted(choice_states.items()):
        for payload in _flush_choice_buffers(
            state,
            chunk_id=chunk_id,
            model=model,
            choice_index=choice_index,
        ):
            yield _encode_sse_json(payload)
        if state.raw_tool_calls_emitted and not state.finish_sent:
            yield _encode_sse_json(
                make_finish_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    finish_reason="tool_calls",
                    choice_index=choice_index,
                ),
            )
            state.finish_sent = True
    yield b"data: [DONE]\n\n"


async def _close_streaming_resources(
    upstream_response: httpx.Response,
    client: httpx.AsyncClient,
) -> None:
    await upstream_response.aclose()
    await client.aclose()


def _rewrite_stream_choice(
    event: JsonObject,
    choice: JsonObject,
    state: StreamChoiceState,
    *,
    chunk_id: str,
    model: str,
    settings: Settings,
) -> list[JsonObject]:
    delta = choice.get("delta")
    finish_reason = choice.get("finish_reason")
    choice_index = _choice_index(choice)
    outputs: list[JsonObject] = []

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
        if finish_reason is not None:
            state.finish_sent = True
        outputs.append(_single_choice_event(event, choice))
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
    if other_delta:
        outputs.append(_choice_delta_event(event, choice, other_delta, finish_reason=None))
        emitted_any_delta = True

    for field_name in _ordered_scan_fields(scanned_text, scan_fields):
        outputs.extend(
            _process_stream_field_text(
                state,
                field_name=field_name,
                text=scanned_text[field_name],
                chunk_id=chunk_id,
                model=model,
                choice_index=choice_index,
                settings=settings,
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
        outputs.append(
            cast(
                "JsonObject",
                make_finish_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    finish_reason=_finish_reason_for_state(finish_reason, state),
                    choice_index=choice_index,
                ),
            ),
        )
        state.finish_sent = True
        emitted_any_delta = True

    if not emitted_any_delta and (not delta or delta.get("tool_calls") == []):
        outputs.append(_choice_delta_event(event, choice, {}, finish_reason=None))

    return outputs


def _process_stream_field_text(
    state: StreamChoiceState,
    *,
    field_name: str,
    text: str,
    chunk_id: str,
    model: str,
    choice_index: int,
    settings: Settings,
) -> list[JsonObject]:
    outputs: list[JsonObject] = []
    state.field_buffers[field_name] = state.field_buffers.get(field_name, "") + text

    while state.field_buffers[field_name]:
        buffer = state.field_buffers[field_name]
        span = find_complete_raw_tool_block_span(buffer)
        if span is not None:
            start, end = span
            prefix = buffer[:start]
            block = buffer[start:end]
            suffix = buffer[end:]

            if len(block) > settings.max_raw_tool_block_chars:
                LOG.warning(
                    "raw tool-call block exceeded max size; passing through as text",
                )
                outputs.append(
                    _field_chunk(chunk_id, model, field_name, prefix + block, choice_index)
                )
                state.field_buffers[field_name] = suffix
                continue

            tool_calls = parse_raw_tool_calls(block)
            if not tool_calls:
                LOG.info("raw tool-call block could not be parsed; passing through as text")
                outputs.append(
                    _field_chunk(chunk_id, model, field_name, prefix + block, choice_index)
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
                    _field_chunk(chunk_id, model, field_name, prefix + block, choice_index)
                )
                state.field_buffers[field_name] = suffix
                continue

            if prefix:
                outputs.append(
                    _field_chunk(chunk_id, model, field_name, prefix, choice_index),
                )
            LOG.info(
                "converted %d raw tool call(s) in streaming chat completion",
                len(tool_calls),
            )
            for tool_chunk in build_tool_call_chunks(
                tool_calls,
                chunk_id=chunk_id,
                model=model,
                argument_chunk_size=settings.tool_argument_chunk_size,
                choice_index=choice_index,
                include_finish=False,
            ):
                outputs.append(cast("JsonObject", tool_chunk))
            state.raw_tool_calls_emitted = True
            state.field_buffers[field_name] = suffix
            continue

        raw_start = find_raw_tool_start(buffer)
        if raw_start is not None:
            if len(buffer) - raw_start > settings.max_raw_tool_block_chars:
                LOG.warning(
                    "incomplete raw tool-call block exceeded max size; passing through as text",
                )
                outputs.append(_field_chunk(chunk_id, model, field_name, buffer, choice_index))
                state.field_buffers[field_name] = ""
                break
            if raw_start > 0:
                outputs.append(
                    _field_chunk(
                        chunk_id,
                        model,
                        field_name,
                        buffer[:raw_start],
                        choice_index,
                    ),
                )
                state.field_buffers[field_name] = buffer[raw_start:]
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
) -> list[JsonObject]:
    outputs: list[JsonObject] = []
    for field_name, buffered_text in list(state.field_buffers.items()):
        if buffered_text:
            outputs.append(_field_chunk(chunk_id, model, field_name, buffered_text, choice_index))
            state.field_buffers[field_name] = ""
    return outputs


def _field_chunk(
    chunk_id: str,
    model: str,
    field_name: str,
    text: str,
    choice_index: int,
) -> JsonObject:
    return {
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


def _body_kwargs(parsed_body: JsonObject | None, raw_body: bytes) -> dict[str, Any]:
    if parsed_body is not None:
        return {"json": parsed_body}
    return {"content": raw_body}


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


def _forward_request_headers(
    request: Request, *, settings: Settings, stream: bool
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

    for key, value in settings.parsed_custom_headers.items():
        lower_key = key.lower()
        if lower_key in HOP_BY_HOP_HEADERS:
            continue
        if stream and lower_key == "accept-encoding":
            continue
        _set_header(headers, key, value)

    return headers


def _set_header(headers: dict[str, str], key: str, value: str) -> None:
    for existing_key in list(headers):
        if existing_key.lower() == key.lower():
            headers.pop(existing_key)
    headers[key] = value


def _forward_response_headers(headers: httpx.Headers) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        forwarded[key] = value
    return forwarded


def _upstream_url(settings: Settings, path: str, query: str) -> str:
    normalized_path = "/" + quote(path.lstrip("/"), safe="/:")
    url = f"{settings.upstream_base_url}{normalized_path}"
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


def _proxy_error(exc: httpx.HTTPError) -> JSONResponse:
    LOG.warning("upstream request failed: %s", exc)
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": "upstream request failed",
                "type": "proxy_error",
            },
        },
    )
