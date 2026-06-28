"""HTTP proxy routes and SSE response rewriting."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from opencode_proxy.compat import (
    JsonObject,
    build_tool_call_chunks,
    convert_chat_completion_response,
    find_raw_tool_start,
    has_complete_raw_tool_block,
    has_raw_tool_prefix,
    parse_raw_tool_calls,
    strip_empty_tool_calls,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from opencode_proxy.settings import Settings

LOG = logging.getLogger(__name__)
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

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


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.api_route(
        CHAT_COMPLETIONS_PATH,
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def chat_completions(request: Request) -> Response:
        return await proxy_chat_completions(request, settings)

    @router.api_route(
        "/models",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def models(request: Request) -> Response:
        return await proxy_passthrough(request, settings, MODELS_PATH)

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
        _sanitize_tools(parsed_body)

    stream = bool(parsed_body.get("stream")) if parsed_body is not None else False
    if stream:
        return await _proxy_streaming_chat_completion(request, settings, parsed_body, body)
    return await _proxy_buffered_chat_completion(request, settings, parsed_body, body)


async def proxy_passthrough(request: Request, settings: Settings, path: str) -> Response:
    body = await request.body()
    headers = _forward_request_headers(request, settings=settings, stream=False)
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
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


async def _proxy_buffered_chat_completion(
    request: Request,
    settings: Settings,
    parsed_body: JsonObject | None,
    raw_body: bytes,
) -> Response:
    headers = _forward_request_headers(request, settings=settings, stream=False)
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
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
        converted, changed = convert_chat_completion_response(response_body)
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

    client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
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

    generator = _rewrite_sse_stream(upstream_response, settings)
    background = BackgroundTask(_close_streaming_resources, upstream_response, client)
    return StreamingResponse(
        generator,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type="text/event-stream",
        background=background,
    )


async def _rewrite_sse_stream(
    upstream_response: httpx.Response,
    settings: Settings,
) -> AsyncIterator[bytes]:
    """Rewrite an SSE chat-completion stream, converting raw tool-call text to
    OpenAI ``tool_calls`` deltas. Only ``choices[0]`` is repaired; ``n>1``
    responses are not supported (OpenCode uses ``n=1``).
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = "unknown"
    pending_text = ""
    monitoring_tool_block = False

    async for line in upstream_response.aiter_lines():
        if not line:
            continue

        event = _parse_sse_data_line(line)
        if event is None:
            yield f"{line}\n\n".encode()
            continue

        if event == "[DONE]":
            if pending_text:
                yield _encode_sse_json(_content_chunk(chunk_id, model, pending_text))
                pending_text = ""
            yield b"data: [DONE]\n\n"
            continue

        if not isinstance(event, dict):
            yield f"{line}\n\n".encode()
            continue

        chunk_id = str(event.get("id") or chunk_id)
        model = str(event.get("model") or model)
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            yield _encode_sse_json(event)
            continue

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            yield _encode_sse_json(event)
            continue

        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            if pending_text:
                yield _encode_sse_json(_content_chunk(chunk_id, model, pending_text))
                pending_text = ""
            yield _encode_sse_json(event)
            continue

        if delta.get("tool_calls"):
            if pending_text:
                yield _encode_sse_json(_content_chunk(chunk_id, model, pending_text))
                pending_text = ""
            yield _encode_sse_json(event)
            continue

        content_value = delta.get("content")
        content_text = content_value if isinstance(content_value, str) else ""
        # non-content delta fields (reasoning_content, reasoning, role, ...) —
        # passed through unchanged so DeepSeek R1 / o1-style reasoning stays in
        # its own field instead of being re-emitted as content.
        other_fields = {k: v for k, v in delta.items() if k != "content"}

        if not content_text:
            cleaned_delta = strip_empty_tool_calls(other_fields)
            if cleaned_delta is not other_fields:
                first_choice["delta"] = cleaned_delta
            yield _encode_sse_json(event)
            continue

        if other_fields:
            cleaned_other = strip_empty_tool_calls(other_fields)
            if cleaned_other:
                passthrough_event = {
                    **event,
                    "choices": [{**first_choice, "delta": cleaned_other}],
                }
                yield _encode_sse_json(passthrough_event)

        pending_text += content_text
        monitoring_tool_block = monitoring_tool_block or has_raw_tool_prefix(pending_text)

        if monitoring_tool_block and has_complete_raw_tool_block(pending_text):
            start = find_raw_tool_start(pending_text)
            prefix = pending_text[:start] if start is not None else ""
            if prefix:
                yield _encode_sse_json(_content_chunk(chunk_id, model, prefix))

            tool_calls = parse_raw_tool_calls(pending_text[start or 0 :])
            if tool_calls:
                LOG.info(
                    "converted %d raw tool call(s) in streaming chat completion", len(tool_calls)
                )
                for tool_chunk in build_tool_call_chunks(
                    tool_calls,
                    chunk_id=chunk_id,
                    model=model,
                    argument_chunk_size=settings.tool_argument_chunk_size,
                ):
                    yield _encode_sse_json(tool_chunk)
                yield b"data: [DONE]\n\n"
                return

            monitoring_tool_block = False

        # Always hold back `stream_guard_chars` of pending content so a split
        # raw tool-call marker (or a false-positive prefix match like the word
        # "tool") can complete without starving the stream until [DONE].
        flush_size = len(pending_text) - settings.stream_guard_chars
        if flush_size > 0:
            yield _encode_sse_json(_content_chunk(chunk_id, model, pending_text[:flush_size]))
            pending_text = pending_text[flush_size:]

    if pending_text:
        yield _encode_sse_json(_content_chunk(chunk_id, model, pending_text))
    yield b"data: [DONE]\n\n"


async def _close_streaming_resources(
    upstream_response: httpx.Response,
    client: httpx.AsyncClient,
) -> None:
    await upstream_response.aclose()
    await client.aclose()


def _content_chunk(chunk_id: str, model: str, text: str) -> JsonObject:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": text},
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


def _sanitize_tools(body: JsonObject) -> None:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return

    function_tools = [
        tool for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"
    ]
    if function_tools:
        body["tools"] = function_tools
    else:
        body.pop("tools", None)


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


def _parse_sse_data_line(line: str) -> JsonObject | str | None:
    if not line.startswith("data:"):
        return None

    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return "[DONE]"

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload

    return parsed if isinstance(parsed, dict) else payload


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
