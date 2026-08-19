"""Ollama REST compatibility routes backed by the shared OpenAI gateway."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from opencode_proxy.compat import (
    RepairStats,
    annotate_empty_completion,
    convert_chat_completion_response,
    is_empty_completion,
)
from opencode_proxy.ollama_models import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaCopyRequest,
    OllamaCreateRequest,
    OllamaDeleteRequest,
    OllamaEmbeddingsRequest,
    OllamaEmbeddingsResponse,
    OllamaEmbedRequest,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaMessage,
    OllamaPullRequest,
    OllamaPullResponse,
    OllamaShowRequest,
)
from opencode_proxy.ollama_streaming import stream_chat_to_ollama, stream_generate_to_ollama
from opencode_proxy.ollama_translate import (
    create_show_response,
    ollama_chat_to_openai,
    ollama_embed_to_openai,
    ollama_generate_to_openai,
    openai_chat_to_ollama,
    openai_chat_to_ollama_generate,
    openai_embeddings_to_ollama,
    openai_models_to_ollama,
    openai_models_to_running,
)
from opencode_proxy.proxy import (
    DEFAULT_TOOL_REPAIR_CONTEXT,
    MODELS_PATH,
    ToolRepairContext,
    _aclose_upstream_and_release_slot,
    _acquire_upstream_slot,
    _add_model_aliases,
    _apply_model_alias,
    _forward_request_headers,
    _forward_response_headers,
    _note_finish_reasons,
    _note_upstream_error,
    _proxy_error,
    _record_repair_stats,
    _record_request_normalizations,
    _record_usage,
    _release_upstream_slot,
    _request_metrics,
    _set_header,
    _thinking_transport,
    _tool_repair_context,
    _upstream_client,
    _upstream_url,
    apply_stream_response_headers,
    apply_target_model,
    send_upstream_with_retries,
)
from opencode_proxy.request_compat import (
    RequestNormalizationStats,
    normalize_reasoning_effort,
    normalize_request,
)
from opencode_proxy.routing import UpstreamTarget, resolve_upstream_target

if TYPE_CHECKING:
    from opencode_proxy.compat import JsonObject
    from opencode_proxy.metrics import ProxyMetrics
    from opencode_proxy.settings import Settings

LOG = logging.getLogger(__name__)


def build_ollama_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse("Ollama is running")

    @router.head("/")
    async def root_head() -> PlainTextResponse:
        return PlainTextResponse("Ollama is running")

    @router.get("/api/version")
    async def version() -> dict[str, str]:
        return {"version": settings.ollama_version}

    @router.post("/api/chat")
    async def chat(payload: OllamaChatRequest, request: Request) -> Response:
        if not payload.messages:
            reason = "unload" if str(payload.keep_alive) == "0" else "load"
            return JSONResponse(
                OllamaChatResponse(
                    model=payload.model,
                    message=OllamaMessage(role="assistant", content=""),
                    done=True,
                    done_reason=reason,
                ).model_dump(exclude_none=True)
            )

        openai_payload = ollama_chat_to_openai(payload)
        alias_error = await _apply_model_alias(request, openai_payload, settings)
        if alias_error is not None:
            return alias_error
        target = resolve_upstream_target(settings, openai_payload)
        apply_target_model(openai_payload, target)
        _normalize_ollama_payload(openai_payload, payload.think, settings, request)
        repair_context = _tool_repair_context(openai_payload, settings)
        overload = await _acquire_upstream_slot(request)
        if overload is not None:
            return overload
        if payload.stream:
            return await _stream_chat(
                request, settings, openai_payload, payload.model, target, repair_context
            )
        try:
            response, repaired = await _request_ollama_completion(
                request, settings, openai_payload, target=target, repair_context=repair_context
            )
            if not isinstance(response, httpx.Response):
                return response
            assert repaired is not None
            return JSONResponse(
                openai_chat_to_ollama(repaired, payload.model).model_dump(exclude_none=True),
                status_code=response.status_code,
            )
        finally:
            await _release_upstream_slot(request)

    @router.post("/api/generate")
    async def generate(payload: OllamaGenerateRequest, request: Request) -> Response:
        if not payload.prompt:
            reason = "unload" if str(payload.keep_alive) == "0" else "load"
            return JSONResponse(
                OllamaGenerateResponse(
                    model=payload.model, done=True, done_reason=reason
                ).model_dump(exclude_none=True)
            )

        openai_payload = ollama_generate_to_openai(payload)
        alias_error = await _apply_model_alias(request, openai_payload, settings)
        if alias_error is not None:
            return alias_error
        target = resolve_upstream_target(settings, openai_payload)
        apply_target_model(openai_payload, target)
        _normalize_ollama_payload(openai_payload, payload.think, settings, request)
        repair_context = _tool_repair_context(openai_payload, settings)
        overload = await _acquire_upstream_slot(request)
        if overload is not None:
            return overload
        if payload.stream:
            return await _stream_generate(
                request, settings, openai_payload, payload.model, target, repair_context
            )
        try:
            response, repaired = await _request_ollama_completion(
                request, settings, openai_payload, target=target, repair_context=repair_context
            )
            if not isinstance(response, httpx.Response):
                return response
            assert repaired is not None
            return JSONResponse(
                openai_chat_to_ollama_generate(repaired, payload.model).model_dump(
                    exclude_none=True
                ),
                status_code=response.status_code,
            )
        finally:
            await _release_upstream_slot(request)

    @router.post("/api/embed")
    async def embed(payload: OllamaEmbedRequest, request: Request) -> Response:
        openai_payload = ollama_embed_to_openai(payload)
        alias_error = await _apply_model_alias(request, openai_payload, settings)
        if alias_error is not None:
            return alias_error
        response = await _request_upstream(request, settings, "/v1/embeddings", openai_payload)
        if isinstance(response, Response):
            return response
        return JSONResponse(
            openai_embeddings_to_ollama(_json_response(response), payload.model).model_dump(
                exclude_none=True
            ),
            status_code=response.status_code,
        )

    @router.post("/api/embeddings")
    async def embeddings(payload: OllamaEmbeddingsRequest, request: Request) -> Response:
        embed_payload = OllamaEmbedRequest(model=payload.model, input=payload.prompt)
        openai_payload = ollama_embed_to_openai(embed_payload)
        alias_error = await _apply_model_alias(request, openai_payload, settings)
        if alias_error is not None:
            return alias_error
        response = await _request_upstream(request, settings, "/v1/embeddings", openai_payload)
        if isinstance(response, Response):
            return response
        ollama_response = openai_embeddings_to_ollama(_json_response(response), payload.model)
        embedding = ollama_response.embeddings[0] if ollama_response.embeddings else []
        return JSONResponse(
            OllamaEmbeddingsResponse(model=payload.model, embedding=embedding).model_dump()
        )

    @router.get("/api/tags")
    async def tags(request: Request) -> Response:
        response = await _request_upstream_raw(request, settings, "GET", MODELS_PATH)
        if isinstance(response, Response):
            return response
        body = _json_response(response)
        if not _add_model_aliases(body, settings):
            return JSONResponse(
                {"error": {"message": "model alias conflict", "type": "alias_conflict"}},
                status_code=409,
            )
        return JSONResponse(openai_models_to_ollama(body).model_dump())

    @router.get("/api/ps")
    async def running_models(request: Request) -> Response:
        response = await _request_upstream_raw(request, settings, "GET", MODELS_PATH)
        if isinstance(response, Response):
            return response
        body = _json_response(response)
        if not _add_model_aliases(body, settings):
            return JSONResponse(
                {"error": {"message": "model alias conflict", "type": "alias_conflict"}},
                status_code=409,
            )
        return JSONResponse(openai_models_to_running(body).model_dump())

    @router.post("/api/show")
    async def show(payload: OllamaShowRequest) -> dict[str, Any]:
        return create_show_response(payload.name).model_dump()

    @router.post("/api/pull")
    async def pull(payload: OllamaPullRequest) -> dict[str, str]:
        LOG.info("Ignoring model pull for %s; models are managed upstream", payload.name)
        return OllamaPullResponse().model_dump()

    @router.post("/api/push")
    async def push(payload: OllamaPullRequest) -> dict[str, str]:
        LOG.info("Ignoring model push for %s; models are managed upstream", payload.name)
        return OllamaPullResponse().model_dump()

    @router.post("/api/copy")
    async def copy(payload: OllamaCopyRequest) -> Response:
        LOG.info("Ignoring model copy %s -> %s", payload.source, payload.destination)
        return Response(status_code=200)

    @router.delete("/api/delete")
    async def delete(payload: OllamaDeleteRequest) -> Response:
        LOG.info("Ignoring model delete for %s", payload.name)
        return Response(status_code=200)

    @router.post("/api/create")
    async def create(payload: OllamaCreateRequest) -> dict[str, str]:
        LOG.info("Ignoring model create for %s", payload.name)
        return {"status": "success"}

    @router.head("/api/blobs/{digest}")
    async def check_blob(digest: str) -> Response:
        return Response(status_code=200)

    @router.post("/api/blobs/{digest}")
    async def push_blob(digest: str) -> Response:
        return Response(status_code=200)

    return router


async def _stream_chat(
    request: Request,
    settings: Settings,
    payload: JsonObject,
    model: str,
    target: UpstreamTarget | None = None,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
) -> Response:
    try:
        response = await _send_streaming(
            request, settings, "/v1/chat/completions", payload, target=target
        )
    except Exception:
        await _release_upstream_slot(request)
        raise
    if isinstance(response, Response):
        await _release_upstream_slot(request)
        return response
    return StreamingResponse(
        stream_chat_to_ollama(request, response, settings, model, repair_context),
        status_code=response.status_code,
        headers=apply_stream_response_headers({}),
        media_type="application/x-ndjson",
        background=BackgroundTask(_aclose_upstream_and_release_slot, response, request),
    )


async def _stream_generate(
    request: Request,
    settings: Settings,
    payload: JsonObject,
    model: str,
    target: UpstreamTarget | None = None,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
) -> Response:
    try:
        response = await _send_streaming(
            request, settings, "/v1/chat/completions", payload, target=target
        )
    except Exception:
        await _release_upstream_slot(request)
        raise
    if isinstance(response, Response):
        await _release_upstream_slot(request)
        return response
    return StreamingResponse(
        stream_generate_to_ollama(request, response, settings, model, repair_context),
        status_code=response.status_code,
        headers=apply_stream_response_headers({}),
        media_type="application/x-ndjson",
        background=BackgroundTask(_aclose_upstream_and_release_slot, response, request),
    )


def _normalize_ollama_payload(
    payload: JsonObject,
    think: bool | str | None,
    settings: Settings,
    request: Request,
) -> None:
    """Apply the same DeepSeek contract used by the native OpenAI route."""
    thinking_transport = _thinking_transport(payload, settings)
    if thinking_transport is None:
        return
    if think is not None:
        payload["reasoning_effort"] = _ollama_reasoning_effort(think)
    if settings.normalize_requests:
        stats = normalize_request(payload, thinking_transport=thinking_transport)
    else:
        # ``think`` is part of the Ollama protocol, so translating it is adapter
        # behavior rather than optional message hygiene.
        stats = RequestNormalizationStats()
        if think is not None:
            normalize_reasoning_effort(payload, stats, transport=thinking_transport)
    _record_request_normalizations(_request_metrics(request), stats)


def _ollama_reasoning_effort(think: bool | str) -> str:
    """Translate Ollama's boolean/string ``think`` control to the shared wire field."""
    if isinstance(think, bool):
        return "high" if think else "off"
    normalized = think.strip().lower()
    if normalized in {"", "on", "true", "enabled"}:
        return "high"
    if normalized in {"off", "false", "none", "disabled"}:
        return "off"
    return normalized


async def _request_ollama_completion(
    request: Request,
    settings: Settings,
    payload: JsonObject,
    *,
    target: UpstreamTarget,
    repair_context: ToolRepairContext,
) -> tuple[httpx.Response | Response, JsonObject | None]:
    """Fetch, repair, and retry a buffered Ollama chat/generate completion."""
    deepseek_profile = _thinking_transport(payload, settings) is not None
    empty_retries_left = settings.empty_response_retries if deepseek_profile else 0
    while True:
        response = await _request_upstream(
            request, settings, "/v1/chat/completions", payload, target=target
        )
        if isinstance(response, Response):
            return response, None

        metrics = _request_metrics(request)
        repaired = _repair_response(
            _json_response(response),
            settings,
            metrics=metrics,
            repair_context=repair_context,
        )
        _record_usage(repaired.get("usage"), metrics)
        _note_finish_reasons(repaired, metrics=metrics, transport="ollama")

        if is_empty_completion(repaired):
            if empty_retries_left > 0:
                empty_retries_left -= 1
                if metrics is not None:
                    metrics.upstream_retries.labels(reason="empty_response").inc()
                LOG.warning(
                    "Ollama upstream returned a completed turn with no output; retrying "
                    "(%d attempt(s) left)",
                    empty_retries_left,
                )
                await response.aclose()
                continue
            if metrics is not None:
                metrics.empty_turns.inc()
            annotate_empty_completion(repaired, settings.empty_turn_notice)

        return response, repaired


async def _send_streaming(
    request: Request,
    settings: Settings,
    path: str,
    payload: JsonObject,
    *,
    target: UpstreamTarget | None = None,
) -> httpx.Response | Response:
    client = _upstream_client(request)
    headers = _forward_request_headers(request, settings=settings, stream=True, target=target)
    _set_header(headers, "Content-Type", "application/json")
    try:
        response = await send_upstream_with_retries(
            client,
            lambda: client.build_request(
                "POST",
                _upstream_url(settings, path, request.url.query, target=target),
                headers=headers,
                json=payload,
            ),
            settings=settings,
            stream=True,
            metrics=_request_metrics(request),
        )
    except httpx.HTTPError as exc:
        return _proxy_error(exc)
    if response.status_code >= 400:
        body = await response.aread()
        _note_upstream_error(response, metrics=_request_metrics(request))
        await response.aclose()
        return Response(
            body,
            status_code=response.status_code,
            headers=_forward_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type.lower():
        body = await response.aread()
        await response.aclose()
        return Response(
            body,
            status_code=response.status_code,
            headers=_forward_response_headers(response.headers),
            media_type=content_type or None,
        )
    return response


async def _request_upstream(
    request: Request,
    settings: Settings,
    path: str,
    payload: JsonObject,
    *,
    target: UpstreamTarget | None = None,
) -> httpx.Response | Response:
    return await _request_upstream_raw(request, settings, "POST", path, payload, target=target)


async def _request_upstream_raw(
    request: Request,
    settings: Settings,
    method: str,
    path: str,
    payload: JsonObject | None = None,
    *,
    target: UpstreamTarget | None = None,
) -> httpx.Response | Response:
    client = _upstream_client(request)
    headers = _forward_request_headers(request, settings=settings, stream=False, target=target)
    if payload is not None:
        _set_header(headers, "Content-Type", "application/json")
    request_kwargs: dict[str, Any] = {}
    if payload is not None:
        request_kwargs["json"] = payload
    try:
        response = await send_upstream_with_retries(
            client,
            lambda: client.build_request(
                method,
                _upstream_url(settings, path, request.url.query, target=target),
                headers=headers,
                **request_kwargs,
            ),
            settings=settings,
            stream=False,
            metrics=_request_metrics(request),
        )
    except httpx.HTTPError as exc:
        return _proxy_error(exc)
    if response.status_code >= 400:
        if path == "/v1/chat/completions":
            _note_upstream_error(response, metrics=_request_metrics(request))
        return Response(
            response.content,
            status_code=response.status_code,
            headers=_forward_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )
    return response


def _json_response(response: httpx.Response) -> JsonObject:
    try:
        parsed = response.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _repair_response(
    body: JsonObject,
    settings: Settings,
    *,
    metrics: ProxyMetrics | None = None,
    repair_context: ToolRepairContext = DEFAULT_TOOL_REPAIR_CONTEXT,
) -> JsonObject:
    stats = RepairStats()
    repaired, _ = convert_chat_completion_response(
        body,
        tool_call_scan_fields=settings.parsed_tool_call_scan_fields,
        max_raw_tool_block_chars=settings.max_raw_tool_block_chars,
        max_tool_calls=settings.max_tool_calls,
        max_tool_argument_chars=settings.max_tool_argument_chars,
        recover_orphan_invokes=repair_context.recover_orphan_invokes,
        declared_tool_names=repair_context.declared_tool_names,
        stats=stats,
    )
    _record_repair_stats(metrics, stats, transport="ollama")
    return repaired
