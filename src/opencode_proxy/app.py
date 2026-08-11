"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from opencode_proxy import __version__
from opencode_proxy.concurrency import UpstreamConcurrencyLimiter
from opencode_proxy.metrics import ProxyMetrics
from opencode_proxy.ollama import build_ollama_router
from opencode_proxy.proxy import MODELS_PATH, build_router, create_upstream_client
from opencode_proxy.settings import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

LOG = logging.getLogger(__name__)

# An upstream that rejects the proxy's credentials is still serving; readiness
# asks whether the upstream is up, not whether this caller is authorized.
AUTH_STATUSES = frozenset({401, 403})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    aliases = settings.parsed_model_aliases
    if aliases:
        LOG.info("configured %d model alias(es): %s", len(aliases), ", ".join(sorted(aliases)))
    else:
        LOG.info("no model aliases configured")

    if settings.max_concurrent_upstream:
        LOG.info("upstream concurrency limit: %d", settings.max_concurrent_upstream)
    else:
        LOG.info("upstream concurrency limit: unlimited")

    upstream_client = create_upstream_client(settings)
    upstream_limiter = (
        UpstreamConcurrencyLimiter(settings.max_concurrent_upstream)
        if settings.max_concurrent_upstream > 0
        else None
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await upstream_client.aclose()

    app = FastAPI(title="OpenCode Proxy", version=__version__, lifespan=lifespan)
    app.state.upstream_client = upstream_client
    app.state.upstream_limiter = upstream_limiter
    app.state.settings = settings
    app.state.metrics = ProxyMetrics.create()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        return await check_upstream_ready(request, settings)

    @app.get("/healthz/config")
    async def healthz_config() -> dict[str, object]:
        return settings.safe_config

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        payload, content_type = request.app.state.metrics.render()
        return Response(content=payload, headers={"Content-Type": content_type})

    # Register Ollama routes before the OpenAI catch-all route so /api/* paths
    # are handled by the native compatibility adapter rather than passthrough.
    app.include_router(build_ollama_router(settings))
    app.include_router(build_router(settings))
    return app


def _not_ready(metrics: ProxyMetrics | None, reason: str, **extra: object) -> JSONResponse:
    if isinstance(metrics, ProxyMetrics):
        metrics.upstream_ready_failures.labels(reason=reason).inc()
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "upstream": reason, **extra},
    )


async def check_upstream_ready(request: Request, settings: Settings) -> JSONResponse:
    """Probe the upstream and report whether it can actually serve a completion.

    Reachability alone is not readiness. ``/v1/models`` is a static registry on
    both vLLM and LiteLLM, so it answers ``200`` long after the engine behind it
    has died; the optional ``UPSTREAM_HEALTH_PATH`` probe is what catches that.
    An auth rejection still counts as ready — the upstream is serving, the
    caller's credentials are the caller's problem — but any other non-2xx does
    not, because a ``404`` means the configured base URL is wrong.
    """
    client = request.app.state.upstream_client
    metrics = getattr(request.app.state, "metrics", None)
    if not isinstance(client, httpx.AsyncClient):
        return _not_ready(metrics, "client_uninitialized")

    timeout = httpx.Timeout(settings.upstream_ready_timeout)
    headers: dict[str, str] = {}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    for key, value in settings.parsed_custom_headers.items():
        headers[key] = value

    health_url = settings.upstream_health_url
    if health_url:
        try:
            health = await client.get(health_url, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            LOG.warning("readiness probe timed out on upstream health path")
            return _not_ready(metrics, "engine_timeout")
        except httpx.HTTPError:
            LOG.warning("readiness probe failed to reach upstream health path")
            return _not_ready(metrics, "unreachable")
        if health.status_code >= 400 and health.status_code not in AUTH_STATUSES:
            LOG.warning("readiness probe saw upstream health status %s", health.status_code)
            return _not_ready(metrics, "engine_unhealthy", upstream_status=health.status_code)

    url = f"{settings.upstream_base_url}{MODELS_PATH}"
    try:
        response = await client.get(url, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        LOG.warning("readiness probe timed out contacting upstream")
        return _not_ready(metrics, "timeout")
    except httpx.HTTPError:
        LOG.warning("readiness probe failed to reach upstream")
        return _not_ready(metrics, "unreachable")

    if response.status_code in AUTH_STATUSES:
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "upstream_status": response.status_code},
        )

    if response.status_code >= 500:
        LOG.warning("readiness probe saw upstream status %s", response.status_code)
        return _not_ready(metrics, "error", upstream_status=response.status_code)

    if response.status_code >= 400:
        LOG.warning(
            "readiness probe saw upstream status %s for %s; check UPSTREAM_URL",
            response.status_code,
            MODELS_PATH,
        )
        return _not_ready(metrics, "unexpected_status", upstream_status=response.status_code)

    if not _lists_any_model(response):
        LOG.warning("readiness probe saw an upstream model list with no servable models")
        return _not_ready(metrics, "no_models", upstream_status=response.status_code)

    return JSONResponse(
        status_code=200,
        content={"status": "ok", "upstream_status": response.status_code},
    )


def _lists_any_model(response: httpx.Response) -> bool:
    """True when the probe body is an OpenAI model list holding at least one model."""
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, list) and bool(data)
