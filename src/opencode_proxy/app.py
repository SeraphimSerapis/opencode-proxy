"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opencode_proxy import __version__
from opencode_proxy.concurrency import UpstreamConcurrencyLimiter
from opencode_proxy.ollama import build_ollama_router
from opencode_proxy.proxy import MODELS_PATH, build_router, create_upstream_client
from opencode_proxy.settings import Settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

LOG = logging.getLogger(__name__)


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

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        return await check_upstream_ready(request, settings)

    @app.get("/healthz/config")
    async def healthz_config() -> dict[str, object]:
        return settings.safe_config

    # Register Ollama routes before the OpenAI catch-all route so /api/* paths
    # are handled by the native compatibility adapter rather than passthrough.
    app.include_router(build_ollama_router(settings))
    app.include_router(build_router(settings))
    return app


async def check_upstream_ready(request: Request, settings: Settings) -> JSONResponse:
    """Probe upstream model discovery; any non-5xx response means ready."""
    client = request.app.state.upstream_client
    if not isinstance(client, httpx.AsyncClient):
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "upstream": "client_uninitialized"},
        )

    timeout = httpx.Timeout(settings.upstream_ready_timeout)
    headers: dict[str, str] = {}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    for key, value in settings.parsed_custom_headers.items():
        headers[key] = value

    url = f"{settings.upstream_base_url}{MODELS_PATH}"
    try:
        response = await client.get(url, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        LOG.warning("readiness probe timed out contacting upstream")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "upstream": "timeout"},
        )
    except httpx.HTTPError:
        LOG.warning("readiness probe failed to reach upstream")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "upstream": "unreachable"},
        )

    if response.status_code >= 500:
        LOG.warning("readiness probe saw upstream status %s", response.status_code)
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "upstream": "error",
                "upstream_status": response.status_code,
            },
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ok", "upstream_status": response.status_code},
    )
