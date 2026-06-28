"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from opencode_proxy import __version__
from opencode_proxy.proxy import build_router
from opencode_proxy.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app = FastAPI(title="OpenCode Proxy", version=__version__)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(build_router(settings))
    return app
