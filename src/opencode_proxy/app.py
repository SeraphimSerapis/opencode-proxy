"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from opencode_proxy import __version__
from opencode_proxy.proxy import build_router
from opencode_proxy.settings import Settings

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

    app = FastAPI(title="OpenCode Proxy", version=__version__)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz/config")
    async def healthz_config() -> dict[str, object]:
        return settings.safe_config

    app.include_router(build_router(settings))
    return app
