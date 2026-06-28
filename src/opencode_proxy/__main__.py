"""Command-line entrypoint for the OpenCode proxy."""

from __future__ import annotations

import uvicorn

from opencode_proxy.settings import Settings


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "opencode_proxy.app:create_app",
        factory=True,
        host=settings.proxy_host,
        port=settings.proxy_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
