"""Runtime settings."""

from __future__ import annotations

import json

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    upstream_url: str = Field(default="http://127.0.0.1:4000", min_length=1)
    proxy_host: str = "0.0.0.0"  # noqa: S104 - container default should be externally reachable.
    proxy_port: int = 9526
    log_level: str = "INFO"
    stream_guard_chars: int = Field(default=192, ge=1)
    tool_argument_chunk_size: int = Field(default=64, ge=1)
    custom_headers: str = Field(
        default="",
        validation_alias=AliasChoices("CUSTOM_HEADERS", "UPSTREAM_HEADERS"),
    )

    @property
    def upstream_base_url(self) -> str:
        return str(self.upstream_url).rstrip("/")

    @property
    def parsed_custom_headers(self) -> dict[str, str]:
        return parse_custom_headers(self.custom_headers)


def parse_custom_headers(raw_headers: str) -> dict[str, str]:
    raw_headers = raw_headers.strip()
    if not raw_headers:
        return {}

    if raw_headers.startswith("{"):
        parsed = json.loads(raw_headers)
        if not isinstance(parsed, dict):
            msg = "CUSTOM_HEADERS JSON must be an object"
            raise ValueError(msg)
        return {
            str(name).strip(): str(value)
            for name, value in parsed.items()
            if str(name).strip() and value is not None
        }

    headers: dict[str, str] = {}
    for line in raw_headers.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            msg = f"Invalid CUSTOM_HEADERS line: {stripped!r}"
            raise ValueError(msg)
        name, value = stripped.split(":", 1)
        name = name.strip()
        if not name:
            msg = f"Invalid CUSTOM_HEADERS line: {stripped!r}"
            raise ValueError(msg)
        headers[name] = value.strip()
    return headers
