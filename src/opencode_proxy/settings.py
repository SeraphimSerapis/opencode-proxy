"""Runtime settings."""

from __future__ import annotations

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    upstream_url: AnyHttpUrl = Field(default="http://127.0.0.1:4000")
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 9526
    log_level: str = "INFO"
    stream_guard_chars: int = 192
    tool_argument_chunk_size: int = 64

    @property
    def upstream_base_url(self) -> str:
        return str(self.upstream_url).rstrip("/")
