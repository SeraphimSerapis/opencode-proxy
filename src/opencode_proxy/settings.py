"""Runtime settings."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, field_validator
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
    upstream_connect_timeout: float = Field(default=10.0, ge=0)
    upstream_read_timeout: float = Field(default=0.0, ge=0)
    upstream_write_timeout: float = Field(default=30.0, ge=0)
    upstream_pool_timeout: float = Field(default=30.0, ge=0)
    stream_guard_chars: int = Field(default=192, ge=1)
    tool_argument_chunk_size: int = Field(default=64, ge=1)
    max_raw_tool_block_chars: int = Field(default=131_072, ge=1)
    max_tool_calls: int = Field(default=32, ge=1)
    max_tool_argument_chars: int = Field(default=262_144, ge=1)
    tool_call_scan_fields: str = Field(
        default="content,reasoning,reasoning_content",
        validation_alias="TOOL_CALL_SCAN_FIELDS",
    )
    sanitize_tools: bool = Field(default=True, validation_alias="SANITIZE_TOOLS")
    request_drop_fields: str = Field(default="", validation_alias="REQUEST_DROP_FIELDS")
    custom_headers: str = Field(
        default="",
        validation_alias=AliasChoices("CUSTOM_HEADERS", "UPSTREAM_HEADERS"),
    )
    model_aliases: str = Field(default="", validation_alias="MODEL_ALIASES")
    alias_conflict_policy: str = Field(default="skip", validation_alias="ALIAS_CONFLICT_POLICY")

    @field_validator("tool_call_scan_fields", mode="before")
    @classmethod
    def validate_tool_call_scan_fields(cls, value: object) -> str:
        parse_tool_call_scan_fields(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list | tuple):
            return ",".join(str(field) for field in value)
        msg = "TOOL_CALL_SCAN_FIELDS must be a comma-separated string or list"
        raise ValueError(msg)

    @field_validator("request_drop_fields", mode="before")
    @classmethod
    def validate_request_drop_fields(cls, value: object) -> str:
        parse_request_drop_fields(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list | tuple):
            return ",".join(str(field) for field in value)
        msg = "REQUEST_DROP_FIELDS must be a comma-separated string or list"
        raise ValueError(msg)

    @field_validator("alias_conflict_policy")
    @classmethod
    def validate_alias_conflict_policy(cls, value: str) -> str:
        if value not in {"skip", "shadow", "error"}:
            msg = "ALIAS_CONFLICT_POLICY must be one of: skip, shadow, error"
            raise ValueError(msg)
        return value

    @property
    def upstream_base_url(self) -> str:
        return str(self.upstream_url).rstrip("/")

    @property
    def upstream_safe_origin(self) -> str:
        parsed = urlsplit(self.upstream_base_url)
        if not parsed.scheme or not parsed.hostname:
            return self.upstream_base_url
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}"

    @property
    def parsed_custom_headers(self) -> dict[str, str]:
        return parse_custom_headers(self.custom_headers)

    @property
    def parsed_tool_call_scan_fields(self) -> tuple[str, ...]:
        return parse_tool_call_scan_fields(self.tool_call_scan_fields)

    @property
    def parsed_request_drop_fields(self) -> tuple[str, ...]:
        return parse_request_drop_fields(self.request_drop_fields)

    @property
    def parsed_model_aliases(self) -> dict[str, str]:
        return parse_model_aliases(self.model_aliases)

    @property
    def safe_config(self) -> dict[str, object]:
        return {
            "upstream": {
                "origin": self.upstream_safe_origin,
                "timeouts": {
                    "connect": self.upstream_connect_timeout,
                    "read": None if self.upstream_read_timeout == 0 else self.upstream_read_timeout,
                    "write": self.upstream_write_timeout,
                    "pool": self.upstream_pool_timeout,
                },
            },
            "streaming": {
                "guard_chars": self.stream_guard_chars,
                "tool_argument_chunk_size": self.tool_argument_chunk_size,
            },
            "tool_call_repair": {
                "scan_fields": list(self.parsed_tool_call_scan_fields),
                "max_raw_tool_block_chars": self.max_raw_tool_block_chars,
                "max_tool_calls": self.max_tool_calls,
                "max_tool_argument_chars": self.max_tool_argument_chars,
            },
            "request_transforms": {
                "sanitize_tools": self.sanitize_tools,
                "drop_fields": list(self.parsed_request_drop_fields),
            },
            "model_aliases": {
                "aliases": sorted(self.parsed_model_aliases),
                "conflict_policy": self.alias_conflict_policy,
            },
            "custom_headers": {
                "names": sorted(self.parsed_custom_headers),
            },
        }


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


def parse_tool_call_scan_fields(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_fields = [field.strip() for field in value.split(",")]
    elif isinstance(value, list | tuple):
        raw_fields = [str(field).strip() for field in value]
    else:
        msg = "TOOL_CALL_SCAN_FIELDS must be a comma-separated string or list"
        raise ValueError(msg)

    allowed = {"content", "reasoning", "reasoning_content"}
    fields: list[str] = []
    for field in raw_fields:
        if not field:
            continue
        if field == "all":
            fields.extend(["content", "reasoning", "reasoning_content"])
            continue
        if field not in allowed:
            msg = f"Unsupported TOOL_CALL_SCAN_FIELDS value: {field!r}"
            raise ValueError(msg)
        fields.append(field)

    deduped = tuple(dict.fromkeys(fields))
    if not deduped:
        msg = "TOOL_CALL_SCAN_FIELDS must include at least one field"
        raise ValueError(msg)
    return deduped


def parse_request_drop_fields(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_fields = [field.strip() for field in value.split(",")]
    elif isinstance(value, list | tuple):
        raw_fields = [str(field).strip() for field in value]
    else:
        msg = "REQUEST_DROP_FIELDS must be a comma-separated string or list"
        raise ValueError(msg)

    fields: list[str] = []
    for field in raw_fields:
        if not field:
            continue
        if not field.replace("_", "").replace("-", "").isalnum():
            msg = f"Unsupported REQUEST_DROP_FIELDS value: {field!r}"
            raise ValueError(msg)
        fields.append(field)
    return tuple(dict.fromkeys(fields))


def parse_model_aliases(raw_aliases: str) -> dict[str, str]:
    raw_aliases = _strip_wrapping_quotes(raw_aliases.strip())
    if not raw_aliases:
        return {}

    if raw_aliases.startswith("{"):
        parsed = json.loads(raw_aliases)
        if not isinstance(parsed, dict):
            msg = "MODEL_ALIASES JSON must be an object"
            raise ValueError(msg)
        return {
            _strip_wrapping_quotes(str(alias).strip()): _strip_wrapping_quotes(str(target).strip())
            for alias, target in parsed.items()
            if _strip_wrapping_quotes(str(alias).strip())
            and _strip_wrapping_quotes(str(target).strip())
        }

    aliases: dict[str, str] = {}
    for item in _split_alias_items(raw_aliases):
        if "=" in item:
            alias, target = item.split("=", 1)
        elif ":" in item:
            alias, target = item.split(":", 1)
        else:
            msg = f"Invalid MODEL_ALIASES item: {item!r}"
            raise ValueError(msg)
        alias = _strip_wrapping_quotes(alias.strip())
        target = _strip_wrapping_quotes(target.strip())
        if not alias or not target:
            msg = f"Invalid MODEL_ALIASES item: {item!r}"
            raise ValueError(msg)
        aliases[alias] = target
    return aliases


def _split_alias_items(raw_aliases: str) -> list[str]:
    items: list[str] = []
    for line in raw_aliases.splitlines():
        for item in line.split(","):
            stripped = item.strip()
            if stripped:
                items.append(stripped)
    return items


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value.strip("'\"").strip()
