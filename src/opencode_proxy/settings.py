"""Runtime settings."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from opencode_proxy.config_file import load_config_file
from opencode_proxy.request_compat import DEFAULT_THINKING_TRANSPORT, THINKING_TRANSPORTS
from opencode_proxy.routing import ModalityRoute, parse_modality_routes

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

LOG = logging.getLogger(__name__)

CONFIG_FILE_ENV = "PROXY_CONFIG_FILE"


@dataclass(frozen=True)
class ModelCompatibility:
    profile: str
    recover_orphan_invokes: bool
    thinking_transport: str = DEFAULT_THINKING_TRANSPORT


class ConfigFileSettingsSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source backed by the optional YAML config file."""

    def __init__(self, settings_cls: type[BaseSettings], path: str) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return load_config_file(self._path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
    )

    upstream_url: str = Field(
        default="http://127.0.0.1:4000",
        min_length=1,
        validation_alias=AliasChoices("UPSTREAM_URL", "OLLAMA_PROXY_UPSTREAM_URL"),
    )
    upstream_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("UPSTREAM_API_KEY", "OLLAMA_PROXY_UPSTREAM_API_KEY"),
    )
    proxy_host: str = "0.0.0.0"  # noqa: S104 - container default should be externally reachable.
    proxy_port: int = 9526
    log_level: str = Field(
        default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "OLLAMA_PROXY_LOG_LEVEL")
    )
    ollama_version: str = Field(
        default="0.5.1",
        validation_alias=AliasChoices("OLLAMA_VERSION", "OLLAMA_PROXY_OLLAMA_VERSION"),
    )
    upstream_connect_timeout: float = Field(default=10.0, ge=0)
    upstream_read_timeout: float = Field(default=0.0, ge=0)
    upstream_write_timeout: float = Field(default=30.0, ge=0)
    upstream_pool_timeout: float = Field(default=30.0, ge=0)
    upstream_ready_timeout: float = Field(default=2.0, ge=0.1)
    # A model-listing probe only proves the upstream HTTP router is up: vLLM and
    # LiteLLM both serve /v1/models from static config, so it still answers 200
    # after the inference engine behind it has died. Point this at an endpoint
    # that actually exercises the engine (vLLM: /health) to catch that.
    upstream_health_path: str = Field(default="", validation_alias="UPSTREAM_HEALTH_PATH")
    # Gap *between* SSE frames. Once tokens are flowing they arrive continuously
    # (measured inter-token p99.9 is ~1.5s), so tens of seconds of mid-stream
    # silence means the turn has stalled, not that the model is thinking.
    upstream_stream_idle_timeout: float = Field(
        default=30.0,
        ge=0,
        validation_alias="UPSTREAM_STREAM_IDLE_TIMEOUT",
    )
    # Wait for the *first* frame, which covers prefill. This is a much larger
    # number than the between-frame gap: nothing is sent while the prompt is
    # being processed, and a long prompt legitimately takes minutes (measured
    # histogram-estimated time-to-first-token p99 tail is ~351s). Use `0` to disable.
    upstream_stream_first_frame_timeout: float = Field(
        default=480.0,
        ge=0,
        validation_alias="UPSTREAM_STREAM_FIRST_FRAME_TIMEOUT",
    )
    sse_keepalive_interval: float = Field(
        default=10.0,
        ge=0,
        validation_alias="SSE_KEEPALIVE_INTERVAL",
    )
    # Emitted when a turn closes with no content and no tool calls, which a
    # reasoning model does when it exhausts max_tokens while still thinking.
    # Set empty to disable and pass the empty turn through untouched.
    empty_turn_notice: str = Field(
        default=(
            "[proxy: the model ended this turn without producing any output "
            "or tool call, usually because it exhausted its token budget while "
            "reasoning. Retry with a larger max_tokens.]"
        ),
        validation_alias="EMPTY_TURN_NOTICE",
    )
    upstream_max_retries: int = Field(
        default=2,
        ge=0,
        validation_alias="UPSTREAM_MAX_RETRIES",
    )
    # A completed turn with no content and no tool call is a failed generation,
    # not an answer, so it is worth one more attempt. Buffered requests only: a
    # stream is only known to be empty once its bytes have reached the client.
    empty_response_retries: int = Field(
        default=1,
        ge=0,
        validation_alias="EMPTY_RESPONSE_RETRIES",
    )
    max_concurrent_upstream: int = Field(
        default=8,
        ge=0,
        validation_alias="MAX_CONCURRENT_UPSTREAM",
    )
    stream_guard_chars: int = Field(default=192, ge=1)
    capture_stream_dir: str = Field(default="", validation_alias="CAPTURE_STREAM_DIR")
    capture_stream_max_bytes: int = Field(
        default=8_388_608,
        ge=0,
        validation_alias="CAPTURE_STREAM_MAX_BYTES",
    )
    capture_stream_include_request: bool = Field(
        default=False,
        validation_alias="CAPTURE_STREAM_INCLUDE_REQUEST",
    )
    tool_argument_chunk_size: int = Field(default=64, ge=1)
    max_raw_tool_block_chars: int = Field(default=131_072, ge=1)
    max_tool_calls: int = Field(default=32, ge=1)
    max_tool_argument_chars: int = Field(default=262_144, ge=1)
    tool_call_scan_fields: str = Field(
        default="content,reasoning,reasoning_content",
        validation_alias="TOOL_CALL_SCAN_FIELDS",
    )
    sanitize_tools: bool = Field(default=True, validation_alias="SANITIZE_TOOLS")
    # Repairs message shapes a DeepSeek-compatible upstream rejects outright
    # (null assistant content, stale reasoning replay, empty tool results).
    normalize_requests: bool = Field(default=True, validation_alias="NORMALIZE_REQUESTS")
    request_drop_fields: str = Field(default="", validation_alias="REQUEST_DROP_FIELDS")
    custom_headers: str = Field(
        default="",
        validation_alias=AliasChoices("CUSTOM_HEADERS", "UPSTREAM_HEADERS"),
    )
    model_aliases: str = Field(default="", validation_alias="MODEL_ALIASES")
    model_compatibility: str = ""
    alias_conflict_policy: str = Field(default="skip", validation_alias="ALIAS_CONFLICT_POLICY")
    config_file: str = Field(default="", validation_alias="PROXY_CONFIG_FILE")
    modality_routes: str = Field(default="", validation_alias="MODALITY_ROUTES")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Init kwargs are normalized to the field's preferred alias, so accept both.
        init_kwargs = getattr(init_settings, "init_kwargs", {})
        path: object = None
        if isinstance(init_kwargs, dict):
            path = init_kwargs.get("config_file") or init_kwargs.get(CONFIG_FILE_ENV)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            ConfigFileSettingsSource(
                settings_cls, str(path or os.environ.get(CONFIG_FILE_ENV, ""))
            ),
            file_secret_settings,
        )

    @field_validator("upstream_url")
    @classmethod
    def normalize_upstream_url(cls, value: str) -> str:
        return strip_upstream_v1_suffix(value)

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

    @field_validator("model_aliases", "model_compatibility", "modality_routes", mode="before")
    @classmethod
    def serialize_mapping_values(cls, value: object) -> object:
        """Accept mappings from the YAML config file, not just env strings."""
        if isinstance(value, dict):
            return json.dumps(value)
        return value

    @field_validator("model_compatibility")
    @classmethod
    def validate_model_compatibility(cls, value: str) -> str:
        parse_model_compatibility(value)
        return value

    @field_validator("modality_routes")
    @classmethod
    def validate_modality_routes(cls, value: str) -> str:
        parse_modality_routes(value, normalize_upstream=strip_upstream_v1_suffix)
        return value

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
    def upstream_health_url(self) -> str:
        """Absolute URL of the optional engine-liveness probe, or empty if unset."""
        path = self.upstream_health_path.strip()
        if not path:
            return ""
        return f"{self.upstream_base_url}/{path.lstrip('/')}"

    @property
    def upstream_safe_origin(self) -> str:
        return safe_origin(self.upstream_base_url)

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

    @cached_property
    def parsed_model_compatibility(self) -> dict[str, ModelCompatibility]:
        return parse_model_compatibility(self.model_compatibility)

    @cached_property
    def parsed_modality_routes(self) -> dict[str, ModalityRoute]:
        return parse_modality_routes(
            self.modality_routes,
            normalize_upstream=strip_upstream_v1_suffix,
        )

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
                    "ready": self.upstream_ready_timeout,
                    "stream_idle": (
                        None
                        if self.upstream_stream_idle_timeout == 0
                        else self.upstream_stream_idle_timeout
                    ),
                    "stream_first_frame": (
                        None
                        if self.upstream_stream_first_frame_timeout == 0
                        else self.upstream_stream_first_frame_timeout
                    ),
                },
                "max_concurrent": self.max_concurrent_upstream or None,
                "max_retries": self.upstream_max_retries,
                "empty_response_retries": self.empty_response_retries,
                "health_path": self.upstream_health_path or None,
            },
            "streaming": {
                "guard_chars": self.stream_guard_chars,
                "tool_argument_chunk_size": self.tool_argument_chunk_size,
                "keepalive_interval": self.sse_keepalive_interval or None,
                "empty_turn_notice": bool(self.empty_turn_notice),
            },
            "capture": {
                "enabled": bool(self.capture_stream_dir),
                "max_bytes": self.capture_stream_max_bytes or None,
                "include_request": self.capture_stream_include_request,
            },
            "tool_call_repair": {
                "scan_fields": list(self.parsed_tool_call_scan_fields),
                "max_raw_tool_block_chars": self.max_raw_tool_block_chars,
                "max_tool_calls": self.max_tool_calls,
                "max_tool_argument_chars": self.max_tool_argument_chars,
            },
            "request_transforms": {
                "sanitize_tools": self.sanitize_tools,
                "normalize_requests": self.normalize_requests,
                "drop_fields": list(self.parsed_request_drop_fields),
            },
            "model_aliases": {
                "aliases": sorted(self.parsed_model_aliases),
                "conflict_policy": self.alias_conflict_policy,
            },
            "model_compatibility": {
                model: {
                    "profile": profile.profile,
                    "recover_orphan_invokes": profile.recover_orphan_invokes,
                    "thinking_transport": profile.thinking_transport,
                }
                for model, profile in sorted(self.parsed_model_compatibility.items())
            },
            "modality_routes": {
                modality: {
                    "origin": safe_origin(route.upstream),
                    "model": route.model,
                    "header_names": sorted(route.headers),
                }
                for modality, route in sorted(self.parsed_modality_routes.items())
            },
            "config_file": self.config_file or None,
            "custom_headers": {
                "names": sorted(self.parsed_custom_headers),
            },
            "ollama": {
                "version": self.ollama_version,
            },
        }


def safe_origin(base_url: str) -> str:
    """Return ``scheme://host[:port]`` so logs and /healthz never leak credentials."""
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.hostname:
        return base_url
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return f"{parsed.scheme}://{host}"


def strip_upstream_v1_suffix(url: str) -> str:
    """Strip a trailing ``/v1`` path so callers can paste OpenAI base URLs safely."""
    cleaned = url.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        return cleaned

    new_path = path[: -len("/v1")]
    normalized = urlunsplit(
        (parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment)
    ).rstrip("/")
    LOG.warning(
        "stripped trailing /v1 from an upstream URL; using %s "
        "(OpenAI paths are appended by the proxy)",
        normalized or cleaned,
    )
    return normalized


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


def parse_model_compatibility(raw: str | dict[str, object]) -> dict[str, ModelCompatibility]:
    if not raw:
        return {}
    parsed: object = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        msg = "model compatibility must be a JSON object"
        raise ValueError(msg)

    profiles: dict[str, ModelCompatibility] = {}
    for raw_model, raw_entry in parsed.items():
        model = str(raw_model).strip()
        if not model or not isinstance(raw_entry, dict):
            msg = "model compatibility entries must map model names to objects"
            raise ValueError(msg)
        profile = raw_entry.get("compatibility")
        recover = raw_entry.get("recover_orphan_invokes", False)
        transport = raw_entry.get("thinking_transport", DEFAULT_THINKING_TRANSPORT)
        if profile != "deepseek_v4":
            msg = f"unsupported compatibility profile for model {model!r}"
            raise ValueError(msg)
        if not isinstance(recover, bool):
            msg = f"recover_orphan_invokes for model {model!r} must be a boolean"
            raise ValueError(msg)
        if transport not in THINKING_TRANSPORTS:
            allowed = ", ".join(sorted(THINKING_TRANSPORTS))
            msg = f"thinking_transport for model {model!r} must be one of: {allowed}"
            raise ValueError(msg)
        profiles[model] = ModelCompatibility(
            profile=profile,
            recover_orphan_invokes=recover,
            thinking_transport=transport,
        )
    return profiles


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
