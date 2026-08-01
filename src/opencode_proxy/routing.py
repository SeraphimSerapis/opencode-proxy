"""Request modality detection and per-modality upstream routing.

Text-only models reject image or audio parts outright, so a deployment can point
those requests at a second multimodal host while normal chat traffic keeps going
to the primary model. This module stays free of FastAPI and settings imports so
detection and routing can be unit tested directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opencode_proxy.settings import Settings

LOG = logging.getLogger(__name__)

VISION = "vision"
AUDIO = "audio"

VISION_PART_TYPES = frozenset({"image_url", "image", "input_image"})
AUDIO_PART_TYPES = frozenset({"input_audio", "audio", "audio_url"})

#: Order used when a request carries several modalities. Audio wins because
#: audio-capable endpoints are the scarcer capability and usually accept images
#: as well.
MODALITY_PRIORITY = (AUDIO, VISION)


@dataclass(frozen=True)
class ModalityRoute:
    """An alternate upstream for requests carrying a given modality."""

    modality: str
    upstream: str
    model: str | None = None
    api_key: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UpstreamTarget:
    """Where a single request should be sent, after routing."""

    base_url: str
    api_key: str = ""
    extra_headers: tuple[tuple[str, str], ...] = ()
    model: str | None = None
    modality: str = ""


def detect_modalities(body: Mapping[str, Any]) -> frozenset[str]:
    """Return the non-text modalities present in an OpenAI chat request body."""
    found: set[str] = set()
    messages = body.get("messages")
    if not isinstance(messages, list):
        return frozenset(found)

    for message in messages:
        if not isinstance(message, dict):
            continue
        # Ollama-shaped messages that reached us unconverted.
        if message.get("images"):
            found.add(VISION)
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in VISION_PART_TYPES:
                found.add(VISION)
            elif part_type in AUDIO_PART_TYPES:
                found.add(AUDIO)
    return frozenset(found)


def default_upstream_target(settings: Settings) -> UpstreamTarget:
    return UpstreamTarget(
        base_url=settings.upstream_base_url,
        api_key=settings.upstream_api_key,
    )


def resolve_upstream_target(settings: Settings, body: Mapping[str, Any]) -> UpstreamTarget:
    """Pick the upstream for ``body``, honouring configured modality routes."""
    modalities = detect_modalities(body)
    if not modalities:
        return default_upstream_target(settings)

    routes = settings.parsed_modality_routes
    for modality in MODALITY_PRIORITY:
        if modality not in modalities:
            continue
        route = routes.get(modality)
        if route is None:
            LOG.warning(
                "request carries %s content but no %s route is configured; "
                "forwarding to the primary upstream",
                modality,
                modality,
            )
            continue
        LOG.info(
            "routing %s request to the %s upstream%s",
            modality,
            modality,
            f" as model {route.model!r}" if route.model else "",
        )
        return UpstreamTarget(
            base_url=route.upstream,
            api_key=route.api_key,
            extra_headers=tuple(route.headers.items()),
            model=route.model,
            modality=modality,
        )
    return default_upstream_target(settings)


def parse_modality_routes(
    value: object,
    *,
    normalize_upstream: object = None,
) -> dict[str, ModalityRoute]:
    """Parse ``MODALITY_ROUTES`` JSON (or an already-parsed mapping) into routes.

    ``normalize_upstream`` is the settings helper that strips a trailing ``/v1``;
    it is injected to keep this module independent of the settings module.
    """
    raw = _as_mapping(value)
    if not raw:
        return {}

    routes: dict[str, ModalityRoute] = {}
    for raw_modality, raw_route in raw.items():
        modality = str(raw_modality).strip().lower()
        if modality not in {VISION, AUDIO}:
            msg = f"Unsupported MODALITY_ROUTES modality: {raw_modality!r}"
            raise ValueError(msg)
        if isinstance(raw_route, str):
            raw_route = {"upstream": raw_route}
        if not isinstance(raw_route, dict):
            msg = f"MODALITY_ROUTES entry for {modality!r} must be a mapping"
            raise ValueError(msg)

        unknown = set(raw_route) - {"upstream", "model", "api_key", "headers"}
        if unknown:
            msg = f"Unsupported MODALITY_ROUTES keys for {modality!r}: {sorted(unknown)}"
            raise ValueError(msg)

        upstream = str(raw_route.get("upstream", "")).strip()
        if not upstream:
            msg = f"MODALITY_ROUTES entry for {modality!r} requires an upstream"
            raise ValueError(msg)
        if callable(normalize_upstream):
            upstream = str(normalize_upstream(upstream))

        raw_headers = raw_route.get("headers") or {}
        if not isinstance(raw_headers, dict):
            msg = f"MODALITY_ROUTES headers for {modality!r} must be a mapping"
            raise ValueError(msg)

        model = raw_route.get("model")
        routes[modality] = ModalityRoute(
            modality=modality,
            upstream=upstream.rstrip("/"),
            model=str(model).strip() if model else None,
            api_key=str(raw_route.get("api_key") or ""),
            headers={str(name): str(header) for name, header in raw_headers.items()},
        )
    return routes


def _as_mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            msg = "MODALITY_ROUTES JSON must be an object"
            raise ValueError(msg)
        return {str(key): item for key, item in parsed.items()}
    msg = "MODALITY_ROUTES must be a JSON object or mapping"
    raise ValueError(msg)
