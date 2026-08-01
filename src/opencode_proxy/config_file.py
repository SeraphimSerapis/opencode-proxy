"""Optional YAML config file for model aliases and modality routes.

Environment variables stay authoritative for deployment wiring; this file exists
so the list-shaped configuration (aliases, alternate upstreams) can be written as
YAML instead of packed into one-line env strings. Anything set in the environment
wins over the same key here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger(__name__)

SUPPORTED_SECTIONS = frozenset({"models", "routes"})


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read ``path`` and return values shaped like the matching settings fields.

    Returns an empty mapping when no path is configured. A configured but
    unreadable or malformed file is an error: silently running with half the
    intended routing would be worse than refusing to start.
    """
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.is_file():
        msg = f"PROXY_CONFIG_FILE does not exist: {config_path}"
        raise ValueError(msg)

    with config_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if document is None:
        return {}
    if not isinstance(document, dict):
        msg = f"{config_path} must contain a YAML mapping at the top level"
        raise ValueError(msg)

    unknown = set(document) - SUPPORTED_SECTIONS
    if unknown:
        msg = f"{config_path} has unsupported section(s): {sorted(unknown)}"
        raise ValueError(msg)

    values: dict[str, Any] = {}
    aliases = _parse_models_section(document.get("models"), config_path)
    if aliases:
        values["model_aliases"] = aliases
    routes = document.get("routes")
    if routes:
        values["modality_routes"] = routes

    LOG.info(
        "loaded %s: %d alias(es), %d modality route(s)",
        config_path,
        len(aliases),
        len(routes) if isinstance(routes, dict) else 0,
    )
    return values


def _parse_models_section(models: object, config_path: Path) -> dict[str, str]:
    """Flatten ``models: {target: {aliases: [...]}}`` into ``{alias: target}``."""
    if models is None:
        return {}
    if not isinstance(models, dict):
        msg = f"{config_path}: 'models' must be a mapping of upstream model to settings"
        raise ValueError(msg)

    aliases: dict[str, str] = {}
    for raw_target, entry in models.items():
        target = str(raw_target).strip()
        if not target:
            msg = f"{config_path}: 'models' has an empty model name"
            raise ValueError(msg)

        if isinstance(entry, list):
            raw_aliases: list[Any] = entry
        elif isinstance(entry, dict):
            unknown = set(entry) - {"aliases"}
            if unknown:
                msg = f"{config_path}: unsupported keys for model {target!r}: {sorted(unknown)}"
                raise ValueError(msg)
            raw_aliases = entry.get("aliases") or []
            if not isinstance(raw_aliases, list):
                msg = f"{config_path}: 'aliases' for model {target!r} must be a list"
                raise ValueError(msg)
        elif entry is None:
            raw_aliases = []
        else:
            msg = f"{config_path}: model {target!r} must map to a list or a mapping"
            raise ValueError(msg)

        for raw_alias in raw_aliases:
            alias = str(raw_alias).strip()
            if not alias:
                msg = f"{config_path}: model {target!r} has an empty alias"
                raise ValueError(msg)
            if alias in aliases and aliases[alias] != target:
                msg = (
                    f"{config_path}: alias {alias!r} is claimed by both "
                    f"{aliases[alias]!r} and {target!r}"
                )
                raise ValueError(msg)
            aliases[alias] = target
    return aliases
