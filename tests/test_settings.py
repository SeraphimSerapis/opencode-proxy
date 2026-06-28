from __future__ import annotations

import pytest

from opencode_proxy.settings import Settings, parse_custom_headers


def test_parse_custom_headers_from_json_object() -> None:
    headers = parse_custom_headers('{"Authorization":"Bearer token","X-Skip-Auth":"true"}')

    assert headers == {"Authorization": "Bearer token", "X-Skip-Auth": "true"}


def test_parse_custom_headers_from_lines() -> None:
    headers = parse_custom_headers(
        """
        Authorization: Bearer token
        X-Skip-Auth: true
        """,
    )

    assert headers == {"Authorization": "Bearer token", "X-Skip-Auth": "true"}


def test_parse_custom_headers_rejects_invalid_line() -> None:
    with pytest.raises(ValueError, match="Invalid CUSTOM_HEADERS line"):
        parse_custom_headers("Authorization")


def test_settings_reads_custom_headers_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_HEADERS", '{"X-Test":"yes"}')

    settings = Settings()

    assert settings.parsed_custom_headers == {"X-Test": "yes"}
