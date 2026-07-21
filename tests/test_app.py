from __future__ import annotations

from fastapi.testclient import TestClient

from opencode_proxy.app import create_app
from opencode_proxy.settings import Settings


def test_healthz() -> None:
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_config_exposes_safe_config() -> None:
    app = create_app(
        Settings(
            upstream_url="http://user:pass@upstream.test:4000/v1",
            custom_headers='{"Authorization":"Bearer secret"}',
            model_aliases='{"alias":"target"}',
        )
    )
    client = TestClient(app)

    response = client.get("/healthz/config")

    assert response.status_code == 200
    body = response.json()
    assert body["upstream"]["origin"] == "http://upstream.test:4000"
    assert body["custom_headers"] == {"names": ["Authorization"]}
    assert body["model_aliases"]["aliases"] == ["alias"]
    assert "secret" not in response.text


def test_app_lifespan_closes_shared_upstream_client() -> None:
    app = create_app(Settings(upstream_url="http://upstream.test"))
    upstream_client = app.state.upstream_client

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert app.state.upstream_client is upstream_client
        assert upstream_client.is_closed is False

    assert upstream_client.is_closed is True
