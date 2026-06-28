from __future__ import annotations

from fastapi.testclient import TestClient

from opencode_proxy.app import create_app


def test_healthz() -> None:
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
