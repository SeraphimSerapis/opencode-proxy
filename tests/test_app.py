from __future__ import annotations

import asyncio

import httpx
import respx
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from opencode_proxy.app import create_app
from opencode_proxy.concurrency import UpstreamConcurrencyLimiter
from opencode_proxy.settings import Settings, strip_upstream_v1_suffix


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
            max_concurrent_upstream=8,
        )
    )
    client = TestClient(app)

    response = client.get("/healthz/config")

    assert response.status_code == 200
    body = response.json()
    assert body["upstream"]["origin"] == "http://upstream.test:4000"
    assert body["upstream"]["max_concurrent"] == 8
    assert body["custom_headers"] == {"names": ["Authorization"]}
    assert body["model_aliases"]["aliases"] == ["alias"]
    assert "secret" not in response.text


def test_strip_upstream_v1_suffix() -> None:
    assert strip_upstream_v1_suffix("http://127.0.0.1:4000/v1") == "http://127.0.0.1:4000"
    assert strip_upstream_v1_suffix("http://127.0.0.1:4000/v1/") == "http://127.0.0.1:4000"
    assert strip_upstream_v1_suffix("http://litellm:4000/openai/v1") == "http://litellm:4000/openai"
    assert strip_upstream_v1_suffix("http://127.0.0.1:4000") == "http://127.0.0.1:4000"


def test_settings_normalizes_upstream_v1_suffix() -> None:
    settings = Settings(upstream_url="http://upstream.test:4000/v1/")
    assert settings.upstream_base_url == "http://upstream.test:4000"


def test_app_lifespan_closes_shared_upstream_client() -> None:
    app = create_app(Settings(upstream_url="http://upstream.test"))
    upstream_client = app.state.upstream_client

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert app.state.upstream_client is upstream_client
        assert upstream_client.is_closed is False

    assert upstream_client.is_closed is True


MODEL_LIST = {"data": [{"id": "deepseek-v4-flash", "object": "model"}]}


@respx.mock
async def test_readyz_ok_when_upstream_reachable() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json=MODEL_LIST)
    )
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "upstream_status": 200}


@respx.mock
async def test_readyz_not_ready_when_upstream_serves_no_models() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["upstream"] == "no_models"


@respx.mock
async def test_readyz_not_ready_on_upstream_4xx_that_is_not_auth() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["upstream"] == "unexpected_status"
    assert body["upstream_status"] == 404


@respx.mock
async def test_readyz_not_ready_when_engine_health_path_fails() -> None:
    health = respx.get("http://upstream.test/health").mock(
        return_value=httpx.Response(503, text="engine dead")
    )
    models = respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json=MODEL_LIST)
    )
    app = create_app(Settings(upstream_url="http://upstream.test", upstream_health_path="/health"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["upstream"] == "engine_unhealthy"
    assert body["upstream_status"] == 503
    assert health.called
    # A dead engine short-circuits: the static model list proves nothing here.
    assert not models.called


@respx.mock
async def test_readyz_ok_when_engine_health_path_passes() -> None:
    respx.get("http://upstream.test/health").mock(return_value=httpx.Response(200))
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json=MODEL_LIST)
    )
    app = create_app(Settings(upstream_url="http://upstream.test", upstream_health_path="/health"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "upstream_status": 200}


@respx.mock
async def test_readyz_failure_is_counted_in_metrics() -> None:
    respx.get("http://upstream.test/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        assert (await client.get("/readyz")).status_code == 503
        metrics = (await client.get("/metrics")).text

    assert 'opencode_proxy_upstream_ready_failures_total{reason="unreachable"} 1.0' in metrics


@respx.mock
async def test_readyz_ok_when_upstream_requires_auth() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["upstream_status"] == 401


@respx.mock
async def test_readyz_not_ready_when_upstream_down() -> None:
    respx.get("http://upstream.test/v1/models").mock(side_effect=httpx.ConnectError("refused"))
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "upstream": "unreachable"}


@respx.mock
async def test_readyz_not_ready_on_upstream_5xx() -> None:
    respx.get("http://upstream.test/v1/models").mock(
        return_value=httpx.Response(503, json={"error": "down"})
    )
    app = create_app(Settings(upstream_url="http://upstream.test"))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["upstream_status"] == 503


async def test_concurrency_limiter_rejects_when_full() -> None:
    limiter = UpstreamConcurrencyLimiter(1)
    assert await limiter.try_acquire() is True
    assert await limiter.try_acquire() is False
    await limiter.release()
    assert await limiter.try_acquire() is True


@respx.mock
async def test_chat_completions_returns_429_when_concurrency_exhausted() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    respx.post("http://upstream.test/v1/chat/completions").mock(side_effect=handler)
    app = create_app(
        Settings(upstream_url="http://upstream.test", max_concurrent_upstream=1),
    )
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        first = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
        )
        await started.wait()
        second = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        metrics_while_active = (await client.get("/metrics")).text
        release.set()
        first_response = await first
        metrics = (await client.get("/metrics")).text

    assert first_response.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["type"] == "proxy_overload"
    assert "opencode_proxy_upstream_active 1.0" in metrics_while_active
    assert "opencode_proxy_upstream_overloads_total 1.0" in metrics
    assert "opencode_proxy_upstream_active 0.0" in metrics
    assert second.headers["retry-after"] == "1"
