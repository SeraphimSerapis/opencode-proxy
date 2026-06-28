# OpenCode Proxy

FastAPI compatibility proxy for running OpenCode against an OpenAI-compatible LiteLLM router when a model emits tool calls as raw text instead of standard `tool_calls` JSON.

```text
OpenCode CLI -> opencode-proxy -> LiteLLM router -> model backend
```

The proxy passes normal OpenAI-compatible traffic through unchanged and repairs known malformed assistant tool-call formats in `/v1/chat/completions` responses.

## Supported Repairs

- DeepSeek DSML `<｜DSML｜tool_calls>` blocks with `<name>` / `<parameters>`.
- DeepSeek DSML invoke blocks such as `<｜DSML｜invoke name="...">`.
- ASCII DSML variants such as `<|DSML|tool_calls>`.
- Qwen-style `<tool_call>` XML blocks.
- Spurious empty streamed `tool_calls: []` chunks from some OpenAI-compatible servers.

Native OpenAI `tool_calls` are passed through unchanged.

## Local Development

```bash
uv sync --dev
uv run uvicorn opencode_proxy.app:create_app --factory --host 0.0.0.0 --port 9526
```

By default the proxy forwards to `http://127.0.0.1:4000`, which is LiteLLM's common local port. Override it with:

```bash
UPSTREAM_URL=http://127.0.0.1:4000 uv run opencode-proxy
```

## OpenCode Provider Example

Point OpenCode at the proxy, not directly at LiteLLM:

```jsonc
{
  "provider": {
    "litellm-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM via OpenCode Proxy",
      "options": {
        "baseURL": "http://127.0.0.1:9526/v1",
        "apiKey": "dummy"
      },
      "models": {
        "your-model": {
          "name": "your-model"
        }
      }
    }
  }
}
```

## Docker

```bash
docker build -t opencode-proxy:local .
docker run --rm -p 9526:9526 \
  -e UPSTREAM_URL=http://host.docker.internal:4000 \
  -e 'CUSTOM_HEADERS={"X-Skip-Auth":"true"}' \
  opencode-proxy:local
```

When this repository is pushed to GitHub, the publish workflow builds:

```bash
docker pull ghcr.io/seraphimserapis/opencode-proxy:latest
docker run --rm -p 9526:9526 \
  -e UPSTREAM_URL=http://host.docker.internal:4000 \
  -e 'CUSTOM_HEADERS={"X-Skip-Auth":"true"}' \
  ghcr.io/seraphimserapis/opencode-proxy:latest
```

## Validation

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

CI also runs a Docker build smoke test.

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `UPSTREAM_URL` | `http://127.0.0.1:4000` | Upstream LiteLLM/OpenAI-compatible base URL. |
| `PROXY_HOST` | `0.0.0.0` | Bind host for `opencode-proxy`. |
| `PROXY_PORT` | `9526` | Bind port for `opencode-proxy`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `STREAM_GUARD_CHARS` | `192` | Text held back while detecting split raw tool-call tags. |
| `TOOL_ARGUMENT_CHUNK_SIZE` | `64` | Size for streamed function argument deltas. |
| `CUSTOM_HEADERS` | unset | Extra headers added to upstream requests. Overrides forwarded client headers. |
| `UPSTREAM_HEADERS` | unset | Alias for `CUSTOM_HEADERS`. |

`CUSTOM_HEADERS` accepts a JSON object:

```bash
CUSTOM_HEADERS='{"Authorization":"Bearer local-dev-token","X-Skip-Auth":"true"}'
```

It also accepts newline-separated `Header: value` pairs, which is useful in `.env` files:

```dotenv
CUSTOM_HEADERS="Authorization: Bearer local-dev-token
X-Skip-Auth: true"
```

Hop-by-hop headers such as `Connection` and `Content-Length` are ignored. For streaming requests, `Accept-Encoding` is also ignored so SSE can be parsed safely.

## API Surface

- `GET /healthz`: local proxy health check.
- `/v1/chat/completions`: proxied to LiteLLM with request tool sanitization and response tool-call repair.
- `/{path:path}`: transparent passthrough for other OpenAI-compatible endpoints such as `/v1/models`.

## Notes

- Set `UPSTREAM_URL` to the LiteLLM router base URL, not the `/v1` path.
- The proxy strips compressed SSE request headers so streamed responses can be parsed line by line.
- If an upstream response already contains standard OpenAI `tool_calls`, it is passed through unchanged.
