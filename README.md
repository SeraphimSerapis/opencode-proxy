# OpenCode Proxy

FastAPI compatibility proxy for running OpenCode against an OpenAI-compatible upstream such as LiteLLM, llama.cpp, or vLLM when a model emits tool calls as raw text instead of standard `tool_calls` JSON.

```text
OpenCode CLI -> opencode-proxy -> OpenAI-compatible upstream -> model backend
Ollama clients -> opencode-proxy -> OpenAI-compatible upstream -> model backend
```

The proxy passes normal OpenAI-compatible traffic through unchanged and repairs known malformed assistant tool-call formats in `/v1/chat/completions` responses. The same process also exposes an Ollama-compatible REST adapter, so OpenCode, Home Assistant, and Ollama clients can share one gateway.

## Supported Repairs

- DeepSeek DSML `<｜DSML｜tool_calls>` blocks with `<name>` / `<parameters>`.
- DeepSeek DSML invoke blocks such as `<｜DSML｜invoke name="...">`.
- ASCII DSML variants such as `<|DSML|tool_calls>`.
- Qwen-style `<tool_call>` XML blocks.
- Qwen-style JSON objects inside `<tool_call>` blocks.
- Poolside / Laguna S 2.1 `<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` blocks.
- Spurious empty streamed `tool_calls: []` chunks from some OpenAI-compatible servers.

Native OpenAI `tool_calls` are passed through unchanged.

The proxy scans `content`, `reasoning`, and `reasoning_content` by default. If a
raw tool-call block is found in a reasoning field, only that raw block is
converted; surrounding reasoning text stays in the original reasoning field.

Streaming responses are parsed as SSE frames, including comments and multiline
`data:` events. If an upstream stream ends without `[DONE]`, pending text is
flushed and the proxy emits a single final `[DONE]`.

## Local Development

```bash
uv sync --dev
uv run uvicorn opencode_proxy.app:create_app --factory --host 0.0.0.0 --port 9526
```

By default the proxy forwards to `http://127.0.0.1:4000`, which is LiteLLM's common local port. Point it at any OpenAI-compatible upstream with:

```bash
UPSTREAM_URL=http://127.0.0.1:4000 uv run opencode-proxy
```

## OpenCode Provider Example

Point OpenCode at the proxy, not directly at the upstream:

```jsonc
{
  "provider": {
    "opencode-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenCode Proxy",
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

## Ollama-compatible API

The unified service implements the commonly used Ollama endpoints:

- `GET /` and `GET /api/version` for client discovery.
- `POST /api/chat` and `POST /api/generate`, including NDJSON streaming, vision images, thinking fields, and tool calls.
- `GET /api/tags`, `GET /api/ps`, and `POST /api/show` for model discovery and synthetic metadata.
- `POST /api/embed` and the deprecated `POST /api/embeddings`.
- Model-management and blob endpoints are safe no-ops because model lifecycle remains upstream.

Ollama clients normally target `http://127.0.0.1:9526` when running the container
directly. In Docker Compose, map both host ports (`11434:9526` and
`9526:9526`) to keep the standard Ollama port and the OpenCode port on one
process.

The adapter forwards incoming `Authorization` headers to LiteLLM. Set
`UPSTREAM_API_KEY` only for clients that cannot provide a key; it is used as a
fallback and never replaces a caller-provided key. This makes LiteLLM virtual
keys and per-user budgets available without running a second proxy.

LiteLLM can also host the repair logic directly through a custom callback using
its `async_post_call_success_hook` and
`async_post_call_streaming_iterator_hook` hooks. The unified adapter remains the
recommended path for Ollama clients because it owns the Ollama request/response
shape while LiteLLM continues to own authentication, routing, budgets, and
spend tracking.

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

For a token-free integration smoke test, run the dependency-free mock upstream
from this repository in a second terminal and point the proxy at it:

```bash
python tools/mock_openai.py
UPSTREAM_URL=http://127.0.0.1:4000 uv run opencode-proxy
curl http://127.0.0.1:9526/api/tags
curl -N http://127.0.0.1:9526/api/chat \
  -H 'content-type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hello"}]}'
```

On Prometheus, the compose stack should include only the unified service. Build
it from `${PROJECTSDIR}/opencode-proxy`, set `UPSTREAM_URL=http://litellm:4000`,
and map both `11434:9526` and `9526:9526` to that one container. Set
`UPSTREAM_API_KEY=${LITELLM_MASTER_KEY}` as a fallback for clients that do not
send a LiteLLM virtual key; caller `Authorization` headers remain per-request.

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `UPSTREAM_URL` | `http://127.0.0.1:4000` | Upstream OpenAI-compatible base URL. |
| `UPSTREAM_API_KEY` | unset | Fallback upstream bearer token. Incoming `Authorization` headers take precedence. `OLLAMA_PROXY_UPSTREAM_URL` and `OLLAMA_PROXY_UPSTREAM_API_KEY` remain accepted aliases for existing Ollama deployments. |
| `PROXY_HOST` | `0.0.0.0` | Bind host for `opencode-proxy`. |
| `PROXY_PORT` | `9526` | Bind port for `opencode-proxy`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | Upstream connect timeout in seconds. |
| `UPSTREAM_READ_TIMEOUT` | `0` | Upstream read timeout in seconds. `0` disables read timeout for long streams. |
| `UPSTREAM_WRITE_TIMEOUT` | `30` | Upstream write timeout in seconds. |
| `UPSTREAM_POOL_TIMEOUT` | `30` | Upstream connection-pool timeout in seconds. |
| `STREAM_GUARD_CHARS` | `192` | Text held back while detecting split raw tool-call tags. |
| `TOOL_ARGUMENT_CHUNK_SIZE` | `64` | Size for streamed function argument deltas. |
| `MAX_RAW_TOOL_BLOCK_CHARS` | `131072` | Maximum raw tool-call block size to convert. Larger blocks pass through as text. |
| `MAX_TOOL_CALLS` | `32` | Maximum tool calls to convert from one raw block. Blocks over the limit pass through as text. |
| `MAX_TOOL_ARGUMENT_CHARS` | `262144` | Maximum serialized argument size per converted tool call. Blocks over the limit pass through as text. |
| `TOOL_CALL_SCAN_FIELDS` | `content,reasoning,reasoning_content` | Comma-separated response fields scanned for raw tool-call blocks. Use `all` for all supported fields. |
| `SANITIZE_TOOLS` | `true` | Drop non-function tools from chat completion requests for OpenCode/upstream compatibility. |
| `REQUEST_DROP_FIELDS` | unset | Comma-separated request body fields to remove before forwarding, for backend-specific quirks. |
| `CUSTOM_HEADERS` | unset | Extra headers added to upstream requests. Overrides forwarded client headers. |
| `UPSTREAM_HEADERS` | unset | Alias for `CUSTOM_HEADERS`. |
| `MODEL_ALIASES` | unset | Model alias map. Request aliases are rewritten to canonical upstream model names. |
| `ALIAS_CONFLICT_POLICY` | `skip` | Model discovery behavior when an alias conflicts with an upstream model id: `skip`, `shadow`, or `error`. |
| `OLLAMA_VERSION` | `0.5.1` | Version reported by `GET /api/version`. |

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

`MODEL_ALIASES` accepts comma-separated `alias=target` pairs, which is usually
the simplest form for Docker Compose and `.env` files:

```bash
MODEL_ALIASES=dsv4-flash=DeepSeek-V4-Flash,deepseek-ai/DeepSeek-V4-Flash-DSpark=DeepSeek-V4-Flash
```

It also accepts newline-separated pairs and JSON object syntax:

```dotenv
MODEL_ALIASES="dsv4-flash=DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Flash-DSpark=DeepSeek-V4-Flash"
```

```bash
MODEL_ALIASES='{"dsv4-flash":"DeepSeek-V4-Flash","deepseek-ai/DeepSeek-V4-Flash-DSpark":"DeepSeek-V4-Flash"}'
```

With these aliases, `/v1/chat/completions` requests for `dsv4-flash` are sent upstream as `DeepSeek-V4-Flash`. `/v1/models` and `/models` also include alias entries so clients can discover them.
On startup, the proxy logs configured alias names. You can also check
`/healthz/config` and confirm `model_aliases.aliases` contains `dsv4-flash`.

If an alias conflicts with a model already returned by upstream discovery,
`ALIAS_CONFLICT_POLICY=skip` keeps the upstream entry, `shadow` replaces the
discovery entry with the alias target metadata, and `error` returns `409`.

## API Surface

- `GET /healthz`: local proxy health check.
- `GET /healthz/config`: safe local config summary, with header values and URL credentials omitted.
- `/v1/chat/completions`: proxied to the upstream with request tool sanitization and response tool-call repair.
- `/v1/models` and `/models`: upstream model discovery with configured alias entries added.
- `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`: Ollama-compatible translations backed by the same upstream client and tool-call repair pipeline.
- `/api/tags`, `/api/ps`, `/api/show`, `/api/version`: Ollama discovery and metadata endpoints.
- `/{path:path}`: transparent passthrough for other OpenAI-compatible endpoints.

## Notes

- Set `UPSTREAM_URL` to the upstream base URL, not the `/v1` path.
- The proxy strips compressed SSE request headers so streamed responses can be parsed line by line.
- If an upstream response already contains standard OpenAI `tool_calls`, it is passed through unchanged.
- `reasoning_content` and `reasoning` fields (DeepSeek R1 / o1-style streaming) are scanned for raw tool-call blocks by default, but ordinary reasoning text remains in reasoning fields.
- Because scanned text is buffered, reasoning deltas from the same upstream event may be emitted before that event's content delta.
- The Docker image runs as a non-root user and includes a `/healthz` healthcheck.
