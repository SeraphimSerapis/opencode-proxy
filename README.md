# OpenCode Proxy

FastAPI compatibility proxy for running OpenCode against an OpenAI-compatible upstream such as LiteLLM, llama.cpp, or vLLM when a model emits tool calls as raw text instead of standard `tool_calls` JSON.

```text
OpenCode CLI -> opencode-proxy -> OpenAI-compatible upstream -> model backend
Ollama clients -> opencode-proxy -> OpenAI-compatible upstream -> model backend
```

The proxy passes normal OpenAI-compatible traffic through unchanged and repairs known malformed assistant tool-call formats in `/v1/chat/completions` responses. The same process also exposes an Ollama-compatible REST adapter, so OpenCode, Home Assistant, and Ollama clients can share one gateway.

## Documentation

Detailed docs live in [`docs/`](docs/index.md) as an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md) bundle:
[request pipeline](docs/architecture/request-pipeline.md),
[tool-call repair](docs/architecture/tool-call-repair.md),
[streaming contract](docs/architecture/streaming-contract.md),
[configuration](docs/reference/configuration.md),
[API surface](docs/reference/api-surface.md),
[decisions](docs/decisions/index.md), and
[runbooks](docs/runbooks/index.md).

## Supported Repairs

- DeepSeek DSML `<｜DSML｜tool_calls>` blocks with `<name>` / `<parameters>`.
- DeepSeek DSML invoke blocks inside the normal outer wrapper.
- Opt-in DeepSeek V4 recovery for canonical orphan
  `<｜DSML｜invoke name="...">` blocks missing only that outer wrapper.
- ASCII DSML variants such as `<|DSML|tool_calls>`.
- Qwen-style `<tool_call>` XML blocks.
- Qwen-style JSON objects inside `<tool_call>` blocks.
- Poolside / Laguna S 2.1 `<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` blocks.
- Spurious empty streamed `tool_calls: []` chunks from some OpenAI-compatible servers.

Native OpenAI `tool_calls` are passed through unchanged except that otherwise
valid calls missing an `id` receive a stable synthetic ID. In a stream, the same
ID is reused for every fragment of one choice/tool index.

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
        "apiKey": "sk-your-litellm-virtual-key"
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

When no fallback is configured, requests without `Authorization` receive the
upstream LiteLLM authentication error. This is the default in the Prometheus
compose deployment.

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
and map both `11434:9526` and `9526:9526` to that one container. The checked-in
compose definition leaves `UPSTREAM_API_KEY` empty, so LiteLLM virtual keys
gate callers through forwarded `Authorization` headers. Set
`PROXY_UPSTREAM_FALLBACK_KEY` only for trusted clients that cannot send a key;
never use the LiteLLM master key as a general client fallback.

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `UPSTREAM_URL` | `http://127.0.0.1:4000` | Upstream OpenAI-compatible base URL. A trailing `/v1` is stripped automatically. |
| `UPSTREAM_API_KEY` | unset | Fallback upstream bearer token. Incoming `Authorization` headers take precedence. `OLLAMA_PROXY_UPSTREAM_URL` and `OLLAMA_PROXY_UPSTREAM_API_KEY` remain accepted aliases for existing Ollama deployments. |
| `PROXY_HOST` | `0.0.0.0` | Bind host for `opencode-proxy`. |
| `PROXY_PORT` | `9526` | Bind port for `opencode-proxy`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | Upstream connect timeout in seconds. |
| `UPSTREAM_READ_TIMEOUT` | `0` | Upstream read timeout in seconds. `0` disables read timeout for long streams. |
| `UPSTREAM_WRITE_TIMEOUT` | `30` | Upstream write timeout in seconds. |
| `UPSTREAM_POOL_TIMEOUT` | `30` | Upstream connection-pool timeout in seconds. |
| `UPSTREAM_READY_TIMEOUT` | `2` | Timeout in seconds for the `/readyz` upstream probe. |
| `SSE_KEEPALIVE_INTERVAL` | `10` | Seconds of upstream silence before the proxy sends an SSE keepalive comment. `0` disables keepalives. |
| `UPSTREAM_MAX_RETRIES` | `2` | Retries for chat/generate requests that fail before any response byte reaches the client. `0` disables retries. |
| `UPSTREAM_STREAM_IDLE_TIMEOUT` | `120` | Seconds to wait for the next upstream SSE frame before flushing buffers, sending a terminal chunk, and closing the client stream. `0` disables the guard. |
| `MAX_CONCURRENT_UPSTREAM` | `8` | Max concurrent chat/generate requests to upstream. Extra requests get `429` with `Retry-After: 1`. `0` disables the limit. |
| `STREAM_GUARD_CHARS` | `192` | Text held back while detecting split raw tool-call tags. |
| `TOOL_ARGUMENT_CHUNK_SIZE` | `64` | Size for streamed function argument deltas. |
| `EMPTY_TURN_NOTICE` | a short explanatory message | Content emitted before the terminal chunk when an upstream-closed turn has no content or tool calls. Set empty to disable. |
| `MAX_RAW_TOOL_BLOCK_CHARS` | `131072` | Maximum raw tool-call block size to convert. Larger blocks pass through as text. |
| `MAX_TOOL_CALLS` | `32` | Maximum raw calls to convert and standard indexes to track for streamed repair. Blocks over the raw limit pass through as text. |
| `MAX_TOOL_ARGUMENT_CHARS` | `262144` | Maximum serialized argument size per converted call and streamed repair buffer. Larger raw blocks pass through as text; standard calls are not repaired past the bound. |
| `TOOL_CALL_SCAN_FIELDS` | `content,reasoning,reasoning_content` | Comma-separated response fields scanned for raw tool-call blocks. Use `all` for all supported fields. |
| `SANITIZE_TOOLS` | `true` | Drop non-function tools from chat completion requests for OpenCode/upstream compatibility. |
| `REQUEST_DROP_FIELDS` | unset | Comma-separated request body fields to remove before forwarding, for backend-specific quirks. |
| `CUSTOM_HEADERS` | unset | Extra headers added to upstream requests. Overrides forwarded client headers. |
| `UPSTREAM_HEADERS` | unset | Alias for `CUSTOM_HEADERS`. |
| `MODEL_ALIASES` | unset | Model alias map. Request aliases are rewritten to canonical upstream model names. |
| `PROXY_CONFIG_FILE` | unset | Path to an optional YAML file holding model aliases and modality routes. Environment variables win over the file. |
| `MODALITY_ROUTES` | unset | JSON map of `vision`/`audio` to an alternate upstream, so image or audio requests can go to a multimodal host. |
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

### Stream liveness and retries

Two guards keep a streamed turn from stalling silently:

- **Keepalives.** While the upstream is quiet, the proxy emits `: keepalive`
  SSE comments every `SSE_KEEPALIVE_INTERVAL` seconds. Comments are ignored by
  SSE clients but stop reverse proxies and load balancers from dropping an idle
  connection during a long reasoning pause. Streamed responses also carry
  `Cache-Control: no-cache` and `X-Accel-Buffering: no` so intermediaries relay
  them token by token instead of buffering.
- **Idle cutoff.** After `UPSTREAM_STREAM_IDLE_TIMEOUT` seconds of silence the
  proxy flushes its buffers, sends a terminal chunk and `[DONE]`, and closes.

When the upstream closes a turn without content or tool calls, the proxy logs
the empty turn and emits `EMPTY_TURN_NOTICE` (enabled by default) before the
terminal chunk so an agent client has something actionable to display. Set it
empty to keep the turn unannotated.

Chat and generate requests are retried up to `UPSTREAM_MAX_RETRIES` times on
transport errors and on upstream `429`, `500`, `502`, `503`, and `504`, with
exponential backoff and jitter. Retries only happen before any response byte has
reached the client, so a stream that has already started is never restarted and
a partially seen answer is never replayed. Transparent passthrough routes are
not retried, because their request body cannot be replayed.

### Config file

Aliases and routes are the two settings that grow into lists, so they can also be
written as YAML instead of packed into one-line env strings:

```yaml
# proxy.yaml
models:
  deepseek-v4-flash:
    aliases: [dsv4-flash, DeepSeek-V4-Flash]
    compatibility: deepseek_v4
    recover_orphan_invokes: true
  gemma-4-e4b: [gemma]        # shorthand: a bare list of aliases

routes:
  vision:
    upstream: http://192.168.10.99:8080
    model: gemma-4-e4b
  audio:
    upstream: http://192.168.10.99:8080
    model: gemma-4-e4b
```

```bash
PROXY_CONFIG_FILE=/etc/opencode-proxy/proxy.yaml
```

The file only carries `models:` and `routes:`; deployment wiring stays in the
environment. Model compatibility is intentionally YAML-only so the temporary
fallback is visible beside the affected model. `MODEL_ALIASES` or
`MODALITY_ROUTES` in the environment replace the matching section entirely. A
configured file that is missing or malformed fails startup rather than silently
running with partial routing.

### Modality routing

Text-only models reject image and audio parts, so a request carrying them can be
sent to a second host instead. The proxy inspects chat requests for
`image_url`/`image`/`input_image` parts (vision) and `input_audio`/`audio`/`audio_url`
parts (audio), including Ollama `images` fields after translation, and forwards
matching requests to the configured route:

```bash
MODALITY_ROUTES='{"vision":{"upstream":"http://192.168.10.99:8080","model":"gemma-4-e4b"}}'
```

A route may also carry its own `api_key` and `headers`. When a route has an
`api_key`, it replaces the caller's `Authorization` so a credential for the
primary upstream is never forwarded to the routed host.

Details worth knowing:

- If a request carries both modalities, the `audio` route wins.
- A modality with no configured route logs a warning and goes to the primary
  upstream unchanged, since the primary model may well support it.
- Routed models are not added to `/v1/models`; they are reachable through
  routing, not by asking for them by name.

## API Surface

- `GET /healthz`: local proxy liveness check.
- `GET /readyz`: readiness check that probes upstream `GET /v1/models`. Connection failures and upstream `5xx` return `503`; auth errors still count as ready.
- `GET /healthz/config`: safe local config summary, with header values and URL credentials omitted.
- `GET /metrics`: proxy-owned Prometheus counters. Scrape vLLM separately for
  cache, KV utilization, queueing, and model latency.
- `/v1/chat/completions`: proxied to the upstream with request tool sanitization and response tool-call repair.
- `/v1/models` and `/models`: upstream model discovery with configured alias entries added.
- `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`: Ollama-compatible translations backed by the same upstream client and tool-call repair pipeline.
- `/api/tags`, `/api/ps`, `/api/show`, `/api/version`: Ollama discovery and metadata endpoints.
- `/{path:path}`: transparent passthrough for other OpenAI-compatible endpoints.

## Notes

- Set `UPSTREAM_URL` to the upstream base URL, not the `/v1` path. If `/v1` is included anyway, the proxy strips it and logs a warning.
- Chat and generate requests are limited by `MAX_CONCURRENT_UPSTREAM` (default `8`) so a shared GPU backend is not stampeded by OpenCode, Home Assistant, and other clients.
- Upstream transport failures return typed `502` bodies (`connection_refused`, `connect_timeout`, `read_timeout`, and related) without leaking hostnames from exception text.
- The proxy strips compressed SSE request headers so streamed responses can be parsed line by line.
- If an upstream response already contains standard OpenAI `tool_calls`, it is passed through unchanged.
- `reasoning_content` and `reasoning` fields (DeepSeek R1 / o1-style streaming) are scanned for raw tool-call blocks by default, but ordinary reasoning text remains in reasoning fields.
- Because scanned text is buffered with `STREAM_GUARD_CHARS`, reasoning deltas from the same upstream event may be emitted before that event's content delta. Before any `content` is emitted, held reasoning/reasoning_content tails are flushed so thinking cannot trail the answer in streaming clients (for example Pi).
- The Docker image runs as a non-root user and includes a `/healthz` healthcheck.
- SSE keepalives, the unbuffered streaming headers, and the pre-first-byte retry rule were adapted from [VisionBridge](https://github.com/thomasunise/visionbridge) (MIT).
