---
type: reference
title: Configuration
description: Every proxy setting, the optional YAML config file, and the precedence rules between environment variables and file contents.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/settings.py
tags: [configuration, environment, yaml, aliases, routing]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# Configuration

Settings are validated by pydantic at startup. A malformed value fails the
process rather than running with a silently wrong default. The live effective
configuration is readable at `GET /healthz/config`, with credentials and header
values omitted.

## Precedence

Highest wins:

1. Constructor arguments (tests)
2. Environment variables
3. `.env` file
4. YAML config file (`PROXY_CONFIG_FILE`)

The YAML file is the lowest-priority source, so an environment variable always
overrides the same section in the file. Only `model_aliases` and
`modality_routes` can come from the file; deployment wiring stays in the
environment.

## Upstream

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_URL` | `http://127.0.0.1:4000` | Backend base URL. A trailing `/v1` is stripped with a warning. |
| `UPSTREAM_API_KEY` | unset | Fallback bearer token. A caller's `Authorization` takes precedence. |
| `UPSTREAM_CONNECT_TIMEOUT` | `10` | Seconds. |
| `UPSTREAM_READ_TIMEOUT` | `0` | Seconds; `0` disables. Intentionally unlimited — see below. |
| `UPSTREAM_WRITE_TIMEOUT` | `30` | Seconds. |
| `UPSTREAM_POOL_TIMEOUT` | `30` | Seconds. |
| `UPSTREAM_READY_TIMEOUT` | `2` | `/readyz` probe timeout. |
| `UPSTREAM_MAX_RETRIES` | `2` | Retries before the first response byte. `0` disables. |
| `MAX_CONCURRENT_UPSTREAM` | `8` | Concurrent chat/generate calls; over the limit returns `429`. `0` disables. |
| `CUSTOM_HEADERS` / `UPSTREAM_HEADERS` | unset | Extra upstream headers. JSON object or newline-separated `Header: value`. |

`UPSTREAM_READ_TIMEOUT=0` is a deliberate choice, not an oversight: a local model
can legitimately take minutes to produce a first token. Silence is bounded by
`UPSTREAM_STREAM_IDLE_TIMEOUT` instead, which distinguishes "thinking" from
"gone" in a way a flat read timeout cannot.

## Streaming

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_STREAM_IDLE_TIMEOUT` | `120` | Seconds of upstream silence before the client stream is terminated cleanly. `0` disables. |
| `SSE_KEEPALIVE_INTERVAL` | `10` | Seconds of silence between `: keepalive` comments. `0` disables. |
| `STREAM_GUARD_CHARS` | `192` | Text held back while watching for a split tool-call marker. |
| `TOOL_ARGUMENT_CHUNK_SIZE` | `64` | Size of streamed function-argument deltas. |

## Tool-call repair

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOOL_CALL_SCAN_FIELDS` | `content,reasoning,reasoning_content` | Fields scanned for raw markup. `all` selects all three. |
| `MAX_RAW_TOOL_BLOCK_CHARS` | `131072` | Larger blocks pass through as text. |
| `MAX_TOOL_CALLS` | `32` | More calls in one block pass through as text. |
| `MAX_TOOL_ARGUMENT_CHARS` | `262144` | Larger arguments pass through as text. |
| `SANITIZE_TOOLS` | `true` | Drop non-`function` tools from requests. |
| `REQUEST_DROP_FIELDS` | unset | Comma-separated request fields removed before forwarding. |

Every limit degrades to passthrough-as-text. None of them drop content.

## Models and routing

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_ALIASES` | unset | Alias map. `alias=target` pairs, newline-separated pairs, or a JSON object. |
| `ALIAS_CONFLICT_POLICY` | `skip` | On collision with an upstream model id: `skip`, `shadow`, or `error`. |
| `MODALITY_ROUTES` | unset | JSON map of `vision`/`audio` to an alternate upstream. |
| `PROXY_CONFIG_FILE` | unset | Path to the YAML file below. |

## Service

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROXY_HOST` | `0.0.0.0` | Bind host. |
| `PROXY_PORT` | `9526` | Bind port. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `OLLAMA_VERSION` | `0.5.1` | Version reported by `GET /api/version`. |

## YAML config file

Aliases and routes are the two settings that grow into lists, which is where
one-line environment strings become unreadable. `PROXY_CONFIG_FILE` points at:

```yaml
models:
  deepseek-v4-flash:
    aliases: [dsv4-flash, DeepSeek-V4-Flash]
  gemma-4-e4b: [gemma]          # bare list is shorthand for aliases

routes:
  vision:
    upstream: http://192.168.10.99:8080
    model: gemma-4-e4b
  audio:
    upstream: http://192.168.10.99:8080
    model: gemma-4-e4b
    api_key: ""                 # optional; replaces the caller's Authorization
    headers:                    # optional
      X-Skip-Auth: "true"
```

Only `models:` and `routes:` are accepted; an unknown section is an error.
A configured file that is missing or malformed fails startup rather than running
with partial routing — half-applied routing would send image requests to a
text-only model and look like a model quality problem.

Template: [`/home/tim/projects/opencode-proxy/proxy.example.yaml`](/home/tim/projects/opencode-proxy/proxy.example.yaml).

## Deployed values

The homelab container currently sets only `UPSTREAM_URL`, `UPSTREAM_API_KEY`,
`LOG_LEVEL`, `OLLAMA_VERSION`, `MODEL_ALIASES`, `TZ`, `PROXY_HOST`, and
`PROXY_PORT`. Everything else runs on defaults. Aliases are still in the
environment, not a config file.
