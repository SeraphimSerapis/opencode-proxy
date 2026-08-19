---
type: reference
title: Configuration
description: Every proxy setting, the optional YAML config file, and the precedence rules between environment variables and file contents.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/settings.py
tags: [configuration, environment, yaml, aliases, routing]
status: active
generated:
  by: codex/gpt-5
  at: 2026-08-15T16:21:21+02:00
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
overrides the same section in the file. Model aliases, model compatibility, and
modality routes can come from the file; deployment wiring stays in the
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
| `UPSTREAM_HEALTH_PATH` | unset | Extra `/readyz` probe that exercises the engine, e.g. `/health` for vLLM. |
| `UPSTREAM_MAX_RETRIES` | `2` | Retries before the first response byte. `0` disables. A `Retry-After` header on a retryable status wins over the backoff curve, clamped to 30s. |
| `EMPTY_RESPONSE_RETRIES` | `1` | Re-sends a buffered request for a `deepseek_v4` profile whose turn completed with no content and no tool call. `0` disables. Streamed turns are never retried. |
| `MAX_CONCURRENT_UPSTREAM` | `8` | Concurrent chat/generate calls; over the limit returns `429`. `0` disables. |
| `CUSTOM_HEADERS` / `UPSTREAM_HEADERS` | unset | Extra upstream headers. JSON object or newline-separated `Header: value`. |

`UPSTREAM_READ_TIMEOUT=0` is a deliberate choice, not an oversight: a local model
can legitimately take minutes to produce a first token. Silence is bounded by
`UPSTREAM_STREAM_IDLE_TIMEOUT` instead, which distinguishes "thinking" from
"gone" in a way a flat read timeout cannot.

## Streaming

| Variable | Default | Purpose |
| --- | --- | --- |
| `UPSTREAM_STREAM_IDLE_TIMEOUT` | `30` | Seconds of silence *between* SSE frames before the client stream is terminated cleanly. `0` disables. |
| `UPSTREAM_STREAM_FIRST_FRAME_TIMEOUT` | `480` | Seconds of silence before the *first* SSE frame, covering prefill. `0` disables. |
| `SSE_KEEPALIVE_INTERVAL` | `10` | Seconds of silence between keepalive ticks. `0` disables. |
| `STREAM_GUARD_CHARS` | `192` | Text held back while watching for a split tool-call marker. |
| `TOOL_ARGUMENT_CHUNK_SIZE` | `64` | Size of streamed function-argument deltas. |
| `EMPTY_TURN_NOTICE` | a short explanatory message | Content emitted before the terminal chunk when a turn the upstream closed itself ends with no content and no tool calls. Set empty to disable. See [turn usability](/decisions/turn-usability.md). |

`/healthz/config` reports whether `EMPTY_TURN_NOTICE` is enabled without
exposing the configured message text.

## Per-request headers

| Header | Values | Effect |
| --- | --- | --- |
| `X-Opencode-Proxy-Keepalive` | `chunk` | Emit keepalive ticks as empty-delta `chat.completion.chunk` frames instead of `: keepalive` comments, for clients whose SSE parser discards comments and which need visible forward progress. Any other value, or the header's absence, keeps comments. Stripped before forwarding upstream. |

## Stream capture

Off by default. When enabled, every streamed turn is written to its own JSONL
file recording the SSE frames received from upstream alongside the bytes sent
to the client, so an output anomaly can be attributed to a layer without
reproducing it. See [diagnose duplicated output](/runbooks/diagnose-duplicated-output.md).

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAPTURE_STREAM_DIR` | empty | Directory for per-turn capture files. Empty disables capture entirely. |
| `CAPTURE_STREAM_MAX_BYTES` | `8388608` | Per-turn cap; a capture that reaches it records a `truncated` marker and stops. `0` disables the cap. |
| `CAPTURE_STREAM_INCLUDE_REQUEST` | `false` | Also record the request body, which carries the prompt. Needed to replay a captured turn. |

Capture writes model output to disk in the clear, and with
`CAPTURE_STREAM_INCLUDE_REQUEST` it writes prompts too. Files are never rotated
or pruned — the operator owns the directory's lifecycle. `/healthz/config`
reports only whether capture is enabled, not the directory path.

## Tool-call repair

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOOL_CALL_SCAN_FIELDS` | `content,reasoning,reasoning_content` | Fields scanned for raw markup. `all` selects all three. |
| `MAX_RAW_TOOL_BLOCK_CHARS` | `131072` | Larger blocks pass through as text. |
| `MAX_TOOL_CALLS` | `32` | More raw calls in one block pass through as text; standard streamed repair tracks at most this many indexes. |
| `MAX_TOOL_ARGUMENT_CHARS` | `262144` | Larger raw arguments pass through as text; standard streamed tool-call repair stops accumulating at this bound. |
| `SANITIZE_TOOLS` | `true` | Drop non-`function` tools from requests. |
| `NORMALIZE_REQUESTS` | `true` | Repair outgoing message and token shapes for models with the `deepseek_v4` compatibility profile. Other models stay transparent. Ollama `think` translation remains active because it is protocol translation. See [conform to DeepSeek's own client](/decisions/deepseek-wire-contract.md). |
| `REQUEST_DROP_FIELDS` | unset | Comma-separated request fields removed before forwarding. |

Every limit degrades to passthrough-as-text. None of them drop content.

The `deepseek_v4` compatibility profile is configured per canonical upstream
model in YAML. Its optional `thinking_transport` picks how "no thinking" is
expressed: `api` (default) sends the DeepSeek API's top-level
`thinking` object, preserving a compatible effort when thinking is enabled;
`chat_template_kwargs` sends
`chat_template_kwargs: {"thinking": false}`, which is what vLLM reads -- it
ignores the API form. Under the vLLM form `reasoning_effort` is translated to a
boolean and dropped, since a chat-template argument cannot carry a level. `recover_orphan_invokes: true` repairs the narrow vLLM #49117
failure only when the request declares function tools, `tool_choice` is not
`none`, the marker is canonical fullwidth-bar V4 DSML in `content`, and the
completed name exactly matches a declared tool. Rejected blocks remain
byte-for-byte text. This fallback is temporary; see the
[DeepSeek V4 runbook](../runbooks/deepseek-v4.md).

`tool_choice` is intentionally preserved for the deployed vLLM path, where the
live tool-call contract uses it to force a call. A direct vendor API deployment
that rejects `tool_choice` should set `REQUEST_DROP_FIELDS=tool_choice` for that
instance rather than changing the shared profile. Do not drop it when callers
depend on `none`, `required`, or a forced function: removing the field changes
that request's semantics.

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
    compatibility: deepseek_v4
    recover_orphan_invokes: true
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

The homelab container sets `UPSTREAM_URL`, `UPSTREAM_API_KEY`,
`UPSTREAM_HEALTH_PATH=/health`, `LOG_LEVEL`, `OLLAMA_VERSION`,
`MAX_CONCURRENT_UPSTREAM`, `MODEL_ALIASES`, `TZ`, `PROXY_HOST`, and
`PROXY_PORT`. Everything else runs on defaults. Aliases are still in the
environment, not a config file.
