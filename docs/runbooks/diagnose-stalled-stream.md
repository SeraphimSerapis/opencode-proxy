---
type: runbook
title: Diagnose a stalled stream
description: Locate the hop responsible when a client keeps showing a turn as in progress after the model has finished.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [troubleshooting, streaming, sse, litellm, vllm, pi]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# Diagnose a stalled stream

## Symptom

The answer is fully rendered but the client still shows a spinner. The model is
idle — no active generation in the backend logs.

Four hops can cause this: the client, LiteLLM, the proxy, or the backend. Work
from the client inward; each step rules out a hop.

## 1. Did the client receive a complete turn?

For pi, the session transcript records every assistant message as it is
finalised:

```bash
ls -t ~/.pi/agent/sessions/*/*.jsonl | head -1
```

```bash
python3 -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
m=rows[-1]['message']
print({k:v for k,v in m.items() if k!='content'})
" <path-to-session.jsonl>
```

* **`stopReason` present with usage** — the client received the terminal chunk
  and finalised the message. It is waiting on the HTTP stream to close, not on
  content. Continue to step 2; the fault is in the stream tail.
* **No final assistant message** — the client never got a terminal chunk. Skip to
  step 3 and look for a missing `finish_reason`.

## 2. Is the proxy holding the connection?

```bash
docker logs opencode-proxy --since 30m 2>&1 | grep -iE "keepalive|idle|terminating|retry"
```

An `upstream sent no SSE frame for …s; terminating the client stream` warning
means the proxy detected the stall and closed the turn cleanly — the fault is
upstream of the proxy, at the backend. No such warning during a hang means the
stall is downstream: LiteLLM or the client.

Note that uvicorn logs its access line when the response *starts*, so a
`200 OK` line proves nothing about completion.

## 3. Reproduce against the proxy directly

This bypasses LiteLLM and the client entirely:

```bash
curl -sS -N -w '\n__code=%{http_code} total=%{time_total}s\n' http://127.0.0.1:9526/v1/chat/completions -H 'content-type: application/json' -d '{"model":"dsv4-flash","stream":true,"stream_options":{"include_usage":true},"max_tokens":120,"messages":[{"role":"user","content":"Count from 1 to 10."}]}' | tail -6
```

A healthy tail ends with a chunk carrying `"finish_reason":"stop"`, an optional
usage chunk, then `data: [DONE]`, and `curl` exits promptly.

* **Ends correctly here but hangs through LiteLLM** — LiteLLM is the stalling
  hop. Check its logs and its own stream timeout settings.
* **Hangs here too** — the backend is not completing the stream. Compare with a
  direct call to `UPSTREAM_URL` to confirm.

## 4. Confirm liveness signalling is active

With defaults, a quiet upstream should produce keepalive comments within 10 s.
To see them without waiting for a slow generation, run a local instance with a
short interval:

```bash
UPSTREAM_URL=http://192.168.10.221:8080 SSE_KEEPALIVE_INTERVAL=0.5 uv run uvicorn opencode_proxy.app:create_app --factory --port 9527
```

```bash
curl -sS -N http://127.0.0.1:9527/v1/chat/completions -H 'content-type: application/json' -d '{"model":"deepseek-v4-flash","stream":true,"max_tokens":60,"messages":[{"role":"user","content":"Think carefully, then say OK."}]}' | grep -c '^: keepalive'
```

A non-zero count confirms keepalives during the time-to-first-token gap. Also
verify the response carries `Cache-Control: no-cache` and `X-Accel-Buffering: no`
(`curl -D -`); without them an intermediary may buffer the whole stream and
produce the same symptom.

## 5. Check the intermediary

Traefik fronts the external route. A buffering or idle-timeout policy on any hop
between client and proxy reproduces this symptom exactly. LAN clients that reach
`:9526` directly bypass Traefik — if the stall only happens through the hostname
route, the reverse proxy is the difference.

## Background

Both proxy-side causes were fixed on 2026-08-01: terminal chunks are now always
synthesised, and silence is bounded by `UPSTREAM_STREAM_IDLE_TIMEOUT`. See
[always terminate a streamed turn](/decisions/stream-termination.md). A stall
observed after that change points at LiteLLM, Traefik, or the client.
