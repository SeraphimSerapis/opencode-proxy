---
type: runbook
title: Build, deploy, and verify
description: Validation gates, the homelab build and deploy path for the locally-built proxy image, smoke tests, and rollback.
resource: /home/tim/docker/compose/ai/opencode-proxy.yml
tags: [deploy, docker, compose, build, rollback, verification]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# Build, deploy, and verify

## Scope

`opencode-proxy` is a locally-built image: the Compose fragment declares
`build.context: ${PROJECTSDIR}/opencode-proxy` with `pull_policy: build`, so
`./utility/pull.sh` never refreshes it. Deploying means rebuilding from the
checkout. See the stack's
[locally-built image runbook](/home/tim/docker/documentation/runbooks/local-image-builds.md)
for the general procedure.

## 1. Validation gates

All four must pass before deploying. Nothing is pushed or deployed on a failure:

```bash
cd /home/tim/projects/opencode-proxy && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest
```

## 2. Build and deploy

From the stack root, which owns the Compose context:

```bash
cd /home/tim/docker && docker compose up -d --build opencode-proxy
```

The container publishes host `9526` and `11434`, both to container `9526`.

## 3. Verify

Liveness — proves only that the proxy process is serving:

```bash
curl -sS http://127.0.0.1:9526/healthz
```

Readiness, which probes the backend and is what the container healthcheck runs.
`docker compose ps` reporting `healthy` therefore means vLLM is serving too:

```bash
docker compose ps opencode-proxy && curl -sS http://127.0.0.1:9526/readyz
```

A `503` names the reason in the `upstream` field — see
[API surface](../reference/api-surface.md) for the full table. The same reasons
are counted in `opencode_proxy_upstream_ready_failures{reason}`, so an alert can
fire on the proxy losing its backend.

Effective configuration — confirms the deployed build picked up the intended
settings, with no credentials in the output:

```bash
curl -sS http://127.0.0.1:9526/healthz/config | python3 -m json.tool
```

End to end through a real generation, including stream termination:

```bash
curl -sS -N http://127.0.0.1:9526/v1/chat/completions -H 'content-type: application/json' -d '{"model":"dsv4-flash","stream":true,"max_tokens":40,"messages":[{"role":"user","content":"Say hi in 3 words."}]}' | tail -4
```

Expect a chunk with `"finish_reason"`, then `data: [DONE]`, and a prompt exit.

The Ollama surface shares the same upstream client and deserves one check:

```bash
curl -sS http://127.0.0.1:11434/api/version && curl -sS http://127.0.0.1:11434/api/tags | head -c 200
```

## 4. Rollback

The previous image is untagged by a rebuild but not deleted. Tag it before
building if a rollback path is wanted:

```bash
docker tag opencode-proxy:latest opencode-proxy:rollback-$(date +%Y%m%d)
```

To roll back, retag and recreate:

```bash
cd /home/tim/docker && docker tag opencode-proxy:rollback-<date> opencode-proxy:latest && docker compose up -d --force-recreate opencode-proxy
```

Reverting source instead means checking out the previous commit and repeating
step 2, since the image is built from the working tree.

## Edge access

`opencode-proxy.${DOMAIN_COFFEE}` is served by three Traefik routers on one
host rule, highest priority first:

| Router | Priority | Matches | Middleware |
| --- | --- | --- | --- |
| `opencode-proxy-local` | 200 | Client IP in the LAN/`LAN_BYPASS_CIDR` ranges | `chain-no-auth` |
| `opencode-proxy-bypass` | 100 | `X-API-KEY: $OPENCODE_PROXY_X_API` | `chain-external-webhook` + header strip |
| `opencode-proxy-rtr` | 1 | anything else | `chain-tinyauth` |

Tinyauth is a browser SSO flow, so it can only be the catch-all: the clients
that actually use this route are headless (OpenAI/JS, opencode, curl) and would
be redirected to a login page. They match the LAN router instead, and off-LAN
headless clients use the `X-API-KEY` header. That key is a Traefik-edge
credential only — `opencode-proxy-strip-x-api-key` clears the header before the
request reaches the proxy, which would otherwise forward it to vLLM.

Direct LAN access on `:9526`/`:11434` bypasses Traefik entirely and is
unaffected by any of this.

## Alerting

Prometheus scrapes `opencode-proxy:9526` and vLLM, and delivers through
Alertmanager to mailrise, which fans `*@tme.coffee` out to Gmail and Discord.
Rules live in `/home/tim/docker/appdata/prometheus/rules/opencode-proxy.yml`:

| Alert | Fires on |
| --- | --- |
| `OpenCodeProxyDown` | Scrape of the proxy fails for 2m. |
| `OpenCodeProxyUpstreamNotReady` | `/readyz` failing for 5m; the `reason` label names which check. |
| `OpenCodeProxyStreamsStalling` | >3 mid-stream idle terminations in 15m. |
| `OpenCodeProxyPrefillTimingOut` | >2 first-frame timeouts in 30m. |
| `VllmDown` / `VllmNoModelServed` | vLLM unscrapeable, or scrapeable with no engine metrics. |

Validate rule changes before reloading, then reload without a restart:

```bash
docker run --rm -v /home/tim/docker/appdata/prometheus/rules:/rules:ro \
  --entrypoint promtool prom/prometheus:v3.13.2 check rules /rules/opencode-proxy.yml
curl -X POST http://127.0.0.1:9090/-/reload
```

`prometheus.yml` is bind-mounted as a *single file*, so rewriting it on the host
replaces the inode and the container keeps reading the old one. Editing it needs
`docker compose up -d --force-recreate prometheus`, not a reload. Files under
`rules/` are a directory mount and do pick up a plain reload.

## Notes

* Clients hold connections to the proxy. Recreating the container drops
  in-flight streamed turns; expect one failed turn in any active session.
* The proxy's upstream is vLLM, not LiteLLM, which calls *into* the proxy — see
  [request pipeline](/architecture/request-pipeline.md). Do not reintroduce
  `depends_on: litellm`; it gates startup on a downstream service.
* Configuration lives in the Compose fragment's `environment:` block, not in the
  project's `.env`, which is for local development only.
