---
okf_version: "0.2"
---

# OpenCode Proxy Knowledge Bundle

Compatibility proxy between OpenAI-compatible and Ollama-compatible clients and a
local model backend. It repairs models that emit tool calls as raw text, keeps
streamed turns terminating cleanly, and can route image or audio requests to a
second multimodal host.

Source: `/home/tim/projects/opencode-proxy`. Deployed as the `opencode-proxy`
Compose service in the homelab stack.

## Knowledge areas

* [Architecture](architecture/) - How a request flows, what the proxy rewrites, and what it guarantees to streaming clients.
* [Reference](reference/) - Configuration surface and HTTP endpoints.
* [Decisions](decisions/) - Design choices with the trade-off that produced them.
* [Runbooks](runbooks/) - Deploy, verify, and diagnose procedures.

## Start here

* New to the project: [request pipeline](architecture/request-pipeline.md).
* Debugging a turn that never finishes: [diagnose a stalled stream](runbooks/diagnose-stalled-stream.md).
* Changing settings: [configuration reference](reference/configuration.md).

See [log.md](log.md) for the change history.
