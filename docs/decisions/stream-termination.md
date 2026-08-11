---
type: decision
title: Always terminate a streamed turn
description: The proxy synthesises terminal chunks and bounds upstream silence with an idle cutoff rather than relying on a read timeout.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [sse, streaming, termination, timeouts, reliability]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# Always terminate a streamed turn

## Context

Clients intermittently kept showing a turn as in progress after the model had
finished — the answer fully rendered, the spinner still running, vLLM idle.

Two independent defects were found, both in the tail of the stream:

1. A terminal `finish_reason` was only synthesised when the proxy itself had
   repaired raw tool calls. Three realistic upstream shapes ended with `[DONE]`
   and no `finish_reason` at all: upstream omitting it, upstream truncating, and
   an error frame mid-stream.
2. With `UPSTREAM_READ_TIMEOUT=0`, an upstream that stopped sending without
   closing the socket held the proxy — and therefore the client — indefinitely.

The client-side evidence separated them. The agent's session transcript showed
the final assistant message persisted with `stopReason: "stop"` and full usage,
meaning it *had* received the finish chunk and finalised the message, and was
still waiting for the HTTP stream to end. That pointed at defect 2 for the
observed case, with defect 1 as a latent second cause.

## Decision

Guarantee that every streamed turn reaching the response-body stage ends with a
terminal chunk and `[DONE]`, regardless of upstream behaviour, and bound silence
with a dedicated idle cutoff.

## Why not just set a read timeout

`UPSTREAM_READ_TIMEOUT` is `0` on purpose. A local model's time to first token is
legitimately long and varies with queue depth and prompt size; any value low
enough to catch a dead stream promptly is also low enough to kill healthy slow
generations. A read timeout also cannot distinguish "thinking" from "gone".

`UPSTREAM_STREAM_IDLE_TIMEOUT` measures the same silence but responds differently:
instead of raising a transport error mid-response, it flushes held buffers, emits
the terminal chunk, sends `[DONE]`, and closes. The client gets a complete,
well-formed turn containing whatever text arrived, rather than a truncated body
or an error.

## Why two silence budgets rather than one

The guard starts counting when body iteration begins, so a single budget has to
cover both the wait for the first frame and the gaps between later ones. Those
are different measurements. Nothing is sent during prefill, which on a long
prompt legitimately takes minutes; once tokens are flowing they arrive
continuously. Measured against the deployed vLLM over seven days:

| | p99 | p99.9 |
| --- | --- | --- |
| Time to first token | 64s | 160s |
| Inter-token latency | 0.15s | 1.5s |

A single flat budget cannot serve both. The original `120` sat *below* the p99.9
of prefill — killing legitimate slow starts — while being ~80× the p99.9 of a
mid-stream gap, so a genuinely dead stream held the client for two minutes.

`UPSTREAM_STREAM_FIRST_FRAME_TIMEOUT` (default `240`) covers prefill with room
above the observed p99.9. `UPSTREAM_STREAM_IDLE_TIMEOUT` (default `30`) covers
the gaps after the first frame, which is still a 20× margin over the inter-token
p99.9. Terminations record which budget expired in
`opencode_proxy_stream_idle_terminations{phase}`.

The observed case was upstream silence (defect 2), not a proven connection reset.
For the separate transport-failure case, an `httpx.TransportError` after SSE
headers have been sent now emits a terminal `finish_reason: "length"` and
`[DONE]`. The same fallback applies to EOF without an upstream `[DONE]`.
That tells clients the partial turn was truncated rather than falsely reporting a
successful `stop`; local proxy bugs still propagate normally. If the failure or
EOF occurs before any upstream choice, the proxy synthesises choice `0` with the
same `length` terminal so agent clients can close the turn.

## Why synthesise rather than pass through

A transparent proxy would forward whatever the upstream sent, including nothing.
The mission statement says "only mutate when needed for client compatibility" —
this is that case. An agent client keying on `finish_reason` to close a turn
hangs forever on a stream that omits it, and the cost of synthesising one is a
single extra chunk that a compliant client ignores when the upstream already
provided its own (`finish_sent` guards against duplicates).

## Consequences

* Truncated streams are reported as `stop`, which is indistinguishable from a
  clean finish for the client. The event is logged, but the client is not told
  the answer was cut short. Signalling that without polluting the conversation
  with error text is unresolved — see the open question in
  [streaming contract](/architecture/streaming-contract.md).
* The terminal chunk carries an empty delta. A test asserting "no empty deltas"
  had to be narrowed to non-terminal chunks.
* Client disconnect remains the one case where neither is emitted, since there
  is nobody to receive them.
