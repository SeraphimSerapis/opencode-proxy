---
type: architecture
title: Streaming contract
description: What the proxy guarantees to a streaming client about how a turn ends, and the mechanisms that keep that promise when the upstream misbehaves.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [sse, streaming, keepalive, timeouts, retries, termination, tool-calls]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-03T09:40:00+02:00
verified: 2026-08-03
---

# Streaming contract

## The guarantee

Every streamed chat completion that reaches the response body stage ends with:

1. a terminal chunk carrying a non-null `finish_reason` for each choice that
   produced output, and
2. a `data: [DONE]` frame,

whether or not the upstream provided either. Agent clients treat a turn as
finished only when they see these; a stream that just stops leaves the client
spinning on a turn the model already completed.

This is a real failure mode, not a hypothetical. It presented as a fully
rendered answer with the client still showing "Working…", while the model had
long since finished. See
[diagnose a stalled stream](/runbooks/diagnose-stalled-stream.md).

A well-formed stream is necessary but not sufficient: a turn can be terminated
correctly and still be unusable, if the tool call it carries doesn't parse or
the turn produced nothing at all. See
[a terminated turn must also be a usable turn](/decisions/turn-usability.md)
for those two mechanisms; this document covers termination itself.

## Mechanisms

### Terminal chunk synthesis

`_finish_sse_stream` closes out every choice whose `finish_sent` is still false.
`_finish_reason_for_state` picks the reason: `tool_calls` if the proxy converted
raw markup or saw standard tool-call deltas without an upstream reason, and the
caller-supplied fallback otherwise.
Four upstream shapes reach this path: no `finish_reason` before `[DONE]`, a
stream that truncates with neither, a transport error mid-stream, and any other
exception raised while rewriting the stream (a proxy-side bug). The last two
share one `except` in `_rewrite_sse_stream` — once SSE headers are sent the
response is already committed to 200, so re-raising cannot produce an HTTP
error, it can only strand the client with a truncated body. The exception is
logged and the turn is closed the same way an idle timeout closes it.

The synthesised reason is `"length"`, not `"stop"`: a stream that ended without
the upstream saying so was truncated, and reporting it as a clean stop would be
false. `"length"` is also what vLLM itself reports for genuine `max_tokens`
truncation, so the two are not currently distinguishable to the client — see
[why synthesise rather than pass through](/decisions/stream-termination.md) for
the reasoning and its known limitation.

Each choice is terminal at most once. Duplicate finish chunks, or deltas that
arrive after a choice has finished, are ignored. If the final argument fragment
and finish reason share an upstream event, the proxy emits the fragment first,
then any append-only repair, then an applicable empty-turn notice, then the
terminal chunk.

### Idle cutoff

`UPSTREAM_STREAM_IDLE_TIMEOUT` (default 120s) bounds silence. `UPSTREAM_READ_TIMEOUT`
defaults to `0` — deliberately unlimited, because a local model's time to first
token is legitimately long — which means an upstream that stops sending without
closing the socket would otherwise hold the client forever. On timeout the proxy
logs a warning, flushes held buffers, emits the terminal chunk and `[DONE]`, and
closes. A bounded, visible failure instead of an unbounded invisible one.

### Keepalives

While the upstream is quiet the proxy emits `: keepalive` SSE comments every
`SSE_KEEPALIVE_INTERVAL` seconds (default 10). Comments are ignored by SSE
parsers; their purpose is to stop reverse proxies and load balancers from
dropping an idle connection during a long reasoning pause, and to give the
client evidence of life.

Streamed responses also carry `Cache-Control: no-cache` and
`X-Accel-Buffering: no` so intermediaries relay tokens rather than buffering.

The implementation detail worth preserving: the pending frame read is held in a
task **across** keepalive ticks. The obvious version —
`asyncio.wait_for(anext(frames), interval)` — cancels the read on every tick and
would corrupt a partially received frame. Silence accumulates across ticks and
still trips the idle cutoff, so the two mechanisms compose rather than conflict.

Keepalive ticks also drive the client-disconnect check, so a departed client is
noticed during a pause rather than only at the next frame.

### Retries, strictly before the first byte

`UPSTREAM_MAX_RETRIES` (default 2) covers transport errors and upstream `429`,
`500`, `502`, `503`, `504`, with exponential backoff plus jitter.

The rule that must not be broken: **never retry once any response byte has
reached the client.** Re-sending then would replay a prompt whose partial answer
the client already rendered. This is enforced structurally rather than by a
flag — all retry logic lives in `send_upstream_with_retries`, which returns
before the response generator is ever iterated. There is no code path from a
started stream back into it.

Transparent passthrough routes are excluded: they forward
`content=request.stream()`, which cannot be replayed.

## What is deliberately *not* guaranteed

* **Frame-for-frame fidelity.** Held text is regrouped, so the client's chunk
  boundaries will not match the upstream's. Concatenated content is preserved.
* **Ordering of non-choice frames against buffered text.** An error frame or SSE
  comment is forwarded immediately and can appear before buffered content that
  logically preceded it. Flushing on every raw frame would defeat the guard
  buffer, which is the worse trade.
* **An unambiguous signal on mid-stream upstream failure.** The synthesised
  `"length"` finish reason means "the proxy closed this, not the upstream," but
  it is indistinguishable from a real `max_tokens` truncation on the wire — see
  above. The client is not told *why* the turn was cut short, only that it was.

## Client disconnect

When `request.is_disconnected()` reports a departed client, the rewriter returns
immediately without emitting the terminal chunk or `[DONE]`. That is correct —
there is nobody to receive them — and it is the one sanctioned exception to the
guarantee above.

## Regression coverage

In [`tests/test_proxy.py`](/home/tim/projects/opencode-proxy/tests/test_proxy.py):
`test_streaming_without_upstream_finish_reason_still_closes_the_turn`,
`test_streaming_truncated_upstream_stream_still_closes_the_turn`,
`test_streaming_idle_upstream_terminates_instead_of_hanging`,
`test_quiet_upstream_gets_keepalive_comments`,
`test_failure_after_the_stream_started_is_not_retried`,
`test_streaming_non_transport_error_still_terminates_the_turn`,
`test_streaming_duplicate_terminal_choice_is_ignored`,
`test_streaming_tool_argument_repair_precedes_finish_without_delta`.

Coverage for the two usability mechanisms (argument repair, empty-turn notice)
lives with them in
[turn usability](/decisions/turn-usability.md#regression-coverage).

Each was confirmed to fail with the corresponding mechanism removed.
