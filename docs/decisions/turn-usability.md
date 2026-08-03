---
type: decision
title: A terminated turn must also be a usable turn
description: The proxy completes truncated tool-call arguments and annotates turns that produce no output, because a well-formed stream can still strand an agent client.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/proxy.py
tags: [sse, streaming, tool-calls, agents, reliability]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-03T06:20:00+02:00
verified: 2026-08-03
---

# A terminated turn must also be a usable turn

## Context

[Always terminate a streamed turn](stream-termination.md) fixed the transport
layer: every turn now ends with a terminal chunk and exactly one `[DONE]`.
Clients still hung.

Driving a real agentic loop against the deployed vLLM — real tools, executed,
results fed back over 10–25 turns — surfaced two failures that the termination
work could not see, because in both cases **the stream is perfectly well
formed**. Correct `finish_reason`, exactly one `[DONE]`, clean close. The turn is
simply unusable by the agent consuming it.

### Truncated tool-call arguments reported as success

The upstream closes a turn with `finish_reason: "tool_calls"` while the streamed
`arguments` are cut off mid-value:

```
finish_reason='tool_calls'  [DONE]
arguments = '{"pattern": "def main|uvicorn|FastAPI'
usage: completion_tokens=85   (of a 16000 budget)
```

Not token exhaustion — 85 tokens of 16000. Replaying that history upstream
returns `HTTP 400 Unterminated string`. It reproduces against vLLM directly with
the proxy out of the path, at roughly three occurrences per eleven ten-turn loop
runs: rare per turn, near-certain across a long session.

An agent that receives an unparseable tool call cannot execute it, so no tool
result is ever produced, so the turn never advances. The spinner runs forever.

### Turns that produce no output at all

A reasoning model can spend its entire `max_tokens` budget thinking and close
the turn having emitted nothing else. Observed on the final synthesis step of a
review task: 12,748 bytes of reasoning, **zero** bytes of content, no tool calls,
`finish_reason: "length"`. The client has nothing to render and nothing to run —
indistinguishable from a hang, and it happens precisely when the work is
otherwise done.

## Decision

The proxy checks that a turn is *actionable*, not merely closed.

**Streamed tool-call arguments are accumulated per index and validated before
the turn closes.** Deltas already on the wire cannot be retracted, so a repair
has to be expressible as an append: `complete_truncated_json` returns the suffix
that makes the accumulated JSON parse, and the proxy emits it as one final
`arguments` fragment ahead of the terminal chunk. It closes open strings and
containers, and fills a dangling `:` with `null`.

It refuses to guess. A trailing `,` would require rewriting bytes the client
already holds, and a fragment like `{"a": tru` could be completed several ways
with different meanings. Those are logged at `error` and passed through
untouched — a wrong repair silently executes the wrong tool call, which is worse
than a call that visibly fails.

**A turn with no content and no tool calls is logged, and optionally annotated**
with a short `[proxy: ...]` note (`EMPTY_TURN_NOTICE`, set empty to disable) so
the client renders a dead end rather than nothing. The note is only emitted when
the *upstream* closed the turn, and it is emitted before that choice's terminal
chunk so the notice remains valid streamed content. When the proxy is
synthesising a terminator after a transport failure or idle timeout, the turn is
empty because the stream broke; blaming the token budget would be wrong, and the
synthetic `finish_reason` already reports the truncation.

## Consequences

* A `tool_calls` turn leaving the proxy carries parseable `arguments`, or the
  failure is in the log at `error` level with the raw fragment.
* The proxy now writes into the assistant's `arguments` and, for empty turns,
  its `content`. Both are observable: every repair logs its suffix.
* Argument fragments are buffered per choice for the duration of a turn, but
  only for valid tool indexes and within `MAX_TOOL_CALLS` /
  `MAX_TOOL_ARGUMENT_CHARS`; malformed or over-limit fragments are never
  guessed into a repair.
* This does not fix the upstream. vLLM still truncates tool calls; the proxy
  makes it survivable. If the upstream is fixed, the repair simply stops firing
  — there is no counter for this yet, so "stopped firing" currently means
  "stopped appearing in the logs," not a dashboard signal.

## Regression coverage

In [`tests/test_proxy.py`](/home/tim/projects/opencode-proxy/tests/test_proxy.py):
`test_streaming_truncated_tool_arguments_are_completed` (parametrized over
whether the last argument fragment and `finish_reason` arrive in the same
event), `test_streaming_valid_tool_arguments_are_left_alone`,
`test_streaming_tool_argument_repair_precedes_finish_without_delta`,
`test_streaming_multiple_tool_argument_repairs_keep_indexes_separate`,
`test_streaming_tool_argument_repair_ignores_invalid_tool_index`,
`test_streaming_tool_argument_repair_buffer_respects_argument_limit`,
`test_streaming_tool_argument_repair_buffer_respects_tool_call_limit`,
`test_streaming_unrepairable_tool_arguments_are_not_guessed`,
`test_streaming_empty_reasoning_only_turn_gets_a_notice`,
`test_streaming_empty_done_without_a_choice_gets_a_notice`,
`test_streaming_empty_turn_notice_can_be_disabled`,
`test_streaming_turn_with_content_gets_no_notice`,
`test_streaming_transport_failure_gets_no_misleading_notice`.

`complete_truncated_json` itself is covered directly in
[`tests/test_compat.py`](/home/tim/projects/opencode-proxy/tests/test_compat.py):
`test_complete_truncated_json_repairs_by_appending`,
`test_complete_truncated_json_returns_empty_suffix_when_already_valid`,
`test_complete_truncated_json_refuses_what_it_cannot_append_to`,
`test_complete_truncated_json_handles_deep_nesting_without_raising`.

The repair test cases mirror shapes captured from the deployed vLLM, not
synthetic edge cases: `'{"pattern": "def main|uvicorn|FastAPI'` and
`'{"path": "src/opencode_proxy/proxy.py'` are the literal truncations observed
during a live agentic loop.
