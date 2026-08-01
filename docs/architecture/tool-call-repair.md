---
type: architecture
title: Tool-call repair
description: How raw tool-call markup emitted as model text is detected mid-stream and converted into standard OpenAI tool_calls deltas.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/compat.py
tags: [tool-calls, dsml, qwen, deepseek, streaming, parsing]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-01T16:20:00+02:00
verified: 2026-08-01
---

# Tool-call repair

## The problem

Some models emit tool calls as text in their own markup instead of populating
the OpenAI `tool_calls` field. The client then renders a wall of pseudo-XML
instead of executing a tool. DeepSeek emits DSML; Qwen emits `<tool_call>` XML
or bare JSON. The proxy's core job is turning that text back into the structured
field the client expects.

## Recognised formats

Detection markers live in `RAW_TOOL_START_MARKERS`; matched pairs in
`RAW_TOOL_BLOCK_PATTERNS`:

| Marker | Origin |
| --- | --- |
| `<｜DSML｜tool_calls>` (fullwidth bar, U+FF5C) | DeepSeek, canonical form |
| `<\|DSML\|tool_calls>` (ASCII pipe) | DeepSeek, degraded tokenisation |
| `<DSML>tool_calls>`, `<DSML: tool_calls>` | DeepSeek, further degraded |
| `<tool_calls>`, `<tool_call ...>` | Qwen XML |

`normalize_raw_tool_markup` folds every variant into the canonical DSML shape
before parsing, so the parser has one grammar to handle rather than five. The
ASCII and spaced variants exist because the fullwidth bar does not always
survive tokenisation intact.

Fixtures for each format: [`/home/tim/projects/opencode-proxy/tests/fixtures/tool_calls`](/home/tim/projects/opencode-proxy/tests/fixtures/tool_calls).
Add a fixture and a test before changing parser behaviour.

## Streaming detection: the guard buffer

A raw tool block arrives split across SSE frames — the opening marker may land
in one delta and the closing marker several frames later. The proxy cannot
forward text as it arrives without risking emitting half a tool block as visible
content, and it cannot buffer the whole response without destroying streaming.

The compromise is `STREAM_GUARD_CHARS` (default 192): text is released to the
client only once that many characters sit behind it, so a marker split across a
frame boundary is still detectable. Per scanned field, the state machine is:

1. **Complete block found** — emit any text before it, convert the block to
   `tool_calls` deltas, continue with the remainder.
2. **Block opener found, no closer yet** — mark the field pending and hold
   everything from the opener onward.
3. **Neither** — release all but the last `STREAM_GUARD_CHARS` characters.

A block that never closes, or exceeds `MAX_RAW_TOOL_BLOCK_CHARS`, or fails to
parse, or breaches `MAX_TOOL_CALLS` / `MAX_TOOL_ARGUMENT_CHARS`, is emitted as
plain text. The proxy degrades to a passthrough rather than dropping content.

## Scanned fields and reasoning order

`TOOL_CALL_SCAN_FIELDS` defaults to `content,reasoning,reasoning_content`.
Reasoning fields are scanned because these models sometimes emit tool markup
inside their thinking block.

That creates an ordering hazard. Held reasoning text and held content text are
separate buffers, so reasoning from an early frame could otherwise be flushed
*after* content from a later one — the client would render thinking below the
answer. Before any `content` delta is emitted, `_flush_reasoning_before_content`
releases held reasoning tails first.

The exception matters: a reasoning field that is mid raw-tool-block keeps its
buffer. Flushing there would leak a half-parsed block as visible text, which is
exactly the failure the guard buffer exists to prevent.

## Emission shape

`build_tool_call_chunks` emits arguments in `TOOL_ARGUMENT_CHUNK_SIZE` slices
(default 64) so clients that render arguments incrementally behave normally.
`state.next_tool_call_index` keeps indices monotonic across several converted
blocks in one response, and `raw_tool_calls_emitted` forces the terminal
`finish_reason` to `tool_calls` even when the upstream said `stop` — a client
that sees `stop` will not execute the tools it was just handed.

Responses that already contain standard `tool_calls` are passed through
untouched. Repair only engages when the field is missing and the markup is in
the text.
