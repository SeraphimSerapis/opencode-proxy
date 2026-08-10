---
type: runbook
title: Diagnose duplicated output
description: Attribute a repeated sentence or fragment to the layer that produced it, using the pi transcript first and stream capture second.
resource: /home/tim/projects/opencode-proxy/src/opencode_proxy/capture.py
tags: [troubleshooting, streaming, sse, litellm, vllm, pi, capture]
status: active
generated:
  by: claude-code/opus-5
  at: 2026-08-10T00:00:00+02:00
verified: 2026-08-10
---

# Diagnose duplicated output

## Symptom

The assistant repeats itself on screen: a sentence appears, cut off partway,
and then appears again in full.

> Usage: codex-usage in any new shell. Single-quoted so tokens are rea
> Usage: codex-usage in any new shell. Single-quoted so tokens are read at call time, ...

Four layers could produce this — vLLM, the proxy, LiteLLM, or the client's
renderer. Do not start by reading proxy code; start by asking whether the
duplicate is in the bytes at all.

## 1. Is the duplicate in the stored message?

pi writes each finalised assistant message to its session transcript. That
record is the model's output as pi received it, before any drawing:

```bash
grep -l "SOME DISTINCTIVE PHRASE" ~/.pi/agent/sessions/*/*.jsonl
```

Then count occurrences inside the assistant message, and check the `role` of
every record that contains the phrase:

```bash
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "tools")
from analyze_capture import find_duplicates

path = Path("<transcript>.jsonl")
for line in path.read_text().splitlines():
    record = json.loads(line)
    message = record.get("message", {})
    parts = message.get("content") if isinstance(message.get("content"), list) else []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if "SOME DISTINCTIVE PHRASE" not in text:
        continue
    print(message.get("role"), message.get("model"), len(find_duplicates(text, "content", 24)))
PY
```

**If the assistant message contains the text once, stop.** The duplicate was
never in the bytes; it is a rendering artifact in pi's TUI, which repainted a
partial line and then the completed line. Nothing upstream is at fault.

Watch for the trap that hid this twice: a `role: user` record can also contain
the duplicate, because pasting the screen text into pi to report the problem
stores it verbatim. Only the `role: assistant` record is evidence.

## 2. If the duplicate *is* in the stored message

Now it is in the bytes and worth attributing. Enable capture on the proxy —
see [configuration](/reference/configuration.md#stream-capture):

```bash
CAPTURE_STREAM_DIR=/var/lib/opencode-proxy/captures
CAPTURE_STREAM_INCLUDE_REQUEST=true   # needed only to replay the prompt later
```

Capture records both sides of every streamed turn: the SSE frames arriving from
vLLM and the bytes the proxy sends to LiteLLM. Reproduce, then analyze:

```bash
python tools/analyze_capture.py /var/lib/opencode-proxy/captures/*.jsonl
```

The tool reconstructs the assistant text from each side independently and
reports duplicated spans in both, which turns an observation into an
attribution:

| Duplicate found in | Meaning |
| --- | --- |
| upstream and client | vLLM produced it; the proxy relayed it faithfully. Tune sampling (`temperature`, `repetition_penalty`). |
| client only | The proxy produced it. Investigate the stream guard in `_process_stream_field_text`. |
| upstream only | The proxy dropped it. Also a proxy bug, in the opposite direction. |
| neither | LiteLLM or the client renderer. Compare against step 1. |

Exit status is `0` when a duplicate was found and `1` when the captures are
clean, so it can gate a loop over many captures.

`--text` prints the reconstructed text from each side when the automatic
detector disagrees with what you saw; `--min-length` lowers the 24-character
floor for short repeats.

## Turning capture back off

Capture writes model output — and, with `CAPTURE_STREAM_INCLUDE_REQUEST`,
prompts — to disk in the clear, and nothing rotates or prunes the directory.
Unset `CAPTURE_STREAM_DIR` and remove the captures once the question is
answered.

## History

Two sightings, 2026-08-03 and 2026-08-10, both on `deepseek-v4-flash` through
the pi → LiteLLM → proxy → vLLM chain, both the closing sentence of a turn cut
mid-word and then repeated. Both were resolved at step 1: the stored assistant
messages each contain the sentence exactly once, so neither the model nor the
proxy duplicated anything. The cut lengths also differed between the two
sightings (63 and 68 characters), which had already ruled out any fixed-size
buffer in the chain — `STREAM_GUARD_CHARS` is 192 and would not cut at either
point.

One follow-up hypothesis was that the proxy closed a truncated turn with
`finish_reason: "length"` and pi's agent loop re-requested, so pi drew a partial
message and then a regenerated one while persisting only the last. Excluded on
evidence: `length` has never been recorded for `deepseek-v4-flash` in 72 pi
sessions (all six instances are a different model from 2026-07-22), neither
sighting's turn contains an `aborted` or `error` record, and pi does persist
partial text when a turn is genuinely interrupted (4 of 69 aborted assistant
messages carry it) — so the absence of a partial message is evidence rather
than a logging gap.

What the two sightings do share is the rendering context, which is why tool
calls feel implicated: both are text-only assistant messages streamed
immediately after a `toolResult`.

**Confirmed 2026-08-10** with capture running, on a third sighting (capture
`20260810T072903`). Both sides of that turn hold the sentence exactly once and
are byte-identical at 2654 characters, so the proxy relayed the model's output
unchanged. The decisive detail is where the repeat began:

```
...just re-save / trigger a refresh in the TRMNL Screenshot plugin (it may cache the old page).
```

In the bytes that is one continuous line — there is no newline before
`Screenshot`. The break the user saw there is the terminal's soft wrap, so the
repeated unit was exactly one *visual* wrapped line, a unit that exists only in
the renderer. pi had drawn 41 of that line's 46 characters, and when the last
five (`age).`) arrived it re-wrapped and repainted the full line without
clearing the partial one.

That also explains the varying cut lengths across the three sightings (63, 68,
and 41 characters): each is simply wherever the token stream had reached when
the repaint fired. It is a pi TUI bug, not a proxy, LiteLLM, or vLLM one, and
the tool-call correlation is incidental — long streamed paragraphs after a tool
result are just where wrapped repaints are most visible.
