#!/usr/bin/env python3
"""Locate duplicated output in a captured turn, and attribute it to a layer.

A capture file (see ``CAPTURE_STREAM_DIR``) holds both sides of one streamed
turn: the SSE frames that arrived from upstream, and the bytes the proxy sent
to its client. This tool reconstructs the assistant text from each side
independently and reports duplicated spans in both, which is what makes the
result an attribution rather than an observation:

  in upstream AND in client  -> upstream produced it; the proxy relayed it
  in client only             -> the proxy produced it
  in neither                 -> a downstream layer produced it, or it was
                                never in the bytes at all (e.g. a rendering
                                artifact in the client's terminal)

Usage:
    python tools/analyze_capture.py CAPTURE.jsonl [CAPTURE.jsonl ...]
    python tools/analyze_capture.py --min-length 24 --text capture.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MIN_LENGTH = 24
MAX_UNIT_CHARS = 4096
TEXT_FIELDS = ("content", "reasoning", "reasoning_content")


@dataclass
class SideText:
    """Assistant text reconstructed from one side of a capture."""

    fields: dict[str, list[str]] = field(default_factory=dict)

    def add(self, field_name: str, text: str) -> None:
        self.fields.setdefault(field_name, []).append(text)

    def joined(self, field_name: str) -> str:
        return "".join(self.fields.get(field_name, ()))

    def field_names(self) -> list[str]:
        preferred = [name for name in TEXT_FIELDS if name in self.fields]
        return preferred + [name for name in self.fields if name not in preferred]


@dataclass(frozen=True)
class Duplicate:
    """A span of text immediately followed by a repeat of itself."""

    field_name: str
    offset: int
    unit: str
    separator: str

    def describe(self) -> str:
        sep = repr(self.separator) if self.separator else "none"
        return (
            f"  {self.field_name} @ {self.offset}: {len(self.unit)} chars repeated "
            f"(separator {sep})\n"
            f"    {self.unit!r}"
        )


def collect_events(path: Path) -> tuple[SideText, SideText, dict[str, object]]:
    """Split a capture into upstream-side and client-side assistant text."""
    upstream = SideText()
    client = SideText()
    meta: dict[str, object] = {"records": 0, "truncated": False}

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            meta["records"] = int(meta["records"]) + 1  # type: ignore[arg-type]
            kind = record.get("kind")
            if kind == "start":
                meta["model"] = record.get("model")
                meta["upstream"] = record.get("upstream")
            elif kind == "truncated" or record.get("truncated"):
                meta["truncated"] = True
            elif kind == "upstream":
                for line in record.get("lines", ()):
                    _absorb_sse_line(str(line), upstream)
            elif kind == "client":
                for line in str(record.get("text", "")).splitlines():
                    _absorb_sse_line(line, client)

    return upstream, client, meta


def _absorb_sse_line(line: str, side: SideText) -> None:
    if not line.startswith("data:"):
        return
    payload = line[len("data:") :].lstrip()
    if not payload or payload == "[DONE]":
        return
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return
    if not isinstance(event, dict):
        return
    for choice in event.get("choices") or ():
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for field_name in TEXT_FIELDS:
            value = delta.get(field_name)
            if isinstance(value, str) and value:
                side.add(field_name, value)


def find_duplicates(text: str, field_name: str, min_length: int) -> list[Duplicate]:
    """Find a segment immediately followed by text that repeats it.

    The segment is delimited by line and sentence boundaries rather than
    searched over every possible length, which keeps the scan linear in the
    size of the turn and matches the shape of the observed fault: a segment is
    emitted, cut off, and then the same segment appears again — either
    identically or continuing past where the first copy stopped.
    """
    duplicates: list[Duplicate] = []
    seen: set[tuple[int, int]] = set()
    boundaries = _candidate_boundaries(text)

    for end_position, end in enumerate(boundaries):
        match = _repeat_ending_at(text, boundaries, end_position, end, min_length)
        if match is None:
            continue
        unit, separator = match
        key = (end - len(unit), len(unit))
        if key in seen:
            continue
        seen.add(key)
        duplicates.append(
            Duplicate(
                field_name=field_name,
                offset=end - len(unit),
                unit=unit,
                separator=separator,
            )
        )
    return duplicates


def _repeat_ending_at(
    text: str,
    boundaries: list[int],
    end_position: int,
    end: int,
    min_length: int,
) -> tuple[str, str] | None:
    """Longest boundary-delimited segment ending at ``end`` that repeats after it.

    A repeated segment can run over several sentences, so its start is searched
    back across boundaries rather than assumed to be the previous one. The
    search is bounded by ``MAX_UNIT_CHARS`` to keep the scan cheap on turns
    that run to hundreds of kilobytes.
    """
    starts = [
        start for start in boundaries[:end_position] if min_length <= end - start <= MAX_UNIT_CHARS
    ]
    for start in starts:  # farthest first, so the longest match wins
        unit = text[start:end].lstrip()
        if len(unit) < min_length:
            continue
        for separator_length in (0, 1, 2):
            start_of_repeat = end + separator_length
            separator = text[end:start_of_repeat]
            if separator and not separator.isspace():
                break
            if text.startswith(unit, start_of_repeat):
                return unit, separator
    return None


def _candidate_boundaries(text: str) -> list[int]:
    """Offsets where a repeated segment plausibly ends: line and sentence breaks."""
    boundaries = [0]
    for index, char in enumerate(text):
        if char == "\n" or (char == " " and index and text[index - 1] in ".!?"):
            boundaries.append(index)
    boundaries.append(len(text))
    return sorted(set(boundaries))


def analyze(path: Path, *, min_length: int, show_text: bool) -> bool:
    """Report on one capture. Returns True when a duplicate was found."""
    upstream, client, meta = collect_events(path)
    print(f"== {path.name}")
    model = meta.get("model")
    print(f"   model={model or 'unknown'} records={meta['records']}", end="")
    print("  [TRUNCATED CAPTURE]" if meta.get("truncated") else "")

    found: dict[str, list[Duplicate]] = {}
    for label, side in (("upstream", upstream), ("client", client)):
        hits: list[Duplicate] = []
        for field_name in side.field_names():
            hits.extend(find_duplicates(side.joined(field_name), field_name, min_length))
        found[label] = hits

    for label, side in (("upstream", upstream), ("client", client)):
        chars = sum(len(side.joined(name)) for name in side.field_names())
        print(f"   {label}: {chars} chars, {len(found[label])} duplicated span(s)")
        for duplicate in found[label]:
            print(duplicate.describe())

    if show_text:
        for label, side in (("upstream", upstream), ("client", client)):
            for field_name in side.field_names():
                print(f"   --- {label}.{field_name} ---")
                print(side.joined(field_name))

    verdict = _attribute(bool(found["upstream"]), bool(found["client"]))
    print(f"   verdict: {verdict}\n")
    return bool(found["upstream"] or found["client"])


def _attribute(in_upstream: bool, in_client: bool) -> str:
    if in_upstream and in_client:
        return "duplicate arrived from upstream and was relayed - not a proxy fault"
    if in_client:
        return "duplicate present only in proxy output - PROXY FAULT"
    if in_upstream:
        return "duplicate arrived from upstream but is absent downstream - proxy dropped it"
    return "no duplicate in either side - look downstream (LiteLLM, client renderer)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=f"shortest repeated span to report (default {DEFAULT_MIN_LENGTH})",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="also print the reconstructed text from each side",
    )
    args = parser.parse_args(argv)

    any_found = False
    for path in args.captures:
        if not path.is_file():
            print(f"== {path}: not a file", file=sys.stderr)
            continue
        any_found |= analyze(path, min_length=args.min_length, show_text=args.text)
    return 0 if any_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
