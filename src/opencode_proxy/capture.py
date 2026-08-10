"""Opt-in raw capture of streamed turns, for diagnosing output anomalies.

Some faults — a sentence duplicated in the model's visible output, say — are
intermittent, need the original prompt to reproduce, and can originate in any
layer of the serving chain. Capture exists so the *next* occurrence localizes
itself: it records the SSE frames arriving from upstream alongside the bytes
this proxy sends to its client, so the two can be compared offline. If the
anomaly is in the upstream frames the proxy is exonerated; if it appears only
in the client bytes the proxy created it; if it is in neither, it was
introduced further downstream.

Capture writes model output to disk in the clear and is therefore off unless
``CAPTURE_STREAM_DIR`` is set. The request body — which carries the prompt, and
so usually the most sensitive part of a turn — is recorded only when
``CAPTURE_STREAM_INCLUDE_REQUEST`` is additionally enabled. Files are never
rotated or pruned; the operator owns the directory's lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

LOG = logging.getLogger(__name__)

CAPTURE_FORMAT_VERSION = 1


class StreamCapture:
    """Append-only JSONL record of one streamed turn.

    Every record is flushed as it is written, so a capture stays readable even
    if the process dies mid-turn — the failure modes worth capturing include
    the ones that strand a stream.
    """

    def __init__(self, path: Path, *, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._written = 0
        self._seq = 0
        self._truncated = False
        self._handle: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def open(self, *, model: str | None, upstream_url: str) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding="utf-8")
        except OSError:
            # Capture is a diagnostic; never let it fail the turn it observes.
            LOG.exception("stream capture could not open %s; continuing uncaptured", self._path)
            self._handle = None
            return
        self._write(
            {
                "kind": "start",
                "format_version": CAPTURE_FORMAT_VERSION,
                "model": model,
                "upstream": upstream_url,
            }
        )

    def request_body(self, body: object) -> None:
        self._write({"kind": "request", "body": body})

    def upstream_frame(self, raw_lines: Iterable[str]) -> None:
        self._write({"kind": "upstream", "lines": list(raw_lines)})

    def upstream_eof(self) -> None:
        self._write({"kind": "upstream_eof"})

    def client_bytes(self, payload: bytes) -> None:
        self._write({"kind": "client", "text": payload.decode("utf-8", errors="replace")})

    def note(self, message: str, **fields: Any) -> None:
        self._write({"kind": "note", "message": message, **fields})

    def close(self, *, reason: str) -> None:
        if self._handle is None:
            return
        self._write({"kind": "end", "reason": reason, "truncated": self._truncated})
        try:
            self._handle.close()
        except OSError:
            LOG.exception("stream capture could not close %s", self._path)
        finally:
            self._handle = None

    def _write(self, record: Mapping[str, Any]) -> None:
        handle = self._handle
        if handle is None or (self._truncated and record.get("kind") != "end"):
            return
        self._seq += 1
        line = json.dumps(
            {"seq": self._seq, "at": time.time(), **record},
            ensure_ascii=False,
        )
        if self._max_bytes and self._written + len(line) > self._max_bytes:
            self._truncated = True
            LOG.warning(
                "stream capture hit CAPTURE_STREAM_MAX_BYTES; truncating %s",
                self._path,
            )
            self._write_line(handle, json.dumps({"seq": self._seq, "kind": "truncated"}))
            return
        self._written += len(line) + 1
        self._write_line(handle, line)

    def _write_line(self, handle: IO[str], line: str) -> None:
        try:
            handle.write(line + "\n")
            handle.flush()
        except OSError:
            LOG.exception("stream capture write failed for %s; disabling", self._path)
            self._handle = None


def open_capture(
    directory: str,
    *,
    max_bytes: int,
    model: str | None,
    upstream_url: str,
) -> StreamCapture | None:
    """Create a capture for one turn, or ``None`` when capture is disabled."""
    if not directory:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    name = f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}.jsonl"
    capture = StreamCapture(Path(directory) / name, max_bytes=max_bytes)
    capture.open(model=model, upstream_url=upstream_url)
    return capture
