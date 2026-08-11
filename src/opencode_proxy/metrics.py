"""Bounded-cardinality Prometheus metrics for proxy-owned behavior."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST


@dataclass(frozen=True)
class ProxyMetrics:
    registry: CollectorRegistry
    orphan_recovery: Counter
    raw_tool_repair: Counter
    synthesized_tool_call_ids: Counter
    tool_argument_repair: Counter
    upstream_retries: Counter
    stream_idle_terminations: Counter
    empty_turns: Counter
    upstream_ready_failures: Counter

    @classmethod
    def create(cls) -> ProxyMetrics:
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            orphan_recovery=Counter(
                "opencode_proxy_orphan_recovery",
                "DeepSeek V4 orphan invoke recovery attempts.",
                ("outcome", "reason"),
                registry=registry,
            ),
            raw_tool_repair=Counter(
                "opencode_proxy_raw_tool_repair",
                "Raw text tool-call blocks converted to OpenAI tool calls.",
                ("format", "field"),
                registry=registry,
            ),
            synthesized_tool_call_ids=Counter(
                "opencode_proxy_synthesized_tool_call_ids",
                "Missing native OpenAI tool-call IDs synthesized by the proxy.",
                ("transport",),
                registry=registry,
            ),
            tool_argument_repair=Counter(
                "opencode_proxy_tool_argument_repair",
                "Streamed tool argument completion outcomes.",
                ("outcome",),
                registry=registry,
            ),
            upstream_retries=Counter(
                "opencode_proxy_upstream_retries",
                "Upstream requests retried before response bytes reached the client.",
                ("reason",),
                registry=registry,
            ),
            stream_idle_terminations=Counter(
                "opencode_proxy_stream_idle_terminations",
                "Streams terminated by the upstream idle guard.",
                registry=registry,
            ),
            empty_turns=Counter(
                "opencode_proxy_empty_turns",
                "Completed upstream turns with no content or tool calls.",
                registry=registry,
            ),
            upstream_ready_failures=Counter(
                "opencode_proxy_upstream_ready_failures",
                "Readiness probes that judged the upstream unable to serve.",
                ("reason",),
                registry=registry,
            ),
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
