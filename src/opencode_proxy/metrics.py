"""Bounded-cardinality Prometheus metrics for proxy-owned behavior."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
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
    request_normalizations: Counter
    upstream_errors: Counter
    finish_reasons: Counter
    usage_tokens: Counter
    upstream_overloads: Counter
    upstream_active: Gauge

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
                ("phase",),
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
            request_normalizations=Counter(
                "opencode_proxy_request_normalizations",
                "Outgoing request message shapes repaired before forwarding.",
                ("kind",),
                registry=registry,
            ),
            upstream_errors=Counter(
                "opencode_proxy_upstream_errors",
                "Chat completion requests answered with an upstream error status.",
                ("type",),
                registry=registry,
            ),
            finish_reasons=Counter(
                "opencode_proxy_finish_reasons",
                "Turn terminators seen by the proxy, unknown values folded into 'other'.",
                ("reason", "transport"),
                registry=registry,
            ),
            usage_tokens=Counter(
                "opencode_proxy_usage_tokens",
                "Upstream-reported token usage, counted disjointly per kind.",
                ("kind",),
                registry=registry,
            ),
            upstream_overloads=Counter(
                "opencode_proxy_upstream_overloads",
                "Requests rejected because the proxy concurrency limit was full.",
                registry=registry,
            ),
            upstream_active=Gauge(
                "opencode_proxy_upstream_active",
                "Currently active chat/generate slots held by the proxy.",
                registry=registry,
            ),
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
