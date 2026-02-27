"""
NEXUS Metrics Collector
In-memory metrics collection — counters, histograms, and gauges.
No external dependencies required (no Prometheus client needed).
"""
from __future__ import annotations

import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Pre-defined metric name constants
# ---------------------------------------------------------------------------
TASK_TOTAL = "nexus_task_total"
TASK_DURATION = "nexus_task_duration_seconds"
LLM_CALLS_TOTAL = "nexus_llm_calls_total"
LLM_COST_TOTAL = "nexus_llm_cost_usd_total"
AGENT_QUALITY = "nexus_agent_quality_score"
CACHE_HIT_TOTAL = "nexus_cache_hit_total"
CIRCUIT_BREAKER_STATE = "nexus_circuit_breaker_state"
CHECKPOINT_OPS = "nexus_checkpoint_operations_total"
GUARDRAIL_TRIGGERS = "nexus_guardrail_triggers_total"


def _label_key(labels: dict[str, str] | None) -> str:
    """Create a deterministic string key from a labels dict."""
    if not labels:
        return "__default__"
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsCollector:
    """
    Thread-safe, in-memory metrics collector.

    Supports three metric types:
    - counter: monotonically increasing value (increment)
    - histogram: observed values (observe) — stores list of observations
    - gauge: point-in-time value (set_gauge)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, float]] = {}
        self._histograms: dict[str, dict[str, list[float]]] = {}
        self._gauges: dict[str, dict[str, float]] = {}

    # ── Counter ──────────────────────────────────────────────────────────────

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = _label_key(labels)
        with self._lock:
            if name not in self._counters:
                self._counters[name] = {}
            self._counters[name][key] = self._counters[name].get(key, 0.0) + value

    # ── Histogram ────────────────────────────────────────────────────────────

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation for a histogram metric."""
        key = _label_key(labels)
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = {}
            if key not in self._histograms[name]:
                self._histograms[name][key] = []
            self._histograms[name][key].append(value)

    # ── Gauge ────────────────────────────────────────────────────────────────

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric to an absolute value."""
        key = _label_key(labels)
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = {}
            self._gauges[name][key] = value

    # ── Retrieval ────────────────────────────────────────────────────────────

    def get(self, name: str) -> dict[str, Any]:
        """
        Get a single metric by name.

        Returns a dict with ``type`` and label-keyed values.
        """
        with self._lock:
            if name in self._counters:
                return {"type": "counter", "values": dict(self._counters[name])}
            if name in self._histograms:
                result: dict[str, Any] = {"type": "histogram", "values": {}}
                for lk, observations in self._histograms[name].items():
                    result["values"][lk] = {
                        "count": len(observations),
                        "sum": sum(observations),
                        "min": min(observations) if observations else 0.0,
                        "max": max(observations) if observations else 0.0,
                        "avg": sum(observations) / len(observations) if observations else 0.0,
                    }
                return result
            if name in self._gauges:
                return {"type": "gauge", "values": dict(self._gauges[name])}
        return {}

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return all collected metrics."""
        result: dict[str, dict[str, Any]] = {}
        with self._lock:
            all_names = set(self._counters) | set(self._histograms) | set(self._gauges)
        for name in sorted(all_names):
            result[name] = self.get(name)
        return result

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
metrics = MetricsCollector()
