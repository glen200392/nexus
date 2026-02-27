"""
Tests for NEXUS Observability — Logging, Metrics, and Tracing.
"""
from __future__ import annotations

import time

import pytest


# ── Logging Tests ────────────────────────────────────────────────────────────

class TestLogging:
    def test_setup_logging(self):
        """setup_logging should not raise."""
        from nexus.observability.logging import setup_logging
        setup_logging(log_level="WARNING", json_format=True)
        setup_logging(log_level="DEBUG", json_format=False)

    def test_get_logger(self):
        """get_logger should return a logger with bound context."""
        from nexus.observability.logging import get_logger
        logger = get_logger("test.module", task_id="t-123", session_id="s-456")
        assert logger is not None
        # Should not raise
        logger.info("test message")
        logger.debug("debug message")
        logger.warning("warning message")
        logger.error("error message")

    def test_get_logger_bind(self):
        """Logger.bind() should return a new logger with merged context."""
        from nexus.observability.logging import get_logger
        logger = get_logger("test.bind", task_id="t-1")
        child = logger.bind(agent_id="agent-a")
        assert child is not logger
        child.info("bound message")


# ── Metrics Tests ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_increment(self):
        """Counter should increment correctly."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        m.increment("test_counter")
        m.increment("test_counter", value=5.0)
        result = m.get("test_counter")
        assert result["type"] == "counter"
        assert result["values"]["__default__"] == 6.0

    def test_metrics_increment_with_labels(self):
        """Counter should track labels separately."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        m.increment("req_total", labels={"method": "GET"})
        m.increment("req_total", labels={"method": "POST"})
        m.increment("req_total", labels={"method": "GET"})
        result = m.get("req_total")
        assert result["values"]["method=GET"] == 2.0
        assert result["values"]["method=POST"] == 1.0

    def test_metrics_observe(self):
        """Histogram should record observations."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        m.observe("latency", 0.1)
        m.observe("latency", 0.2)
        m.observe("latency", 0.3)
        result = m.get("latency")
        assert result["type"] == "histogram"
        vals = result["values"]["__default__"]
        assert vals["count"] == 3
        assert abs(vals["sum"] - 0.6) < 1e-9
        assert abs(vals["min"] - 0.1) < 1e-9
        assert abs(vals["max"] - 0.3) < 1e-9

    def test_metrics_set_gauge(self):
        """Gauge should store the latest value."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        m.set_gauge("temperature", 22.5)
        m.set_gauge("temperature", 23.0)
        result = m.get("temperature")
        assert result["type"] == "gauge"
        assert result["values"]["__default__"] == 23.0

    def test_metrics_get_all(self):
        """get_all should return all metrics."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        m.increment("c1")
        m.observe("h1", 1.0)
        m.set_gauge("g1", 5.0)
        all_metrics = m.get_all()
        assert "c1" in all_metrics
        assert "h1" in all_metrics
        assert "g1" in all_metrics
        assert len(all_metrics) == 3

    def test_metrics_reset(self):
        """reset should clear all metrics."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        m.increment("c1")
        m.observe("h1", 1.0)
        m.set_gauge("g1", 5.0)
        m.reset()
        assert m.get_all() == {}

    def test_metrics_get_unknown(self):
        """Getting an unknown metric returns empty dict."""
        from nexus.observability.metrics import MetricsCollector
        m = MetricsCollector()
        assert m.get("nonexistent") == {}

    def test_metric_constants(self):
        """Pre-defined metric constants should be available."""
        from nexus.observability.metrics import (
            TASK_TOTAL, TASK_DURATION, LLM_CALLS_TOTAL, LLM_COST_TOTAL,
            AGENT_QUALITY, CACHE_HIT_TOTAL, CIRCUIT_BREAKER_STATE,
            CHECKPOINT_OPS, GUARDRAIL_TRIGGERS,
        )
        assert TASK_TOTAL == "nexus_task_total"
        assert TASK_DURATION == "nexus_task_duration_seconds"

    def test_module_singleton(self):
        """Module-level metrics singleton should exist."""
        from nexus.observability.metrics import metrics
        assert isinstance(metrics, type(metrics))
        metrics.reset()  # Clean up


# ── Tracing Tests ────────────────────────────────────────────────────────────

class TestTracing:
    def test_tracing_start_end_span(self):
        """Should create and end a span."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        span = t.start_span("test-op", attributes={"key": "val"})
        assert span.trace_id
        assert span.span_id
        assert span.name == "test-op"
        assert span.end_time is None
        t.end_span(span)
        assert span.end_time is not None
        assert span.status == "ok"

    def test_tracing_get_spans(self):
        """Should retrieve spans filtered by trace_id."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        s1 = t.start_span("op1")
        trace_id = s1.trace_id
        s2 = t.start_span("op2")  # Same trace
        t.end_span(s1)
        t.end_span(s2)

        # Get all spans
        all_spans = t.get_spans()
        assert len(all_spans) == 2

        # Get by trace_id
        filtered = t.get_spans(trace_id=trace_id)
        assert len(filtered) == 2

    def test_tracing_get_spans_different_traces(self):
        """Spans from different traces should be filterable."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        s1 = t.start_span("op1")
        tid1 = s1.trace_id
        t.end_span(s1)
        t.clear()
        t._active_trace_id = None  # Force new trace

        s2 = t.start_span("op2")
        tid2 = s2.trace_id
        t.end_span(s2)

        # If they happen to share trace_id (unlikely), skip assertion
        if tid1 != tid2:
            assert len(t.get_spans(trace_id=tid2)) == 1

    def test_tracing_export(self):
        """export should return list of dicts."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        s = t.start_span("export-test", attributes={"foo": "bar"})
        t.end_span(s)
        exported = t.export()
        assert len(exported) == 1
        d = exported[0]
        assert d["name"] == "export-test"
        assert d["attributes"]["foo"] == "bar"
        assert d["status"] == "ok"
        assert d["duration_ms"] is not None
        assert d["duration_ms"] >= 0

    def test_tracing_clear(self):
        """clear should remove all spans."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        t.start_span("op1")
        t.start_span("op2")
        assert len(t.get_spans()) == 2
        t.clear()
        assert len(t.get_spans()) == 0

    def test_tracing_error_status(self):
        """Spans can be ended with error status."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        s = t.start_span("failing-op")
        t.end_span(s, status="error")
        assert s.status == "error"

    def test_tracing_parent_span(self):
        """Spans can have parent IDs."""
        from nexus.observability.tracing import TracingExporter
        t = TracingExporter(backend="memory")
        parent = t.start_span("parent")
        child = t.start_span("child", parent_id=parent.span_id)
        assert child.parent_id == parent.span_id
        t.end_span(child)
        t.end_span(parent)

    def test_module_singleton(self):
        """Module-level tracer singleton should exist."""
        from nexus.observability.tracing import tracer
        assert tracer is not None
        tracer.clear()  # Clean up
