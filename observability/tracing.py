"""
NEXUS Distributed Tracing
In-memory span collection compatible with OpenTelemetry concepts.
No external dependencies required.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """Represents a single tracing span."""

    trace_id: str
    span_id: str
    name: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def duration_ms(self) -> float | None:
        """Return span duration in milliseconds, or None if still open."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize span to a plain dict."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "attributes": dict(self.attributes),
            "status": self.status,
        }


class TracingExporter:
    """
    In-memory tracing exporter.

    Backends:
    - "memory": store spans in memory (default, useful for testing/inspection)
    - "console": print spans to stderr on end_span
    - "none": discard all spans
    """

    def __init__(self, backend: str = "console") -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._spans: list[Span] = []
        self._active_trace_id: str | None = None

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> Span:
        """Create and register a new span."""
        # Inherit trace_id from parent or generate new
        trace_id = self._active_trace_id or uuid.uuid4().hex[:16]
        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            name=name,
            parent_id=parent_id,
            start_time=time.time(),
            attributes=dict(attributes or {}),
        )
        self._active_trace_id = trace_id
        if self._backend != "none":
            with self._lock:
                self._spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        """Mark a span as ended."""
        span.end_time = time.time()
        span.status = status

        if self._backend == "console":
            import sys
            dur = span.duration_ms()
            print(
                f"[TRACE] {span.name} trace={span.trace_id} span={span.span_id} "
                f"status={status} duration_ms={dur:.2f}" if dur else
                f"[TRACE] {span.name} trace={span.trace_id} span={span.span_id} status={status}",
                file=sys.stderr,
            )

    def get_spans(self, trace_id: str | None = None) -> list[Span]:
        """Retrieve spans, optionally filtered by trace_id."""
        with self._lock:
            if trace_id is None:
                return list(self._spans)
            return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        """Remove all stored spans."""
        with self._lock:
            self._spans.clear()
        self._active_trace_id = None

    def export(self) -> list[dict[str, Any]]:
        """Export all spans as a list of dicts."""
        with self._lock:
            return [s.to_dict() for s in self._spans]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
tracer = TracingExporter(backend="memory")
