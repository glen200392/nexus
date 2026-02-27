"""
NEXUS Structured Logging
Provides JSON-formatted structured logging with optional structlog support.
Falls back to stdlib logging with a JSON formatter when structlog is unavailable.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Try optional structlog; fall back gracefully
# ---------------------------------------------------------------------------
try:
    import structlog  # type: ignore
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

# ---------------------------------------------------------------------------
# JSON Formatter (stdlib fallback)
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "caller": f"{record.pathname}:{record.lineno}",
        }
        # Merge any extra context bound via `get_logger`
        if hasattr(record, "_nexus_context"):
            log_entry.update(record._nexus_context)
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class _ColorFormatter(logging.Formatter):
    """Human-friendly colored output for development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = self.formatTime(record, self.datefmt)
        ctx = ""
        if hasattr(record, "_nexus_context"):
            ctx = " " + " ".join(f"{k}={v}" for k, v in record._nexus_context.items())
        return (
            f"{color}{ts} [{record.levelname:>8s}]{self.RESET} "
            f"{record.name}: {record.getMessage()}{ctx}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONFIGURED = False


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """
    Configure NEXUS logging.

    Parameters
    ----------
    log_level : str
        Python log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json_format : bool
        If True, use JSON output (production). If False, use colored dev output.
    """
    global _CONFIGURED

    if _HAS_STRUCTLOG:
        processors = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
        ]
        if json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        root = logging.getLogger()
        # Remove existing handlers to avoid duplicates
        for h in root.handlers[:]:
            root.removeHandler(h)

        handler = logging.StreamHandler(sys.stderr)
        if json_format:
            handler.setFormatter(_JSONFormatter())
        else:
            handler.setFormatter(_ColorFormatter())
        root.addHandler(handler)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    _CONFIGURED = True


class _ContextLogger:
    """Thin wrapper around stdlib Logger that carries bound context."""

    def __init__(self, logger: logging.Logger, context: dict[str, Any]) -> None:
        self._logger = logger
        self._context = dict(context)

    def _log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        record = self._logger.makeRecord(
            self._logger.name, level, "(nexus)", 0, msg, args, None
        )
        record._nexus_context = self._context  # type: ignore[attr-defined]
        self._logger.handle(record)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def bind(self, **kwargs: Any) -> _ContextLogger:
        new_ctx = {**self._context, **kwargs}
        return _ContextLogger(self._logger, new_ctx)


def get_logger(name: str, **context: Any) -> _ContextLogger:
    """
    Return a logger with bound context (task_id, session_id, agent_id, etc.).
    """
    if not _CONFIGURED:
        setup_logging()
    logger = logging.getLogger(name)
    return _ContextLogger(logger, context)


def bind_context(**kwargs: Any) -> None:
    """
    Utility to add context to the root nexus logger.
    Primarily useful when structlog is available; with stdlib this is a no-op
    since context is bound per-logger instance via get_logger().
    """
    if _HAS_STRUCTLOG:
        structlog.contextvars.bind_contextvars(**kwargs)
