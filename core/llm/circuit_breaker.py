"""
NEXUS Circuit Breaker — Provider Health Tracking (v2)
Prevents cascading failures by tracking provider health and temporarily
disabling unhealthy providers.

States:
  CLOSED   → Normal operation, requests flow through
  OPEN     → Provider disabled, all requests fail-fast
  HALF_OPEN → Testing recovery, limited requests allowed

Transitions:
  CLOSED → OPEN: failure_threshold consecutive failures
  OPEN → HALF_OPEN: recovery_timeout elapsed
  HALF_OPEN → CLOSED: success_threshold successes
  HALF_OPEN → OPEN: any failure
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("nexus.circuit_breaker")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ProviderHealth:
    """Health metrics for a single LLM provider."""
    provider: str
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_calls: int = 0
    total_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_error: str = ""
    opened_at: float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_failures / self.total_calls

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "failure_rate": round(self.failure_rate, 3),
            "last_error": self.last_error,
        }


class CircuitBreaker:
    """
    Tracks health of LLM providers and circuit-breaks on failures.

    Usage:
        cb = CircuitBreaker()
        if cb.is_available("anthropic"):
            try:
                result = await call_anthropic(...)
                cb.record_success("anthropic")
            except Exception:
                cb.record_failure("anthropic")
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._providers: dict[str, ProviderHealth] = {}

    def _get_health(self, provider: str) -> ProviderHealth:
        if provider not in self._providers:
            self._providers[provider] = ProviderHealth(provider=provider)
        return self._providers[provider]

    def is_available(self, provider: str) -> bool:
        """Check if a provider is available for requests."""
        health = self._get_health(provider)

        if health.state == CircuitState.CLOSED:
            return True

        if health.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            elapsed = time.time() - health.opened_at
            if elapsed >= self.recovery_timeout:
                health.state = CircuitState.HALF_OPEN
                health.consecutive_successes = 0
                logger.info("Circuit breaker %s: OPEN → HALF_OPEN (recovery timeout elapsed)",
                            provider)
                return True
            return False

        # HALF_OPEN: allow limited requests for testing
        return True

    def record_success(self, provider: str) -> None:
        """Record a successful call."""
        health = self._get_health(provider)
        health.total_calls += 1
        health.consecutive_failures = 0
        health.consecutive_successes += 1
        health.last_success_time = time.time()

        if health.state == CircuitState.HALF_OPEN:
            if health.consecutive_successes >= self.success_threshold:
                health.state = CircuitState.CLOSED
                logger.info("Circuit breaker %s: HALF_OPEN → CLOSED (recovered)", provider)

    def record_failure(self, provider: str, error: str = "") -> None:
        """Record a failed call."""
        health = self._get_health(provider)
        health.total_calls += 1
        health.total_failures += 1
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.last_failure_time = time.time()
        health.last_error = error

        if health.state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN → back to OPEN
            health.state = CircuitState.OPEN
            health.opened_at = time.time()
            logger.warning("Circuit breaker %s: HALF_OPEN → OPEN (failure during recovery)", provider)

        elif health.state == CircuitState.CLOSED:
            if health.consecutive_failures >= self.failure_threshold:
                health.state = CircuitState.OPEN
                health.opened_at = time.time()
                logger.warning("Circuit breaker %s: CLOSED → OPEN (%d consecutive failures)",
                               provider, health.consecutive_failures)

    def get_health(self, provider: str) -> ProviderHealth:
        return self._get_health(provider)

    def get_all_health(self) -> dict[str, ProviderHealth]:
        return dict(self._providers)

    def reset(self, provider: str) -> None:
        """Manually reset a provider's circuit breaker."""
        if provider in self._providers:
            self._providers[provider] = ProviderHealth(provider=provider)
            logger.info("Circuit breaker %s: manually reset", provider)
