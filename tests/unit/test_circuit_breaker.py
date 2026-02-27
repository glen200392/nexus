"""
Tests for nexus.core.llm.circuit_breaker.CircuitBreaker.
"""

import time
import pytest
from nexus.core.llm.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:

    def test_initial_state_allows_execution(self):
        cb = CircuitBreaker()
        assert cb.is_available("test-service") is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("svc")
        cb.record_failure("svc")
        assert cb.is_available("svc") is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=9999)
        for _ in range(3):
            cb.record_failure("svc")
        assert cb.is_available("svc") is False
        assert cb.get_health("svc").state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("svc")
        cb.record_failure("svc")
        cb.record_success("svc")
        # After a success, failures reset — still closed
        cb.record_failure("svc")
        cb.record_failure("svc")
        assert cb.is_available("svc") is True

    def test_independent_services(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=9999)
        cb.record_failure("a")
        cb.record_failure("a")
        assert cb.is_available("a") is False
        assert cb.is_available("b") is True

    def test_record_success_on_unknown_service(self):
        cb = CircuitBreaker()
        cb.record_success("unknown")
        assert cb.is_available("unknown") is True

    def test_multiple_services_isolation(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=9999)
        cb.record_failure("x")
        cb.record_failure("y")
        assert cb.is_available("x") is True
        assert cb.is_available("y") is True
        cb.record_failure("x")
        assert cb.is_available("x") is False
        assert cb.is_available("y") is True

    def test_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01, success_threshold=1)
        cb.record_failure("svc")
        cb.record_failure("svc")
        assert cb.get_health("svc").state == CircuitState.OPEN

        time.sleep(0.02)  # Wait for recovery timeout
        assert cb.is_available("svc") is True  # Should transition to HALF_OPEN
        assert cb.get_health("svc").state == CircuitState.HALF_OPEN

        cb.record_success("svc")
        assert cb.get_health("svc").state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure("svc")
        cb.record_failure("svc")

        time.sleep(0.02)
        cb.is_available("svc")  # Trigger HALF_OPEN
        cb.record_failure("svc")
        assert cb.get_health("svc").state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure("svc")
        assert cb.is_available("svc") is False
        cb.reset("svc")
        assert cb.is_available("svc") is True

    def test_get_all_health(self):
        cb = CircuitBreaker()
        cb.record_success("a")
        cb.record_failure("b")
        health = cb.get_all_health()
        assert "a" in health
        assert "b" in health
