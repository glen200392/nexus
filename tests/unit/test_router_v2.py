"""
Tests for nexus.core.llm.router_v2.
"""

import pytest
from unittest.mock import MagicMock

from nexus.core.llm.router_v2 import (
    ModelCapability,
    ModelConfigV2,
    RoutingRequestV2,
    RoutingDecisionV2,
    LLMRouterV2,
)


# ---------------------------------------------------------------------------
# ModelCapability tests
# ---------------------------------------------------------------------------

class TestModelCapability:

    def test_has_true(self):
        cap = ModelCapability(reasoning=True, code=True)
        assert cap.has("reasoning") is True
        assert cap.has("code") is True

    def test_has_false(self):
        cap = ModelCapability()
        assert cap.has("vision") is False

    def test_has_unknown_field(self):
        cap = ModelCapability()
        assert cap.has("nonexistent") is False

    def test_matches_all(self):
        cap = ModelCapability(reasoning=True, code=True, vision=True)
        ok, matched = cap.matches(["reasoning", "code"])
        assert ok is True
        assert set(matched) == {"reasoning", "code"}

    def test_matches_partial(self):
        cap = ModelCapability(reasoning=True)
        ok, matched = cap.matches(["reasoning", "vision"])
        assert ok is False
        assert matched == ["reasoning"]

    def test_matches_empty(self):
        cap = ModelCapability()
        ok, matched = cap.matches([])
        assert ok is True
        assert matched == []


# ---------------------------------------------------------------------------
# LLMRouterV2 tests
# ---------------------------------------------------------------------------

class TestLLMRouterV2:

    def _make_router(self):
        return LLMRouterV2()

    def test_default_registry_has_models(self):
        router = self._make_router()
        assert len(router.models) >= 10

    def test_route_basic(self):
        router = self._make_router()
        req = RoutingRequestV2()
        decision = router.route(req)
        assert isinstance(decision, RoutingDecisionV2)
        assert decision.primary is not None
        assert decision.primary.model_id != ""

    def test_route_privacy_restricted(self):
        router = self._make_router()
        req = RoutingRequestV2(privacy_tier="restricted")
        decision = router.route(req)
        assert decision.primary.is_local is True
        assert "restricted" in decision.primary.privacy_tiers

    def test_route_privacy_confidential(self):
        router = self._make_router()
        req = RoutingRequestV2(privacy_tier="confidential")
        decision = router.route(req)
        assert "confidential" in decision.primary.privacy_tiers

    def test_route_requires_vision(self):
        router = self._make_router()
        req = RoutingRequestV2(required_capabilities=["vision"])
        decision = router.route(req)
        assert decision.primary.capabilities.has("vision")

    def test_route_requires_audio(self):
        router = self._make_router()
        req = RoutingRequestV2(required_capabilities=["audio"])
        decision = router.route(req)
        assert decision.primary.capabilities.has("audio")

    def test_route_requires_extended_thinking(self):
        router = self._make_router()
        req = RoutingRequestV2(required_capabilities=["extended_thinking"])
        decision = router.route(req)
        assert decision.primary.capabilities.has("extended_thinking")

    def test_route_prefer_local(self):
        router = self._make_router()
        req = RoutingRequestV2(prefer_local=True)
        decision = router.route(req)
        # Local preference should boost local models
        assert decision.primary.is_local is True

    def test_route_no_candidates_raises(self):
        router = LLMRouterV2(models=[])
        req = RoutingRequestV2()
        with pytest.raises(ValueError, match="No model"):
            router.route(req)

    def test_route_max_latency(self):
        router = self._make_router()
        req = RoutingRequestV2(max_latency_ms=300)
        decision = router.route(req)
        assert decision.primary.avg_latency_ms <= 300

    def test_explain_returns_string(self):
        router = self._make_router()
        req = RoutingRequestV2()
        explanation = router.explain(req)
        assert isinstance(explanation, str)
        assert "Selected model" in explanation

    def test_explain_failed_routing(self):
        router = LLMRouterV2(models=[])
        req = RoutingRequestV2()
        explanation = router.explain(req)
        assert "failed" in explanation.lower()

    def test_list_models_all(self):
        router = self._make_router()
        models = router.list_models()
        assert len(models) >= 10

    def test_list_models_by_privacy(self):
        router = self._make_router()
        restricted = router.list_models(privacy_tier="restricted")
        assert all("restricted" in m.privacy_tiers for m in restricted)
        assert len(restricted) < len(router.models)

    def test_list_models_by_capability(self):
        router = self._make_router()
        vision_models = router.list_models(capability="vision")
        assert all(m.capabilities.has("vision") for m in vision_models)

    def test_route_with_circuit_breaker(self):
        cb = MagicMock()
        cb.can_execute.return_value = True
        router = LLMRouterV2(circuit_breaker=cb)
        req = RoutingRequestV2()
        decision = router.route(req)
        assert decision.primary is not None

    def test_route_circuit_breaker_filters_unhealthy(self):
        cb = MagicMock()
        # Block all models by returning False
        cb.can_execute.return_value = False
        router = LLMRouterV2(circuit_breaker=cb)
        req = RoutingRequestV2()
        # Should fall back to all models when everything is tripped
        decision = router.route(req)
        assert decision.primary is not None

    def test_decision_has_fallback(self):
        router = self._make_router()
        req = RoutingRequestV2()
        decision = router.route(req)
        # With 10+ models there should be a fallback
        assert decision.fallback is not None

    def test_decision_estimated_cost(self):
        router = self._make_router()
        req = RoutingRequestV2(estimated_input_tokens=1000)
        decision = router.route(req)
        assert decision.estimated_cost_usd >= 0
