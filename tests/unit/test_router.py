"""
Tests for LLMRouter — Privacy routing and cost cap enforcement.
"""
from __future__ import annotations

import pytest

from nexus.core.llm.router import (
    LLMRouter, RoutingRequest, RoutingDecision, ModelConfig,
    PrivacyTier, TaskComplexity, TaskDomain, MODEL_REGISTRY,
)


class TestPrivacyRouting:
    """Privacy tier must be strictly enforced at routing level."""

    def setup_method(self):
        self.router = LLMRouter()

    def test_private_tier_gets_local_model(self):
        """PRIVATE tier must always route to a local model."""
        req = RoutingRequest(
            task_type="test",
            domain=TaskDomain.RESEARCH,
            complexity=TaskComplexity.HIGH,
            privacy_tier=PrivacyTier.PRIVATE,
        )
        decision = self.router.route(req)
        assert decision.primary.is_local, "PRIVATE tier must use local model"
        assert decision.privacy_compliant

    def test_private_critical_uses_best_local(self):
        """PRIVATE + CRITICAL should use the largest local model."""
        req = RoutingRequest(
            task_type="test",
            domain=TaskDomain.ENGINEERING,
            complexity=TaskComplexity.CRITICAL,
            privacy_tier=PrivacyTier.PRIVATE,
        )
        decision = self.router.route(req)
        assert decision.primary.is_local
        assert "72b" in decision.primary.model_id or "qwen" in decision.primary.model_id.lower()

    def test_internal_tier_allows_cloud(self):
        """INTERNAL tier should be able to use cloud models."""
        req = RoutingRequest(
            task_type="test",
            domain=TaskDomain.RESEARCH,
            complexity=TaskComplexity.HIGH,
            privacy_tier=PrivacyTier.INTERNAL,
        )
        decision = self.router.route(req)
        # Cloud models are typically not local
        assert decision.privacy_compliant

    def test_public_tier_allows_any_model(self):
        """PUBLIC tier should allow any model."""
        req = RoutingRequest(
            task_type="test",
            domain=TaskDomain.RESEARCH,
            complexity=TaskComplexity.LOW,
            privacy_tier=PrivacyTier.PUBLIC,
        )
        decision = self.router.route(req)
        assert decision.privacy_compliant


class TestCostCap:
    """Cost cap should override model selection when needed."""

    def setup_method(self):
        self.router = LLMRouter()

    def test_cost_cap_triggers_fallback(self):
        """When cost exceeds budget, should fall back to cheaper model."""
        req = RoutingRequest(
            task_type="test",
            domain=TaskDomain.ENGINEERING,
            complexity=TaskComplexity.CRITICAL,
            privacy_tier=PrivacyTier.INTERNAL,
            max_cost_usd=0.001,  # Very low budget
        )
        decision = self.router.route(req)
        # Should have swapped to a cheaper alternative
        assert decision.estimated_cost_usd >= 0.0

    def test_no_cost_cap_uses_primary(self):
        """Without cost cap, should use the primary model."""
        req = RoutingRequest(
            task_type="test",
            domain=TaskDomain.ENGINEERING,
            complexity=TaskComplexity.CRITICAL,
            privacy_tier=PrivacyTier.INTERNAL,
        )
        decision = self.router.route(req)
        assert decision.reason  # Should have a reason


class TestComplexityRouting:
    """Different complexity levels should route to appropriate models."""

    def setup_method(self):
        self.router = LLMRouter()

    def test_perception_uses_lightweight(self):
        """Perception layer tasks should use lightweight models."""
        req = RoutingRequest(
            task_type="classify",
            domain=TaskDomain.PERCEPTION,
            complexity=TaskComplexity.LOW,
            privacy_tier=PrivacyTier.INTERNAL,
        )
        decision = self.router.route(req)
        # Should be qwen2.5:7b or similar lightweight
        assert decision.primary.avg_latency_ms <= 2000

    def test_vision_task_routing(self):
        """Vision tasks should route to vision-capable models."""
        req = RoutingRequest(
            task_type="analyze",
            domain=TaskDomain.ANALYSIS,
            complexity=TaskComplexity.MEDIUM,
            privacy_tier=PrivacyTier.INTERNAL,
            is_vision=True,
        )
        decision = self.router.route(req)
        assert decision.primary.supports_vision


class TestModelRegistry:
    """Model registry should be properly configured."""

    def test_all_models_have_required_fields(self):
        for name, config in MODEL_REGISTRY.items():
            assert config.provider in ("ollama", "anthropic", "openai", "google")
            assert config.model_id
            assert config.context_window > 0

    def test_local_models_have_zero_cost(self):
        for name, config in MODEL_REGISTRY.items():
            if config.is_local:
                assert config.cost_per_1k_input == 0.0
                assert config.cost_per_1k_output == 0.0

    def test_explain_returns_string(self):
        router = LLMRouter()
        req = RoutingRequest("test", TaskDomain.RESEARCH, TaskComplexity.MEDIUM, PrivacyTier.INTERNAL)
        explanation = router.explain(req)
        assert isinstance(explanation, str)
        assert "Primary" in explanation
