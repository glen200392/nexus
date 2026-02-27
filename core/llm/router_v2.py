"""
NEXUS v2 — Capability-based LLM Router.

Routes requests to the optimal model based on capabilities, privacy,
cost, latency, and health constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Optional

from nexus.core.llm.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelCapability:
    """Boolean capability flags for a model."""
    reasoning: bool = False
    code: bool = False
    vision: bool = False
    audio: bool = False
    tool_use: bool = False
    structured_output: bool = False
    extended_thinking: bool = False
    long_context: bool = False

    def has(self, name: str) -> bool:
        """Return True if the named capability is present and True."""
        return getattr(self, name, False)

    def matches(self, required: list[str]) -> tuple[bool, list[str]]:
        """Check whether all *required* capabilities are satisfied.

        Returns (all_matched, list_of_matched_names).
        """
        matched = [cap for cap in required if self.has(cap)]
        return len(matched) == len(required), matched


@dataclass
class ModelConfigV2:
    """Full configuration for a single model."""
    provider: str
    model_id: str
    display_name: str
    context_window: int
    max_output: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    capabilities: ModelCapability
    privacy_tiers: list[str] = field(default_factory=lambda: ["public"])
    is_local: bool = False
    avg_latency_ms: float = 500.0
    endpoint: str = ""


@dataclass
class RoutingRequestV2:
    """A request describing what the caller needs from an LLM."""
    task_type: str = "general"
    domain: str = "general"
    complexity: str = "medium"          # low | medium | high
    privacy_tier: str = "public"        # public | internal | confidential | restricted
    required_capabilities: list[str] = field(default_factory=list)
    estimated_input_tokens: int = 500
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    prefer_local: bool = False
    stream: bool = False


@dataclass
class RoutingDecisionV2:
    """Result of the routing pipeline."""
    primary: ModelConfigV2
    fallback: ModelConfigV2 | None = None
    reason: str = ""
    estimated_cost_usd: float = 0.0
    privacy_compliant: bool = True
    capabilities_match: list[str] = field(default_factory=list)
    latency_estimate_ms: float = 0.0
    alternative_models: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in model registry
# ---------------------------------------------------------------------------

def _build_default_registry() -> list[ModelConfigV2]:
    """Return the default set of known models."""
    return [
        ModelConfigV2(
            provider="ollama", model_id="qwen2.5:latest",
            display_name="Qwen 2.5", context_window=32_768,
            max_output=8_192, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
            capabilities=ModelCapability(reasoning=True, code=True, tool_use=True, structured_output=True),
            privacy_tiers=["public", "internal", "confidential", "restricted"],
            is_local=True, avg_latency_ms=200.0,
        ),
        ModelConfigV2(
            provider="ollama", model_id="qwen2.5:latest",
            display_name="Qwen 2.5 (32B alias)", context_window=32_768,
            max_output=8_192, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
            capabilities=ModelCapability(reasoning=True, code=True, tool_use=True, structured_output=True, long_context=True),
            privacy_tiers=["public", "internal", "confidential", "restricted"],
            is_local=True, avg_latency_ms=800.0,
        ),
        ModelConfigV2(
            provider="ollama", model_id="llama3.2:latest",
            display_name="Llama 3.2", context_window=128_000,
            max_output=8_192, cost_per_1k_input=0.0, cost_per_1k_output=0.0,
            capabilities=ModelCapability(reasoning=True, code=True, tool_use=True, long_context=True),
            privacy_tiers=["public", "internal", "confidential", "restricted"],
            is_local=True, avg_latency_ms=250.0,
        ),
        ModelConfigV2(
            provider="anthropic", model_id="claude-opus-4-6",
            display_name="Claude Opus 4.6", context_window=200_000,
            max_output=32_000, cost_per_1k_input=0.015, cost_per_1k_output=0.075,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, tool_use=True,
                structured_output=True, extended_thinking=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=3000.0,
        ),
        ModelConfigV2(
            provider="anthropic", model_id="claude-sonnet-4-6",
            display_name="Claude Sonnet 4.6", context_window=200_000,
            max_output=16_000, cost_per_1k_input=0.003, cost_per_1k_output=0.015,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, tool_use=True,
                structured_output=True, extended_thinking=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=1500.0,
        ),
        ModelConfigV2(
            provider="anthropic", model_id="claude-haiku-4-5",
            display_name="Claude Haiku 4.5", context_window=200_000,
            max_output=8_192, cost_per_1k_input=0.001, cost_per_1k_output=0.005,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, tool_use=True,
                structured_output=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=600.0,
        ),
        ModelConfigV2(
            provider="openai", model_id="gpt-4o",
            display_name="GPT-4o", context_window=128_000,
            max_output=16_384, cost_per_1k_input=0.005, cost_per_1k_output=0.015,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, audio=True,
                tool_use=True, structured_output=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=1200.0,
        ),
        ModelConfigV2(
            provider="openai", model_id="o4-mini",
            display_name="o4-mini", context_window=128_000,
            max_output=65_536, cost_per_1k_input=0.003, cost_per_1k_output=0.012,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, tool_use=True,
                structured_output=True, extended_thinking=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=2000.0,
        ),
        ModelConfigV2(
            provider="google", model_id="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash", context_window=1_000_000,
            max_output=8_192, cost_per_1k_input=0.0001, cost_per_1k_output=0.0004,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, audio=True,
                tool_use=True, structured_output=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=400.0,
        ),
        ModelConfigV2(
            provider="google", model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro", context_window=1_000_000,
            max_output=65_536, cost_per_1k_input=0.007, cost_per_1k_output=0.021,
            capabilities=ModelCapability(
                reasoning=True, code=True, vision=True, audio=True,
                tool_use=True, structured_output=True, extended_thinking=True, long_context=True,
            ),
            privacy_tiers=["public", "internal"],
            avg_latency_ms=2500.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class LLMRouterV2:
    """Capability-based LLM router with privacy, cost, and latency awareness."""

    def __init__(
        self,
        models: list[ModelConfigV2] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.models = models if models is not None else _build_default_registry()
        self.circuit_breaker = circuit_breaker

    # ----- public API -----

    def route(self, request: RoutingRequestV2) -> RoutingDecisionV2:
        """Run the full routing pipeline and return a decision."""
        candidates = list(self.models)

        candidates = self._filter_privacy(candidates, request.privacy_tier)
        candidates = self._filter_capabilities(candidates, request.required_capabilities)
        candidates = self._filter_context_window(candidates, request.estimated_input_tokens)
        candidates = self._filter_health(candidates)

        if request.max_latency_ms is not None:
            candidates = [m for m in candidates if m.avg_latency_ms <= request.max_latency_ms]

        if request.max_cost_usd is not None:
            candidates = [
                m for m in candidates
                if self._estimate_cost(m, request.estimated_input_tokens) <= request.max_cost_usd
            ]

        if not candidates:
            raise ValueError("No model satisfies the routing constraints")

        scored = self._score_candidates(candidates, request)
        scored.sort(key=lambda pair: pair[1], reverse=True)

        primary = scored[0][0]
        fallback = scored[1][0] if len(scored) > 1 else None
        alternatives = [m.model_id for m, _ in scored[2:6]]

        _, caps_matched = primary.capabilities.matches(request.required_capabilities)
        est_cost = self._estimate_cost(primary, request.estimated_input_tokens)

        return RoutingDecisionV2(
            primary=primary,
            fallback=fallback,
            reason=self._build_reason(primary, request),
            estimated_cost_usd=est_cost,
            privacy_compliant=request.privacy_tier in primary.privacy_tiers,
            capabilities_match=caps_matched,
            latency_estimate_ms=primary.avg_latency_ms,
            alternative_models=alternatives,
        )

    def explain(self, request: RoutingRequestV2) -> str:
        """Return a human-readable explanation of the routing decision."""
        try:
            decision = self.route(request)
        except ValueError as exc:
            return f"Routing failed: {exc}"

        lines = [
            f"Selected model: {decision.primary.display_name} ({decision.primary.model_id})",
            f"Provider: {decision.primary.provider}",
            f"Reason: {decision.reason}",
            f"Estimated cost: ${decision.estimated_cost_usd:.6f}",
            f"Privacy compliant: {decision.privacy_compliant}",
            f"Matched capabilities: {', '.join(decision.capabilities_match) or 'none required'}",
            f"Latency estimate: {decision.latency_estimate_ms:.0f} ms",
        ]
        if decision.fallback:
            lines.append(f"Fallback: {decision.fallback.display_name}")
        if decision.alternative_models:
            lines.append(f"Alternatives: {', '.join(decision.alternative_models)}")
        return "\n".join(lines)

    def list_models(
        self,
        privacy_tier: str | None = None,
        capability: str | None = None,
    ) -> list[ModelConfigV2]:
        """Return models filtered by optional privacy tier and/or capability."""
        result = list(self.models)
        if privacy_tier:
            result = [m for m in result if privacy_tier in m.privacy_tiers]
        if capability:
            result = [m for m in result if m.capabilities.has(capability)]
        return result

    # ----- pipeline stages -----

    def _filter_privacy(self, models: list[ModelConfigV2], tier: str) -> list[ModelConfigV2]:
        return [m for m in models if tier in m.privacy_tiers]

    def _filter_capabilities(self, models: list[ModelConfigV2], required: list[str]) -> list[ModelConfigV2]:
        if not required:
            return models
        return [m for m in models if m.capabilities.matches(required)[0]]

    def _filter_context_window(self, models: list[ModelConfigV2], tokens: int) -> list[ModelConfigV2]:
        return [m for m in models if m.context_window >= tokens]

    def _filter_health(self, models: list[ModelConfigV2]) -> list[ModelConfigV2]:
        if self.circuit_breaker is None:
            return models
        healthy = []
        for m in models:
            key = f"{m.provider}:{m.model_id}"
            if self.circuit_breaker.can_execute(key):
                healthy.append(m)
        return healthy or models  # fall back to all if everything is tripped

    # ----- scoring -----

    def _score_candidates(
        self, candidates: list[ModelConfigV2], request: RoutingRequestV2,
    ) -> list[tuple[ModelConfigV2, float]]:
        scored: list[tuple[ModelConfigV2, float]] = []
        for m in candidates:
            score = 0.0

            # Capability coverage
            if request.required_capabilities:
                _, matched = m.capabilities.matches(request.required_capabilities)
                score += 30 * (len(matched) / len(request.required_capabilities))
            else:
                score += 30

            # Cost efficiency (lower is better, max 25 pts)
            cost = self._estimate_cost(m, request.estimated_input_tokens)
            if cost == 0:
                score += 25
            elif request.max_cost_usd and request.max_cost_usd > 0:
                score += 25 * max(0, 1 - cost / request.max_cost_usd)
            else:
                score += max(0, 25 - cost * 1000)

            # Latency (lower is better, max 20 pts)
            if request.max_latency_ms and request.max_latency_ms > 0:
                score += 20 * max(0, 1 - m.avg_latency_ms / request.max_latency_ms)
            else:
                score += max(0, 20 - m.avg_latency_ms / 500)

            # Local preference
            if request.prefer_local and m.is_local:
                score += 15

            # Complexity bonus — prefer stronger models for high complexity
            cap_count = sum(
                1 for f in fields(m.capabilities) if getattr(m.capabilities, f.name)
            )
            if request.complexity == "high":
                score += cap_count * 1.5
            elif request.complexity == "medium":
                score += cap_count * 0.5

            scored.append((m, score))
        return scored

    # ----- helpers -----

    @staticmethod
    def _estimate_cost(model: ModelConfigV2, input_tokens: int) -> float:
        est_output = min(input_tokens, model.max_output)
        return (input_tokens / 1000) * model.cost_per_1k_input + (est_output / 1000) * model.cost_per_1k_output

    @staticmethod
    def _build_reason(model: ModelConfigV2, request: RoutingRequestV2) -> str:
        parts: list[str] = []
        if model.is_local:
            parts.append("local model (zero cost, full privacy)")
        else:
            parts.append(f"cloud model via {model.provider}")
        if request.required_capabilities:
            parts.append(f"satisfies {', '.join(request.required_capabilities)}")
        if request.complexity == "high":
            parts.append("chosen for high-complexity task")
        return "; ".join(parts) if parts else "best overall match"
