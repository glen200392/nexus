"""
NEXUS LLM Router — Layer 0 Foundation
Routes tasks to the appropriate LLM based on complexity, privacy tier,
domain, and cost constraints.
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("nexus.llm.router")


# ─── Enums ──────────────────────────────────────────────────────────────────

class PrivacyTier(str, Enum):
    PRIVATE  = "PRIVATE"   # Local-only. Zero data egress.
    INTERNAL = "INTERNAL"  # Cloud OK, no PII in prompts.
    PUBLIC   = "PUBLIC"    # Any model, standard data.


class TaskComplexity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class TaskDomain(str, Enum):
    RESEARCH     = "research"
    ENGINEERING  = "engineering"
    OPERATIONS   = "operations"
    CREATIVE     = "creative"
    ANALYSIS     = "analysis"
    PERCEPTION   = "perception"   # Layer 2 classification tasks
    ORCHESTRATION = "orchestration"  # Layer 3 planning tasks


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RoutingRequest:
    task_type: str
    domain: TaskDomain
    complexity: TaskComplexity
    privacy_tier: PrivacyTier
    is_vision: bool = False
    is_batch: bool = False
    requires_streaming: bool = False
    max_cost_usd: Optional[float] = None   # None = no cap
    preferred_latency_ms: Optional[int] = None


@dataclass
class ModelConfig:
    provider: str       # "ollama" | "anthropic" | "openai" | "google"
    model_id: str       # e.g. "qwen2.5:32b", "claude-sonnet-4-6"
    context_window: int
    cost_per_1k_input:  float  # USD (0.0 for local)
    cost_per_1k_output: float
    avg_latency_ms: int
    supports_vision: bool = False
    supports_tools: bool = True
    is_local: bool = False
    endpoint: Optional[str] = None  # Ollama URL for local models

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass
class RoutingDecision:
    primary: ModelConfig
    fallback: ModelConfig
    reason: str
    estimated_cost_usd: float
    privacy_compliant: bool
    timestamp: float = field(default_factory=time.time)


# ─── Model Registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelConfig] = {

    # ── Local Models (Ollama) ──────────────────────────────────────────────
    "qwen2.5:7b": ModelConfig(
        provider="ollama", model_id="qwen2.5:7b",
        context_window=32_768,
        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        avg_latency_ms=800,
        is_local=True,
        endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),
    "qwen2.5:32b": ModelConfig(
        provider="ollama", model_id="qwen2.5:32b",
        context_window=32_768,
        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        avg_latency_ms=2_500,
        is_local=True,
        endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),
    "qwen2.5:72b": ModelConfig(
        provider="ollama", model_id="qwen2.5:72b",
        context_window=32_768,
        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        avg_latency_ms=6_000,
        is_local=True,
        endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),
    "llama3.1:8b": ModelConfig(
        provider="ollama", model_id="llama3.1:8b",
        context_window=128_000,
        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        avg_latency_ms=700,
        is_local=True,
        endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),
    "phi4:14b": ModelConfig(
        provider="ollama", model_id="phi4:14b",
        context_window=16_384,
        cost_per_1k_input=0.0, cost_per_1k_output=0.0,
        avg_latency_ms=1_200,
        is_local=True,
        endpoint=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ),

    # ── Anthropic Claude ─────────────────────────────────────────────────────
    "claude-opus-4-6": ModelConfig(
        provider="anthropic", model_id="claude-opus-4-6",
        context_window=200_000,
        cost_per_1k_input=0.015, cost_per_1k_output=0.075,
        avg_latency_ms=4_000,
        supports_vision=True, supports_tools=True,
    ),
    "claude-sonnet-4-6": ModelConfig(
        provider="anthropic", model_id="claude-sonnet-4-6",
        context_window=200_000,
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
        avg_latency_ms=2_000,
        supports_vision=True, supports_tools=True,
    ),
    "claude-haiku-4-5": ModelConfig(
        provider="anthropic", model_id="claude-haiku-4-5-20251001",
        context_window=200_000,
        cost_per_1k_input=0.00025, cost_per_1k_output=0.00125,
        avg_latency_ms=600,
        supports_vision=True, supports_tools=True,
    ),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4o": ModelConfig(
        provider="openai", model_id="gpt-4o",
        context_window=128_000,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
        avg_latency_ms=2_500,
        supports_vision=True, supports_tools=True,
    ),

    # ── Google Gemini ────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelConfig(
        provider="google", model_id="gemini-2.0-flash",
        context_window=1_000_000,
        cost_per_1k_input=0.000075, cost_per_1k_output=0.0003,
        avg_latency_ms=500,
        supports_vision=True, supports_tools=True,
    ),
}


# ─── Routing Rules ───────────────────────────────────────────────────────────
# Priority-ordered list of (condition_fn, primary_model_key, fallback_model_key, reason)

_ROUTING_RULES: list[tuple] = [
    # Vision tasks → GPT-4o (best multimodal), fallback Claude
    (lambda r: r.is_vision and r.privacy_tier != PrivacyTier.PRIVATE,
     "gpt-4o", "claude-sonnet-4-6", "Vision task → GPT-4o"),

    # PRIVATE tier: always local, regardless of complexity
    (lambda r: r.privacy_tier == PrivacyTier.PRIVATE and r.complexity == TaskComplexity.CRITICAL,
     "qwen2.5:72b", "qwen2.5:32b", "PRIVATE + CRITICAL → best local"),
    (lambda r: r.privacy_tier == PrivacyTier.PRIVATE and r.complexity == TaskComplexity.HIGH,
     "qwen2.5:72b", "qwen2.5:32b", "PRIVATE + HIGH → large local"),
    (lambda r: r.privacy_tier == PrivacyTier.PRIVATE and r.complexity == TaskComplexity.MEDIUM,
     "qwen2.5:32b", "qwen2.5:7b", "PRIVATE + MEDIUM → mid local"),
    (lambda r: r.privacy_tier == PrivacyTier.PRIVATE,
     "qwen2.5:7b", "llama3.1:8b", "PRIVATE default → small local"),

    # Perception / classification: always lightweight (Layer 2)
    (lambda r: r.domain == TaskDomain.PERCEPTION,
     "qwen2.5:7b", "claude-haiku-4-5", "Perception layer → lightweight"),

    # Critical decisions → Claude Opus (best reasoning)
    (lambda r: r.complexity == TaskComplexity.CRITICAL,
     "claude-opus-4-6", "claude-sonnet-4-6", "CRITICAL → Claude Opus"),

    # High-complexity engineering → Claude Sonnet (excellent code)
    (lambda r: r.complexity == TaskComplexity.HIGH and r.domain == TaskDomain.ENGINEERING,
     "claude-sonnet-4-6", "qwen2.5:72b", "HIGH engineering → Claude Sonnet"),

    # High-complexity research → Claude Sonnet
    (lambda r: r.complexity == TaskComplexity.HIGH,
     "claude-sonnet-4-6", "qwen2.5:72b", "HIGH task → Claude Sonnet"),

    # Batch + low complexity → Gemini Flash (cost-optimized)
    (lambda r: r.is_batch and r.complexity == TaskComplexity.LOW,
     "gemini-2.0-flash", "claude-haiku-4-5", "Batch low-complexity → Gemini Flash"),

    # Medium complexity: Cloud Sonnet for quality
    (lambda r: r.complexity == TaskComplexity.MEDIUM,
     "claude-sonnet-4-6", "qwen2.5:32b", "MEDIUM → Claude Sonnet"),

    # Low complexity default
    (lambda r: True,
     "claude-haiku-4-5", "qwen2.5:7b", "Default → Claude Haiku"),
]


# ─── Router ──────────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Routes RoutingRequest → RoutingDecision.

    Usage:
        router = LLMRouter()
        decision = router.route(RoutingRequest(
            task_type="summarize",
            domain=TaskDomain.RESEARCH,
            complexity=TaskComplexity.MEDIUM,
            privacy_tier=PrivacyTier.INTERNAL,
        ))
        # decision.primary → ModelConfig for claude-sonnet-4-6
    """

    def __init__(self, override_rules: Optional[list] = None):
        self._rules = override_rules or _ROUTING_RULES

    def route(self, request: RoutingRequest) -> RoutingDecision:
        for condition, primary_key, fallback_key, reason in self._rules:
            try:
                if condition(request):
                    primary  = MODEL_REGISTRY[primary_key]
                    fallback = MODEL_REGISTRY[fallback_key]

                    # Cost cap override: if primary exceeds budget, use fallback
                    if (request.max_cost_usd is not None
                            and not primary.is_local
                            and primary.cost_per_1k_input > request.max_cost_usd / 100):
                        logger.info(
                            "Cost cap triggered: %s → fallback %s",
                            primary.display_name, fallback.display_name,
                        )
                        primary, fallback = fallback, primary
                        reason += " [cost-capped to fallback]"

                    # Latency override
                    if (request.preferred_latency_ms is not None
                            and primary.avg_latency_ms > request.preferred_latency_ms * 1.5):
                        logger.info(
                            "Latency override: %s too slow, using fallback %s",
                            primary.display_name, fallback.display_name,
                        )
                        primary, fallback = fallback, primary
                        reason += " [latency-shifted to fallback]"

                    return RoutingDecision(
                        primary=primary,
                        fallback=fallback,
                        reason=reason,
                        estimated_cost_usd=self._estimate_cost(primary),
                        privacy_compliant=self._is_privacy_compliant(primary, request),
                    )
            except Exception as exc:
                logger.warning("Rule evaluation error: %s", exc)
                continue

        # Should never reach here, but just in case
        default = MODEL_REGISTRY["qwen2.5:7b"]
        return RoutingDecision(
            primary=default, fallback=default,
            reason="Fallthrough default → local safe",
            estimated_cost_usd=0.0,
            privacy_compliant=True,
        )

    def _estimate_cost(self, model: ModelConfig, assumed_tokens: int = 2000) -> float:
        if model.is_local:
            return 0.0
        return (assumed_tokens / 1000) * (model.cost_per_1k_input + model.cost_per_1k_output)

    def _is_privacy_compliant(self, model: ModelConfig, request: RoutingRequest) -> bool:
        if request.privacy_tier == PrivacyTier.PRIVATE and not model.is_local:
            return False
        return True

    def explain(self, request: RoutingRequest) -> str:
        """Return a human-readable explanation of routing decision."""
        decision = self.route(request)
        lines = [
            f"Task: {request.task_type} | Domain: {request.domain.value} | "
            f"Complexity: {request.complexity.value} | Privacy: {request.privacy_tier.value}",
            f"→ Primary:  {decision.primary.display_name}",
            f"→ Fallback: {decision.fallback.display_name}",
            f"→ Reason:   {decision.reason}",
            f"→ Est. cost: ${decision.estimated_cost_usd:.4f}",
            f"→ Privacy-compliant: {decision.privacy_compliant}",
        ]
        return "\n".join(lines)


# ─── Global singleton ────────────────────────────────────────────────────────
_router_instance: Optional[LLMRouter] = None

def get_router() -> LLMRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = LLMRouter()
    cases = [
        RoutingRequest("classify_intent",  TaskDomain.PERCEPTION,    TaskComplexity.LOW,      PrivacyTier.PRIVATE),
        RoutingRequest("code_review",      TaskDomain.ENGINEERING,   TaskComplexity.HIGH,     PrivacyTier.INTERNAL),
        RoutingRequest("strategic_plan",   TaskDomain.ORCHESTRATION, TaskComplexity.CRITICAL, PrivacyTier.INTERNAL),
        RoutingRequest("batch_summarize",  TaskDomain.RESEARCH,      TaskComplexity.LOW,      PrivacyTier.PUBLIC, is_batch=True),
        RoutingRequest("analyze_chart",    TaskDomain.ANALYSIS,      TaskComplexity.MEDIUM,   PrivacyTier.INTERNAL, is_vision=True),
    ]
    for req in cases:
        print(router.explain(req))
        print()
