"""
NEXUS Critic Agent — Quality Gate
Scores outputs from other agents (0.0–1.0).
Used in feedback loops and adversarial workflows.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity

logger = logging.getLogger("nexus.agents.critic")


@dataclass
class CritiqueResult:
    score: float                          # 0.0–1.0 overall quality
    dimension_scores: dict[str, float]   # per-dimension breakdown
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    verdict: str                          # "accept" | "revise" | "reject"


class CriticAgent(BaseAgent):
    agent_id   = "critic_agent"
    agent_name = "Quality Critic Agent"
    description = "Scores agent outputs and provides improvement feedback"
    domain     = TaskDomain.RESEARCH    # Cross-domain
    default_complexity = TaskComplexity.LOW

    # Default rubric — overridden per-swarm in YAML config
    DEFAULT_RUBRIC: dict[str, float] = {
        "accuracy":      0.35,
        "completeness":  0.25,
        "clarity":       0.20,
        "usefulness":    0.20,
    }

    CODE_RUBRIC: dict[str, float] = {
        "correctness":   0.40,
        "test_coverage": 0.25,
        "security":      0.20,
        "readability":   0.15,
    }

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rubric: Optional[dict[str, float]] = None,
        accept_threshold: float = 0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm      = llm_client or get_client()
        self.rubric    = rubric or self.DEFAULT_RUBRIC
        self.threshold = accept_threshold

    def _build_system_prompt(self, context: AgentContext) -> str:
        rubric_lines = "\n".join(
            f"  - {dim} (weight: {w:.0%})" for dim, w in self.rubric.items()
        )
        return (
            f"You are a rigorous quality evaluator. Score the given output against this rubric:\n"
            f"{rubric_lines}\n\n"
            "Be strict but fair. A score of 1.0 means publication-ready.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "dimension_scores": {"accuracy": 0.8, ...},\n'
            '  "overall_score": 0.82,\n'
            '  "strengths": ["point 1", ...],\n'
            '  "weaknesses": ["point 1", ...],\n'
            '  "suggestions": ["specific actionable improvement", ...],\n'
            '  "verdict": "accept|revise|reject"\n'
            "}"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        # The output to critique is the last assistant message in context
        output_to_critique = ""
        for msg in reversed(context.history):
            if msg["role"] == "assistant":
                output_to_critique = msg["content"]
                break

        if not output_to_critique:
            output_to_critique = context.user_message

        # Use fast model for critique (cost control)
        from nexus.core.llm.router import MODEL_REGISTRY
        if context.privacy_tier.value == "PRIVATE":
            model = MODEL_REGISTRY["qwen2.5:7b"]
        else:
            model = MODEL_REGISTRY["claude-haiku-4-5"]

        messages = [
            Message(
                "user",
                f"Original task: {context.user_message}\n\n"
                f"Output to evaluate:\n{output_to_critique[:3000]}\n\n"
                "Evaluate this output against the rubric."
            )
        ]

        llm_resp = await self._llm.chat(
            messages=messages,
            model=model,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.1,
            max_tokens=1024,
        )

        critique = self._parse_critique(llm_resp.content)

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=critique.__dict__,
            quality_score=critique.score,
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
            cost_usd=llm_resp.cost_usd,
            llm_used=model.display_name,
        )

    def _parse_critique(self, raw: str) -> CritiqueResult:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                scores = data.get("dimension_scores", {})
                # Compute weighted score
                overall = data.get("overall_score")
                if overall is None:
                    overall = sum(
                        scores.get(dim, 0.5) * weight
                        for dim, weight in self.rubric.items()
                    )
                verdict = data.get("verdict", "")
                if not verdict:
                    verdict = "accept" if overall >= self.threshold else "revise"
                return CritiqueResult(
                    score=min(1.0, max(0.0, float(overall))),
                    dimension_scores=scores,
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    suggestions=data.get("suggestions", []),
                    verdict=verdict,
                )
            except Exception:
                pass
        return CritiqueResult(
            score=0.5,
            dimension_scores={},
            strengths=[],
            weaknesses=["Could not parse critique"],
            suggestions=["Retry with clearer output"],
            verdict="revise",
        )
