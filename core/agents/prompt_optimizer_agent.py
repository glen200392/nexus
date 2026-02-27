"""
NEXUS Prompt Optimizer Agent
DSPy-inspired systematic prompt improvement.

Workflow:
  1. Load the current active prompt for a target agent
  2. Analyze weaknesses using prompt_optimizer.md system prompt
  3. Generate 3–5 concrete prompt variants (complete rewrites, not suggestions)
  4. Save each variant via prompt_versioning skill (as candidate versions)
  5. Return comparison table with predicted quality deltas
  6. Actual A/B testing happens naturally as tasks execute and record quality

Operations (context.metadata["operation"]):
  analyze      — diagnose weaknesses in the current prompt
  generate     — produce prompt variants and save them
  recommend    — review quality stats and recommend which version to promote
  full_cycle   — analyze + generate in one pass (default)
  promote_best — activate the version with highest avg_quality

Integration:
  Uses prompt_versioning skill (must be loaded in SkillRegistry)
  Reads from prompt_versioning.list() to find recorded quality data
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.prompt_optimizer")

_PROMPT_OPTIMIZER_MD = Path("config/prompts/system/prompt_optimizer.md")


class PromptOptimizerAgent(BaseAgent):
    agent_id   = "prompt_optimizer_agent"
    agent_name = "Prompt Optimizer Agent"
    description = (
        "DSPy-style systematic prompt optimization. Analyzes current system "
        "prompts, generates concrete variants with predicted quality deltas, "
        "saves candidates via prompt_versioning skill for natural A/B testing."
    )
    domain             = TaskDomain.ENGINEERING
    default_complexity = TaskComplexity.HIGH
    default_privacy    = PrivacyTier.INTERNAL

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _optimizer_system_prompt(self) -> str:
        if _PROMPT_OPTIMIZER_MD.exists():
            return _PROMPT_OPTIMIZER_MD.read_text(encoding="utf-8")
        return (
            "You are a prompt engineering specialist. Analyze prompts and generate "
            "improved variants. Return JSON with keys: analysis, variants (each with "
            "full_prompt, predicted_quality_delta, hypothesis), recommended_ab_test."
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        operation  = context.metadata.get("operation", "full_cycle")
        target_agent = context.metadata.get("target_agent_id", "master_orchestrator")

        # Load current prompt
        current_prompt = await self._load_current_prompt(target_agent, context)

        if operation == "analyze":
            return await self._analyze(current_prompt, target_agent, context)
        if operation == "recommend":
            return await self._recommend(target_agent, context)
        if operation == "promote_best":
            return await self._promote_best(target_agent, context)

        # Default: full_cycle (analyze + generate)
        return await self._full_cycle(current_prompt, target_agent, context)

    # ── Load current prompt ────────────────────────────────────────────────────

    async def _load_current_prompt(self, agent_id: str, context: AgentContext) -> str:
        """Try skill first, then fall back to config/prompts/system/<agent_id>.md."""
        # Try via skill
        if self.skill_registry:
            pv_skill = self.skill_registry.get("prompt_versioning")
            if pv_skill:
                result = await pv_skill.run(operation="get_active", agent_id=agent_id)
                if "content" in result:
                    return result["content"]

        # Fallback: load from .md file
        md_path = Path(f"config/prompts/system/{agent_id}.md")
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")

        return context.metadata.get("current_prompt", "")

    # ── Analyze ────────────────────────────────────────────────────────────────

    async def _analyze(self, current_prompt: str, agent_id: str,
                       context: AgentContext) -> AgentResult:
        if not current_prompt:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error=f"No active prompt found for '{agent_id}'. Use 'save' in prompt_versioning first.",
            )

        decision = self.route_llm(context)
        sample_tasks = context.metadata.get("sample_tasks", [])
        sample_txt   = "\n".join(f"- {t}" for t in sample_tasks[:5]) if sample_tasks else "(none provided)"

        resp = await self._llm.chat(
            messages=[Message("user",
                f"Analyze this system prompt for agent '{agent_id}':\n\n"
                f"```\n{current_prompt[:4000]}\n```\n\n"
                f"Sample tasks this agent handles:\n{sample_txt}\n\n"
                "Provide weakness analysis only (not variants yet)."
            )],
            model=decision.primary,
            system=self._optimizer_system_prompt(),
            privacy_tier=context.privacy_tier,
            temperature=0.4,
            max_tokens=1200,
        )
        analysis = self._parse_json(resp.content)
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={"agent_id": agent_id, "analysis": analysis, "current_prompt_length": len(current_prompt)},
            quality_score=0.8,
        )

    # ── Full cycle ─────────────────────────────────────────────────────────────

    async def _full_cycle(self, current_prompt: str, agent_id: str,
                          context: AgentContext) -> AgentResult:
        if not current_prompt:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error=f"No prompt found for agent '{agent_id}'.",
            )

        decision     = self.route_llm(context)
        sample_tasks = context.metadata.get("sample_tasks", [])
        sample_txt   = "\n".join(f"- {t}" for t in sample_tasks[:5]) if sample_tasks else "(none provided)"
        num_variants = int(context.metadata.get("num_variants", 3))

        resp = await self._llm.chat(
            messages=[Message("user",
                f"Optimize this system prompt for agent '{agent_id}'.\n\n"
                f"Current prompt:\n```\n{current_prompt[:3000]}\n```\n\n"
                f"Sample tasks:\n{sample_txt}\n\n"
                f"Generate {num_variants} complete prompt variants with quality delta predictions."
            )],
            model=decision.primary,
            system=self._optimizer_system_prompt(),
            privacy_tier=context.privacy_tier,
            temperature=0.6,
            max_tokens=3000,
        )

        data     = self._parse_json(resp.content)
        variants = data.get("variants", [])
        saved    = []

        # Save each variant via prompt_versioning skill
        if self.skill_registry:
            pv_skill = self.skill_registry.get("prompt_versioning")
            if pv_skill and variants:
                for i, variant in enumerate(variants):
                    prompt_text = variant.get("full_prompt", "")
                    if not prompt_text:
                        continue
                    save_result = await pv_skill.run(
                        operation="save",
                        agent_id=agent_id,
                        content=prompt_text,
                        changelog=(
                            f"Optimizer variant {variant.get('id', i+1)}: "
                            f"{variant.get('hypothesis', '')[:120]}"
                        ),
                        bump="minor",
                        tags=["optimizer_candidate", f"variant_{i+1}"],
                    )
                    if "version" in save_result:
                        variant["saved_as_version"] = save_result["version"]
                        saved.append(save_result["version"])

        await self.remember(
            content=f"Prompt optimization for {agent_id}: {len(variants)} variants generated. "
                    f"Saved: {saved}",
            context=context,
            doc_type=DocumentType.EPISODIC,
            tags=["prompt_optimizer", agent_id],
        )

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "agent_id":            agent_id,
                "variants_generated":  len(variants),
                "variants_saved":      saved,
                "variants":            variants,
                "recommended_ab_test": data.get("recommended_ab_test", ""),
                "analysis":            data.get("analysis", {}),
                "metrics_to_track":    data.get("metrics_to_track", ["quality_score"]),
                "next_step": (
                    f"Run tasks with these agents, then use prompt_versioning.compare "
                    f"to evaluate versions after at least 10 tasks each."
                ),
            },
            quality_score=0.85,
        )

    # ── Recommend ─────────────────────────────────────────────────────────────

    async def _recommend(self, agent_id: str, context: AgentContext) -> AgentResult:
        """Review quality data and recommend which version to promote."""
        if not self.skill_registry:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error="No skill_registry available",
            )
        pv_skill = self.skill_registry.get("prompt_versioning")
        if not pv_skill:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error="prompt_versioning skill not loaded",
            )

        versions = await pv_skill.run(operation="list", agent_id=agent_id)
        vs       = versions.get("versions", [])
        if not vs:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={"message": f"No versions found for '{agent_id}'"},
            )

        # Find best by avg_quality (minimum 3 tasks for statistical confidence)
        candidates = [v for v in vs if v.get("task_count", 0) >= 3]
        if not candidates:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={
                    "message":  "Not enough data yet. Need ≥3 tasks per version for recommendation.",
                    "versions": vs,
                },
            )

        best = max(candidates, key=lambda v: v.get("avg_quality", 0))
        active = next((v for v in vs if v.get("is_active")), None)
        improvement = 0.0
        if active:
            improvement = best["avg_quality"] - active.get("avg_quality", 0)

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "recommended_version": best["version"],
                "avg_quality":         best["avg_quality"],
                "task_count":          best["task_count"],
                "current_active":      active["version"] if active else None,
                "quality_improvement": round(improvement, 4),
                "should_promote":      improvement > 0.02,
                "all_versions":        vs,
            },
            quality_score=0.9,
        )

    # ── Promote best ──────────────────────────────────────────────────────────

    async def _promote_best(self, agent_id: str, context: AgentContext) -> AgentResult:
        """Activate the highest-quality version."""
        recommend_result = await self._recommend(agent_id, context)
        output = recommend_result.output or {}
        if not output.get("should_promote"):
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={"promoted": False, "reason": "No significant improvement found", **output},
            )
        if not self.skill_registry:
            return recommend_result
        pv_skill = self.skill_registry.get("prompt_versioning")
        if not pv_skill:
            return recommend_result

        version = output.get("recommended_version", "")
        result  = await pv_skill.run(operation="set_active", agent_id=agent_id, version=version)
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "promoted":       True,
                "new_version":    version,
                "quality_gain":   output.get("quality_improvement"),
                "activate_result": result,
            },
            quality_score=0.9,
        )

    def _parse_json(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"analysis": {"raw": raw[:500]}, "variants": []}
