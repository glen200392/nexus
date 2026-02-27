"""
NEXUS Memory Agent — Knowledge Consolidation & Distillation
Inspired by neuroscience: during "sleep" (idle time), the brain
consolidates short-term episodic memories into long-term semantic knowledge.

This agent:
  1. CONSOLIDATE — scan recent episodic records, extract key learnings
  2. DISTILL     — compress multiple related records into one dense fact
  3. FORGET      — apply temporal decay, remove low-quality/expired records
  4. REFLECT     — find patterns across task history, generate meta-insights
  5. REPORT      — return memory health statistics

Always runs with PRIVATE privacy tier (memory contains personal context).
Uses qwen2.5:32b locally (never sends memory to cloud).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier, MODEL_REGISTRY
from nexus.knowledge.rag.schema import (
    DocumentType, MemoryRecord, MemoryType, RetrievalConfig, RetrievalMode
)

logger = logging.getLogger("nexus.agents.memory")


@dataclass
class MemoryHealthReport:
    total_records:      int = 0
    episodic_count:     int = 0
    semantic_count:     int = 0
    avg_quality_score:  float = 0.0
    distilled_count:    int = 0    # records successfully distilled
    forgotten_count:    int = 0    # records pruned
    insights_found:     int = 0
    top_domains:        list[str] = field(default_factory=list)
    recommendations:    list[str] = field(default_factory=list)


class MemoryAgent(BaseAgent):
    agent_id   = "memory_agent"
    agent_name = "Memory Consolidation Agent"
    description = (
        "Consolidates episodic memory into semantic knowledge. "
        "Runs distillation, temporal decay, and pattern reflection. "
        "Always uses local models — never sends memory to cloud."
    )
    domain           = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.MEDIUM
    default_privacy  = PrivacyTier.PRIVATE    # Memory is always private

    # Memory model: always local, never cloud
    _MEMORY_MODEL = "qwen2.5:32b"

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a memory consolidation system. Your role is to distill multiple "
            "related experiences into compact, high-quality knowledge.\n\n"
            "When given a set of task records:\n"
            "1. Extract the key learnings (what worked, what didn't)\n"
            "2. Identify recurring patterns\n"
            "3. Formulate reusable principles\n"
            "4. Flag anything surprising or worth remembering\n\n"
            "Return compact JSON:\n"
            "{\n"
            '  "distilled_knowledge": "2-3 sentences of dense, reusable insight",\n'
            '  "patterns": ["pattern 1", "pattern 2"],\n'
            '  "principles": ["principle that applies broadly"],\n'
            '  "surprises": ["anything unexpected"],\n'
            '  "quality_score": 0.85\n'
            "}"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        # Force PRIVATE regardless of what was passed
        context.privacy_tier = PrivacyTier.PRIVATE

        operation = context.metadata.get("operation", "consolidate")
        report    = MemoryHealthReport()

        if operation == "consolidate":
            await self._consolidate(context, report)
        elif operation == "forget":
            await self._apply_decay(context, report)
        elif operation == "reflect":
            await self._reflect(context, report)
        elif operation == "full":
            await self._consolidate(context, report)
            await self._apply_decay(context, report)
            await self._reflect(context, report)
        else:
            # Default: full maintenance cycle
            await self._consolidate(context, report)
            await self._apply_decay(context, report)

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=report.__dict__,
            quality_score=0.75,
            llm_used=self._MEMORY_MODEL,
        )

    # ── Consolidation ─────────────────────────────────────────────────────────

    async def _consolidate(self, context: AgentContext, report: MemoryHealthReport) -> None:
        """
        Read recent episodic records, group by domain,
        distill each group into a semantic fact.
        """
        if self.memory_store is None:
            logger.warning("No memory store; skipping consolidation")
            return

        # Retrieve recent episodic memory
        episodic_cfg = RetrievalConfig(
            mode=RetrievalMode.TEMPORAL,
            top_k=20,
            score_threshold=0.0,    # Get all recent, regardless of quality
            apply_temporal_decay=False,
            filters={"memory_type": MemoryType.EPISODIC.value},
        )
        recent = await self.memory_store.retrieve(
            query="recent task execution",
            config=episodic_cfg,
            privacy_tier=PrivacyTier.PRIVATE,
        )
        report.episodic_count = len(recent)
        if not recent:
            return

        # Group by domain
        domain_groups: dict[str, list[MemoryRecord]] = {}
        for rec in recent:
            d = rec.domain or "general"
            domain_groups.setdefault(d, []).append(rec)

        report.top_domains = sorted(domain_groups.keys(),
                                    key=lambda d: len(domain_groups[d]), reverse=True)[:5]

        # Distill each domain group
        model = MODEL_REGISTRY.get(self._MEMORY_MODEL, MODEL_REGISTRY["qwen2.5:7b"])

        for domain, records in list(domain_groups.items())[:5]:   # max 5 domains
            if len(records) < 2:
                continue   # Need at least 2 to distill

            combined = "\n\n---\n\n".join(
                f"[Task {i+1}] Quality:{r.quality_score:.2f}\n{r.content[:400]}"
                for i, r in enumerate(records[:8])
            )

            resp = await self._llm.chat(
                messages=[
                    Message("user",
                        f"Domain: {domain}\n\n"
                        f"Recent experiences ({len(records)} tasks):\n\n{combined}\n\n"
                        "Distill these into reusable knowledge."
                    )
                ],
                model=model,
                system=self._build_system_prompt(context),
                privacy_tier=PrivacyTier.PRIVATE,
                temperature=0.3,
                max_tokens=800,
            )

            data = self._parse_json(resp.content)
            distilled = data.get("distilled_knowledge", "")
            if not distilled:
                continue

            # Write distilled knowledge to semantic memory
            doc_id = await self.remember(
                content=distilled,
                context=context,
                doc_type=DocumentType.SUMMARY,
                quality_score=float(data.get("quality_score", 0.7)),
                tags=["distilled", domain, "consolidated"] + data.get("patterns", [])[:3],
            )
            if doc_id:
                report.distilled_count += 1

        logger.info(
            "Consolidation: %d episodic → %d distilled",
            len(recent), report.distilled_count,
        )

    # ── Temporal Decay (Forgetting) ───────────────────────────────────────────

    async def _apply_decay(self, context: AgentContext, report: MemoryHealthReport) -> None:
        """
        Mark low-quality or expired records for deletion.
        Actual deletion uses a soft-delete approach (quality_score → 0).
        Real pruning is delegated to storage layer (batch job).
        """
        if self.memory_store is None:
            return

        now = time.time()
        # Retrieve all records (use broad query)
        all_cfg = RetrievalConfig(
            mode=RetrievalMode.TEMPORAL,
            top_k=100,
            score_threshold=0.0,
            apply_temporal_decay=False,
        )
        all_recs = await self.memory_store.retrieve(
            query="all", config=all_cfg, privacy_tier=PrivacyTier.PRIVATE
        )

        forgotten = 0
        for rec in all_recs:
            # Compute decay
            days_old = (now - rec.created_at) / 86400
            decayed_score = rec.quality_score * max(0.1, 1.0 - 0.005 * days_old)

            # Flag for deletion: very old + low quality + rarely accessed
            should_forget = (
                days_old > 90 and decayed_score < 0.3 and rec.access_count < 2
            )
            # Or explicitly expired
            if rec.expires_at and now > rec.expires_at:
                should_forget = True

            if should_forget:
                rec.quality_score = 0.0  # soft-delete marker
                forgotten += 1

        report.forgotten_count = forgotten
        logger.info("Decay applied: %d records marked for pruning", forgotten)

    # ── Reflection ────────────────────────────────────────────────────────────

    async def _reflect(self, context: AgentContext, report: MemoryHealthReport) -> None:
        """
        Find meta-patterns across all episodic memory.
        Generate high-level insights about working patterns, recurring issues, etc.
        """
        if self.memory_store is None:
            return

        # Retrieve high-quality episodic records for reflection
        reflect_cfg = RetrievalConfig(
            mode=RetrievalMode.HYBRID,
            top_k=15,
            score_threshold=0.6,
            filters={"memory_type": MemoryType.EPISODIC.value},
        )
        records = await self.memory_store.retrieve(
            query="successful task patterns recurring",
            config=reflect_cfg,
            privacy_tier=PrivacyTier.PRIVATE,
        )
        if not records:
            return

        combined = "\n\n".join(r.content[:300] for r in records[:10])
        model = MODEL_REGISTRY.get(self._MEMORY_MODEL, MODEL_REGISTRY["qwen2.5:7b"])

        resp = await self._llm.chat(
            messages=[
                Message("user",
                    f"Review these {len(records)} task records and identify:\n"
                    "1. What types of tasks succeed most?\n"
                    "2. What patterns lead to failure?\n"
                    "3. Which domains need improvement?\n"
                    "4. Any surprising insights?\n\n"
                    f"Records:\n{combined}"
                )
            ],
            model=model,
            system=(
                "You are analyzing an AI agent's task history to identify patterns. "
                "Be analytical, specific, and actionable. "
                "Return JSON with keys: patterns, failures, improvements, surprises."
            ),
            privacy_tier=PrivacyTier.PRIVATE,
            temperature=0.4,
        )

        data = self._parse_json(resp.content)
        improvements = data.get("improvements", [])
        report.recommendations.extend(improvements[:3])
        report.insights_found = (
            len(data.get("patterns", [])) +
            len(data.get("surprises", []))
        )

        # Store the reflection as a meta-insight
        reflection_text = resp.content[:1000]
        await self.remember(
            content=f"Reflection on {len(records)} tasks:\n{reflection_text}",
            context=context,
            doc_type=DocumentType.SUMMARY,
            quality_score=0.8,
            tags=["reflection", "meta_insight", "pattern"],
        )

    def _parse_json(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {}
