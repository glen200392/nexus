"""
NEXUS Bias Auditor Agent
Systematically audits AI system outputs for demographic bias,
representational harm, and allocational disparities.

Operations (context.metadata["operation"]):
  audit_output   — analyze a single text output for bias (default)
  batch_audit    — audit recent task outputs from memory/audit log
  generate_report — full bias audit report with trend analysis
  flag_for_review — mark a specific output for human review

Uses: bias_auditor.md system prompt + Fairlearn concepts (rule-based proxy)
Always uses INTERNAL tier — bias analysis itself is not private data.

Integration:
  GovernanceManager can call BiasAuditorAgent.audit_output()
  automatically for HIGH_RISK tasks (EU AI Act compliance).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.bias_auditor")

_REPORT_DIR = Path("data/bias_reports")

# ── Lightweight rule-based bias signals (supplement LLM analysis) ─────────────

_STEREOTYPING_PATTERNS = [
    (r"\b(?:all|every|most)\s+(?:women|men|blacks|whites|asians|elderly|young)\s+(?:are|tend|prefer)",
     "demographic_generalization"),
    (r"\b(?:naturally|inherently|biologically)\s+(?:better|worse|less|more)\s+at",
     "essentialist_claim"),
    (r"\b(?:females?|males?)\s+(?:are\s+)?(?:too|not\s+smart|emotional|weak|strong)",
     "gender_stereotype"),
    (r"\b(?:typical|classic|standard)\s+(?:asian|black|white|hispanic|female|male)\b",
     "demographic_profiling"),
]

_EXCLUSIONARY_LANGUAGE = [
    (r"\b(?:mankind|manpower|chairman|stewardess|fireman|policeman)\b", "gendered_language"),
    (r"\b(?:crazy|insane|psycho|retarded|lame)\b", "ableist_language"),
    (r"\bnormal\s+(?:people|person|human)\b", "othering_language"),
]


@dataclass
class BiasSignal:
    dimension:   str    # demographic_parity | representational_harm | allocational_harm
    pattern:     str
    evidence:    str
    severity:    str    # informational | minor | moderate | severe
    match_text:  str


@dataclass
class BiasAuditResult:
    overall_risk:          str = "low"    # low | medium | high | critical
    bias_signals:          list[BiasSignal] = field(default_factory=list)
    llm_analysis:          dict = field(default_factory=dict)
    affected_groups:       list[str] = field(default_factory=list)
    recommendations:       list[str] = field(default_factory=list)
    requires_human_review: bool = False
    audit_timestamp:       str = ""
    input_snippet:         str = ""


class BiasAuditorAgent(BaseAgent):
    agent_id   = "bias_auditor_agent"
    agent_name = "Bias Auditor Agent"
    description = (
        "Audits AI outputs for demographic bias, representational harm, "
        "and allocational disparities. Combines rule-based pattern matching "
        "with LLM-based fairness analysis. Flags outputs for human review "
        "when bias risk is high or critical."
    )
    domain             = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.MEDIUM
    default_privacy    = PrivacyTier.INTERNAL

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Load bias_auditor.md system prompt."""
        prompt_file = Path("config/prompts/system/bias_auditor.md")
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        # Inline fallback
        return (
            "You are an AI Fairness and Bias Auditor. Analyze the given text for demographic bias, "
            "representational harm, and allocational disparities. "
            "Return structured JSON with keys: overall_bias_risk, dimensions, affected_groups, "
            "severity, recommendations, requires_human_review."
        )

    def _rule_based_scan(self, text: str) -> list[BiasSignal]:
        """Fast pattern-matching scan before LLM call."""
        signals: list[BiasSignal] = []
        for pattern, label in _STEREOTYPING_PATTERNS:
            for m in re.finditer(pattern, text, re.I):
                signals.append(BiasSignal(
                    dimension="representational_harm",
                    pattern=label,
                    evidence=f"Pattern '{label}' detected",
                    severity="moderate",
                    match_text=m.group()[:100],
                ))
        for pattern, label in _EXCLUSIONARY_LANGUAGE:
            for m in re.finditer(pattern, text, re.I):
                signals.append(BiasSignal(
                    dimension="representational_harm",
                    pattern=label,
                    evidence=f"Exclusionary language: '{label}'",
                    severity="minor",
                    match_text=m.group()[:100],
                ))
        return signals

    def _overall_risk(self, signals: list[BiasSignal], llm_result: dict) -> str:
        """Combine rule-based and LLM signals into overall risk level."""
        llm_risk = llm_result.get("overall_bias_risk", "low")
        if llm_risk == "critical":
            return "critical"
        severe_count   = sum(1 for s in signals if s.severity == "severe")
        moderate_count = sum(1 for s in signals if s.severity == "moderate")
        if severe_count > 0 or llm_risk == "high":
            return "high"
        if moderate_count >= 2 or llm_risk == "medium":
            return "medium"
        return "low"

    async def execute(self, context: AgentContext) -> AgentResult:
        operation = context.metadata.get("operation", "audit_output")

        if operation == "batch_audit":
            return await self._batch_audit(context)
        if operation == "generate_report":
            return await self._generate_report(context)
        if operation == "flag_for_review":
            return self._flag_for_review(context)

        # Default: audit_output
        return await self._audit_output(context)

    async def _audit_output(self, context: AgentContext) -> AgentResult:
        """Audit a single text output."""
        text_to_audit = context.metadata.get("output_text", context.user_message)
        if not text_to_audit:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error="No text to audit. Set output_text in metadata.",
            )

        # Step 1: Rule-based scan (fast, no LLM cost)
        signals = self._rule_based_scan(text_to_audit)

        # Step 2: LLM analysis
        decision = self.route_llm(context)
        resp = await self._llm.chat(
            messages=[Message("user",
                f"Analyze this AI output for bias:\n\n```\n{text_to_audit[:3000]}\n```\n\n"
                f"Rule-based pre-scan found {len(signals)} signal(s): "
                f"{[s.pattern for s in signals]}\n\n"
                "Provide structured bias analysis."
            )],
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.2,
            max_tokens=1000,
        )

        llm_result = self._parse_json(resp.content)
        overall    = self._overall_risk(signals, llm_result)

        result = BiasAuditResult(
            overall_risk=overall,
            bias_signals=signals,
            llm_analysis=llm_result,
            affected_groups=llm_result.get("affected_groups", []),
            recommendations=llm_result.get("recommendations", []),
            requires_human_review=(overall in ("high", "critical")
                                   or llm_result.get("requires_human_review", False)),
            audit_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            input_snippet=text_to_audit[:200],
        )

        # Persist high-risk findings
        if result.requires_human_review:
            self._save_report(result, context.task_id)
            logger.warning(
                "Bias audit flagged for human review (risk=%s, task=%s)",
                overall, context.task_id,
            )

        await self.remember(
            content=f"Bias audit [{overall}]: {context.task_id}\n{json.dumps(llm_result)[:300]}",
            context=context,
            doc_type=DocumentType.EPISODIC,
            tags=["bias_audit", overall, "governance"],
        )

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "overall_risk":          result.overall_risk,
                "requires_human_review": result.requires_human_review,
                "bias_signals":          [s.__dict__ for s in result.bias_signals],
                "llm_analysis":          result.llm_analysis,
                "affected_groups":       result.affected_groups,
                "recommendations":       result.recommendations,
                "audit_timestamp":       result.audit_timestamp,
            },
            quality_score=0.85,
        )

    async def _batch_audit(self, context: AgentContext) -> AgentResult:
        """Audit multiple outputs from recent task history."""
        outputs = context.metadata.get("outputs", [])
        if not outputs:
            # Try to pull from memory
            if self.memory_store:
                from nexus.knowledge.rag.schema import RetrievalConfig, RetrievalMode, MemoryType
                cfg = RetrievalConfig(mode=RetrievalMode.TEMPORAL, top_k=10, score_threshold=0.0)
                records = await self.memory_store.retrieve(
                    "recent task output", cfg, PrivacyTier.INTERNAL
                )
                outputs = [r.content[:1000] for r in records]

        if not outputs:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error="No outputs to audit. Pass outputs list in metadata.",
            )

        results = []
        high_risk_count = 0
        for i, text in enumerate(outputs[:10]):  # cap at 10
            signals = self._rule_based_scan(text)
            risk    = "high" if len(signals) > 2 else "medium" if signals else "low"
            if risk in ("high", "critical"):
                high_risk_count += 1
            results.append({
                "index":         i,
                "snippet":       text[:200],
                "signal_count":  len(signals),
                "risk":          risk,
                "signals":       [s.pattern for s in signals],
            })

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "audited_count":    len(results),
                "high_risk_count":  high_risk_count,
                "results":          results,
                "overall_health":   "good" if high_risk_count == 0 else "attention_required",
            },
            quality_score=0.8,
        )

    async def _generate_report(self, context: AgentContext) -> AgentResult:
        """Generate a full bias audit report with trend analysis."""
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        reports = list(_REPORT_DIR.glob("*.json"))

        if not reports:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={"message": "No bias reports saved yet. Run audit_output on high-risk tasks first."},
            )

        all_reports = []
        for fp in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            try:
                all_reports.append(json.loads(fp.read_text(encoding="utf-8")))
            except Exception:
                pass

        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for r in all_reports:
            risk_counts[r.get("overall_risk", "low")] = risk_counts.get(r.get("overall_risk", "low"), 0) + 1

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "total_audits":    len(all_reports),
                "risk_distribution": risk_counts,
                "requires_attention": risk_counts.get("high", 0) + risk_counts.get("critical", 0),
                "recent_reports":  all_reports[:5],
            },
            quality_score=0.85,
        )

    def _flag_for_review(self, context: AgentContext) -> AgentResult:
        """Manually flag an output for human review."""
        output_text = context.metadata.get("output_text", "")
        reason      = context.metadata.get("reason", "Manual flag")
        flag = {
            "task_id":   context.task_id,
            "reason":    reason,
            "flagged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "snippet":   output_text[:300],
        }
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fp = _REPORT_DIR / f"flag_{context.task_id[:8]}_{int(time.time())}.json"
        fp.write_text(json.dumps(flag, indent=2, ensure_ascii=False), encoding="utf-8")
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={"flagged": True, "file": str(fp)},
        )

    def _save_report(self, result: BiasAuditResult, task_id: str) -> None:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        fp = _REPORT_DIR / f"audit_{task_id[:8]}_{int(time.time())}.json"
        fp.write_text(json.dumps({
            "overall_risk":          result.overall_risk,
            "affected_groups":       result.affected_groups,
            "requires_human_review": result.requires_human_review,
            "recommendations":       result.recommendations,
            "audit_timestamp":       result.audit_timestamp,
            "input_snippet":         result.input_snippet,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    def _parse_json(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"overall_bias_risk": "low", "recommendations": [raw[:300]]}
