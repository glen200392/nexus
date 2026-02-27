"""
NEXUS Cost Optimizer Agent
Monitors LLM spend, enforces budgets, and recommends routing adjustments.

Operations (context.metadata["operation"]):
  check_budget   — compare actual spend vs configured budgets, flag overruns
  report         — generate detailed cost breakdown by agent/model/day
  optimize       — analyze routing decisions and suggest cheaper alternatives
  set_budget     — update budget limits (stored in config/governance/budgets.yaml)
  alert_check    — run a fast check and return alerts only (for scheduled triggers)

Budget config: config/governance/budgets.yaml
  daily_limit_usd:   50.0
  monthly_limit_usd: 1000.0
  per_agent_daily:   {code_agent: 10.0, browser_agent: 5.0}
  auto_downgrade:    true   # automatically adjust routing when over budget

Routing downgrade rules (applied when daily limit is exceeded):
  claude-opus-4  → claude-sonnet-4
  claude-sonnet-4 → claude-haiku-4
  gpt-4o         → gpt-4o-mini
  gemini-1.5-pro → gemini-1.5-flash
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.cost_optimizer")

_BUDGET_FILE = Path(os.environ.get("NEXUS_BUDGET_FILE", "config/governance/budgets.yaml"))

# Model downgrade chains
_DOWNGRADE_CHAIN = {
    "claude-opus-4-6":        "claude-sonnet-4-6",
    "claude-sonnet-4-6":      "claude-haiku-4-5-20251001",
    "gpt-4o":                 "gpt-4o-mini",
    "gpt-4o-mini":            "gpt-4o-mini",          # already cheapest
    "gemini-1.5-pro":         "gemini-1.5-flash",
    "gemini-1.5-flash":       "gemini-1.5-flash",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}


@dataclass
class BudgetStatus:
    daily_spend_usd:   float = 0.0
    monthly_spend_usd: float = 0.0
    daily_limit_usd:   float = 50.0
    monthly_limit_usd: float = 1000.0
    daily_pct:         float = 0.0
    monthly_pct:       float = 0.0
    alerts:            list[str] = field(default_factory=list)
    top_agents:        list[dict] = field(default_factory=list)
    top_models:        list[dict] = field(default_factory=list)
    auto_downgrade_active: bool = False


class CostOptimizerAgent(BaseAgent):
    agent_id   = "cost_optimizer_agent"
    agent_name = "Cost Optimizer Agent"
    description = (
        "Monitors LLM API spend, enforces budget limits, recommends model "
        "routing downgrades, and generates cost breakdown reports. "
        "Integrates with GovernanceManager audit log."
    )
    domain             = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.LOW
    default_privacy    = PrivacyTier.INTERNAL

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    # ── Budget config I/O ──────────────────────────────────────────────────────

    def _load_budgets(self) -> dict:
        _BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _BUDGET_FILE.exists():
            return yaml.safe_load(_BUDGET_FILE.read_text(encoding="utf-8")) or {}
        default = {
            "daily_limit_usd":   50.0,
            "monthly_limit_usd": 1000.0,
            "alert_pct":         80,    # alert at 80% of limit
            "auto_downgrade":    True,
            "per_agent_daily":   {},
        }
        self._save_budgets(default)
        return default

    def _save_budgets(self, budgets: dict) -> None:
        _BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BUDGET_FILE.write_text(yaml.dump(budgets, allow_unicode=True), encoding="utf-8")

    # ── Cost data retrieval ────────────────────────────────────────────────────

    def _get_cost_data(self) -> dict:
        """Pull cost data from GovernanceManager audit log if available."""
        try:
            from nexus.core.governance import GovernanceManager
            gov = GovernanceManager()
            return gov.audit.get_cost_summary()
        except Exception as exc:
            logger.warning("Could not access audit log: %s", exc)
            return {}

    def _compute_budget_status(self, cost_data: dict, budgets: dict) -> BudgetStatus:
        status = BudgetStatus(
            daily_limit_usd=budgets.get("daily_limit_usd", 50.0),
            monthly_limit_usd=budgets.get("monthly_limit_usd", 1000.0),
        )

        today     = time.strftime("%Y-%m-%d", time.gmtime())
        this_month = time.strftime("%Y-%m", time.gmtime())

        # Sum from audit data
        by_date  = cost_data.get("by_date", {})
        by_agent = cost_data.get("by_agent", {})
        by_model = cost_data.get("by_model", {})

        status.daily_spend_usd   = by_date.get(today, 0.0)
        status.monthly_spend_usd = sum(
            v for k, v in by_date.items()
            if k.startswith(this_month)
        )
        status.daily_pct   = round(100 * status.daily_spend_usd   / max(status.daily_limit_usd, 0.01), 1)
        status.monthly_pct = round(100 * status.monthly_spend_usd / max(status.monthly_limit_usd, 0.01), 1)

        # Alerts
        alert_threshold = budgets.get("alert_pct", 80)
        if status.daily_pct >= 100:
            status.alerts.append(f"🚨 DAILY budget EXCEEDED: ${status.daily_spend_usd:.2f} / ${status.daily_limit_usd:.2f}")
            if budgets.get("auto_downgrade"):
                status.auto_downgrade_active = True
        elif status.daily_pct >= alert_threshold:
            status.alerts.append(f"⚠️  Daily budget at {status.daily_pct}%: ${status.daily_spend_usd:.2f} / ${status.daily_limit_usd:.2f}")

        if status.monthly_pct >= 100:
            status.alerts.append(f"🚨 MONTHLY budget EXCEEDED: ${status.monthly_spend_usd:.2f} / ${status.monthly_limit_usd:.2f}")
        elif status.monthly_pct >= alert_threshold:
            status.alerts.append(f"⚠️  Monthly budget at {status.monthly_pct}%: ${status.monthly_spend_usd:.2f} / ${status.monthly_limit_usd:.2f}")

        # Per-agent daily limits
        per_agent_limits = budgets.get("per_agent_daily", {})
        for agent_id, limit in per_agent_limits.items():
            agent_spend = by_agent.get(agent_id, 0.0)
            if agent_spend > limit:
                status.alerts.append(
                    f"⚠️  {agent_id} over daily budget: ${agent_spend:.3f} / ${limit:.2f}"
                )

        # Top consumers
        status.top_agents = sorted(
            [{"agent": k, "cost_usd": round(v, 4)} for k, v in by_agent.items()],
            key=lambda x: x["cost_usd"], reverse=True,
        )[:5]
        status.top_models = sorted(
            [{"model": k, "cost_usd": round(v, 4)} for k, v in by_model.items()],
            key=lambda x: x["cost_usd"], reverse=True,
        )[:5]

        return status

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a cost optimization agent. Monitor LLM spending, "
            "enforce budget constraints, and recommend model downgrades when needed."
        )

    # ── Agent Execution ────────────────────────────────────────────────────────

    async def execute(self, context: AgentContext) -> AgentResult:
        operation = context.metadata.get("operation", "check_budget")
        budgets   = self._load_budgets()
        cost_data = self._get_cost_data()

        if operation == "set_budget":
            return self._set_budget(context, budgets)
        if operation == "alert_check":
            return self._alert_check(cost_data, budgets)
        if operation == "report":
            return await self._report(cost_data, budgets, context)
        if operation == "optimize":
            return await self._optimize(cost_data, budgets, context)

        # Default: check_budget
        return self._check_budget(cost_data, budgets, context)

    def _check_budget(self, cost_data: dict, budgets: dict, context: AgentContext) -> AgentResult:
        status = self._compute_budget_status(cost_data, budgets)
        output = {
            "daily_spend_usd":       round(status.daily_spend_usd, 4),
            "daily_limit_usd":       status.daily_limit_usd,
            "daily_pct":             status.daily_pct,
            "monthly_spend_usd":     round(status.monthly_spend_usd, 4),
            "monthly_limit_usd":     status.monthly_limit_usd,
            "monthly_pct":           status.monthly_pct,
            "alerts":                status.alerts,
            "auto_downgrade_active": status.auto_downgrade_active,
            "top_agents":            status.top_agents,
            "top_models":            status.top_models,
            "downgrade_chain":       _DOWNGRADE_CHAIN if status.auto_downgrade_active else {},
        }
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True, output=output,
            quality_score=1.0,
        )

    def _alert_check(self, cost_data: dict, budgets: dict) -> AgentResult:
        status = self._compute_budget_status(cost_data, budgets)
        return AgentResult(
            agent_id=self.agent_id, task_id="",
            success=True,
            output={"alerts": status.alerts, "has_alerts": bool(status.alerts)},
        )

    def _set_budget(self, context: AgentContext, budgets: dict) -> AgentResult:
        updates = context.metadata.get("budget_updates", {})
        budgets.update(updates)
        self._save_budgets(budgets)
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={"saved": True, "budgets": budgets},
        )

    async def _report(self, cost_data: dict, budgets: dict, context: AgentContext) -> AgentResult:
        """Generate a narrative cost report using LLM."""
        status    = self._compute_budget_status(cost_data, budgets)
        decision  = self.route_llm(context)
        stats_txt = json.dumps({
            "daily":   {"spent": status.daily_spend_usd, "limit": status.daily_limit_usd, "pct": status.daily_pct},
            "monthly": {"spent": status.monthly_spend_usd, "limit": status.monthly_limit_usd, "pct": status.monthly_pct},
            "top_agents": status.top_agents,
            "top_models": status.top_models,
            "alerts": status.alerts,
        }, indent=2)

        resp = await self._llm.chat(
            messages=[Message("user",
                f"Generate a concise cost management report for the NEXUS AI platform.\n\n"
                f"Cost data:\n{stats_txt}\n\n"
                "Include: executive summary, cost breakdown, optimization recommendations, "
                "and projected monthly total if current trend continues."
            )],
            model=decision.primary,
            system=(
                "You are a FinOps analyst for an AI platform. "
                "Be specific with numbers. Format as Markdown. "
                "Return JSON: {\"summary\": \"...\", \"recommendations\": [], \"projected_monthly\": 0.0}"
            ),
            privacy_tier=context.privacy_tier,
        )
        insights = {}
        try:
            import re
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                insights = json.loads(m.group())
        except Exception:
            insights = {"summary": resp.content[:500]}

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={**status.__dict__, "insights": insights},
            quality_score=0.85,
        )

    async def _optimize(self, cost_data: dict, budgets: dict, context: AgentContext) -> AgentResult:
        """Analyze routing and suggest cheaper alternatives."""
        status   = self._compute_budget_status(cost_data, budgets)
        decision = self.route_llm(context)

        suggestions = []

        # Rule-based optimizations
        for model_entry in status.top_models:
            model = model_entry["model"]
            cost  = model_entry["cost_usd"]
            if model in _DOWNGRADE_CHAIN and _DOWNGRADE_CHAIN[model] != model:
                downgraded = _DOWNGRADE_CHAIN[model]
                # Estimate savings: downgraded models are typically 5–20x cheaper
                est_savings_pct = 70 if "opus" in model or "gpt-4o" in model and "mini" not in model else 40
                suggestions.append({
                    "action":        f"Downgrade {model} → {downgraded}",
                    "estimated_savings_pct": est_savings_pct,
                    "affected_cost": cost,
                    "recommendation": f"Route non-critical tasks to {downgraded}",
                })

        # LLM-generated recommendations
        resp = await self._llm.chat(
            messages=[Message("user",
                f"Top cost drivers: {json.dumps(status.top_agents)}\n"
                f"Top models: {json.dumps(status.top_models)}\n"
                f"Daily spend: ${status.daily_spend_usd:.2f} ({status.daily_pct}% of limit)\n\n"
                "Suggest 3 specific, actionable cost optimizations for this AI agent platform."
            )],
            model=decision.primary,
            system="You are a FinOps AI optimization expert. Be specific and quantitative. "
                   "Return JSON: {\"optimizations\": [{\"action\": str, \"impact\": str, \"effort\": \"low|medium|high\"}]}",
            privacy_tier=context.privacy_tier,
        )
        try:
            import re
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                llm_suggestions = json.loads(m.group()).get("optimizations", [])
                suggestions.extend(llm_suggestions)
        except Exception:
            pass

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={
                "suggestions":           suggestions,
                "auto_downgrade_chain":  _DOWNGRADE_CHAIN,
                "current_daily_spend":   round(status.daily_spend_usd, 4),
                "downgrade_active":      status.auto_downgrade_active,
            },
            quality_score=0.8,
        )
