"""
Model Card Generator Skill
Auto-generate and maintain NIST AI RMF + EU AI Act-aligned model cards
for NEXUS agents. Stores cards as YAML in config/model_cards/.

Sections follow the NIST AI RMF Trustworthy AI framework and
Google's Model Card Toolkit format, extended with EU AI Act
risk classification fields.

Operations:
  generate      — create a new model card for an agent
  update        — update specific fields of an existing card
  validate      — check card completeness against required fields
  export_md     — export card as formatted Markdown
  list_cards    — list all agents with cards
  get_card      — retrieve a card as dict
  add_eval      — append evaluation results to a card
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional
import os

import yaml

from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="model_card_generator",
    description=(
        "Generate and maintain NIST AI RMF + EU AI Act aligned model cards "
        "for each NEXUS agent. Tracks intended use, limitations, performance metrics, "
        "ethical considerations, and EU AI Act risk classification."
    ),
    version="1.0.0",
    domains=["governance", "operations"],
    triggers=["model card", "transparency", "ai governance", "nist", "eu ai act",
              "模型卡", "治理文件", "透明度", "EU AI法案"],
    requires=["pyyaml"],
    is_local=True,
)

_CARDS_DIR = Path(
    os.environ.get("NEXUS_MODEL_CARDS_DIR", "config/model_cards")
)

# Required fields for a card to be considered "complete"
_REQUIRED_FIELDS = [
    "model_details.name",
    "model_details.version",
    "model_details.description",
    "intended_use.primary_uses",
    "intended_use.out_of_scope",
    "factors.relevant_factors",
    "metrics.performance_measures",
    "ethical_considerations.sensitive_data",
    "eu_ai_act.risk_category",
    "eu_ai_act.human_oversight_mechanism",
]


def _blank_card(agent_id: str, description: str = "") -> dict:
    """Return a blank card skeleton following NIST AI RMF + EU AI Act."""
    return {
        # ── NIST AI RMF: Govern ──────────────────────────────────────────────
        "model_details": {
            "name":           agent_id,
            "version":        "1.0.0",
            "description":    description,
            "type":           "LLM-orchestrated agent",
            "created_at":     time.strftime("%Y-%m-%d", time.gmtime()),
            "last_updated":   time.strftime("%Y-%m-%d", time.gmtime()),
            "maintainers":    [],
            "license":        "Internal use only",
            "contact":        "",
        },
        # ── NIST AI RMF: Map ─────────────────────────────────────────────────
        "intended_use": {
            "primary_uses":        [],   # e.g. ["code review", "automated testing"]
            "primary_users":       [],   # e.g. ["developers", "QA engineers"]
            "out_of_scope":        [],   # explicitly prohibited uses
            "deployment_context":  "",
        },
        # ── NIST AI RMF: Measure ─────────────────────────────────────────────
        "factors": {
            "relevant_factors":       [],   # demographic / environmental / instrumental
            "evaluation_factors":     [],
        },
        "metrics": {
            "performance_measures":   [],   # e.g. [{"metric": "quality_score", "threshold": 0.75}]
            "decision_thresholds":    [],
            "approaches_to_uncertainty": "",
        },
        "evaluation_data": {
            "datasets":     [],
            "motivation":   "",
            "preprocessing": "",
        },
        "training_data": {
            "description": "NEXUS agents use pre-trained foundation models (no fine-tuning by default)",
            "sources":     [],
        },
        "quantitative_analyses": {
            "unitary_results":    [],   # [{date, metric, value, model}]
            "intersectional_results": [],
        },
        # ── NIST AI RMF: Manage ───────────────────────────────────────────────
        "ethical_considerations": {
            "sensitive_data":      False,
            "pii_handling":        "PII scrubbed before cloud calls via PIIScrubber",
            "bias_mitigations":    [],
            "risks":               [],
            "use_cases_to_avoid":  [],
        },
        "caveats_and_recommendations": {
            "caveats":         [],
            "recommendations": [],
            "known_issues":    [],
        },
        # ── EU AI Act Extension ───────────────────────────────────────────────
        "eu_ai_act": {
            "risk_category":              "minimal",  # prohibited | high | limited | minimal
            "article_references":         [],          # e.g. ["Art. 6", "Art. 14"]
            "human_oversight_mechanism":  "confirmation gate in MasterOrchestrator for destructive tasks",
            "transparency_measures":      ["audit logging", "quality scoring", "AgentCard at /.well-known/agent.json"],
            "technical_documentation":    "config/model_cards/" + agent_id + ".yaml",
            "conformity_assessment":      "not required (minimal risk)",
        },
    }


def _deep_get(d: dict, dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    cur  = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


class Skill(BaseSkill):
    meta = SKILL_META

    def _card_path(self, agent_id: str) -> Path:
        _CARDS_DIR.mkdir(parents=True, exist_ok=True)
        return _CARDS_DIR / f"{agent_id}.yaml"

    def _load_card(self, agent_id: str) -> Optional[dict]:
        fp = self._card_path(agent_id)
        if not fp.exists():
            return None
        return yaml.safe_load(fp.read_text(encoding="utf-8"))

    def _save_card(self, agent_id: str, card: dict) -> None:
        if "model_details" in card:
            card["model_details"]["last_updated"] = time.strftime("%Y-%m-%d", time.gmtime())
        fp = self._card_path(agent_id)
        fp.write_text(yaml.dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8")

    async def run(
        self,
        operation: str,
        agent_id: str = "",
        description: str = "",
        updates: Optional[dict] = None,
        eval_result: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        op = operation.lower()

        # ── generate ──────────────────────────────────────────────────────────
        if op == "generate":
            if not agent_id:
                return {"error": "agent_id required"}
            existing = self._load_card(agent_id)
            if existing:
                return {"error": f"Card already exists for {agent_id}. Use 'update' to modify.", "path": str(self._card_path(agent_id))}
            card = _blank_card(agent_id, description)
            self._save_card(agent_id, card)
            return {
                "generated": True,
                "agent_id":  agent_id,
                "path":      str(self._card_path(agent_id)),
                "completeness": self._check_completeness(card),
            }

        # ── get_card ──────────────────────────────────────────────────────────
        if op == "get_card":
            if not agent_id:
                return {"error": "agent_id required"}
            card = self._load_card(agent_id)
            if card is None:
                return {"error": f"No card found for {agent_id}. Run 'generate' first."}
            return card

        # ── update ────────────────────────────────────────────────────────────
        if op == "update":
            if not agent_id:
                return {"error": "agent_id required"}
            card = self._load_card(agent_id) or _blank_card(agent_id)
            if updates:
                card = self._deep_merge(card, updates)
            self._save_card(agent_id, card)
            return {
                "updated":     True,
                "agent_id":    agent_id,
                "completeness": self._check_completeness(card),
            }

        # ── validate ──────────────────────────────────────────────────────────
        if op == "validate":
            if not agent_id:
                return {"error": "agent_id required"}
            card = self._load_card(agent_id)
            if card is None:
                return {"error": f"No card found for {agent_id}"}
            return self._check_completeness(card)

        # ── export_md ─────────────────────────────────────────────────────────
        if op == "export_md":
            if not agent_id:
                return {"error": "agent_id required"}
            card = self._load_card(agent_id)
            if card is None:
                return {"error": f"No card found for {agent_id}"}
            md = self._to_markdown(card)
            out_path = _CARDS_DIR / f"{agent_id}.md"
            out_path.write_text(md, encoding="utf-8")
            return {"markdown": md, "path": str(out_path)}

        # ── list_cards ────────────────────────────────────────────────────────
        if op == "list_cards":
            _CARDS_DIR.mkdir(parents=True, exist_ok=True)
            cards = []
            for fp in sorted(_CARDS_DIR.glob("*.yaml")):
                raw = yaml.safe_load(fp.read_text(encoding="utf-8"))
                completeness = self._check_completeness(raw)
                cards.append({
                    "agent_id":       fp.stem,
                    "description":    _deep_get(raw, "model_details.description") or "",
                    "risk_category":  _deep_get(raw, "eu_ai_act.risk_category") or "unknown",
                    "last_updated":   _deep_get(raw, "model_details.last_updated") or "",
                    "completeness_pct": completeness["completeness_pct"],
                })
            return {"cards": cards, "total": len(cards)}

        # ── add_eval ──────────────────────────────────────────────────────────
        if op == "add_eval":
            if not agent_id or not eval_result:
                return {"error": "agent_id and eval_result required"}
            card = self._load_card(agent_id) or _blank_card(agent_id)
            entry = {
                "date":   time.strftime("%Y-%m-%d", time.gmtime()),
                **eval_result,
            }
            card.setdefault("quantitative_analyses", {}).setdefault("unitary_results", []).append(entry)
            self._save_card(agent_id, card)
            return {"added": True, "agent_id": agent_id, "eval": entry}

        return {"error": f"Unknown operation: {op}"}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _check_completeness(self, card: dict) -> dict:
        missing = []
        for field in _REQUIRED_FIELDS:
            val = _deep_get(card, field)
            if val is None or val == [] or val == "":
                missing.append(field)
        pct = round(100 * (1 - len(missing) / len(_REQUIRED_FIELDS)), 1)
        return {
            "completeness_pct": pct,
            "missing_fields":   missing,
            "is_complete":      pct == 100.0,
        }

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _to_markdown(self, card: dict) -> str:
        md = card.get("model_details", {})
        lines = [
            f"# Model Card: {md.get('name', 'Unknown')}",
            f"*Version {md.get('version', '1.0.0')} — Last updated: {md.get('last_updated', '')}*",
            "",
            f"## Description",
            md.get("description", ""),
            "",
            "## Intended Use",
        ]
        iu = card.get("intended_use", {})
        for use in iu.get("primary_uses", []):
            lines.append(f"- {use}")
        lines += [
            "",
            "### Out of Scope",
        ]
        for oos in iu.get("out_of_scope", []):
            lines.append(f"- ❌ {oos}")

        lines += [
            "",
            "## Performance Metrics",
        ]
        for m in card.get("metrics", {}).get("performance_measures", []):
            if isinstance(m, dict):
                lines.append(f"- **{m.get('metric', '')}**: threshold = {m.get('threshold', '')}")
            else:
                lines.append(f"- {m}")

        lines += [
            "",
            "## Ethical Considerations",
        ]
        ec = card.get("ethical_considerations", {})
        lines.append(f"- Sensitive data: {ec.get('sensitive_data', False)}")
        lines.append(f"- PII handling: {ec.get('pii_handling', '')}")
        for risk in ec.get("risks", []):
            lines.append(f"- ⚠️ {risk}")

        lines += [
            "",
            "## EU AI Act",
        ]
        eu = card.get("eu_ai_act", {})
        lines.append(f"- **Risk Category**: `{eu.get('risk_category', 'unknown')}`")
        lines.append(f"- **Human Oversight**: {eu.get('human_oversight_mechanism', '')}")
        for m in eu.get("transparency_measures", []):
            lines.append(f"- Transparency: {m}")

        return "\n".join(lines)
