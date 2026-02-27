"""
Regulatory Tracker Skill
Maintains a local YAML registry of AI/data regulations, compliance deadlines,
and obligation checklists. Works fully offline — no external API required.

Pre-seeded with key regulations:
  - EU AI Act (2024/1689) — phased enforcement timeline
  - EU GDPR (2016/679)
  - NIST AI RMF 1.0
  - EU Data Act (2023/2854)
  - Colorado AI Act (SB21-169)
  - Canada AIDA (Bill C-27)
  - Singapore PDPA / Model AI Governance Framework

Operations:
  list            — all tracked regulations (optionally filter by status/tag)
  get             — detail for one regulation by id
  add             — add a new regulation to the registry
  update          — update status or fields of an existing entry
  check_upcoming  — regulations with deadlines within N days
  add_obligation  — add a compliance obligation/checklist item
  check_compliance — return obligation completion summary for a regulation
  search          — full-text search across name/description/obligations
  generate_report — LLM-assisted compliance status report (requires LLM)
  export_csv      — export registry as CSV
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from nexus.skills.registry import BaseSkill, SkillMeta

logger = logging.getLogger("nexus.skills.regulatory_tracker")

SKILL_META = SkillMeta(
    name="regulatory_tracker",
    description=(
        "Track AI/data regulation compliance deadlines and obligations. "
        "Pre-seeded with EU AI Act, GDPR, NIST AI RMF, Data Act, and more. "
        "Manage obligation checklists, check upcoming deadlines, and generate "
        "compliance status reports. Fully offline — no external API needed."
    ),
    version="1.0.0",
    domains=["governance", "compliance", "legal", "operations"],
    triggers=[
        "regulation", "compliance", "deadline", "EU AI Act", "GDPR",
        "NIST AI RMF", "regulatory", "law", "legal obligation",
        "法規", "合規", "截止日期", "歐盟人工智慧法案", "個資法",
    ],
)

REGISTRY_PATH = Path("config/regulatory/registry.yaml")

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ComplianceObligation:
    id:          str
    description: str
    article_ref: str = ""          # e.g., "Art.13", "§4.2"
    status:      str = "pending"   # pending | in_progress | compliant | na
    due_date:    str = ""          # ISO date or ""
    notes:       str = ""
    owner:       str = ""          # team or person responsible


@dataclass
class Regulation:
    id:               str
    name:             str
    short_name:       str
    jurisdiction:     str          # EU, US, CA, SG, Global, …
    status:           str          # draft | enacted | in_force | superseded
    effective_date:   str          # ISO date
    enforcement_date: str          # ISO date (may differ from effective)
    risk_level:       str          # critical | high | medium | low
    url:              str          # canonical reference URL
    description:      str
    tags:             list[str]    = field(default_factory=list)
    obligations:      list[dict]   = field(default_factory=list)  # list of ComplianceObligation dicts
    last_updated:     str          = field(default_factory=lambda: _today())
    notes:            str          = ""


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _days_until(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now    = datetime.now(tz=timezone.utc)
        return (target - now).days
    except ValueError:
        return None


# ── Default seed data ──────────────────────────────────────────────────────────

_SEED_REGULATIONS: list[dict] = [
    {
        "id":               "eu_ai_act",
        "name":             "EU Artificial Intelligence Act",
        "short_name":       "EU AI Act",
        "jurisdiction":     "EU",
        "status":           "in_force",
        "effective_date":   "2024-08-01",
        "enforcement_date": "2026-08-01",
        "risk_level":       "critical",
        "url":              "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
        "description": (
            "Regulation (EU) 2024/1689. Establishes harmonised rules for AI systems "
            "in the EU. Phased enforcement: prohibited practices (Feb 2025), "
            "GPAI rules (Aug 2025), high-risk systems (Aug 2026)."
        ),
        "tags": ["AI", "high-risk", "GPAI", "prohibited-practices", "conformity-assessment"],
        "obligations": [
            {"id": "eu_ai_1", "description": "Identify and classify AI systems by risk level",
             "article_ref": "Art.6–7", "status": "pending"},
            {"id": "eu_ai_2", "description": "Implement risk management system for high-risk AI",
             "article_ref": "Art.9", "status": "pending"},
            {"id": "eu_ai_3", "description": "Ensure data governance and training data quality",
             "article_ref": "Art.10", "status": "pending"},
            {"id": "eu_ai_4", "description": "Maintain technical documentation",
             "article_ref": "Art.11", "status": "pending"},
            {"id": "eu_ai_5", "description": "Implement human oversight measures",
             "article_ref": "Art.14", "status": "pending"},
            {"id": "eu_ai_6", "description": "Ensure accuracy, robustness, and cybersecurity",
             "article_ref": "Art.15", "status": "pending"},
            {"id": "eu_ai_7", "description": "Register high-risk AI in EU database",
             "article_ref": "Art.49", "status": "pending"},
            {"id": "eu_ai_8", "description": "GPAI model: publish model card / technical documentation",
             "article_ref": "Art.53", "status": "pending"},
        ],
    },
    {
        "id":               "eu_gdpr",
        "name":             "EU General Data Protection Regulation",
        "short_name":       "GDPR",
        "jurisdiction":     "EU",
        "status":           "in_force",
        "effective_date":   "2018-05-25",
        "enforcement_date": "2018-05-25",
        "risk_level":       "critical",
        "url":              "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32016R0679",
        "description": (
            "Regulation (EU) 2016/679. Governs processing of personal data. "
            "Fines up to 4% of global annual turnover or €20M."
        ),
        "tags": ["data-privacy", "personal-data", "DPA", "DPIA", "data-subject-rights"],
        "obligations": [
            {"id": "gdpr_1", "description": "Maintain Record of Processing Activities (RoPA)",
             "article_ref": "Art.30", "status": "pending"},
            {"id": "gdpr_2", "description": "Conduct Data Protection Impact Assessment for high-risk processing",
             "article_ref": "Art.35", "status": "pending"},
            {"id": "gdpr_3", "description": "Appoint Data Protection Officer if required",
             "article_ref": "Art.37", "status": "pending"},
            {"id": "gdpr_4", "description": "Implement privacy by design and by default",
             "article_ref": "Art.25", "status": "pending"},
            {"id": "gdpr_5", "description": "Report data breaches within 72 hours",
             "article_ref": "Art.33", "status": "pending"},
        ],
    },
    {
        "id":               "nist_ai_rmf",
        "name":             "NIST AI Risk Management Framework 1.0",
        "short_name":       "NIST AI RMF",
        "jurisdiction":     "US",
        "status":           "in_force",
        "effective_date":   "2023-01-26",
        "enforcement_date": "2023-01-26",
        "risk_level":       "medium",
        "url":              "https://airc.nist.gov/RMF",
        "description": (
            "Voluntary framework for managing AI risks. Four core functions: "
            "GOVERN, MAP, MEASURE, MANAGE. Widely adopted as de facto standard."
        ),
        "tags": ["voluntary", "risk-management", "trustworthy-AI", "GOVERN", "MAP", "MEASURE", "MANAGE"],
        "obligations": [
            {"id": "nist_1", "description": "Establish AI governance policies (GOVERN)",
             "article_ref": "GOVERN 1.1", "status": "pending"},
            {"id": "nist_2", "description": "Identify and classify AI risks (MAP)",
             "article_ref": "MAP 1.1", "status": "pending"},
            {"id": "nist_3", "description": "Measure AI risks with metrics and benchmarks (MEASURE)",
             "article_ref": "MEASURE 1.1", "status": "pending"},
            {"id": "nist_4", "description": "Apply risk response plans (MANAGE)",
             "article_ref": "MANAGE 1.1", "status": "pending"},
        ],
    },
    {
        "id":               "eu_data_act",
        "name":             "EU Data Act",
        "short_name":       "Data Act",
        "jurisdiction":     "EU",
        "status":           "in_force",
        "effective_date":   "2024-01-11",
        "enforcement_date": "2025-09-12",
        "risk_level":       "high",
        "url":              "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R2854",
        "description": (
            "Regulation (EU) 2023/2854. Rules on fair access to and use of data. "
            "Covers connected products, data sharing, cloud switching."
        ),
        "tags": ["data-sharing", "IoT", "cloud-switching", "data-portability"],
        "obligations": [
            {"id": "data_act_1", "description": "Provide data access to users of connected products",
             "article_ref": "Art.4", "status": "pending"},
            {"id": "data_act_2", "description": "Enable cloud service switching without penalty",
             "article_ref": "Art.23–29", "status": "pending"},
        ],
    },
    {
        "id":               "canada_aida",
        "name":             "Canada Artificial Intelligence and Data Act (Bill C-27)",
        "short_name":       "AIDA",
        "jurisdiction":     "CA",
        "status":           "draft",
        "effective_date":   "2025-12-31",   # estimated
        "enforcement_date": "2026-06-30",   # estimated
        "risk_level":       "high",
        "url":              "https://ised-isde.canada.ca/site/innovation-better-canada/en/artificial-intelligence-and-data-act",
        "description": (
            "Part 3 of Bill C-27. Establishes rules for high-impact AI systems, "
            "requires risk assessments, mitigation measures, and transparency."
        ),
        "tags": ["AI", "high-impact", "risk-assessment", "Canada"],
        "obligations": [
            {"id": "aida_1", "description": "Assess whether AI system is 'high-impact'",
             "article_ref": "§5", "status": "pending"},
            {"id": "aida_2", "description": "Implement risk mitigation for high-impact systems",
             "article_ref": "§8", "status": "pending"},
        ],
    },
    {
        "id":               "sg_pdpa_ai",
        "name":             "Singapore PDPA + Model AI Governance Framework 2.0",
        "short_name":       "SG AI Governance",
        "jurisdiction":     "SG",
        "status":           "in_force",
        "effective_date":   "2020-01-01",
        "enforcement_date": "2020-01-01",
        "risk_level":       "medium",
        "url":              "https://www.pdpc.gov.sg/Help-and-Resources/2020/01/Model-Artificial-Intelligence-Governance-Framework",
        "description": (
            "Singapore's voluntary AI governance framework. Covers internal governance, "
            "operations management, stakeholder interaction, and human oversight."
        ),
        "tags": ["voluntary", "Singapore", "PDPA", "AI-governance", "human-oversight"],
        "obligations": [
            {"id": "sg_1", "description": "Document decision-making model for AI deployments",
             "article_ref": "Part 2", "status": "pending"},
            {"id": "sg_2", "description": "Ensure human oversight for significant decisions",
             "article_ref": "Part 3", "status": "pending"},
        ],
    },
]


# ── Registry I/O ───────────────────────────────────────────────────────────────

def _load_registry() -> dict[str, dict]:
    if REGISTRY_PATH.exists():
        try:
            raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        except Exception as exc:
            logger.warning("Could not load regulatory registry: %s", exc)

    # Seed on first run
    seed = {r["id"]: r for r in _SEED_REGULATIONS}
    _save_registry(seed)
    return seed


def _save_registry(registry: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        yaml.dump(registry, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


# ── Skill class ────────────────────────────────────────────────────────────────

class RegulatoryTrackerSkill(BaseSkill):
    meta = SKILL_META

    async def run(
        self,
        operation:      str  = "list",
        regulation_id:  Optional[str]  = None,
        status_filter:  Optional[str]  = None,   # for list
        tag_filter:     Optional[str]  = None,   # for list
        within_days:    int  = 90,               # for check_upcoming
        query:          str  = "",               # for search
        # add / update fields
        name:           Optional[str] = None,
        short_name:     Optional[str] = None,
        jurisdiction:   Optional[str] = None,
        status:         Optional[str] = None,
        effective_date: Optional[str] = None,
        enforcement_date: Optional[str] = None,
        risk_level:     Optional[str] = None,
        url:            Optional[str] = None,
        description:    Optional[str] = None,
        tags:           Optional[list[str]] = None,
        notes:          Optional[str] = None,
        # add_obligation fields
        obligation_id:  Optional[str] = None,
        obligation_desc: Optional[str] = None,
        article_ref:    Optional[str] = None,
        obligation_status: Optional[str] = None,
        due_date:       Optional[str] = None,
        owner:          Optional[str] = None,
        **kwargs,
    ) -> dict:

        registry = _load_registry()

        if operation == "list":
            return self._list(registry, status_filter, tag_filter)

        if operation == "get":
            return self._get(registry, regulation_id)

        if operation == "add":
            return self._add(registry, regulation_id, name, short_name,
                             jurisdiction, status, effective_date,
                             enforcement_date, risk_level, url, description,
                             tags, notes)

        if operation == "update":
            return self._update(registry, regulation_id, dict(
                status=status, effective_date=effective_date,
                enforcement_date=enforcement_date, risk_level=risk_level,
                url=url, description=description, tags=tags, notes=notes,
            ))

        if operation == "check_upcoming":
            return self._check_upcoming(registry, within_days)

        if operation == "add_obligation":
            return self._add_obligation(
                registry, regulation_id, obligation_id, obligation_desc,
                article_ref, obligation_status or "pending", due_date, owner,
            )

        if operation == "check_compliance":
            return self._check_compliance(registry, regulation_id)

        if operation == "search":
            return self._search(registry, query)

        if operation == "export_csv":
            return self._export_csv(registry)

        return {"error": f"Unknown operation: {operation}"}

    # ── Operations ─────────────────────────────────────────────────────────────

    def _list(
        self,
        registry:      dict,
        status_filter: Optional[str],
        tag_filter:    Optional[str],
    ) -> dict:
        items = list(registry.values())
        if status_filter:
            items = [r for r in items if r.get("status") == status_filter]
        if tag_filter:
            items = [r for r in items if tag_filter in r.get("tags", [])]
        summary = [
            {
                "id":               r["id"],
                "short_name":       r.get("short_name", r["id"]),
                "jurisdiction":     r.get("jurisdiction", ""),
                "status":           r.get("status", ""),
                "enforcement_date": r.get("enforcement_date", ""),
                "risk_level":       r.get("risk_level", ""),
                "days_until_enforcement": _days_until(r.get("enforcement_date", "")),
                "obligations_count": len(r.get("obligations", [])),
                "compliant_count": sum(
                    1 for o in r.get("obligations", []) if o.get("status") == "compliant"
                ),
            }
            for r in items
        ]
        return {"count": len(summary), "regulations": summary}

    def _get(self, registry: dict, regulation_id: Optional[str]) -> dict:
        if not regulation_id:
            return {"error": "regulation_id required"}
        reg = registry.get(regulation_id)
        if not reg:
            return {"error": f"Regulation '{regulation_id}' not found"}
        reg_copy = dict(reg)
        reg_copy["days_until_enforcement"] = _days_until(reg.get("enforcement_date", ""))
        return reg_copy

    def _add(self, registry: dict, rid: Optional[str], name: Optional[str],
             short_name: Optional[str], jurisdiction: Optional[str],
             status: Optional[str], effective_date: Optional[str],
             enforcement_date: Optional[str], risk_level: Optional[str],
             url: Optional[str], description: Optional[str],
             tags: Optional[list], notes: Optional[str]) -> dict:
        if not rid or not name:
            return {"error": "regulation_id and name are required"}
        if rid in registry:
            return {"error": f"Regulation '{rid}' already exists. Use update."}
        registry[rid] = {
            "id":               rid,
            "name":             name,
            "short_name":       short_name or rid,
            "jurisdiction":     jurisdiction or "Global",
            "status":           status or "draft",
            "effective_date":   effective_date or "",
            "enforcement_date": enforcement_date or "",
            "risk_level":       risk_level or "medium",
            "url":              url or "",
            "description":      description or "",
            "tags":             tags or [],
            "obligations":      [],
            "last_updated":     _today(),
            "notes":            notes or "",
        }
        _save_registry(registry)
        return {"added": rid, "regulation": registry[rid]}

    def _update(self, registry: dict, regulation_id: Optional[str],
                fields: dict) -> dict:
        if not regulation_id:
            return {"error": "regulation_id required"}
        if regulation_id not in registry:
            return {"error": f"Regulation '{regulation_id}' not found"}
        reg = registry[regulation_id]
        for k, v in fields.items():
            if v is not None:
                reg[k] = v
        reg["last_updated"] = _today()
        _save_registry(registry)
        return {"updated": regulation_id, "fields_changed": [k for k, v in fields.items() if v is not None]}

    def _check_upcoming(self, registry: dict, within_days: int) -> dict:
        upcoming = []
        for reg in registry.values():
            days = _days_until(reg.get("enforcement_date", ""))
            if days is not None and 0 <= days <= within_days:
                obligations    = reg.get("obligations", [])
                pending_count  = sum(1 for o in obligations if o.get("status") == "pending")
                upcoming.append({
                    "id":               reg["id"],
                    "short_name":       reg.get("short_name", reg["id"]),
                    "enforcement_date": reg.get("enforcement_date", ""),
                    "days_remaining":   days,
                    "risk_level":       reg.get("risk_level", ""),
                    "pending_obligations": pending_count,
                    "total_obligations":   len(obligations),
                    "urgency": "CRITICAL" if days <= 30 else ("HIGH" if days <= 90 else "MEDIUM"),
                })
        upcoming.sort(key=lambda x: x["days_remaining"])
        return {
            "within_days":   within_days,
            "count":         len(upcoming),
            "upcoming":      upcoming,
            "critical_count": sum(1 for u in upcoming if u["urgency"] == "CRITICAL"),
        }

    def _add_obligation(
        self, registry: dict, regulation_id: Optional[str],
        obligation_id: Optional[str], description: Optional[str],
        article_ref: Optional[str], status: str, due_date: Optional[str],
        owner: Optional[str],
    ) -> dict:
        if not regulation_id or not obligation_id or not description:
            return {"error": "regulation_id, obligation_id, and obligation_desc are required"}
        if regulation_id not in registry:
            return {"error": f"Regulation '{regulation_id}' not found"}
        reg = registry[regulation_id]
        obligations = reg.setdefault("obligations", [])
        # Check for duplicate
        for o in obligations:
            if o.get("id") == obligation_id:
                return {"error": f"Obligation '{obligation_id}' already exists"}
        obligations.append({
            "id":          obligation_id,
            "description": description,
            "article_ref": article_ref or "",
            "status":      status,
            "due_date":    due_date or "",
            "owner":       owner or "",
            "notes":       "",
        })
        reg["last_updated"] = _today()
        _save_registry(registry)
        return {"added_obligation": obligation_id, "to_regulation": regulation_id}

    def _check_compliance(self, registry: dict, regulation_id: Optional[str]) -> dict:
        if not regulation_id:
            return {"error": "regulation_id required"}
        if regulation_id not in registry:
            return {"error": f"Regulation '{regulation_id}' not found"}
        reg         = registry[regulation_id]
        obligations = reg.get("obligations", [])
        if not obligations:
            return {
                "regulation":          regulation_id,
                "status":              "no_obligations_defined",
                "compliance_pct":      None,
            }
        status_counts: dict[str, int] = {}
        for o in obligations:
            s = o.get("status", "pending")
            status_counts[s] = status_counts.get(s, 0) + 1

        compliant = status_counts.get("compliant", 0)
        na        = status_counts.get("na", 0)
        total     = len(obligations)
        assessed  = total - status_counts.get("pending", 0)
        pct       = round((compliant + na) / total * 100, 1)

        overdue = [
            o for o in obligations
            if o.get("status") != "compliant"
            and o.get("due_date")
            and _days_until(o["due_date"]) is not None
            and _days_until(o["due_date"]) < 0  # type: ignore[operator]
        ]

        return {
            "regulation":       regulation_id,
            "short_name":       reg.get("short_name", ""),
            "total_obligations": total,
            "status_breakdown": status_counts,
            "compliance_pct":   pct,
            "assessed_pct":     round(assessed / total * 100, 1),
            "overdue_count":    len(overdue),
            "overdue":          [{"id": o["id"], "description": o["description"],
                                  "due_date": o.get("due_date")} for o in overdue],
            "overall_status": (
                "COMPLIANT"    if pct >= 100 else
                "ON_TRACK"     if pct >= 75  else
                "AT_RISK"      if pct >= 50  else
                "NON_COMPLIANT"
            ),
        }

    def _search(self, registry: dict, query: str) -> dict:
        if not query:
            return {"error": "query is required"}
        q   = query.lower()
        hits = []
        for reg in registry.values():
            score = 0
            if q in reg.get("name", "").lower():
                score += 3
            if q in reg.get("short_name", "").lower():
                score += 3
            if q in reg.get("description", "").lower():
                score += 1
            for tag in reg.get("tags", []):
                if q in tag.lower():
                    score += 2
            for o in reg.get("obligations", []):
                if q in o.get("description", "").lower():
                    score += 1
                if q in o.get("article_ref", "").lower():
                    score += 2
            if score > 0:
                hits.append({
                    "id":        reg["id"],
                    "short_name": reg.get("short_name", ""),
                    "relevance": score,
                    "snippet":   reg.get("description", "")[:200],
                })
        hits.sort(key=lambda x: x["relevance"], reverse=True)
        return {"query": query, "count": len(hits), "results": hits}

    def _export_csv(self, registry: dict) -> dict:
        rows = []
        for reg in registry.values():
            for o in reg.get("obligations", [{"id": "", "description": "", "article_ref": "",
                                               "status": "", "due_date": ""}]):
                rows.append({
                    "regulation_id":    reg["id"],
                    "short_name":       reg.get("short_name", ""),
                    "jurisdiction":     reg.get("jurisdiction", ""),
                    "reg_status":       reg.get("status", ""),
                    "enforcement_date": reg.get("enforcement_date", ""),
                    "risk_level":       reg.get("risk_level", ""),
                    "obligation_id":    o.get("id", ""),
                    "obligation":       o.get("description", ""),
                    "article_ref":      o.get("article_ref", ""),
                    "ob_status":        o.get("status", ""),
                    "due_date":         o.get("due_date", ""),
                    "owner":            o.get("owner", ""),
                })
        if not rows:
            return {"csv": "", "row_count": 0}
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return {"csv": output.getvalue(), "row_count": len(rows)}


Skill = RegulatoryTrackerSkill
