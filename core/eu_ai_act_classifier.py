"""
NEXUS EU AI Act Classifier — Core Compliance Module
Rule-based + optional LLM review for task risk classification.

Implements Regulation (EU) 2024/1689 (EU AI Act, in force Aug 2024):
  - Article 5:     Prohibited AI practices
  - Annex III:     High-risk AI system categories
  - Article 52:    Transparency obligations (limited risk)
  - Articles 51–56: General Purpose AI (GPAI) obligations

Integration:
  GovernanceManager.check_eu_compliance(perceived_task) → ComplianceCheck
  MasterOrchestrator respects ComplianceCheck.requires_human_oversight

Usage:
    classifier = EUAIActClassifier()
    check = classifier.classify(
        task_type="employment_screening",
        domain="hr",
        description="Rank job candidates by predicted performance",
        metadata={"processes_biometric": False, "affects_fundamental_rights": True},
    )
    # check.risk_level → RiskLevel.HIGH_RISK
    # check.applicable_articles → ["Annex III §4", "Art. 9", "Art. 14"]
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("nexus.governance.eu_ai_act")


# ── Risk Levels ────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    PROHIBITED   = "prohibited"    # Art. 5 — must be blocked
    HIGH_RISK    = "high_risk"     # Annex III — strict obligations
    LIMITED_RISK = "limited_risk"  # Art. 52 — transparency required
    MINIMAL_RISK = "minimal"       # No specific obligations
    GPAI         = "gpai"          # General Purpose AI — Art. 51–56


@dataclass
class ComplianceCheck:
    risk_level:               RiskLevel
    applicable_articles:      list[str] = field(default_factory=list)
    requires_human_oversight: bool = False
    requires_documentation:   bool = False
    requires_transparency:    bool = False
    is_blocked:               bool = False       # True for prohibited practices
    reasoning:                str = ""
    mitigation_required:      list[str] = field(default_factory=list)
    confidence:               str = "high"       # high | medium | low


# ── Prohibited Practice Patterns (Article 5) ──────────────────────────────────
# These cause is_blocked=True — task should not proceed

_PROHIBITED_PATTERNS = [
    (r"social\s+scor(?:e|ing)", "Art. 5(1)(c): Social scoring by public authorities"),
    (r"subliminal\s+(?:manipulation|message|technique)", "Art. 5(1)(a): Subliminal manipulation"),
    (r"real.?time\s+biometric\s+(?:surveillance|identification)\s+(?:in\s+)?public",
     "Art. 5(1)(h): Real-time remote biometric surveillance in public"),
    (r"exploit\s+(?:vulnerabilit|weakness).{0,40}(?:age|disability|mental|social)",
     "Art. 5(1)(b): Exploitation of vulnerable groups"),
    (r"criminal\s+(?:prediction|profil|risk\s+scor).{0,30}(?:biometric|face|gait)",
     "Art. 5(1)(e): Biometric-based criminal risk prediction"),
    (r"emotion\s+recogni(?:tion|ze).{0,40}(?:workplace|school|education|law.?enforce)",
     "Art. 5(1)(f): Emotion recognition in workplace/education"),
]

# ── High-Risk Annex III Categories ────────────────────────────────────────────

_HIGH_RISK_RULES = [
    # §1 — Biometric
    (r"biometric.{0,30}(?:categori|identif|verification)",
     ["Annex III §1", "Art. 9", "Art. 10", "Art. 13", "Art. 14"],
     "Biometric identification or categorization system"),

    # §2 — Critical infrastructure
    (r"(?:power\s+grid|water\s+supply|financial\s+infrastructure|traffic\s+management).{0,30}(?:control|automat|manag)",
     ["Annex III §2", "Art. 9", "Art. 15"],
     "Critical infrastructure management"),

    # §3 — Education
    (r"(?:student|pupil|applicant).{0,40}(?:admission|grade|score|evaluat|assess).{0,30}(?:school|university|education|exam)",
     ["Annex III §3", "Art. 13", "Art. 14"],
     "Educational assessment or admission decision"),

    # §4 — Employment
    (r"(?:recruit|hir(?:e|ing)|screen).{0,40}(?:candidate|applicant|employee|worker|resume|cv)",
     ["Annex III §4", "Art. 9", "Art. 13", "Art. 14"],
     "Employment recruitment or screening"),
    (r"(?:promot|terminat|evaluat).{0,40}(?:employee|worker|staff).{0,30}(?:automat|AI|model)",
     ["Annex III §4", "Art. 14"],
     "Automated employee evaluation or promotion/termination"),

    # §5 — Essential services (credit, insurance, public benefits)
    (r"(?:credit\s+scor|loan\s+(?:decision|approv)|creditworthiness)",
     ["Annex III §5", "Art. 9", "Art. 13"],
     "Credit scoring or lending decision"),
    (r"(?:insurance\s+(?:pric|risk|premium|underwriting))",
     ["Annex III §5", "Art. 9", "Art. 13"],
     "Insurance risk assessment"),
    (r"(?:social\s+benefit|welfare\s+(?:eligibility|decision))",
     ["Annex III §5", "Art. 13", "Art. 14"],
     "Social benefit eligibility determination"),

    # §6 — Law enforcement
    (r"(?:law\s+enforce|police|crime\s+predict|criminal\s+justice|detention\s+risk)",
     ["Annex III §6", "Art. 9", "Art. 14"],
     "Law enforcement or criminal justice application"),

    # §7 — Migration / border control
    (r"(?:asylum|migration|visa|border\s+control).{0,30}(?:automat|assess|predict|decision)",
     ["Annex III §7", "Art. 9", "Art. 14"],
     "Migration or border control decision"),

    # §8 — Administration of justice
    (r"(?:judicial|court|legal\s+decision|sentence|verdict).{0,30}(?:automat|AI|predict)",
     ["Annex III §8", "Art. 14"],
     "Administration of justice or legal decision"),

    # Medical devices (covered by other regulations but mention for awareness)
    (r"(?:medical\s+diagnosis|clinical\s+decision|patient\s+triage).{0,30}(?:automat|AI|model)",
     ["Annex III §1 (medical)", "Art. 9", "Art. 13"],
     "Medical diagnosis or clinical decision support"),
]

# ── Limited Risk Patterns (Article 52 — Transparency) ─────────────────────────

_LIMITED_RISK_PATTERNS = [
    (r"(?:chatbot|virtual\s+assistant|conversational\s+AI)", "Art. 52(1): Must disclose AI nature to users"),
    (r"(?:deepfake|synthetic\s+(?:image|video|audio)|AI.?generat(?:ed)?\s+(?:image|video|audio))",
     "Art. 52(3): Must label AI-generated content"),
    (r"(?:emotion\s+recogni(?:tion|ze)|affect\s+detect)",
     "Art. 52(2): Must inform users of emotion recognition (outside Art. 5)"),
]

# ── GPAI Detection (Articles 51–56) ───────────────────────────────────────────

_GPAI_PATTERNS = [
    r"train(?:ing)?\s+(?:a\s+)?(?:foundation|base|large)\s+model",
    r"fine.?tun(?:e|ing)\s+(?:llm|gpt|claude|gemini|llama)",
    r"deploy(?:ing)?\s+(?:a\s+)?(?:general.?purpose|foundation)\s+(?:ai|model)",
]

# ── Domain → Risk Level mapping (fast-path heuristic) ─────────────────────────

_DOMAIN_RISK_HINTS: dict[str, RiskLevel] = {
    "hr":            RiskLevel.HIGH_RISK,
    "legal":         RiskLevel.HIGH_RISK,
    "medical":       RiskLevel.HIGH_RISK,
    "financial":     RiskLevel.HIGH_RISK,
    "law_enforcement": RiskLevel.HIGH_RISK,
    "education":     RiskLevel.HIGH_RISK,
    "research":      RiskLevel.MINIMAL_RISK,
    "creative":      RiskLevel.MINIMAL_RISK,
    "operations":    RiskLevel.MINIMAL_RISK,
    "analysis":      RiskLevel.LIMITED_RISK,
}


# ── Classifier ─────────────────────────────────────────────────────────────────

class EUAIActClassifier:
    """
    Classifies AI tasks by EU AI Act risk level.
    Uses rule-based matching (fast, deterministic) for primary classification.
    Optionally calls an LLM for ambiguous cases (low confidence).
    """

    def classify(
        self,
        task_type: str = "",
        domain: str = "",
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> ComplianceCheck:
        """
        Classify a task and return a ComplianceCheck.

        Args:
            task_type:   Short task category (e.g. "employment_screening")
            domain:      NEXUS domain (hr, research, legal, operations…)
            description: Natural language task description
            metadata:    Extra signals (processes_biometric, affects_public, …)
        """
        text = f"{task_type} {domain} {description}".lower()
        meta = metadata or {}

        # ── Step 1: Check for prohibited practices (hard block) ────────────
        for pattern, article in _PROHIBITED_PATTERNS:
            if re.search(pattern, text, re.I):
                logger.warning("EU AI Act PROHIBITED practice detected: %s", article)
                return ComplianceCheck(
                    risk_level=RiskLevel.PROHIBITED,
                    applicable_articles=[article, "Art. 5"],
                    requires_human_oversight=True,
                    is_blocked=True,
                    reasoning=f"Task matches prohibited AI practice: {article}",
                    confidence="high",
                )

        # ── Step 2: Check for high-risk categories ─────────────────────────
        high_risk_match = None
        for pattern, articles, reason in _HIGH_RISK_RULES:
            if re.search(pattern, text, re.I):
                high_risk_match = (articles, reason)
                break

        # Also check metadata signals
        if meta.get("affects_fundamental_rights") or meta.get("processes_biometric"):
            if not high_risk_match:
                high_risk_match = (
                    ["Annex III", "Art. 9", "Art. 14"],
                    "Metadata indicates fundamental rights impact or biometric processing",
                )

        # Domain hint escalation
        domain_risk = _DOMAIN_RISK_HINTS.get(domain.lower(), RiskLevel.MINIMAL_RISK)
        if domain_risk == RiskLevel.HIGH_RISK and not high_risk_match:
            high_risk_match = (
                ["Annex III", "Art. 9"],
                f"Domain '{domain}' is typically high-risk under EU AI Act",
            )

        if high_risk_match:
            articles, reason = high_risk_match
            mitigations = [
                "Implement risk management system (Art. 9)",
                "Ensure training data governance (Art. 10)",
                "Create technical documentation (Art. 11)",
                "Enable human oversight mechanism (Art. 14)",
                "Register in EU AI database if public authority use (Art. 49)",
            ]
            return ComplianceCheck(
                risk_level=RiskLevel.HIGH_RISK,
                applicable_articles=articles,
                requires_human_oversight=True,
                requires_documentation=True,
                requires_transparency=True,
                is_blocked=False,
                reasoning=reason,
                mitigation_required=mitigations,
                confidence="high",
            )

        # ── Step 3: Check for GPAI ─────────────────────────────────────────
        for pattern in _GPAI_PATTERNS:
            if re.search(pattern, text, re.I):
                return ComplianceCheck(
                    risk_level=RiskLevel.GPAI,
                    applicable_articles=["Art. 51", "Art. 53", "Art. 54"],
                    requires_documentation=True,
                    requires_transparency=True,
                    reasoning="Task involves training or deploying a general-purpose AI model",
                    mitigation_required=[
                        "Comply with transparency obligations (Art. 53)",
                        "Maintain technical documentation",
                        "Respect copyright law for training data (Art. 53(1)(c))",
                    ],
                    confidence="high",
                )

        # ── Step 4: Check for limited risk (transparency obligations) ──────
        for pattern, article in _LIMITED_RISK_PATTERNS:
            if re.search(pattern, text, re.I):
                return ComplianceCheck(
                    risk_level=RiskLevel.LIMITED_RISK,
                    applicable_articles=["Art. 52", article],
                    requires_transparency=True,
                    reasoning=f"Task requires transparency disclosure: {article}",
                    mitigation_required=["Disclose AI nature to end users"],
                    confidence="high",
                )

        # ── Step 5: Minimal risk (no specific obligations) ─────────────────
        return ComplianceCheck(
            risk_level=RiskLevel.MINIMAL_RISK,
            applicable_articles=[],
            reasoning="No EU AI Act high-risk indicators detected",
            confidence="high" if domain in _DOMAIN_RISK_HINTS else "medium",
        )

    def gate_task(self, check: ComplianceCheck) -> tuple[bool, str]:
        """
        Returns (should_proceed, explanation).
        PROHIBITED → False (block).
        HIGH_RISK  → True but add confirmation requirement.
        Others     → True.
        """
        if check.is_blocked:
            return False, (
                f"Task BLOCKED: EU AI Act prohibited practice detected.\n"
                f"Articles: {', '.join(check.applicable_articles)}\n"
                f"Reason: {check.reasoning}"
            )
        if check.risk_level == RiskLevel.HIGH_RISK:
            return True, (
                f"⚠️  HIGH-RISK AI system (EU AI Act Annex III).\n"
                f"Human oversight required. Articles: {', '.join(check.applicable_articles)}\n"
                f"Required mitigations: {'; '.join(check.mitigation_required[:3])}"
            )
        return True, f"EU AI Act: {check.risk_level.value} — {check.reasoning}"


# ── Singleton ──────────────────────────────────────────────────────────────────
_classifier: Optional[EUAIActClassifier] = None

def get_classifier() -> EUAIActClassifier:
    global _classifier
    if _classifier is None:
        _classifier = EUAIActClassifier()
    return _classifier
