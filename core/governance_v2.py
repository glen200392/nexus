"""
NEXUS GovernanceV2 — Extended PII Scrubbing & EU AI Act Classification
Builds on the v1 PIIScrubber with additional patterns and reversible pseudonymization.
Adds EUAIActClassifierV2 with keyword-based risk classification.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("nexus.governance_v2")


# ── Additional PII Patterns ─────────────────────────────────────────────────

_V2_PII_RULES = [
    # V1 patterns (inherited)
    (r"\b[A-Z]\d{9}\b",                               "TAIWAN_ID"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  "CREDIT_CARD"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",                "PHONE"),
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w+\b",   "EMAIL"),
    (r'(?:password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+', "CREDENTIAL"),
    (r"\b(?:sk-|pk-)[A-Za-z0-9]{20,}\b",              "API_KEY"),
    # V2 additional patterns
    (r"\b[A-Z][A-D]\d{8}\b",                           "TAIWAN_RESIDENCE_PERMIT"),
    (r"\b[A-Z]\d{8}\b",                                "PASSPORT"),
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b",                     "JAPANESE_MY_NUMBER"),
    (r"\b\d{6}-[1-4]\d{6}\b",                          "KOREAN_ID"),
    (r"\bMRN[-:]?\s?\d{6,10}\b",                       "MEDICAL_RECORD"),
    # SSN
    (r"\b\d{3}-\d{2}-\d{4}\b",                         "SSN"),
]

_COMPILED_V2_PII = [(re.compile(p, re.IGNORECASE), label) for p, label in _V2_PII_RULES]


class PIIScrubberV2:
    """
    Extended PII scrubber with additional patterns and reversible pseudonymization.

    Supports:
      - All v1 PII patterns (Taiwan ID, credit card, phone, email, credentials, API keys)
      - Taiwan residence permit, passport, Japanese My Number, Korean ID, medical records, SSN
      - Reversible tokenization: replace PII with tokens like [PII_EMAIL_1] and restore later
      - Non-reversible scrubbing: replace all PII with [REDACTED]
    """

    def scrub(self, text: str) -> str:
        """
        Non-reversible PII scrubbing. Replaces all detected PII with [REDACTED].

        Args:
            text: Input text that may contain PII.

        Returns:
            Scrubbed text with PII replaced by [REDACTED].
        """
        result = text
        for pattern, label in _COMPILED_V2_PII:
            result = pattern.sub("[REDACTED]", result)
        return result

    def tokenize(self, text: str) -> tuple[str, dict]:
        """
        Reversible pseudonymization.

        Replaces PII with tokens like [PII_EMAIL_1], [PII_SSN_1], etc.
        Returns the scrubbed text and a token map for restoration.

        Args:
            text: Input text that may contain PII.

        Returns:
            Tuple of (tokenized_text, token_map) where token_map maps
            tokens to original values.
        """
        token_map: dict[str, str] = {}
        counters: dict[str, int] = {}
        result = text

        for pattern, label in _COMPILED_V2_PII:
            matches = list(pattern.finditer(result))
            # Process in reverse order to preserve positions
            for match in reversed(matches):
                original = match.group()
                count = counters.get(label, 0) + 1
                counters[label] = count
                token = f"[PII_{label}_{count}]"
                token_map[token] = original
                result = result[:match.start()] + token + result[match.end():]

        return result, token_map

    def detokenize(self, text: str, token_map: dict) -> str:
        """
        Restore original PII values from a token map.

        Args:
            text: Tokenized text containing [PII_*] tokens.
            token_map: Mapping from tokens to original values.

        Returns:
            Text with tokens replaced by original PII values.
        """
        result = text
        for token, original in token_map.items():
            result = result.replace(token, original)
        return result

    def detect(self, text: str) -> list[str]:
        """
        Detect PII types present in text without modifying it.

        Returns:
            List of detected PII type labels.
        """
        found: list[str] = []
        for pattern, label in _COMPILED_V2_PII:
            if pattern.search(text):
                found.append(label)
        return found


# ── EU AI Act Classifier V2 ─────────────────────────────────────────────────

# Prohibited practices (Article 5) — keyword patterns
_PROHIBITED_PATTERNS = [
    (r"social\s+scor(e|ing)", "social_scoring"),
    (r"real[\-\s]?time\s+biometric", "real_time_biometric"),
    (r"manipulat(e|ion|ing)\s+(of\s+)?vulnerable", "manipulation_vulnerable"),
    (r"predictive\s+polic(e|ing)", "predictive_policing"),
    (r"emotion\s+recogni(tion|ze|zing)\s+(at\s+|in\s+)?work", "emotion_recognition_workplace"),
    (r"untargeted\s+facial\s+scrap(e|ing)", "untargeted_facial_scraping"),
    (r"subliminal\s+manipulat", "subliminal_manipulation"),
    (r"exploit(ing|ation)\s+(of\s+)?(children|minors|elderly|disabled)", "exploitation_vulnerable_groups"),
    (r"mass\s+surveillance", "mass_surveillance"),
    (r"cognitive\s+behavio(u)?ral\s+manipulat", "cognitive_behavioral_manipulation"),
    (r"biometric\s+categori(z|s)ation\s+.*(race|religion|sexual|political)", "biometric_categorization"),
    (r"indiscriminate\s+.*(scraping|collection)\s+.*facial", "indiscriminate_facial_data"),
]

# High-risk patterns (Annex III)
_HIGH_RISK_PATTERNS = [
    (r"credit\s+scor(e|ing)", "credit_scoring"),
    (r"employment\s+screening", "employment_screening"),
    (r"education\s+assess(ment|ing)", "education_assessment"),
    (r"migration\s+manage(ment|ing)", "migration_management"),
    (r"law\s+enforcement\s+profil(e|ing)", "law_enforcement_profiling"),
    (r"biometric\s+identif(y|ication)", "biometric_identification"),
    (r"critical\s+infrastructure", "critical_infrastructure"),
    (r"access\s+to\s+(essential\s+)?(public\s+)?services", "access_public_services"),
    (r"recruit(ment|ing)\s+.*(automat|ai|algorithm)", "automated_recruitment"),
    (r"insurance\s+.*(pric|risk|scor)", "insurance_risk_assessment"),
    (r"border\s+control", "border_control"),
    (r"asylum\s+.*(process|assess|evaluat)", "asylum_processing"),
    (r"judicial\s+.*(decision|sentenc)", "judicial_decision"),
    (r"democratic\s+process", "democratic_process"),
    (r"safety\s+component\s+.*(vehicle|medical|machinery)", "safety_component"),
    (r"medical\s+device\s+.*(diagnos|treat)", "medical_device_diagnostic"),
    (r"worker\s+.*(monitor|surveill|track)", "worker_monitoring"),
    (r"student\s+.*(monitor|surveill|track|evaluat)", "student_monitoring"),
]

_COMPILED_PROHIBITED = [(re.compile(p, re.IGNORECASE), label) for p, label in _PROHIBITED_PATTERNS]
_COMPILED_HIGH_RISK = [(re.compile(p, re.IGNORECASE), label) for p, label in _HIGH_RISK_PATTERNS]


class EUAIActClassifierV2:
    """
    EU AI Act risk classifier using keyword pattern matching.

    Classifies task descriptions into risk levels:
      - prohibited: Matches Article 5 banned practices
      - high: Matches Annex III high-risk AI systems
      - limited: Contains transparency-relevant keywords
      - minimal: No specific obligations
    """

    def classify(self, task_description: str) -> dict:
        """
        Classify a task description according to EU AI Act risk levels.

        Args:
            task_description: Free-text description of the AI task.

        Returns:
            Dict with keys:
              - risk_level: "prohibited", "high", "limited", or "minimal"
              - risk_score: 0.0 to 1.0
              - flags: list of matched pattern labels
              - compliant: True if risk_level is not "prohibited"
        """
        if not task_description or not task_description.strip():
            return {
                "risk_level": "minimal",
                "risk_score": 0.0,
                "flags": [],
                "compliant": True,
            }

        flags: list[str] = []

        # Check prohibited patterns
        prohibited_matches: list[str] = []
        for pattern, label in _COMPILED_PROHIBITED:
            if pattern.search(task_description):
                prohibited_matches.append(label)
                flags.append(f"prohibited:{label}")

        if prohibited_matches:
            return {
                "risk_level": "prohibited",
                "risk_score": 1.0,
                "flags": flags,
                "compliant": False,
            }

        # Check high-risk patterns
        high_risk_matches: list[str] = []
        for pattern, label in _COMPILED_HIGH_RISK:
            if pattern.search(task_description):
                high_risk_matches.append(label)
                flags.append(f"high_risk:{label}")

        if high_risk_matches:
            # Score based on number of matched patterns
            risk_score = min(0.9, 0.5 + 0.1 * len(high_risk_matches))
            return {
                "risk_level": "high",
                "risk_score": risk_score,
                "flags": flags,
                "compliant": True,
            }

        # Check for limited risk indicators
        limited_patterns = [
            r"chatbot",
            r"deep\s*fake",
            r"generated\s+content",
            r"synthetic\s+(media|image|video|audio)",
            r"ai[\-\s]generated",
        ]
        for p in limited_patterns:
            if re.search(p, task_description, re.IGNORECASE):
                flags.append("limited_risk:transparency_required")
                return {
                    "risk_level": "limited",
                    "risk_score": 0.3,
                    "flags": flags,
                    "compliant": True,
                }

        # Default: minimal risk
        return {
            "risk_level": "minimal",
            "risk_score": 0.0,
            "flags": [],
            "compliant": True,
        }
