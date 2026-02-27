"""
NEXUS GuardrailsEngine — Content Safety & Compliance
Checks inputs and outputs against configurable rules for:
  - PII detection and scrubbing
  - Prompt injection blocking
  - Shell injection blocking
  - Output length warnings
  - Custom regex/static analysis rules

Rules are loaded from YAML config files in config/guardrails/.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("nexus.orchestrator.guardrails")


# ── Enums & Data Classes ─────────────────────────────────────────────────────

class GuardrailAction(str, Enum):
    BLOCK = "block"
    SCRUB = "scrub"
    WARN = "warn"
    FLAG = "flag"


@dataclass
class GuardrailRule:
    """A single guardrail rule definition."""
    name: str
    type: str            # "regex" or "static_analysis"
    stage: str           # "input" or "output"
    action: GuardrailAction
    pattern: str = ""
    description: str = ""


@dataclass
class GuardrailResult:
    """Result of checking content against a single rule."""
    rule_name: str
    action: GuardrailAction
    triggered: bool
    detail: str = ""
    scrubbed_content: str | None = None


# ── GuardrailsEngine ─────────────────────────────────────────────────────────

class GuardrailsEngine:
    """
    Configurable content safety engine.

    Usage:
        engine = GuardrailsEngine()
        count = engine.load_rules_from_config("config/guardrails")
        results = await engine.check_input(user_message)
        if engine.has_blocking(results):
            raise PermissionError("Input blocked by guardrails")

        output = "..."
        results = await engine.check_output(output)
    """

    def __init__(self):
        self._rules: list[GuardrailRule] = []

    def load_rules_from_config(self, config_dir: str | Path) -> int:
        """
        Load guardrail rules from YAML files in the given directory.
        Returns the number of rules loaded.
        """
        config_path = Path(config_dir)
        if not config_path.exists():
            logger.warning("Guardrails config dir not found: %s", config_path)
            return 0

        count = 0
        for yaml_file in sorted(config_path.glob("*.yaml")):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "rules" in data:
                    for rule_data in data["rules"]:
                        rule = GuardrailRule(
                            name=rule_data["name"],
                            type=rule_data.get("type", "regex"),
                            stage=rule_data.get("stage", "output"),
                            action=GuardrailAction(rule_data.get("action", "warn")),
                            pattern=rule_data.get("pattern", ""),
                            description=rule_data.get("description", ""),
                        )
                        self._rules.append(rule)
                        count += 1
                    logger.debug("Loaded %d rules from %s", len(data["rules"]), yaml_file.name)
            except Exception as exc:
                logger.error("Failed to load guardrails from %s: %s", yaml_file, exc)

        logger.info("Loaded %d guardrail rules from %s", count, config_path)
        return count

    def add_rule(self, rule: GuardrailRule) -> None:
        """Add a single guardrail rule."""
        self._rules.append(rule)

    def get_rule(self, name: str) -> GuardrailRule | None:
        """Get a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                return rule
        return None

    def list_rules(self) -> list[GuardrailRule]:
        """Return all registered rules."""
        return list(self._rules)

    async def check_input(
        self, content: str, rule_names: list[str] | None = None,
    ) -> list[GuardrailResult]:
        """
        Check content against all input-stage rules.
        Optionally filter to specific rule names.
        """
        return self._check_stage(content, "input", rule_names)

    async def check_output(
        self, content: str, rule_names: list[str] | None = None,
    ) -> list[GuardrailResult]:
        """
        Check content against all output-stage rules.
        Optionally filter to specific rule names.
        """
        return self._check_stage(content, "output", rule_names)

    def _check_stage(
        self, content: str, stage: str, rule_names: list[str] | None = None,
    ) -> list[GuardrailResult]:
        """Run all rules for a given stage against content."""
        results: list[GuardrailResult] = []
        for rule in self._rules:
            if rule.stage != stage:
                continue
            if rule_names is not None and rule.name not in rule_names:
                continue

            if rule.type == "regex":
                result = self._check_regex(content, rule)
            elif rule.type == "static_analysis":
                result = self._check_static(content, rule)
            else:
                logger.warning("Unknown rule type '%s' for rule '%s'", rule.type, rule.name)
                continue

            results.append(result)

        return results

    def _check_regex(self, content: str, rule: GuardrailRule) -> GuardrailResult:
        """Check content against a regex pattern."""
        try:
            match = re.search(rule.pattern, content)
            triggered = match is not None

            scrubbed = None
            if triggered and rule.action == GuardrailAction.SCRUB:
                scrubbed = re.sub(rule.pattern, "[REDACTED]", content)

            detail = ""
            if triggered:
                detail = f"Pattern matched: {rule.description or rule.pattern}"
                if match:
                    detail += f" (found: '{match.group()[:50]}')"

            return GuardrailResult(
                rule_name=rule.name,
                action=rule.action,
                triggered=triggered,
                detail=detail,
                scrubbed_content=scrubbed,
            )
        except re.error as exc:
            logger.error("Invalid regex in rule '%s': %s", rule.name, exc)
            return GuardrailResult(
                rule_name=rule.name,
                action=rule.action,
                triggered=False,
                detail=f"Regex error: {exc}",
            )

    def _check_static(self, content: str, rule: GuardrailRule) -> GuardrailResult:
        """
        Static analysis check. Supports:
          - max_chars:N — warn/block if content exceeds N characters
          - keyword:word1,word2 — check for keyword presence
        """
        pattern = rule.pattern.strip()

        if pattern.startswith("max_chars:"):
            try:
                max_chars = int(pattern.split(":", 1)[1])
                triggered = len(content) > max_chars
                detail = (
                    f"Content length {len(content)} exceeds limit {max_chars}"
                    if triggered else ""
                )
                return GuardrailResult(
                    rule_name=rule.name,
                    action=rule.action,
                    triggered=triggered,
                    detail=detail,
                )
            except ValueError:
                return GuardrailResult(
                    rule_name=rule.name,
                    action=rule.action,
                    triggered=False,
                    detail=f"Invalid max_chars value in pattern: {pattern}",
                )

        if pattern.startswith("keyword:"):
            keywords = [k.strip() for k in pattern.split(":", 1)[1].split(",")]
            found = [kw for kw in keywords if kw.lower() in content.lower()]
            triggered = len(found) > 0
            detail = f"Keywords found: {found}" if triggered else ""
            scrubbed = None
            if triggered and rule.action == GuardrailAction.SCRUB:
                scrubbed = content
                for kw in found:
                    scrubbed = re.sub(re.escape(kw), "[REDACTED]", scrubbed, flags=re.IGNORECASE)
            return GuardrailResult(
                rule_name=rule.name,
                action=rule.action,
                triggered=triggered,
                detail=detail,
                scrubbed_content=scrubbed,
            )

        # Fallback: treat pattern as a simple substring check
        triggered = pattern.lower() in content.lower() if pattern else False
        return GuardrailResult(
            rule_name=rule.name,
            action=rule.action,
            triggered=triggered,
            detail=f"Pattern '{pattern}' found in content" if triggered else "",
        )

    def has_blocking(self, results: list[GuardrailResult]) -> bool:
        """Return True if any result has BLOCK action and was triggered."""
        return any(
            r.triggered and r.action == GuardrailAction.BLOCK
            for r in results
        )
