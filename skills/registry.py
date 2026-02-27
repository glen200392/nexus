"""
NEXUS Skill Registry — Layer 4 Tool Management
Discovers, loads, and manages skills from the skills/implementations/ directory.
Each skill is a SKILL.md (documentation) + Python module (implementation).
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nexus.skills")

SKILLS_DIR = Path(__file__).parent / "implementations"


# ─── Skill Metadata ───────────────────────────────────────────────────────────

@dataclass
class SkillMeta:
    """
    Describes what a skill can do and when to use it.
    Parsed from SKILL.md frontmatter.
    """
    name:        str
    description: str
    version:     str = "1.0.0"
    domains:     list[str] = field(default_factory=list)   # which domains can use it
    triggers:    list[str] = field(default_factory=list)   # keywords that suggest this skill
    requires:    list[str] = field(default_factory=list)   # pip packages needed
    is_local:    bool = True                               # False if requires external service
    module_path: str = ""
    skill_md:    str = ""   # full SKILL.md content for agent context


# ─── Skill Interface ──────────────────────────────────────────────────────────

class BaseSkill:
    """
    All skills must implement this interface.

    Skill module = skills/implementations/<name>/__init__.py
    Must contain: SKILL_META (SkillMeta), Skill(BaseSkill) class

    Example:
        # skills/implementations/excel_designer/__init__.py
        SKILL_META = SkillMeta(name="excel_designer", ...)
        class Skill(BaseSkill):
            async def run(self, data, template=None): ...
    """
    meta: SkillMeta

    async def run(self, **kwargs) -> Any:
        raise NotImplementedError

    def describe(self) -> str:
        """Return SKILL.md content for injection into agent context."""
        return self.meta.skill_md or f"Skill: {self.meta.name}\n{self.meta.description}"


# ─── Skill Registry ───────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Auto-discovers and loads all skills from skills/implementations/.
    Provides lookup by name and by domain/keyword.

    File structure expected:
        skills/implementations/
            excel_designer/
                __init__.py      ← contains SKILL_META and Skill class
                SKILL.md         ← documentation
                assets/          ← optional templates
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        self._dir   = skills_dir or SKILLS_DIR
        self._skills: dict[str, BaseSkill] = {}
        self._meta:   dict[str, SkillMeta] = {}

    def load_all(self) -> int:
        """Discover and load all skills. Returns count loaded."""
        count = 0
        if not self._dir.exists():
            logger.warning("Skills dir not found: %s", self._dir)
            return 0

        for skill_dir in self._dir.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("_"):
                try:
                    self._load_skill(skill_dir)
                    count += 1
                except Exception as exc:
                    logger.warning("Failed to load skill '%s': %s", skill_dir.name, exc)

        logger.info("Loaded %d skills from %s", count, self._dir)
        return count

    def _load_skill(self, skill_dir: Path) -> None:
        init_file = skill_dir / "__init__.py"
        if not init_file.exists():
            return

        # Load module dynamically
        # NOTE: module must be registered in sys.modules BEFORE exec_module so that
        # Python 3.13's dataclasses can resolve cls.__module__ via sys.modules.
        import sys as _sys
        module_name = f"nexus.skills.implementations.{skill_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, init_file)
        module = importlib.util.module_from_spec(spec)
        _sys.modules[module_name] = module   # register before exec so @dataclass works
        spec.loader.exec_module(module)

        # Require SKILL_META and Skill class
        if not hasattr(module, "SKILL_META") or not hasattr(module, "Skill"):
            logger.warning("Skill '%s' missing SKILL_META or Skill class", skill_dir.name)
            return

        meta: SkillMeta = module.SKILL_META

        # Load SKILL.md if present
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            meta.skill_md = skill_md.read_text(encoding="utf-8")

        meta.module_path = str(init_file)

        # Instantiate
        skill_instance: BaseSkill = module.Skill()
        skill_instance.meta = meta

        self._skills[meta.name] = skill_instance
        self._meta[meta.name]   = meta
        logger.debug("Loaded skill: %s v%s", meta.name, meta.version)

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def register(self, skill: BaseSkill) -> None:
        """Manually register a skill (for testing or dynamic loading)."""
        self._skills[skill.meta.name] = skill
        self._meta[skill.meta.name]   = skill.meta

    def find_by_domain(self, domain: str) -> list[SkillMeta]:
        return [m for m in self._meta.values() if domain in m.domains or "any" in m.domains]

    def find_by_keyword(self, text: str) -> list[SkillMeta]:
        text_lower = text.lower()
        return [
            m for m in self._meta.values()
            if any(t.lower() in text_lower for t in m.triggers)
        ]

    def list_all(self) -> list[SkillMeta]:
        return list(self._meta.values())

    def describe_all(self) -> str:
        """Returns markdown listing of all skills — can be injected into agent context."""
        if not self._meta:
            return "No skills loaded."
        lines = ["## Available Skills\n"]
        for meta in self._meta.values():
            lines.append(f"### `{meta.name}` v{meta.version}")
            lines.append(f"{meta.description}")
            if meta.domains:
                lines.append(f"Domains: {', '.join(meta.domains)}")
            lines.append("")
        return "\n".join(lines)


# ─── Global singleton ────────────────────────────────────────────────────────
_registry_instance: Optional[SkillRegistry] = None

def get_registry() -> SkillRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SkillRegistry()
        _registry_instance.load_all()
    return _registry_instance
