"""
Prompt Versioning Skill
Version-control system prompts with semantic versioning, quality tracking,
and one-command rollback. Stores versions as YAML under
config/prompts/versions/<agent_id>/.

Operations:
  save        — save current prompt as a new version
  load        — load a specific version
  list        — list all versions for an agent
  compare     — diff two versions side-by-side
  rollback    — activate a previous version
  set_active  — promote a version to active
  get_active  — return the currently active prompt for an agent
  record_quality — update quality stats for a version after task execution
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="prompt_versioning",
    description=(
        "Version-control system prompts with semantic versioning. "
        "Track quality scores per version, compare diffs, and rollback instantly. "
        "Stores versioned YAML in config/prompts/versions/."
    ),
    version="1.0.0",
    domains=["operations", "governance", "engineering"],
    triggers=["prompt version", "rollback prompt", "prompt history",
              "system prompt", "提示詞版本", "回滾", "版本控制"],
    requires=["pyyaml"],
    is_local=True,
)

# Default storage root (can be overridden via NEXUS_VERSIONS_DIR env)
_VERSIONS_ROOT = Path(
    os.environ.get("NEXUS_VERSIONS_DIR", "config/prompts/versions")
)


@dataclass
class PromptVersion:
    agent_id:      str
    version:       str        # semantic: "1.0.0"
    content:       str        # full system prompt text
    created_at:    float = field(default_factory=time.time)
    changelog:     str = ""
    is_active:     bool = False
    # Quality statistics (updated as tasks are executed with this version)
    task_count:    int   = 0
    avg_quality:   float = 0.0
    total_cost_usd: float = 0.0
    tags:          list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)
        )
        return d


class Skill(BaseSkill):
    meta = SKILL_META

    def _agent_dir(self, agent_id: str) -> Path:
        d = _VERSIONS_ROOT / agent_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _version_file(self, agent_id: str, version: str) -> Path:
        safe = version.replace(".", "_")
        return self._agent_dir(agent_id) / f"v{safe}.yaml"

    def _load_version(self, agent_id: str, version: str) -> Optional[PromptVersion]:
        fp = self._version_file(agent_id, version)
        if not fp.exists():
            return None
        raw = yaml.safe_load(fp.read_text(encoding="utf-8"))
        raw.pop("created_at_iso", None)
        return PromptVersion(**raw)

    def _save_version(self, pv: PromptVersion) -> None:
        fp = self._version_file(pv.agent_id, pv.version)
        fp.write_text(yaml.dump(pv.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _list_versions(self, agent_id: str) -> list[PromptVersion]:
        d = self._agent_dir(agent_id)
        versions = []
        for fp in sorted(d.glob("v*.yaml")):
            raw = yaml.safe_load(fp.read_text(encoding="utf-8"))
            raw.pop("created_at_iso", None)
            versions.append(PromptVersion(**raw))
        # Sort by semantic version (descending)
        def _ver_key(pv: PromptVersion):
            try:
                parts = [int(x) for x in pv.version.split(".")]
                return parts
            except Exception:
                return [0, 0, 0]
        versions.sort(key=_ver_key, reverse=True)
        return versions

    def _next_version(self, agent_id: str, bump: str = "minor") -> str:
        versions = self._list_versions(agent_id)
        if not versions:
            return "1.0.0"
        latest = versions[0].version
        try:
            major, minor, patch = [int(x) for x in latest.split(".")]
        except Exception:
            return "1.0.0"
        if bump == "major":
            return f"{major+1}.0.0"
        elif bump == "patch":
            return f"{major}.{minor}.{patch+1}"
        else:  # minor
            return f"{major}.{minor+1}.0"

    def _deactivate_all(self, agent_id: str) -> None:
        for pv in self._list_versions(agent_id):
            if pv.is_active:
                pv.is_active = False
                self._save_version(pv)

    async def run(
        self,
        operation: str,
        agent_id: str = "master_orchestrator",
        content: str = "",
        version: str = "",
        version_a: str = "",
        version_b: str = "",
        changelog: str = "",
        bump: str = "minor",        # major | minor | patch
        quality_score: float = 0.0,
        cost_usd: float = 0.0,
        tags: Optional[list[str]] = None,
        **kwargs,
    ) -> Any:
        op = operation.lower()

        # ── save ──────────────────────────────────────────────────────────────
        if op == "save":
            if not content:
                return {"error": "content is required for save"}
            new_ver = self._next_version(agent_id, bump)
            pv = PromptVersion(
                agent_id=agent_id,
                version=new_ver,
                content=content,
                changelog=changelog,
                tags=tags or [],
                is_active=False,
            )
            self._save_version(pv)
            return {
                "saved":    True,
                "version":  new_ver,
                "agent_id": agent_id,
                "file":     str(self._version_file(agent_id, new_ver)),
            }

        # ── load ──────────────────────────────────────────────────────────────
        if op == "load":
            if not version:
                # Load active version
                for pv in self._list_versions(agent_id):
                    if pv.is_active:
                        return pv.to_dict()
                return {"error": "No active version found"}
            pv = self._load_version(agent_id, version)
            if pv is None:
                return {"error": f"Version {version} not found for {agent_id}"}
            return pv.to_dict()

        # ── list ──────────────────────────────────────────────────────────────
        if op == "list":
            versions = self._list_versions(agent_id)
            return {
                "agent_id": agent_id,
                "versions": [
                    {
                        "version":    v.version,
                        "is_active":  v.is_active,
                        "created_at": time.strftime("%Y-%m-%d %H:%M", time.gmtime(v.created_at)),
                        "task_count": v.task_count,
                        "avg_quality": round(v.avg_quality, 3),
                        "changelog":  v.changelog,
                        "tags":       v.tags,
                    }
                    for v in versions
                ],
                "total": len(versions),
            }

        # ── compare ───────────────────────────────────────────────────────────
        if op == "compare":
            if not version_a or not version_b:
                return {"error": "version_a and version_b required"}
            pva = self._load_version(agent_id, version_a)
            pvb = self._load_version(agent_id, version_b)
            if pva is None:
                return {"error": f"Version {version_a} not found"}
            if pvb is None:
                return {"error": f"Version {version_b} not found"}

            import difflib
            diff = list(difflib.unified_diff(
                pva.content.splitlines(keepends=True),
                pvb.content.splitlines(keepends=True),
                fromfile=f"v{version_a}",
                tofile=f"v{version_b}",
                lineterm="",
            ))
            quality_delta = pvb.avg_quality - pva.avg_quality
            return {
                "version_a":       version_a,
                "version_b":       version_b,
                "diff":            "".join(diff) or "(no differences)",
                "quality_delta":   round(quality_delta, 3),
                "quality_a":       pva.avg_quality,
                "quality_b":       pvb.avg_quality,
                "task_count_a":    pva.task_count,
                "task_count_b":    pvb.task_count,
                "recommendation":  (
                    f"v{version_b} is better by {quality_delta:+.3f}"
                    if quality_delta > 0.02 else
                    f"v{version_a} is better by {-quality_delta:+.3f}"
                    if quality_delta < -0.02 else
                    "Statistically similar — no clear winner yet"
                ),
            }

        # ── set_active / rollback ─────────────────────────────────────────────
        if op in ("set_active", "rollback"):
            if not version:
                return {"error": "version required"}
            pv = self._load_version(agent_id, version)
            if pv is None:
                return {"error": f"Version {version} not found for {agent_id}"}
            self._deactivate_all(agent_id)
            pv.is_active = True
            self._save_version(pv)
            return {
                "activated": True,
                "version":   version,
                "agent_id":  agent_id,
                "content_preview": pv.content[:200],
            }

        # ── get_active ────────────────────────────────────────────────────────
        if op == "get_active":
            for pv in self._list_versions(agent_id):
                if pv.is_active:
                    return {"version": pv.version, "content": pv.content, "agent_id": agent_id}
            return {"error": f"No active version for {agent_id}"}

        # ── record_quality ────────────────────────────────────────────────────
        if op == "record_quality":
            if not version:
                return {"error": "version required"}
            pv = self._load_version(agent_id, version)
            if pv is None:
                return {"error": f"Version {version} not found"}
            # Incremental moving average
            n = pv.task_count
            pv.avg_quality = (pv.avg_quality * n + quality_score) / (n + 1)
            pv.task_count  += 1
            pv.total_cost_usd += cost_usd
            self._save_version(pv)
            return {
                "updated":    True,
                "version":    version,
                "new_avg":    round(pv.avg_quality, 4),
                "task_count": pv.task_count,
            }

        return {"error": f"Unknown operation: {op}. Valid: save, load, list, compare, rollback, set_active, get_active, record_quality"}
