"""
NEXUS Perception Layer — Layer 2
Transforms raw TaskEvents into fully-analyzed PerceivedTasks.
Uses a lightweight local LLM (qwen2.5:7b) — always runs privately.

Analyzes:
  - Intent: What does the user/trigger want?
  - Domain: Which swarm should handle this?
  - Complexity: Low/Medium/High/Critical?
  - Privacy tier: Can we use cloud models?
  - Required capabilities: Which agents/skills are needed?
  - Context: Relevant memories to pre-load?
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from nexus.core.llm.router import PrivacyTier, TaskComplexity, TaskDomain
from nexus.core.orchestrator.trigger import TaskEvent

logger = logging.getLogger("nexus.perception")


# ─── Perceived Task ──────────────────────────────────────────────────────────

@dataclass
class PerceivedTask:
    """
    Fully-analyzed task ready for the Management Layer (Layer 3).
    This is the contract between Layer 2 and Layer 3.
    """
    # From original event
    event_id:      str = ""
    session_id:    str = ""
    user_message:  str = ""

    # Perception analysis results
    intent:        str = ""            # 1-sentence description of what's wanted
    task_type:     str = "general"     # research | code | write | analyze | operate…
    domain:        TaskDomain = TaskDomain.RESEARCH
    complexity:    TaskComplexity = TaskComplexity.MEDIUM
    privacy_tier:  PrivacyTier = PrivacyTier.INTERNAL

    # Capability requirements
    required_agents:  list[str] = field(default_factory=list)
    required_skills:  list[str] = field(default_factory=list)
    required_mcp:     list[str] = field(default_factory=list)

    # Risk assessment
    requires_confirmation: bool = False   # True if action is irreversible/external
    has_pii:               bool = False   # True if input contains personal data
    is_destructive:        bool = False   # True if could delete/overwrite

    # Context hints
    key_entities:    list[str] = field(default_factory=list)  # names, files, URLs
    memory_queries:  list[str] = field(default_factory=list)  # what to retrieve from memory
    language:        str = "zh-TW"

    # Pause / resume
    is_resume:       bool = False
    resume_task_id:  Optional[str] = None
    checkpoint:      Optional[dict] = None

    # Confidence of perception analysis
    analysis_confidence: float = 0.8
    analyzed_at:         float = field(default_factory=time.time)


# ─── PII Detector ────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    r"\b[A-Z]\d{9}\b",                          # Taiwan ID (A123456789)
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",           # Phone
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w+\b",  # Email
    r"\b(?:password|passwd|secret|api.?key)\s*[:=]\s*\S+",  # Credentials
]

import re as _re
_PII_RE = [_re.compile(p, _re.IGNORECASE) for p in _PII_PATTERNS]


def detect_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PII_RE)


# ─── Rule-Based Fast Path ─────────────────────────────────────────────────────
# Quick heuristics that don't require an LLM call.
# If these match confidently, skip the LLM analysis.

_FAST_PATH_RULES: list[dict] = [
    {
        "keywords": ["git commit", "git push", "git status", "git log"],
        "task_type": "code", "domain": TaskDomain.ENGINEERING,
        "complexity": TaskComplexity.LOW, "agents": ["code_agent"],
    },
    {
        "keywords": ["search", "find information", "research", "查詢", "搜尋"],
        "task_type": "research", "domain": TaskDomain.RESEARCH,
        "complexity": TaskComplexity.MEDIUM, "agents": ["web_agent", "rag_agent"],
    },
    {
        "keywords": ["summarize", "summary", "摘要", "總結"],
        "task_type": "write", "domain": TaskDomain.CREATIVE,
        "complexity": TaskComplexity.LOW, "agents": ["writer_agent"],
    },
    {
        "keywords": ["analyze data", "chart", "graph", "分析", "視覺化"],
        "task_type": "analyze", "domain": TaskDomain.ANALYSIS,
        "complexity": TaskComplexity.MEDIUM, "agents": ["analyst_agent"],
    },
    {
        "keywords": ["write code", "implement", "debug", "fix bug", "寫程式"],
        "task_type": "code", "domain": TaskDomain.ENGINEERING,
        "complexity": TaskComplexity.HIGH, "agents": ["code_agent", "test_agent"],
    },
]


# ─── Perception Engine ───────────────────────────────────────────────────────

class PerceptionEngine:
    """
    Converts raw TaskEvents into PerceivedTasks.

    Strategy:
      1. Check for pause/resume signals
      2. Detect PII → force PRIVATE tier
      3. Try rule-based fast path (no LLM call)
      4. If ambiguous, call local LLM (qwen2.5:7b) for structured analysis
      5. Build PerceivedTask with memory retrieval hints
    """

    # System prompt for the perception LLM
    # Uses qwen2.5:7b — always local, always private
    ANALYSIS_PROMPT = """You are a task analysis engine. Analyze the user's request and output ONLY valid JSON.

Classify the task with these fields:
{
  "intent": "one sentence: what the user wants",
  "task_type": "research|code|write|analyze|operate|decide|general",
  "domain": "research|engineering|operations|creative|analysis|orchestration",
  "complexity": "low|medium|high|critical",
  "privacy_tier": "PRIVATE|INTERNAL|PUBLIC",
  "required_agents": ["list of agent ids needed"],
  "required_skills": ["list of skill names needed"],
  "key_entities": ["important names, files, URLs mentioned"],
  "memory_queries": ["1-3 queries to search past memory for context"],
  "requires_confirmation": false,
  "is_destructive": false,
  "language": "zh-TW|en|other"
}

Privacy rules:
- PRIVATE: contains personal names, credentials, medical, financial data → use local models only
- INTERNAL: company/project data, no PII → cloud models OK
- PUBLIC: generic questions → any model

Complexity rules:
- critical: strategic decisions, irreversible actions, cross-system impact
- high: multi-step reasoning, code architecture, long research
- medium: standard tasks with some judgment required
- low: simple lookups, formatting, classification"""

    def __init__(
        self,
        llm_caller=None,    # Async callable: (prompt, system) → str (injected at startup)
        memory_store=None,  # For context pre-loading
    ):
        self.llm     = llm_caller
        self.memory  = memory_store

    async def analyze(self, event: TaskEvent) -> PerceivedTask:
        """Main entry point. Convert TaskEvent → PerceivedTask."""
        text = event.raw_input.strip()

        # ── 1. Pause / Resume detection ──────────────────────────────────────
        if text.startswith("__PAUSE__"):
            tid = text.replace("__PAUSE__", "")
            return PerceivedTask(event_id=event.event_id, user_message=text,
                                 task_type="pause", intent=f"Pause task {tid}",
                                 complexity=TaskComplexity.LOW, privacy_tier=PrivacyTier.PRIVATE)

        if text.startswith("__RESUME__") or event.resume_task_id:
            return PerceivedTask(
                event_id=event.event_id,
                user_message=text,
                task_type="resume",
                intent="Resume paused task",
                complexity=TaskComplexity.LOW,
                privacy_tier=PrivacyTier.PRIVATE,
                is_resume=True,
                resume_task_id=event.resume_task_id,
                checkpoint=event.checkpoint,
            )

        # ── 2. PII detection → force PRIVATE ─────────────────────────────────
        has_pii = detect_pii(text)
        forced_private = has_pii or event.hint_privacy == "PRIVATE"

        # ── 3. Rule-based fast path ───────────────────────────────────────────
        fast = self._try_fast_path(text, event)
        if fast is not None:
            if forced_private:
                fast.privacy_tier = PrivacyTier.PRIVATE
            fast.has_pii = has_pii
            return fast

        # ── 4. LLM-based analysis ─────────────────────────────────────────────
        perceived = await self._llm_analyze(text, event)
        if forced_private:
            perceived.privacy_tier = PrivacyTier.PRIVATE
        perceived.has_pii = has_pii

        # ── 5. Memory pre-loading hints ───────────────────────────────────────
        if not perceived.memory_queries:
            perceived.memory_queries = [text[:200]]  # fallback: use raw input

        return perceived

    def _try_fast_path(self, text: str, event: TaskEvent) -> Optional[PerceivedTask]:
        """Returns PerceivedTask if a rule matches confidently, else None."""
        text_lower = text.lower()
        for rule in _FAST_PATH_RULES:
            if any(kw in text_lower for kw in rule["keywords"]):
                return PerceivedTask(
                    event_id=event.event_id,
                    session_id=event.session_id,
                    user_message=text,
                    intent=f"Fast-path: {rule['task_type']} task",
                    task_type=rule["task_type"],
                    domain=rule["domain"],
                    complexity=rule["complexity"],
                    privacy_tier=PrivacyTier(event.hint_privacy or "INTERNAL"),
                    required_agents=rule["agents"],
                    analysis_confidence=0.75,
                )
        return None

    async def _llm_analyze(self, text: str, event: TaskEvent) -> PerceivedTask:
        """Use local LLM to analyze the task. Falls back to defaults on failure."""
        if self.llm is None:
            logger.warning("No LLM caller injected; using perception defaults")
            return PerceivedTask(
                event_id=event.event_id,
                session_id=event.session_id,
                user_message=text,
                intent=text[:100],
                privacy_tier=PrivacyTier(event.hint_privacy or "INTERNAL"),
            )

        try:
            raw = await self.llm(
                prompt=f"Analyze this task:\n\n{text}",
                system=self.ANALYSIS_PROMPT,
                model="qwen2.5:7b",   # Always local — perception is always private
            )
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON in LLM response")
            data: dict = json.loads(json_match.group())

            return PerceivedTask(
                event_id=event.event_id,
                session_id=event.session_id,
                user_message=text,
                intent=data.get("intent", text[:100]),
                task_type=data.get("task_type", "general"),
                domain=TaskDomain(data.get("domain", "research")),
                complexity=TaskComplexity(data.get("complexity", "medium")),
                privacy_tier=PrivacyTier(data.get("privacy_tier", "INTERNAL")),
                required_agents=data.get("required_agents", []),
                required_skills=data.get("required_skills", []),
                key_entities=data.get("key_entities", []),
                memory_queries=data.get("memory_queries", [text[:200]]),
                requires_confirmation=data.get("requires_confirmation", False),
                is_destructive=data.get("is_destructive", False),
                language=data.get("language", "zh-TW"),
                analysis_confidence=0.9,
            )

        except Exception as exc:
            logger.warning("LLM perception analysis failed: %s", exc)
            return PerceivedTask(
                event_id=event.event_id,
                session_id=event.session_id,
                user_message=text,
                intent=text[:100],
                privacy_tier=PrivacyTier(event.hint_privacy or "INTERNAL"),
                analysis_confidence=0.3,
            )
