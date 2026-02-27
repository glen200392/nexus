"""
NEXUS Governance — Cross-cutting Security, Privacy, and Quality
Runs at every layer boundary to enforce rules before data flows through.

Responsibilities:
  - PII scrubbing before cloud model calls
  - Audit logging (who did what, when, with which model)
  - Quality feedback loop: collect Critic scores → optimize prompts
  - Knowledge distillation: extract learnings from completed tasks
  - Cost tracking and alerting
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nexus.governance")

AUDIT_DB_PATH = Path(__file__).parent.parent / "data" / "audit.db"


# ─── Audit Record ────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """Immutable record of every agent action. Stored in SQLite."""
    record_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:    float = field(default_factory=time.time)
    event_type:   str = ""         # task_started | task_completed | tool_called | model_called
    task_id:      str = ""
    agent_id:     str = ""
    action:       str = ""         # human-readable description
    model_used:   str = ""
    privacy_tier: str = "INTERNAL"
    cost_usd:     float = 0.0
    tokens_used:  int = 0
    success:      bool = True
    quality_score: float = 0.0
    error:        Optional[str] = None
    # Payload hash (not content — for tamper detection)
    payload_hash: str = ""


# ─── PII Scrubber ────────────────────────────────────────────────────────────

_PII_RULES = [
    # (pattern, replacement_label)
    (r"\b[A-Z]\d{9}\b",                               "[TAIWAN_ID]"),
    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  "[CREDIT_CARD]"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",                "[PHONE]"),
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w+\b",   "[EMAIL]"),
    (r'(?:password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+', "[CREDENTIAL]"),
    (r"\b(?:sk-|pk-)[A-Za-z0-9]{20,}\b",              "[API_KEY]"),
]

_COMPILED_PII = [(re.compile(p, re.IGNORECASE), r) for p, r in _PII_RULES]


class PIIScrubber:
    """Removes PII patterns before sending text to cloud models."""

    @staticmethod
    def scrub(text: str) -> tuple[str, list[str]]:
        """
        Returns (scrubbed_text, list_of_detected_types).
        Detected types tells caller what was found for audit logging.
        """
        found = []
        for pattern, replacement in _COMPILED_PII:
            matches = pattern.findall(text)
            if matches:
                found.append(replacement.strip("[]"))
                text = pattern.sub(replacement, text)
        return text, found

    @staticmethod
    def hash_for_audit(text: str) -> str:
        """One-way hash of original content for tamper detection."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─── Audit Logger ─────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Writes immutable audit records to SQLite.
    This is the single source of truth for what happened and when.
    """

    def __init__(self, db_path: Path = AUDIT_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                record_id     TEXT PRIMARY KEY,
                timestamp     REAL NOT NULL,
                event_type    TEXT,
                task_id       TEXT,
                agent_id      TEXT,
                action        TEXT,
                model_used    TEXT,
                privacy_tier  TEXT,
                cost_usd      REAL DEFAULT 0,
                tokens_used   INTEGER DEFAULT 0,
                success       INTEGER DEFAULT 1,
                quality_score REAL DEFAULT 0,
                error         TEXT,
                payload_hash  TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task ON audit_log(task_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent ON audit_log(agent_id)"
        )
        self._conn.commit()

    def log(self, record: AuditRecord) -> None:
        try:
            self._conn.execute(
                """INSERT INTO audit_log VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.record_id, record.timestamp, record.event_type,
                    record.task_id, record.agent_id, record.action,
                    record.model_used, record.privacy_tier, record.cost_usd,
                    record.tokens_used, int(record.success), record.quality_score,
                    record.error, record.payload_hash,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.error("Audit log write failed: %s", exc)

    def get_task_history(self, task_id: str) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT * FROM audit_log WHERE task_id = ? ORDER BY timestamp",
            (task_id,)
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_cost_summary(self, since_timestamp: Optional[float] = None) -> dict:
        since = since_timestamp or (time.time() - 86400)  # last 24h
        cursor = self._conn.execute(
            """SELECT agent_id, model_used, SUM(cost_usd) as total_cost,
                      COUNT(*) as call_count, AVG(quality_score) as avg_quality
               FROM audit_log
               WHERE timestamp > ?
               GROUP BY agent_id, model_used
               ORDER BY total_cost DESC""",
            (since,)
        )
        cols = [d[0] for d in cursor.description]
        return {"rows": [dict(zip(cols, row)) for row in cursor.fetchall()]}


# ─── Quality Optimizer ────────────────────────────────────────────────────────

class QualityOptimizer:
    """
    Collects quality scores from completed tasks and uses them to:
    1. Track which prompt versions perform best
    2. Identify low-performing agent + model combinations
    3. Extract knowledge from high-quality outputs for the knowledge base
    4. Alert when quality drops below threshold
    """

    QUALITY_ALERT_THRESHOLD = 0.5   # Avg quality below this triggers alert
    MIN_SAMPLES_FOR_ANALYSIS = 10   # Don't analyze until we have enough data

    def __init__(self, audit_logger: AuditLogger, memory_store=None):
        self.audit   = audit_logger
        self.memory  = memory_store
        self._scores: dict[str, list[float]] = {}   # agent_id → [scores]

    async def record(self, completed_task) -> None:
        """Called by Master Orchestrator after each task completion."""
        agent_id = getattr(completed_task, "assigned_swarm", "unknown")
        score    = getattr(completed_task, "quality_score", 0.5)

        if agent_id not in self._scores:
            self._scores[agent_id] = []
        self._scores[agent_id].append(score)

        # Log to audit
        self.audit.log(AuditRecord(
            event_type="task_completed",
            task_id=completed_task.task_id,
            agent_id=agent_id,
            action=f"Task completed via {completed_task.workflow_pattern.value}",
            quality_score=score,
            success=(completed_task.status.value == "completed"),
            cost_usd=completed_task.total_cost_usd,
        ))

        # Knowledge distillation: if high quality, save to memory
        if score >= 0.85 and self.memory and completed_task.final_result:
            await self._distill_knowledge(completed_task)

        # Quality alert
        self._check_quality_trend(agent_id)

    async def _distill_knowledge(self, task) -> None:
        """Extract high-quality outputs as reusable knowledge."""
        from nexus.knowledge.rag.schema import MemoryRecord, DocumentType, MemoryType
        record = MemoryRecord(
            content=str(task.final_result)[:2000],
            memory_type=MemoryType.SEMANTIC,
            doc_type=DocumentType.FACT,
            source=f"task:{task.task_id}",
            domain=task.domain,
            quality_score=task.quality_score,
            tags=["auto_distilled", task.domain, task.workflow_pattern.value],
        )
        await self.memory.upsert(record)
        logger.info("Knowledge distilled from high-quality task %s", task.task_id[:8])

    def _check_quality_trend(self, agent_id: str) -> None:
        scores = self._scores.get(agent_id, [])
        if len(scores) < self.MIN_SAMPLES_FOR_ANALYSIS:
            return
        recent = scores[-10:]  # last 10 tasks
        avg = sum(recent) / len(recent)
        if avg < self.QUALITY_ALERT_THRESHOLD:
            logger.warning(
                "⚠️  Quality alert: agent '%s' avg score %.2f (last 10 tasks). "
                "Consider updating system prompt or switching models.",
                agent_id, avg,
            )

    def report(self) -> dict:
        return {
            agent: {
                "avg_score": sum(scores) / len(scores),
                "task_count": len(scores),
                "trend": "↑" if len(scores) >= 3 and scores[-1] > scores[-3] else "↓",
            }
            for agent, scores in self._scores.items()
            if scores
        }


# ─── Governance Manager (facade) ─────────────────────────────────────────────

class GovernanceManager:
    """
    Single entry point for all governance functions.
    Injected into agents and orchestrators.
    """

    def __init__(self, db_path: Path = AUDIT_DB_PATH, memory_store=None):
        self.scrubber  = PIIScrubber()
        self.audit     = AuditLogger(db_path)
        self.optimizer = QualityOptimizer(self.audit, memory_store)

    def guard_cloud_call(
        self,
        text: str,
        privacy_tier: str,
        model: str,
    ) -> tuple[str, bool]:
        """
        Call before sending text to a cloud model.
        Returns (safe_text, ok_to_proceed).
        Blocks call if PRIVATE tier requests a cloud model.
        """
        if privacy_tier == "PRIVATE" and not model.startswith("ollama/"):
            logger.error(
                "PRIVACY VIOLATION BLOCKED: PRIVATE tier text was about to be sent "
                "to cloud model '%s'. Aborting.", model
            )
            return text, False

        scrubbed, found = self.scrubber.scrub(text)
        if found:
            logger.info("PII scrubbed before cloud call: %s", found)
            self.audit.log(AuditRecord(
                event_type="pii_scrubbed",
                action=f"Scrubbed: {', '.join(found)}",
                model_used=model,
                privacy_tier=privacy_tier,
                payload_hash=self.scrubber.hash_for_audit(text),
            ))
        return scrubbed, True
