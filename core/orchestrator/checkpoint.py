"""
NEXUS CheckpointStore — Durable Execution State (v2)
Persists graph execution state to SQLite for crash-safe pause/resume/fork.

Each checkpoint captures:
  - The full graph state at a specific step
  - Thread lineage (parent_id for forked executions)
  - Metadata (agent, timing, cost)

Backend: SQLite via aiosqlite (swappable to Redis/PostgreSQL)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nexus.checkpoint")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "checkpoints.db"


@dataclass
class Checkpoint:
    """Snapshot of graph execution state at a specific step."""
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    node_name: str = ""
    step: int = 0
    state_json: str = "{}"  # Serialized GraphState
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent_id: Optional[str] = None  # For forked threads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> Checkpoint:
        meta = row.get("metadata", "{}")
        if isinstance(meta, str):
            meta = json.loads(meta)
        return cls(
            checkpoint_id=row["checkpoint_id"],
            thread_id=row["thread_id"],
            node_name=row.get("node_name", ""),
            step=row.get("step", 0),
            state_json=row.get("state_json", "{}"),
            metadata=meta,
            created_at=row.get("created_at", 0.0),
            parent_id=row.get("parent_id"),
        )


class CheckpointStore:
    """
    Persistent checkpoint storage backed by SQLite.

    Usage:
        store = CheckpointStore()
        await store.initialize()

        # Save during execution
        await store.save(Checkpoint(thread_id="t1", step=3, state_json=...))

        # Resume after crash
        cp = await store.load_latest("t1")

        # Fork execution
        new_thread = await store.fork("t1", step=2)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db = None

    async def initialize(self) -> None:
        """Create database and tables if needed."""
        import aiosqlite
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id     TEXT NOT NULL,
                node_name     TEXT DEFAULT '',
                step          INTEGER DEFAULT 0,
                state_json    TEXT DEFAULT '{}',
                metadata      TEXT DEFAULT '{}',
                created_at    REAL NOT NULL,
                parent_id     TEXT
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread ON checkpoints(thread_id, step)"
        )
        await self._db.commit()
        logger.info("CheckpointStore initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> None:
        if self._db is None:
            await self.initialize()

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint. Returns checkpoint_id."""
        await self._ensure_db()
        await self._db.execute(
            """INSERT OR REPLACE INTO checkpoints
               (checkpoint_id, thread_id, node_name, step, state_json, metadata, created_at, parent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.checkpoint_id,
                checkpoint.thread_id,
                checkpoint.node_name,
                checkpoint.step,
                checkpoint.state_json,
                json.dumps(checkpoint.metadata, ensure_ascii=False),
                checkpoint.created_at,
                checkpoint.parent_id,
            ),
        )
        await self._db.commit()
        logger.debug("Checkpoint saved: thread=%s step=%d", checkpoint.thread_id, checkpoint.step)
        return checkpoint.checkpoint_id

    async def load(self, thread_id: str, step: int | None = None) -> Checkpoint | None:
        """Load a specific checkpoint by thread_id and optional step."""
        await self._ensure_db()
        if step is not None:
            cursor = await self._db.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? AND step = ? ORDER BY created_at DESC LIMIT 1",
                (thread_id, step),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
                (thread_id,),
            )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Checkpoint.from_row(dict(row))

    async def load_latest(self, thread_id: str) -> Checkpoint | None:
        """Load the most recent checkpoint for a thread."""
        return await self.load(thread_id)

    async def list_checkpoints(self, thread_id: str) -> list[Checkpoint]:
        """List all checkpoints for a thread, ordered by step."""
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY step ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        return [Checkpoint.from_row(dict(r)) for r in rows]

    async def fork(self, thread_id: str, step: int) -> str:
        """
        Fork execution from a specific step, creating a new thread.
        Returns the new thread_id.
        """
        source = await self.load(thread_id, step)
        if source is None:
            raise ValueError(f"No checkpoint found for thread={thread_id} step={step}")

        new_thread_id = str(uuid.uuid4())
        forked = Checkpoint(
            thread_id=new_thread_id,
            node_name=source.node_name,
            step=source.step,
            state_json=source.state_json,
            metadata={**source.metadata, "forked_from": thread_id, "fork_step": step},
            parent_id=source.checkpoint_id,
        )
        await self.save(forked)
        logger.info("Forked thread %s → %s at step %d", thread_id[:8], new_thread_id[:8], step)
        return new_thread_id

    async def delete_thread(self, thread_id: str) -> int:
        """Delete all checkpoints for a thread. Returns count deleted."""
        await self._ensure_db()
        cursor = await self._db.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        await self._db.commit()
        return cursor.rowcount

    async def cleanup(self, older_than_days: int = 30) -> int:
        """Remove checkpoints older than N days. Returns count deleted."""
        await self._ensure_db()
        cutoff = time.time() - (older_than_days * 86400)
        cursor = await self._db.execute(
            "DELETE FROM checkpoints WHERE created_at < ?", (cutoff,)
        )
        await self._db.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info("Cleaned up %d old checkpoints (>%d days)", count, older_than_days)
        return count
