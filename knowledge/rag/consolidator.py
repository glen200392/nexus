"""
NEXUS MemoryConsolidator — Memory Maintenance & Optimization
Performs periodic consolidation of the RAG store:
  - Deduplication: merge near-duplicate documents
  - Stale decay: remove unused old documents
  - Promotion: promote frequently-accessed episodic memories to semantic
  - Statistics and reporting
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("nexus.rag.consolidator")


@dataclass
class ConsolidationReport:
    """Report of consolidation operations performed."""
    deduplicated: int = 0
    summarized: int = 0
    decayed: int = 0
    promoted: int = 0
    extracted: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class MemoryConsolidator:
    """
    Performs memory consolidation on a RAGEngineV2 store.

    Operations:
      - Deduplicate near-identical documents (cosine similarity > threshold)
      - Decay stale documents (no access, older than stale_days)
      - Promote frequently-accessed episodic memories to semantic collection
    """

    def __init__(
        self,
        rag_engine: Any = None,
        similarity_threshold: float = 0.95,
        stale_days: int = 30,
        promotion_threshold: int = 5,
    ):
        """
        Args:
            rag_engine: A RAGEngineV2 instance (or compatible).
            similarity_threshold: Cosine similarity above which docs are considered duplicates.
            stale_days: Days after which unused documents are candidates for decay.
            promotion_threshold: Access count above which episodic memories are promoted.
        """
        self._engine = rag_engine
        self._similarity_threshold = similarity_threshold
        self._stale_days = stale_days
        self._promotion_threshold = promotion_threshold

    async def run(self) -> ConsolidationReport:
        """
        Run all consolidation steps and return a report.

        Returns:
            ConsolidationReport with counts of each operation.
        """
        start = time.time()
        report = ConsolidationReport()

        if self._engine is None:
            report.errors.append("No RAG engine configured")
            report.duration_ms = (time.time() - start) * 1000
            return report

        try:
            report.deduplicated = await self.deduplicate()
        except Exception as exc:
            report.errors.append(f"Deduplication failed: {exc}")
            logger.exception("Deduplication failed: %s", exc)

        try:
            report.decayed = await self.decay_stale()
        except Exception as exc:
            report.errors.append(f"Stale decay failed: {exc}")
            logger.exception("Stale decay failed: %s", exc)

        try:
            report.promoted = await self.promote_frequent()
        except Exception as exc:
            report.errors.append(f"Promotion failed: {exc}")
            logger.exception("Promotion failed: %s", exc)

        report.duration_ms = (time.time() - start) * 1000
        logger.info(
            "Consolidation complete: dedup=%d, decayed=%d, promoted=%d (%.0fms)",
            report.deduplicated, report.decayed, report.promoted, report.duration_ms,
        )
        return report

    async def deduplicate(self) -> int:
        """
        Find documents with cosine similarity > threshold and merge them.

        Keeps the newer document and deletes the older one.

        Returns:
            Number of documents removed.
        """
        if self._engine is None:
            return 0

        db = await self._engine._ensure_db()
        cursor = await db.execute(
            "SELECT doc_id, embedding, created_at FROM documents WHERE embedding != '[]' ORDER BY created_at"
        )
        rows = await cursor.fetchall()

        if len(rows) < 2:
            return 0

        # Load embeddings
        docs: list[tuple[str, list[float], float]] = []
        for row in rows:
            try:
                embedding = json.loads(row[1])
                if embedding:
                    docs.append((row[0], embedding, row[2]))
            except (json.JSONDecodeError, ValueError):
                continue

        # Find duplicates (O(n^2) — acceptable for consolidation batches)
        to_delete: set[str] = set()
        removed = 0

        for i in range(len(docs)):
            if docs[i][0] in to_delete:
                continue
            for j in range(i + 1, len(docs)):
                if docs[j][0] in to_delete:
                    continue
                sim = _cosine_similarity(docs[i][1], docs[j][1])
                if sim >= self._similarity_threshold:
                    # Delete the older document
                    older_id = docs[i][0] if docs[i][2] <= docs[j][2] else docs[j][0]
                    to_delete.add(older_id)

        for doc_id in to_delete:
            await self._engine.delete(doc_id)
            removed += 1

        if removed:
            logger.info("Deduplicated %d documents", removed)
        return removed

    async def decay_stale(self) -> int:
        """
        Remove documents with access_count=0 older than stale_days.

        Returns:
            Number of documents removed.
        """
        if self._engine is None:
            return 0

        db = await self._engine._ensure_db()
        cutoff = time.time() - (self._stale_days * 86400)

        cursor = await db.execute(
            "SELECT doc_id FROM documents WHERE access_count = 0 AND created_at < ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()

        removed = 0
        for row in rows:
            await self._engine.delete(row[0])
            removed += 1

        if removed:
            logger.info("Decayed %d stale documents", removed)
        return removed

    async def promote_frequent(self) -> int:
        """
        Promote episodic memories with high access count to the semantic collection.

        Returns:
            Number of documents promoted.
        """
        if self._engine is None:
            return 0

        db = await self._engine._ensure_db()

        cursor = await db.execute(
            """SELECT doc_id FROM documents
               WHERE collection = 'episodic'
               AND access_count >= ?""",
            (self._promotion_threshold,),
        )
        rows = await cursor.fetchall()

        promoted = 0
        for row in rows:
            await db.execute(
                "UPDATE documents SET collection = 'semantic' WHERE doc_id = ?",
                (row[0],),
            )
            promoted += 1

        if promoted:
            await db.commit()
            logger.info("Promoted %d episodic memories to semantic", promoted)
        return promoted

    async def get_stats(self) -> dict:
        """Return consolidation statistics."""
        if self._engine is None:
            return {"error": "No RAG engine configured"}

        db = await self._engine._ensure_db()

        cursor = await db.execute("SELECT COUNT(*) FROM documents")
        total = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT COUNT(*) FROM documents WHERE access_count = 0"
        )
        zero_access = (await cursor.fetchone())[0]

        cutoff = time.time() - (self._stale_days * 86400)
        cursor = await db.execute(
            "SELECT COUNT(*) FROM documents WHERE access_count = 0 AND created_at < ?",
            (cutoff,),
        )
        stale_candidates = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT COUNT(*) FROM documents WHERE collection = 'episodic' AND access_count >= ?",
            (self._promotion_threshold,),
        )
        promotion_candidates = (await cursor.fetchone())[0]

        return {
            "total_documents": total,
            "zero_access_documents": zero_access,
            "stale_candidates": stale_candidates,
            "promotion_candidates": promotion_candidates,
        }


# ── Utility ──────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
