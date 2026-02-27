"""
Tests for MemoryConsolidator — deduplication, decay, promotion.
"""
from __future__ import annotations

import pytest

from nexus.knowledge.rag.consolidator import ConsolidationReport, MemoryConsolidator
from nexus.knowledge.rag.engine_v2 import RAGEngineV2


@pytest.fixture
async def rag_engine(tmp_path):
    """Create a RAGEngineV2 with a temporary database."""
    db_path = tmp_path / "test_consolidator.db"
    engine = RAGEngineV2(db_path=db_path, similarity_threshold=0.0)
    yield engine
    await engine.close()


class TestMemoryConsolidator:

    @pytest.mark.asyncio
    async def test_run_empty(self, rag_engine):
        """Consolidating an empty store should return a clean report."""
        consolidator = MemoryConsolidator(rag_engine=rag_engine)
        report = await consolidator.run()

        assert isinstance(report, ConsolidationReport)
        assert report.deduplicated == 0
        assert report.decayed == 0
        assert report.promoted == 0
        assert report.errors == []
        assert report.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_no_engine(self):
        """Consolidating without an engine should report an error."""
        consolidator = MemoryConsolidator(rag_engine=None)
        report = await consolidator.run()

        assert len(report.errors) > 0
        assert "No RAG engine" in report.errors[0]

    @pytest.mark.asyncio
    async def test_consolidation_report(self):
        """ConsolidationReport should have correct default values."""
        report = ConsolidationReport()
        assert report.deduplicated == 0
        assert report.summarized == 0
        assert report.decayed == 0
        assert report.promoted == 0
        assert report.extracted == 0
        assert report.errors == []
        assert report.duration_ms == 0.0

    @pytest.mark.asyncio
    async def test_stats(self, rag_engine):
        """Stats should reflect document store state."""
        await rag_engine.add(text="Document one", collection="episodic")
        await rag_engine.add(text="Document two", collection="semantic")

        consolidator = MemoryConsolidator(rag_engine=rag_engine)
        stats = await consolidator.get_stats()

        assert stats["total_documents"] == 2
        assert stats["zero_access_documents"] == 2
        assert stats["stale_candidates"] == 0  # Just created, not stale yet
        assert stats["promotion_candidates"] == 0  # access_count=0

    @pytest.mark.asyncio
    async def test_decay_stale(self, rag_engine):
        """Stale documents (old + zero access) should be decayed."""
        import time

        # Add a document and manually backdate it
        doc_id = await rag_engine.add(text="Old stale document")
        db = await rag_engine._ensure_db()
        old_time = time.time() - 60 * 86400  # 60 days ago
        await db.execute(
            "UPDATE documents SET created_at = ? WHERE doc_id = ?",
            (old_time, doc_id),
        )
        await db.commit()

        consolidator = MemoryConsolidator(rag_engine=rag_engine, stale_days=30)
        decayed = await consolidator.decay_stale()

        assert decayed == 1

    @pytest.mark.asyncio
    async def test_promote_frequent(self, rag_engine):
        """Frequently accessed episodic memories should be promoted."""
        doc_id = await rag_engine.add(text="Important memory", collection="episodic")

        # Simulate high access count
        db = await rag_engine._ensure_db()
        await db.execute(
            "UPDATE documents SET access_count = 10 WHERE doc_id = ?",
            (doc_id,),
        )
        await db.commit()

        consolidator = MemoryConsolidator(rag_engine=rag_engine, promotion_threshold=5)
        promoted = await consolidator.promote_frequent()

        assert promoted == 1

        # Verify collection was updated
        cursor = await db.execute(
            "SELECT collection FROM documents WHERE doc_id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        assert row[0] == "semantic"
