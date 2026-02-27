"""
Tests for RAGEngineV2 — hybrid retrieval with BM25 + semantic + recency.
"""
from __future__ import annotations

import math
import time

import pytest

from nexus.knowledge.rag.engine_v2 import RAGEngineV2


@pytest.fixture
async def rag_engine(tmp_path):
    """Create a RAGEngineV2 with a temporary database."""
    db_path = tmp_path / "test_rag.db"
    engine = RAGEngineV2(db_path=db_path, similarity_threshold=0.0)
    yield engine
    await engine.close()


class TestRAGEngineV2:

    @pytest.mark.asyncio
    async def test_add_and_query(self, rag_engine):
        """Add a document and query it back via BM25."""
        doc_id = await rag_engine.add(
            text="Machine learning is a subset of artificial intelligence",
            metadata={"source": "test"},
            collection="default",
        )
        assert doc_id is not None

        results = await rag_engine.query("machine learning artificial intelligence")
        assert len(results) > 0
        assert results[0]["doc_id"] == doc_id
        assert "machine learning" in results[0]["text"].lower()

    @pytest.mark.asyncio
    async def test_query_no_results(self, rag_engine):
        """Query on empty engine should return empty list."""
        results = await rag_engine.query("nonexistent topic")
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_collections(self, rag_engine):
        """Documents in different collections should be retrievable separately."""
        await rag_engine.add(text="Python programming", collection="code")
        await rag_engine.add(text="Quantum physics theory", collection="science")

        code_results = await rag_engine.query("Python", collection="code")
        assert len(code_results) > 0
        assert all(r["collection"] == "code" for r in code_results)

        science_results = await rag_engine.query("quantum", collection="science")
        assert len(science_results) > 0
        assert all(r["collection"] == "science" for r in science_results)

    @pytest.mark.asyncio
    async def test_delete_document(self, rag_engine):
        """Deleting a document should remove it from results."""
        doc_id = await rag_engine.add(text="Temporary document for deletion test")

        deleted = await rag_engine.delete(doc_id)
        assert deleted is True

        results = await rag_engine.query("temporary document deletion")
        assert all(r["doc_id"] != doc_id for r in results)

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, rag_engine):
        """Deleting a nonexistent document should return False."""
        deleted = await rag_engine.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_stats(self, rag_engine):
        """Stats should reflect the number of documents."""
        await rag_engine.add(text="Doc one", collection="a")
        await rag_engine.add(text="Doc two", collection="a")
        await rag_engine.add(text="Doc three", collection="b")

        stats = await rag_engine.get_stats()
        assert stats["total_documents"] == 3
        assert stats["collection_count"] == 2
        assert stats["per_collection"]["a"] == 2
        assert stats["per_collection"]["b"] == 1

    @pytest.mark.asyncio
    async def test_get_collections(self, rag_engine):
        """get_collections should return all unique collection names."""
        await rag_engine.add(text="Doc in alpha", collection="alpha")
        await rag_engine.add(text="Doc in beta", collection="beta")

        collections = await rag_engine.get_collections()
        assert "alpha" in collections
        assert "beta" in collections

    def test_recency_score(self):
        """Recency score should follow exponential decay."""
        engine = RAGEngineV2()

        # Just created (score should be near 1.0)
        score_now = engine._recency_score(time.time())
        assert score_now > 0.99

        # 10 days old: exp(-0.1 * 10) = exp(-1) ~ 0.368
        ten_days_ago = time.time() - 10 * 86400
        score_10d = engine._recency_score(ten_days_ago)
        assert abs(score_10d - math.exp(-1)) < 0.01

        # 30 days old: exp(-0.1 * 30) = exp(-3) ~ 0.05
        thirty_days_ago = time.time() - 30 * 86400
        score_30d = engine._recency_score(thirty_days_ago)
        assert abs(score_30d - math.exp(-3)) < 0.01

        # Scores should decrease with age
        assert score_now > score_10d > score_30d
