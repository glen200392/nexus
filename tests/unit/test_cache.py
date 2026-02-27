"""
Tests for nexus.core.llm.cache.LLMSemanticCache.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nexus.core.llm.cache import LLMSemanticCache, CacheEntry


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_cache.db"


@pytest.fixture
def cache(tmp_db):
    """Create a cache instance with a temporary DB."""
    return LLMSemanticCache(db_path=tmp_db)


class TestCacheEntry:

    def test_is_expired_false(self):
        import time
        entry = CacheEntry(
            query_hash="abc",
            query_text="hello",
            response_json="{}",
            model="test",
            created_at=time.time(),
            ttl=3600,
        )
        assert entry.is_expired is False

    def test_is_expired_true(self):
        entry = CacheEntry(
            query_hash="abc",
            query_text="hello",
            response_json="{}",
            model="test",
            created_at=0,  # epoch
            ttl=1,
        )
        assert entry.is_expired is True


class TestLLMSemanticCache:

    @pytest.mark.asyncio
    async def test_put_and_get_exact(self, cache):
        await cache.put("hello world", {"content": "Hi!"}, "gpt-4o")
        result = await cache.get("hello world", "gpt-4o")
        assert result is not None
        assert result["content"] == "Hi!"

    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        result = await cache.get("nonexistent query", "gpt-4o")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_wrong_model(self, cache):
        await cache.put("hello", {"content": "Hi!"}, "gpt-4o")
        result = await cache.get("hello", "claude-sonnet-4-6")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        await cache.put("query1", {"content": "resp1"}, "m1")
        await cache.invalidate("query1")
        result = await cache.get("query1", "m1")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        await cache.put("q1", {"c": "1"}, "m1")
        await cache.put("q2", {"c": "2"}, "m1")
        await cache.clear()
        assert await cache.get("q1", "m1") is None
        assert await cache.get("q2", "m1") is None

    @pytest.mark.asyncio
    async def test_stats_initial(self, cache):
        stats = cache.stats()
        assert stats["total_entries"] == 0
        assert stats["total_hits"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_after_hits(self, cache):
        await cache.put("q", {"c": "r"}, "m")
        await cache.get("q", "m")       # hit
        await cache.get("miss", "m")     # miss
        stats = cache.stats()
        assert stats["total_hits"] == 1
        assert stats["total_lookups"] == 2
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self, cache):
        await cache.put("q", {"c": "r"}, "m", ttl=0)
        # TTL=0 means it expires immediately
        import asyncio
        await asyncio.sleep(0.01)
        result = await cache.get("q", "m")
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_match(self, tmp_db):
        """Test semantic matching with a mock embedding manager."""
        emb_mgr = AsyncMock()
        # Return similar embeddings for similar queries
        emb_mgr.embed = AsyncMock(side_effect=[
            [1.0, 0.0, 0.0],   # put: original
            [0.99, 0.1, 0.0],  # get: similar query → should match
        ])
        cache = LLMSemanticCache(db_path=tmp_db, embedding_manager=emb_mgr)
        await cache.put("What is Python?", {"content": "A programming language"}, "m1")
        result = await cache.get("Tell me about Python", "m1")
        # Cosine similarity of [1,0,0] and [0.99,0.1,0] ≈ 0.995 > 0.92
        assert result is not None
        assert result["content"] == "A programming language"

    @pytest.mark.asyncio
    async def test_semantic_no_match_below_threshold(self, tmp_db):
        """Test that dissimilar embeddings do not match."""
        emb_mgr = AsyncMock()
        emb_mgr.embed = AsyncMock(side_effect=[
            [1.0, 0.0, 0.0],   # put
            [0.0, 1.0, 0.0],   # get: orthogonal → no match
        ])
        cache = LLMSemanticCache(db_path=tmp_db, embedding_manager=emb_mgr)
        await cache.put("What is Python?", {"content": "A lang"}, "m1")
        result = await cache.get("Unrelated question", "m1")
        assert result is None

    def test_cosine_similarity(self):
        sim = LLMSemanticCache._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-6

        sim = LLMSemanticCache._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 1e-6

        sim = LLMSemanticCache._cosine_similarity([], [])
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_overwrite_existing(self, cache):
        await cache.put("q", {"v": 1}, "m")
        await cache.put("q", {"v": 2}, "m")
        result = await cache.get("q", "m")
        assert result["v"] == 2
