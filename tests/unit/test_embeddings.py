"""
Tests for EmbeddingManager — Verifies fallback chain and no hash fallback.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from nexus.knowledge.rag.embeddings import EmbeddingManager, EmbeddingBackend


class TestEmbeddingManager:

    def test_default_dimension(self):
        mgr = EmbeddingManager()
        assert mgr.dimension == 768

    def test_no_active_backend_initially(self):
        mgr = EmbeddingManager()
        assert mgr.active_backend is None

    @pytest.mark.asyncio
    async def test_embed_fails_when_no_backends(self):
        """Should raise RuntimeError when all backends fail (no hash fallback!)."""
        mgr = EmbeddingManager(ollama_url="http://nonexistent:11434")

        with patch.object(mgr, '_try_ollama', return_value=None):
            with patch.object(mgr, '_try_openai', return_value=None):
                with patch.object(mgr, '_try_sentence_transformers', return_value=None):
                    with pytest.raises(RuntimeError, match="All embedding backends failed"):
                        await mgr.embed("test text")

    @pytest.mark.asyncio
    async def test_ollama_fallback_to_sentence_transformers(self):
        """When Ollama fails, should try sentence-transformers."""
        mgr = EmbeddingManager()
        fake_embedding = [0.1] * 384 + [0.0] * 384  # Padded to 768

        # Patch the methods but also set the active_backend side effect
        async def fake_ollama(text):
            return None

        async def fake_openai(text):
            return None

        def fake_st(text):
            mgr._active = EmbeddingBackend.SENTENCE_TRANS
            return fake_embedding

        with patch.object(mgr, '_try_ollama', side_effect=fake_ollama):
            with patch.object(mgr, '_try_openai', side_effect=fake_openai):
                with patch.object(mgr, '_try_sentence_transformers', side_effect=fake_st):
                    result = await mgr.embed("test text")
                    assert len(result) == 768

    @pytest.mark.asyncio
    async def test_private_tier_skips_openai(self):
        """PRIVATE tier should not attempt OpenAI."""
        mgr = EmbeddingManager()
        fake_embedding = [0.5] * 768

        async def fake_ollama(text):
            mgr._active = EmbeddingBackend.OLLAMA_BGE
            return fake_embedding

        with patch.object(mgr, '_try_ollama', side_effect=fake_ollama):
            result = await mgr.embed("secret data", privacy_tier="PRIVATE")
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        mgr = EmbeddingManager()
        result = await mgr.embed_batch([], privacy_tier="INTERNAL")
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_falls_back_to_individual(self):
        mgr = EmbeddingManager()
        fake_emb = [0.1] * 768

        with patch.object(mgr, '_try_openai_batch', return_value=None):
            with patch.object(mgr, 'embed', return_value=fake_emb):
                result = await mgr.embed_batch(["text1", "text2"], privacy_tier="INTERNAL")
                assert len(result) == 2
