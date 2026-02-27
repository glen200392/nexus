"""
NEXUS EmbeddingManager — Unified Embedding Provider (v2)
Replaces v1 EmbeddingProvider with proper fallback chain and NO hash fallback.

Fallback chain:
  1. Ollama bge-m3 (local, private)
  2. OpenAI text-embedding-3-small (cloud)
  3. sentence-transformers all-MiniLM-L6-v2 (local, lightweight)

The SHA256 hash fallback from v1 is REMOVED — it produces fake embeddings
that poison vector search results with random noise.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger("nexus.embeddings")


class EmbeddingBackend(str, Enum):
    OLLAMA_BGE = "ollama_bge"
    OPENAI = "openai"
    SENTENCE_TRANS = "sentence_transformers"


class EmbeddingManager:
    """
    Unified embedding provider with graceful fallback.

    Usage:
        mgr = EmbeddingManager()
        embedding = await mgr.embed("Hello world", privacy_tier="INTERNAL")
        batch = await mgr.embed_batch(["text1", "text2"], privacy_tier="INTERNAL")
    """

    # Standard dimension for bge-m3 and text-embedding-3-small (truncated)
    DEFAULT_DIMENSION = 768

    def __init__(
        self,
        preferred_backend: Optional[EmbeddingBackend] = None,
        ollama_url: Optional[str] = None,
    ):
        self._preferred = preferred_backend
        self._ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._active: Optional[EmbeddingBackend] = None
        self._sentence_model = None  # Lazy-loaded
        self._dimension: int = self.DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def active_backend(self) -> Optional[EmbeddingBackend]:
        return self._active

    async def embed(self, text: str, privacy_tier: str = "INTERNAL") -> list[float]:
        """
        Embed a single text string.
        Respects privacy: PRIVATE tier forces local-only backends.
        """
        is_private = privacy_tier == "PRIVATE"

        # Try backends in priority order
        if not is_private or True:  # Ollama is always local
            result = await self._try_ollama(text)
            if result is not None:
                return result

        if not is_private:
            result = await self._try_openai(text)
            if result is not None:
                return result

        # Final fallback: sentence-transformers (always local)
        result = self._try_sentence_transformers(text)
        if result is not None:
            return result

        raise RuntimeError(
            "All embedding backends failed. Ensure at least one is available: "
            "Ollama with bge-m3, OpenAI API key, or sentence-transformers installed."
        )

    async def embed_batch(self, texts: list[str], privacy_tier: str = "INTERNAL") -> list[list[float]]:
        """Embed multiple texts. Uses batch API where available."""
        if not texts:
            return []

        is_private = privacy_tier == "PRIVATE"

        # Try OpenAI batch first (most efficient for batches)
        if not is_private:
            result = await self._try_openai_batch(texts)
            if result is not None:
                return result

        # Fall back to individual embedding
        results = []
        for text in texts:
            emb = await self.embed(text, privacy_tier)
            results.append(emb)
        return results

    # ── Backend Implementations ────────────────────────────────────────────────

    async def _try_ollama(self, text: str) -> list[float] | None:
        """Embed via Ollama bge-m3 (local)."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/embeddings",
                    json={"model": "bge-m3", "prompt": text},
                )
                resp.raise_for_status()
                embedding = resp.json()["embedding"]
                self._active = EmbeddingBackend.OLLAMA_BGE
                self._dimension = len(embedding)
                return embedding
        except Exception as exc:
            logger.debug("Ollama bge-m3 embedding failed: %s", exc)
            return None

    async def _try_openai(self, text: str) -> list[float] | None:
        """Embed via OpenAI text-embedding-3-small."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            resp = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
                dimensions=self.DEFAULT_DIMENSION,
            )
            self._active = EmbeddingBackend.OPENAI
            self._dimension = len(resp.data[0].embedding)
            return resp.data[0].embedding
        except Exception as exc:
            logger.debug("OpenAI embedding failed: %s", exc)
            return None

    async def _try_openai_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Batch embed via OpenAI."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            resp = await client.embeddings.create(
                model="text-embedding-3-small",
                input=[t[:8000] for t in texts],
                dimensions=self.DEFAULT_DIMENSION,
            )
            self._active = EmbeddingBackend.OPENAI
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        except Exception as exc:
            logger.debug("OpenAI batch embedding failed: %s", exc)
            return None

    def _try_sentence_transformers(self, text: str) -> list[float] | None:
        """Embed via sentence-transformers all-MiniLM-L6-v2 (local)."""
        try:
            if self._sentence_model is None:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded sentence-transformers all-MiniLM-L6-v2")

            embedding = self._sentence_model.encode(text).tolist()
            self._active = EmbeddingBackend.SENTENCE_TRANS
            # MiniLM produces 384-dim; pad to 768 for consistency
            if len(embedding) < self.DEFAULT_DIMENSION:
                embedding = embedding + [0.0] * (self.DEFAULT_DIMENSION - len(embedding))
            self._dimension = len(embedding)
            return embedding
        except Exception as exc:
            logger.debug("sentence-transformers embedding failed: %s", exc)
            return None
