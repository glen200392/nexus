"""
NEXUS RAG Engine — Layer 5 Memory Implementation
Manages all five memory types with hybrid retrieval.

Memory hierarchy:
  Working   → Python objects in process (no persistence)
  Short-term → Redis with TTL (session context)
  Episodic  → ChromaDB (task history, searchable)
  Semantic  → ChromaDB (knowledge base, documents)
  Procedural → File system (skills, prompts)
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional

from nexus.knowledge.rag.schema import (
    COLLECTION_CONFIGS, CHUNKING_STRATEGIES, MemoryRecord, EpisodicRecord,
    MemoryType, RetrievalConfig, RetrievalMode, PrivacyTier,
)

logger = logging.getLogger("nexus.rag")

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ─── Embedding Provider ───────────────────────────────────────────────────────

class EmbeddingProvider:
    """
    Wraps local (bge-m3 via Ollama) and cloud (OpenAI) embedding models.
    Selects based on privacy tier.
    """

    def __init__(self):
        self._local_client  = None  # lazy init
        self._openai_client = None

    async def embed(self, text: str, privacy_tier: PrivacyTier) -> list[float]:
        if privacy_tier == PrivacyTier.PRIVATE:
            return await self._embed_local(text)
        try:
            return await self._embed_openai(text)
        except Exception:
            logger.warning("OpenAI embedding failed; falling back to local")
            return await self._embed_local(text)

    async def _embed_local(self, text: str) -> list[float]:
        """Use bge-m3 via Ollama (local, private)."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "bge-m3", "prompt": text},
                )
                return resp.json()["embedding"]
        except Exception as exc:
            logger.warning("Local embedding failed: %s; using fallback hash", exc)
            # Deterministic pseudo-embedding for fallback (not for production)
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h] * 3  # 768 floats

    async def _embed_openai(self, text: str) -> list[float]:
        """Use text-embedding-3-small via OpenAI."""
        import openai
        client = openai.AsyncOpenAI()
        resp = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],  # OpenAI token limit
        )
        return resp.data[0].embedding


# ─── Vector Store Adapter ─────────────────────────────────────────────────────

class VectorStoreAdapter:
    """
    Wraps ChromaDB for persistent vector storage.
    One collection per memory type (see COLLECTION_CONFIGS).
    """

    def __init__(self, persist_dir: Path = DATA_DIR / "vector_store"):
        self._dir = persist_dir
        self._client = None
        self._collections: dict = {}

    def _ensure_client(self):
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=str(self._dir))
                logger.info("ChromaDB initialized at %s", self._dir)
            except ImportError:
                logger.error("chromadb not installed. Run: pip install chromadb")
                raise

    def _get_collection(self, name: str):
        self._ensure_client()
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def upsert(
        self,
        collection: str,
        record: MemoryRecord,
        embedding: list[float],
    ) -> str:
        col = self._get_collection(collection)
        meta = record.to_metadata_dict()
        # ChromaDB metadata must be flat scalar types
        meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        col.upsert(
            ids=[record.doc_id],
            embeddings=[embedding],
            documents=[record.content],
            metadatas=[meta],
        )
        return record.doc_id

    async def query(
        self,
        collection: str,
        embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        col = self._get_collection(collection)
        kwargs: dict = {"query_embeddings": [embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            {
                "doc_id":    results["ids"][0][i],
                "content":   results["documents"][0][i],
                "metadata":  results["metadatas"][0][i],
                "distance":  results["distances"][0][i],
                "score":     1.0 - results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]


# ─── RAG Engine ───────────────────────────────────────────────────────────────

class RAGEngine:
    """
    High-level memory interface used by all agents.
    Handles: upsert, hybrid retrieval, chunking, time decay.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreAdapter] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self.vector  = vector_store or VectorStoreAdapter()
        self.embedder = embedder or EmbeddingProvider()

    # ── Write ──────────────────────────────────────────────────────────────────

    async def upsert(self, record: MemoryRecord) -> str:
        """Store a memory record with its embedding."""
        collection = self._route_collection(record)
        embedding  = await self.embedder.embed(record.content, record.privacy_tier)
        doc_id     = await self.vector.upsert(collection, record, embedding)
        logger.debug("Memory stored: %s in %s", doc_id[:8], collection)
        return doc_id

    async def ingest_document(
        self,
        content: str,
        source: str,
        domain: str = "general",
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
        doc_type: str = "document",
    ) -> list[str]:
        """
        Chunk a long document and store each chunk.
        Returns list of doc_ids.
        """
        from nexus.knowledge.rag.schema import DocumentType
        chunks    = self._chunk(content, doc_type)
        doc_ids   = []
        parent_id = None

        for i, chunk in enumerate(chunks):
            record = MemoryRecord(
                content=chunk,
                source=source,
                domain=domain,
                privacy_tier=privacy_tier,
                doc_type=DocumentType(doc_type),
                chunk_index=i,
                total_chunks=len(chunks),
                parent_id=parent_id,
            )
            if i == 0:
                parent_id = record.doc_id
            doc_id = await self.upsert(record)
            doc_ids.append(doc_id)

        logger.info(
            "Ingested %d chunks from '%s'", len(doc_ids), source
        )
        return doc_ids

    # ── Read ───────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
    ) -> list[MemoryRecord]:
        """
        Hybrid retrieval: semantic vector search + optional keyword filter.
        Returns ranked MemoryRecords.
        """
        from nexus.knowledge.rag.schema import RetrievalConfig, RetrievalMode
        cfg = config or RetrievalConfig()

        # Embed query
        q_embedding = await self.embedder.embed(query, privacy_tier)

        # Build privacy filter
        where: dict = {}
        if cfg.filters:
            where.update(cfg.filters)

        # Query knowledge_base collection
        raw = await self.vector.query(
            collection="knowledge_base",
            embedding=q_embedding,
            n_results=cfg.top_k * 2,   # Over-fetch for re-ranking
            where=where if where else None,
        )

        # Filter by score threshold
        filtered = [r for r in raw if r["score"] >= cfg.score_threshold]

        # Apply temporal decay
        if cfg.apply_temporal_decay:
            filtered = self._apply_decay(filtered, cfg.temporal_decay_rate)

        # Re-rank and return top_k
        filtered.sort(key=lambda r: r["score"], reverse=True)
        top = filtered[:cfg.top_k]

        # Convert to MemoryRecord objects
        records = []
        for item in top:
            meta = item["metadata"]
            r = MemoryRecord(
                doc_id=item["doc_id"],
                content=item["content"],
                quality_score=float(meta.get("quality_score", 0.5)),
                domain=meta.get("domain", "general"),
                source=meta.get("source", ""),
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            )
            records.append(r)

        logger.debug(
            "Retrieved %d/%d records for query '%s…'",
            len(records), len(raw), query[:40],
        )
        return records

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _route_collection(self, record: MemoryRecord) -> str:
        """Map memory type to collection name."""
        mapping = {
            MemoryType.SEMANTIC:  "knowledge_base",
            MemoryType.EPISODIC:  "episodic_memory",
            MemoryType.PROCEDURAL: "agent_personas",
            MemoryType.SHORT_TERM: "conversation_history",
            MemoryType.WORKING:   "knowledge_base",  # working mem → semantic store
        }
        return mapping.get(record.memory_type, "knowledge_base")

    def _chunk(self, text: str, doc_type: str = "document") -> list[str]:
        """Split text into chunks based on doc_type strategy."""
        strategy = CHUNKING_STRATEGIES.get(doc_type, CHUNKING_STRATEGIES["document"])
        size     = strategy["chunk_size"]
        overlap  = strategy["chunk_overlap"]

        if strategy["strategy"] == "none" or len(text) <= size:
            return [text]

        chunks = []
        start  = 0
        while start < len(text):
            end = start + size
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ["。", ".\n", ". ", "\n\n", "\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start + size // 2:
                        end = pos + len(sep)
                        break
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c.strip()]

    def _apply_decay(self, results: list[dict], decay_rate: float) -> list[dict]:
        """Multiply score by time-based decay factor."""
        now = time.time()
        for r in results:
            created = float(r["metadata"].get("created_at", now))
            days_old = (now - created) / 86400
            decay = max(0.1, 1.0 - decay_rate * days_old)
            r["score"] *= decay
        return results


# ─── Global singleton ────────────────────────────────────────────────────────
_engine_instance: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
