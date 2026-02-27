"""
NEXUS RAGEngineV2 — Hybrid Retrieval with BM25 + Semantic + Recency
SQLite-backed document store with aiosqlite for async operations.

Improvements over v1:
  - Hybrid retrieval: semantic (vector), BM25 (keyword), and recency scoring
  - Weighted rank fusion with configurable weights
  - Privacy-tier-aware collections
  - Async SQLite backend (no ChromaDB dependency)
  - Per-collection document management
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger("nexus.rag.engine_v2")

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "rag_v2.db"
_DEFAULT_WEIGHTS = {"semantic": 0.6, "bm25": 0.3, "recency": 0.1}


class RAGEngineV2:
    """
    V2 RAG engine with hybrid retrieval (semantic + BM25 + recency).
    Uses aiosqlite for persistent storage and an optional EmbeddingManager
    for vector embeddings.
    """

    def __init__(
        self,
        embedding_manager: Any = None,
        db_path: str | Path | None = None,
        similarity_threshold: float = 0.7,
    ):
        self._db_path = str(db_path or _DEFAULT_DB)
        self._similarity_threshold = similarity_threshold
        self._embedding_manager = embedding_manager
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def _ensure_db(self) -> aiosqlite.Connection:
        """Lazily initialize the database connection and schema."""
        if self._db is None or not self._initialized:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id       TEXT PRIMARY KEY,
                    text         TEXT NOT NULL,
                    embedding    TEXT,
                    metadata     TEXT DEFAULT '{}',
                    collection   TEXT DEFAULT 'default',
                    privacy_tier TEXT DEFAULT 'public',
                    created_at   REAL NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection ON documents(collection)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_privacy ON documents(privacy_tier)"
            )
            await self._db.commit()
            self._initialized = True
        return self._db

    # ── Write Operations ─────────────────────────────────────────────────────

    async def add(
        self,
        text: str,
        metadata: dict | None = None,
        collection: str = "default",
        privacy_tier: str = "public",
    ) -> str:
        """
        Add a document to the store.

        Args:
            text: Document text content.
            metadata: Optional metadata dict.
            collection: Collection name for grouping.
            privacy_tier: Privacy classification (public, internal, private).

        Returns:
            The generated document ID.
        """
        db = await self._ensure_db()
        doc_id = str(uuid.uuid4())

        # Generate embedding if manager is available
        embedding_json = "[]"
        if self._embedding_manager is not None:
            try:
                embedding = await self._embedding_manager.embed(text, privacy_tier=privacy_tier)
                embedding_json = json.dumps(embedding)
            except Exception as exc:
                logger.warning("Embedding generation failed: %s", exc)

        metadata_json = json.dumps(metadata or {})
        created_at = time.time()

        await db.execute(
            """INSERT INTO documents (doc_id, text, embedding, metadata, collection, privacy_tier, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, text, embedding_json, metadata_json, collection, privacy_tier, created_at),
        )
        await db.commit()

        logger.debug("Added document %s to collection '%s'", doc_id[:8], collection)
        return doc_id

    async def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Returns:
            True if a document was deleted, False if not found.
        """
        db = await self._ensure_db()
        cursor = await db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        await db.commit()
        return cursor.rowcount > 0

    # ── Query Operations ─────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
        privacy_tier: str = "public",
        weights: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval combining semantic search, BM25, and recency.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            collection: Filter by collection (None = all).
            privacy_tier: Privacy tier for embedding generation.
            weights: Custom weights dict with keys "semantic", "bm25", "recency".

        Returns:
            List of result dicts with: doc_id, text, score, metadata, collection.
        """
        db = await self._ensure_db()
        w = weights or dict(_DEFAULT_WEIGHTS)

        # Generate query embedding if manager is available
        query_embedding: list[float] | None = None
        if self._embedding_manager is not None:
            try:
                query_embedding = await self._embedding_manager.embed(
                    query, privacy_tier=privacy_tier
                )
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)

        # Run retrieval strategies
        semantic_scores = await self._semantic_search(
            query_embedding, top_k * 3, collection, privacy_tier
        ) if query_embedding else []

        bm25_scores = await self._bm25_search(query, top_k * 3, collection, privacy_tier)

        # Gather all candidate doc_ids
        all_ids = set()
        for doc_id, _ in semantic_scores:
            all_ids.add(doc_id)
        for doc_id, _ in bm25_scores:
            all_ids.add(doc_id)

        if not all_ids:
            return []

        # Get recency scores
        recency_scores: dict[str, float] = {}
        for doc_id in all_ids:
            cursor = await db.execute(
                "SELECT created_at FROM documents WHERE doc_id = ?", (doc_id,)
            )
            row = await cursor.fetchone()
            if row:
                recency_scores[doc_id] = self._recency_score(row[0])

        # Merge results
        results = self._merge_results(
            dict(semantic_scores),
            dict(bm25_scores),
            recency_scores,
            w,
        )

        # Fetch full documents for top results
        results.sort(key=lambda r: r["score"], reverse=True)
        top_results = results[:top_k]

        final: list[dict] = []
        for r in top_results:
            cursor = await db.execute(
                "SELECT text, metadata, collection FROM documents WHERE doc_id = ?",
                (r["doc_id"],),
            )
            row = await cursor.fetchone()
            if row:
                # Increment access count
                await db.execute(
                    "UPDATE documents SET access_count = access_count + 1 WHERE doc_id = ?",
                    (r["doc_id"],),
                )
                final.append({
                    "doc_id": r["doc_id"],
                    "text": row[0],
                    "score": r["score"],
                    "metadata": json.loads(row[1]) if row[1] else {},
                    "collection": row[2],
                })
        await db.commit()

        return final

    # ── Collection Management ────────────────────────────────────────────────

    async def get_collections(self) -> list[str]:
        """Return list of all collection names."""
        db = await self._ensure_db()
        cursor = await db.execute("SELECT DISTINCT collection FROM documents ORDER BY collection")
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_stats(self) -> dict:
        """Return statistics about the document store."""
        db = await self._ensure_db()

        cursor = await db.execute("SELECT COUNT(*) FROM documents")
        total = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(DISTINCT collection) FROM documents")
        collection_count = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT collection, COUNT(*) FROM documents GROUP BY collection"
        )
        per_collection = {row[0]: row[1] for row in await cursor.fetchall()}

        return {
            "total_documents": total,
            "collection_count": collection_count,
            "per_collection": per_collection,
        }

    # ── Internal: Semantic Search ────────────────────────────────────────────

    async def _semantic_search(
        self,
        query_embedding: list[float] | None,
        top_k: int,
        collection: str | None,
        privacy_tier: str,
    ) -> list[tuple[str, float]]:
        """
        Vector similarity search using cosine similarity.

        Returns list of (doc_id, similarity_score) tuples.
        """
        if query_embedding is None:
            return []

        db = await self._ensure_db()

        # Build query
        sql = "SELECT doc_id, embedding FROM documents WHERE embedding != '[]'"
        params: list = []

        if collection:
            sql += " AND collection = ?"
            params.append(collection)

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

        results: list[tuple[str, float]] = []
        for row in rows:
            doc_id = row[0]
            try:
                doc_embedding = json.loads(row[1])
                if not doc_embedding:
                    continue
                similarity = _cosine_similarity(query_embedding, doc_embedding)
                if similarity >= self._similarity_threshold:
                    results.append((doc_id, similarity))
            except (json.JSONDecodeError, ValueError):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Internal: BM25 Search ────────────────────────────────────────────────

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        collection: str | None,
        privacy_tier: str,
    ) -> list[tuple[str, float]]:
        """
        Simple BM25-like keyword search using term frequency.

        Returns list of (doc_id, bm25_score) tuples.
        """
        db = await self._ensure_db()

        # Tokenize query
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        # Build query
        sql = "SELECT doc_id, text FROM documents WHERE 1=1"
        params: list = []
        if collection:
            sql += " AND collection = ?"
            params.append(collection)

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

        if not rows:
            return []

        # Simple BM25 scoring
        total_docs = len(rows)
        k1 = 1.5
        b = 0.75

        # Compute average document length
        avg_dl = sum(len(_tokenize(row[1])) for row in rows) / total_docs if total_docs > 0 else 1

        results: list[tuple[str, float]] = []

        for row in rows:
            doc_id = row[0]
            doc_text = row[1]
            doc_terms = _tokenize(doc_text)
            dl = len(doc_terms)
            if dl == 0:
                continue

            score = 0.0
            term_freq: dict[str, int] = {}
            for t in doc_terms:
                term_freq[t] = term_freq.get(t, 0) + 1

            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue

                # Count documents containing term
                df = sum(1 for r in rows if term in r[1].lower())
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm

            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── Internal: Recency Score ──────────────────────────────────────────────

    def _recency_score(self, created_at: float) -> float:
        """
        Compute recency score using exponential decay.
        score = exp(-lambda * days_old) where lambda = 0.1
        """
        days_old = (time.time() - created_at) / 86400.0
        decay_lambda = 0.1
        return math.exp(-decay_lambda * max(0, days_old))

    # ── Internal: Result Merging ─────────────────────────────────────────────

    def _merge_results(
        self,
        semantic: dict[str, float],
        bm25: dict[str, float],
        recency_scores: dict[str, float],
        weights: dict[str, float],
    ) -> list[dict]:
        """
        Weighted rank fusion of semantic, BM25, and recency scores.

        Each score type is normalized to [0, 1] before weighting.
        """
        all_ids = set(semantic.keys()) | set(bm25.keys())

        # Normalize scores to [0, 1]
        semantic_norm = _normalize_scores(semantic)
        bm25_norm = _normalize_scores(bm25)

        results: list[dict] = []
        for doc_id in all_ids:
            s_score = semantic_norm.get(doc_id, 0.0)
            b_score = bm25_norm.get(doc_id, 0.0)
            r_score = recency_scores.get(doc_id, 0.5)

            combined = (
                weights.get("semantic", 0.6) * s_score
                + weights.get("bm25", 0.3) * b_score
                + weights.get("recency", 0.1) * r_score
            )

            results.append({"doc_id": doc_id, "score": combined})

        return results

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            self._initialized = False


# ── Utility Functions ────────────────────────────────────────────────────────

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


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\w+', text.lower())


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return {k: 1.0 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
