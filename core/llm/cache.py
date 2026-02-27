"""
NEXUS v2 — Semantic LLM response cache with SQLite backend.

Supports exact-hash matching and optional cosine-similarity matching
when an EmbeddingManager is provided.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/cache/llm_cache.db")


@dataclass
class CacheEntry:
    """Single row in the cache table."""
    query_hash: str
    query_text: str
    response_json: str
    model: str
    embedding: str = ""          # JSON-encoded float list
    created_at: float = 0.0
    ttl: int = 3600
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl


class LLMSemanticCache:
    """SQLite-backed LLM response cache with optional semantic matching."""

    SIMILARITY_THRESHOLD = 0.92

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_manager: Any | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = embedding_manager

        self._total_lookups = 0
        self._total_hits = 0
        self._saved_cost_usd = 0.0

        self._init_db()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash   TEXT PRIMARY KEY,
                    query_text   TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    embedding    TEXT DEFAULT '',
                    created_at   REAL NOT NULL,
                    ttl          INTEGER NOT NULL DEFAULT 3600,
                    hit_count    INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get(self, query: str, model: str) -> dict | None:
        """Look up a cached response.

        1. Try semantic similarity if an EmbeddingManager is available.
        2. Fall back to exact hash match.
        """
        self._total_lookups += 1

        # Try semantic match first
        if self.embedding_manager is not None:
            result = await self._semantic_lookup(query, model)
            if result is not None:
                self._total_hits += 1
                return result

        # Exact hash fallback
        result = self._exact_lookup(query, model)
        if result is not None:
            self._total_hits += 1
        return result

    async def put(
        self,
        query: str,
        response: dict,
        model: str,
        ttl: int = 3600,
    ) -> None:
        """Store a response in the cache."""
        qhash = self._hash(query)
        embedding_json = ""
        if self.embedding_manager is not None:
            try:
                emb = await self.embedding_manager.embed(query)
                embedding_json = json.dumps(emb if isinstance(emb, list) else emb.tolist())
            except Exception:
                logger.debug("Failed to generate embedding for cache entry")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache
                    (query_hash, query_text, response_json, model, embedding, created_at, ttl, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (qhash, query, json.dumps(response), model, embedding_json, time.time(), ttl),
            )

    async def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        qhash = self._hash(query)
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE query_hash = ?", (qhash,))

    async def clear(self) -> None:
        """Remove all entries."""
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")
        self._total_hits = 0
        self._total_lookups = 0
        self._saved_cost_usd = 0.0

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM cache").fetchone()
        total_entries = row[0] if row else 0
        total_db_hits = row[1] if row else 0
        hit_rate = (self._total_hits / self._total_lookups) if self._total_lookups > 0 else 0.0
        return {
            "hit_rate": round(hit_rate, 4),
            "total_entries": total_entries,
            "total_hits": self._total_hits,
            "total_db_hits": total_db_hits,
            "total_lookups": self._total_lookups,
            "saved_cost_usd": round(self._saved_cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _exact_lookup(self, query: str, model: str) -> dict | None:
        qhash = self._hash(query)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json, created_at, ttl FROM cache WHERE query_hash = ? AND model = ?",
                (qhash, model),
            ).fetchone()
        if row is None:
            return None
        response_json, created_at, ttl = row
        if time.time() > created_at + ttl:
            return None
        # bump hit count
        with self._connect() as conn:
            conn.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?", (qhash,))
        return json.loads(response_json)

    async def _semantic_lookup(self, query: str, model: str) -> dict | None:
        try:
            query_emb = await self.embedding_manager.embed(query)
            if isinstance(query_emb, list):
                query_vec = query_emb
            else:
                query_vec = query_emb.tolist()
        except Exception:
            return None

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT query_hash, response_json, embedding, created_at, ttl FROM cache WHERE model = ?",
                (model,),
            ).fetchall()

        best_score = 0.0
        best_response: dict | None = None
        best_hash: str | None = None

        for qhash, resp_json, emb_json, created_at, ttl in rows:
            if time.time() > created_at + ttl:
                continue
            if not emb_json:
                continue
            try:
                stored_vec = json.loads(emb_json)
            except json.JSONDecodeError:
                continue

            sim = self._cosine_similarity(query_vec, stored_vec)
            if sim > best_score:
                best_score = sim
                best_response = json.loads(resp_json)
                best_hash = qhash

        if best_score >= self.SIMILARITY_THRESHOLD and best_response is not None:
            with self._connect() as conn:
                conn.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?", (best_hash,))
            return best_response
        return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
