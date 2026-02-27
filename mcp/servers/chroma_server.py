"""
NEXUS MCP Server — ChromaDB Vector Store
Provides direct query/management access to the NEXUS knowledge base.

Tools:
  - query_collection      Search a collection by text (semantic similarity)
  - list_collections      Show all collections with doc counts
  - get_collection_info   Metadata + sample docs from a collection
  - add_documents         Ingest text(s) into a collection
  - delete_documents      Remove docs by ID from a collection
  - peek_collection       Return the first N docs (for inspection)

Environment:
  CHROMA_HOST  (default: localhost)
  CHROMA_PORT  (default: 8000)
  CHROMA_MODE  (default: local) — "local" uses PersistentClient,
                                   "http"  uses HttpClient
  CHROMA_PATH  (default: ./data/vector_store) — for local mode
"""
from __future__ import annotations

import json
import os
import sys
import uuid
import logging
from typing import Any

logger = logging.getLogger("nexus.mcp.chroma")

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_MODE = os.environ.get("CHROMA_MODE", "local")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/vector_store")

# ── ChromaDB client init (lazy) ────────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import chromadb
        if CHROMA_MODE == "http":
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        else:
            import os as _os
            _os.makedirs(CHROMA_PATH, exist_ok=True)
            _client = chromadb.PersistentClient(path=CHROMA_PATH)
        return _client
    except Exception as exc:
        raise RuntimeError(f"ChromaDB client init failed: {exc}") from exc


def _get_or_create_collection(name: str):
    client = _get_client()
    return client.get_or_create_collection(name)


# ── Tool Handlers ──────────────────────────────────────────────────────────────

def query_collection(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Semantic search in a ChromaDB collection.
    Returns ranked docs with distances and metadata.
    """
    try:
        col = _get_or_create_collection(collection_name)
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results":   min(n_results, col.count() or 1),
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        hits = []
        docs      = (results.get("documents")  or [[]])[0]
        metas     = (results.get("metadatas")  or [[]])[0]
        distances = (results.get("distances")  or [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            hits.append({
                "content":  doc,
                "metadata": meta or {},
                "distance": round(float(dist), 4),
                "score":    round(max(0.0, 1.0 - float(dist)), 4),
            })

        return {
            "collection": collection_name,
            "query":      query_text,
            "hits":       hits,
            "total":      len(hits),
        }
    except Exception as exc:
        return {"error": str(exc)}


def list_collections() -> dict:
    """List all collections with name, count, and metadata."""
    try:
        client = _get_client()
        cols   = client.list_collections()
        result = []
        for col in cols:
            try:
                count = col.count()
            except Exception:
                count = -1
            result.append({
                "name":     col.name,
                "count":    count,
                "metadata": col.metadata or {},
            })
        return {"collections": result, "total": len(result)}
    except Exception as exc:
        return {"error": str(exc)}


def get_collection_info(collection_name: str, peek_n: int = 3) -> dict:
    """Get collection stats and a sample of documents."""
    try:
        col   = _get_or_create_collection(collection_name)
        count = col.count()
        info: dict[str, Any] = {
            "name":     collection_name,
            "count":    count,
            "metadata": col.metadata or {},
        }
        if count > 0 and peek_n > 0:
            peek = col.peek(limit=peek_n)
            samples = []
            docs  = peek.get("documents") or []
            metas = peek.get("metadatas") or []
            ids   = peek.get("ids") or []
            for i, doc in enumerate(docs):
                samples.append({
                    "id":       ids[i] if i < len(ids) else "",
                    "content":  (doc or "")[:300],
                    "metadata": metas[i] if i < len(metas) else {},
                })
            info["samples"] = samples
        return info
    except Exception as exc:
        return {"error": str(exc)}


def add_documents(
    collection_name: str,
    documents: list[str],
    metadatas: list[dict] | None = None,
    ids: list[str] | None = None,
) -> dict:
    """
    Add text documents to a collection.
    ChromaDB handles embedding internally if an embedding function is set;
    otherwise uses its default (all-MiniLM-L6-v2 from sentence-transformers).
    """
    try:
        if not documents:
            return {"error": "No documents provided"}

        col = _get_or_create_collection(collection_name)

        # Auto-generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]
        if not metadatas:
            metadatas = [{} for _ in documents]

        col.add(documents=documents, metadatas=metadatas, ids=ids)
        return {
            "added":      len(documents),
            "collection": collection_name,
            "ids":        ids,
        }
    except Exception as exc:
        return {"error": str(exc)}


def delete_documents(collection_name: str, ids: list[str]) -> dict:
    """Delete documents by ID from a collection."""
    try:
        if not ids:
            return {"error": "No IDs provided"}
        col = _get_or_create_collection(collection_name)
        col.delete(ids=ids)
        return {"deleted": len(ids), "collection": collection_name}
    except Exception as exc:
        return {"error": str(exc)}


def peek_collection(collection_name: str, limit: int = 10) -> dict:
    """Return the first N documents from a collection."""
    try:
        col  = _get_or_create_collection(collection_name)
        peek = col.peek(limit=limit)
        docs  = peek.get("documents") or []
        metas = peek.get("metadatas") or []
        ids   = peek.get("ids") or []
        rows  = []
        for i, doc in enumerate(docs):
            rows.append({
                "id":       ids[i] if i < len(ids) else "",
                "content":  (doc or "")[:400],
                "metadata": metas[i] if i < len(metas) else {},
            })
        return {"collection": collection_name, "docs": rows, "count": col.count()}
    except Exception as exc:
        return {"error": str(exc)}


# ── MCP Tool Schema ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "query_collection",
        "description": "Semantic search in a ChromaDB collection by text query.",
        "inputSchema": {
            "type": "object",
            "required": ["collection_name", "query_text"],
            "properties": {
                "collection_name": {"type": "string", "description": "Target collection name"},
                "query_text":      {"type": "string", "description": "Natural language query"},
                "n_results":       {"type": "integer", "default": 5, "description": "Number of results to return"},
                "where":           {"type": "object",  "description": "Metadata filter dict (optional)"},
            },
        },
    },
    {
        "name": "list_collections",
        "description": "List all ChromaDB collections with their document counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_collection_info",
        "description": "Get stats and sample documents for a specific collection.",
        "inputSchema": {
            "type": "object",
            "required": ["collection_name"],
            "properties": {
                "collection_name": {"type": "string"},
                "peek_n":          {"type": "integer", "default": 3, "description": "Number of sample docs to return"},
            },
        },
    },
    {
        "name": "add_documents",
        "description": "Add text documents to a ChromaDB collection.",
        "inputSchema": {
            "type": "object",
            "required": ["collection_name", "documents"],
            "properties": {
                "collection_name": {"type": "string"},
                "documents":       {"type": "array", "items": {"type": "string"}},
                "metadatas":       {"type": "array", "items": {"type": "object"}, "description": "Metadata list (parallel to documents)"},
                "ids":             {"type": "array", "items": {"type": "string"}, "description": "Explicit doc IDs (auto-generated if omitted)"},
            },
        },
    },
    {
        "name": "delete_documents",
        "description": "Delete documents by ID from a ChromaDB collection.",
        "inputSchema": {
            "type": "object",
            "required": ["collection_name", "ids"],
            "properties": {
                "collection_name": {"type": "string"},
                "ids":             {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "peek_collection",
        "description": "Return the first N documents from a collection for inspection.",
        "inputSchema": {
            "type": "object",
            "required": ["collection_name"],
            "properties": {
                "collection_name": {"type": "string"},
                "limit":           {"type": "integer", "default": 10},
            },
        },
    },
]

TOOL_MAP = {
    "query_collection":   query_collection,
    "list_collections":   list_collections,
    "get_collection_info": get_collection_info,
    "add_documents":      add_documents,
    "delete_documents":   delete_documents,
    "peek_collection":    peek_collection,
}


# ── MCP Stdio Loop ─────────────────────────────────────────────────────────────

def _respond(result: dict, req_id) -> None:
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _error(req_id, code: int, message: str) -> None:
    msg = json.dumps({
        "jsonrpc": "2.0", "id": req_id,
        "error":   {"code": code, "message": message},
    })
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def main():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")

        if method == "initialize":
            _respond({
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo":      {"name": "chroma", "version": "1.0"},
            }, req_id)

        elif method == "tools/list":
            _respond({"tools": TOOLS}, req_id)

        elif method == "tools/call":
            params    = req.get("params", {})
            tool_name = params.get("name", "")
            args      = params.get("arguments", {})
            fn        = TOOL_MAP.get(tool_name)
            if fn is None:
                _error(req_id, -32601, f"Unknown tool: {tool_name}")
                continue
            try:
                result = fn(**{k: v for k, v in args.items()})
                _respond({
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }, req_id)
            except Exception as exc:
                _respond({
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}]
                }, req_id)

        elif method == "notifications/initialized":
            pass  # Ignore

        else:
            _error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
