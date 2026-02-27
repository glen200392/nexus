"""
NEXUS MCP Server — arXiv Paper Monitor
Queries the arXiv Atom API and monitors keyword feeds for AI research papers.
Uses Python stdlib only (urllib + xml.etree.ElementTree).

Tools:
  arxiv_search          Search papers by keyword + category
  arxiv_get_paper       Fetch metadata for a specific arXiv ID
  arxiv_get_recent      Get recent papers from a category (cs.AI, cs.LG, cs.CL…)
  arxiv_add_keyword     Add a keyword to the monitoring list
  arxiv_list_keywords   List all monitored keywords
  arxiv_remove_keyword  Remove a keyword
  arxiv_run_monitor     Run all monitored keyword queries, return new papers

Environment:
  ARXIV_MONITOR_FILE  (default: data/arxiv_monitor.json) — persists keywords + seen IDs
  ARXIV_MAX_RESULTS   (default: 20)

arXiv categories relevant to NEXUS:
  cs.AI   — Artificial Intelligence
  cs.LG   — Machine Learning
  cs.CL   — Computation and Language (NLP / LLMs)
  cs.NE   — Neural and Evolutionary Computing
  cs.CR   — Cryptography and Security (AI safety)
  stat.ML — Statistics / ML
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

ARXIV_BASE        = "https://export.arxiv.org/api/query"
MONITOR_FILE      = Path(os.environ.get("ARXIV_MONITOR_FILE", "data/arxiv_monitor.json"))
MAX_RESULTS       = int(os.environ.get("ARXIV_MAX_RESULTS", "20"))
NS                = {"atom": "http://www.w3.org/2005/Atom",
                     "arxiv": "http://arxiv.org/schemas/atom"}


# ── Persistence ────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MONITOR_FILE.exists():
        try:
            return json.loads(MONITOR_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"keywords": [], "seen_ids": [], "last_run": None}


def _save_state(state: dict) -> None:
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Keep seen_ids bounded at 5000
    state["seen_ids"] = state.get("seen_ids", [])[-5000:]
    MONITOR_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ── arXiv Atom Parsing ─────────────────────────────────────────────────────────

def _fetch_arxiv(search_query: str, max_results: int = MAX_RESULTS,
                 sort_by: str = "submittedDate") -> list[dict]:
    params = {
        "search_query": search_query,
        "max_results":  max_results,
        "sortBy":       sort_by,
        "sortOrder":    "descending",
    }
    url = ARXIV_BASE + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        return [{"error": str(exc)}]

    try:
        root    = ET.fromstring(raw)
        entries = root.findall("atom:entry", NS)
    except ET.ParseError as exc:
        return [{"error": f"XML parse error: {exc}"}]

    papers = []
    for entry in entries:
        def _text(tag: str, ns: str = "atom") -> str:
            el = entry.find(f"{ns}:{tag}", NS)
            return (el.text or "").strip() if el is not None else ""

        arxiv_id = _text("id")
        # Extract short ID from URL
        short_id = arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id

        authors = [
            (a.find("atom:name", NS).text or "").strip()
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]
        # PDF link
        pdf_link = ""
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib.get("href", "")
                break
        if not pdf_link:
            pdf_link = arxiv_id.replace("/abs/", "/pdf/")

        papers.append({
            "id":           short_id,
            "title":        _text("title").replace("\n", " "),
            "abstract":     _text("summary")[:800].replace("\n", " "),
            "authors":      authors[:5],
            "categories":   categories,
            "published":    _text("published")[:10],
            "updated":      _text("updated")[:10],
            "arxiv_url":    arxiv_id,
            "pdf_url":      pdf_link,
        })
    return papers


# ── Tool Implementations ───────────────────────────────────────────────────────

def arxiv_search(
    query: str,
    categories: Optional[list[str]] = None,
    max_results: int = 10,
    date_range_days: int = 0,
) -> dict:
    """Search arXiv papers. categories: ['cs.AI', 'cs.LG', ...]"""
    q_parts = [f"all:{urllib.parse.quote(query)}"]
    if categories:
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        q_parts.append(f"({cat_query})")
    search_q = " AND ".join(q_parts)
    papers   = _fetch_arxiv(search_q, max_results=min(max_results, 50))
    # Filter by date if requested
    if date_range_days > 0 and papers:
        cutoff = time.strftime(
            "%Y-%m-%d",
            time.gmtime(time.time() - date_range_days * 86400)
        )
        papers = [p for p in papers if p.get("published", "") >= cutoff]
    return {"query": query, "papers": papers, "count": len(papers)}


def arxiv_get_paper(arxiv_id: str) -> dict:
    """Fetch metadata for a specific paper by arXiv ID (e.g. '2303.08774')."""
    # Strip version suffix
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id.split("/")[-1] else arxiv_id
    papers   = _fetch_arxiv(f"id:{clean_id}", max_results=1)
    if not papers or "error" in papers[0]:
        return {"error": f"Paper not found: {arxiv_id}"}
    return papers[0]


def arxiv_get_recent(
    category: str = "cs.AI",
    max_results: int = 10,
) -> dict:
    """Get the most recently submitted papers in a category."""
    papers = _fetch_arxiv(f"cat:{category}", max_results=min(max_results, 50))
    return {"category": category, "papers": papers, "count": len(papers)}


def arxiv_add_keyword(
    keyword: str,
    categories: Optional[list[str]] = None,
    label: str = "",
) -> dict:
    """Add a keyword to the monitoring list."""
    state = _load_state()
    for kw in state.get("keywords", []):
        if kw["keyword"].lower() == keyword.lower():
            return {"error": f"Keyword '{keyword}' already monitored"}
    entry = {
        "keyword":    keyword,
        "categories": categories or ["cs.AI", "cs.LG", "cs.CL"],
        "label":      label or keyword,
        "added_at":   time.strftime("%Y-%m-%d", time.gmtime()),
    }
    state.setdefault("keywords", []).append(entry)
    _save_state(state)
    return {"added": True, "keyword": keyword, "total_keywords": len(state["keywords"])}


def arxiv_list_keywords() -> dict:
    """List all monitored keywords."""
    state = _load_state()
    return {
        "keywords":     state.get("keywords", []),
        "last_run":     state.get("last_run"),
        "seen_ids_count": len(state.get("seen_ids", [])),
    }


def arxiv_remove_keyword(keyword: str) -> dict:
    """Remove a keyword from monitoring."""
    state = _load_state()
    before = len(state.get("keywords", []))
    state["keywords"] = [k for k in state.get("keywords", [])
                         if k["keyword"].lower() != keyword.lower()]
    _save_state(state)
    removed = before - len(state["keywords"])
    return {"removed": removed > 0, "keyword": keyword}


def arxiv_run_monitor() -> dict:
    """Run all monitored keyword queries and return only NEW papers."""
    state    = _load_state()
    keywords = state.get("keywords", [])
    if not keywords:
        return {
            "new_papers": [],
            "message":    "No keywords monitored. Use arxiv_add_keyword first.",
        }
    seen_ids = set(state.get("seen_ids", []))
    new_papers = []

    for kw_entry in keywords:
        kw         = kw_entry["keyword"]
        categories = kw_entry.get("categories", ["cs.AI", "cs.LG", "cs.CL"])
        q_parts    = [f"all:{urllib.parse.quote(kw)}"]
        if categories:
            cat_q = " OR ".join(f"cat:{c}" for c in categories)
            q_parts.append(f"({cat_q})")
        papers = _fetch_arxiv(" AND ".join(q_parts), max_results=MAX_RESULTS)
        for p in papers:
            if "error" in p:
                continue
            pid = p["id"]
            if pid not in seen_ids:
                p["matched_keyword"] = kw
                p["matched_label"]   = kw_entry.get("label", kw)
                new_papers.append(p)
                seen_ids.add(pid)

    state["seen_ids"] = list(seen_ids)
    state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(state)

    # Sort by publication date (newest first)
    new_papers.sort(key=lambda p: p.get("published", ""), reverse=True)
    return {
        "new_papers":     new_papers,
        "new_count":      len(new_papers),
        "keywords_checked": len(keywords),
        "last_run":       state["last_run"],
    }


# ── MCP Schema ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "arxiv_search",
        "description": "Search arXiv papers by keyword, optionally filtered by category.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query":           {"type": "string"},
                "categories":      {"type": "array", "items": {"type": "string"}, "description": "e.g. ['cs.AI','cs.LG']"},
                "max_results":     {"type": "integer", "default": 10},
                "date_range_days": {"type": "integer", "default": 0, "description": "Only papers within last N days (0 = all)"},
            },
        },
    },
    {
        "name": "arxiv_get_paper",
        "description": "Get metadata for a specific arXiv paper by ID.",
        "inputSchema": {
            "type": "object",
            "required": ["arxiv_id"],
            "properties": {"arxiv_id": {"type": "string", "description": "e.g. '2303.08774' or '2303.08774v2'"}},
        },
    },
    {
        "name": "arxiv_get_recent",
        "description": "Get the most recently submitted papers in an arXiv category.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category":    {"type": "string", "default": "cs.AI", "description": "cs.AI, cs.LG, cs.CL, cs.NE…"},
                "max_results": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "arxiv_add_keyword",
        "description": "Add a keyword to the monitoring list.",
        "inputSchema": {
            "type": "object",
            "required": ["keyword"],
            "properties": {
                "keyword":    {"type": "string"},
                "categories": {"type": "array", "items": {"type": "string"}, "description": "Default: ['cs.AI','cs.LG','cs.CL']"},
                "label":      {"type": "string", "description": "Human-readable label for this alert"},
            },
        },
    },
    {
        "name": "arxiv_list_keywords",
        "description": "List all monitored keywords and their categories.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "arxiv_remove_keyword",
        "description": "Stop monitoring a keyword.",
        "inputSchema": {
            "type": "object",
            "required": ["keyword"],
            "properties": {"keyword": {"type": "string"}},
        },
    },
    {
        "name": "arxiv_run_monitor",
        "description": "Query all monitored keywords and return only papers not seen before.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_MAP = {
    "arxiv_search":         arxiv_search,
    "arxiv_get_paper":      arxiv_get_paper,
    "arxiv_get_recent":     arxiv_get_recent,
    "arxiv_add_keyword":    arxiv_add_keyword,
    "arxiv_list_keywords":  arxiv_list_keywords,
    "arxiv_remove_keyword": arxiv_remove_keyword,
    "arxiv_run_monitor":    arxiv_run_monitor,
}


# ── MCP Stdio Loop ─────────────────────────────────────────────────────────────

def _respond(req_id, result: dict) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
    sys.stdout.flush()

def _error(req_id, code: int, message: str) -> None:
    sys.stdout.write(json.dumps({
        "jsonrpc": "2.0", "id": req_id,
        "error":   {"code": code, "message": message},
    }) + "\n")
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
            _respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo":      {"name": "arxiv_monitor", "version": "1.0"},
            })
        elif method == "tools/list":
            _respond(req_id, {"tools": TOOLS})
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
                _respond(req_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]})
            except Exception as exc:
                _respond(req_id, {"content": [{"type": "text", "text": json.dumps({"error": str(exc)})}]})
        elif method == "notifications/initialized":
            pass
        else:
            _error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
