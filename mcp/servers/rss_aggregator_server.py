"""
NEXUS MCP Server — RSS / Atom Feed Aggregator
Monitors AI technology blogs, model release pages, and governance bodies.
Uses Python stdlib only (urllib + xml.etree.ElementTree).

Pre-configured feeds (all AI governance / tech relevant):
  Anthropic Blog       — model releases, safety research
  OpenAI Blog          — GPT updates, policy
  HuggingFace Blog     — open models, datasets, libraries
  Google DeepMind      — research, Gemini updates
  NIST AI              — AI RMF updates, standards
  EU AI Act tracker    — legislative updates
  Ars Technica AI      — industry news
  MIT Technology Review AI — research trends

Tools:
  rss_add_feed          Register a new RSS/Atom feed
  rss_list_feeds        List all registered feeds
  rss_remove_feed       Remove a feed
  rss_fetch_feed        Fetch and return items from one feed
  rss_fetch_all         Fetch all feeds, return merged & sorted items
  rss_search_items      Search across cached items
  rss_get_stats         Show fetch stats per feed
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

FEEDS_FILE  = Path(os.environ.get("NEXUS_RSS_FILE", "data/rss_feeds.json"))
CACHE_FILE  = Path(os.environ.get("NEXUS_RSS_CACHE", "data/rss_cache.json"))
MAX_ITEMS   = int(os.environ.get("NEXUS_RSS_MAX_ITEMS", "50"))   # per feed
CACHE_TTL_S = int(os.environ.get("NEXUS_RSS_CACHE_TTL", "3600")) # 1 hour

# ── Default feed registry ──────────────────────────────────────────────────────
DEFAULT_FEEDS = [
    {
        "id":       "anthropic",
        "name":     "Anthropic News",
        "url":      "https://www.anthropic.com/rss.xml",
        "category": "model_release",
        "tags":     ["claude", "safety", "alignment"],
    },
    {
        "id":       "openai",
        "name":     "OpenAI Blog",
        "url":      "https://openai.com/blog/rss.xml",
        "category": "model_release",
        "tags":     ["gpt", "api", "policy"],
    },
    {
        "id":       "huggingface",
        "name":     "Hugging Face Blog",
        "url":      "https://huggingface.co/blog/feed.xml",
        "category": "open_source",
        "tags":     ["transformers", "datasets", "models"],
    },
    {
        "id":       "deepmind",
        "name":     "Google DeepMind Blog",
        "url":      "https://deepmind.google/blog/rss.xml",
        "category": "research",
        "tags":     ["gemini", "research", "safety"],
    },
    {
        "id":       "nist_ai",
        "name":     "NIST AI Program",
        "url":      "https://www.nist.gov/artificial-intelligence/rss.xml",
        "category": "governance",
        "tags":     ["nist", "rmf", "standards", "governance"],
    },
    {
        "id":       "mit_tech_review_ai",
        "name":     "MIT Technology Review — AI",
        "url":      "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
        "category": "news",
        "tags":     ["ai", "research", "industry"],
    },
    {
        "id":       "ars_ai",
        "name":     "Ars Technica — AI",
        "url":      "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "category": "news",
        "tags":     ["llm", "industry", "policy"],
    },
    {
        "id":       "langchain_blog",
        "name":     "LangChain Blog",
        "url":      "https://blog.langchain.dev/rss/",
        "category": "tooling",
        "tags":     ["agents", "rag", "langchain", "langgraph"],
    },
]


# ── State management ───────────────────────────────────────────────────────────

def _load_feeds() -> list[dict]:
    FEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if FEEDS_FILE.exists():
        try:
            return json.loads(FEEDS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # First run: seed with defaults
    _save_feeds(DEFAULT_FEEDS)
    return DEFAULT_FEEDS


def _save_feeds(feeds: list[dict]) -> None:
    FEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEEDS_FILE.write_text(json.dumps(feeds, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ── RSS/Atom Parser ────────────────────────────────────────────────────────────

NS_ATOM  = "http://www.w3.org/2005/Atom"
NS_MEDIA = "http://search.yahoo.com/mrss/"
NS_DC    = "http://purl.org/dc/elements/1.1/"


def _parse_feed(xml_text: str, feed_id: str, feed_name: str) -> list[dict]:
    """Parse RSS 2.0 or Atom 1.0 XML into a list of item dicts."""
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    tag = root.tag.lower()

    def _clean(text: str) -> str:
        if not text:
            return ""
        # Strip basic HTML tags
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        return " ".join(text.split())[:600]

    if "feed" in tag:
        # Atom
        for entry in root.findall(f"{{{NS_ATOM}}}entry"):
            def _t(name: str) -> str:
                el = entry.find(f"{{{NS_ATOM}}}{name}")
                return (el.text or "").strip() if el is not None else ""
            link = ""
            for l in entry.findall(f"{{{NS_ATOM}}}link"):
                if l.attrib.get("rel", "alternate") == "alternate":
                    link = l.attrib.get("href", "")
                    break
            if not link:
                link_el = entry.find(f"{{{NS_ATOM}}}link")
                if link_el is not None:
                    link = link_el.attrib.get("href", "")
            items.append({
                "feed_id":   feed_id,
                "feed_name": feed_name,
                "title":     _clean(_t("title")),
                "link":      link,
                "summary":   _clean(_t("summary") or _t("content")),
                "published": _t("published") or _t("updated"),
                "author":    _t("author"),
            })
    else:
        # RSS 2.0
        for item in root.findall(".//item"):
            def _r(name: str) -> str:
                el = item.find(name)
                return (el.text or "").strip() if el is not None else ""
            items.append({
                "feed_id":   feed_id,
                "feed_name": feed_name,
                "title":     _clean(_r("title")),
                "link":      _r("link"),
                "summary":   _clean(_r("description")),
                "published": _r("pubDate"),
                "author":    _r(f"{{{NS_DC}}}creator") or _r("author"),
            })

    return items[:MAX_ITEMS]


def _fetch_xml(url: str) -> tuple[str, Optional[str]]:
    """Fetch RSS/Atom XML. Returns (xml_text, error_or_None)."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NEXUS-RSS-Aggregator/1.0 (+https://github.com/nexus)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace"), None
    except Exception as exc:
        return "", str(exc)


# ── Tool Implementations ───────────────────────────────────────────────────────

def rss_add_feed(
    id: str,
    name: str,
    url: str,
    category: str = "general",
    tags: Optional[list[str]] = None,
) -> dict:
    """Register a new RSS or Atom feed."""
    feeds = _load_feeds()
    if any(f["id"] == id for f in feeds):
        return {"error": f"Feed ID '{id}' already exists"}
    entry = {"id": id, "name": name, "url": url, "category": category, "tags": tags or []}
    feeds.append(entry)
    _save_feeds(feeds)
    return {"added": True, "feed_id": id, "total_feeds": len(feeds)}


def rss_list_feeds() -> dict:
    """List all registered feeds."""
    feeds  = _load_feeds()
    cache  = _load_cache()
    result = []
    for f in feeds:
        cached = cache.get(f["id"], {})
        result.append({
            "id":           f["id"],
            "name":         f["name"],
            "url":          f["url"],
            "category":     f["category"],
            "tags":         f.get("tags", []),
            "last_fetched": cached.get("fetched_at"),
            "item_count":   len(cached.get("items", [])),
        })
    return {"feeds": result, "total": len(result)}


def rss_remove_feed(id: str) -> dict:
    """Remove a feed by ID."""
    feeds = _load_feeds()
    before = len(feeds)
    feeds  = [f for f in feeds if f["id"] != id]
    _save_feeds(feeds)
    return {"removed": len(feeds) < before, "feed_id": id}


def rss_fetch_feed(feed_id: str, force_refresh: bool = False) -> dict:
    """Fetch items from a specific feed (cached by default)."""
    feeds  = _load_feeds()
    feed   = next((f for f in feeds if f["id"] == feed_id), None)
    if feed is None:
        return {"error": f"Feed '{feed_id}' not found"}

    cache  = _load_cache()
    cached = cache.get(feed_id, {})
    now    = time.time()

    # Use cache if fresh
    if (not force_refresh
            and cached.get("items")
            and (now - cached.get("fetched_ts", 0)) < CACHE_TTL_S):
        return {
            "feed_id":   feed_id,
            "name":      feed["name"],
            "items":     cached["items"],
            "cached":    True,
            "fetched_at": cached.get("fetched_at"),
        }

    xml_text, err = _fetch_xml(feed["url"])
    if err:
        return {"error": f"Fetch failed for {feed['url']}: {err}"}

    items = _parse_feed(xml_text, feed_id, feed["name"])
    cache[feed_id] = {
        "items":       items,
        "fetched_ts":  now,
        "fetched_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
    }
    _save_cache(cache)
    return {
        "feed_id":   feed_id,
        "name":      feed["name"],
        "items":     items,
        "cached":    False,
        "item_count": len(items),
    }


def rss_fetch_all(
    force_refresh: bool = False,
    category_filter: Optional[str] = None,
    max_items_per_feed: int = 5,
) -> dict:
    """Fetch all feeds and return merged, time-sorted items."""
    feeds  = _load_feeds()
    all_items: list[dict] = []
    errors:    list[dict] = []

    for feed in feeds:
        if category_filter and feed.get("category") != category_filter:
            continue
        result = rss_fetch_feed(feed["id"], force_refresh=force_refresh)
        if "error" in result:
            errors.append({"feed_id": feed["id"], "error": result["error"]})
        else:
            all_items.extend(result.get("items", [])[:max_items_per_feed])

    # Sort by published date (best-effort — ISO strings sort lexicographically)
    all_items.sort(key=lambda x: x.get("published", ""), reverse=True)
    return {
        "items":       all_items,
        "total":       len(all_items),
        "feeds_fetched": len(feeds),
        "errors":      errors,
    }


def rss_search_items(
    query: str,
    category_filter: Optional[str] = None,
    max_results: int = 20,
) -> dict:
    """Search cached items by title/summary keyword."""
    cache  = _load_cache()
    feeds  = {f["id"]: f for f in _load_feeds()}
    query_lower = query.lower()
    hits   = []

    for feed_id, cached in cache.items():
        feed     = feeds.get(feed_id, {})
        category = feed.get("category", "")
        if category_filter and category != category_filter:
            continue
        for item in cached.get("items", []):
            text = (item.get("title", "") + " " + item.get("summary", "")).lower()
            if query_lower in text:
                hits.append(item)

    hits.sort(key=lambda x: x.get("published", ""), reverse=True)
    return {"query": query, "hits": hits[:max_results], "total": len(hits)}


def rss_get_stats() -> dict:
    """Show cache stats per feed."""
    cache  = _load_cache()
    feeds  = _load_feeds()
    stats  = []
    for feed in feeds:
        cached = cache.get(feed["id"], {})
        stats.append({
            "id":           feed["id"],
            "name":         feed["name"],
            "category":     feed["category"],
            "item_count":   len(cached.get("items", [])),
            "last_fetched": cached.get("fetched_at", "never"),
        })
    return {"feeds": stats, "total_cached_items": sum(len(c.get("items", [])) for c in cache.values())}


# ── MCP Schema ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "rss_add_feed",
        "description": "Register a new RSS or Atom feed for monitoring.",
        "inputSchema": {
            "type": "object",
            "required": ["id", "name", "url"],
            "properties": {
                "id":       {"type": "string", "description": "Unique feed identifier"},
                "name":     {"type": "string"},
                "url":      {"type": "string"},
                "category": {"type": "string", "description": "model_release | research | governance | tooling | news"},
                "tags":     {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "rss_list_feeds",
        "description": "List all registered RSS/Atom feeds with cache stats.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "rss_remove_feed",
        "description": "Remove a feed by ID.",
        "inputSchema": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
        },
    },
    {
        "name": "rss_fetch_feed",
        "description": "Fetch items from a specific feed (cached by default).",
        "inputSchema": {
            "type": "object",
            "required": ["feed_id"],
            "properties": {
                "feed_id":       {"type": "string"},
                "force_refresh": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "rss_fetch_all",
        "description": "Fetch all feeds and return merged items sorted by date.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force_refresh":       {"type": "boolean", "default": False},
                "category_filter":     {"type": "string", "description": "Only fetch feeds in this category"},
                "max_items_per_feed":  {"type": "integer", "default": 5},
            },
        },
    },
    {
        "name": "rss_search_items",
        "description": "Search across all cached feed items by keyword.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query":           {"type": "string"},
                "category_filter": {"type": "string"},
                "max_results":     {"type": "integer", "default": 20},
            },
        },
    },
    {
        "name": "rss_get_stats",
        "description": "Show cache statistics per feed.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_MAP = {
    "rss_add_feed":    rss_add_feed,
    "rss_list_feeds":  rss_list_feeds,
    "rss_remove_feed": rss_remove_feed,
    "rss_fetch_feed":  rss_fetch_feed,
    "rss_fetch_all":   rss_fetch_all,
    "rss_search_items": rss_search_items,
    "rss_get_stats":   rss_get_stats,
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
                "serverInfo":      {"name": "rss_aggregator", "version": "1.0"},
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
