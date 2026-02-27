"""
NEXUS MCP Fetch Server
Fetch web pages and convert to clean Markdown for LLM consumption.
Tools: fetch_url, fetch_markdown, fetch_links, fetch_json

Key feature: HTML → Markdown conversion preserves structure
(headings, lists, tables, code blocks) while stripping navigation/ads.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any


def _ok(data: Any) -> dict:
    text = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
    return {"content": [{"type": "text", "text": text}]}

def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "isError": True}


def _html_to_markdown(html: str) -> str:
    """
    Convert HTML to readable Markdown without requiring external libraries.
    Handles: headings, paragraphs, lists, links, code, tables, bold/italic.
    """
    # Remove script/style/nav/footer blocks
    for tag in ["script", "style", "nav", "footer", "header", "aside", "noscript"]:
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.I)

    # Headings
    for i in range(6, 0, -1):
        html = re.sub(rf"<h{i}[^>]*>(.*?)</h{i}>", rf"\n{'#'*i} \1\n", html, flags=re.DOTALL | re.I)

    # Bold / italic
    html = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<(em|i)[^>]*>(.*?)</\1>",     r"*\2*",   html, flags=re.DOTALL | re.I)

    # Code
    html = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", html, flags=re.DOTALL | re.I)

    # Links
    html = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r"[\2](\1)", html, flags=re.DOTALL | re.I)

    # Images
    html = re.sub(r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*src=["\']([^"\']*)["\'][^>]*/?>',
                  r"![\1](\2)", html, flags=re.I)
    html = re.sub(r'<img[^>]*src=["\']([^"\']*)["\'][^>]*/?>',
                  r"![image](\1)", html, flags=re.I)

    # Lists
    html = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<[ou]l[^>]*>", "\n", html, flags=re.I)
    html = re.sub(r"</[ou]l>",     "\n", html, flags=re.I)

    # Table (simple)
    html = re.sub(r"<tr[^>]*>",    "\n| ",   html, flags=re.I)
    html = re.sub(r"<t[dh][^>]*>", " | ",    html, flags=re.I)
    html = re.sub(r"</t[dhr]>",    "",       html, flags=re.I)
    html = re.sub(r"</tr>",        " |",     html, flags=re.I)
    html = re.sub(r"<table[^>]*>", "\n",     html, flags=re.I)
    html = re.sub(r"</table>",     "\n",     html, flags=re.I)

    # Paragraphs / breaks
    html = re.sub(r"<br\s*/?>",    "\n",    html, flags=re.I)
    html = re.sub(r"<p[^>]*>",     "\n\n",  html, flags=re.I)
    html = re.sub(r"</p>",         "\n",    html, flags=re.I)
    html = re.sub(r"<div[^>]*>",   "\n",    html, flags=re.I)
    html = re.sub(r"</div>",        "",     html, flags=re.I)

    # Strip remaining tags
    html = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
    html = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), html)

    # Clean whitespace
    html = re.sub(r"\n{4,}", "\n\n\n", html)
    html = re.sub(r" {3,}", "  ", html)
    return html.strip()


def fetch_url(url: str, max_chars: int = 20000, timeout: int = 15) -> dict:
    """Fetch a URL and return raw content with metadata."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 NEXUS-Fetch/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", errors="replace")
            return _ok({
                "url":          url,
                "status":       resp.status,
                "content_type": content_type,
                "content":      raw[:max_chars],
                "truncated":    len(raw) > max_chars,
                "char_count":   len(raw),
            })
    except Exception as exc:
        return _err(f"Fetch failed: {exc}")


def fetch_markdown(url: str, max_chars: int = 15000, timeout: int = 15) -> dict:
    """Fetch a URL and return content converted to clean Markdown."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 NEXUS-Fetch/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", errors="replace")

        if "html" in content_type.lower() or raw.strip().startswith("<"):
            markdown = _html_to_markdown(raw)
        else:
            markdown = raw  # Already plain text/markdown

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.DOTALL | re.I)
        title = title_match.group(1).strip() if title_match else url

        return _ok({
            "url":       url,
            "title":     title,
            "markdown":  markdown[:max_chars],
            "truncated": len(markdown) > max_chars,
            "word_count": len(markdown.split()),
        })
    except Exception as exc:
        return _err(f"Fetch failed: {exc}")


def fetch_links(url: str, timeout: int = 10) -> dict:
    """Extract all links from a webpage."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NEXUS/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        links = re.findall(r'href=["\']([^"\'#][^"\']*)["\']', html, re.I)
        # Make relative links absolute
        base = "/".join(url.split("/")[:3])
        abs_links = []
        for link in links:
            if link.startswith("http"):
                abs_links.append(link)
            elif link.startswith("/"):
                abs_links.append(base + link)
        abs_links = list(dict.fromkeys(abs_links))  # deduplicate
        return _ok({"url": url, "links": abs_links[:100], "total": len(abs_links)})
    except Exception as exc:
        return _err(f"Failed to fetch links: {exc}")


def fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch a JSON API endpoint."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "NEXUS-Fetch/1.0",
                "Accept":     "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return _ok(data)
    except Exception as exc:
        return _err(f"JSON fetch failed: {exc}")


TOOLS = {
    "fetch_url": {
        "fn": fetch_url,
        "description": "Fetch raw content from a URL (HTML, text, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
                "timeout":   {"type": "integer", "default": 15},
            },
            "required": ["url"],
        },
    },
    "fetch_markdown": {
        "fn": fetch_markdown,
        "description": "Fetch a webpage and convert HTML to clean Markdown for LLM reading",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":       {"type": "string"},
                "max_chars": {"type": "integer", "default": 15000},
                "timeout":   {"type": "integer", "default": 15},
            },
            "required": ["url"],
        },
    },
    "fetch_links": {
        "fn": fetch_links,
        "description": "Extract all hyperlinks from a webpage",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":     {"type": "string"},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["url"],
        },
    },
    "fetch_json": {
        "fn": fetch_json,
        "description": "Fetch and parse a JSON API endpoint",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url":     {"type": "string"},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["url"],
        },
    },
}


def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", "")
    rpc_id = msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
            "serverInfo": {"name": "nexus-fetch", "version": "1.0.0"},
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {
            "tools": [{"name": n, "description": s["description"], "inputSchema": s["inputSchema"]}
                      for n, s in TOOLS.items()]
        }}
    if method == "tools/call":
        params = msg.get("params", {})
        name   = params.get("name", "")
        args   = params.get("arguments", {})
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": _err(f"Unknown tool: {name}")}
        try:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": TOOLS[name]["fn"](**args)}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": _err(str(exc))}
    if method.startswith("notifications/"): return None
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown: {method}"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
