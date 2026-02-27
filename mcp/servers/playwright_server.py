"""
NEXUS MCP Server — Playwright Browser Automation
Provides browser control tools via MCP over stdio.

A single Chromium browser instance is kept alive for the session.
Multiple pages (tabs) are tracked by page_id.

Tools:
  - browser_navigate      Go to a URL
  - browser_screenshot    Take a page screenshot (returns base64 PNG)
  - browser_click         Click element by CSS selector
  - browser_fill          Fill an input field
  - browser_extract_text  Extract visible text from page (or selector)
  - browser_get_html      Get the full or partial HTML of the page
  - browser_evaluate      Run JavaScript in the page context
  - browser_new_page      Open a new tab (returns page_id)
  - browser_close_page    Close a tab
  - browser_list_pages    List all open tabs

Environment:
  PLAYWRIGHT_HEADLESS  (default: true)  — set to "false" to see browser
  PLAYWRIGHT_TIMEOUT   (default: 30000) — default timeout in ms
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger("nexus.mcp.playwright")

# ── Config ─────────────────────────────────────────────────────────────────────
HEADLESS = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower() != "false"
TIMEOUT  = int(os.environ.get("PLAYWRIGHT_TIMEOUT", "30000"))

# ── Browser State ──────────────────────────────────────────────────────────────
_playwright  = None
_browser     = None
_context     = None
_pages: dict[str, object] = {}          # page_id → Page
_page_counter = 0


async def _ensure_browser():
    """Lazy-init: start Playwright + Chromium on first call."""
    global _playwright, _browser, _context
    if _browser is not None:
        return
    try:
        from playwright.async_api import async_playwright
        _playwright = await async_playwright().start()
        _browser    = await _playwright.chromium.launch(headless=HEADLESS)
        _context    = await _browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        logger.info("Playwright Chromium started (headless=%s)", HEADLESS)
    except ImportError:
        raise RuntimeError(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        )


def _make_page_id() -> str:
    global _page_counter
    _page_counter += 1
    return f"page_{_page_counter}"


async def _get_page(page_id: Optional[str] = None):
    """Return the requested page, or create a default one."""
    await _ensure_browser()
    if page_id and page_id in _pages:
        return _pages[page_id], page_id
    # Create a new page if none exist or no ID given
    pid  = page_id or _make_page_id()
    page = await _context.new_page()
    page.set_default_timeout(TIMEOUT)
    _pages[pid] = page
    return page, pid


# ── Tool Handlers (all async) ──────────────────────────────────────────────────

async def browser_navigate(url: str, page_id: Optional[str] = None, wait_until: str = "domcontentloaded") -> dict:
    """Navigate to a URL. Creates default page if page_id not given."""
    page, pid = await _get_page(page_id)
    try:
        resp = await page.goto(url, wait_until=wait_until, timeout=TIMEOUT)
        return {
            "page_id": pid,
            "url":     page.url,
            "status":  resp.status if resp else None,
            "title":   await page.title(),
        }
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_screenshot(
    page_id: Optional[str] = None,
    full_page: bool = False,
    selector: Optional[str] = None,
) -> dict:
    """Capture a screenshot. Returns base64-encoded PNG."""
    page, pid = await _get_page(page_id)
    try:
        if selector:
            el = await page.query_selector(selector)
            if el is None:
                return {"error": f"Selector not found: {selector}", "page_id": pid}
            data = await el.screenshot()
        else:
            data = await page.screenshot(full_page=full_page)
        b64 = base64.b64encode(data).decode()
        return {
            "page_id":  pid,
            "format":   "png",
            "encoding": "base64",
            "data":     b64,
            "url":      page.url,
        }
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_click(
    selector: str,
    page_id: Optional[str] = None,
    button: str = "left",
    delay: int = 0,
) -> dict:
    """Click an element identified by CSS selector."""
    page, pid = await _get_page(page_id)
    try:
        await page.click(selector, button=button, delay=delay, timeout=TIMEOUT)
        return {"page_id": pid, "clicked": selector, "url": page.url}
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_fill(
    selector: str,
    value: str,
    page_id: Optional[str] = None,
    clear_first: bool = True,
) -> dict:
    """Fill a form input field."""
    page, pid = await _get_page(page_id)
    try:
        if clear_first:
            await page.fill(selector, "", timeout=TIMEOUT)
        await page.fill(selector, value, timeout=TIMEOUT)
        return {"page_id": pid, "selector": selector, "filled": True}
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_extract_text(
    page_id: Optional[str] = None,
    selector: Optional[str] = None,
    max_chars: int = 8000,
) -> dict:
    """Extract visible text from the page or a specific element."""
    page, pid = await _get_page(page_id)
    try:
        if selector:
            el = await page.query_selector(selector)
            if el is None:
                return {"error": f"Selector not found: {selector}", "page_id": pid}
            text = await el.inner_text()
        else:
            text = await page.inner_text("body")
        return {
            "page_id": pid,
            "url":     page.url,
            "text":    text[:max_chars],
            "length":  len(text),
        }
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_get_html(
    page_id: Optional[str] = None,
    selector: Optional[str] = None,
    max_chars: int = 20000,
) -> dict:
    """Get the HTML of the page or a specific element."""
    page, pid = await _get_page(page_id)
    try:
        if selector:
            el = await page.query_selector(selector)
            if el is None:
                return {"error": f"Selector not found: {selector}", "page_id": pid}
            html = await el.inner_html()
        else:
            html = await page.content()
        return {
            "page_id": pid,
            "url":     page.url,
            "html":    html[:max_chars],
            "length":  len(html),
        }
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_evaluate(
    script: str,
    page_id: Optional[str] = None,
) -> dict:
    """Execute JavaScript in the page context and return the result."""
    page, pid = await _get_page(page_id)
    try:
        result = await page.evaluate(script)
        return {"page_id": pid, "result": result}
    except Exception as exc:
        return {"error": str(exc), "page_id": pid}


async def browser_new_page() -> dict:
    """Open a new browser tab and return its page_id."""
    await _ensure_browser()
    pid  = _make_page_id()
    page = await _context.new_page()
    page.set_default_timeout(TIMEOUT)
    _pages[pid] = page
    return {"page_id": pid, "url": "about:blank"}


async def browser_close_page(page_id: str) -> dict:
    """Close a browser tab."""
    if page_id not in _pages:
        return {"error": f"Unknown page_id: {page_id}"}
    try:
        await _pages[page_id].close()
        del _pages[page_id]
        return {"closed": page_id}
    except Exception as exc:
        return {"error": str(exc)}


async def browser_list_pages() -> dict:
    """List all open browser tabs."""
    info = []
    for pid, page in _pages.items():
        try:
            title = await page.title()
            url   = page.url
        except Exception:
            title, url = "", "unknown"
        info.append({"page_id": pid, "url": url, "title": title})
    return {"pages": info, "count": len(info)}


# ── Tool Schema ────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "browser_navigate",
        "description": "Navigate a browser tab to a URL.",
        "inputSchema": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url":        {"type": "string"},
                "page_id":    {"type": "string", "description": "Tab to use (creates one if omitted)"},
                "wait_until": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle"], "default": "domcontentloaded"},
            },
        },
    },
    {
        "name": "browser_screenshot",
        "description": "Take a screenshot of the page. Returns base64-encoded PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page_id":   {"type": "string"},
                "full_page": {"type": "boolean", "default": False},
                "selector":  {"type": "string", "description": "CSS selector for element screenshot"},
            },
        },
    },
    {
        "name": "browser_click",
        "description": "Click an element by CSS selector.",
        "inputSchema": {
            "type": "object",
            "required": ["selector"],
            "properties": {
                "selector": {"type": "string"},
                "page_id":  {"type": "string"},
                "button":   {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                "delay":    {"type": "integer", "default": 0, "description": "Milliseconds to hold button"},
            },
        },
    },
    {
        "name": "browser_fill",
        "description": "Fill a text input or textarea.",
        "inputSchema": {
            "type": "object",
            "required": ["selector", "value"],
            "properties": {
                "selector":    {"type": "string"},
                "value":       {"type": "string"},
                "page_id":     {"type": "string"},
                "clear_first": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "browser_extract_text",
        "description": "Extract visible text from the page or a specific element.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page_id":   {"type": "string"},
                "selector":  {"type": "string", "description": "CSS selector (optional, defaults to body)"},
                "max_chars": {"type": "integer", "default": 8000},
            },
        },
    },
    {
        "name": "browser_get_html",
        "description": "Get raw HTML of the page or a specific element.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page_id":   {"type": "string"},
                "selector":  {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
            },
        },
    },
    {
        "name": "browser_evaluate",
        "description": "Run JavaScript in the page and return the result.",
        "inputSchema": {
            "type": "object",
            "required": ["script"],
            "properties": {
                "script":  {"type": "string", "description": "JS expression to evaluate"},
                "page_id": {"type": "string"},
            },
        },
    },
    {
        "name": "browser_new_page",
        "description": "Open a new browser tab. Returns page_id.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_close_page",
        "description": "Close a browser tab by page_id.",
        "inputSchema": {
            "type": "object",
            "required": ["page_id"],
            "properties": {"page_id": {"type": "string"}},
        },
    },
    {
        "name": "browser_list_pages",
        "description": "List all open browser tabs with their URLs and titles.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

ASYNC_TOOL_MAP = {
    "browser_navigate":     browser_navigate,
    "browser_screenshot":   browser_screenshot,
    "browser_click":        browser_click,
    "browser_fill":         browser_fill,
    "browser_extract_text": browser_extract_text,
    "browser_get_html":     browser_get_html,
    "browser_evaluate":     browser_evaluate,
    "browser_new_page":     browser_new_page,
    "browser_close_page":   browser_close_page,
    "browser_list_pages":   browser_list_pages,
}


# ── Async MCP Stdio Loop ───────────────────────────────────────────────────────

def _write(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


async def handle_request(req: dict) -> None:
    req_id = req.get("id")
    method = req.get("method", "")

    if method == "initialize":
        _write({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo":      {"name": "playwright", "version": "1.0"},
            },
        })

    elif method == "tools/list":
        _write({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})

    elif method == "tools/call":
        params    = req.get("params", {})
        tool_name = params.get("name", "")
        args      = params.get("arguments", {})
        fn        = ASYNC_TOOL_MAP.get(tool_name)
        if fn is None:
            _write({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            })
            return
        try:
            result = await fn(**{k: v for k, v in args.items()})
            _write({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                },
            })
        except Exception as exc:
            _write({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}]
                },
            })

    elif method == "notifications/initialized":
        pass  # Ignore notification

    else:
        _write({
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })


async def main():
    """
    Async MCP stdio loop.
    Uses loop.run_in_executor for non-blocking stdin reads so the asyncio
    event loop remains free for Playwright's browser callbacks.
    """
    loop = asyncio.get_event_loop()

    while True:
        try:
            raw = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, OSError):
            break
        if not raw:
            break
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue
        await handle_request(req)

    # Cleanup
    global _browser, _playwright
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()


if __name__ == "__main__":
    asyncio.run(main())
