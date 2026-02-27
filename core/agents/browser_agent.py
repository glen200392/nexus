"""
NEXUS Browser Agent — Automated Web Browsing
Full browser automation using Playwright (local Chrome/Chromium).
Falls back to httpx + HTML parsing for simple fetch tasks.

Capabilities:
  - Navigate to URLs, click elements, fill forms
  - Extract structured content from pages
  - Take screenshots (useful for visual analysis tasks)
  - Handle JavaScript-rendered pages (SPA, React apps)
  - Login flows (with credential management)

Privacy: page content stays local; screenshots are in-memory.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.browser")


class BrowserAgent(BaseAgent):
    agent_id   = "browser_agent"
    agent_name = "Browser Automation Agent"
    description = (
        "Automates a real web browser: navigate, click, fill forms, extract content, screenshot"
    )
    domain     = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.MEDIUM

    def __init__(self, llm_client: Optional[LLMClient] = None, headless: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._llm      = llm_client or get_client()
        self._headless = headless
        self._playwright = None   # lazy-initialized
        self._browser    = None
        self._page       = None

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a browser automation agent. Given a task and page content, "
            "decide what browser actions to take.\n\n"
            "Available actions (return as JSON array of action objects):\n"
            '  {"action": "navigate",    "url": "https://..."}\n'
            '  {"action": "click",       "selector": "CSS or text selector"}\n'
            '  {"action": "fill",        "selector": "input selector", "value": "text"}\n'
            '  {"action": "extract",     "selector": "CSS selector", "attribute": "text|href|src"}\n'
            '  {"action": "screenshot",  "name": "description"}\n'
            '  {"action": "scroll",      "direction": "down|up", "amount": 3}\n'
            '  {"action": "wait",        "ms": 1000}\n'
            '  {"action": "done",        "summary": "what was accomplished"}\n\n'
            "Return ONLY a JSON object:\n"
            '{"plan": "what you will do", "actions": [...]}'
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        task = context.user_message
        url  = context.metadata.get("url", "")

        # Extract URL from task if not explicitly provided
        if not url:
            urls = re.findall(r"https?://[^\s\"']+", task)
            url = urls[0] if urls else ""

        # Phase 1: Get initial page content (fast fetch, no browser yet)
        initial_content = ""
        if url:
            initial_content = await self._fast_fetch(url)

        # Phase 2: LLM plans the action sequence
        decision = self.route_llm(context)
        page_summary = initial_content[:2000] if initial_content else "Page not yet loaded"

        llm_resp = await self._llm.chat(
            messages=[
                Message("user",
                    f"Task: {task}\n"
                    f"Starting URL: {url or 'none'}\n\n"
                    f"Current page content preview:\n{page_summary}\n\n"
                    "Plan the browser actions to complete this task."
                )
            ],
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.1,
        )

        plan_data = self._parse_plan(llm_resp.content)
        actions   = plan_data.get("actions", [])

        if not actions:
            # No browser actions needed — just return fetched content
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={"content": initial_content, "url": url, "actions_taken": []},
                tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
                cost_usd=llm_resp.cost_usd,
                llm_used=decision.primary.display_name,
            )

        # Phase 3: Execute browser actions
        artifacts    = []
        results_log  = []
        total_cost   = llm_resp.cost_usd
        final_content = initial_content

        try:
            browser_available = await self._ensure_browser()
            if not browser_available:
                # Fall back to httpx-only mode
                return await self._httpx_fallback(task, url, context, decision, initial_content)

            if url:
                await self._page.goto(url, wait_until="networkidle", timeout=15000)

            for action in actions:
                result = await self._execute_action(action)
                results_log.append(result)

                if result.get("type") == "screenshot":
                    artifacts.append(result)
                elif result.get("type") == "done":
                    break

            final_content = await self._page.content()

        except Exception as exc:
            logger.warning("Browser execution error: %s; falling back to httpx", exc)
            return await self._httpx_fallback(task, url, context, decision, initial_content)
        finally:
            await self._close_browser()

        # Phase 4: LLM summarizes what was accomplished
        summary_resp = await self._llm.chat(
            messages=[
                Message("user", f"Task: {task}"),
                Message("assistant", json.dumps(plan_data)),
                Message("user",
                    f"Actions completed. Final page summary:\n{final_content[:3000]}\n\n"
                    "Summarize what was accomplished and any key information extracted."
                ),
            ],
            model=decision.primary,
            system="You are a helpful browser assistant. Summarize what happened concisely.",
            privacy_tier=context.privacy_tier,
        )
        total_cost += summary_resp.cost_usd

        # Store extracted content to memory
        if final_content:
            await self.remember(
                content=f"URL: {url}\nTask: {task}\nContent: {final_content[:1000]}",
                context=context,
                doc_type=DocumentType.DOCUMENT,
                tags=["browser", "web_content", url.split("/")[2] if "/" in url else "web"],
            )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output={
                "summary":      summary_resp.content,
                "url":          url,
                "actions_taken": results_log,
                "content_length": len(final_content),
            },
            tokens_used=(llm_resp.tokens_in + llm_resp.tokens_out +
                         summary_resp.tokens_in + summary_resp.tokens_out),
            cost_usd=total_cost,
            llm_used=decision.primary.display_name,
            artifacts=artifacts,
        )

    # ── Browser Control ───────────────────────────────────────────────────────

    async def _ensure_browser(self) -> bool:
        """Initialize Playwright browser. Returns False if not installed."""
        try:
            from playwright.async_api import async_playwright
            if self._playwright is None:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self._headless,
                    args=["--no-sandbox", "--disable-gpu"],
                )
                self._page = await self._browser.new_page()
                # Set reasonable viewport
                await self._page.set_viewport_size({"width": 1280, "height": 900})
            return True
        except ImportError:
            logger.warning("playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
        except Exception as exc:
            logger.warning("Browser init failed: %s", exc)
            return False

    async def _close_browser(self):
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        finally:
            self._browser    = None
            self._playwright = None
            self._page       = None

    async def _execute_action(self, action: dict) -> dict:
        """Execute a single browser action and return result."""
        action_type = action.get("action", "")

        try:
            if action_type == "navigate":
                await self._page.goto(
                    action["url"], wait_until="networkidle", timeout=15000
                )
                return {"type": "navigate", "url": action["url"], "success": True}

            elif action_type == "click":
                await self._page.click(action["selector"], timeout=5000)
                await self._page.wait_for_load_state("networkidle", timeout=5000)
                return {"type": "click", "selector": action["selector"], "success": True}

            elif action_type == "fill":
                await self._page.fill(action["selector"], action.get("value", ""))
                return {"type": "fill", "selector": action["selector"], "success": True}

            elif action_type == "extract":
                attr = action.get("attribute", "text")
                elements = await self._page.query_selector_all(action.get("selector", "body"))
                texts = []
                for el in elements[:20]:
                    if attr == "text":
                        texts.append(await el.inner_text())
                    elif attr == "href":
                        texts.append(await el.get_attribute("href") or "")
                    else:
                        texts.append(await el.get_attribute(attr) or "")
                return {"type": "extract", "data": texts, "success": True}

            elif action_type == "screenshot":
                screenshot_bytes = await self._page.screenshot(
                    type="png", full_page=False
                )
                b64 = base64.b64encode(screenshot_bytes).decode()
                return {
                    "type": "screenshot",
                    "name": action.get("name", "screenshot"),
                    "data": b64,
                    "mime": "image/png",
                    "success": True,
                }

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                amount    = int(action.get("amount", 3)) * 300
                delta_y   = amount if direction == "down" else -amount
                await self._page.mouse.wheel(0, delta_y)
                return {"type": "scroll", "direction": direction, "success": True}

            elif action_type == "wait":
                ms = int(action.get("ms", 1000))
                await asyncio.sleep(ms / 1000)
                return {"type": "wait", "ms": ms, "success": True}

            elif action_type == "done":
                return {"type": "done", "summary": action.get("summary", ""), "success": True}

            return {"type": action_type, "success": False, "error": "Unknown action"}

        except Exception as exc:
            logger.warning("Action %s failed: %s", action_type, exc)
            return {"type": action_type, "success": False, "error": str(exc)}

    # ── Fallback (httpx, no browser required) ────────────────────────────────

    async def _fast_fetch(self, url: str) -> str:
        """Quick HTTP fetch + HTML strip, no browser required."""
        try:
            import httpx
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            }
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                html = r.text
            # Strip tags
            text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", text).strip()[:10000]
        except Exception as exc:
            logger.debug("Fast fetch failed: %s", exc)
            return ""

    async def _httpx_fallback(
        self, task: str, url: str, context: AgentContext, decision, content: str
    ) -> AgentResult:
        """Return result using only fetched content, no browser."""
        if not content and url:
            content = await self._fast_fetch(url)
        output = {
            "summary": f"Fetched {url} (browser unavailable, used httpx).\n{content[:2000]}",
            "url": url,
            "actions_taken": [],
            "note": "Playwright unavailable. Install with: pip install playwright && playwright install chromium",
        }
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=bool(content), output=output,
            llm_used=decision.primary.display_name,
        )

    def _parse_plan(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"plan": raw[:200], "actions": []}
