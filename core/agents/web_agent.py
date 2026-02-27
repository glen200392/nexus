"""
NEXUS Web Agent — Layer 4 Execution
Searches the internet and fetches web content.
Providers tried in order: Brave Search API → DuckDuckGo → direct fetch.
Always stores results to episodic memory.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.web")


class WebAgent(BaseAgent):
    agent_id   = "web_agent"
    agent_name = "Web Research Agent"
    description = "Searches the internet and fetches web content for research tasks"
    domain     = TaskDomain.RESEARCH
    default_complexity = TaskComplexity.MEDIUM

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a precise web research agent. Your task:\n"
            "1. Analyze the user's query to identify the key search terms\n"
            "2. Review the search results provided\n"
            "3. Extract and synthesize the most relevant information\n"
            "4. ALWAYS cite sources with [Source: URL]\n"
            "5. Be factual — if you're unsure, say so\n"
            "6. Return a structured JSON response with:\n"
            '   {"summary": "...", "key_findings": [...], '
            '"sources": [{"title": "...", "url": "...", "relevance": 0.9}]}'
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        query = context.user_message

        # Step 1: Perform web search
        self._logger.info("Searching: %s…", query[:60])
        search_results = await self._search(query)

        if not search_results:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=context.task_id,
                success=False,
                output=None,
                error="No search results found",
            )

        # Step 2: Synthesize with LLM
        decision = self.route_llm(context)
        results_text = self._format_results(search_results)

        messages = [
            Message("user",
                f"Query: {query}\n\n"
                f"Search results:\n{results_text}\n\n"
                "Synthesize these into a structured JSON response."
            )
        ]

        llm_resp = await self._llm.chat(
            messages=messages,
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
        )

        # Step 3: Parse response
        parsed = self._parse_llm_output(llm_resp.content)

        # Step 4: Store to memory
        await self.remember(
            content=json.dumps(parsed, ensure_ascii=False),
            context=context,
            doc_type=DocumentType.FACT,
            tags=["web_search", context.domain.value],
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=parsed,
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
            cost_usd=llm_resp.cost_usd,
            llm_used=decision.primary.display_name,
            artifacts=[
                {"type": "source", "url": s.get("url", ""), "title": s.get("title", "")}
                for s in parsed.get("sources", [])
            ],
        )

    # ── Search Providers (tried in order) ────────────────────────────────────

    async def _search(self, query: str) -> list[dict]:
        """Try providers in order until one succeeds."""
        providers = [
            self._search_brave,
            self._search_duckduckgo,
        ]
        for provider in providers:
            try:
                results = await provider(query)
                if results:
                    self._logger.debug(
                        "Search via %s: %d results", provider.__name__, len(results)
                    )
                    return results
            except Exception as exc:
                self._logger.debug("Provider %s failed: %s", provider.__name__, exc)
        return []

    async def _search_brave(self, query: str) -> list[dict]:
        """Brave Search API — best quality, requires BRAVE_API_KEY."""
        import os
        import httpx

        api_key = os.getenv("BRAVE_API_KEY", "")
        if not api_key:
            raise ValueError("No BRAVE_API_KEY set")

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 10, "text_decorations": False},
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            )
            r.raise_for_status()
            data = r.json()

        return [
            {
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("description", ""),
            }
            for item in data.get("web", {}).get("results", [])
        ]

    async def _search_duckduckgo(self, query: str) -> list[dict]:
        """DuckDuckGo via duckduckgo-search library — no API key needed."""
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=10):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results

    async def fetch_url(self, url: str) -> str:
        """Fetch and parse a single URL, returning clean text."""
        import httpx
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            html = r.text

        # Simple HTML → text extraction (no BS4 dependency required)
        text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]  # Limit to avoid huge token counts

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_results(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r['title']}\n"
                f"    URL: {r['url']}\n"
                f"    {r['snippet']}\n"
            )
        return "\n".join(lines)

    def _parse_llm_output(self, raw: str) -> dict:
        """Extract JSON from LLM response, fall back to plain text."""
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"summary": raw, "key_findings": [], "sources": []}
