"""
NEXUS RAG Agent — Layer 4 + Layer 5 Bridge
Queries the local knowledge base and returns relevant context.
This agent is the primary bridge between Execution and Memory layers.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity
from nexus.knowledge.rag.engine import RAGEngine, get_rag_engine
from nexus.knowledge.rag.schema import (
    DocumentType, MemoryRecord, RetrievalConfig, RetrievalMode
)

logger = logging.getLogger("nexus.agents.rag")


class RAGAgent(BaseAgent):
    agent_id   = "rag_agent"
    agent_name = "RAG Knowledge Agent"
    description = "Retrieves relevant knowledge from the local vector store"
    domain     = TaskDomain.RESEARCH
    default_complexity = TaskComplexity.LOW  # Retrieval itself is cheap

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rag_engine: Optional[RAGEngine] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()
        self._rag = rag_engine or get_rag_engine()

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a knowledge retrieval and synthesis agent with access to a local "
            "knowledge base. Your task:\n"
            "1. Review the retrieved documents carefully\n"
            "2. Identify which documents are most relevant to the query\n"
            "3. Synthesize a coherent, accurate answer based ONLY on the retrieved content\n"
            "4. For every claim, cite the source document [doc_id: X]\n"
            "5. If the knowledge base doesn't contain relevant information, say so explicitly\n"
            "6. Return structured JSON:\n"
            '   {"answer": "...", "confidence": 0.8, '
            '"citations": [{"doc_id": "...", "excerpt": "...", "relevance": 0.9}], '
            '"knowledge_gaps": ["what is missing"]}'
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        query = context.user_message

        # Step 1: Build retrieval queries (decompose query for better recall)
        queries = await self._expand_queries(query, context)

        # Step 2: Retrieve from all query variations
        all_records: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        retrieval_cfg = RetrievalConfig(
            mode=RetrievalMode.HYBRID,
            top_k=6,
            score_threshold=0.55,
            apply_temporal_decay=True,
        )

        for q in queries:
            records = await self._rag.retrieve(
                query=q,
                config=retrieval_cfg,
                privacy_tier=context.privacy_tier,
            )
            for r in records:
                if r.doc_id not in seen_ids:
                    all_records.append(r)
                    seen_ids.add(r.doc_id)

        if not all_records:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=context.task_id,
                success=True,
                output={
                    "answer": "知識庫中沒有找到相關資料。",
                    "confidence": 0.0,
                    "citations": [],
                    "knowledge_gaps": [query],
                },
                quality_score=0.3,
            )

        # Step 3: LLM synthesis over retrieved documents
        context_text = self._format_records(all_records)
        decision = self.route_llm(context)

        messages = [
            Message(
                "user",
                f"Query: {query}\n\n"
                f"Retrieved knowledge ({len(all_records)} documents):\n\n"
                f"{context_text}\n\n"
                "Synthesize an answer from the above knowledge."
            )
        ]

        llm_resp = await self._llm.chat(
            messages=messages,
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
        )

        parsed = self._parse_response(llm_resp.content)

        # Step 4: Update access counts (mark these docs as used)
        for rec in all_records:
            rec.access_count += 1

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=parsed,
            quality_score=parsed.get("confidence", 0.5),
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
            cost_usd=llm_resp.cost_usd,
            llm_used=decision.primary.display_name,
            artifacts=[
                {"type": "citation", "doc_id": c.get("doc_id", ""), "excerpt": c.get("excerpt", "")}
                for c in parsed.get("citations", [])
            ],
        )

    # ── Document Ingestion (used by orchestrator to add knowledge) ────────────

    async def ingest(
        self,
        content: str,
        source: str,
        domain: str = "general",
        privacy_tier=None,
    ) -> list[str]:
        """Ingest a document into the knowledge base. Returns doc_ids."""
        from nexus.knowledge.rag.schema import PrivacyTier as PT
        tier = privacy_tier or PT.INTERNAL
        return await self._rag.ingest_document(
            content=content,
            source=source,
            domain=domain,
            privacy_tier=tier,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _expand_queries(self, query: str, context: AgentContext) -> list[str]:
        """
        Generate 2-3 query variations for better recall.
        Uses fast local model for this cheap task.
        """
        from nexus.core.llm.router import MODEL_REGISTRY, TaskComplexity
        fast_model = MODEL_REGISTRY["qwen2.5:7b"]
        try:
            r = await self._llm.chat(
                messages=[Message(
                    "user",
                    f"Generate 3 different search queries for retrieving information about:\n"
                    f'"{query}"\n\n'
                    "Return ONLY a JSON array of 3 strings, no explanation:\n"
                    '["query1", "query2", "query3"]'
                )],
                model=fast_model,
                system="You are a query expansion engine. Output only valid JSON arrays.",
                privacy_tier=context.privacy_tier,
                max_tokens=200,
                temperature=0.3,
            )
            arr = json.loads(re.search(r"\[.*\]", r.content, re.DOTALL).group())
            return [query] + [str(q) for q in arr[:2]]
        except Exception:
            return [query]  # fallback: just use original

    def _format_records(self, records: list[MemoryRecord]) -> str:
        parts = []
        for i, r in enumerate(records, 1):
            source_line = f"[Source: {r.source}]" if r.source else ""
            parts.append(
                f"[Doc {i} | id:{r.doc_id[:8]} | score:{r.quality_score:.2f}] {source_line}\n"
                f"{r.content[:600]}\n"
            )
        return "\n---\n".join(parts)

    def _parse_response(self, raw: str) -> dict:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {
            "answer": raw,
            "confidence": 0.5,
            "citations": [],
            "knowledge_gaps": [],
        }
