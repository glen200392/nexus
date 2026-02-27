"""
NEXUS Writer Agent — Final Output Synthesizer
Takes multiple agent outputs and synthesizes a polished final response.
Adapts tone, format, and length to match the request type.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.writer")


class WriterAgent(BaseAgent):
    agent_id   = "writer_agent"
    agent_name = "Synthesis & Writer Agent"
    description = "Synthesizes multi-source findings into polished output"
    domain     = TaskDomain.CREATIVE
    default_complexity = TaskComplexity.MEDIUM

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        output_format = context.metadata.get("output_format", "markdown")
        language      = context.metadata.get("language", "zh-TW")
        tone          = context.metadata.get("tone", "professional")

        return (
            f"You are an expert technical writer. Your job is to synthesize multiple "
            f"research findings into a single, coherent, high-quality response.\n\n"
            f"Requirements:\n"
            f"- Language: {language}\n"
            f"- Format: {output_format}\n"
            f"- Tone: {tone}\n"
            f"- Always cite sources when making factual claims\n"
            f"- Organize information logically with clear headings\n"
            f"- Be comprehensive but avoid redundancy\n"
            f"- Start with an executive summary, then details\n"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        # Collect all prior agent outputs from conversation history
        agent_outputs = self._extract_agent_outputs(context)
        original_query = context.user_message

        if not agent_outputs:
            # Nothing to synthesize — just answer directly
            agent_outputs = [{"source": "direct", "content": original_query}]

        decision = self.route_llm(context)

        synthesis_input = self._format_inputs(original_query, agent_outputs)
        messages = [Message("user", synthesis_input)]

        llm_resp = await self._llm.chat(
            messages=messages,
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.5,
        )

        # Store final synthesis to memory
        await self.remember(
            content=f"Q: {original_query}\n\nA: {llm_resp.content[:2000]}",
            context=context,
            doc_type=DocumentType.SUMMARY,
            quality_score=0.7,
            tags=["synthesis", "final_answer", context.domain.value],
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=llm_resp.content,
            quality_score=0.75,   # Will be overridden by Critic
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
            cost_usd=llm_resp.cost_usd,
            llm_used=decision.primary.display_name,
        )

    def _extract_agent_outputs(self, context: AgentContext) -> list[dict]:
        outputs = []
        for msg in context.history:
            if msg["role"] == "assistant":
                content = msg["content"]
                # Try to parse as JSON (structured agent output)
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        outputs.append({"source": "agent", "content": content, "data": data})
                        continue
                except Exception:
                    pass
                outputs.append({"source": "agent", "content": content})
        return outputs

    def _format_inputs(self, query: str, outputs: list[dict]) -> str:
        parts = [f"# Original Query\n{query}\n"]
        for i, out in enumerate(outputs, 1):
            src = out.get("source", f"source_{i}")
            content = out.get("content", "")
            # If structured data, pretty-print key parts
            data = out.get("data", {})
            if "summary" in data:
                content = data["summary"]
            elif "answer" in data:
                content = data["answer"]
            elif "key_findings" in data:
                content = "\n".join(f"- {f}" for f in data["key_findings"])
            parts.append(f"## Input from {src}\n{content[:1500]}\n")
        parts.append("# Task\nSynthesize all the above into a comprehensive, "
                     "well-structured final response.")
        return "\n".join(parts)
