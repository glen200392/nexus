"""
NEXUS Base Agent — Layer 4 Foundation
All specialized agents inherit from this class.
Handles: LLM routing, memory access, skill loading, MCP tool use, audit logging.
"""
from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from nexus.core.llm.router import (
    LLMRouter, RoutingRequest, PrivacyTier, TaskComplexity, TaskDomain,
    get_router,
)
from nexus.knowledge.rag.schema import (
    MemoryRecord, EpisodicRecord, MemoryType, DocumentType, RetrievalConfig, RetrievalMode
)

logger = logging.getLogger("nexus.agents")


# ─── Agent Context (working memory for one task) ─────────────────────────────

@dataclass
class AgentContext:
    """
    Short-lived working memory for a single task execution.
    Created by orchestrator, passed down to agents.
    """
    task_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id:    str = ""
    user_message:  str = ""
    history:       list[dict] = field(default_factory=list)  # [{role, content}]
    retrieved_memory: list[MemoryRecord] = field(default_factory=list)
    active_tools:  list[str] = field(default_factory=list)
    privacy_tier:  PrivacyTier = PrivacyTier.INTERNAL
    domain:        TaskDomain = TaskDomain.RESEARCH
    complexity:    TaskComplexity = TaskComplexity.MEDIUM
    metadata:      dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None   # set if this is a sub-task
    started_at:    float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def token_estimate(self) -> int:
        """Rough token estimate from history (4 chars ≈ 1 token)."""
        total_chars = sum(len(m["content"]) for m in self.history)
        return total_chars // 4


# ─── Agent Result ─────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    agent_id:      str
    task_id:       str
    success:       bool
    output:        Any                  # final output (str, dict, list…)
    quality_score: float = 0.5         # set by Critic Agent
    tokens_used:   int = 0
    cost_usd:      float = 0.0
    duration_ms:   int = 0
    llm_used:      str = ""
    error:         Optional[str] = None
    artifacts:     list[dict] = field(default_factory=list)  # files, URLs, code blocks
    memory_written: list[str] = field(default_factory=list)  # doc_ids written to memory


# ─── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent(abc.ABC):
    """
    Abstract base for all NEXUS agents.

    Subclass contract:
        • Override `agent_id`, `agent_name`, `domain`, `default_complexity`
        • Implement `_build_system_prompt(context) -> str`
        • Implement `execute(context) -> AgentResult`

    Provided automatically:
        • LLM routing via self.route_llm(context)
        • Memory read/write via self.remember() / self.recall()
        • Skill loading via self.use_skill(skill_name, **kwargs)
        • Audit logging
    """

    # ── Agent Identity (override in subclass) ─────────────────────────────────
    agent_id:          str = "base_agent"
    agent_name:        str = "Base Agent"
    description:       str = "Abstract base agent"
    domain:            TaskDomain = TaskDomain.RESEARCH
    default_complexity: TaskComplexity = TaskComplexity.MEDIUM
    default_privacy:   PrivacyTier = PrivacyTier.INTERNAL
    version:           str = "1.0.0"

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        memory_store=None,        # Injected RAG engine (duck-typed)
        skill_registry=None,      # Injected skill registry
        mcp_client=None,          # Injected MCP client
        config: Optional[dict] = None,
    ):
        self.router        = router or get_router()
        self.memory_store  = memory_store
        self.skill_registry = skill_registry
        self.mcp_client    = mcp_client
        self.config        = config or {}
        self._logger       = logging.getLogger(f"nexus.agents.{self.agent_id}")
        self._call_count   = 0
        self._total_cost   = 0.0

    # ── Core Interface ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def _build_system_prompt(self, context: AgentContext) -> str:
        """Build the system prompt for this agent given the current context."""
        ...

    @abc.abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute the main task. Called by the orchestrator.
        Returns AgentResult with output, quality, cost, artifacts.
        """
        ...

    # ── LLM Routing Helper ────────────────────────────────────────────────────

    def route_llm(
        self,
        context: AgentContext,
        complexity_override: Optional[TaskComplexity] = None,
        is_vision: bool = False,
    ):
        """Returns a RoutingDecision for this context."""
        return self.router.route(RoutingRequest(
            task_type=self.agent_id,
            domain=self.domain,
            complexity=complexity_override or context.complexity,
            privacy_tier=context.privacy_tier,
            is_vision=is_vision,
        ))

    # ── Memory Interface ──────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        context: AgentContext,
        config: Optional[RetrievalConfig] = None,
    ) -> list[MemoryRecord]:
        """Retrieve relevant memories for the current task."""
        if self.memory_store is None:
            return []
        cfg = config or RetrievalConfig(
            mode=RetrievalMode.HYBRID,
            top_k=5,
            filters={"privacy_tier": context.privacy_tier.value},
        )
        try:
            results = await self.memory_store.retrieve(query, config=cfg)
            self._logger.debug("Recalled %d memories for: %s…", len(results), query[:50])
            return results
        except Exception as exc:
            self._logger.warning("Memory recall failed: %s", exc)
            return []

    async def remember(
        self,
        content: str,
        context: AgentContext,
        doc_type: DocumentType = DocumentType.FACT,
        quality_score: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Write a memory record. Returns doc_id or None on failure."""
        if self.memory_store is None:
            return None
        record = MemoryRecord(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            doc_type=doc_type,
            source=f"agent:{self.agent_id}",
            agent_id=self.agent_id,
            task_id=context.task_id,
            session_id=context.session_id,
            domain=self.domain.value,
            privacy_tier=context.privacy_tier,
            quality_score=quality_score,
            tags=tags or [],
        )
        try:
            doc_id = await self.memory_store.upsert(record)
            self._logger.debug("Wrote memory: %s", doc_id)
            return doc_id
        except Exception as exc:
            self._logger.warning("Memory write failed: %s", exc)
            return None

    async def _record_episode(self, context: AgentContext, result: AgentResult) -> None:
        """Write episodic memory record after task completion."""
        if self.memory_store is None:
            return
        summary = (
            f"Task: {context.user_message[:200]}\n"
            f"Result: {'SUCCESS' if result.success else 'FAILURE'}\n"
            f"Output: {str(result.output)[:500]}"
        )
        record = EpisodicRecord(
            content=summary,
            source=f"agent:{self.agent_id}",
            agent_id=self.agent_id,
            task_id=context.task_id,
            session_id=context.session_id,
            domain=self.domain.value,
            privacy_tier=context.privacy_tier,
            quality_score=result.quality_score,
            agent_chain=[self.agent_id],
            llm_used=result.llm_used,
            duration_ms=result.duration_ms,
            success=result.success,
            cost_usd=result.cost_usd,
            tokens_used=result.tokens_used,
        )
        await self.memory_store.upsert(record)

    # ── Skill Interface ───────────────────────────────────────────────────────

    async def use_skill(self, skill_name: str, **kwargs) -> Any:
        """
        Execute a registered skill. Skills are Python modules in skills/implementations/.
        Example: await self.use_skill("excel_designer", data=df, template="report")
        """
        if self.skill_registry is None:
            raise RuntimeError("No skill registry injected")
        skill = self.skill_registry.get(skill_name)
        if skill is None:
            raise ValueError(f"Skill '{skill_name}' not found in registry")
        self._logger.info("Executing skill: %s", skill_name)
        return await skill.run(**kwargs)

    # ── MCP Tool Interface ────────────────────────────────────────────────────

    async def call_mcp_tool(self, server: str, tool: str, params: dict) -> Any:
        """
        Call an MCP server tool via the NEXUS MCP client.
        Example: await self.call_mcp_tool("github", "create_issue", {...})
        """
        if self.mcp_client is None:
            raise RuntimeError("No MCP client injected")
        self._logger.info("MCP call: %s/%s", server, tool)
        return await self.mcp_client.call(server=server, tool=tool, params=params)

    # ── Lifecycle Hooks ───────────────────────────────────────────────────────

    async def before_execute(self, context: AgentContext) -> None:
        """Override to add pre-execution logic (e.g., validate inputs)."""
        pass

    async def after_execute(self, context: AgentContext, result: AgentResult) -> None:
        """Override to add post-execution logic (e.g., cleanup, notifications)."""
        await self._record_episode(context, result)

    # ── Orchestrated Execute (called by Harness) ──────────────────────────────

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Entry point called by the Harness.
        Wraps execute() with lifecycle hooks, timing, and error handling.
        """
        self._call_count += 1
        start = time.time()
        self._logger.info("Agent %s starting task %s", self.agent_id, context.task_id)

        await self.before_execute(context)

        try:
            result = await self.execute(context)
        except Exception as exc:
            self._logger.exception("Agent %s failed: %s", self.agent_id, exc)
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=context.task_id,
                success=False,
                output=None,
                error=str(exc),
                duration_ms=int((time.time() - start) * 1000),
            )

        result.duration_ms = int((time.time() - start) * 1000)
        self._total_cost += result.cost_usd

        await self.after_execute(context, result)

        self._logger.info(
            "Agent %s done | success=%s | %dms | $%.4f",
            self.agent_id, result.success, result.duration_ms, result.cost_usd,
        )
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "agent_id":    self.agent_id,
            "version":     self.version,
            "call_count":  self._call_count,
            "total_cost":  self._total_cost,
            "domain":      self.domain.value,
        }
