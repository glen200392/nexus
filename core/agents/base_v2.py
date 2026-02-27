"""
NEXUS BaseAgentV2 — Enhanced Agent Foundation
Extends BaseAgent with:
  - Handoff support (transfer, consult, escalate)
  - Extended thinking (LLM with longer reasoning)
  - Streaming output
  - Tool execution stubs
  - Per-agent guardrail rules

All v1 agents continue to work; v2 agents get extra capabilities.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent

logger = logging.getLogger("nexus.agents.v2")


class BaseAgentV2(BaseAgent):
    """
    Enhanced base agent with handoff, streaming, and tool support.

    New class attributes (override in subclass):
      - tools: list of tool names this agent can use
      - handoff_targets: agent_ids this agent can hand off to
      - guardrail_rules: rule names to apply to this agent's I/O
      - supports_streaming: whether this agent supports streaming output
      - max_thinking_tokens: max tokens for extended thinking (0 = disabled)
    """

    # ── V2 Class Attributes ───────────────────────────────────────────────────
    tools: list[str] = []
    handoff_targets: list[str] = []
    guardrail_rules: list[str] = []
    supports_streaming: bool = False
    max_thinking_tokens: int = 0

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        config: Optional[dict] = None,
        llm_client=None,
        handoff_manager=None,
        **kwargs,
    ):
        # Pass through to BaseAgent
        super().__init__(config=config, **kwargs)

        # Override agent_id if provided
        if agent_id is not None:
            self.agent_id = agent_id

        self.llm_client = llm_client
        self.handoff_manager = handoff_manager

    # ── Handoff Interface ─────────────────────────────────────────────────────

    async def handoff(
        self,
        to_agent: str,
        context: dict,
        reason: str = "",
    ) -> AgentResult:
        """
        Hand off work to another agent via the HandoffManager.
        Returns AgentResult wrapping the handoff result.
        """
        if self.handoff_manager is None:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=context.get("task_id", ""),
                success=False,
                output=None,
                error="No handoff manager configured",
            )

        from nexus.core.orchestrator.handoff import HandoffRequest, HandoffType

        request = HandoffRequest(
            from_agent=self.agent_id,
            to_agent=to_agent,
            type=HandoffType.TRANSFER,
            context=context,
            reason=reason,
        )

        result = await self.handoff_manager.execute(request)

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.get("task_id", ""),
            success=result.success,
            output=result.output,
            error=result.error or None,
        )

    # ── Extended Thinking ─────────────────────────────────────────────────────

    async def think(self, prompt: str, system: str = "") -> str:
        """
        Use the LLM with extended thinking for deeper reasoning.
        Falls back to standard call if extended thinking is not supported.
        """
        if self.llm_client is None:
            return ""

        try:
            kwargs: dict[str, Any] = {
                "prompt": prompt,
            }
            if system:
                kwargs["system"] = system
            if self.max_thinking_tokens > 0:
                kwargs["max_thinking_tokens"] = self.max_thinking_tokens

            result = await self.llm_client(** kwargs)
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            self._logger.warning("Think failed: %s", exc)
            return ""

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(self, context: AgentContext) -> AsyncIterator[str]:
        """
        Stream the agent's output token by token.
        Default implementation runs execute() and yields the full output.
        Override in subclasses for true streaming.
        """
        result = await self.run(context)
        output = str(result.output) if result.output else ""
        if output:
            yield output

    # ── Tool Execution ────────────────────────────────────────────────────────

    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name. Stub implementation.
        Override in subclasses or inject a tool registry.
        """
        if tool_name not in self.tools:
            raise ValueError(
                f"Tool '{tool_name}' not available for agent '{self.agent_id}'. "
                f"Available tools: {self.tools}"
            )

        # Delegate to skill registry if available
        if self.skill_registry is not None:
            skill = self.skill_registry.get(tool_name)
            if skill is not None:
                return await skill.run(**kwargs)

        # Delegate to MCP client if available
        if self.mcp_client is not None:
            return await self.mcp_client.call(tool=tool_name, params=kwargs)

        raise RuntimeError(
            f"No execution backend for tool '{tool_name}'. "
            "Inject skill_registry or mcp_client."
        )

    # ── Default Implementations ───────────────────────────────────────────────

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Default system prompt for v2 agents."""
        return (
            f"You are {self.agent_name}, a NEXUS v2 agent.\n"
            f"Agent ID: {self.agent_id}\n"
            f"Domain: {self.domain.value}\n"
            f"Task: {context.user_message}\n"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Default execute implementation using LLM client.
        Override in subclasses for specialized behavior.
        """
        if self.llm_client is not None:
            system = self._build_system_prompt(context)
            output = await self.think(context.user_message, system=system)
            return AgentResult(
                agent_id=self.agent_id,
                task_id=context.task_id,
                success=True,
                output=output,
            )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=f"[{self.agent_id}] processed: {context.user_message[:100]}",
        )
