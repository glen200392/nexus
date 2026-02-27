"""
NEXUS HandoffManager — Inter-Agent Handoff Protocol
Manages transfers, consultations, escalations, and delegations between agents.

Handoff types:
  - TRANSFER: Full control transfer from one agent to another
  - CONSULT: Ask another agent a question, return answer to requester
  - ESCALATE: Move task up the supervision chain
  - DELEGATE: Assign a sub-task to a subordinate agent
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("nexus.orchestrator.handoff")


# ── Handoff Types ─────────────────────────────────────────────────────────────

class HandoffType(str, Enum):
    TRANSFER = "transfer"
    CONSULT = "consult"
    ESCALATE = "escalate"
    DELEGATE = "delegate"


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class HandoffRequest:
    """Request to hand off work between agents."""
    from_agent: str
    to_agent: str
    type: HandoffType
    context: dict
    reason: str
    depth: int = 0


@dataclass
class HandoffResult:
    """Result of a handoff operation."""
    success: bool
    output: str
    from_agent: str
    to_agent: str
    type: HandoffType
    elapsed_ms: float = 0.0
    error: str = ""


# ── Escalation Chain ─────────────────────────────────────────────────────────

_ESCALATION_MAP: dict[str, str] = {
    "code_agent": "ml_pipeline_agent",
    "data_analyst_agent": "ml_pipeline_agent",
    "web_agent": "browser_agent",
    "planner": "critic",
}

# Default escalation target for any agent not in the map
_DEFAULT_ESCALATION_TARGET = "planner"


# ── HandoffManager ───────────────────────────────────────────────────────────

class HandoffManager:
    """
    Manages inter-agent handoffs within a swarm or across swarms.

    Features:
      - Transfer: full control transfer (agent A stops, agent B takes over)
      - Consult: agent A asks agent B a question, gets answer back
      - Escalate: move task to a supervisor in the escalation chain
      - Delegate: assign a sub-task (same as transfer but preserves parent context)
      - Depth protection: prevents infinite handoff loops
    """

    def __init__(self, swarm_registry, max_depth: int = 5):
        self.swarm_registry = swarm_registry
        self.max_depth = max_depth
        self._handoff_log: list[HandoffResult] = []

    async def execute(self, request: HandoffRequest) -> HandoffResult:
        """
        Dispatch a handoff request based on its type.
        Returns HandoffResult with success/failure and output.
        """
        # Depth protection
        if request.depth >= self.max_depth:
            logger.warning(
                "Handoff depth limit reached (%d >= %d): %s → %s",
                request.depth, self.max_depth, request.from_agent, request.to_agent,
            )
            return HandoffResult(
                success=False,
                output="",
                from_agent=request.from_agent,
                to_agent=request.to_agent,
                type=request.type,
                error=f"Max handoff depth ({self.max_depth}) exceeded",
            )

        start = time.time()
        try:
            if request.type == HandoffType.TRANSFER:
                result = await self.transfer(
                    request.from_agent, request.to_agent,
                    request.context, request.reason,
                )
            elif request.type == HandoffType.CONSULT:
                question = request.context.get("question", request.reason)
                answer = await self.consult(
                    request.from_agent, request.to_agent,
                    question, request.context,
                )
                result = HandoffResult(
                    success=True,
                    output=answer,
                    from_agent=request.from_agent,
                    to_agent=request.to_agent,
                    type=HandoffType.CONSULT,
                )
            elif request.type == HandoffType.ESCALATE:
                result = await self.escalate(
                    request.from_agent, request.context, request.reason,
                )
            elif request.type == HandoffType.DELEGATE:
                # Delegate is similar to transfer but preserves parent context
                result = await self.transfer(
                    request.from_agent, request.to_agent,
                    request.context, request.reason,
                )
                result.type = HandoffType.DELEGATE
            else:
                result = HandoffResult(
                    success=False,
                    output="",
                    from_agent=request.from_agent,
                    to_agent=request.to_agent,
                    type=request.type,
                    error=f"Unknown handoff type: {request.type}",
                )
        except Exception as exc:
            logger.exception("Handoff failed: %s → %s (%s)", request.from_agent, request.to_agent, exc)
            result = HandoffResult(
                success=False,
                output="",
                from_agent=request.from_agent,
                to_agent=request.to_agent,
                type=request.type,
                error=str(exc),
            )

        result.elapsed_ms = (time.time() - start) * 1000
        self._handoff_log.append(result)
        return result

    async def transfer(
        self,
        from_agent: str,
        to_agent: str,
        context: dict,
        reason: str,
    ) -> HandoffResult:
        """
        Full control transfer from one agent to another.
        The target agent runs with the provided context.
        """
        agent = self._resolve_agent(to_agent)
        if agent is None:
            return HandoffResult(
                success=False,
                output="",
                from_agent=from_agent,
                to_agent=to_agent,
                type=HandoffType.TRANSFER,
                error=f"Agent '{to_agent}' not found in any registered swarm",
            )

        logger.info("Transfer: %s → %s (reason: %s)", from_agent, to_agent, reason)

        try:
            from nexus.core.agents.base import AgentContext
            agent_context = AgentContext(
                user_message=context.get("user_message", reason),
                metadata={
                    "handoff_from": from_agent,
                    "handoff_reason": reason,
                    **context,
                },
            )
            result = await agent.run(agent_context)
            return HandoffResult(
                success=result.success,
                output=str(result.output) if result.output else "",
                from_agent=from_agent,
                to_agent=to_agent,
                type=HandoffType.TRANSFER,
                error=result.error or "",
            )
        except Exception as exc:
            return HandoffResult(
                success=False,
                output="",
                from_agent=from_agent,
                to_agent=to_agent,
                type=HandoffType.TRANSFER,
                error=str(exc),
            )

    async def consult(
        self,
        requester: str,
        consultant: str,
        question: str,
        context: dict,
    ) -> str:
        """
        Ask another agent a question and return the answer.
        The requester retains control after receiving the answer.
        """
        agent = self._resolve_agent(consultant)
        if agent is None:
            raise ValueError(f"Consultant agent '{consultant}' not found in any registered swarm")

        logger.info("Consult: %s asking %s: %s", requester, consultant, question[:80])

        from nexus.core.agents.base import AgentContext
        agent_context = AgentContext(
            user_message=question,
            metadata={
                "consult_from": requester,
                **context,
            },
        )
        result = await agent.run(agent_context)
        return str(result.output) if result.output else ""

    async def escalate(
        self,
        from_agent: str,
        context: dict,
        reason: str,
    ) -> HandoffResult:
        """
        Escalate a task to the next agent in the supervision chain.
        Uses the escalation map to determine the target.
        """
        target = self._get_escalation_target(from_agent)
        if target is None:
            return HandoffResult(
                success=False,
                output="",
                from_agent=from_agent,
                to_agent="",
                type=HandoffType.ESCALATE,
                error=f"No escalation target for agent '{from_agent}'",
            )

        logger.info("Escalate: %s → %s (reason: %s)", from_agent, target, reason)
        return await self.transfer(from_agent, target, context, reason)

    def _get_escalation_target(self, from_agent: str) -> str | None:
        """
        Determine the escalation target for a given agent.

        Chain:
          - code_agent → ml_pipeline_agent
          - data_analyst_agent → ml_pipeline_agent
          - web_agent → browser_agent
          - planner → critic
          - any other → planner (default)
        """
        if from_agent in _ESCALATION_MAP:
            return _ESCALATION_MAP[from_agent]
        # Default: escalate to planner (unless already planner)
        return _DEFAULT_ESCALATION_TARGET

    def _resolve_agent(self, agent_id: str):
        """
        Find an agent across all registered swarms.
        Returns the agent instance or None.
        """
        if self.swarm_registry is None:
            return None

        # SwarmRegistry with to_dict() method
        swarms = {}
        if hasattr(self.swarm_registry, "to_dict"):
            swarms = self.swarm_registry.to_dict()
        elif hasattr(self.swarm_registry, "_swarms"):
            swarms = self.swarm_registry._swarms
        elif isinstance(self.swarm_registry, dict):
            swarms = self.swarm_registry

        for swarm in swarms.values():
            try:
                if hasattr(swarm, "get_agent"):
                    return swarm.get_agent(agent_id)
                elif isinstance(swarm, dict) and agent_id in swarm:
                    return swarm[agent_id]
            except (ValueError, KeyError):
                continue
        return None

    @property
    def handoff_log(self) -> list[HandoffResult]:
        """Return the handoff history."""
        return list(self._handoff_log)
