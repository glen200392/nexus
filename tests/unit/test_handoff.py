"""Tests for HandoffManager — inter-agent handoff protocol."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from nexus.core.orchestrator.handoff import (
    HandoffManager,
    HandoffRequest,
    HandoffResult,
    HandoffType,
)
from nexus.core.agents.base import AgentResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mock_agent(agent_id: str, output: str = "done", success: bool = True):
    """Create a mock agent that returns a predictable AgentResult."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.run = AsyncMock(return_value=AgentResult(
        agent_id=agent_id,
        task_id="test-task",
        success=success,
        output=output,
    ))
    return agent


def _make_mock_swarm(agents: dict):
    """Create a mock swarm with the given agents."""
    swarm = MagicMock()
    swarm.get_agent = MagicMock(side_effect=lambda aid: agents.get(aid) or (_ for _ in ()).throw(ValueError(f"Not found: {aid}")))
    return swarm


def _make_registry(swarm_agents: dict):
    """Create a mock swarm registry with a single swarm containing given agents."""
    swarm = _make_mock_swarm(swarm_agents)
    registry = MagicMock()
    registry.to_dict.return_value = {"test_swarm": swarm}
    return registry


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_transfer_success():
    """Transfer should call the target agent and return its output."""
    agent_a = _make_mock_agent("agent_a", output="result from A")
    agent_b = _make_mock_agent("agent_b", output="result from B")
    registry = _make_registry({"agent_a": agent_a, "agent_b": agent_b})

    manager = HandoffManager(swarm_registry=registry, max_depth=5)
    result = await manager.transfer("agent_a", "agent_b", {"user_message": "hello"}, "need help")

    assert result.success is True
    assert result.output == "result from B"
    assert result.from_agent == "agent_a"
    assert result.to_agent == "agent_b"
    assert result.type == HandoffType.TRANSFER
    agent_b.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_consult_returns_output():
    """Consult should call the consultant agent and return its string output."""
    consultant = _make_mock_agent("expert", output="42 is the answer")
    registry = _make_registry({"expert": consultant})

    manager = HandoffManager(swarm_registry=registry)
    answer = await manager.consult("asker", "expert", "what is the answer?", {})

    assert answer == "42 is the answer"
    consultant.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_escalate_follows_chain():
    """Escalate from code_agent should go to ml_pipeline_agent."""
    target_agent = _make_mock_agent("ml_pipeline_agent", output="escalated result")
    registry = _make_registry({"ml_pipeline_agent": target_agent})

    manager = HandoffManager(swarm_registry=registry)
    result = await manager.escalate("code_agent", {"user_message": "need help"}, "too complex")

    assert result.success is True
    assert result.output == "escalated result"
    assert result.to_agent == "ml_pipeline_agent"


@pytest.mark.asyncio
async def test_depth_limit():
    """Handoffs exceeding max_depth should fail with an error."""
    registry = _make_registry({})
    manager = HandoffManager(swarm_registry=registry, max_depth=5)

    request = HandoffRequest(
        from_agent="agent_a",
        to_agent="agent_b",
        type=HandoffType.TRANSFER,
        context={},
        reason="loop",
        depth=5,
    )
    result = await manager.execute(request)

    assert result.success is False
    assert "depth" in result.error.lower()


@pytest.mark.asyncio
async def test_unknown_agent():
    """Transfer to unknown agent should return error gracefully."""
    registry = _make_registry({})
    manager = HandoffManager(swarm_registry=registry)

    result = await manager.transfer("agent_a", "nonexistent", {}, "test")

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_escalation_targets():
    """Verify each escalation chain mapping."""
    manager = HandoffManager(swarm_registry=None)

    assert manager._get_escalation_target("code_agent") == "ml_pipeline_agent"
    assert manager._get_escalation_target("data_analyst_agent") == "ml_pipeline_agent"
    assert manager._get_escalation_target("web_agent") == "browser_agent"
    assert manager._get_escalation_target("planner") == "critic"
    # Any unknown agent defaults to planner
    assert manager._get_escalation_target("random_agent") == "planner"


@pytest.mark.asyncio
async def test_execute_dispatches_by_type():
    """execute() should dispatch to the correct handler based on HandoffType."""
    agent = _make_mock_agent("target", output="executed")
    registry = _make_registry({"target": agent})
    manager = HandoffManager(swarm_registry=registry)

    request = HandoffRequest(
        from_agent="source",
        to_agent="target",
        type=HandoffType.TRANSFER,
        context={"user_message": "test"},
        reason="test",
        depth=0,
    )
    result = await manager.execute(request)

    assert result.success is True
    assert result.elapsed_ms >= 0
    assert len(manager.handoff_log) == 1
