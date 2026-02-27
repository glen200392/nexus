"""Tests for BaseAgentV2 — enhanced agent foundation."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from nexus.core.agents.base import AgentContext, AgentResult
from nexus.core.agents.base_v2 import BaseAgentV2


# ── Concrete Subclass for Testing ─────────────────────────────────────────────

class ConcreteAgentV2(BaseAgentV2):
    """Concrete subclass of BaseAgentV2 for testing."""
    agent_id = "test_v2_agent"
    agent_name = "Test V2 Agent"
    description = "A testable v2 agent"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_class_attributes_defaults():
    """V2 agents should have sensible default class attributes."""
    assert BaseAgentV2.tools == []
    assert BaseAgentV2.handoff_targets == []
    assert BaseAgentV2.guardrail_rules == []
    assert BaseAgentV2.supports_streaming is False
    assert BaseAgentV2.max_thinking_tokens == 0


def test_instance_creation():
    """V2 agent can be created with keyword arguments."""
    agent = ConcreteAgentV2(agent_id="custom_id")
    assert agent.agent_id == "custom_id"
    assert agent.llm_client is None
    assert agent.handoff_manager is None


def test_custom_class_attributes():
    """Subclasses can override class attributes."""

    class SpecialAgent(BaseAgentV2):
        agent_id = "special"
        agent_name = "Special Agent"
        tools = ["web_search", "calculator"]
        handoff_targets = ["expert_agent"]
        guardrail_rules = ["pii_email"]
        supports_streaming = True
        max_thinking_tokens = 4096

    agent = SpecialAgent()
    assert agent.tools == ["web_search", "calculator"]
    assert agent.handoff_targets == ["expert_agent"]
    assert agent.guardrail_rules == ["pii_email"]
    assert agent.supports_streaming is True
    assert agent.max_thinking_tokens == 4096


@pytest.mark.asyncio
async def test_handoff_delegates_to_manager():
    """handoff() should delegate to the injected HandoffManager."""
    mock_manager = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.output = "handoff result"
    mock_result.error = ""
    mock_manager.execute = AsyncMock(return_value=mock_result)

    agent = ConcreteAgentV2(handoff_manager=mock_manager)
    result = await agent.handoff("expert", {"task_id": "t1", "user_message": "help"}, reason="need expertise")

    assert result.success is True
    assert result.output == "handoff result"
    mock_manager.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_handoff_without_manager():
    """handoff() without a manager should return error."""
    agent = ConcreteAgentV2()
    result = await agent.handoff("expert", {"task_id": "t1"})

    assert result.success is False
    assert "No handoff manager" in result.error


@pytest.mark.asyncio
async def test_think_calls_llm():
    """think() should call the LLM client with the prompt."""
    mock_llm = AsyncMock(return_value="deep thought")
    agent = ConcreteAgentV2(llm_client=mock_llm)

    result = await agent.think("what is life?", system="You are a philosopher.")
    assert result == "deep thought"
    mock_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_think_without_llm():
    """think() without LLM client should return empty string."""
    agent = ConcreteAgentV2()
    result = await agent.think("question")
    assert result == ""


@pytest.mark.asyncio
async def test_stream_yields_output():
    """stream() should yield the agent's output."""
    agent = ConcreteAgentV2()
    context = AgentContext(user_message="test message")

    chunks = []
    async for chunk in agent.stream(context):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "test_v2_agent" in chunks[0] or "test message" in chunks[0]


@pytest.mark.asyncio
async def test_execute_with_llm():
    """execute() with LLM client should call think() and return result."""
    mock_llm = AsyncMock(return_value="LLM output")
    agent = ConcreteAgentV2(llm_client=mock_llm)
    context = AgentContext(user_message="analyze this")

    result = await agent.execute(context)
    assert result.success is True
    assert result.output == "LLM output"


@pytest.mark.asyncio
async def test_execute_without_llm():
    """execute() without LLM should return a default processed message."""
    agent = ConcreteAgentV2()
    context = AgentContext(user_message="analyze this")

    result = await agent.execute(context)
    assert result.success is True
    assert "test_v2_agent" in result.output
