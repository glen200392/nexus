"""
NEXUS Test Configuration — Shared Fixtures
Provides mock LLM, mock MCP, test event factory, and other shared fixtures.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure nexus package is importable
_ROOT = Path(__file__).resolve().parent.parent
_PARENT = _ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


# ── Mock LLM Response ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    from nexus.core.llm.client import LLMResponse

    def _make(content="Mock response", cost=0.001, tokens_in=100, tokens_out=50):
        return LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model_used="mock/test-model",
            finish_reason="stop",
        )
    return _make


@pytest.fixture
def mock_llm_caller():
    """Async callable that simulates LLM perception analysis."""
    async def _caller(prompt: str, system: str, model: str = "qwen2.5:7b") -> str:
        return '{"intent": "test task", "task_type": "research", "domain": "research", "complexity": "medium", "privacy_tier": "INTERNAL", "required_agents": ["web_agent"], "required_skills": [], "key_entities": [], "memory_queries": ["test"], "requires_confirmation": false, "is_destructive": false, "language": "en"}'
    return _caller


# ── Mock Agent ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_agent():
    """Factory for creating mock agents."""
    from nexus.core.agents.base import AgentResult

    def _make(agent_id="test_agent", success=True, output="Test output", quality=0.8, cost=0.01):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.run = AsyncMock(return_value=AgentResult(
            agent_id=agent_id,
            task_id="test-task",
            success=success,
            output=output,
            quality_score=quality,
            cost_usd=cost,
        ))
        return agent
    return _make


# ── Mock Swarm ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_swarm(mock_agent):
    """Create a mock swarm with configurable agents."""
    def _make(swarm_id="test_swarm", agent_ids=None):
        swarm = MagicMock()
        swarm.swarm_id = swarm_id

        agents = {}
        for aid in (agent_ids or ["agent_a", "agent_b"]):
            agents[aid] = mock_agent(agent_id=aid)

        swarm.get_agent = MagicMock(side_effect=lambda aid: agents.get(aid, mock_agent(agent_id=aid)))

        from nexus.core.agents.base import AgentContext
        swarm.build_context = MagicMock(return_value=AgentContext(
            user_message="test message",
        ))

        swarm.merge_parallel_results = MagicMock(return_value={
            "findings": ["finding1", "finding2"],
            "sources": [],
            "errors": [],
            "best_output": "merged output",
            "quality_score": 0.85,
        })

        return swarm
    return _make


# ── Task Event Factory ────────────────────────────────────────────────────────

@pytest.fixture
def event_factory():
    """Factory for creating test TaskEvents."""
    from nexus.core.orchestrator.trigger import TaskEvent, TriggerSource, TriggerPriority

    def _make(
        raw_input="Test task",
        priority=TriggerPriority.NORMAL,
        source=TriggerSource.CLI,
        hint_domain=None,
        hint_privacy=None,
    ):
        return TaskEvent(
            event_id=str(uuid.uuid4()),
            source=source,
            priority=priority,
            raw_input=raw_input,
            hint_domain=hint_domain,
            hint_privacy=hint_privacy,
            session_id="test-session",
        )
    return _make


# ── Governance Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_audit_db(tmp_path):
    """Temporary audit database for testing."""
    return tmp_path / "test_audit.db"


@pytest.fixture
def governance_manager(tmp_audit_db):
    """GovernanceManager with temporary database."""
    from nexus.core.governance import GovernanceManager
    return GovernanceManager(db_path=tmp_audit_db)
