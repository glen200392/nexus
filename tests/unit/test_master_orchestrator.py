"""
Tests for MasterOrchestrator — Layer 3 Core
Tests all 6 workflow patterns and bug fixes.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nexus.core.orchestrator.master import (
    MasterOrchestrator, OrchestratedTask, ResourcePool,
    WorkflowPattern, TaskStatus, Harness,
)


# ── Workflow Planning ──────────────────────────────────────────────────────────

class TestPlanWorkflow:
    """Test _plan_workflow routing logic."""

    def setup_method(self):
        self.orch = MasterOrchestrator()

    def test_research_task_gets_parallel_pattern(self):
        """Research tasks should use PARALLEL pattern."""
        perceived = {"task_type": "research", "domain": "research", "user_message": "test"}
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.PARALLEL

    def test_code_task_gets_feedback_loop(self):
        """Code tasks should use FEEDBACK_LOOP for quality."""
        perceived = {"task_type": "code", "domain": "engineering", "user_message": "fix bug"}
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.FEEDBACK_LOOP
        assert task.quality_threshold == 0.8

    def test_decision_task_gets_adversarial(self):
        """Decision tasks should use ADVERSARIAL pattern."""
        perceived = {"task_type": "decision", "domain": "research", "user_message": "decide"}
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.ADVERSARIAL

    def test_critical_complexity_gets_hierarchical(self):
        """Critical complexity should trigger HIERARCHICAL delegation."""
        perceived = {"task_type": "general", "complexity": "critical", "user_message": "test"}
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.HIERARCHICAL

    def test_default_gets_sequential(self):
        """Default tasks should use SEQUENTIAL."""
        perceived = {"task_type": "general", "complexity": "low", "user_message": "hello"}
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.SEQUENTIAL

    def test_swarm_assignment(self):
        """Domain should map to correct swarm."""
        perceived = {"task_type": "general", "domain": "engineering", "user_message": "test"}
        task = self.orch._plan_workflow(perceived)
        assert task.assigned_swarm == "engineering_swarm"

    def test_unknown_domain_falls_back_to_research(self):
        """Unknown domain should fall back to research_swarm."""
        perceived = {"task_type": "general", "domain": "unknown_domain", "user_message": "test"}
        task = self.orch._plan_workflow(perceived)
        assert task.assigned_swarm == "research_swarm"


class TestBug1EnumConversion:
    """Bug 1: enum objects passed from PerceivedTask.__dict__ should work."""

    def setup_method(self):
        self.orch = MasterOrchestrator()

    def test_enum_domain_is_handled(self):
        """Enum domain values should be converted to strings."""
        from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier

        perceived = {
            "task_type": "research",
            "domain": TaskDomain.RESEARCH,  # enum, not string!
            "complexity": TaskComplexity.MEDIUM,
            "privacy_tier": PrivacyTier.INTERNAL,
            "user_message": "test enum handling",
        }
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.PARALLEL
        assert task.assigned_swarm == "research_swarm"

    def test_string_values_still_work(self):
        """Plain string values should continue to work."""
        perceived = {
            "task_type": "code",
            "domain": "engineering",
            "complexity": "high",
            "user_message": "test string values",
        }
        task = self.orch._plan_workflow(perceived)
        assert task.workflow_pattern == WorkflowPattern.FEEDBACK_LOOP


# ── Workflow Execution ─────────────────────────────────────────────────────────

class TestWorkflowExecution:
    """Test actual workflow execution with mock agents."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_swarm):
        agents = ["agent_a", "agent_b", "agent_c"]
        swarm = mock_swarm(agent_ids=agents)
        self.orch = MasterOrchestrator(swarm_registry={"test_swarm": swarm})
        self.swarm = swarm

    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_agent):
        """Sequential should run agents one after another."""
        agents = [mock_agent("a"), mock_agent("b")]
        context = MagicMock()
        context.add_message = MagicMock()

        task = OrchestratedTask(workflow_pattern=WorkflowPattern.SEQUENTIAL)
        result = await self.orch._execute_sequential(task, agents, context)
        assert result.success
        assert len(task.intermediate_results) == 2

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_agent):
        """Parallel should run all agents concurrently."""
        agents = [mock_agent("a"), mock_agent("b"), mock_agent("c")]
        context = MagicMock()

        task = OrchestratedTask(workflow_pattern=WorkflowPattern.PARALLEL, parallel_slots=3)
        results = await self.orch._execute_parallel(task, agents, context)
        assert len(results) == 3
        assert all(hasattr(r, "success") for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, mock_agent):
        """Pipeline should pass output from each agent to the next."""
        agents = [mock_agent("a"), mock_agent("b")]
        context = MagicMock()
        context.messages = []

        task = OrchestratedTask(workflow_pattern=WorkflowPattern.PIPELINE)
        result = await self.orch._execute_pipeline(task, agents, context)
        assert result.success
        assert len(task.intermediate_results) == 2


# ── Checkpoint Persistence (Bug 3) ────────────────────────────────────────────

class TestCheckpointPersistence:
    """Bug 3: Checkpoints should persist to JSON files."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.orch = MasterOrchestrator()
        self.orch.CHECKPOINT_DIR = tmp_path / "checkpoints"

    def test_save_checkpoint_creates_file(self):
        task = OrchestratedTask(
            original_input="test",
            domain="research",
            workflow_pattern=WorkflowPattern.SEQUENTIAL,
            assigned_swarm="research_swarm",
            status=TaskStatus.PAUSED,
        )
        task.intermediate_results = [{"agent": "a", "success": True}]
        self.orch._save_checkpoint(task)

        path = self.orch._checkpoint_path(task.task_id)
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["task_id"] == task.task_id
        assert data["status"] == "paused"
        assert len(data["intermediate_results"]) == 1

    def test_load_checkpoint_returns_data(self):
        task = OrchestratedTask(original_input="test", status=TaskStatus.PAUSED)
        self.orch._save_checkpoint(task)
        loaded = self.orch._load_checkpoint(task.task_id)
        assert loaded is not None
        assert loaded["task_id"] == task.task_id

    def test_load_missing_checkpoint_returns_none(self):
        assert self.orch._load_checkpoint("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_pause_saves_checkpoint(self):
        task = OrchestratedTask(status=TaskStatus.RUNNING)
        self.orch._active_tasks[task.task_id] = task

        result = await self.orch.pause_task(task.task_id)
        assert result is True
        assert task.status == TaskStatus.PAUSED
        assert self.orch._checkpoint_path(task.task_id).exists()


# ── Cost Tracking (Bug 4) ─────────────────────────────────────────────────────

class TestCostTracking:
    """Bug 4: cost_usd should be tracked in intermediate_results."""

    @pytest.mark.asyncio
    async def test_sequential_tracks_cost(self, mock_agent):
        orch = MasterOrchestrator()
        agents = [mock_agent("a", cost=0.05), mock_agent("b", cost=0.03)]
        context = MagicMock()
        context.add_message = MagicMock()

        task = OrchestratedTask()
        await orch._execute_sequential(task, agents, context)

        assert all("cost_usd" in r for r in task.intermediate_results)
        total = sum(r["cost_usd"] for r in task.intermediate_results)
        assert total == pytest.approx(0.08, abs=0.001)


# ── Resource Pool ──────────────────────────────────────────────────────────────

class TestResourcePool:
    """Test resource pool slot management."""

    @pytest.mark.asyncio
    async def test_acquire_release_llm_slot(self):
        pool = ResourcePool(max_concurrent_llm_calls=2)
        assert await pool.acquire_llm_slot(timeout=1)
        assert pool._active_llm_calls == 1
        await pool.release_llm_slot()
        assert pool._active_llm_calls == 0

    @pytest.mark.asyncio
    async def test_llm_slot_limit(self):
        pool = ResourcePool(max_concurrent_llm_calls=1)
        assert await pool.acquire_llm_slot(timeout=1)
        # Second acquire should timeout
        assert not await pool.acquire_llm_slot(timeout=0.2)

    def test_cost_recording(self):
        pool = ResourcePool()
        pool.record_cost(0.05)
        pool.record_cost(0.03)
        assert pool.status()["total_cost_usd"] == pytest.approx(0.08)
