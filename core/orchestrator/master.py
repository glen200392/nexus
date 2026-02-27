"""
NEXUS Master Orchestrator — Layer 3 Core
The brain of the system. Receives PerceivedTasks from Layer 2,
builds workflow DAGs, assigns Domain Swarms, manages resources,
and collects quality feedback for continuous optimization.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("nexus.orchestrator.master")


# ─── Workflow Patterns ────────────────────────────────────────────────────────

class WorkflowPattern(str, Enum):
    SEQUENTIAL     = "sequential"      # A → B → C
    PARALLEL       = "parallel"        # A ∥ B ∥ C → merge
    PIPELINE       = "pipeline"        # A output feeds B input
    FEEDBACK_LOOP  = "feedback_loop"   # Execute → Critic → retry if score < threshold
    HIERARCHICAL   = "hierarchical"    # Delegate to sub-swarms
    ADVERSARIAL    = "adversarial"     # Proposal ↔ Critic → Judge


class TaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    PAUSED     = "paused"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


# ─── Task Definition ──────────────────────────────────────────────────────────

@dataclass
class OrchestratedTask:
    """
    A fully-analyzed task ready for workflow execution.
    Created by Master Orchestrator after receiving a PerceivedTask.
    """
    task_id:          str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id:   Optional[str] = None
    original_input:   str = ""

    # Routing decisions
    domain:           str = "research"
    workflow_pattern: WorkflowPattern = WorkflowPattern.SEQUENTIAL
    assigned_swarm:   str = ""          # e.g. "research_swarm"
    agent_sequence:   list[str] = field(default_factory=list)  # agent_ids in order

    # Execution config
    max_retries:      int = 2
    timeout_seconds:  int = 300
    quality_threshold: float = 0.7      # Critic must score above this to accept
    parallel_slots:   int = 3           # max concurrent agents in parallel pattern

    # State
    status:           TaskStatus = TaskStatus.PENDING
    retry_count:      int = 0
    created_at:       float = field(default_factory=time.time)
    started_at:       Optional[float] = None
    completed_at:     Optional[float] = None
    checkpoint:       Optional[dict] = None   # for pause/resume

    # Results
    intermediate_results: list[dict] = field(default_factory=list)
    final_result:     Optional[Any] = None
    quality_score:    float = 0.0
    total_cost_usd:   float = 0.0


# ─── Resource Pool ────────────────────────────────────────────────────────────

@dataclass
class ResourcePool:
    """Tracks available LLM slots and memory budget."""
    max_concurrent_llm_calls: int = 5
    max_concurrent_agents:    int = 10
    memory_budget_mb:         int = 4096
    max_cost_per_task_usd:    float = 1.0

    _active_llm_calls: int = 0
    _active_agents:    int = 0
    _total_cost:       float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire_llm_slot(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            async with self._lock:
                if self._active_llm_calls < self.max_concurrent_llm_calls:
                    self._active_llm_calls += 1
                    return True
            await asyncio.sleep(0.1)
        return False

    async def release_llm_slot(self) -> None:
        async with self._lock:
            self._active_llm_calls = max(0, self._active_llm_calls - 1)

    async def acquire_agent_slot(self) -> bool:
        async with self._lock:
            if self._active_agents < self.max_concurrent_agents:
                self._active_agents += 1
                return True
        return False

    async def release_agent_slot(self) -> None:
        async with self._lock:
            self._active_agents = max(0, self._active_agents - 1)

    def record_cost(self, cost_usd: float) -> None:
        self._total_cost += cost_usd

    def status(self) -> dict:
        return {
            "active_llm_calls": self._active_llm_calls,
            "active_agents":    self._active_agents,
            "total_cost_usd":   self._total_cost,
        }


# ─── Harness ──────────────────────────────────────────────────────────────────

class Harness:
    """
    Task execution sandbox. Wraps agent calls with:
      - Timeout enforcement
      - Retry logic with exponential backoff
      - Resource acquisition/release
      - Pause/resume via checkpointing
      - Error isolation (one agent failure doesn't kill the whole workflow)
    """

    def __init__(self, resource_pool: ResourcePool):
        self.pool = resource_pool

    async def run_agent(
        self,
        agent,                   # BaseAgent instance
        context,                 # AgentContext
        task: OrchestratedTask,
    ):
        """Run a single agent with full harness protection."""
        from nexus.core.agents.base import AgentResult

        if not await self.pool.acquire_agent_slot():
            raise RuntimeError("Agent slot exhausted")

        acquired_llm = await self.pool.acquire_llm_slot(timeout=task.timeout_seconds)
        if not acquired_llm:
            await self.pool.release_agent_slot()
            raise TimeoutError(f"LLM slot not available after {task.timeout_seconds}s")

        try:
            result = await asyncio.wait_for(
                agent.run(context),
                timeout=task.timeout_seconds,
            )
            self.pool.record_cost(result.cost_usd)
            return result

        except asyncio.TimeoutError:
            logger.warning("Agent %s timed out after %ds", agent.agent_id, task.timeout_seconds)
            return AgentResult(
                agent_id=agent.agent_id,
                task_id=context.task_id,
                success=False,
                output=None,
                error=f"Timeout after {task.timeout_seconds}s",
            )
        except Exception as exc:
            logger.exception("Agent %s crashed: %s", agent.agent_id, exc)
            return AgentResult(
                agent_id=agent.agent_id,
                task_id=context.task_id,
                success=False,
                output=None,
                error=str(exc),
            )
        finally:
            await self.pool.release_llm_slot()
            await self.pool.release_agent_slot()

    async def run_with_retry(self, agent, context, task: OrchestratedTask):
        """Run with exponential backoff retry."""
        for attempt in range(task.max_retries + 1):
            result = await self.run_agent(agent, context, task)
            if result.success:
                return result
            if attempt < task.max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s…
                logger.info("Retry %d/%d in %ds", attempt + 1, task.max_retries, wait)
                await asyncio.sleep(wait)
        return result  # return last failure


# ─── Master Orchestrator ──────────────────────────────────────────────────────

class MasterOrchestrator:
    """
    Central coordinator for the entire NEXUS system.

    Responsibilities:
      1. Receive PerceivedTask from Perception Layer
      2. Build OrchestratedTask with workflow plan
      3. Route to the appropriate Domain Swarm
      4. Collect quality scores and feed back to optimization loop
      5. Manage checkpoints for pause/resume
    """

    def __init__(
        self,
        resource_pool: Optional[ResourcePool] = None,
        swarm_registry: Optional[dict] = None,
        quality_optimizer=None,
    ):
        self.pool         = resource_pool or ResourcePool()
        self.harness      = Harness(self.pool)
        # Accept either a SwarmRegistry object or a plain dict
        if hasattr(swarm_registry, "to_dict"):
            self.swarms = swarm_registry.to_dict()
        else:
            self.swarms = swarm_registry or {}
        self.optimizer    = quality_optimizer
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._active_tasks: dict[str, OrchestratedTask] = {}
        self._completed_tasks: list[OrchestratedTask] = []
        self._running = False

    # ── Routing Logic ─────────────────────────────────────────────────────────

    def _plan_workflow(self, perceived: dict) -> OrchestratedTask:
        """
        Convert a PerceivedTask dict into a full OrchestratedTask with workflow plan.
        This is the core routing intelligence of Layer 3.
        """
        # Bug 1 fix: Convert enum objects to their string values for comparison
        _raw_domain     = perceived.get("domain", "research")
        _raw_complexity = perceived.get("complexity", "medium")
        _raw_task_type  = perceived.get("task_type", "general")

        domain     = _raw_domain.value if hasattr(_raw_domain, "value") else _raw_domain
        complexity = _raw_complexity.value if hasattr(_raw_complexity, "value") else _raw_complexity
        task_type  = _raw_task_type.value if hasattr(_raw_task_type, "value") else _raw_task_type

        task = OrchestratedTask(
            original_input=perceived.get("user_message", ""),
            domain=domain,
        )

        # ── Workflow pattern selection ──────────────────────────────────────
        if task_type in ("research", "analysis"):
            # Parallel search + synthesis
            task.workflow_pattern = WorkflowPattern.PARALLEL
            task.agent_sequence   = ["web_agent", "rag_agent", "analyst_agent", "writer_agent"]

        elif task_type in ("code", "engineering"):
            # Sequential with feedback loop for quality
            task.workflow_pattern = WorkflowPattern.FEEDBACK_LOOP
            task.agent_sequence   = ["planner_agent", "code_agent", "test_agent", "critic_agent"]
            task.quality_threshold = 0.8  # higher bar for code

        elif task_type == "decision":
            # Adversarial for high-stakes decisions
            task.workflow_pattern = WorkflowPattern.ADVERSARIAL
            task.agent_sequence   = ["proposal_agent", "critic_agent", "judge_agent"]

        elif complexity == "critical":
            # Hierarchical delegation
            task.workflow_pattern = WorkflowPattern.HIERARCHICAL
            task.agent_sequence   = ["domain_orchestrator"]

        else:
            # Default: simple sequential
            task.workflow_pattern = WorkflowPattern.SEQUENTIAL
            task.agent_sequence   = ["rag_agent", "writer_agent"]

        # ── Swarm assignment ────────────────────────────────────────────────
        # Maps perceived domain → registered swarm_id (must match config/agents/*.yaml)
        DOMAIN_TO_SWARM = {
            "research":      "research_swarm",
            "engineering":   "engineering_swarm",
            "operations":    "maintenance_swarm",   # maintenance_swarm owns operations domain
            "creative":      "research_swarm",      # no dedicated creative swarm; research is closest
            "analysis":      "data_swarm",          # data_swarm handles analysis tasks
            "data_science":  "data_swarm",
            "ml":            "ml_swarm",
            "governance":    "governance_swarm",
            "compliance":    "governance_swarm",
        }
        task.assigned_swarm = DOMAIN_TO_SWARM.get(domain, "research_swarm")

        logger.info(
            "Planned task %s: pattern=%s swarm=%s agents=%s",
            task.task_id[:8], task.workflow_pattern.value,
            task.assigned_swarm, task.agent_sequence,
        )
        return task

    # ── Workflow Executors ────────────────────────────────────────────────────

    async def _execute_sequential(
        self, task: OrchestratedTask, agents: list, context
    ) -> Any:
        """Run agents one after another. Output of each feeds into next."""
        result = None
        for agent in agents:
            if result is not None:
                # Inject previous result into context
                context.add_message("assistant", str(result.output))
            result = await self.harness.run_with_retry(agent, context, task)
            task.intermediate_results.append({
                "agent": agent.agent_id,
                "success": result.success,
                "output": str(result.output)[:500],
                "cost_usd": getattr(result, "cost_usd", 0.0),
            })
            if not result.success:
                break
        return result

    async def _execute_parallel(
        self, task: OrchestratedTask, agents: list, context
    ) -> list:
        """Run agents concurrently. Collect all results."""
        sem = asyncio.Semaphore(task.parallel_slots)

        async def run_guarded(agent):
            async with sem:
                return await self.harness.run_with_retry(agent, context, task)

        results = await asyncio.gather(
            *[run_guarded(a) for a in agents],
            return_exceptions=True,
        )
        successful = [r for r in results if hasattr(r, "success") and r.success]
        logger.info(
            "Parallel execution: %d/%d succeeded", len(successful), len(agents)
        )
        return results

    async def _execute_feedback_loop(
        self, task: OrchestratedTask, agents: list, context
    ) -> Any:
        """
        Execute → Critic evaluates → retry if quality below threshold.
        Requires a 'critic_agent' in the last position of agent_sequence.
        """
        executor_agents = agents[:-1]
        critic_agent    = agents[-1]
        best_result     = None
        best_score      = 0.0

        for attempt in range(task.max_retries + 1):
            # Execute
            exec_result = await self._execute_sequential(task, executor_agents, context)
            # Critique
            context.add_message("assistant", str(exec_result.output))
            context.metadata["critique_attempt"] = attempt
            critique = await self.harness.run_agent(critic_agent, context, task)
            score = critique.quality_score

            if score > best_score:
                best_score  = score
                best_result = exec_result
                best_result.quality_score = score

            logger.info(
                "Feedback loop attempt %d/%d: quality=%.2f threshold=%.2f",
                attempt + 1, task.max_retries + 1, score, task.quality_threshold,
            )

            if score >= task.quality_threshold:
                break
            if attempt < task.max_retries:
                context.add_message(
                    "user",
                    f"Quality score {score:.2f} below threshold. "
                    f"Critic feedback: {critique.output}\nPlease improve.",
                )

        return best_result

    async def _execute_adversarial(
        self, task: OrchestratedTask, agents: list, context
    ) -> Any:
        """
        Proposal ↔ Critic debate → Judge decides.
        agents = [proposal_agent, critic_agent, judge_agent]
        """
        proposal_agent, critic_agent, judge_agent = agents[0], agents[1], agents[2]
        rounds = 3

        for i in range(rounds):
            proposal = await self.harness.run_agent(proposal_agent, context, task)
            context.add_message("assistant", f"Proposal: {proposal.output}")

            critique = await self.harness.run_agent(critic_agent, context, task)
            context.add_message("user", f"Critique: {critique.output}")

            if critique.quality_score >= 0.85:
                break  # Critic accepts the proposal

        return await self.harness.run_agent(judge_agent, context, task)

    async def _execute_pipeline(
        self, task: OrchestratedTask, agents: list, context
    ) -> Any:
        """
        Pipeline execution: output of each agent feeds as input to the next.
        Unlike sequential, each agent receives ONLY the previous agent's output.
        """
        result = None
        for i, agent in enumerate(agents):
            if result is not None:
                # Replace context with previous output only
                context.messages = [
                    {"role": "user", "content": str(result.output)},
                ]
            result = await self.harness.run_with_retry(agent, context, task)
            task.intermediate_results.append({
                "agent": agent.agent_id,
                "step": i,
                "success": result.success,
                "output": str(result.output)[:500],
                "cost_usd": getattr(result, "cost_usd", 0.0),
            })
            if not result.success:
                break
        return result

    async def _execute_hierarchical(
        self, task: OrchestratedTask, agents: list, context
    ) -> Any:
        """
        Hierarchical execution: first agent acts as coordinator, delegates to sub-agents.
        The coordinator agent plans sub-tasks; remaining agents execute them.
        """
        if not agents:
            raise ValueError("No agents for hierarchical execution")

        coordinator = agents[0]
        sub_agents = agents[1:] if len(agents) > 1 else agents

        # Step 1: Coordinator plans the work
        coord_result = await self.harness.run_with_retry(coordinator, context, task)
        task.intermediate_results.append({
            "agent": coordinator.agent_id,
            "role": "coordinator",
            "success": coord_result.success,
            "output": str(coord_result.output)[:500],
            "cost_usd": getattr(coord_result, "cost_usd", 0.0),
        })

        if not coord_result.success or not sub_agents:
            return coord_result

        # Step 2: Sub-agents execute in parallel
        context.add_message("assistant", f"Coordinator plan: {coord_result.output}")
        parallel_results = await self._execute_parallel(task, sub_agents, context)

        # Step 3: Coordinator synthesizes results
        for r in parallel_results:
            if hasattr(r, "output") and r.output:
                context.add_message("assistant", str(r.output)[:500])

        context.add_message("user", "Synthesize all sub-agent results into a final answer.")
        final = await self.harness.run_agent(coordinator, context, task)
        task.intermediate_results.append({
            "agent": coordinator.agent_id,
            "role": "synthesizer",
            "success": final.success,
            "output": str(final.output)[:500],
            "cost_usd": getattr(final, "cost_usd", 0.0),
        })
        return final

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def dispatch(self, perceived_task: dict) -> OrchestratedTask:
        """
        Primary entry point. Called by Perception Layer.
        Builds workflow, executes, stores results.
        """
        # ── EU AI Act compliance gate (rule-based, zero-latency, no LLM cost) ──
        try:
            from nexus.core.eu_ai_act_classifier import get_classifier
            classifier = get_classifier()
            check = classifier.classify(
                task_type=perceived_task.get("task_type", ""),
                domain=perceived_task.get("domain", ""),
                description=perceived_task.get("user_message", ""),
            )
            allowed, explanation = classifier.gate_task(check)
            if not allowed:
                blocked = OrchestratedTask(
                    original_input=perceived_task.get("user_message", ""),
                    status=TaskStatus.CANCELLED,
                )
                blocked.final_result  = {"blocked": True, "reason": explanation}
                blocked.completed_at  = time.time()
                self._completed_tasks.append(blocked)
                logger.warning("Task BLOCKED by EU AI Act gate: %s", explanation)
                return blocked
            if check.requires_human_oversight:
                logger.warning(
                    "EU AI Act HIGH-RISK task — human oversight required. Articles: %s",
                    check.applicable_articles,
                )
                perceived_task["requires_confirmation"] = True
                perceived_task["eu_ai_act_risk"]        = check.risk_level.value
                perceived_task["applicable_articles"]   = check.applicable_articles
        except Exception as exc:
            logger.warning("EU AI Act compliance check skipped (non-fatal): %s", exc)

        task = self._plan_workflow(perceived_task)
        task.status     = TaskStatus.RUNNING
        task.started_at = time.time()
        self._active_tasks[task.task_id] = task

        try:
            swarm = self.swarms.get(task.assigned_swarm)
            if swarm is None:
                raise ValueError(f"Swarm '{task.assigned_swarm}' not registered")

            agents  = [swarm.get_agent(aid) for aid in task.agent_sequence]
            context = swarm.build_context(perceived_task, task)

            # Execute based on pattern (Bug 2 fix: added PIPELINE + HIERARCHICAL)
            if task.workflow_pattern == WorkflowPattern.SEQUENTIAL:
                result = await self._execute_sequential(task, agents, context)
            elif task.workflow_pattern == WorkflowPattern.PARALLEL:
                results = await self._execute_parallel(task, agents, context)
                # Bug 5 fix: merge parallel results via swarm
                result = swarm.merge_parallel_results(results)
            elif task.workflow_pattern == WorkflowPattern.PIPELINE:
                result = await self._execute_pipeline(task, agents, context)
            elif task.workflow_pattern == WorkflowPattern.FEEDBACK_LOOP:
                result = await self._execute_feedback_loop(task, agents, context)
            elif task.workflow_pattern == WorkflowPattern.ADVERSARIAL:
                result = await self._execute_adversarial(task, agents, context)
            elif task.workflow_pattern == WorkflowPattern.HIERARCHICAL:
                result = await self._execute_hierarchical(task, agents, context)
            else:
                result = await self._execute_sequential(task, agents, context)

            task.final_result   = result.get("best_output", result) if isinstance(result, dict) else (result.output if hasattr(result, "output") else result)
            task.quality_score  = result.get("quality_score", 0.5) if isinstance(result, dict) else getattr(result, "quality_score", 0.5)
            # Bug 4 fix: accumulate cost_usd from intermediate results
            task.total_cost_usd = sum(
                r.get("cost_usd", 0.0) for r in task.intermediate_results
            )
            task.status         = TaskStatus.COMPLETED

        except Exception as exc:
            logger.exception("Task %s failed: %s", task.task_id[:8], exc)
            task.status = TaskStatus.FAILED

        finally:
            task.completed_at = time.time()
            self._active_tasks.pop(task.task_id, None)
            self._completed_tasks.append(task)

            # Feed to continuous optimization loop
            if self.optimizer:
                await self.optimizer.record(task)

        return task

    # ── Pause / Resume (Bug 3 fix: JSON file checkpoint persistence) ─────────

    CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "data" / "checkpoints"

    def _checkpoint_path(self, task_id: str) -> Path:
        return self.CHECKPOINT_DIR / f"{task_id}.json"

    def _save_checkpoint(self, task: OrchestratedTask) -> None:
        """Persist checkpoint to JSON file."""
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "task_id": task.task_id,
            "status": task.status.value,
            "domain": task.domain,
            "workflow_pattern": task.workflow_pattern.value,
            "assigned_swarm": task.assigned_swarm,
            "agent_sequence": task.agent_sequence,
            "intermediate_results": task.intermediate_results,
            "quality_score": task.quality_score,
            "total_cost_usd": task.total_cost_usd,
            "original_input": task.original_input,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_step": len(task.intermediate_results),
        }
        self._checkpoint_path(task.task_id).write_text(
            _json.dumps(checkpoint_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Checkpoint saved for task %s", task.task_id[:8])

    def _load_checkpoint(self, task_id: str) -> dict | None:
        """Load checkpoint from JSON file."""
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return None
        try:
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load checkpoint for %s: %s", task_id[:8], exc)
            return None

    async def pause_task(self, task_id: str) -> bool:
        task = self._active_tasks.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.status     = TaskStatus.PAUSED
            task.checkpoint = {"intermediate": task.intermediate_results}
            self._save_checkpoint(task)
            logger.info("Task %s paused", task_id[:8])
            return True
        return False

    async def resume_task(self, task_id: str) -> bool:
        task = self._active_tasks.get(task_id)
        if task and task.status == TaskStatus.PAUSED:
            task.status = TaskStatus.RUNNING
            logger.info("Task %s resumed from in-memory checkpoint", task_id[:8])
            return True

        # Try loading from file checkpoint
        checkpoint = self._load_checkpoint(task_id)
        if checkpoint:
            task = OrchestratedTask(
                task_id=checkpoint["task_id"],
                original_input=checkpoint.get("original_input", ""),
                domain=checkpoint.get("domain", "research"),
                workflow_pattern=WorkflowPattern(checkpoint.get("workflow_pattern", "sequential")),
                assigned_swarm=checkpoint.get("assigned_swarm", ""),
                agent_sequence=checkpoint.get("agent_sequence", []),
                intermediate_results=checkpoint.get("intermediate_results", []),
                quality_score=checkpoint.get("quality_score", 0.0),
                total_cost_usd=checkpoint.get("total_cost_usd", 0.0),
                status=TaskStatus.RUNNING,
                started_at=checkpoint.get("started_at"),
            )
            self._active_tasks[task_id] = task
            logger.info("Task %s resumed from file checkpoint (step %d)",
                        task_id[:8], checkpoint.get("completed_step", 0))
            return True

        return False

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "active_tasks":    len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "resource_pool":   self.pool.status(),
            "registered_swarms": list(self.swarms.keys()),
        }
