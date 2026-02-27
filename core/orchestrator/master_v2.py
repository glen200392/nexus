"""
NEXUS MasterOrchestratorV2 — Graph-Based Orchestration Engine
Builds on the v1 MasterOrchestrator with:
  - Graph-based workflow execution via WorkflowGraph/CompiledGraph
  - Integrated guardrails (input/output)
  - Handoff-aware agent execution
  - Checkpoint-based pause/resume/fork
  - Per-pattern graph builders (sequential, parallel, pipeline, feedback_loop, adversarial, hierarchical)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from nexus.core.orchestrator.graph import (
    CompiledGraph,
    END,
    GraphState,
    WorkflowGraph,
)
from nexus.core.orchestrator.master import (
    OrchestratedTask,
    ResourcePool,
    TaskStatus,
    WorkflowPattern,
)

logger = logging.getLogger("nexus.orchestrator.master_v2")


class MasterOrchestratorV2:
    """
    V2 orchestrator using graph-based execution with guardrails and handoffs.

    Improvements over v1:
      - Composable graph topology per workflow pattern
      - Automatic checkpointing at each graph step
      - Input/output guardrails integrated into dispatch pipeline
      - Handoff support for inter-agent delegation
      - Fork/resume from any checkpoint
    """

    def __init__(
        self,
        *,
        swarm_registry=None,
        checkpoint_store=None,
        handoff_manager=None,
        guardrails_engine=None,
        llm_client=None,
        resource_pool=None,
    ):
        self.swarm_registry = swarm_registry
        self.checkpoint_store = checkpoint_store
        self.handoff_manager = handoff_manager
        self.guardrails_engine = guardrails_engine
        self.llm_client = llm_client
        self.resource_pool = resource_pool or ResourcePool()

        self._active_tasks: dict[str, dict] = {}
        self._completed_tasks: list[dict] = []

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def dispatch(self, perceived_task) -> dict:
        """
        Main entry point. Receives a PerceivedTask (or dict-like object),
        builds a graph, runs guardrails, executes, and returns results.

        Returns dict with keys:
          task_id, output, quality_score, cost_usd, checkpoints, errors
        """
        task_id = str(uuid.uuid4())
        start = time.time()
        errors: list[str] = []
        checkpoints: list[str] = []
        result: dict | None = None

        # Extract task info
        if hasattr(perceived_task, "user_message"):
            user_message = perceived_task.user_message
        elif isinstance(perceived_task, dict):
            user_message = perceived_task.get("user_message", "")
        else:
            user_message = str(perceived_task)

        # Track active task
        self._active_tasks[task_id] = {
            "task_id": task_id,
            "status": "running",
            "started_at": start,
        }

        try:
            # 1. Run input guardrails
            if self.guardrails_engine is not None:
                input_results = await self.guardrails_engine.check_input(user_message)
                if self.guardrails_engine.has_blocking(input_results):
                    blocked_details = [
                        r.detail for r in input_results
                        if r.triggered and r.action.value == "block"
                    ]
                    result = {
                        "task_id": task_id,
                        "output": None,
                        "quality_score": 0.0,
                        "cost_usd": 0.0,
                        "checkpoints": [],
                        "errors": [f"Input blocked: {'; '.join(blocked_details)}"],
                    }
                    return result

            # 2. Build graph from perceived task
            graph = self._build_graph(perceived_task)

            # 3. Compile and execute graph
            compiled = graph.compile(checkpoint_store=self.checkpoint_store)

            initial_state = GraphState(
                task_id=task_id,
                data={
                    "user_message": user_message,
                    "perceived_task": self._perceived_to_dict(perceived_task),
                },
                messages=[{"role": "user", "content": user_message}],
            )

            state = await compiled.run(initial_state)

            # 4. Run output guardrails
            output = self._extract_output(state)
            if self.guardrails_engine is not None and output:
                output_results = await self.guardrails_engine.check_output(str(output))
                # Apply scrubbing if needed
                for r in output_results:
                    if r.triggered and r.scrubbed_content is not None:
                        output = r.scrubbed_content

            # 5. Build result
            quality_score = state.avg_quality if state.quality_scores else 0.5
            cost_usd = state.total_cost_usd
            errors.extend(state.errors)

            result = {
                "task_id": task_id,
                "output": output,
                "quality_score": quality_score,
                "cost_usd": cost_usd,
                "checkpoints": checkpoints,
                "errors": errors,
            }

        except Exception as exc:
            logger.exception("Dispatch failed for task %s: %s", task_id[:8], exc)
            result = {
                "task_id": task_id,
                "output": None,
                "quality_score": 0.0,
                "cost_usd": 0.0,
                "checkpoints": [],
                "errors": [str(exc)],
            }

        finally:
            self._active_tasks.pop(task_id, None)
            if result is not None:
                self._completed_tasks.append(result)

        return result

    async def resume(self, thread_id: str) -> dict:
        """
        Resume execution from a checkpoint.
        Loads the latest checkpoint for the thread and continues execution.
        """
        if self.checkpoint_store is None:
            return {
                "task_id": "",
                "output": None,
                "quality_score": 0.0,
                "cost_usd": 0.0,
                "checkpoints": [],
                "errors": ["No checkpoint store configured"],
            }

        checkpoint = await self.checkpoint_store.load_latest(thread_id)
        if checkpoint is None:
            return {
                "task_id": "",
                "output": None,
                "quality_score": 0.0,
                "cost_usd": 0.0,
                "checkpoints": [],
                "errors": [f"No checkpoint found for thread {thread_id}"],
            }

        state = GraphState.from_json(checkpoint.state_json)
        task_id = state.task_id

        # Determine which agents to use based on checkpoint metadata
        graph_name = checkpoint.metadata.get("graph", "workflow")
        perceived = state.data.get("perceived_task", {})
        agents = perceived.get("agent_sequence", ["rag_agent", "writer_agent"])
        pattern = perceived.get("workflow_pattern", "sequential")

        # Build the appropriate graph
        graph = self._build_graph_for_pattern(pattern, agents)
        compiled = graph.compile(checkpoint_store=self.checkpoint_store)

        # Resume from the checkpoint step
        state = await compiled.run(state, resume_from_step=checkpoint.step)

        output = self._extract_output(state)

        return {
            "task_id": task_id,
            "output": output,
            "quality_score": state.avg_quality if state.quality_scores else 0.5,
            "cost_usd": state.total_cost_usd,
            "checkpoints": [checkpoint.checkpoint_id],
            "errors": state.errors,
        }

    async def pause(self, task_id: str) -> bool:
        """Pause a running task. Returns True if successfully paused."""
        if task_id in self._active_tasks:
            self._active_tasks[task_id]["status"] = "paused"
            logger.info("Task %s paused", task_id[:8])
            return True
        return False

    async def fork(self, thread_id: str, step: int | None = None) -> str:
        """
        Fork execution from a checkpoint, creating a new thread.
        Returns the new thread_id.
        """
        if self.checkpoint_store is None:
            raise RuntimeError("No checkpoint store configured")

        if step is not None:
            new_thread_id = await self.checkpoint_store.fork(thread_id, step)
        else:
            # Fork from latest checkpoint
            latest = await self.checkpoint_store.load_latest(thread_id)
            if latest is None:
                raise ValueError(f"No checkpoint found for thread {thread_id}")
            new_thread_id = await self.checkpoint_store.fork(thread_id, latest.step)

        logger.info("Forked thread %s → %s", thread_id[:8], new_thread_id[:8])
        return new_thread_id

    # ── Graph Building ────────────────────────────────────────────────────────

    def _build_graph(self, perceived_task) -> WorkflowGraph:
        """Build a workflow graph based on the perceived task."""
        # Extract workflow info
        if hasattr(perceived_task, "suggested_pattern") and perceived_task.suggested_pattern:
            pattern = perceived_task.suggested_pattern
        elif hasattr(perceived_task, "task_type"):
            pattern = self._infer_pattern(perceived_task.task_type)
        elif isinstance(perceived_task, dict):
            pattern = perceived_task.get("suggested_pattern", "") or \
                      self._infer_pattern(perceived_task.get("task_type", "general"))
        else:
            pattern = "sequential"

        # Get agents
        if hasattr(perceived_task, "required_agents"):
            agents = perceived_task.required_agents or []
        elif isinstance(perceived_task, dict):
            agents = perceived_task.get("required_agents", []) or perceived_task.get("agent_sequence", [])
        else:
            agents = []

        if not agents:
            agents = ["rag_agent", "writer_agent"]

        return self._build_graph_for_pattern(pattern, agents)

    def _build_graph_for_pattern(self, pattern: str, agents: list[str]) -> WorkflowGraph:
        """Build a graph for a specific pattern and agent list."""
        builders = {
            "sequential": self._build_sequential_graph,
            "parallel": self._build_parallel_graph,
            "pipeline": self._build_pipeline_graph,
            "feedback_loop": self._build_feedback_loop_graph,
            "adversarial": self._build_adversarial_graph,
            "hierarchical": self._build_hierarchical_graph,
        }
        builder = builders.get(pattern, self._build_sequential_graph)
        return builder(agents)

    def _build_sequential_graph(self, agents: list[str]) -> WorkflowGraph:
        """Build a sequential graph: A -> B -> C."""
        graph = WorkflowGraph("sequential")
        if not agents:
            return graph

        for agent_id in agents:
            graph.add_node(agent_id, self._make_agent_executor(agent_id))

        # Chain edges
        for i in range(len(agents) - 1):
            graph.add_edge(agents[i], agents[i + 1])

        # Last node goes to END
        graph.add_edge(agents[-1], END)

        graph.set_entry_point(agents[0])
        graph.set_finish_point(agents[-1])
        return graph

    def _build_parallel_graph(self, agents: list[str]) -> WorkflowGraph:
        """
        Build a parallel graph: all agents run from a dispatcher,
        results merge at a collector node.
        """
        graph = WorkflowGraph("parallel")
        if not agents:
            return graph

        # Create a dispatcher node that fans out
        async def dispatcher(state: GraphState) -> GraphState:
            state.data["parallel_agents"] = agents
            return state

        # Create a collector node that gathers results
        async def collector(state: GraphState) -> GraphState:
            # Results are already in state.agent_results from parallel execution
            results = [r for r in state.agent_results if r.get("success")]
            if results:
                best = max(results, key=lambda r: r.get("quality", 0.0))
                state.data["final_output"] = best.get("output", "")
            return state

        graph.add_node("dispatcher", dispatcher)

        # Add agent nodes
        for agent_id in agents:
            graph.add_node(agent_id, self._make_agent_executor(agent_id))
            graph.add_edge("dispatcher", agent_id)
            graph.add_edge(agent_id, "collector")

        graph.add_node("collector", collector)
        graph.add_edge("collector", END)

        graph.set_entry_point("dispatcher")
        graph.set_finish_point("collector")
        return graph

    def _build_pipeline_graph(self, agents: list[str]) -> WorkflowGraph:
        """Build a pipeline graph: output of each feeds input of next."""
        graph = WorkflowGraph("pipeline")
        if not agents:
            return graph

        for agent_id in agents:
            graph.add_node(agent_id, self._make_agent_executor(agent_id))

        for i in range(len(agents) - 1):
            graph.add_edge(agents[i], agents[i + 1])

        graph.add_edge(agents[-1], END)
        graph.set_entry_point(agents[0])
        graph.set_finish_point(agents[-1])
        return graph

    def _build_feedback_loop_graph(self, agents: list[str]) -> WorkflowGraph:
        """
        Build a feedback loop graph.
        Last agent is critic; if quality < threshold, loop back to first executor.
        """
        graph = WorkflowGraph("feedback_loop")
        if not agents:
            return graph

        if len(agents) < 2:
            return self._build_sequential_graph(agents)

        executors = agents[:-1]
        critic_id = agents[-1]

        # Add executor nodes
        for agent_id in executors:
            graph.add_node(agent_id, self._make_agent_executor(agent_id))

        # Chain executors
        for i in range(len(executors) - 1):
            graph.add_edge(executors[i], executors[i + 1])

        # Add critic node
        graph.add_node(critic_id, self._make_agent_executor(critic_id))
        graph.add_edge(executors[-1], critic_id)

        # Conditional: if quality >= threshold → END, else → back to first executor
        def quality_check(state: GraphState) -> str:
            if state.quality_scores and state.quality_scores[-1] >= 0.7:
                return "pass"
            # Limit retries
            retry_count = state.data.get("feedback_retries", 0)
            if retry_count >= 2:
                return "pass"
            state.data["feedback_retries"] = retry_count + 1
            return "fail"

        graph.add_conditional_edge(
            critic_id,
            quality_check,
            {"pass": END, "fail": executors[0]},
        )

        graph.set_entry_point(executors[0])
        graph.set_finish_point(critic_id)
        return graph

    def _build_adversarial_graph(self, agents: list[str]) -> WorkflowGraph:
        """
        Build an adversarial graph: proposer, critic, and judge.
        Proposer and critic debate, judge makes final decision.
        """
        graph = WorkflowGraph("adversarial")
        if len(agents) < 3:
            return self._build_sequential_graph(agents)

        proposer_id = agents[0]
        critic_id = agents[1]
        judge_id = agents[2]

        graph.add_node(proposer_id, self._make_agent_executor(proposer_id))
        graph.add_node(critic_id, self._make_agent_executor(critic_id))
        graph.add_node(judge_id, self._make_agent_executor(judge_id))

        graph.add_edge(proposer_id, critic_id)

        # Conditional: if critic score >= 0.85, go to judge; else loop back
        def debate_check(state: GraphState) -> str:
            if state.quality_scores and state.quality_scores[-1] >= 0.85:
                return "accept"
            debate_rounds = state.data.get("debate_rounds", 0)
            if debate_rounds >= 3:
                return "accept"
            state.data["debate_rounds"] = debate_rounds + 1
            return "reject"

        graph.add_conditional_edge(
            critic_id,
            debate_check,
            {"accept": judge_id, "reject": proposer_id},
        )

        graph.add_edge(judge_id, END)
        graph.set_entry_point(proposer_id)
        graph.set_finish_point(judge_id)
        return graph

    def _build_hierarchical_graph(self, agents: list[str]) -> WorkflowGraph:
        """
        Build a hierarchical graph: coordinator delegates to sub-agents,
        then synthesizes results.
        """
        graph = WorkflowGraph("hierarchical")
        if not agents:
            return graph

        coordinator_id = agents[0]
        sub_agents = agents[1:] if len(agents) > 1 else []

        # Coordinator plans
        graph.add_node(coordinator_id, self._make_agent_executor(coordinator_id))

        if sub_agents:
            # Sub-agents execute
            for agent_id in sub_agents:
                graph.add_node(agent_id, self._make_agent_executor(agent_id))
                graph.add_edge(coordinator_id, agent_id)

            # Synthesizer node
            async def synthesize(state: GraphState) -> GraphState:
                results = state.agent_results
                state.data["final_output"] = "\n".join(
                    r.get("output", "")[:500] for r in results if r.get("success")
                )
                return state

            graph.add_node("synthesizer", synthesize)
            for agent_id in sub_agents:
                graph.add_edge(agent_id, "synthesizer")

            graph.add_edge("synthesizer", END)
            graph.set_finish_point("synthesizer")
        else:
            graph.add_edge(coordinator_id, END)
            graph.set_finish_point(coordinator_id)

        graph.set_entry_point(coordinator_id)
        return graph

    # ── Agent Executor Factory ────────────────────────────────────────────────

    def _make_agent_executor(self, agent_id: str):
        """
        Create an async executor function for a given agent.
        Returns an async function (GraphState) -> GraphState.
        """
        async def executor(state: GraphState) -> GraphState:
            agent = self._resolve_agent(agent_id)
            if agent is None:
                state.errors.append(f"Agent '{agent_id}' not found")
                state.add_result(agent_id, "", success=False, error=f"Agent not found: {agent_id}")
                return state

            try:
                from nexus.core.agents.base import AgentContext

                context = AgentContext(
                    task_id=state.task_id,
                    user_message=state.data.get("user_message", ""),
                    metadata=state.data,
                )

                # Add message history
                for msg in state.messages:
                    context.add_message(msg.get("role", "user"), msg.get("content", ""))

                result = await agent.run(context)

                state.add_result(
                    agent_id=agent_id,
                    output=result.output,
                    cost=result.cost_usd,
                    quality=result.quality_score,
                    success=result.success,
                    error=result.error or "",
                )

                # Add agent output to messages for next node
                if result.output:
                    state.messages.append({
                        "role": "assistant",
                        "content": str(result.output)[:2000],
                    })

                # Update data with latest output
                state.data["last_output"] = str(result.output)[:2000] if result.output else ""
                state.data["last_agent"] = agent_id

            except Exception as exc:
                logger.exception("Agent executor '%s' failed: %s", agent_id, exc)
                state.errors.append(f"Agent '{agent_id}' error: {exc}")
                state.add_result(agent_id, "", success=False, error=str(exc))

            return state

        return executor

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_agent(self, agent_id: str):
        """Find an agent in the swarm registry."""
        if self.swarm_registry is None:
            return None

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

    def _infer_pattern(self, task_type: str) -> str:
        """Infer workflow pattern from task type."""
        mapping = {
            "research": "parallel",
            "analysis": "parallel",
            "code": "feedback_loop",
            "engineering": "feedback_loop",
            "decision": "adversarial",
            "write": "sequential",
            "operate": "sequential",
        }
        return mapping.get(task_type, "sequential")

    def _perceived_to_dict(self, perceived_task) -> dict:
        """Convert a perceived task to a dict for storage."""
        if isinstance(perceived_task, dict):
            return perceived_task
        result = {}
        for attr in ("user_message", "task_type", "domain", "complexity",
                      "required_agents", "suggested_pattern", "workflow_pattern",
                      "agent_sequence"):
            val = getattr(perceived_task, attr, None)
            if val is not None:
                if hasattr(val, "value"):
                    result[attr] = val.value
                else:
                    result[attr] = val
        return result

    def _extract_output(self, state: GraphState) -> Any:
        """Extract the final output from graph state."""
        # Check for explicit final_output
        if "final_output" in state.data:
            return state.data["final_output"]
        # Use last successful agent result
        if state.agent_results:
            for r in reversed(state.agent_results):
                if r.get("success") and r.get("output"):
                    return r["output"]
        # Use last_output from data
        return state.data.get("last_output", "")

    def status(self) -> dict:
        """Return orchestrator status."""
        return {
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "resource_pool": self.resource_pool.status(),
        }
