"""
NEXUS WorkflowGraph — Graph-Based Orchestration (v2)
Replaces the v1 pattern-based dispatch with a composable, checkpointed graph executor.

Key concepts:
  - WorkflowGraph: Builder for defining nodes, edges, and conditional routing
  - CompiledGraph: Executable graph with automatic checkpointing per step
  - GraphState: Shared mutable state passed between nodes
  - END sentinel: Special target marking graph completion

Usage:
    graph = WorkflowGraph("research")
    graph.add_node("search", search_fn)
    graph.add_node("analyze", analyze_fn)
    graph.add_node("write", write_fn)
    graph.add_edge("search", "analyze")
    graph.add_conditional_edge("analyze", quality_check, {"pass": "write", "fail": "search"})
    graph.set_entry_point("search")
    graph.set_finish_point("write")

    compiled = graph.compile(checkpoint_store)
    result = await compiled.run(initial_state)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger("nexus.graph")

END = "__end__"


@dataclass
class GraphState:
    """Shared mutable state flowing through the graph."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_node: str = ""
    step: int = 0
    data: dict = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    agent_results: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    quality_scores: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, s: str) -> GraphState:
        data = json.loads(s)
        return cls(**data)

    @property
    def avg_quality(self) -> float:
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @property
    def is_failed(self) -> bool:
        return len(self.errors) > 0

    def add_result(self, agent_id: str, output: Any, cost: float = 0.0,
                   quality: float = 0.5, success: bool = True, error: str = "") -> None:
        self.agent_results.append({
            "agent_id": agent_id,
            "output": str(output)[:2000],
            "cost_usd": cost,
            "quality": quality,
            "success": success,
            "step": self.step,
        })
        self.total_cost_usd += cost
        if quality > 0:
            self.quality_scores.append(quality)
        if error:
            self.errors.append(error)


# ── Node & Edge Types ──────────────────────────────────────────────────────────

# A node executor is an async function: (GraphState) -> GraphState
NodeExecutor = Callable[[GraphState], Any]  # Actually Awaitable[GraphState]

# A condition is an async/sync function: (GraphState) -> str (target node name)
ConditionFn = Callable[[GraphState], Any]  # Returns str


@dataclass
class _Edge:
    source: str
    target: str


@dataclass
class _ConditionalEdge:
    source: str
    condition: ConditionFn
    targets: dict[str, str]  # condition_result → target_node


# ── WorkflowGraph (Builder) ──────────────────────────────────────────────────

class WorkflowGraph:
    """
    Builder for defining graph topology.
    Fluent API: graph.add_node(...).add_edge(...).compile()
    """

    def __init__(self, name: str = "workflow"):
        self.name = name
        self._nodes: dict[str, NodeExecutor] = {}
        self._edges: list[_Edge] = []
        self._conditional_edges: list[_ConditionalEdge] = []
        self._entry_point: str | None = None
        self._finish_points: set[str] = set()

    def add_node(self, name: str, executor: NodeExecutor) -> WorkflowGraph:
        """Add a node with its executor function."""
        if name == END:
            raise ValueError("Cannot use END as a node name")
        self._nodes[name] = executor
        return self

    def add_edge(self, source: str, target: str) -> WorkflowGraph:
        """Add a direct edge from source to target."""
        self._edges.append(_Edge(source, target))
        return self

    def add_conditional_edge(
        self, source: str, condition: ConditionFn, targets: dict[str, str]
    ) -> WorkflowGraph:
        """Add a conditional edge: condition(state) returns key → targets[key] is next node."""
        self._conditional_edges.append(_ConditionalEdge(source, condition, targets))
        return self

    def set_entry_point(self, name: str) -> WorkflowGraph:
        self._entry_point = name
        return self

    def set_finish_point(self, name: str) -> WorkflowGraph:
        self._finish_points.add(name)
        return self

    def validate(self) -> list[str]:
        """Check for common errors. Returns list of warnings."""
        warnings = []
        if not self._entry_point:
            warnings.append("No entry point set")
        elif self._entry_point not in self._nodes:
            warnings.append(f"Entry point '{self._entry_point}' not in nodes")

        # Check for orphan nodes (no incoming or outgoing edges)
        connected = set()
        for e in self._edges:
            connected.add(e.source)
            connected.add(e.target)
        for ce in self._conditional_edges:
            connected.add(ce.source)
            connected.update(ce.targets.values())

        for name in self._nodes:
            if name not in connected and name != self._entry_point:
                warnings.append(f"Orphan node: '{name}' has no edges")

        # Check that edge targets exist
        for e in self._edges:
            if e.source not in self._nodes:
                warnings.append(f"Edge source '{e.source}' not in nodes")
            if e.target not in self._nodes and e.target != END:
                warnings.append(f"Edge target '{e.target}' not in nodes")

        return warnings

    def compile(self, checkpoint_store=None) -> CompiledGraph:
        """Compile into an executable graph."""
        warnings = self.validate()
        if warnings:
            for w in warnings:
                logger.warning("Graph validation: %s", w)

        return CompiledGraph(
            name=self.name,
            nodes=dict(self._nodes),
            edges=list(self._edges),
            conditional_edges=list(self._conditional_edges),
            entry_point=self._entry_point or "",
            finish_points=set(self._finish_points),
            checkpoint_store=checkpoint_store,
        )

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the graph."""
        lines = ["graph TD"]
        for name in self._nodes:
            lines.append(f"    {name}[{name}]")
        for e in self._edges:
            target = "END" if e.target == END else e.target
            lines.append(f"    {e.source} --> {target}")
        for ce in self._conditional_edges:
            for label, target in ce.targets.items():
                target_name = "END" if target == END else target
                lines.append(f"    {ce.source} -->|{label}| {target_name}")
        if self._entry_point:
            lines.append(f"    style {self._entry_point} fill:#90EE90")
        return "\n".join(lines)


# ── CompiledGraph (Executor) ─────────────────────────────────────────────────

class CompiledGraph:
    """
    Executable graph with automatic checkpointing.
    Produced by WorkflowGraph.compile().
    """

    def __init__(
        self,
        name: str,
        nodes: dict[str, NodeExecutor],
        edges: list[_Edge],
        conditional_edges: list[_ConditionalEdge],
        entry_point: str,
        finish_points: set[str],
        checkpoint_store=None,
    ):
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point
        self._finish_points = finish_points
        self._checkpoint = checkpoint_store

        # Build adjacency map for quick lookup
        self._next: dict[str, str | None] = {}
        for e in edges:
            self._next[e.source] = e.target

        self._cond_next: dict[str, _ConditionalEdge] = {}
        for ce in conditional_edges:
            self._cond_next[ce.source] = ce

    def _get_next_node(self, current: str, state: GraphState) -> str | None:
        """Determine the next node given current node and state."""
        # Check conditional edges first
        if current in self._cond_next:
            ce = self._cond_next[current]
            result = ce.condition(state)
            if asyncio.iscoroutine(result):
                raise RuntimeError("Use run_step for async conditions")
            target = ce.targets.get(result)
            if target:
                return target
            logger.warning("Condition returned '%s' but no matching target; available: %s",
                           result, list(ce.targets.keys()))
            return None

        # Check direct edges
        return self._next.get(current)

    async def _get_next_node_async(self, current: str, state: GraphState) -> str | None:
        """Async version of next-node resolution."""
        if current in self._cond_next:
            ce = self._cond_next[current]
            result = ce.condition(state)
            if asyncio.iscoroutine(result):
                result = await result
            target = ce.targets.get(result)
            return target

        return self._next.get(current)

    async def _save_checkpoint(self, state: GraphState) -> None:
        """Save checkpoint if store is available."""
        if self._checkpoint is None:
            return
        from nexus.core.orchestrator.checkpoint import Checkpoint
        cp = Checkpoint(
            thread_id=state.thread_id,
            node_name=state.current_node,
            step=state.step,
            state_json=state.to_json(),
            metadata={"graph": self.name, "task_id": state.task_id},
        )
        await self._checkpoint.save(cp)

    async def run(
        self,
        initial_state: GraphState,
        *,
        resume_from_step: int | None = None,
        max_steps: int = 50,
    ) -> GraphState:
        """
        Execute the graph from entry point (or resume point) to completion.

        Args:
            initial_state: Starting state
            resume_from_step: If set, load checkpoint and resume from this step
            max_steps: Safety limit to prevent infinite loops
        """
        state = initial_state

        # Resume from checkpoint if requested
        if resume_from_step is not None and self._checkpoint:
            cp = await self._checkpoint.load(state.thread_id, resume_from_step)
            if cp:
                state = GraphState.from_json(cp.state_json)
                logger.info("Resumed from step %d, node=%s", cp.step, cp.node_name)
            else:
                logger.warning("No checkpoint at step %d; starting from beginning", resume_from_step)

        if not state.current_node:
            state.current_node = self._entry_point

        for _ in range(max_steps):
            current = state.current_node

            if current == END:
                await self._save_checkpoint(state)
                logger.info("Graph '%s' completed (reached END), step %d",
                            self.name, state.step)
                return state

            if current not in self._nodes:
                state.errors.append(f"Node '{current}' not found in graph")
                return state

            # Execute the current node
            executor = self._nodes[current]
            try:
                state = await executor(state)
            except Exception as exc:
                logger.exception("Node '%s' failed: %s", current, exc)
                state.errors.append(f"Node '{current}' error: {exc}")
                return state

            state.step += 1

            # Checkpoint after each step
            await self._save_checkpoint(state)

            # If this was a finish point, we're done after executing it
            if current in self._finish_points:
                logger.info("Graph '%s' completed at finish point '%s', step %d",
                            self.name, current, state.step)
                return state

            # Determine next node
            next_node = await self._get_next_node_async(current, state)
            if next_node is None:
                logger.info("Graph '%s' ended (no next node from '%s')", self.name, current)
                return state

            state.current_node = next_node

        logger.warning("Graph '%s' hit max_steps limit (%d)", self.name, max_steps)
        state.errors.append(f"Max steps ({max_steps}) exceeded")
        return state

    async def run_step(self, state: GraphState) -> GraphState:
        """Execute exactly one step and return."""
        current = state.current_node or self._entry_point
        state.current_node = current

        if current == END or current in self._finish_points:
            return state

        if current not in self._nodes:
            state.errors.append(f"Node '{current}' not found")
            return state

        executor = self._nodes[current]
        state = await executor(state)
        state.step += 1

        await self._save_checkpoint(state)

        next_node = await self._get_next_node_async(current, state)
        state.current_node = next_node or END

        return state

    async def stream(self, initial_state: GraphState, *, max_steps: int = 50) -> AsyncIterator[GraphState]:
        """Yield state after each step for real-time streaming."""
        state = initial_state
        if not state.current_node:
            state.current_node = self._entry_point

        for _ in range(max_steps):
            current = state.current_node

            if current == END:
                yield state
                return

            if current not in self._nodes:
                state.errors.append(f"Node '{current}' not found")
                yield state
                return

            executor = self._nodes[current]
            try:
                state = await executor(state)
            except Exception as exc:
                state.errors.append(f"Node '{current}' error: {exc}")
                yield state
                return

            state.step += 1
            await self._save_checkpoint(state)

            yield state

            if current in self._finish_points:
                return

            next_node = await self._get_next_node_async(current, state)
            if next_node is None:
                return
            state.current_node = next_node

    def visualize(self) -> str:
        """Generate Mermaid diagram."""
        lines = ["graph TD"]
        for name in self._nodes:
            lines.append(f"    {name}[{name}]")
        for e in self._edges:
            target = "END" if e.target == END else e.target
            lines.append(f"    {e.source} --> {target}")
        for ce in self._conditional_edges:
            for label, target in ce.targets.items():
                target_name = "END" if target == END else target
                lines.append(f"    {ce.source} -->|{label}| {target_name}")
        return "\n".join(lines)
