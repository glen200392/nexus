"""
Tests for WorkflowGraph and CompiledGraph — Graph-based orchestration.
"""
from __future__ import annotations

import pytest

from nexus.core.orchestrator.graph import (
    WorkflowGraph, CompiledGraph, GraphState, END,
)


# ── Helper node executors ──────────────────────────────────────────────────────

async def search_node(state: GraphState) -> GraphState:
    state.data["search_done"] = True
    state.add_result("search_agent", "found results", cost=0.01)
    return state

async def analyze_node(state: GraphState) -> GraphState:
    state.data["analyzed"] = True
    state.add_result("analyst_agent", "analysis complete", cost=0.02, quality=0.8)
    return state

async def write_node(state: GraphState) -> GraphState:
    state.data["written"] = True
    state.add_result("writer_agent", "final report", cost=0.03, quality=0.9)
    return state

async def failing_node(state: GraphState) -> GraphState:
    raise RuntimeError("Intentional failure")


def quality_check(state: GraphState) -> str:
    if state.avg_quality >= 0.7:
        return "pass"
    return "fail"


# ── Graph Building Tests ──────────────────────────────────────────────────────

class TestWorkflowGraph:

    def test_build_simple_graph(self):
        g = WorkflowGraph("test")
        g.add_node("a", search_node)
        g.add_node("b", write_node)
        g.add_edge("a", "b")
        g.set_entry_point("a")
        g.set_finish_point("b")

        warnings = g.validate()
        assert len(warnings) == 0

    def test_validate_missing_entry_point(self):
        g = WorkflowGraph("test")
        g.add_node("a", search_node)
        warnings = g.validate()
        assert any("entry point" in w.lower() for w in warnings)

    def test_validate_orphan_node(self):
        g = WorkflowGraph("test")
        g.add_node("a", search_node)
        g.add_node("orphan", write_node)
        g.add_edge("a", END)
        g.set_entry_point("a")
        warnings = g.validate()
        assert any("orphan" in w.lower() for w in warnings)

    def test_fluent_api(self):
        g = (WorkflowGraph("test")
             .add_node("a", search_node)
             .add_node("b", write_node)
             .add_edge("a", "b")
             .set_entry_point("a")
             .set_finish_point("b"))
        assert len(g._nodes) == 2
        assert len(g._edges) == 1

    def test_cannot_use_end_as_node_name(self):
        g = WorkflowGraph("test")
        with pytest.raises(ValueError, match="END"):
            g.add_node(END, search_node)

    def test_visualize_mermaid(self):
        g = WorkflowGraph("test")
        g.add_node("a", search_node)
        g.add_node("b", write_node)
        g.add_edge("a", "b")
        g.set_entry_point("a")

        mermaid = g.visualize()
        assert "graph TD" in mermaid
        assert "a --> b" in mermaid


# ── Graph Execution Tests ─────────────────────────────────────────────────────

class TestCompiledGraph:

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        g = (WorkflowGraph("seq")
             .add_node("search", search_node)
             .add_node("analyze", analyze_node)
             .add_node("write", write_node)
             .add_edge("search", "analyze")
             .add_edge("analyze", "write")
             .set_entry_point("search")
             .set_finish_point("write"))

        compiled = g.compile()
        state = await compiled.run(GraphState())

        assert state.data["search_done"]
        assert state.data["analyzed"]
        assert state.data["written"]
        assert len(state.agent_results) == 3
        assert state.total_cost_usd == pytest.approx(0.06)

    @pytest.mark.asyncio
    async def test_conditional_edge(self):
        g = (WorkflowGraph("conditional")
             .add_node("analyze", analyze_node)
             .add_node("write", write_node)
             .add_node("retry", search_node)
             .add_conditional_edge("analyze", quality_check, {"pass": "write", "fail": "retry"})
             .add_edge("retry", "analyze")
             .set_entry_point("analyze")
             .set_finish_point("write"))

        compiled = g.compile()
        state = await compiled.run(GraphState())

        # analyze gives quality 0.8 >= 0.7, so should go to "write"
        assert state.data.get("written")
        assert not state.data.get("search_done")  # retry not triggered

    @pytest.mark.asyncio
    async def test_node_failure_captured(self):
        g = (WorkflowGraph("fail")
             .add_node("bad", failing_node)
             .set_entry_point("bad"))

        compiled = g.compile()
        state = await compiled.run(GraphState())

        assert state.is_failed
        assert any("Intentional failure" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """Infinite loop should be caught by max_steps."""
        g = (WorkflowGraph("loop")
             .add_node("a", search_node)
             .add_node("b", analyze_node)
             .add_edge("a", "b")
             .add_edge("b", "a")
             .set_entry_point("a"))

        compiled = g.compile()
        state = await compiled.run(GraphState(), max_steps=5)

        assert state.step == 5
        assert any("Max steps" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_stream_yields_per_step(self):
        g = (WorkflowGraph("stream")
             .add_node("a", search_node)
             .add_node("b", write_node)
             .add_edge("a", "b")
             .set_entry_point("a")
             .set_finish_point("b"))

        compiled = g.compile()
        states = []
        async for s in compiled.stream(GraphState()):
            states.append(s)

        assert len(states) == 2  # One per node
        assert states[0].data.get("search_done")
        assert states[1].data.get("written")


class TestGraphState:

    def test_serialization_roundtrip(self):
        state = GraphState(task_id="test", current_node="search", step=3)
        state.data["key"] = "value"

        json_str = state.to_json()
        restored = GraphState.from_json(json_str)

        assert restored.task_id == "test"
        assert restored.step == 3
        assert restored.data["key"] == "value"

    def test_add_result(self):
        state = GraphState()
        state.add_result("agent1", "output1", cost=0.05, quality=0.9)
        state.add_result("agent2", "output2", cost=0.03, quality=0.7)

        assert len(state.agent_results) == 2
        assert state.total_cost_usd == pytest.approx(0.08)
        assert state.avg_quality == pytest.approx(0.8)

    def test_is_failed(self):
        state = GraphState()
        assert not state.is_failed
        state.errors.append("something went wrong")
        assert state.is_failed
