"""
NEXUS Planner Agent — Task Decomposition into Executable DAG
Given a complex task, this agent:
  1. Decomposes it into atomic sub-tasks
  2. Assigns the right agent to each sub-task
  3. Determines dependencies between sub-tasks (DAG)
  4. Estimates complexity and LLM cost per sub-task
  5. Returns an ExecutionPlan that MasterOrchestrator can directly dispatch

Uses claude-opus for CRITICAL tasks, claude-sonnet otherwise.
Output is consumed by MasterOrchestrator._plan_workflow().
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskComplexity, TaskDomain

logger = logging.getLogger("nexus.agents.planner")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """One atomic unit of work in an execution plan."""
    id:           str          # e.g. "t1", "t2"
    description:  str
    agent_id:     str          # which agent executes this
    depends_on:   list[str]   # list of subtask ids that must complete first
    complexity:   str = "medium"   # low | medium | high | critical
    domain:       str = "research"
    can_parallel: bool = True      # can run in parallel with siblings
    estimated_tokens: int = 1000
    output_used_by: list[str] = field(default_factory=list)  # which tasks consume this output


@dataclass
class ExecutionPlan:
    """A fully decomposed task ready for MasterOrchestrator dispatch."""
    plan_id:    str
    goal:       str              # original user request
    subtasks:   list[SubTask]
    workflow:   str = "sequential"  # sequential | parallel | hierarchical
    total_estimated_cost: float = 0.0
    rationale:  str = ""


class PlannerAgent(BaseAgent):
    agent_id   = "planner_agent"
    agent_name = "Task Planner Agent"
    description = "Decomposes complex tasks into an executable DAG of sub-tasks"
    domain     = TaskDomain.ORCHESTRATION
    default_complexity = TaskComplexity.HIGH

    # Available agents that can be assigned to sub-tasks
    AVAILABLE_AGENTS = [
        "web_agent",    "rag_agent",    "code_agent",
        "shell_agent",  "writer_agent", "critic_agent",
        "data_agent",   "browser_agent",
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        agents_desc = "\n".join([
            "  - web_agent:     search the internet for current information",
            "  - rag_agent:     query the local knowledge base",
            "  - code_agent:    write, debug, and execute code",
            "  - shell_agent:   execute shell commands on the local machine",
            "  - browser_agent: automate a web browser (click, fill forms, screenshot)",
            "  - writer_agent:  synthesize multiple inputs into polished output",
            "  - critic_agent:  evaluate quality of other agents' outputs",
            "  - data_agent:    analyze data with pandas/matplotlib",
        ])
        return (
            "You are an expert task planner. Decompose the given task into a minimal set "
            "of atomic sub-tasks that can be executed by specialized agents.\n\n"
            "Available agents:\n"
            f"{agents_desc}\n\n"
            "Rules:\n"
            "1. Each sub-task must be completable by exactly ONE agent\n"
            "2. Identify dependencies: if task B needs task A's output, B depends_on A\n"
            "3. Tasks with no dependencies can run in parallel\n"
            "4. Keep the plan minimal — don't create unnecessary steps\n"
            "5. End with a writer_agent step if the final output is a document/report\n"
            "6. End with a critic_agent step if quality is critical\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "goal": "restate the user goal in one sentence",\n'
            '  "workflow": "sequential|parallel|hierarchical",\n'
            '  "rationale": "why this decomposition",\n'
            '  "subtasks": [\n'
            '    {\n'
            '      "id": "t1",\n'
            '      "description": "specific task for this agent",\n'
            '      "agent_id": "web_agent",\n'
            '      "depends_on": [],\n'
            '      "complexity": "low|medium|high|critical",\n'
            '      "domain": "research|engineering|operations|creative|analysis",\n'
            '      "can_parallel": true\n'
            '    }\n'
            '  ]\n'
            "}"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        task = context.user_message

        # Use opus for critical/high complexity planning
        from nexus.core.llm.router import MODEL_REGISTRY
        if context.complexity in (TaskComplexity.CRITICAL, TaskComplexity.HIGH):
            if context.privacy_tier.value != "PRIVATE":
                model = MODEL_REGISTRY.get("claude-opus-4-6") or MODEL_REGISTRY["claude-sonnet-4-6"]
            else:
                model = MODEL_REGISTRY["qwen2.5:72b"]
        else:
            decision = self.route_llm(context)
            model = decision.primary

        # Retrieve relevant past plans from memory
        past_plans = await self.recall(f"planning {task}", context)
        history_context = ""
        if past_plans:
            history_context = (
                "\n\nRelevant past plans for context:\n" +
                "\n".join(r.content[:300] for r in past_plans[:2])
            )

        llm_resp = await self._llm.chat(
            messages=[
                Message("user", f"Decompose this task into sub-tasks:\n\n{task}{history_context}")
            ],
            model=model,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.2,
        )

        raw_plan = self._parse_plan(llm_resp.content, task)
        plan     = self._validate_and_enrich(raw_plan)

        # Store plan to memory for future reference
        await self.remember(
            content=f"Plan for: {task}\n{json.dumps(raw_plan, ensure_ascii=False, indent=2)[:1000]}",
            context=context,
            tags=["plan", "task_decomposition", context.domain.value],
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output=plan.__dict__,
            quality_score=0.8,
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
            cost_usd=llm_resp.cost_usd,
            llm_used=model.display_name,
        )

    # ── Plan parsing & validation ─────────────────────────────────────────────

    def _parse_plan(self, raw: str, original_task: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        # Fallback: single-step plan
        return {
            "goal": original_task,
            "workflow": "sequential",
            "rationale": "Could not parse structured plan; using single-step fallback",
            "subtasks": [
                {
                    "id": "t1",
                    "description": original_task,
                    "agent_id": "rag_agent",
                    "depends_on": [],
                    "complexity": "medium",
                    "domain": "research",
                    "can_parallel": False,
                }
            ],
        }

    def _validate_and_enrich(self, raw: dict) -> ExecutionPlan:
        """Validate the plan, fix invalid agent IDs, compute cost estimates."""
        import uuid

        subtasks = []
        for st in raw.get("subtasks", []):
            agent_id = st.get("agent_id", "rag_agent")
            # Fix invalid agent IDs
            if agent_id not in self.AVAILABLE_AGENTS:
                logger.warning("Unknown agent '%s', remapping to rag_agent", agent_id)
                agent_id = "rag_agent"

            subtasks.append(SubTask(
                id=st.get("id", f"t{len(subtasks)+1}"),
                description=st.get("description", ""),
                agent_id=agent_id,
                depends_on=st.get("depends_on", []),
                complexity=st.get("complexity", "medium"),
                domain=st.get("domain", "research"),
                can_parallel=st.get("can_parallel", True),
                estimated_tokens={"low": 500, "medium": 1500, "high": 4000, "critical": 8000}
                    .get(st.get("complexity", "medium"), 1500),
            ))

        # Detect if any tasks have no dependencies → can run in parallel
        has_parallel = any(not st.depends_on for st in subtasks)
        workflow = raw.get("workflow", "sequential")
        if has_parallel and workflow == "sequential" and len(subtasks) > 2:
            workflow = "parallel"

        total_cost = sum(
            (st.estimated_tokens / 1000) * 0.003  # rough estimate: sonnet pricing
            for st in subtasks
        )

        return ExecutionPlan(
            plan_id=str(uuid.uuid4())[:8],
            goal=raw.get("goal", ""),
            subtasks=subtasks,
            workflow=workflow,
            total_estimated_cost=total_cost,
            rationale=raw.get("rationale", ""),
        )

    def visualize(self, plan: ExecutionPlan) -> str:
        """Return ASCII DAG visualization of the plan."""
        lines = [f"Plan: {plan.goal}", f"Workflow: {plan.workflow}", ""]
        for st in plan.subtasks:
            deps = f" ← [{', '.join(st.depends_on)}]" if st.depends_on else ""
            parallel = " ∥" if st.can_parallel else ""
            lines.append(f"  [{st.id}] {st.agent_id}: {st.description[:60]}{deps}{parallel}")
        lines.append(f"\nEst. cost: ${plan.total_estimated_cost:.4f}")
        return "\n".join(lines)
