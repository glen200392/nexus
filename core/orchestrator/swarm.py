"""
NEXUS Swarm Coordinator — Layer 3 Domain Management
Reads swarm YAML configs, instantiates agents, and provides
the execution environment that Master Orchestrator delegates to.

Each Swarm is a self-contained unit that knows:
  - Which agents it contains
  - What workflow pattern to use
  - How to build AgentContext for its agents
  - How to merge parallel results
"""
from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import Any, Optional

from nexus.core.agents.base import AgentContext, BaseAgent
from nexus.core.llm.router import PrivacyTier, TaskComplexity, TaskDomain

logger = logging.getLogger("nexus.orchestrator.swarm")

CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "agents"


# ── Agent Factory ─────────────────────────────────────────────────────────────

def _build_agent(agent_id: str, config: dict, shared: dict) -> BaseAgent:
    """
    Instantiate the correct agent class for a given agent_id.
    shared: dict with injected dependencies (router, memory_store, mcp_client, llm_client)
    """
    from nexus.core.agents.web_agent   import WebAgent
    from nexus.core.agents.rag_agent   import RAGAgent
    from nexus.core.agents.code_agent  import CodeAgent
    from nexus.core.agents.critic_agent import CriticAgent
    from nexus.core.agents.writer_agent import WriterAgent

    from nexus.core.agents.shell_agent   import ShellAgent
    from nexus.core.agents.planner_agent import PlannerAgent
    from nexus.core.agents.browser_agent import BrowserAgent
    from nexus.core.agents.data_agent   import DataAgent
    from nexus.core.agents.memory_agent import MemoryAgent
    from nexus.core.agents.email_agent          import EmailAgent
    from nexus.core.agents.a2a_agent            import A2AAgent
    from nexus.core.agents.cost_optimizer_agent    import CostOptimizerAgent
    from nexus.core.agents.bias_auditor_agent      import BiasAuditorAgent
    from nexus.core.agents.ml_pipeline_agent       import MLPipelineAgent
    from nexus.core.agents.prompt_optimizer_agent  import PromptOptimizerAgent

    registry = {
        "web_agent":     WebAgent,
        "rag_agent":     RAGAgent,
        "code_agent":    CodeAgent,
        "critic_agent":  CriticAgent,
        "writer_agent":  WriterAgent,
        "shell_agent":   ShellAgent,
        "planner_agent": PlannerAgent,
        "browser_agent": BrowserAgent,
        "data_agent":    DataAgent,
        "memory_agent":  MemoryAgent,
        "email_agent":           EmailAgent,
        "a2a_agent":             A2AAgent,
        "cost_optimizer_agent":  CostOptimizerAgent,
        "bias_auditor_agent":    BiasAuditorAgent,
        "ml_pipeline_agent":     MLPipelineAgent,
        "prompt_optimizer_agent": PromptOptimizerAgent,
        # ── YAML-defined role aliases (map to closest existing implementation) ──
        # analyst_agent: synthesis + structured analysis → WriterAgent
        "analyst_agent":   WriterAgent,
        # test_agent: writes & executes tests → CodeAgent (same sandbox capabilities)
        "test_agent":      CodeAgent,
        # review_agent: code review → CriticAgent (quality evaluation)
        "review_agent":    CriticAgent,
        # proposal_agent: structured proposals → PlannerAgent (task decomposition)
        "proposal_agent":  PlannerAgent,
        # judge_agent: decides between competing proposals → CriticAgent
        "judge_agent":     CriticAgent,
    }

    cls = registry.get(agent_id)
    if cls is None:
        raise ValueError(
            f"Unknown agent_id '{agent_id}'. "
            f"Available: {list(registry.keys())}"
        )

    kwargs: dict = {}
    if shared.get("router"):
        kwargs["router"] = shared["router"]
    if shared.get("memory_store"):
        kwargs["memory_store"] = shared["memory_store"]
    if shared.get("mcp_client"):
        kwargs["mcp_client"] = shared["mcp_client"]
    if shared.get("llm_client"):
        kwargs["llm_client"] = shared["llm_client"]
    if shared.get("skill_registry"):
        kwargs["skill_registry"] = shared["skill_registry"]

    # Critic uses custom rubric from config
    if agent_id == "critic_agent" and "scoring_rubric" in config:
        kwargs["rubric"] = config["scoring_rubric"]

    return cls(**kwargs)


# ── Swarm ──────────────────────────────────────────────────────────────────────

class Swarm:
    """
    A domain-specific collection of agents with a shared execution context.
    Instantiated by SwarmRegistry.load() from a YAML config file.
    """

    def __init__(self, config: dict, shared_deps: dict):
        self.swarm_id    = config["swarm_id"]
        self.description = config.get("description", "")
        self.domain      = config.get("domain", "research")
        self._config     = config
        self._agents: dict[str, BaseAgent] = {}

        # Instantiate all configured agents
        for aid, acfg in config.get("agents", {}).items():
            try:
                agent = _build_agent(aid, acfg, shared_deps)
                self._agents[aid] = agent
                logger.debug("Swarm %s: loaded agent %s", self.swarm_id, aid)
            except Exception as exc:
                logger.error("Failed to load agent %s: %s", aid, exc)

        # Workflow defaults from config
        defaults = config.get("workflow_defaults", {})
        self.default_pattern    = defaults.get("pattern", "sequential")
        self.quality_threshold  = defaults.get("quality_threshold", 0.75)
        self.timeout_seconds    = defaults.get("timeout_seconds", 300)
        self.max_retries        = defaults.get("max_retries", 2)

        logger.info(
            "Swarm '%s' ready with agents: %s",
            self.swarm_id, list(self._agents.keys()),
        )

    def get_agent(self, agent_id: str) -> BaseAgent:
        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(
                f"Agent '{agent_id}' not in swarm '{self.swarm_id}'. "
                f"Available: {list(self._agents.keys())}"
            )
        return agent

    def build_context(self, perceived: dict, task) -> AgentContext:
        """
        Build an AgentContext from perceived task data and OrchestratedTask.
        This is the context that all agents in this swarm share.
        """
        privacy_str = perceived.get("privacy_tier", "INTERNAL")
        complexity_str = perceived.get("complexity", "medium")
        domain_str = perceived.get("domain", "research")

        try:
            privacy = PrivacyTier(privacy_str)
        except ValueError:
            privacy = PrivacyTier.INTERNAL
        try:
            complexity = TaskComplexity(complexity_str)
        except ValueError:
            complexity = TaskComplexity.MEDIUM
        try:
            domain = TaskDomain(domain_str)
        except ValueError:
            domain = TaskDomain.RESEARCH

        ctx = AgentContext(
            task_id=task.task_id,
            user_message=perceived.get("user_message", ""),
            privacy_tier=privacy,
            complexity=complexity,
            domain=domain,
            metadata={
                "intent":         perceived.get("intent", ""),
                "task_type":      perceived.get("task_type", "general"),
                "key_entities":   perceived.get("key_entities", []),
                "language":       perceived.get("language", "zh-TW"),
                "output_format":  "markdown",
            },
        )
        return ctx

    def merge_parallel_results(self, results: list) -> dict:
        """
        Merge outputs from parallel agent execution.
        Aggregates findings, deduplicates sources, picks best quality.
        """
        merged: dict = {
            "findings": [],
            "sources":  [],
            "errors":   [],
        }
        best_quality = 0.0
        best_output  = None

        for r in results:
            if hasattr(r, "success") and r.success:
                out = r.output
                if isinstance(out, dict):
                    merged["findings"].extend(out.get("key_findings", []))
                    merged["sources"].extend(out.get("sources", []))
                    if r.quality_score > best_quality:
                        best_quality = r.quality_score
                        best_output  = out
                elif isinstance(out, str):
                    merged["findings"].append(out[:500])
            elif hasattr(r, "error") and r.error:
                merged["errors"].append(r.error)

        # Deduplicate sources by URL
        seen_urls: set = set()
        deduped_sources = []
        for s in merged["sources"]:
            url = s.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                deduped_sources.append(s)
        merged["sources"] = deduped_sources

        merged["best_output"]  = best_output
        merged["quality_score"] = best_quality
        return merged

    def status(self) -> dict:
        return {
            "swarm_id":   self.swarm_id,
            "domain":     self.domain,
            "agents":     [a.status() for a in self._agents.values()],
        }


# ── Swarm Registry ────────────────────────────────────────────────────────────

class SwarmRegistry:
    """
    Loads all swarm configs from config/agents/*.yaml and
    instantiates Swarm objects ready for the Master Orchestrator.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._dir    = config_dir or CONFIG_DIR
        self._swarms: dict[str, Swarm] = {}

    def load_all(self, shared_deps: dict) -> int:
        """
        Load all YAML configs from the config/agents/ directory.
        Returns count of swarms loaded.
        shared_deps: dict with router, memory_store, llm_client, mcp_client, skill_registry
        """
        if not self._dir.exists():
            logger.warning("Swarm config dir not found: %s", self._dir)
            return 0

        count = 0
        for yaml_file in self._dir.glob("*.yaml"):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                if config and "swarm_id" in config:
                    swarm = Swarm(config, shared_deps)
                    self._swarms[swarm.swarm_id] = swarm
                    count += 1
            except Exception as exc:
                logger.error("Failed to load swarm config %s: %s", yaml_file, exc)

        logger.info("Loaded %d swarms from %s", count, self._dir)
        return count

    def get(self, swarm_id: str) -> Optional[Swarm]:
        return self._swarms.get(swarm_id)

    def register(self, swarm: Swarm) -> None:
        self._swarms[swarm.swarm_id] = swarm

    def list_all(self) -> list[dict]:
        return [s.status() for s in self._swarms.values()]

    def to_dict(self) -> dict[str, Swarm]:
        """Return raw dict for injection into MasterOrchestrator.swarms."""
        return dict(self._swarms)
