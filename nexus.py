"""
NEXUS — Main Entry Point
Enterprise AI Agent Management Platform

Usage:
    python nexus.py init              # Initialize databases and config
    python nexus.py start             # Start all services
    python nexus.py task "prompt"     # Submit a one-shot task
    python nexus.py status            # Show system status
    python nexus.py pause <task_id>   # Pause a running task
    python nexus.py resume <task_id>  # Resume a paused task
    python nexus.py costs             # Show cost report
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import logging
from pathlib import Path

# ── Package path bootstrap ────────────────────────────────────────────────────
# Allow `from nexus.core.xxx import ...` when running nexus.py directly.
# Adds the parent of the nexus/ directory to sys.path.
_ROOT = Path(__file__).resolve().parent        # /Users/.../nexus
_PARENT = _ROOT.parent                          # /Users/...
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("nexus")


# ─── Bootstrap ────────────────────────────────────────────────────────────────

async def init_system() -> dict:
    """Initialize all components and return the assembled system."""
    logger.info("Initializing NEXUS...")

    # Ensure data directories exist
    for d in ["data", "data/vector_store", "data/graph", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Layer 0: Infrastructure ───────────────────────────────────────────────
    from nexus.core.llm.router import LLMRouter
    from nexus.core.llm.client import LLMClient
    from nexus.skills.registry import SkillRegistry
    from nexus.core.governance import GovernanceManager
    from nexus.knowledge.rag.engine import RAGEngine
    from nexus.mcp.client import MCPClient

    router     = LLMRouter()
    governance = GovernanceManager()
    # Bug 9 fix: inject GovernanceManager into LLMClient
    llm_client = LLMClient(governance=governance)
    skills     = SkillRegistry()
    skills.load_all()
    from nexus.core.eu_ai_act_classifier import get_classifier as get_eu_classifier
    eu_classifier = get_eu_classifier()   # pre-warm rule engine
    rag_engine = RAGEngine()
    mcp_client = MCPClient()

    # Bug 10 fix: connect DataLineageTracker
    from nexus.core.data_lineage import DataLineageTracker
    data_lineage = DataLineageTracker()

    # Connect all MCP servers
    mcp_servers = {
        "filesystem":          "mcp/servers/filesystem_server.py",
        "git":                 "mcp/servers/git_server.py",
        "fetch":               "mcp/servers/fetch_server.py",
        "sqlite":              "mcp/servers/sqlite_server.py",
        "sequential_thinking": "mcp/servers/sequential_thinking_server.py",
        # Round 2 servers
        "github":              "mcp/servers/github_server.py",
        "chroma":              "mcp/servers/chroma_server.py",
        "playwright":          "mcp/servers/playwright_server.py",
        # Round 3 servers
        "slack":               "mcp/servers/slack_server.py",
        "postgres":            "mcp/servers/postgres_server.py",
        "prometheus":          "mcp/servers/prometheus_server.py",
        # Wave 1 — AI governance & monitoring
        "arxiv_monitor":       "mcp/servers/arxiv_monitor_server.py",
        "rss_aggregator":      "mcp/servers/rss_aggregator_server.py",
        # Wave 3 — MLOps
        "mlflow":              "mcp/servers/mlflow_server.py",
    }
    for server_name, script in mcp_servers.items():
        try:
            await mcp_client.connect(
                server_name,
                [sys.executable, str(Path(__file__).parent / script)],
            )
            logger.info("MCP server connected: %s", server_name)
        except Exception as exc:
            logger.warning("MCP server '%s' failed: %s", server_name, exc)

    # ── Layer 1: Trigger Manager ──────────────────────────────────────────────
    from nexus.core.orchestrator.trigger import TriggerManager
    trigger_mgr = TriggerManager()
    cli_trigger = trigger_mgr.build_cli_trigger()

    # ── Layer 2: Perception Engine ────────────────────────────────────────────
    from nexus.core.orchestrator.perception import PerceptionEngine
    perception = PerceptionEngine(
        llm_caller=_build_llm_caller(router, llm_client),
        memory_store=rag_engine,
    )

    # ── Layer 3: Swarms + Master Orchestrator ─────────────────────────────────
    from nexus.core.orchestrator.swarm import SwarmRegistry
    from nexus.core.orchestrator.master import MasterOrchestrator

    shared_deps = {
        "router":         router,
        "llm_client":     llm_client,
        "memory_store":   rag_engine,
        "mcp_client":     mcp_client,
        "skill_registry": skills,
    }
    swarm_registry = SwarmRegistry()
    swarm_registry.load_all(shared_deps)

    orchestrator = MasterOrchestrator(
        swarm_registry=swarm_registry.to_dict(),
        quality_optimizer=governance.optimizer,
    )

    # ── Layer 5: Dashboard injection ─────────────────────────────────────────
    from nexus.api.dashboard import inject as dashboard_inject
    from nexus.api.webhook import set_event_queue
    dashboard_inject(orchestrator, governance, cli_trigger)
    set_event_queue(trigger_mgr.queue)

    logger.info("NEXUS initialized ✓")
    logger.info("  Skills loaded:  %d", len(skills.list_all()))
    logger.info("  Swarms loaded:  %d", len(swarm_registry.list_all()))
    logger.info("  MCP servers:    %s", mcp_client.list_servers())

    components = {
        "router":        router,
        "llm_client":    llm_client,
        "skills":        skills,
        "governance":    governance,
        "rag_engine":    rag_engine,
        "mcp_client":    mcp_client,
        "trigger_mgr":   trigger_mgr,
        "cli_trigger":   cli_trigger,
        "perception":    perception,
        "orchestrator":  orchestrator,
        "swarm_registry": swarm_registry,
        "data_lineage":  data_lineage,
    }

    if os.environ.get("NEXUS_ROUTER_VERSION") == "v2":
        components = _init_v2_components(components)

    return components


def _init_v2_components(components: dict) -> dict:
    """
    Initialize NEXUS v2 components (controlled by NEXUS_ROUTER_VERSION=v2).
    Purely additive — does not replace any v1 components.
    """
    from nexus.core.llm.circuit_breaker import CircuitBreaker
    from nexus.core.llm.router_v2 import LLMRouterV2
    from nexus.core.llm.cache import LLMSemanticCache
    from nexus.core.llm.client_v2 import LLMClientV2
    from nexus.core.orchestrator.checkpoint import CheckpointStore
    from nexus.core.orchestrator.guardrails import GuardrailsEngine
    from nexus.core.orchestrator.handoff import HandoffManager
    from nexus.core.orchestrator.master_v2 import MasterOrchestratorV2
    from nexus.plugins.loader import PluginLoader

    circuit_breaker = CircuitBreaker()
    router_v2 = LLMRouterV2(circuit_breaker=circuit_breaker)

    cache_path = Path("data/cache/llm_cache.db")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = LLMSemanticCache(db_path=str(cache_path))

    governance = components.get("governance")
    client_v2 = LLMClientV2(
        router=router_v2,
        cache=cache,
        circuit_breaker=circuit_breaker,
        governance=governance,
    )

    checkpoint_path = Path("data/checkpoints.db")
    checkpoint_store = CheckpointStore(db_path=str(checkpoint_path))

    guardrails_engine = GuardrailsEngine()
    guardrails_config_dir = Path("config/guardrails")
    if guardrails_config_dir.exists():
        guardrails_engine.load_rules_from_config(str(guardrails_config_dir))

    handoff_manager = HandoffManager(swarm_registry=components.get("swarm_registry"))

    orchestrator_v2 = MasterOrchestratorV2(
        swarm_registry=components.get("swarm_registry"),
        checkpoint_store=checkpoint_store,
        handoff_manager=handoff_manager,
        guardrails_engine=guardrails_engine,
        llm_client=client_v2,
    )

    plugin_loader = PluginLoader()
    try:
        plugin_loader.load_all()
    except Exception as exc:
        logger.warning("Plugin loading failed: %s", exc)

    components["circuit_breaker"] = circuit_breaker
    components["router_v2"] = router_v2
    components["cache"] = cache
    components["llm_client_v2"] = client_v2
    components["checkpoint_store"] = checkpoint_store
    components["guardrails_engine"] = guardrails_engine
    components["handoff_manager"] = handoff_manager
    components["orchestrator_v2"] = orchestrator_v2
    components["plugin_loader"] = plugin_loader

    logger.info("NEXUS v2 components initialized ✓")
    return components


def _build_llm_caller(router, llm_client):
    """
    Build the async LLM caller function using LLMClient.chat() (Bug 8 fix).
    This ensures governance guards, cost tracking, and provider abstraction apply.
    """
    from nexus.core.llm.client import Message as LLMMessage
    from nexus.core.llm.router import MODEL_REGISTRY, PrivacyTier

    async def call_llm(prompt: str, system: str, model: str = "qwen2.5:7b") -> str:
        try:
            model_config = MODEL_REGISTRY.get(model)
            if model_config is None:
                # Fallback: try to find a matching model
                model_config = MODEL_REGISTRY.get("qwen2.5:7b")
            messages = [LLMMessage(role="user", content=prompt)]
            resp = await llm_client.chat(
                messages=messages,
                model=model_config,
                system=system,
                privacy_tier=PrivacyTier.PRIVATE,  # Perception always uses local
            )
            return resp.content
        except Exception as exc:
            logger.warning("LLM call failed (%s): %s", model, exc)
            return "{}"  # Perception engine handles empty JSON gracefully
    return call_llm


# ─── Main Event Loop ──────────────────────────────────────────────────────────

async def run_task(system: dict, prompt: str) -> None:
    """Submit a task and wait for result."""
    from nexus.core.orchestrator.trigger import TriggerPriority

    cli: "CLITrigger" = system["cli_trigger"]
    perception: "PerceptionEngine" = system["perception"]
    orchestrator: "MasterOrchestrator" = system["orchestrator"]
    trigger_mgr: "TriggerManager" = system["trigger_mgr"]
    governance: "GovernanceManager" = system["governance"]

    # Layer 1 → submit
    event_id = await cli.submit(prompt)
    logger.info("Event submitted: %s", event_id[:8])

    # Layer 2 → perceive (use next_event() to unwrap PriorityQueue tuples)
    event = await trigger_mgr.next_event()
    perceived = await perception.analyze(event)
    logger.info(
        "Perceived: domain=%s complexity=%s privacy=%s agents=%s",
        perceived.domain.value, perceived.complexity.value,
        perceived.privacy_tier.value, perceived.required_agents,
    )

    # Confirmation gate for destructive actions
    if perceived.requires_confirmation or perceived.is_destructive:
        print(f"\n⚠️  This task requires confirmation:")
        print(f"   Intent: {perceived.intent}")
        print(f"   Destructive: {perceived.is_destructive}")
        confirm = input("Proceed? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Task cancelled.")
            return

    # Layer 3 → orchestrate
    task = await orchestrator.dispatch(perceived.__dict__)

    # Output
    print(f"\n{'='*60}")
    print(f"Task: {task.task_id[:8]}")
    print(f"Status: {task.status.value}")
    print(f"Quality: {task.quality_score:.2f}")
    print(f"Cost: ${task.total_cost_usd:.4f}")
    if task.final_result:
        print(f"\nResult:\n{task.final_result}")
    print('='*60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

async def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "init":
        await init_system()
        print("NEXUS initialized successfully.")

    elif cmd == "start":
        system = await init_system()
        print("NEXUS running. Type your task and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                prompt = input("nexus> ").strip()
                if prompt:
                    await run_task(system, prompt)
            except (KeyboardInterrupt, EOFError):
                print("\nShutting down NEXUS.")
                break

    elif cmd == "task":
        if len(args) < 2:
            print("Usage: nexus.py task \"your prompt here\"")
            return
        system = await init_system()
        await run_task(system, " ".join(args[1:]))

    elif cmd == "status":
        system = await init_system()
        orch = system["orchestrator"]
        gov  = system["governance"]
        print(json.dumps({
            "orchestrator": orch.status(),
            "quality":      gov.optimizer.report(),
            "costs":        gov.audit.get_cost_summary(),
        }, indent=2))

    elif cmd == "costs":
        system = await init_system()
        summary = system["governance"].audit.get_cost_summary()
        print(json.dumps(summary, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
