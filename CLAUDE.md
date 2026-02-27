# NEXUS — Claude Code Project Instructions

## Overview
Enterprise AI Agent orchestration platform — 19 agents, 10 skills, 14 MCP servers.
See @README.md for full description, @ARCHITECTURE.md for system design.

## Environment Setup
```bash
cd /Users/tsunglunho/nexus
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands
```bash
# Run the platform
python nexus.py init          # First time setup
python nexus.py start         # Interactive mode
python nexus.py task "..."    # One-shot task
python nexus.py status        # System status

# Web dashboard
uvicorn api.dashboard:app --host 0.0.0.0 --port 7800

# Testing
pytest tests/unit -x -q                    # Unit tests (fast, no deps)
pytest tests/integration -x -q             # Integration tests
pytest tests/e2e -x -q                     # E2E tests
pytest tests/unit -k "test_name" -x -q     # Single test

# Type checking
python -m mypy core/ --ignore-missing-imports
```

## Code Style
- Python 3.11+, type hints required on all public functions
- Async by default — use `async def` for I/O-bound operations
- Pydantic v2 for data models, FastAPI for API endpoints
- Follow existing agent pattern in `core/agents/base.py` when creating new agents
- Follow existing skill pattern in `skills/implementations/` when creating new skills

## Architecture (5 layers)
- Layer 0: Infrastructure (LLMRouter, GovernanceManager, SkillRegistry)
- Layer 1: Trigger (CLI, REST API, APScheduler, FileWatcher)
- Layer 2: Perception (Intent, PII, Domain classification)
- Layer 3: Orchestration (EU AI Act gate, Workflow planner, Swarms)
- Layer 4: Execution (Agents, Skills, MCP Servers, Swarms)
- Layer 5: Memory & Knowledge (RAG, ChromaDB, Data Lineage)

## Key Patterns
- All agents inherit from `core/agents/base.py` — NEVER create standalone agent classes
- All skills auto-discovered via `skills/registry.py` — drop module in `skills/implementations/`
- MCP servers follow stdio protocol via `mcp/client.py`
- Config YAML in `config/agents/` for swarm definitions
- System prompts in `config/prompts/system/`

## Governance (IMPORTANT)
- EU AI Act compliance is mandatory — every task goes through classification
- PII scrubber runs on all inputs/outputs
- Cost tracking is always active — respect budget limits in .env
- NEVER bypass governance checks, even in tests

## Don'ts
- NEVER modify .env directly (use .env.example as reference)
- NEVER commit API keys or credentials
- NEVER skip EU AI Act classification gate
- Don't run full test suite for quick iteration — use `pytest -k` for single tests
