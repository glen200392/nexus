# NEXUS — Enterprise AI Agent Management Platform

> **Local-first. Governed. Composable. Self-improving.**

NEXUS is an enterprise-grade, privacy-preserving AI Agent orchestration platform. It coordinates multiple specialized AI agents to complete complex tasks — while enforcing EU AI Act compliance, budget controls, bias auditing, and automatic prompt quality improvement.

---

## What NEXUS Does

Instead of a single AI model trying to do everything, NEXUS dispatches each task to a **team of specialized agents** that work together:

```
You: "Analyze Q3 sales data, find trends, generate a board report"

NEXUS:
  data_agent        → loads CSV, computes statistics
  ml_pipeline_agent → trains trend prediction model
  bias_auditor_agent→ verifies no demographic bias
  critic_agent      → quality-gates the analysis (score: 0.87)
  writer_agent      → produces board-ready Markdown report
  EU AI Act gate    → classifies risk, flags obligations
  cost tracker      → records $0.023 spent
```

---

## Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Private-by-default** | `PRIVATE` tasks → local Ollama only, never leave the machine |
| **Governed** | EU AI Act gate on every task; PII scrubber across all layers |
| **Composable** | 19 Agents + 10 Skills + 14 MCP Servers, mix and match |
| **Self-improving** | DSPy-style prompt optimization loop with A/B quality tracking |
| **Observable** | Immutable audit log + data lineage graph + Prometheus metrics |

---

## Architecture Overview

```
┌─ Layer 0: Infrastructure ──────────────────────────────────────┐
│  LLMRouter (26 models)  ·  LLMClient  ·  GovernanceManager    │
│  EUAIActClassifier  ·  SkillRegistry  ·  RAGEngine             │
└────────────────────────────────────────────────────────────────┘
             ↑ injected into all layers
┌─ Layer 1: Trigger ─────────────────────────────────────────────┐
│  CLI  ·  REST API  ·  APScheduler  ·  File Watcher             │
└────────────────────────────────────────────────────────────────┘
             ↓ TaskEvent
┌─ Layer 2: Perception Engine ───────────────────────────────────┐
│  Intent  ·  PII Detection  ·  Domain/Complexity Classification │
└────────────────────────────────────────────────────────────────┘
             ↓ PerceivedTask
┌─ Layer 3: Orchestration ───────────────────────────────────────┐
│  EU AI Act Gate  ·  Workflow Planner  ·  SwarmRegistry          │
│  Harness (timeout/retry/checkpoint)  ·  ResourcePool           │
└────────────────────────────────────────────────────────────────┘
             ↓ dispatches to Swarms
┌─ Layer 4: Execution ───────────────────────────────────────────┐
│  19 Agents  ·  10 Skills  ·  14 MCP Servers  ·  10 Swarms      │
└────────────────────────────────────────────────────────────────┘
             ↕ read/write
┌─ Layer 5: Memory & Knowledge ──────────────────────────────────┐
│  RAGEngine (BM25 + Semantic)  ·  ChromaDB  ·  Data Lineage     │
└────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full deep-dive.

---

## Quick Start

### Prerequisites

```bash
# Required: Python 3.11+
python --version

# Required: Ollama (for local/private LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b        # Minimum — perception + private tasks
ollama pull qwen2.5:32b       # Recommended — better reasoning

# Optional: ChromaDB for persistent vector memory
pip install chromadb
```

### Install

```bash
git clone https://github.com/glen200392/nexus.git
cd nexus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys (at minimum, set OLLAMA_BASE_URL)
```

### Run

```bash
# Initialize databases and config directories
python nexus.py init

# Interactive mode
python nexus.py start

# One-shot task
python nexus.py task "Research the latest developments in LLM inference optimization"

# System status
python nexus.py status

# Cost report
python nexus.py costs

# Web dashboard (SSE real-time updates)
uvicorn nexus.api.dashboard:app --host 0.0.0.0 --port 7800
```

---

## Platform Stats

| Dimension | Count |
|-----------|-------|
| Agents | **19** |
| Skills | **10** |
| MCP Servers | **14** |
| Swarm Configurations | **10** |
| Workflow Patterns | **6** |
| Memory Types | **5** |
| LLM Models Supported | **26** |
| Pre-loaded Regulations | **6** |

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full 5-layer architecture, data flows, design decisions |
| [DEVELOPMENT.md](DEVELOPMENT.md) | How to add agents, skills, MCP servers |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Local, Docker, and cloud deployment guides |
| [ROADMAP.md](ROADMAP.md) | Planned features and future direction |
| [docs/agents/](docs/agents/) | All 19 agents — capabilities and usage |
| [docs/skills/](docs/skills/) | All 10 skills — API reference |
| [docs/mcp/](docs/mcp/) | All 14 MCP servers — configuration |
| [docs/governance/](docs/governance/) | EU AI Act, bias auditing, cost management |
| [docs/workflows/](docs/workflows/) | End-to-end workflow examples |
| [docs/memory/](docs/memory/) | Memory system and data lineage |

---

## Project Structure

```
nexus/
├── nexus.py                    # Main entry point & system bootstrap
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
│
├── core/
│   ├── agents/                 # 19 specialized agent implementations
│   ├── orchestrator/           # 5-layer orchestration (trigger→perception→master→swarm)
│   ├── llm/                    # LLM router + unified client (26 models)
│   ├── governance.py           # PII scrubber + audit logger + quality optimizer
│   ├── eu_ai_act_classifier.py # EU AI Act risk classification (rule-based)
│   └── data_lineage.py         # Neo4j/NetworkX data provenance graph
│
├── skills/
│   ├── registry.py             # Auto-discovery and loading
│   └── implementations/        # 10 pluggable skill modules
│
├── mcp/
│   ├── client.py               # MCP stdio client
│   └── servers/                # 14 MCP server implementations
│
├── knowledge/
│   └── rag/                    # RAGEngine: hybrid BM25+semantic retrieval
│
├── api/
│   ├── dashboard.py            # FastAPI web dashboard + SSE
│   └── webhook.py              # Event streaming
│
└── config/
    ├── agents/                 # 10 swarm YAML configurations
    ├── prompts/system/         # System prompts for each agent role
    └── routing/                # LLM routing rules
```

---

## AI Governance

NEXUS has first-class AI governance built in at every layer:

- **EU AI Act (2024/1689)**: Every task is classified before execution. Prohibited practices (Art. 5) are blocked. High-risk systems (Annex III) require human oversight.
- **Bias Auditing**: Rule-based + LLM analysis of outputs for demographic bias, representational harm, and allocational disparities.
- **Cost Controls**: Per-day and per-month budgets with automatic model downgrade chains when thresholds are exceeded.
- **Data Lineage**: Full provenance graph of every data transformation (Neo4j primary, NetworkX fallback).
- **Immutable Audit Log**: Every LLM call recorded in SQLite with cost, quality, model, and privacy tier.

See [docs/governance/](docs/governance/) for full details.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and how to add new agents, skills, or MCP servers.
