# NEXUS Architecture

> A deep-dive into the 5-layer design, data flows, and key architectural decisions.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Layer 0 — Infrastructure](#layer-0--infrastructure)
3. [Layer 1 — Trigger Management](#layer-1--trigger-management)
4. [Layer 2 — Perception Engine](#layer-2--perception-engine)
5. [Layer 3 — Orchestration](#layer-3--orchestration)
6. [Layer 4 — Execution](#layer-4--execution)
7. [Layer 5 — Memory & Knowledge](#layer-5--memory--knowledge)
8. [Cross-Cutting Concerns](#cross-cutting-concerns)
9. [Data Flow Walkthrough](#data-flow-walkthrough)
10. [Key Architectural Decisions](#key-architectural-decisions)

---

## Design Philosophy

NEXUS is built on four principles:

**1. Privacy is routing, not middleware**
Privacy tier (`PRIVATE` / `INTERNAL` / `PUBLIC`) is embedded in every LLM call at the routing level. A `PRIVATE` task _cannot_ reach a cloud model regardless of what the calling code says — the router enforces it structurally.

**2. Governance is infrastructure, not a feature**
The EU AI Act classifier, PII scrubber, and audit logger are not optional modules. They run on every task before any agent executes, baked into `dispatch()`.

**3. Composition over inheritance**
Agents, Skills, and MCP Servers are three independent, orthogonal axes. Any agent can use any skill and any MCP server. New capability is added by creating one file, not modifying existing classes.

**4. Quality is a feedback loop**
Every agent execution records a quality score. Those scores accumulate into prompt version statistics. PromptOptimizerAgent turns those statistics into better prompts automatically.

---

## Layer 0 — Infrastructure

Everything in Layer 0 is initialized once in `nexus.py:init_system()` and injected as `shared_deps` into all higher layers.

### LLM Router (`core/llm/router.py`)

Routes tasks to the appropriate model based on three signals:

| Signal | Options |
|--------|---------|
| `PrivacyTier` | `PRIVATE` → local Ollama only; `INTERNAL` → small cloud OK; `PUBLIC` → best available |
| `TaskComplexity` | `LOW` → haiku/7b; `MEDIUM` → sonnet/32b; `HIGH` → opus/72b; `CRITICAL` → opus-4-6 |
| `TaskDomain` | `ENGINEERING` prefers stronger reasoning; `CREATIVE` accepts less |

**26 registered models** across 4 providers:
- Local (Ollama): `qwen2.5:7b/32b/72b`, `llama3.1:8b`, `mistral`, `neural-chat`
- Anthropic: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Google: `gemini-1.5-pro`, `gemini-1.5-flash`

**Routing decision output:**
```python
RoutingDecision(
    primary="claude-sonnet-4-6",
    fallback="claude-haiku-4-5-20251001",
    reasoning="MEDIUM complexity, INTERNAL privacy → sonnet",
    estimated_cost_usd=0.004,
)
```

### LLM Client (`core/llm/client.py`)

Unified async interface across all providers. Handles:
- Automatic provider dispatch based on model name prefix
- Rate limiting with exponential backoff
- Token counting and cost calculation per call
- `PRIVATE` tier → always routes to local Ollama, never calls external APIs

### Governance Manager (`core/governance.py`)

Three components that run at every layer boundary:

**PIIScrubber**: Detects and redacts sensitive data before any LLM call:
- Taiwan National ID (`A123456789` → `[TAIWAN_ID]`)
- Credit card numbers → `[CREDIT_CARD]`
- Email addresses → `[EMAIL]`
- Phone numbers → `[PHONE]`
- API keys / credentials → `[CREDENTIAL]`

**AuditLogger**: Append-only SQLite log at `data/audit.db`. Records every LLM call with: timestamp, event_type, task_id, agent_id, model, privacy_tier, tokens_in, tokens_out, cost_usd, quality_score.

**QualityOptimizer**: Collects Critic scores across tasks, tracks per-agent quality trends, triggers PromptOptimizerAgent when quality dips below threshold.

### EU AI Act Classifier (`core/eu_ai_act_classifier.py`)

Rule-based classifier — zero LLM cost, sub-millisecond latency.

**3-layer classification:**
1. **Article 5 (Prohibited)** — 6 regex patterns → `is_blocked=True`
2. **Annex III (High-Risk)** — 12 rules covering §1–§8 categories → `requires_human_oversight=True`
3. **Article 52 (Limited Risk)** — transparency obligations
4. **Minimal Risk** — default

Used in `MasterOrchestrator.dispatch()` as the first gate before any workflow planning.

---

## Layer 1 — Trigger Management

**`TriggerManager`** normalizes all input sources into `TaskEvent` objects and queues them for the Perception Engine.

```
CLI Input    ──┐
REST API     ──┤
APScheduler  ──┼──→  TaskEvent{id, message, priority, session_id}
File Watcher ──┤         → asyncio.Queue
A2A Incoming ──┘
```

**Priority levels**: `NORMAL` (default) → `HIGH` → `CRITICAL`

CRITICAL tasks bypass the queue and are injected directly into the orchestrator.

---

## Layer 2 — Perception Engine

Transforms raw `TaskEvent` into a fully-analyzed `PerceivedTask` that tells the orchestrator exactly how to handle the request.

**Three analysis stages:**

```
Stage 1: Fast-path rules (0ms, no LLM)
  regex matching → domain, basic complexity, destructive flags

Stage 2: PII Scanner (0ms, no LLM)
  multi-pattern regex → has_pii, privacy_tier escalation

Stage 3: LLM Analysis (qwen2.5:7b, always local)
  → intent, task_type, required_agents, required_skills, key_entities
```

**`PerceivedTask` output fields:**
```python
PerceivedTask(
    intent="技術趨勢分析",
    task_type="research",
    domain="research",
    complexity="medium",
    privacy_tier="INTERNAL",
    required_agents=["web_agent", "rag_agent", "writer_agent"],
    required_skills=[],
    required_mcp=["arxiv_monitor", "fetch"],
    is_destructive=False,
    requires_confirmation=False,
    has_pii=False,
    language="zh-TW",
)
```

**Destructive operation detection** triggers `requires_confirmation=True`:
- File deletion: `rm -rf`, `delete`, `truncate`
- Git: `push --force`, `reset --hard`, `branch -D`
- Database: `DROP TABLE`, `TRUNCATE`, `DELETE FROM` (without WHERE)
- Process: `kill`, `pkill`, `docker rm`

---

## Layer 3 — Orchestration

The brain of NEXUS. `MasterOrchestrator.dispatch()` is the central control flow.

### Dispatch sequence

```python
async def dispatch(perceived_task):
    # 1. EU AI Act compliance gate (rule-based, ~0ms)
    check = classifier.classify(...)
    if check.is_blocked:
        return blocked_result(explanation)
    if check.requires_human_oversight:
        perceived_task["requires_confirmation"] = True

    # 2. Plan workflow
    task = _plan_workflow(perceived_task)

    # 3. Execute via swarm
    swarm = swarm_registry[task.assigned_swarm]
    result = await execute_pattern(task, swarm)

    # 4. Feed quality back to optimizer
    await optimizer.record(task)
    return task
```

### Workflow Patterns

| Pattern | Topology | Use Case |
|---------|----------|----------|
| `SEQUENTIAL` | A → B → C | Reports, simple pipelines |
| `PARALLEL` | A ∥ B → merge | Multi-source research |
| `PIPELINE` | A.output → B.input | ETL, multi-step transformation |
| `FEEDBACK_LOOP` | Execute → Critic → retry | Code generation, writing |
| `HIERARCHICAL` | Orchestrator → sub-swarms | Large compound tasks |
| `ADVERSARIAL` | Proposal ↔ Critic → Judge | High-stakes decisions |

### Resource Pool

Prevents system overload:
- Max 5 concurrent LLM calls
- Max 10 concurrent agents
- 4GB memory budget
- $1.00 max cost per task (configurable)

### Harness

Every agent execution is wrapped in the `Harness`:
- `asyncio.wait_for()` timeout enforcement
- Exponential backoff retry (1s, 2s, 4s)
- Resource slot acquisition/release
- Error isolation (one agent failure doesn't crash the workflow)
- Checkpoint save/restore for pause/resume

---

## Layer 4 — Execution

### Agents

19 agents, each with a single responsibility. All inherit `BaseAgent` and implement:

```python
class MyAgent(BaseAgent):
    agent_id   = "my_agent"
    domain     = TaskDomain.RESEARCH
    default_complexity = TaskComplexity.MEDIUM
    default_privacy    = PrivacyTier.INTERNAL

    async def execute(self, context: AgentContext) -> AgentResult:
        ...
```

**See [docs/agents/README.md](docs/agents/README.md) for full agent reference.**

### Skills

10 reusable modules callable from any agent:

```python
result = await self.skill_registry.get("synthetic_data_generator").run(
    operation="generate",
    source_file="data/sales.csv",
    n_rows=1000,
    use_copula=True,
)
```

**See [docs/skills/README.md](docs/skills/README.md) for full skill reference.**

### MCP Servers

14 external service bridges using JSON-RPC 2.0 over stdio:

```python
result = await self.mcp_client.call("arxiv_monitor", "run_monitor", {
    "keywords": ["LLM inference", "speculative decoding"]
})
```

**See [docs/mcp/README.md](docs/mcp/README.md) for full MCP reference.**

### Swarms (Declarative Workflow Config)

Each swarm is a YAML file in `config/agents/`. No Python required to create a new workflow:

```yaml
swarm_id: my_swarm
domain: research
agents:
  web_agent:
    description: "Search for current information"
  critic_agent:
    scoring_rubric:
      accuracy: 0.40
      completeness: 0.30
      clarity: 0.30
  writer_agent:
    description: "Synthesize findings"
workflow_defaults:
  pattern: sequential
  quality_threshold: 0.75
  timeout_seconds: 180
```

---

## Layer 5 — Memory & Knowledge

### 5 Memory Types

```
Working Memory     AgentContext (task lifetime only)
Short-term         Redis TTL 48h (session dialogue)
Episodic           ChromaDB (task execution records)
Semantic           ChromaDB (knowledge base: facts, documents, summaries)
Procedural         File system (skill definitions, system prompts)
```

### RAGEngine Retrieval

```
Query
  │
  ├── BM25 sparse search (keyword)
  ├── Embedding dense search (semantic, bge-m3 or text-embedding-3-small)
  └── Temporal decay bonus (recent records score higher)
       │
       ▼
  Hybrid Score = 0.6 × semantic + 0.4 × BM25
       │
  Privacy filter (only records readable by caller's tier)
       │
  Top-K results
```

### Data Lineage (`core/data_lineage.py`)

Tracks full data provenance as a directed graph:

```
task_abc  --CONSUMED-->  file_sales_q3.csv
task_abc  --PRODUCED-->  report_board.md
report_board.md  --STORED_IN-->  knowledge_doc_xyz
```

**Backend selection**: Neo4j (if available) → NetworkX + JSON fallback. Same API either way.

---

## Cross-Cutting Concerns

### Privacy Enforcement Flow

```
Any LLM call
  → LLMClient.chat(privacy_tier=X)
  → LLMRouter.route(privacy_tier=X)
  → if PRIVATE: model must be in OLLAMA_LOCAL_MODELS
  → PIIScrubber.scrub(messages) before sending
  → AuditLogger.log(model, cost, privacy_tier)
```

### Cost Control Chain

```
CostOptimizerAgent (hourly check)
  → audit.db: today_total > 90% daily_limit?
  → Auto-downgrade routing table:
    claude-opus-4-6   → claude-sonnet-4-6 → claude-haiku-4-5
    gpt-4o            → gpt-4o-mini
    gemini-1.5-pro    → gemini-1.5-flash
  → Slack alert via MCP
```

### Prompt Quality Loop

```
Task execution → AgentResult.quality_score
  → prompt_versioning.record_quality() (moving average)
  → [weekly] PromptOptimizerAgent.full_cycle()
    → generate 3–5 complete prompt variants
    → save each via prompt_versioning (semver)
  → real tasks accumulate quality stats per version (≥3 required)
  → recommend() → promote_best() if improvement > 2%
```

---

## Data Flow Walkthrough

Complete flow for: `nexus task "Analyze employee attrition risk"`

```
1. CLITrigger.submit() → TaskEvent{id, message}

2. PerceptionEngine:
   - PII: "employee" → escalate privacy_tier to PRIVATE
   - LLM (local): intent="人員分析", domain="analysis"
   - Output: PerceivedTask{privacy=PRIVATE, domain=analysis}

3. MasterOrchestrator.dispatch():
   - EU AI Act: classify("employee attrition")
     → HIGH_RISK (Annex III §4: employment)
     → requires_human_oversight=True
   - Pause for human confirmation ← user types "confirm"
   - _plan_workflow() → data_swarm, SEQUENTIAL

4. data_swarm execution (all LLM calls use local Ollama):
   - data_agent: load CSV, pandas stats (PII scrubbed)
   - rag_agent: retrieve HR domain knowledge
   - critic_agent: quality gate (score: 0.81)
   - writer_agent: synthesize report (scrubbed, no real names)

5. Memory write:
   - Episodic: task execution record
   - DataLineageTracker: employee_data.csv → attrition_report.md
   - AuditLogger: all LLM calls, cost=$0.00 (local models)

6. Return: OrchestratedTask{quality=0.81, cost=$0.00, status=COMPLETED}
```

---

## Key Architectural Decisions

### ADR-1: Privacy tier is a routing constraint, not a filter

**Decision**: Privacy tier is enforced inside `LLMRouter`, making it impossible for any agent to accidentally use a cloud model for private data.

**Alternative considered**: Middleware that intercepts LLM calls and blocks unauthorized ones.

**Reasoning**: Structural enforcement is safer than behavioral enforcement. A middleware can be bypassed; a router constraint cannot.

---

### ADR-2: EU AI Act classifier is rule-based, not LLM-based

**Decision**: `EUAIActClassifier` uses pure Python regex, not an LLM.

**Reasoning**: (1) Zero latency — runs before any LLM call is made. (2) Deterministic — same input always produces same classification. (3) No cost. (4) Cannot be "jailbroken" by crafted inputs.

---

### ADR-3: Swarms are YAML, not Python

**Decision**: Workflow configurations live in `config/agents/*.yaml`, not in Python code.

**Reasoning**: Business users and non-engineers can create new workflows without touching Python. The YAML also serves as living documentation of what each swarm does.

---

### ADR-4: Skills are sync Python, agents are async

**Decision**: Skills are implemented as `async def run()` but internally call sync libraries (openpyxl, pdfplumber, scikit-learn).

**Reasoning**: Heavy computation (ML training, PDF parsing) runs in `asyncio.run_in_executor()` to avoid blocking the event loop. The skill interface is async to be compatible with the agent execution model.

---

### ADR-5: Data lineage has a fallback backend

**Decision**: `DataLineageTracker` auto-selects Neo4j if available, falls back to NetworkX + JSON.

**Reasoning**: Neo4j provides full Cypher query capability but requires a running server. For development and small deployments, the NetworkX fallback provides identical API with zero infrastructure dependencies.
