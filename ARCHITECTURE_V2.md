# NEXUS v2 Architecture — Next-Generation Design

> Complete architecture redesign informed by Anthropic MCP/Agent SDK, Google A2A/ADK,
> OpenAI Agents SDK, and the top 50 GitHub AI projects (2024–2026 trends).

---

## Executive Summary

NEXUS v2 evolves from a **5-layer pipeline** to a **stateful graph-based orchestration platform**
with first-class support for:

- **MCP Streamable HTTP** (replacing stdio-only MCP)
- **A2A Protocol** (Google's Agent-to-Agent standard, 50+ partners)
- **Durable Execution** (LangGraph-style checkpoint persistence)
- **Handoffs & Guardrails** (OpenAI Agents SDK primitives)
- **Multi-Modal Input/Output** (vision, audio, structured data)
- **Plugin-First Architecture** (Dify-inspired hot-reloadable extensions)
- **Agent Skills & Tool Search** (Anthropic's deferred-loading pattern)

**Key metrics target:**
| Metric | v1 | v2 Target |
|--------|-----|-----------|
| Agents | 19 | 19+ (same, composable) |
| Skills | 10 | 10+ (hot-reloadable) |
| MCP Servers | 14 | 14+ (stdio + HTTP transport) |
| LLM Models | 26 | 30+ (Gemini 3, Claude 4.6 1M, GPT-5) |
| Workflow Patterns | 6 | 8+ (+ map-reduce, scatter-gather) |
| Checkpoint/Resume | ❌ | ✅ Durable execution |
| Agent Federation | Basic | Full A2A + AgentCard discovery |
| Transport | stdio only | stdio + Streamable HTTP + WebSocket |

---

## Table of Contents

1. [Design Principles (Updated)](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Layer 0 — Foundation Platform](#3-layer-0--foundation-platform)
4. [Layer 1 — Ingestion & Triggers](#4-layer-1--ingestion--triggers)
5. [Layer 2 — Perception & Routing](#5-layer-2--perception--routing)
6. [Layer 3 — Stateful Orchestration](#6-layer-3--stateful-orchestration)
7. [Layer 4 — Execution Runtime](#7-layer-4--execution-runtime)
8. [Layer 5 — Memory & Knowledge](#8-layer-5--memory--knowledge)
9. [Layer 6 — Federation & A2A](#9-layer-6--federation--a2a)
10. [Cross-Cutting: Governance & Compliance](#10-cross-cutting-governance--compliance)
11. [Cross-Cutting: Observability & Tracing](#11-cross-cutting-observability--tracing)
12. [Cross-Cutting: Cost Intelligence](#12-cross-cutting-cost-intelligence)
13. [MCP v2 — Streamable HTTP Transport](#13-mcp-v2--streamable-http-transport)
14. [LLM Router v2 — Multi-Provider Intelligence](#14-llm-router-v2--multi-provider-intelligence)
15. [Plugin System](#15-plugin-system)
16. [Security Model](#16-security-model)
17. [Deployment Architecture](#17-deployment-architecture)
18. [Migration Path from v1](#18-migration-path-from-v1)
19. [Architectural Decision Records](#19-architectural-decision-records)

---

## 1. Design Principles

### Retained from v1 (proven correct)
1. **Privacy is routing, not middleware** — structural enforcement at LLMRouter level
2. **Governance is infrastructure** — EU AI Act + PII + audit run on every task
3. **Composition over inheritance** — agents, skills, MCP servers are orthogonal
4. **Quality is a feedback loop** — scores → prompt optimization → A/B testing

### New in v2
5. **Durable execution by default** — every workflow step checkpoints state; crash-safe resume
6. **Federation-native** — A2A protocol for inter-agent communication across organizations
7. **Transport-agnostic MCP** — Streamable HTTP as primary, stdio as fallback
8. **Handoff as primitive** — agents can transfer context to other agents mid-execution (OpenAI pattern)
9. **Guardrails are declarative** — input/output validation rules in YAML, not code
10. **Plugin-first extensibility** — agents, skills, MCP servers are all hot-reloadable plugins

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEXUS v2 Platform                                │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 6: Federation (A2A Protocol)                              │   │
│  │  AgentCard ◄──► Discovery ◄──► Delegate ◄──► Broadcast          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 5: Memory & Knowledge                                     │   │
│  │  Working │ Short-term │ Episodic │ Semantic │ Procedural │ Graph │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 4: Execution Runtime                                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌──────────────┐ ┌──────────────────┐ │   │
│  │  │ Agents  │ │ Skills  │ │ MCP Servers  │ │ External Plugins │ │   │
│  │  │ (19+)   │ │ (10+)   │ │ (14+, HTTP)  │ │ (hot-reload)     │ │   │
│  │  └─────────┘ └─────────┘ └──────────────┘ └──────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 3: Stateful Orchestration (Graph-Based)                   │   │
│  │  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐ │   │
│  │  │ Workflow  │ │ Checkpoint   │ │ Handoff    │ │ Guardrails   │ │   │
│  │  │ Graph    │ │ Persistence  │ │ Manager    │ │ Engine       │ │   │
│  │  └──────────┘ └──────────────┘ └────────────┘ └──────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 2: Perception & Intelligent Routing                       │   │
│  │  Fast-path │ PII │ LLM Analysis │ Privacy │ Complexity │ Domain │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 1: Ingestion & Triggers                                   │   │
│  │  CLI │ REST │ WebSocket │ Scheduler │ FileWatch │ A2A │ Webhook │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Layer 0: Foundation Platform                                    │   │
│  │  LLMRouter │ LLMClient │ Governance │ RAGEngine │ MCPManager   │   │
│  │  PluginLoader │ ConfigStore │ SecretVault │ TracingExporter     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ═══════════ Cross-Cutting ════════════════════════════════════════════ │
│  Governance │ Observability/Tracing │ Cost Intelligence │ Security     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 0 — Foundation Platform

### 3.1 LLM Router v2

**What changed:** Support for latest models + intelligent fallback chains + cost-aware routing.

```python
# config/routing/llm_routing_v2.yaml
models:
  # ── Anthropic ──────────────────────────────
  claude-opus-4-6:
    provider: anthropic
    context_window: 200000
    max_output: 32000
    cost_per_1k_input: 0.015
    cost_per_1k_output: 0.075
    capabilities: [reasoning, code, vision, tool_use, extended_thinking]
    privacy: [PUBLIC, INTERNAL]

  claude-sonnet-4-6:
    provider: anthropic
    context_window: 1048576    # 1M context — key differentiator
    max_output: 64000
    cost_per_1k_input: 0.003
    cost_per_1k_output: 0.015
    capabilities: [reasoning, code, vision, tool_use, extended_thinking, long_context]
    privacy: [PUBLIC, INTERNAL]

  claude-haiku-4-5:
    provider: anthropic
    context_window: 200000
    max_output: 8192
    cost_per_1k_input: 0.0008
    cost_per_1k_output: 0.004
    capabilities: [fast, code, vision, tool_use]
    privacy: [PUBLIC, INTERNAL]

  # ── Google ─────────────────────────────────
  gemini-2.5-pro:
    provider: google
    context_window: 1048576
    max_output: 65536
    cost_per_1k_input: 0.00125    # <128K pricing
    cost_per_1k_output: 0.01
    capabilities: [reasoning, code, vision, audio, tool_use, grounding, thinking]
    privacy: [PUBLIC, INTERNAL]

  gemini-2.5-flash:
    provider: google
    context_window: 1048576
    max_output: 65536
    cost_per_1k_input: 0.000075
    cost_per_1k_output: 0.0003
    capabilities: [fast, code, vision, audio, tool_use, thinking]
    privacy: [PUBLIC, INTERNAL]

  # ── OpenAI ─────────────────────────────────
  gpt-4.1:
    provider: openai
    context_window: 1047576
    max_output: 32768
    cost_per_1k_input: 0.002
    cost_per_1k_output: 0.008
    capabilities: [reasoning, code, vision, tool_use, structured_output]
    privacy: [PUBLIC, INTERNAL]

  gpt-4.1-mini:
    provider: openai
    context_window: 1047576
    max_output: 32768
    cost_per_1k_input: 0.0004
    cost_per_1k_output: 0.0016
    capabilities: [fast, code, vision, tool_use, structured_output]
    privacy: [PUBLIC, INTERNAL]

  o4-mini:
    provider: openai
    context_window: 200000
    max_output: 100000
    cost_per_1k_input: 0.0011
    cost_per_1k_output: 0.0044
    capabilities: [deep_reasoning, code, vision, tool_use]
    privacy: [PUBLIC, INTERNAL]

  # ── Local (Ollama) ─────────────────────────
  qwen2.5:latest:
    provider: ollama
    context_window: 32768
    capabilities: [reasoning, code, multilingual]
    privacy: [PRIVATE, INTERNAL, PUBLIC]    # can handle ALL tiers

  llama3.2:latest:
    provider: ollama
    context_window: 131072
    capabilities: [reasoning, code]
    privacy: [PRIVATE, INTERNAL, PUBLIC]

  gemma3:4b:
    provider: ollama
    context_window: 8192
    capabilities: [fast, multilingual]
    privacy: [PRIVATE, INTERNAL, PUBLIC]

  deepseek-r1:latest:
    provider: ollama
    context_window: 65536
    capabilities: [deep_reasoning, code, math]
    privacy: [PRIVATE, INTERNAL, PUBLIC]

routing_rules:
  # Capability-based routing (new in v2)
  - match: {requires: "long_context", input_tokens_gt: 100000}
    prefer: [claude-sonnet-4-6, gemini-2.5-pro, gpt-4.1]
    reason: "Long context task — models with 1M+ window"

  - match: {requires: "deep_reasoning"}
    prefer: [o4-mini, claude-opus-4-6, deepseek-r1:latest]
    reason: "Complex reasoning — chain-of-thought models"

  - match: {requires: "vision"}
    prefer: [claude-sonnet-4-6, gemini-2.5-flash, gpt-4.1]
    reason: "Multi-modal input"

  - match: {privacy: "PRIVATE"}
    prefer: [qwen2.5:latest, llama3.2:latest, gemma3:4b, deepseek-r1:latest]
    reason: "PRIVATE data — local models only"

  - match: {complexity: "LOW", cost_priority: true}
    prefer: [gemini-2.5-flash, claude-haiku-4-5, gpt-4.1-mini, gemma3:4b]
    reason: "Low complexity — cheapest capable model"

  - match: {complexity: "HIGH"}
    prefer: [claude-opus-4-6, o4-mini, gemini-2.5-pro]
    reason: "High complexity — strongest reasoning"

  # Domain-specific
  - match: {domain: "ENGINEERING"}
    prefer: [claude-sonnet-4-6, gpt-4.1, deepseek-r1:latest]
    reason: "Code generation — strong coding models"

  - match: {domain: "CREATIVE"}
    prefer: [claude-sonnet-4-6, gemini-2.5-pro]
    reason: "Creative writing — expressive models"

  - match: {domain: "ANALYSIS"}
    prefer: [o4-mini, claude-opus-4-6, gemini-2.5-pro]
    reason: "Data analysis — reasoning-heavy"

  # Default fallback chain
  default_chain: [claude-sonnet-4-6, gemini-2.5-flash, qwen2.5:latest]
```

**New routing capabilities:**
- **Capability matching**: Route based on required capabilities (vision, audio, long_context, deep_reasoning)
- **Cost-aware**: Factor in estimated cost; downgrade when budget is tight
- **Input-length aware**: Switch to 1M-context models when input exceeds 100K tokens
- **Latency-aware**: Track P50/P99 latency per model and prefer faster ones for real-time use

### 3.2 LLM Client v2

```python
class LLMClientV2:
    """
    Unified async client — now with:
    - Structured output (JSON mode for all providers)
    - Tool use / function calling (native per provider)
    - Streaming with Server-Sent Events
    - Extended thinking / chain-of-thought capture
    - Multi-modal: text + image + audio + file
    - Automatic retry with circuit breaker
    - OpenTelemetry span creation per call
    """

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        system: str = "",
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
        tools: list[ToolDef] | None = None,
        structured_output: type[BaseModel] | None = None,  # Pydantic model
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,        # Enable extended thinking (Claude/o4)
        thinking_budget: int = 10000,   # Max thinking tokens
    ) -> LLMResponse | AsyncIterator[LLMChunk]:
        ...
```

### 3.3 Configuration Store

**New:** Centralized, typed configuration with hot-reload support.

```python
class ConfigStore:
    """
    Reads from YAML files + env vars + Vault secrets.
    Supports:
    - Live reload via file watcher (no restart needed)
    - Typed access: config.get("llm.default_model", str)
    - Layered: defaults < config file < env var < runtime override
    - Schema validation via Pydantic
    """
```

### 3.4 Secret Vault

```python
class SecretVault:
    """
    Manages API keys and credentials:
    - .env file (development)
    - OS keychain (macOS Keychain, Linux libsecret)
    - HashiCorp Vault / AWS Secrets Manager (production)
    - In-memory cache with TTL
    - Audit trail for secret access
    """
```

---

## 4. Layer 1 — Ingestion & Triggers

### Extended trigger sources (v2 additions in bold)

| Trigger | Transport | Use Case |
|---------|-----------|----------|
| CLI | stdin/stdout | Interactive terminal |
| REST API | HTTP POST | External integrations |
| **WebSocket** | **WS** | **Real-time bidirectional (chat UIs)** |
| Scheduler | APScheduler | Periodic tasks |
| File Watcher | watchdog | File system events |
| **A2A Inbound** | **HTTP POST** | **Incoming tasks from remote agents** |
| Webhook | HTTP POST | GitHub, Slack, etc. |
| **Message Queue** | **Redis Streams / NATS** | **High-throughput async tasks** |
| **Git Hook** | Shell | Pre-commit, post-push |

### TaskEvent v2

```python
@dataclass
class TaskEvent:
    id:           str
    message:      str
    priority:     Priority           # NORMAL | HIGH | CRITICAL
    session_id:   str
    source:       TriggerSource      # CLI | REST | WS | A2A | SCHEDULER | ...
    attachments:  list[Attachment]   # NEW: files, images, audio
    parent_id:    str | None         # NEW: for sub-task tracking
    a2a_source:   str | None         # NEW: remote agent URL if A2A
    metadata:     dict               # Extensible context
    created_at:   datetime
```

---

## 5. Layer 2 — Perception & Routing

### Enhanced pipeline

```
TaskEvent
  │
  ├─► Stage 1: Fast-path rules (0ms, regex)
  │     domain, complexity, destructive flags
  │
  ├─► Stage 2: PII Scanner (0ms, regex)
  │     multi-pattern → privacy_tier escalation
  │
  ├─► Stage 3: Attachment Analysis (NEW)
  │     file type detection, image OCR preview, size checks
  │
  ├─► Stage 4: LLM Analysis (local model, always private)
  │     intent, task_type, agents, skills, entities
  │     NEW: required_capabilities (vision, long_context, etc.)
  │     NEW: estimated_input_tokens (for routing decisions)
  │
  └─► Output: PerceivedTask v2
```

### PerceivedTask v2

```python
@dataclass
class PerceivedTask:
    # ... existing fields ...
    required_capabilities: list[str]  # ["vision", "long_context", "code"]
    estimated_tokens:      int        # For router input-length decisions
    suggested_pattern:     str        # Workflow pattern hint
    handoff_eligible:      bool       # Can this be handed off mid-execution?
    guardrails:            list[str]  # Which guardrail rules to apply
```

---

## 6. Layer 3 — Stateful Orchestration

This is the **biggest change in v2**. The orchestrator moves from a simple pattern dispatcher to a **stateful execution graph** inspired by LangGraph.

### 6.1 Workflow Graph

```python
class WorkflowGraph:
    """
    Directed graph where:
    - Nodes = agent/skill executions (steps)
    - Edges = transitions with conditions
    - State = accumulated data flowing through the graph

    Inspired by LangGraph's StateGraph + OpenAI's Handoff pattern.
    """

    def add_node(self, name: str, executor: Callable) -> None: ...
    def add_edge(self, source: str, target: str) -> None: ...
    def add_conditional_edge(
        self, source: str, condition: Callable, targets: dict[str, str]
    ) -> None: ...
    def set_entry_point(self, name: str) -> None: ...
    def set_finish_point(self, name: str) -> None: ...
    def compile(self) -> CompiledGraph: ...
```

**Example: Research workflow as a graph**

```python
graph = WorkflowGraph()

# Nodes
graph.add_node("search",    web_agent.execute)
graph.add_node("retrieve",  rag_agent.execute)
graph.add_node("synthesize", writer_agent.execute)
graph.add_node("review",    critic_agent.execute)

# Edges
graph.add_edge("search", "retrieve")       # search then retrieve
graph.add_edge("retrieve", "synthesize")    # retrieve then synthesize
graph.add_conditional_edge(
    "review",
    lambda state: "synthesize" if state["quality"] < 0.75 else "end",
    {"synthesize": "synthesize", "end": END},
)

# Flow: search → retrieve → synthesize → review → (loop or end)
graph.add_edge("synthesize", "review")
graph.set_entry_point("search")
```

### 6.2 Checkpoint Persistence (Durable Execution)

```python
class CheckpointStore:
    """
    Persists workflow state after every node execution.
    Enables:
    - Crash recovery: resume from last successful step
    - Pause/resume: user can pause, close terminal, resume later
    - Time-travel debugging: replay from any checkpoint
    - Branching: fork execution from a checkpoint

    Backends:
    - SQLite (default, local)
    - Redis (distributed)
    - PostgreSQL (production)
    """

    async def save(self, thread_id: str, step: int, state: dict) -> str: ...
    async def load(self, thread_id: str, step: int | None = None) -> dict: ...
    async def list_checkpoints(self, thread_id: str) -> list[CheckpointMeta]: ...
    async def fork(self, thread_id: str, step: int) -> str: ...  # Returns new thread_id
```

**Checkpoint data structure:**
```python
@dataclass
class Checkpoint:
    thread_id:   str        # Conversation/workflow thread
    step:        int        # Step number in the graph
    node_name:   str        # Which node just completed
    state:       dict       # Full accumulated state
    metadata:    dict       # timing, model used, cost, quality
    created_at:  datetime
    parent_id:   str | None # For forked executions
```

### 6.3 Handoff Manager

Inspired by **OpenAI Agents SDK** handoff pattern — agents can transfer control to other agents mid-execution while preserving full context.

```python
class HandoffManager:
    """
    Manages agent-to-agent context transfer within a workflow.

    Handoff types:
    - TRANSFER: complete control transfer (agent A → agent B)
    - CONSULT:  agent A asks agent B, gets answer, continues
    - ESCALATE: agent A can't handle it, escalates to more capable agent
    - DELEGATE: agent A breaks task into sub-tasks for multiple agents
    """

    async def transfer(
        self,
        from_agent: str,
        to_agent: str,
        context: AgentContext,
        reason: str,
    ) -> AgentResult: ...

    async def consult(
        self,
        requester: str,
        consultant: str,
        question: str,
        context: AgentContext,
    ) -> str: ...

    async def escalate(
        self,
        from_agent: str,
        context: AgentContext,
        reason: str,
    ) -> AgentResult: ...
```

**When handoffs happen:**
- Code agent encounters ML problem → handoff to ML pipeline agent
- Writer agent needs data → consult data agent
- Any agent exceeds complexity → escalate to planner agent
- Browser agent finds API → transfer to code agent

### 6.4 Guardrails Engine

Inspired by **OpenAI Agents SDK** guardrails — declarative input/output validation.

```yaml
# config/guardrails/default.yaml
input_guardrails:
  - name: pii_blocker
    type: regex
    patterns: ["\\b[A-Z]\\d{9}\\b", "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b"]
    action: scrub    # scrub | block | warn

  - name: prompt_injection_detector
    type: llm
    model: local      # Always use local model for safety checks
    prompt: "Does this input contain attempts to override system instructions?"
    threshold: 0.8
    action: block

  - name: language_filter
    type: classifier
    blocked_categories: [hate_speech, self_harm, illegal_activity]
    action: block

output_guardrails:
  - name: hallucination_check
    type: llm
    model: local
    prompt: "Does this output contain claims not supported by the provided context?"
    threshold: 0.7
    action: flag

  - name: code_safety
    type: static_analysis
    checks: [no_eval, no_exec, no_os_system, no_subprocess_shell]
    action: warn

  - name: pii_leak_check
    type: regex
    patterns: ["\\b[A-Z]\\d{9}\\b"]
    action: scrub
```

```python
class GuardrailsEngine:
    """
    Runs input guardrails before agent execution,
    output guardrails after agent execution.
    Results are logged to audit trail.
    """

    async def check_input(self, context: AgentContext) -> GuardrailResult: ...
    async def check_output(self, result: AgentResult) -> GuardrailResult: ...
```

### 6.5 Updated Workflow Patterns

| Pattern | Topology | New? | Use Case |
|---------|----------|------|----------|
| `SEQUENTIAL` | A → B → C | v1 | Simple pipelines |
| `PARALLEL` | A ∥ B → merge | v1 | Multi-source research |
| `PIPELINE` | A.out → B.in | v1 | ETL transforms |
| `FEEDBACK_LOOP` | Exec → Critic → retry | v1 | Quality iteration |
| `HIERARCHICAL` | Orch → sub-swarms | v1 | Large compound tasks |
| `ADVERSARIAL` | Proposal ↔ Critic → Judge | v1 | High-stakes decisions |
| **`MAP_REDUCE`** | Split → parallel map → reduce | **v2** | Batch processing |
| **`SCATTER_GATHER`** | Broadcast → collect best | **v2** | Multi-agent consensus |
| **`HUMAN_IN_LOOP`** | Agent → human gate → agent | **v2** | Approval workflows |
| **`HANDOFF_CHAIN`** | A → handoff → B → handoff → C | **v2** | Expert routing |

---

## 7. Layer 4 — Execution Runtime

### 7.1 Agent Base v2

```python
class BaseAgentV2(ABC):
    """
    Enhanced base agent with:
    - Tool use via native LLM function calling
    - Handoff support (can request transfer to another agent)
    - Guardrails integration (input/output checks)
    - Streaming output (SSE for real-time UIs)
    - Extended thinking capture
    """

    # Existing
    agent_id:          str
    domain:            TaskDomain
    default_complexity: TaskComplexity
    default_privacy:   PrivacyTier

    # New in v2
    tools:             list[ToolDef]       # Native tool definitions
    handoff_targets:   list[str]           # Agents this agent can hand off to
    guardrail_rules:   list[str]           # Which guardrail configs apply
    max_thinking_tokens: int = 0           # 0 = no extended thinking

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult: ...

    # New helper methods
    async def use_tool(self, tool_name: str, **kwargs) -> Any: ...
    async def handoff(self, to_agent: str, reason: str) -> AgentResult: ...
    async def think(self, prompt: str) -> str: ...  # Extended thinking
    async def stream(self, context: AgentContext) -> AsyncIterator[str]: ...
```

### 7.2 Skill System v2 — Hot-Reloadable

```python
class SkillRegistryV2:
    """
    Enhanced with:
    - Hot reload: file watcher on skills/implementations/
    - Deferred loading: skills loaded on first use (Anthropic Tool Search pattern)
    - Dependency resolution: skills can declare dependencies on other skills
    - Sandboxed execution: skills run in subprocess with resource limits
    """

    def load_on_demand(self, name: str) -> BaseSkill:
        """Load a skill only when first requested (deferred loading)."""
        ...

    def reload(self, name: str) -> None:
        """Hot-reload a skill without restarting the system."""
        ...

    def search(self, query: str) -> list[SkillMeta]:
        """Semantic search for skills by description (Tool Search pattern)."""
        ...
```

### 7.3 MCP v2 Transport

```python
class MCPTransport(Protocol):
    """Transport-agnostic MCP communication."""
    async def send(self, message: dict) -> dict: ...
    async def stream(self, message: dict) -> AsyncIterator[dict]: ...
    async def close(self) -> None: ...

class StdioTransport(MCPTransport):
    """Legacy stdio transport (subprocess)."""
    ...

class StreamableHTTPTransport(MCPTransport):
    """
    New in MCP spec — replaces deprecated SSE transport.
    Uses standard HTTP POST with streaming response.
    Supports:
    - Stateless and stateful sessions
    - Server-initiated notifications via SSE
    - Automatic reconnection
    - OAuth 2.1 authentication
    """
    ...

class MCPClientV2:
    """
    Enhanced MCP client:
    - Auto-selects transport (HTTP preferred, stdio fallback)
    - Connection pooling for HTTP transport
    - Tool result caching with TTL
    - Concurrent requests to multiple servers
    - Server health monitoring
    """

    async def call(
        self,
        server: str,
        tool: str,
        args: dict,
        *,
        transport: str = "auto",     # "auto" | "http" | "stdio"
        timeout: float = 30.0,
        cache_ttl: float = 0.0,      # 0 = no cache
    ) -> Any: ...

    async def discover_tools(self, server: str) -> list[ToolDef]: ...
    async def health_check(self, server: str) -> bool: ...
```

**MCP Server configuration v2:**

```yaml
# config/mcp/servers.yaml
servers:
  filesystem:
    transport: stdio          # Local process
    command: ["python", "-m", "nexus.mcp.servers.filesystem_server"]
    auto_start: true

  github:
    transport: http           # Streamable HTTP (new)
    url: "http://localhost:3100/mcp"
    auth:
      type: bearer
      token_env: GITHUB_TOKEN
    health_check: "/health"

  chroma:
    transport: http
    url: "http://localhost:8200/mcp"
    connection_pool_size: 5

  # Remote MCP server (cloud-hosted)
  hosted_search:
    transport: http
    url: "https://mcp.example.com/search"
    auth:
      type: oauth2
      client_id_env: SEARCH_CLIENT_ID
      client_secret_env: SEARCH_CLIENT_SECRET
      token_url: "https://auth.example.com/token"
```

---

## 8. Layer 5 — Memory & Knowledge

### 8.1 Enhanced Memory Types

| Type | Backend | TTL | New Changes |
|------|---------|-----|-------------|
| Working | AgentContext | Task lifetime | Checkpoint-backed |
| Short-term | Redis | 48h | Stream-based (Redis Streams) |
| Episodic | ChromaDB | Permanent | **Privacy-tier indexed** |
| Semantic | ChromaDB | Permanent | **Hybrid retrieval v2** |
| Procedural | File system | Permanent | **Hot-reloadable** |
| **Graph** | **Neo4j / NetworkX** | **Permanent** | **Entity relationships** |

### 8.2 RAG Engine v2 — Improved Retrieval

```python
class RAGEngineV2:
    """
    Enhancements:
    1. Privacy-tier mandatory (not optional filter)
    2. sentence-transformers fallback (not SHA256 hash)
    3. Re-ranking with cross-encoder
    4. Contextual compression (extract only relevant chunks)
    5. Multi-index: separate collections per document type
    6. Hybrid scoring: configurable weights
    """

    async def retrieve(
        self,
        query: str,
        *,
        privacy_tier: PrivacyTier,           # REQUIRED (was optional in v1)
        top_k: int = 10,
        rerank: bool = True,                 # Cross-encoder reranking
        compress: bool = False,              # Contextual compression
        doc_types: list[DocumentType] | None = None,
        time_decay: bool = True,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.3,
        recency_weight: float = 0.1,
    ) -> list[RetrievalResult]: ...
```

**Embedding fallback chain (fixed from v1):**
```
1. Ollama bge-m3 (local, best quality)
2. OpenAI text-embedding-3-small (cloud, fast)
3. sentence-transformers all-MiniLM-L6-v2 (local, no server needed)  ← NEW
4. ❌ SHA256 hash (REMOVED — produced random, useless embeddings)
```

### 8.3 Memory Consolidation (Background Process)

```python
class MemoryConsolidator:
    """
    Periodic background task that:
    1. Clusters similar episodic memories into semantic summaries
    2. Decays rarely-accessed memories (lower retrieval priority)
    3. Promotes frequently-accessed patterns to procedural memory
    4. Deduplicates near-identical entries (cosine similarity > 0.95)
    5. Generates "lessons learned" from failure patterns

    Schedule: runs every 6 hours or after 100 new memories
    """
```

---

## 9. Layer 6 — Federation & A2A

### 9.1 A2A Protocol Implementation

Based on **Google's Agent-to-Agent protocol** (donated to Linux Foundation AAIF).

```
NEXUS Instance A                    NEXUS Instance B
     │                                     │
     ├─► GET /.well-known/agent.json       │  (Discovery)
     │   ◄── AgentCard{name, skills, auth} │
     │                                     │
     ├─► POST /tasks/send                  │  (Delegation)
     │   {message, skill_id, metadata}     │
     │   ◄── {task_id, status: "working"}  │
     │                                     │
     ├─► POST /tasks/get                   │  (Polling)
     │   ◄── {status: "completed",         │
     │        artifacts: [{text: "..."}]}   │
     │                                     │
     ├─► POST /tasks/sendSubscribe         │  (Streaming — NEW)
     │   ◄── SSE stream of status updates  │
```

### 9.2 AgentCard (Published at `/.well-known/agent.json`)

```json
{
  "name": "NEXUS",
  "url": "https://nexus.example.com",
  "version": "2.0",
  "description": "Enterprise AI Agent Orchestration Platform",
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "skills": [
    {"id": "research", "name": "Research", "description": "Web search + RAG"},
    {"id": "code", "name": "Code Generation", "description": "Write & execute code"},
    {"id": "data", "name": "Data Analysis", "description": "CSV/Excel analysis"},
    {"id": "browser", "name": "Browser Control", "description": "Web automation"},
    {"id": "planning", "name": "Task Planning", "description": "DAG decomposition"},
    {"id": "writing", "name": "Writing", "description": "Long-form documents"},
    {"id": "memory", "name": "Memory", "description": "Knowledge recall"},
    {"id": "ml", "name": "ML Pipeline", "description": "Model training & eval"},
    {"id": "governance", "name": "Governance", "description": "Compliance & bias audit"}
  ],
  "authentication": {
    "schemes": ["Bearer", "OAuth2"]
  },
  "provider": {
    "organization": "NEXUS Platform",
    "url": "https://nexus.example.com"
  }
}
```

### 9.3 Multi-Instance Federation

```
                    ┌─────────────┐
                    │  Discovery  │
                    │  Registry   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                  │
    ┌────▼────┐      ┌────▼────┐       ┌────▼────┐
    │ NEXUS-1 │◄────►│ NEXUS-2 │◄─────►│ NEXUS-3 │
    │ Research │      │ Code    │       │ Data    │
    │ Cluster  │      │ Cluster │       │ Cluster │
    └─────────┘      └─────────┘       └─────────┘
         │                                    │
         └──── A2A Tasks/Send ────────────────┘
```

---

## 10. Cross-Cutting: Governance & Compliance

### 10.1 EU AI Act Classifier v2

```python
class EUAIActClassifierV2:
    """
    Enhancements:
    - Article 5 prohibited AI detection (6 → 12 patterns)
    - Annex III high-risk categories (12 → 18 rules)
    - NEW: Article 50 transparency obligations for GPAI
    - NEW: Risk scoring (0.0–1.0) instead of binary classification
    - NEW: Explanation generation (why this classification)
    - NEW: Audit trail of all classifications
    """
```

### 10.2 PII Scrubber v2

```python
class PIIScrubberV2:
    """
    Enhancements over v1:
    - Taiwan National ID, Resident Certificate, Passport
    - GDPR entity detection (EU patterns)
    - Japanese My Number, Korean RRN
    - Medical record numbers
    - Biometric data indicators
    - Named entity recognition (spaCy) for person/org names
    - Configurable: scrub vs. tokenize (reversible pseudonymization)
    """
```

### 10.3 Audit Logger v2

```python
class AuditLoggerV2:
    """
    Enhancements:
    - Encrypted at rest (Fernet symmetric encryption)
    - Structured JSON format (not just SQLite columns)
    - Log rotation and archival
    - Export to SIEM (Splunk, ELK) via structured logging
    - Tamper-proof: append-only with hash chains
    - Query API: filter by agent, model, date range, cost
    """
```

---

## 11. Cross-Cutting: Observability & Tracing

### OpenTelemetry Integration

```python
class TracingExporter:
    """
    Every LLM call, agent execution, and MCP request creates an
    OpenTelemetry span. Distributed tracing across the full pipeline.

    Backends:
    - Console (development)
    - Jaeger (local)
    - Datadog / New Relic / Grafana Tempo (production)
    """
```

**Trace hierarchy:**
```
Task[task_123]
  └─ Perception[perception_123]
       ├─ FastPath[fast_path_123]
       ├─ PIIScan[pii_scan_123]
       └─ LLMAnalysis[llm_analysis_123]
            └─ LLMCall[llm_call_123] (model=qwen2.5, tokens=150)
  └─ Orchestration[orch_123]
       ├─ EUAIActCheck[eu_check_123]
       └─ WorkflowExec[workflow_123]
            ├─ Step[web_agent] (duration=2.3s, cost=$0.003)
            ├─ Step[rag_agent] (duration=0.8s, cost=$0.001)
            ├─ Step[writer_agent] (duration=3.1s, cost=$0.005)
            └─ Step[critic_agent] (duration=1.2s, cost=$0.002)
  └─ MemoryWrite[memory_123]
```

### Metrics Dashboard

```yaml
# Key metrics exposed via Prometheus
nexus_tasks_total{status, domain, pattern}
nexus_llm_calls_total{provider, model, privacy_tier}
nexus_llm_cost_usd{provider, model}
nexus_llm_latency_seconds{provider, model, quantile}
nexus_agent_quality_score{agent_id}
nexus_mcp_calls_total{server, tool, status}
nexus_guardrail_triggers_total{rule, action}
nexus_checkpoint_operations_total{operation}
nexus_a2a_tasks_total{remote_agent, status}
nexus_memory_operations_total{type, operation}
```

---

## 12. Cross-Cutting: Cost Intelligence

### Cost Optimizer v2

```python
class CostIntelligenceEngine:
    """
    Beyond simple budget limits:
    - Real-time cost tracking per task/agent/model
    - Predictive budgeting: forecast monthly spend from current trends
    - Smart downgrade: downgrade models only for low-complexity sub-tasks
    - Cost attribution: per-user, per-team, per-project cost reports
    - Token optimization: detect and eliminate redundant context
    - Caching layer: skip LLM calls for previously-seen queries
    """
```

**Downgrade chain v2 (updated models):**
```
claude-opus-4-6     → claude-sonnet-4-6    → claude-haiku-4-5
gemini-2.5-pro      → gemini-2.5-flash
gpt-4.1             → gpt-4.1-mini
o4-mini             → gpt-4.1-mini
deepseek-r1:latest  → qwen2.5:latest       → gemma3:4b
```

**Smart caching:**
```python
class LLMCache:
    """
    Semantic cache: if a new query is >0.92 cosine similar to a
    cached query AND the cached response is <24h old, return cached.
    Saves ~30-40% of LLM costs in typical usage.
    """
```

---

## 13. MCP v2 — Streamable HTTP Transport

### Why Streamable HTTP?

The MCP specification deprecated SSE transport in favor of **Streamable HTTP** (2025):
- **Standard HTTP**: works through firewalls, proxies, load balancers
- **Stateless by default**: no persistent connections needed
- **Optional streaming**: server can upgrade to SSE when needed
- **OAuth 2.1**: first-class authentication
- **Better scaling**: connection pooling, HTTP/2 multiplexing

### Migration from stdio

```
v1 (stdio only):
  MCPClient → subprocess.Popen → stdin/stdout JSON-RPC

v2 (hybrid):
  MCPClient → StdioTransport (local servers, unchanged)
            → StreamableHTTPTransport (remote servers, NEW)
            → Auto-detect based on server config
```

### Remote MCP Servers

```yaml
# v2 enables connecting to hosted MCP services
remote_servers:
  - name: anthropic_web_search
    url: "https://mcp.anthropic.com/search"
    auth: oauth2

  - name: github_mcp
    url: "https://api.github.com/mcp"
    auth: bearer
    token_env: GITHUB_TOKEN

  - name: company_knowledge_base
    url: "https://kb.internal.company/mcp"
    auth: bearer
    token_env: KB_TOKEN
```

---

## 14. LLM Router v2 — Multi-Provider Intelligence

### Capability-Based Routing

```python
class LLMRouterV2:
    """
    New routing dimensions:
    1. Required capabilities (vision, audio, long_context, tools, thinking)
    2. Input token count (route to 1M-context models when needed)
    3. Latency requirement (real-time vs. batch)
    4. Cost budget remaining
    5. Provider health (circuit breaker on failing providers)
    6. Geographic compliance (EU data → EU endpoints)
    """

    async def route(
        self,
        task: RoutingRequest,
    ) -> RoutingDecision:
        # 1. Filter by privacy tier
        candidates = self._filter_privacy(task.privacy_tier)

        # 2. Filter by required capabilities
        candidates = self._filter_capabilities(candidates, task.capabilities)

        # 3. Filter by context window (input tokens)
        candidates = self._filter_context_window(candidates, task.estimated_tokens)

        # 4. Score by preference (complexity, domain, cost, latency)
        scored = self._score_candidates(candidates, task)

        # 5. Apply circuit breaker (skip unhealthy providers)
        scored = self._apply_circuit_breaker(scored)

        # 6. Select primary + fallback
        return RoutingDecision(
            primary=scored[0].model_id,
            fallback=scored[1].model_id if len(scored) > 1 else None,
            reasoning=self._explain(scored, task),
            estimated_cost=scored[0].estimated_cost,
        )
```

### Provider Health Monitoring

```python
class ProviderCircuitBreaker:
    """
    Tracks success/failure rates per provider.
    States: CLOSED (healthy) → OPEN (failing) → HALF_OPEN (testing)

    - 3 consecutive failures → OPEN (skip this provider for 60s)
    - After 60s → HALF_OPEN (try one request)
    - If succeeds → CLOSED; if fails → OPEN again
    """
```

---

## 15. Plugin System

Inspired by **Dify's plugin-first architecture** — everything is extensible.

### Plugin Types

| Type | Interface | Discovery | Hot-Reload |
|------|-----------|-----------|------------|
| Agent | `BaseAgentV2` | `plugins/agents/` | ✅ |
| Skill | `BaseSkill` | `plugins/skills/` | ✅ |
| MCP Server | `MCPServer` | `plugins/mcp/` | ✅ |
| Guardrail | `GuardrailRule` | `plugins/guardrails/` | ✅ |
| Trigger | `TriggerSource` | `plugins/triggers/` | ✅ |
| Embedding | `EmbeddingProvider` | `plugins/embeddings/` | ✅ |

### Plugin Manifest

```yaml
# plugins/agents/my_custom_agent/plugin.yaml
name: my_custom_agent
version: "1.0.0"
type: agent
description: "Custom agent for XYZ domain"
author: "team@example.com"
requires:
  python: ">=3.11"
  packages: ["pandas", "scikit-learn"]
  skills: ["excel_designer"]
  mcp: ["filesystem"]
entry_point: "agent.py:MyCustomAgent"
```

### Plugin Loader

```python
class PluginLoader:
    """
    - Scans plugins/ directory on startup
    - Validates manifest and dependencies
    - Loads in sandboxed environment
    - Watches for file changes → hot-reload
    - Reports health status per plugin
    """

    def load_all(self) -> PluginReport: ...
    def reload(self, plugin_name: str) -> bool: ...
    def unload(self, plugin_name: str) -> bool: ...
    def list_available(self) -> list[PluginManifest]: ...
```

---

## 16. Security Model

### Defense in Depth

```
Layer 1: Network
  - TLS 1.3 for all external communications
  - mTLS for inter-NEXUS A2A calls
  - API rate limiting per client

Layer 2: Authentication
  - OAuth 2.1 for MCP HTTP transport
  - Bearer tokens for API access
  - API key rotation via SecretVault

Layer 3: Authorization
  - Role-based access: admin, operator, viewer
  - Privacy tier enforcement at router level
  - Per-agent permission scopes

Layer 4: Data Protection
  - PII scrubbing before any LLM call
  - Audit log encryption (Fernet)
  - Memory encryption at rest (optional)
  - No PII in checkpoint state

Layer 5: Runtime
  - Skill execution in sandboxed subprocess
  - Shell command allowlist
  - Resource limits (CPU, memory, time)
  - Guardrails on all inputs/outputs
```

### Prompt Injection Defense

```python
class PromptInjectionDetector:
    """
    Multi-layer defense:
    1. Static rules: detect common injection patterns
    2. LLM classifier: local model checks for override attempts
    3. Output verification: ensure response follows expected format
    4. Canary tokens: embed hidden markers in system prompts
    """
```

---

## 17. Deployment Architecture

### Local Development

```bash
# Minimal: Ollama only (zero cost)
python nexus.py init
python nexus.py start

# Full local: Docker Compose
docker compose up -d  # Ollama, Redis, ChromaDB, Neo4j, Prometheus, Grafana
python nexus.py start
```

### Production (Kubernetes)

```
┌─────────────────────────────────────────────────┐
│  Kubernetes Cluster                              │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ NEXUS API    │  │ NEXUS Workers (3 pods)    │ │
│  │ (FastAPI)    │  │ - Agent execution         │ │
│  │ - REST       │  │ - Checkpoint persistence  │ │
│  │ - WebSocket  │  │ - MCP server management   │ │
│  │ - A2A        │  │                           │ │
│  └──────────────┘  └──────────────────────────┘ │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ Redis        │  │ PostgreSQL   │             │
│  │ (sessions +  │  │ (checkpoints │             │
│  │  cache)      │  │  + audit)    │             │
│  └──────────────┘  └──────────────┘             │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐             │
│  │ ChromaDB     │  │ Neo4j        │             │
│  │ (vectors)    │  │ (graph)      │             │
│  └──────────────┘  └──────────────┘             │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ Observability: Prometheus + Grafana +     │   │
│  │               Jaeger (tracing)            │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

---

## 18. Migration Path from v1

### Phase 1: Foundation (Weeks 1–2)
- [ ] Add `CheckpointStore` (SQLite backend)
- [ ] Implement `WorkflowGraph` alongside existing pattern executor
- [ ] Add `sentence-transformers` embedding fallback (remove SHA256)
- [ ] Make `privacy_tier` required in RAG retrieval
- [ ] Add Fernet encryption to `AuditLogger`

### Phase 2: Transport & Routing (Weeks 3–4)
- [ ] Implement `StreamableHTTPTransport` for MCP
- [ ] Update `LLMRouter` with capability-based routing
- [ ] Add new models (Gemini 2.5, GPT-4.1, o4-mini, DeepSeek-R1)
- [ ] Implement `ProviderCircuitBreaker`
- [ ] Add `LLMCache` (semantic caching layer)

### Phase 3: Orchestration (Weeks 5–6)
- [ ] Implement `HandoffManager`
- [ ] Implement `GuardrailsEngine` with YAML config
- [ ] Add `MAP_REDUCE` and `SCATTER_GATHER` patterns
- [ ] Add `HUMAN_IN_LOOP` pattern
- [ ] Integrate checkpoint save/restore into Harness

### Phase 4: Observability & Federation (Weeks 7–8)
- [ ] Add OpenTelemetry tracing throughout pipeline
- [ ] Implement A2A `/tasks/sendSubscribe` (streaming)
- [ ] Add Prometheus metrics for all new components
- [ ] Implement `CostIntelligenceEngine` (predictive budgeting)
- [ ] Deploy Grafana dashboards

### Phase 5: Plugin System & Polish (Weeks 9–10)
- [ ] Implement `PluginLoader` with hot-reload
- [ ] Migrate existing skills to plugin format
- [ ] Add `PromptInjectionDetector`
- [ ] Add `MemoryConsolidator` background process
- [ ] Performance testing & optimization

### Backward Compatibility
- All existing agents work unchanged (BaseAgent → BaseAgentV2 is additive)
- Existing YAML swarm configs remain valid
- stdio MCP servers continue to work alongside HTTP
- Existing .env configuration is preserved
- Existing audit.db data is migrated (not lost)

---

## 19. Architectural Decision Records

### ADR-6: Stateful graph orchestration over static patterns

**Decision**: Replace pattern-based dispatch with a stateful `WorkflowGraph`.

**Context**: LangGraph (30k+ stars), CrewAI, and AutoGen all converge on graph-based workflow execution. Static patterns (sequential, parallel) can't express conditional branching, human-in-the-loop, or dynamic agent selection.

**Trade-offs**:
- (+) Supports any workflow topology
- (+) Checkpoint persistence enables crash recovery
- (+) Time-travel debugging for complex failures
- (-) More complex than pattern dispatch
- (-) Requires checkpoint storage backend

**Migration**: Existing patterns become pre-built graph templates. No breaking changes.

---

### ADR-7: Streamable HTTP as primary MCP transport

**Decision**: Adopt Streamable HTTP for new MCP servers; keep stdio for local servers.

**Context**: MCP specification deprecated SSE transport (2025). Streamable HTTP is the new standard, supported by Anthropic, OpenAI, Google, and the AAIF (Linux Foundation).

**Trade-offs**:
- (+) Works through firewalls and proxies
- (+) Standard OAuth 2.1 authentication
- (+) Connection pooling, HTTP/2 multiplexing
- (+) Remote MCP servers become possible
- (-) More infrastructure than subprocess stdio

---

### ADR-8: Handoffs as first-class primitive

**Decision**: Implement agent handoffs at the orchestration layer.

**Context**: OpenAI Agents SDK (2025) identified Handoffs, Guardrails, and Tracing as the three essential agent primitives. CrewAI and AutoGen have similar delegation patterns.

**Trade-offs**:
- (+) Natural expert routing (code agent → ML agent)
- (+) Reduces orchestrator complexity (agents self-organize)
- (-) Potential infinite handoff loops (mitigated by max_handoffs)

---

### ADR-9: Declarative guardrails over code-based validation

**Decision**: Guardrails defined in YAML, not Python.

**Context**: OpenAI Agents SDK guardrails, Anthropic's constitutional AI approach, and Dify's visual guardrail builder all trend toward declarative safety rules.

**Trade-offs**:
- (+) Non-engineers can configure safety rules
- (+) Auditable (YAML is human-readable)
- (+) Hot-reloadable without code changes
- (-) Less flexible than arbitrary Python code (mitigated by custom rule types)

---

### ADR-10: Plugin-first architecture

**Decision**: All extensibility through plugins (not core code changes).

**Context**: Dify (114k+ stars) proved that plugin-first architecture dramatically improves community contribution. LangChain's integration explosion (700+ integrations) followed a similar pattern.

**Trade-offs**:
- (+) Community can extend without forking
- (+) Hot-reloadable
- (+) Sandboxed execution limits blast radius
- (-) Plugin API must be stable (versioned interfaces)
- (-) Performance overhead of sandboxing

---

## Appendix A: Industry Research Sources

| Source | Key Insight Applied to NEXUS v2 |
|--------|-------------------------------|
| **LangGraph** (30k stars) | Stateful graph execution, checkpoint persistence, time-travel |
| **AutoGPT** (167k stars) | Autonomous agent loops, goal decomposition |
| **Dify** (114k stars) | Plugin architecture, visual workflows, RAG pipeline |
| **CrewAI** (44.6k stars) | Role-based agents, structured collaboration |
| **AutoGen** (30k stars) | Multi-agent conversations, human-in-loop |
| **Anthropic MCP** | Streamable HTTP, Tool Search, deferred loading |
| **Google A2A** | Agent-to-Agent protocol, AgentCard, federation |
| **OpenAI Agents SDK** | Handoffs, Guardrails, Tracing as primitives |
| **Google ADK** | Interactions API, background execution |
| **n8n** (77k stars) | Visual workflow automation, 400+ integrations |
| **Langflow** (145k stars) | Visual agent builder, component marketplace |
| **Ollama** (120k+ stars) | Local LLM hosting, model management |
| **DeepSeek** (100k+ stars) | Open-source reasoning models, MoE architecture |

---

## Appendix B: Model Comparison Matrix (2026)

| Model | Provider | Context | Cost (1K in/out) | Strengths |
|-------|----------|---------|-------------------|-----------|
| Claude Opus 4.6 | Anthropic | 200K | $15/$75 | Deepest reasoning |
| Claude Sonnet 4.6 | Anthropic | 1M | $3/$15 | Best balance, huge context |
| Claude Haiku 4.5 | Anthropic | 200K | $0.8/$4 | Fast, cheap |
| Gemini 2.5 Pro | Google | 1M | $1.25/$10 | Multimodal, thinking |
| Gemini 2.5 Flash | Google | 1M | $0.075/$0.3 | Extremely cheap |
| GPT-4.1 | OpenAI | 1M | $2/$8 | Strong coding |
| GPT-4.1-mini | OpenAI | 1M | $0.4/$1.6 | Good balance |
| o4-mini | OpenAI | 200K | $1.1/$4.4 | Deep reasoning |
| DeepSeek-R1 | Local/API | 64K | Free (local) | Open-source reasoning |
| Qwen 2.5 | Local | 32K | Free (local) | Multilingual |
| Llama 3.2 | Local | 128K | Free (local) | General purpose |

---

*Document version: 2.0.0 — Generated 2026-02-27*
*Based on industry research across 50+ GitHub projects and 4 major AI providers*
