# NEXUS v2 — Complete Architecture Plan

> 此文件為 NEXUS v2 的完整架構規劃，涵蓋目錄結構、每個模組介面、
> 資料流、狀態機、設定格式、資料庫結構、API、測試策略及遷移路徑。
> 所有設計均基於 v1 程式碼審計 + 業界研究（Anthropic MCP / Google A2A /
> OpenAI Agents SDK / LangGraph / Dify）。

---

## 目錄

- [Part 1: 設計原則與全域約束](#part-1-設計原則與全域約束)
- [Part 2: 目錄結構（完整檔案清單）](#part-2-目錄結構)
- [Part 3: Layer 0 — Foundation Platform](#part-3-layer-0--foundation-platform)
- [Part 4: Layer 1 — Trigger & Ingestion](#part-4-layer-1--trigger--ingestion)
- [Part 5: Layer 2 — Perception & Routing](#part-5-layer-2--perception--routing)
- [Part 6: Layer 3 — Stateful Orchestration](#part-6-layer-3--stateful-orchestration)
- [Part 7: Layer 4 — Execution Runtime](#part-7-layer-4--execution-runtime)
- [Part 8: Layer 5 — Memory & Knowledge](#part-8-layer-5--memory--knowledge)
- [Part 9: Layer 6 — Federation (A2A)](#part-9-layer-6--federation-a2a)
- [Part 10: Cross-Cutting — Governance](#part-10-cross-cutting--governance)
- [Part 11: Cross-Cutting — Observability](#part-11-cross-cutting--observability)
- [Part 12: Cross-Cutting — Security](#part-12-cross-cutting--security)
- [Part 13: API & Dashboard](#part-13-api--dashboard)
- [Part 14: Configuration Schema](#part-14-configuration-schema)
- [Part 15: Database Schema](#part-15-database-schema)
- [Part 16: 狀態機與生命週期](#part-16-狀態機與生命週期)
- [Part 17: 資料流（End-to-End）](#part-17-資料流)
- [Part 18: 錯誤處理策略](#part-18-錯誤處理策略)
- [Part 19: 測試策略](#part-19-測試策略)
- [Part 20: 效能需求與限制](#part-20-效能需求與限制)
- [Part 21: 遷移計畫（v1 → v2）](#part-21-遷移計畫)
- [Part 22: 技術選型理由](#part-22-技術選型理由)

---

## Part 1: 設計原則與全域約束

### 1.1 保留原則（v1 已驗證）

| # | 原則 | 實施方式 |
|---|------|----------|
| P1 | Privacy is routing | `PrivacyTier` 嵌入 `LLMRouter.route()`，PRIVATE 任務結構上無法觸及雲端模型 |
| P2 | Governance is infrastructure | EU AI Act + PII + Audit 在 `dispatch()` 前執行，不可跳過 |
| P3 | Composition over inheritance | Agent / Skill / MCP 三軸正交，新增功能 = 新增一個檔案 |
| P4 | Quality feedback loop | quality_score → prompt_versioning → PromptOptimizer → A/B 測試 |

### 1.2 新增原則（v2）

| # | 原則 | 來源 | 實施方式 |
|---|------|------|----------|
| P5 | Durable execution | LangGraph | 每個 workflow step 產出 checkpoint，crash-safe resume |
| P6 | Federation-native | Google A2A | A2A AgentCard + task delegation 為一等公民 |
| P7 | Transport-agnostic MCP | Anthropic AAIF | Streamable HTTP 為主，stdio 為 fallback |
| P8 | Handoff as primitive | OpenAI Agents SDK | Agent 可在執行中轉移控制權給另一個 Agent |
| P9 | Declarative guardrails | OpenAI Agents SDK | Input/Output 驗證規則以 YAML 宣告 |
| P10 | Plugin-first | Dify | Agent / Skill / MCP / Guardrail 全部支援熱載入 |

### 1.3 全域約束

```
MAX_CONCURRENT_LLM_CALLS   = 5
MAX_CONCURRENT_AGENTS       = 10
MAX_TASK_COST_USD           = 1.00   (可設定)
MAX_TASK_TIMEOUT_SEC        = 600
MAX_HANDOFF_DEPTH           = 5      (防止無限遞迴)
MAX_FEEDBACK_LOOP_ROUNDS    = 3
MAX_CHECKPOINT_SIZE_MB      = 10
EMBEDDING_DIM               = 1024   (bge-m3)
CHUNK_SIZE                  = 512    (tokens)
CHUNK_OVERLAP               = 64     (tokens)
```

---

## Part 2: 目錄結構

```
nexus/                              # 專案根目錄
├── nexus.py                        # 主入口：init_system(), CLI dispatcher
├── ARCHITECTURE_PLAN.md            # 本文件
├── ARCHITECTURE_V2.md              # 設計概覽
├── ARCHITECTURE.md                 # v1 設計文件（保留）
├── README.md
├── ROADMAP.md
├── DEVELOPMENT.md
├── DEPLOYMENT.md
├── CLAUDE.md
├── .env                            # 環境變數（本地開發）
├── .env.example                    # 環境變數範本
├── requirements.txt                # Python 依賴
├── pyproject.toml                  # ★ NEW: 套件定義 + 工具設定
├── docker-compose.yml              # 基礎設施服務
├── docker-compose.prod.yml         # 生產部署
├── Dockerfile                      # NEXUS 容器
│
├── core/                           # Layer 0: Foundation
│   ├── __init__.py
│   ├── agents/                     # 所有 Agent 實作
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseAgent (v1) — 保留
│   │   ├── base_v2.py              # ★ NEW: BaseAgentV2 (extends BaseAgent)
│   │   ├── web_agent.py
│   │   ├── rag_agent.py
│   │   ├── code_agent.py
│   │   ├── shell_agent.py
│   │   ├── data_agent.py
│   │   ├── memory_agent.py
│   │   ├── email_agent.py
│   │   ├── browser_agent.py
│   │   ├── ml_pipeline_agent.py
│   │   ├── writer_agent.py
│   │   ├── bias_auditor_agent.py
│   │   ├── cost_optimizer_agent.py
│   │   ├── prompt_optimizer_agent.py
│   │   ├── planner_agent.py
│   │   ├── critic_agent.py
│   │   └── a2a_agent.py
│   │
│   ├── llm/                        # LLM 路由與客戶端
│   │   ├── __init__.py
│   │   ├── router.py               # LLMRouter (v1) — 保留
│   │   ├── router_v2.py            # ★ NEW: 能力路由 + 斷路器
│   │   ├── client.py               # LLMClient (v1) — 保留
│   │   ├── client_v2.py            # ★ NEW: structured output + streaming + thinking
│   │   ├── cache.py                # ★ NEW: 語意快取 (semantic cache)
│   │   └── circuit_breaker.py      # ★ NEW: Provider 健康監控
│   │
│   ├── orchestrator/               # Layer 2-3: 感知 + 編排
│   │   ├── __init__.py
│   │   ├── trigger.py              # Layer 1: TriggerManager — 保留 + 擴充
│   │   ├── perception.py           # Layer 2: PerceptionEngine — 保留 + 擴充
│   │   ├── master.py               # Layer 3: MasterOrchestrator — 保留
│   │   ├── master_v2.py            # ★ NEW: Graph-based orchestration
│   │   ├── swarm.py                # SwarmRegistry — 保留
│   │   ├── graph.py                # ★ NEW: WorkflowGraph + CompiledGraph
│   │   ├── checkpoint.py           # ★ NEW: CheckpointStore
│   │   ├── handoff.py              # ★ NEW: HandoffManager
│   │   └── guardrails.py           # ★ NEW: GuardrailsEngine
│   │
│   ├── governance.py               # PII + Audit + Quality — 保留 + 修補
│   ├── governance_v2.py            # ★ NEW: 加密審計 + PIIv2
│   ├── eu_ai_act_classifier.py     # EU AI Act — 保留
│   └── data_lineage.py             # DataLineageTracker — 保留
│
├── knowledge/                      # Layer 5: Memory & Knowledge
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py
│       ├── schema.py               # MemoryRecord, RetrievalConfig — 保留
│       ├── engine.py               # RAGEngine (v1) — 保留
│       ├── engine_v2.py            # ★ NEW: privacy mandatory + reranker + consolidator
│       ├── embeddings.py           # ★ NEW: EmbeddingManager (多後端 + fallback chain)
│       └── consolidator.py         # ★ NEW: MemoryConsolidator (背景定期整理)
│
├── mcp/                            # MCP Protocol
│   ├── __init__.py
│   ├── client.py                   # MCPClient stdio — 保留
│   ├── client_v2.py                # ★ NEW: 多 transport (stdio + HTTP)
│   ├── transport/                  # ★ NEW: 傳輸層抽象
│   │   ├── __init__.py
│   │   ├── base.py                 # MCPTransport protocol
│   │   ├── stdio.py                # StdioTransport (v1 行為)
│   │   └── http.py                 # StreamableHTTPTransport
│   └── servers/                    # 14 個 MCP 伺服器 — 全部保留
│       ├── filesystem_server.py
│       ├── git_server.py
│       ├── sqlite_server.py
│       ├── fetch_server.py
│       ├── sequential_thinking_server.py
│       ├── github_server.py
│       ├── chroma_server.py
│       ├── playwright_server.py
│       ├── slack_server.py
│       ├── postgres_server.py
│       ├── prometheus_server.py
│       ├── arxiv_monitor_server.py
│       ├── rss_aggregator_server.py
│       └── mlflow_server.py
│
├── skills/                         # Layer 4: Skills
│   ├── __init__.py
│   ├── registry.py                 # SkillRegistry — 保留
│   └── implementations/            # 10 個 skill — 全部保留
│       ├── excel_designer/
│       ├── pdf_reader/
│       ├── pptx_builder/
│       ├── shell_executor/
│       ├── notebook_executor/
│       ├── synthetic_data_generator/
│       ├── rag_evaluator/
│       ├── regulatory_tracker/
│       ├── prompt_versioning/
│       └── model_card_generator/
│
├── plugins/                        # ★ NEW: Plugin system
│   ├── __init__.py
│   ├── loader.py                   # PluginLoader (hot-reload)
│   ├── manifest.py                 # PluginManifest (YAML 解析)
│   ├── sandbox.py                  # SandboxedExecutor (subprocess + 資源限制)
│   ├── agents/                     # 第三方 Agent 插件
│   ├── skills/                     # 第三方 Skill 插件
│   ├── mcp/                        # 第三方 MCP 插件
│   └── guardrails/                 # 第三方 Guardrail 規則
│
├── api/                            # HTTP API
│   ├── __init__.py
│   ├── dashboard.py                # FastAPI dashboard — 保留
│   ├── webhook.py                  # Webhook handler — 保留
│   ├── a2a_endpoints.py            # ★ NEW: A2A protocol endpoints
│   ├── mcp_http_handler.py         # ★ NEW: NEXUS 作為 MCP HTTP 伺服器
│   └── ws_handler.py              # ★ NEW: WebSocket 即時通訊
│
├── observability/                  # ★ NEW: 可觀測性
│   ├── __init__.py
│   ├── tracing.py                  # OpenTelemetry span 管理
│   ├── metrics.py                  # Prometheus 指標定義
│   └── logging.py                  # Structured logging 設定
│
├── config/                         # 設定檔
│   ├── agents/                     # 10 Swarm YAML — 保留
│   │   ├── research_swarm.yaml
│   │   ├── engineering_swarm.yaml
│   │   ├── data_swarm.yaml
│   │   ├── data_ops_swarm.yaml
│   │   ├── governance_swarm.yaml
│   │   ├── optimization_swarm.yaml
│   │   ├── communications_swarm.yaml
│   │   ├── maintenance_swarm.yaml
│   │   ├── federation_swarm.yaml
│   │   └── ml_swarm.yaml
│   ├── prompts/
│   │   ├── system/                 # 6 系統提示 — 保留
│   │   └── domain/                 # 領域特定提示
│   ├── routing/
│   │   ├── llm_routing.yaml        # v1 路由 — 保留
│   │   └── llm_routing_v2.yaml     # ★ NEW: 能力路由
│   ├── guardrails/                 # ★ NEW
│   │   ├── default.yaml            # 預設 guardrail 規則
│   │   ├── pii.yaml                # PII 專用規則
│   │   └── code_safety.yaml        # 程式碼安全規則
│   ├── governance/
│   │   └── budgets.yaml            # 預算設定 — 保留
│   ├── mcp/
│   │   └── servers.yaml            # ★ NEW: MCP 伺服器統一設定
│   └── plugins/
│       └── registry.yaml           # ★ NEW: 已安裝插件清單
│
├── data/                           # 執行期資料（git-ignored）
│   ├── audit.db                    # SQLite 審計日誌
│   ├── checkpoints.db              # ★ NEW: SQLite 檢查點
│   ├── vector_store/               # ChromaDB 持久化
│   ├── lineage/                    # NetworkX JSON fallback
│   ├── cache/                      # ★ NEW: 語意快取
│   └── plugins/                    # ★ NEW: 插件執行期資料
│
├── tests/                          # 測試
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_router.py
│   │   ├── test_router_v2.py       # ★ NEW
│   │   ├── test_client.py
│   │   ├── test_governance.py
│   │   ├── test_eu_classifier.py
│   │   ├── test_checkpoint.py      # ★ NEW
│   │   ├── test_graph.py           # ★ NEW
│   │   ├── test_handoff.py         # ★ NEW
│   │   ├── test_guardrails.py      # ★ NEW
│   │   ├── test_cache.py           # ★ NEW
│   │   └── test_embeddings.py      # ★ NEW
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_a2a.py             # ★ NEW
│   │   ├── test_mcp_http.py        # ★ NEW
│   │   └── test_checkpoint_recovery.py  # ★ NEW
│   └── e2e/
│       ├── test_research_flow.py
│       └── test_code_flow.py
│
└── docs/
    ├── agents/README.md
    ├── skills/README.md
    ├── mcp/README.md
    ├── governance/README.md
    ├── workflows/README.md
    ├── memory/README.md
    ├── plugins/README.md           # ★ NEW
    └── a2a/README.md               # ★ NEW
```

**檔案統計：**

| 類別 | v1 檔案數 | v2 新增 | v2 總計 |
|------|----------|---------|---------|
| Core (agents) | 18 | 1 | 19 |
| Core (llm) | 2 | 3 | 5 |
| Core (orchestrator) | 4 | 5 | 9 |
| Core (governance) | 3 | 1 | 4 |
| Knowledge | 2 | 3 | 5 |
| MCP | 15 | 5 | 20 |
| Skills | 11 | 0 | 11 |
| Plugins | 0 | 5 | 5 |
| API | 2 | 3 | 5 |
| Observability | 0 | 3 | 3 |
| Config YAML | 12 | 5 | 17 |
| Tests | 5 | 10 | 15 |
| **Total** | **74** | **44** | **118** |

---

## Part 3: Layer 0 — Foundation Platform

### 3.1 LLM Router v2 (`core/llm/router_v2.py`)

**v1 問題：**
- 僅支援 privacy + complexity + domain 三維路由
- 不知道模型有哪些能力（vision, long_context, thinking）
- 無法根據 input token 數動態選模型
- Provider 故障時無自動切換

**v2 設計：**

```python
# ─── 資料結構 ───────────────────────────────────────────

@dataclass
class ModelCapability:
    """模型能力描述"""
    reasoning:      bool = True
    code:           bool = True
    vision:         bool = False
    audio:          bool = False
    tool_use:       bool = True
    structured_output: bool = False
    extended_thinking: bool = False
    long_context:   bool = False    # context_window > 200K
    grounding:      bool = False    # search grounding (Gemini)

@dataclass
class ModelConfigV2:
    """擴展的模型設定"""
    provider:         str               # "ollama" | "anthropic" | "openai" | "google"
    model_id:         str               # 實際模型 ID
    display_name:     str               # 顯示名稱
    context_window:   int               # 最大 token 數
    max_output:       int               # 最大輸出 token
    cost_per_1k_in:   float             # USD per 1K input tokens
    cost_per_1k_out:  float             # USD per 1K output tokens
    capabilities:     ModelCapability
    privacy_tiers:    list[PrivacyTier] # 允許的 privacy tier
    is_local:         bool = False
    avg_latency_ms:   int = 1000
    endpoint:         str | None = None # 自訂端點

@dataclass
class RoutingRequestV2:
    """擴展的路由請求"""
    task_type:          str
    domain:             TaskDomain
    complexity:         TaskComplexity
    privacy_tier:       PrivacyTier
    # ─ v2 新增 ─
    required_capabilities: list[str] = field(default_factory=list)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    max_cost_usd:       float | None = None
    max_latency_ms:     int | None = None
    prefer_local:       bool = False
    stream:             bool = False

@dataclass
class RoutingDecisionV2:
    """擴展的路由決策"""
    primary:            ModelConfigV2
    fallback:           ModelConfigV2 | None
    reasoning:          str
    estimated_cost_usd: float
    privacy_compliant:  bool
    # ─ v2 新增 ─
    capabilities_match: list[str]       # 實際匹配的能力
    latency_estimate_ms: int
    alternative_models: list[str]       # 其他可用模型（按優先排序）

# ─── Router 類別 ────────────────────────────────────────

class LLMRouterV2:
    """
    能力感知 + 成本感知 + 延遲感知 + 健康感知路由器。

    路由流程：
    1. filter_privacy(privacy_tier)      → 排除不合規模型
    2. filter_capabilities(required)      → 排除缺少能力的模型
    3. filter_context_window(tokens)      → 排除 context 不夠的模型
    4. filter_health(circuit_breaker)     → 排除故障中的 provider
    5. score_candidates(cost, latency, domain) → 評分
    6. select_primary_and_fallback()      → 選出 primary + fallback
    """

    def __init__(
        self,
        models: dict[str, ModelConfigV2] | None = None,
        rules_file: Path | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ): ...

    def route(self, request: RoutingRequestV2) -> RoutingDecisionV2: ...

    def _filter_privacy(
        self, models: dict, tier: PrivacyTier
    ) -> dict: ...

    def _filter_capabilities(
        self, models: dict, required: list[str]
    ) -> dict: ...

    def _filter_context_window(
        self, models: dict, tokens: int
    ) -> dict: ...

    def _score_candidates(
        self, models: dict, request: RoutingRequestV2
    ) -> list[tuple[str, float]]: ...

    def explain(self, request: RoutingRequestV2) -> str: ...

    def list_models(
        self,
        privacy_tier: PrivacyTier | None = None,
        capability: str | None = None,
    ) -> list[ModelConfigV2]: ...
```

**與 v1 的關係：** `LLMRouterV2` 是獨立類別，不繼承 `LLMRouter`。v1 `LLMRouter` 保留不動，`nexus.py` 的 `init_system()` 可選擇使用哪個版本（透過環境變數 `NEXUS_ROUTER_VERSION=v2`）。

### 3.2 Circuit Breaker (`core/llm/circuit_breaker.py`)

```python
class CircuitState(str, Enum):
    CLOSED    = "closed"      # 正常運作
    OPEN      = "open"        # 故障中，跳過
    HALF_OPEN = "half_open"   # 測試中

@dataclass
class ProviderHealth:
    provider:        str
    state:           CircuitState = CircuitState.CLOSED
    failure_count:   int = 0
    success_count:   int = 0
    last_failure_at: float | None = None
    last_success_at: float | None = None
    open_until:      float | None = None    # OPEN 狀態到期時間

class CircuitBreaker:
    """
    每個 provider 獨立追蹤健康狀態。

    設定參數：
      failure_threshold  = 3     (連續失敗 N 次 → OPEN)
      recovery_timeout   = 60.0  (OPEN 持續 N 秒 → HALF_OPEN)
      success_threshold  = 1     (HALF_OPEN 成功 N 次 → CLOSED)
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
    ): ...

    def is_available(self, provider: str) -> bool: ...
    def record_success(self, provider: str) -> None: ...
    def record_failure(self, provider: str) -> None: ...
    def get_health(self, provider: str) -> ProviderHealth: ...
    def get_all_health(self) -> dict[str, ProviderHealth]: ...
```

### 3.3 LLM Semantic Cache (`core/llm/cache.py`)

```python
@dataclass
class CacheEntry:
    query_hash:      str
    query_embedding: list[float]
    response:        LLMResponse
    model_used:      str
    created_at:      float
    ttl_seconds:     float
    hit_count:       int = 0

class LLMSemanticCache:
    """
    語意快取：相似 query 直接回傳快取結果。

    流程：
    1. 計算 query embedding
    2. 在快取中找 cosine_similarity > threshold 的 entry
    3. 若找到且未過期 → cache hit，回傳
    4. 若未找到 → cache miss，正常呼叫 LLM 後存入

    設定：
      similarity_threshold = 0.92
      default_ttl          = 86400  (24 小時)
      max_entries          = 10000
      backend              = "sqlite"  (data/cache/llm_cache.db)
    """

    def __init__(
        self,
        embedder: EmbeddingManager,
        similarity_threshold: float = 0.92,
        default_ttl: float = 86400,
        max_entries: int = 10000,
        db_path: Path = DATA_DIR / "cache" / "llm_cache.db",
    ): ...

    async def get(
        self, query: str, model: str | None = None
    ) -> LLMResponse | None: ...

    async def put(
        self, query: str, response: LLMResponse, ttl: float | None = None
    ) -> None: ...

    async def invalidate(self, query: str) -> None: ...
    async def clear(self) -> None: ...
    def stats(self) -> dict: ...   # hit_rate, total_entries, saved_cost_usd
```

### 3.4 LLM Client v2 (`core/llm/client_v2.py`)

```python
class LLMClientV2:
    """
    擴展 v1 LLMClient：
    + Structured output (JSON mode / Pydantic schema)
    + Streaming (SSE AsyncIterator)
    + Extended thinking (Claude / o-series)
    + Multi-modal input (image, audio attachments)
    + Token counting before call (pre-flight check)
    + OpenTelemetry span per call
    + Cost recording to AuditLogger
    """

    def __init__(
        self,
        governance: GovernanceManager | None = None,
        cache: LLMSemanticCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        tracer: TracingExporter | None = None,
    ): ...

    async def chat(
        self,
        messages: list[Message],
        model: ModelConfigV2,
        *,
        system: str = "",
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
        tools: list[ToolDef] | None = None,
        structured_output: type | None = None,    # Pydantic BaseModel
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,
        thinking_budget: int = 10000,
        cache_ttl: float = 0.0,                   # 0 = 不快取
        attachments: list[Attachment] | None = None,
    ) -> LLMResponse | AsyncIterator[LLMChunk]: ...

    async def count_tokens(
        self, messages: list[Message], model: ModelConfigV2
    ) -> int: ...

    # Provider adapters (internal)
    async def _ollama(self, ...) -> LLMResponse: ...
    async def _anthropic(self, ...) -> LLMResponse: ...
    async def _openai(self, ...) -> LLMResponse: ...
    async def _google(self, ...) -> LLMResponse: ...

@dataclass
class LLMChunk:
    """Streaming chunk"""
    content: str = ""
    thinking: str = ""          # Extended thinking content
    tool_call: dict | None = None
    is_final: bool = False
    usage: dict | None = None   # tokens_in, tokens_out (final chunk only)

@dataclass
class Attachment:
    """Multi-modal attachment"""
    type: str           # "image" | "audio" | "file"
    data: bytes | None = None
    url: str | None = None
    media_type: str = ""    # "image/png", "audio/wav"
```

---

## Part 4: Layer 1 — Trigger & Ingestion

### 4.1 TaskEvent v2

```python
@dataclass
class TaskEvent:
    """v2 擴展"""
    # v1 欄位（保留）
    id:           str = field(default_factory=lambda: str(uuid.uuid4()))
    message:      str = ""
    priority:     Priority = Priority.NORMAL
    session_id:   str = ""
    source:       str = "cli"           # cli | rest | ws | a2a | scheduler | ...
    metadata:     dict = field(default_factory=dict)
    created_at:   float = field(default_factory=time.time)

    # v2 新增
    attachments:  list[Attachment] = field(default_factory=list)
    parent_id:    str | None = None     # sub-task parent
    a2a_source:   str | None = None     # 來源 A2A agent URL
    thread_id:    str | None = None     # conversation thread (for checkpoint)
    reply_to:     str | None = None     # WebSocket reply target
```

### 4.2 TriggerManager v2 擴充

```
v1 sources:  CLI, REST, Scheduler, FileWatcher, GitHook, Heartbeat, MQ
v2 新增:      WebSocket, A2A inbound

新增 transport:
  WebSocket → ws_handler.py 接收 → TriggerManager.submit()
  A2A       → a2a_endpoints.py 接收 → TriggerManager.submit()
```

**不需要重寫 `trigger.py`**。只需在 `api/ws_handler.py` 和 `api/a2a_endpoints.py` 呼叫 `trigger_manager.submit(event)`。

---

## Part 5: Layer 2 — Perception & Routing

### 5.1 PerceivedTask v2

```python
@dataclass
class PerceivedTask:
    """v2 擴展（v1 所有欄位保留）"""
    # v1 欄位
    intent:              str = ""
    task_type:           str = ""
    domain:              str = "research"
    complexity:          str = "medium"
    privacy_tier:        str = "INTERNAL"
    required_agents:     list[str] = field(default_factory=list)
    required_skills:     list[str] = field(default_factory=list)
    required_mcp:        list[str] = field(default_factory=list)
    is_destructive:      bool = False
    requires_confirmation: bool = False
    has_pii:             bool = False
    language:            str = "en"
    key_entities:        list[str] = field(default_factory=list)

    # v2 新增
    required_capabilities: list[str] = field(default_factory=list)   # ["vision", "long_context"]
    estimated_tokens:      int = 0
    suggested_pattern:     str = ""               # workflow pattern hint
    handoff_eligible:      bool = False
    guardrail_rules:       list[str] = field(default_factory=list)  # 適用的 guardrail 設定
    has_attachments:       bool = False
    attachment_types:      list[str] = field(default_factory=list)  # ["image", "file"]
```

### 5.2 Perception Pipeline v2

```
TaskEvent
  │
  ├─► Stage 1: Fast-path rules (regex, 0ms)
  │     → domain, complexity, destructive, language
  │     → 不變
  │
  ├─► Stage 2: PII Scanner (regex, 0ms)
  │     → has_pii, privacy_tier escalation
  │     → 不變
  │
  ├─► Stage 3: Attachment Analysis (★ NEW, <10ms)
  │     → has_attachments, attachment_types
  │     → 若有 image → required_capabilities += ["vision"]
  │     → 若 text > 100K tokens → required_capabilities += ["long_context"]
  │
  ├─► Stage 4: LLM Analysis (local model)
  │     → intent, task_type, agents, skills, entities
  │     → 不變（但 prompt 加入 attachment context）
  │
  └─► Stage 5: Guardrail Selection (★ NEW, rule-based)
        → 根據 domain + privacy_tier 選擇適用的 guardrail rules
        → guardrail_rules: ["default", "pii"] for PRIVATE tasks
```

---

## Part 6: Layer 3 — Stateful Orchestration

這是 v2 最大的變更區塊。

### 6.1 WorkflowGraph (`core/orchestrator/graph.py`)

```python
END = "__end__"

@dataclass
class GraphNode:
    name:     str
    executor: Callable[[dict], Awaitable[dict]]   # async fn(state) -> state
    metadata: dict = field(default_factory=dict)

@dataclass
class GraphEdge:
    source:    str
    target:    str
    condition: Callable[[dict], str] | None = None  # None = unconditional

@dataclass
class GraphState:
    """Workflow 執行過程中的累積狀態"""
    task_id:        str
    thread_id:      str
    current_node:   str = ""
    step:           int = 0
    data:           dict = field(default_factory=dict)     # 累積資料
    messages:       list[dict] = field(default_factory=list)
    agent_results:  list[AgentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    quality_scores: list[float] = field(default_factory=list)
    started_at:     float = field(default_factory=time.time)
    errors:         list[str] = field(default_factory=list)

class WorkflowGraph:
    """
    定義 workflow 拓撲。
    呼叫 compile() 後產出 CompiledGraph 供執行。
    """

    def __init__(self, name: str = ""): ...

    def add_node(self, name: str, executor: Callable) -> "WorkflowGraph": ...

    def add_edge(self, source: str, target: str) -> "WorkflowGraph": ...

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[dict], str],
        targets: dict[str, str],    # condition_result → target_node
    ) -> "WorkflowGraph": ...

    def set_entry_point(self, name: str) -> "WorkflowGraph": ...

    def set_finish_point(self, name: str) -> "WorkflowGraph": ...

    def compile(
        self,
        checkpoint_store: CheckpointStore | None = None,
    ) -> "CompiledGraph": ...

    def validate(self) -> list[str]: ...    # 驗證圖的合法性

class CompiledGraph:
    """
    可執行的 workflow 圖。
    每一步自動存 checkpoint。
    """

    async def run(
        self,
        initial_state: GraphState,
        *,
        resume_from_step: int | None = None,  # 從 checkpoint 恢復
        max_steps: int = 50,
    ) -> GraphState: ...

    async def run_step(
        self, state: GraphState
    ) -> GraphState: ...           # 執行一步

    async def stream(
        self, initial_state: GraphState
    ) -> AsyncIterator[GraphState]: ...  # 每步 yield

    def visualize(self) -> str: ...      # 輸出 Mermaid diagram
```

### 6.2 Checkpoint Persistence (`core/orchestrator/checkpoint.py`)

```python
@dataclass
class Checkpoint:
    checkpoint_id:  str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id:      str = ""
    step:           int = 0
    node_name:      str = ""
    state_json:     str = ""          # GraphState 序列化
    metadata:       dict = field(default_factory=dict)
    created_at:     float = field(default_factory=time.time)
    parent_id:      str | None = None  # fork 用

class CheckpointStore:
    """
    Backend: SQLite (default) | Redis | PostgreSQL

    Schema:
      CREATE TABLE checkpoints (
        checkpoint_id  TEXT PRIMARY KEY,
        thread_id      TEXT NOT NULL,
        step           INTEGER NOT NULL,
        node_name      TEXT NOT NULL,
        state_json     TEXT NOT NULL,
        metadata_json  TEXT,
        created_at     REAL NOT NULL,
        parent_id      TEXT,
        UNIQUE(thread_id, step)
      );
      CREATE INDEX idx_thread ON checkpoints(thread_id);
    """

    def __init__(
        self,
        backend: str = "sqlite",       # "sqlite" | "redis" | "postgres"
        db_path: Path = DATA_DIR / "checkpoints.db",
        redis_url: str | None = None,
        postgres_dsn: str | None = None,
    ): ...

    async def save(self, checkpoint: Checkpoint) -> str: ...

    async def load(
        self, thread_id: str, step: int | None = None
    ) -> Checkpoint | None: ...     # step=None → latest

    async def load_latest(self, thread_id: str) -> Checkpoint | None: ...

    async def list_checkpoints(
        self, thread_id: str
    ) -> list[Checkpoint]: ...

    async def fork(
        self, thread_id: str, step: int
    ) -> str: ...                    # 回傳新 thread_id

    async def delete_thread(self, thread_id: str) -> int: ...

    async def cleanup(
        self, older_than_days: int = 30
    ) -> int: ...                    # 清理舊 checkpoint
```

### 6.3 Handoff Manager (`core/orchestrator/handoff.py`)

```python
class HandoffType(str, Enum):
    TRANSFER  = "transfer"    # 完全移交
    CONSULT   = "consult"     # 諮詢後返回
    ESCALATE  = "escalate"    # 升級到更強的 agent
    DELEGATE  = "delegate"    # 分派子任務

@dataclass
class HandoffRequest:
    from_agent:  str
    to_agent:    str
    type:        HandoffType
    reason:      str
    context:     AgentContext
    payload:     dict = field(default_factory=dict)

@dataclass
class HandoffResult:
    success:     bool
    from_agent:  str
    to_agent:    str
    type:        HandoffType
    result:      AgentResult | None = None
    error:       str | None = None

class HandoffManager:
    """
    管理 Agent 間的控制權轉移。

    防護措施：
    - max_depth = 5（防止無限遞迴）
    - 每個 handoff 記錄到 audit log
    - handoff 鏈路追蹤（tracing span）
    """

    def __init__(
        self,
        swarm_registry: SwarmRegistry,
        max_depth: int = 5,
        tracer: TracingExporter | None = None,
        audit: AuditLogger | None = None,
    ): ...

    async def execute(self, request: HandoffRequest) -> HandoffResult: ...

    async def transfer(
        self, from_agent: str, to_agent: str,
        context: AgentContext, reason: str
    ) -> HandoffResult: ...

    async def consult(
        self, requester: str, consultant: str,
        question: str, context: AgentContext
    ) -> str: ...                   # 回傳諮詢結果文字

    async def escalate(
        self, from_agent: str, context: AgentContext, reason: str
    ) -> HandoffResult: ...         # 自動選擇更高能力的 agent

    def _get_escalation_target(self, from_agent: str) -> str | None: ...
```

**Escalation 鏈（預設）：**
```
任何 agent → planner_agent → critic_agent (human review)
code_agent → ml_pipeline_agent
data_agent → ml_pipeline_agent
web_agent  → browser_agent
```

### 6.4 Guardrails Engine (`core/orchestrator/guardrails.py`)

```python
class GuardrailAction(str, Enum):
    BLOCK  = "block"    # 拒絕執行
    SCRUB  = "scrub"    # 清洗後繼續
    WARN   = "warn"     # 記錄警告，繼續
    FLAG   = "flag"     # 標記供人工審查

@dataclass
class GuardrailRule:
    name:        str
    type:        str              # "regex" | "llm" | "classifier" | "static_analysis"
    stage:       str              # "input" | "output"
    action:      GuardrailAction
    config:      dict = field(default_factory=dict)
    # config 依 type 而不同：
    #   regex:          {"patterns": [...]}
    #   llm:            {"model": "local", "prompt": "...", "threshold": 0.8}
    #   classifier:     {"blocked_categories": [...]}
    #   static_analysis: {"checks": ["no_eval", "no_exec"]}

@dataclass
class GuardrailResult:
    passed:      bool
    rule_name:   str
    action:      GuardrailAction
    details:     str = ""
    modified_content: str | None = None   # if action == SCRUB

class GuardrailsEngine:
    """
    從 YAML 載入規則，在 agent 執行前後檢查。

    設定載入優先順序：
    1. config/guardrails/default.yaml（全域）
    2. swarm YAML 中的 guardrail_rules 覆寫
    3. PerceivedTask.guardrail_rules 動態選擇
    """

    def __init__(
        self,
        rules_dir: Path = CONFIG_DIR / "guardrails",
    ): ...

    def load_rules(self) -> int: ...          # 載入所有 YAML

    async def check_input(
        self,
        context: AgentContext,
        rule_names: list[str] | None = None,  # None = 全部
    ) -> list[GuardrailResult]: ...

    async def check_output(
        self,
        result: AgentResult,
        rule_names: list[str] | None = None,
    ) -> list[GuardrailResult]: ...

    def get_rule(self, name: str) -> GuardrailRule | None: ...
    def list_rules(self) -> list[GuardrailRule]: ...
```

### 6.5 Master Orchestrator v2 (`core/orchestrator/master_v2.py`)

```python
class MasterOrchestratorV2:
    """
    v2 編排器：在 v1 基礎上新增 graph / checkpoint / handoff / guardrail。

    dispatch() 流程：
    1. EU AI Act gate（不變）
    2. Plan workflow → WorkflowGraph 或 legacy pattern
    3. Guardrails input check
    4. Execute via CompiledGraph（帶 checkpoint）
    5. Guardrails output check
    6. Quality feedback
    7. Memory write
    """

    def __init__(
        self,
        resource_pool: ResourcePool,
        swarm_registry: SwarmRegistry,
        checkpoint_store: CheckpointStore,
        handoff_manager: HandoffManager,
        guardrails: GuardrailsEngine,
        quality_optimizer: QualityOptimizer | None = None,
        tracer: TracingExporter | None = None,
    ): ...

    async def dispatch(
        self, perceived_task: dict
    ) -> OrchestratedTask: ...

    async def resume(
        self, thread_id: str
    ) -> OrchestratedTask: ...       # 從 checkpoint 恢復

    async def pause(
        self, task_id: str
    ) -> bool: ...

    async def fork(
        self, thread_id: str, step: int
    ) -> str: ...                    # fork 到新 thread

    def _build_graph(
        self, perceived: dict, swarm: Swarm
    ) -> WorkflowGraph: ...          # 根據 pattern 建圖

    def _build_sequential_graph(self, agents, swarm) -> WorkflowGraph: ...
    def _build_parallel_graph(self, agents, swarm) -> WorkflowGraph: ...
    def _build_feedback_loop_graph(self, agents, swarm) -> WorkflowGraph: ...
    def _build_map_reduce_graph(self, agents, swarm) -> WorkflowGraph: ...
    def _build_scatter_gather_graph(self, agents, swarm) -> WorkflowGraph: ...
```

### 6.6 預建圖模板

| Pattern | 圖結構 | 何時使用 |
|---------|--------|----------|
| SEQUENTIAL | `A → B → C → END` | 簡單流水線 |
| PARALLEL | `START → [A∥B∥C] → merge → END` | 多源研究 |
| PIPELINE | `A → B → C`（output chain） | ETL 轉換 |
| FEEDBACK_LOOP | `exec → critic →(pass)→ END / →(fail)→ exec` | 品質迭代 |
| ADVERSARIAL | `proposer → critic → judge → END` | 高風險決策 |
| HIERARCHICAL | `planner → [sub_graph_1 ∥ sub_graph_2] → merge` | 大型複合任務 |
| MAP_REDUCE | `split → [map_1 ∥ ... ∥ map_N] → reduce → END` | 批次處理 |
| SCATTER_GATHER | `broadcast → [agent_1 ∥ ... ∥ agent_N] → select_best` | 多 Agent 共識 |
| HUMAN_IN_LOOP | `agent → human_gate →(approve)→ continue / →(reject)→ revise` | 審批流程 |
| HANDOFF_CHAIN | `A →(handoff)→ B →(handoff)→ C → END` | 專家接力 |

---

## Part 7: Layer 4 — Execution Runtime

### 7.1 BaseAgent v2 (`core/agents/base_v2.py`)

```python
class BaseAgentV2(BaseAgent):
    """
    繼承 v1 BaseAgent，新增 v2 能力。
    現有 19 個 agent 可選擇是否升級到 V2。

    新增能力：
    - Native tool definitions (for LLM function calling)
    - Handoff support
    - Streaming output
    - Extended thinking
    - Guardrail hooks
    """

    # 新增 class attributes
    tools:               list[ToolDef] = []
    handoff_targets:     list[str] = []        # 可 handoff 的目標 agent
    guardrail_rules:     list[str] = ["default"]
    supports_streaming:  bool = False
    max_thinking_tokens: int = 0               # 0 = 不使用 thinking

    # 新增 methods
    async def handoff(
        self, to_agent: str, reason: str
    ) -> AgentResult: ...

    async def think(self, prompt: str) -> str: ...

    async def stream(
        self, context: AgentContext
    ) -> AsyncIterator[str]: ...

    async def use_native_tool(
        self, tool_name: str, **kwargs
    ) -> Any: ...
```

**與 v1 的關係：** `BaseAgentV2(BaseAgent)` — 繼承，所有現有 agent 不需修改。想用 v2 功能的 agent 改繼承 `BaseAgentV2` 即可。

### 7.2 Tool Definition

```python
@dataclass
class ToolDef:
    """LLM function calling 工具定義"""
    name:        str
    description: str
    parameters:  dict    # JSON Schema
    handler:     Callable | None = None

    def to_anthropic(self) -> dict: ...
    def to_openai(self) -> dict: ...
    def to_google(self) -> dict: ...
```

### 7.3 Skill Registry 擴充

```
v1 SkillRegistry:
  - load_all() 啟動時一次載入
  - get(name) 取得 skill

v2 擴充（加在現有 registry.py 上，不新建檔案）:
  + load_on_demand(name)    # 延遲載入（Tool Search pattern）
  + reload(name)            # 熱載入
  + search(query)           # 語意搜尋 skill
```

---

## Part 8: Layer 5 — Memory & Knowledge

### 8.1 Embedding Manager (`knowledge/rag/embeddings.py`)

```python
class EmbeddingBackend(str, Enum):
    OLLAMA_BGE     = "ollama_bge_m3"
    OPENAI         = "openai_text_embedding_3_small"
    SENTENCE_TRANS = "sentence_transformers"  # ★ 新的 fallback

class EmbeddingManager:
    """
    統一 embedding 介面 + fallback chain。

    Fallback 順序：
    1. Ollama bge-m3 (local, 1024-dim, best quality)
    2. OpenAI text-embedding-3-small (cloud, 1536-dim)
    3. sentence-transformers all-MiniLM-L6-v2 (local, 384-dim, no server)
    ❌ SHA256 hash 已移除（v1 bug）
    """

    def __init__(self): ...

    async def embed(
        self,
        text: str,
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
    ) -> list[float]: ...

    async def embed_batch(
        self,
        texts: list[str],
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
    ) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...         # 當前使用的 backend dimension

    @property
    def active_backend(self) -> str: ...    # 當前使用的 backend 名稱
```

### 8.2 RAG Engine v2 (`knowledge/rag/engine_v2.py`)

```python
class RAGEngineV2(RAGEngine):
    """
    繼承 v1 RAGEngine，修復 + 增強：

    修復：
    1. privacy_tier 為必填參數（不再有 default INTERNAL）
    2. embedding fallback 改用 sentence-transformers（移除 SHA256）

    增強：
    3. Cross-encoder reranking（改善 top-K 品質）
    4. Multi-collection 索引（episodic / semantic / conversation 分開）
    5. Contextual compression（只提取相關段落）
    """

    async def retrieve(
        self,
        query: str,
        *,
        privacy_tier: PrivacyTier,           # ★ REQUIRED（v1 有 default）
        top_k: int = 10,
        rerank: bool = True,                 # ★ NEW
        compress: bool = False,              # ★ NEW
        doc_types: list[DocumentType] | None = None,
        time_decay: bool = True,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.3,
        recency_weight: float = 0.1,        # ★ NEW: 明確分離
    ) -> list[MemoryRecord]: ...

    async def _rerank(
        self,
        query: str,
        candidates: list[MemoryRecord],
        top_k: int,
    ) -> list[MemoryRecord]: ...

    async def _compress(
        self,
        query: str,
        record: MemoryRecord,
    ) -> MemoryRecord: ...
```

### 8.3 Memory Consolidator (`knowledge/rag/consolidator.py`)

```python
class MemoryConsolidator:
    """
    背景定期任務：整理記憶。

    觸發時機：
    - 每 6 小時（定時器）
    - 累積 100 條新記憶後

    操作：
    1. 去重：cosine_similarity > 0.95 的記錄合併
    2. 摘要：同一 task 的多條 episodic 記憶合併為摘要
    3. 衰減：access_count = 0 且 > 30 天的記錄降低 decay_factor
    4. 晉升：被頻繁存取的 episodic 記憶晉升為 semantic
    5. 教訓：從失敗模式中提取 "lessons learned"
    """

    def __init__(
        self,
        rag_engine: RAGEngineV2,
        embedder: EmbeddingManager,
        llm_client: LLMClientV2 | None = None,
    ): ...

    async def run(self) -> ConsolidationReport: ...
    async def deduplicate(self) -> int: ...
    async def summarize_episodes(self) -> int: ...
    async def decay_stale(self) -> int: ...
    async def promote_frequent(self) -> int: ...
    async def extract_lessons(self) -> int: ...

@dataclass
class ConsolidationReport:
    deduplicated: int
    summarized:   int
    decayed:      int
    promoted:     int
    lessons:      int
    duration_ms:  int
```

---

## Part 9: Layer 6 — Federation (A2A)

### 9.1 A2A Endpoints (`api/a2a_endpoints.py`)

```python
# FastAPI router

# Discovery
GET  /.well-known/agent.json       → AgentCard JSON

# Task lifecycle (JSON-RPC style)
POST /a2a/tasks/send               → 提交任務
POST /a2a/tasks/get                → 查詢任務狀態
POST /a2a/tasks/cancel             → 取消任務
POST /a2a/tasks/sendSubscribe      → ★ NEW: streaming (SSE)

# Admin
GET  /a2a/known-agents             → 列出已知遠端 agent
POST /a2a/discover                 → 發現新的遠端 agent
```

### 9.2 A2A Agent 擴充

v1 `A2AAgent` 已支援 delegate / discover / list / broadcast。v2 新增：

```python
# 新增 operation:
#   subscribe  — 使用 SSE streaming 監聽遠端任務進度
#   register   — 向 discovery registry 註冊自己

# a2a_agent.py 修改：
async def _subscribe(self, context: AgentContext) -> AgentResult:
    """SSE streaming delegation"""
    ...

async def _register(self, context: AgentContext) -> AgentResult:
    """Register self with discovery service"""
    ...
```

---

## Part 10: Cross-Cutting — Governance

### 10.1 PII Scrubber v2

```
v1: 6 個 regex 模式
v2 新增：
  - 台灣居留證號碼 (2 letters + 8 digits)
  - 台灣護照號碼
  - 日本 My Number
  - 韓國居民登錄號碼
  - 醫療紀錄號碼
  - 人名偵測（spaCy NER，可選）
  - 可逆 pseudonymization 模式：
    PIIScrubber.tokenize(text) → (scrubbed_text, token_map)
    PIIScrubber.detokenize(scrubbed_text, token_map) → original
```

### 10.2 Audit Logger v2

```
v1: 明文 SQLite
v2 新增：
  - Fernet 加密 payload
  - JSON 結構化記錄
  - Hash chain（每條記錄包含前一條的 hash，防竄改）
  - Log rotation（每月一個 .db 檔）
  - 匯出 API：to_json(), to_csv()
  - Query API：filter by agent/model/date/cost
```

### 10.3 EU AI Act Classifier v2

```
v1: 6 prohibited + 8 high-risk + 3 limited
v2 新增：
  - 12 prohibited patterns（+6）
  - 18 high-risk rules（+10）
  - Article 50 GPAI 透明義務
  - Risk scoring: 0.0–1.0（不再只是二元分類）
  - 分類解釋（reasoning 欄位）
  - 所有分類記錄到 audit log
```

---

## Part 11: Cross-Cutting — Observability

### 11.1 Tracing (`observability/tracing.py`)

```python
class TracingExporter:
    """
    OpenTelemetry 分散式追蹤。

    Backends:
    - Console (development)
    - Jaeger (local: http://localhost:16686)
    - OTLP (production: Datadog / New Relic / Grafana Tempo)

    Span 層級：
    - task (root span)
      - perception
        - fast_path
        - pii_scan
        - llm_analysis → llm_call
      - orchestration
        - eu_ai_check
        - guardrail_input
        - workflow_execution
          - step_1 (agent_name) → llm_call, mcp_call, skill_call
          - step_2 ...
          - handoff (if any)
        - guardrail_output
      - memory_write
    """

    def __init__(
        self,
        service_name: str = "nexus",
        backend: str = "console",    # "console" | "jaeger" | "otlp"
        endpoint: str | None = None,
    ): ...

    def start_span(
        self, name: str, parent: Span | None = None, attributes: dict | None = None
    ) -> Span: ...

    def end_span(self, span: Span) -> None: ...

    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[Span]: ...

    def inject_context(self, headers: dict) -> dict: ...    # for A2A
    def extract_context(self, headers: dict) -> Context: ...
```

### 11.2 Metrics (`observability/metrics.py`)

```python
# Prometheus 指標定義

# Task 級
nexus_tasks_total             {status, domain, pattern}           Counter
nexus_task_duration_seconds   {domain, pattern}                   Histogram
nexus_task_quality_score      {domain, agent_id}                  Histogram

# LLM 級
nexus_llm_calls_total         {provider, model, privacy_tier}     Counter
nexus_llm_cost_usd            {provider, model}                   Counter
nexus_llm_latency_seconds     {provider, model}                   Histogram
nexus_llm_tokens_total        {provider, model, direction}        Counter
nexus_llm_cache_hits_total    {}                                  Counter
nexus_llm_cache_misses_total  {}                                  Counter

# Agent 級
nexus_agent_executions_total  {agent_id, success}                 Counter
nexus_agent_duration_seconds  {agent_id}                          Histogram
nexus_agent_quality           {agent_id}                          Gauge

# MCP 級
nexus_mcp_calls_total         {server, tool, status}              Counter
nexus_mcp_duration_seconds    {server, tool}                      Histogram

# Orchestration 級
nexus_checkpoint_ops_total    {operation}                         Counter
nexus_handoff_total           {from_agent, to_agent, type}        Counter
nexus_guardrail_triggers      {rule, action}                      Counter

# A2A 級
nexus_a2a_tasks_total         {remote_agent, direction, status}   Counter

# Memory 級
nexus_memory_ops_total        {type, operation}                   Counter
nexus_memory_consolidation    {operation}                         Counter

# System 級
nexus_circuit_breaker_state   {provider}                          Gauge
nexus_active_agents           {}                                  Gauge
nexus_active_llm_calls        {}                                  Gauge
```

### 11.3 Structured Logging (`observability/logging.py`)

```python
# 使用 structlog，每條 log 帶有：
# - task_id
# - agent_id
# - trace_id (OpenTelemetry correlation)
# - privacy_tier
# - cost_usd
# - 不包含 PII（log 前 scrub）

def setup_logging(
    level: str = "INFO",
    format: str = "json",       # "json" | "console"
    log_file: Path | None = None,
) -> None: ...
```

---

## Part 12: Cross-Cutting — Security

### 12.1 安全分層

```
Layer 1: Network
  ├─ TLS 1.3 (所有外部通訊)
  ├─ mTLS (A2A inter-instance)
  └─ Rate limiting (per client IP, FastAPI middleware)

Layer 2: Authentication
  ├─ Bearer token (API access)
  ├─ OAuth 2.1 (MCP HTTP transport)
  └─ API key rotation (SecretVault)

Layer 3: Authorization
  ├─ RBAC: admin / operator / viewer
  ├─ Privacy tier enforcement (LLMRouter)
  └─ Per-agent permission scope

Layer 4: Data Protection
  ├─ PII scrubbing (每個 LLM call 前)
  ├─ Audit log 加密 (Fernet)
  ├─ Checkpoint state 不含 PII
  └─ Memory encryption at rest (optional)

Layer 5: Runtime
  ├─ Skill sandbox (subprocess + resource limits)
  ├─ Shell command allowlist
  ├─ Max token / cost / timeout limits
  └─ Guardrails on all I/O
```

### 12.2 Prompt Injection Defense

```python
class PromptInjectionDetector:
    """
    多層防護：
    1. Static rules: 偵測常見注入模式
       - "ignore previous instructions"
       - "you are now"
       - "system:" 在 user message 中
    2. LLM classifier: local model 檢查是否有覆寫意圖
    3. Output verification: 確保回應符合預期格式
    4. Canary tokens: 在 system prompt 中嵌入隱藏標記

    實作為 GuardrailRule，設定在 config/guardrails/default.yaml
    """
```

---

## Part 13: API & Dashboard

### 13.1 API Endpoints 完整清單

```
# ── 系統 ──────────────────────────
GET   /health                       → 健康檢查
GET   /health/detailed              → 所有組件狀態
GET   /status                       → 系統儀表板 data

# ── Task ──────────────────────────
POST  /api/task                     → 提交任務
GET   /api/task/{task_id}           → 查詢任務狀態
POST  /api/task/{task_id}/pause     → 暫停任務
POST  /api/task/{task_id}/resume    → 恢復任務
POST  /api/task/{task_id}/cancel    → 取消任務
GET   /api/task/{task_id}/checkpoints → 列出 checkpoints
POST  /api/task/{task_id}/fork/{step} → fork execution

# ── Streaming ─────────────────────
WS    /ws                           → ★ NEW: WebSocket 即時通訊
GET   /api/events                   → SSE 事件流（v1 保留）

# ── A2A Protocol ──────────────────
GET   /.well-known/agent.json       → ★ NEW: AgentCard
POST  /a2a/tasks/send               → ★ NEW: 接收任務
POST  /a2a/tasks/get                → ★ NEW: 查詢任務
POST  /a2a/tasks/cancel             → ★ NEW: 取消任務
POST  /a2a/tasks/sendSubscribe      → ★ NEW: SSE streaming
GET   /a2a/known-agents             → ★ NEW: 列出遠端 agent
POST  /a2a/discover                 → ★ NEW: 發現遠端 agent

# ── Memory ────────────────────────
POST  /api/memory/ingest            → 寫入知識庫
GET   /api/memory/search            → 搜尋知識庫
POST  /api/memory/consolidate       → ★ NEW: 手動觸發整理

# ── LLM ───────────────────────────
GET   /api/llm/models               → 列出所有模型
GET   /api/llm/health               → ★ NEW: Provider 健康狀態
GET   /api/llm/cache/stats          → ★ NEW: 快取命中率

# ── Governance ────────────────────
GET   /api/audit/query              → 查詢審計日誌
GET   /api/audit/costs              → 費用報表
GET   /api/audit/export             → 匯出審計記錄

# ── Plugins ───────────────────────
GET   /api/plugins                  → ★ NEW: 列出插件
POST  /api/plugins/{name}/reload    → ★ NEW: 熱載入

# ── MCP ───────────────────────────
GET   /api/mcp/servers              → 列出 MCP 伺服器
GET   /api/mcp/tools                → 列出所有 tools
POST  /mcp                          → ★ NEW: NEXUS 作為 MCP HTTP 伺服器
```

### 13.2 WebSocket 協議 (`api/ws_handler.py`)

```python
# Client → Server
{
    "type": "task",
    "message": "分析最新的 AI 趨勢",
    "session_id": "sess_123",
    "attachments": []
}

# Server → Client (streaming)
{"type": "status",  "task_id": "t_123", "status": "perception"}
{"type": "status",  "task_id": "t_123", "status": "orchestrating"}
{"type": "stream",  "task_id": "t_123", "agent": "web_agent", "chunk": "搜尋中..."}
{"type": "stream",  "task_id": "t_123", "agent": "writer_agent", "chunk": "## 報告\n"}
{"type": "result",  "task_id": "t_123", "output": {...}, "quality": 0.85}
{"type": "error",   "task_id": "t_123", "error": "..."}
```

---

## Part 14: Configuration Schema

### 14.1 環境變數 (`.env`)

```bash
# ── Provider API Keys ─────────────
ANTHROPIC_API_KEY=               # Anthropic Claude
OPENAI_API_KEY=                  # OpenAI
GOOGLE_API_KEY=                  # Google Gemini
OLLAMA_BASE_URL=http://localhost:11434

# ── Database ──────────────────────
REDIS_URL=redis://localhost:6379
CHROMA_HOST=localhost
CHROMA_PORT=8000
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# ── NEXUS Config ──────────────────
NEXUS_BASE_URL=http://localhost:8080
NEXUS_ROUTER_VERSION=v2           # "v1" | "v2"
NEXUS_LOG_LEVEL=INFO
NEXUS_LOG_FORMAT=json             # "json" | "console"
NEXUS_TRACING_BACKEND=console     # "console" | "jaeger" | "otlp"
NEXUS_TRACING_ENDPOINT=           # Jaeger/OTLP endpoint

# ── Governance ────────────────────
NEXUS_BUDGET_DAILY_USD=50.0
NEXUS_BUDGET_MONTHLY_USD=1000.0
NEXUS_AUDIT_ENCRYPTION_KEY=       # Fernet key (auto-generated if empty)

# ── A2A Federation ────────────────
A2A_AGENT_URLS=                   # comma-separated
A2A_API_KEY=

# ── MCP ───────────────────────────
GITHUB_TOKEN=
SLACK_BOT_TOKEN=
```

### 14.2 MCP 伺服器設定 (`config/mcp/servers.yaml`)

```yaml
servers:
  # ── Local (stdio transport) ─────────
  filesystem:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.filesystem_server"]
    auto_start: true

  git:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.git_server"]
    auto_start: true

  sqlite:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.sqlite_server"]
    auto_start: true
    env:
      SQLITE_DB_PATH: "data/audit.db"

  fetch:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.fetch_server"]
    auto_start: true

  sequential_thinking:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.sequential_thinking_server"]
    auto_start: true

  playwright:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.playwright_server"]
    auto_start: false    # 按需啟動

  # ── Cloud (stdio, needs tokens) ────
  github:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.github_server"]
    auto_start: false
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"

  slack:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.slack_server"]
    auto_start: false
    env:
      SLACK_BOT_TOKEN: "${SLACK_BOT_TOKEN}"

  # ── Database ───────────────────────
  chroma:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.chroma_server"]
    auto_start: true
    env:
      CHROMA_HOST: "${CHROMA_HOST}"

  postgres:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.postgres_server"]
    auto_start: false

  # ── Monitoring ─────────────────────
  prometheus:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.prometheus_server"]
    auto_start: false

  arxiv_monitor:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.arxiv_monitor_server"]
    auto_start: false

  rss_aggregator:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.rss_aggregator_server"]
    auto_start: false

  mlflow:
    transport: stdio
    command: ["python", "-m", "nexus.mcp.servers.mlflow_server"]
    auto_start: false

  # ── Remote HTTP (v2 新增) ──────────
  # 以下為範例，實際需要時再啟用
  # remote_search:
  #   transport: http
  #   url: "https://mcp.example.com/search"
  #   auth:
  #     type: oauth2
  #     client_id_env: SEARCH_CLIENT_ID
  #     client_secret_env: SEARCH_CLIENT_SECRET
  #     token_url: "https://auth.example.com/token"
  #   health_check: "/health"
  #   connection_pool_size: 5
```

### 14.3 Guardrail 設定 (`config/guardrails/default.yaml`)

```yaml
input_guardrails:
  - name: pii_blocker
    type: regex
    action: scrub
    config:
      patterns:
        - "\\b[A-Z]\\d{9}\\b"                              # Taiwan ID
        - "\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b"  # Credit card

  - name: prompt_injection_detector
    type: regex
    action: warn
    config:
      patterns:
        - "(?i)ignore\\s+(previous|above|all)\\s+instructions"
        - "(?i)you\\s+are\\s+now\\s+"
        - "(?i)\\bsystem:\\s*"

  - name: destructive_command_check
    type: regex
    action: block
    config:
      patterns:
        - "rm\\s+-rf\\s+/"
        - "DROP\\s+DATABASE"
        - ":(){ :|:& };:"    # fork bomb

output_guardrails:
  - name: pii_leak_check
    type: regex
    action: scrub
    config:
      patterns:
        - "\\b[A-Z]\\d{9}\\b"

  - name: code_safety
    type: static_analysis
    action: warn
    config:
      checks:
        - no_eval
        - no_exec
        - no_os_system
        - no_subprocess_shell_true
```

---

## Part 15: Database Schema

### 15.1 Audit DB (`data/audit.db`)

```sql
-- v1 schema（保留）
CREATE TABLE IF NOT EXISTS audit_log (
    record_id     TEXT PRIMARY KEY,
    timestamp     REAL NOT NULL,
    event_type    TEXT NOT NULL,
    task_id       TEXT,
    agent_id      TEXT,
    action        TEXT,
    model_used    TEXT,
    privacy_tier  TEXT,
    cost_usd      REAL DEFAULT 0.0,
    tokens_used   INTEGER DEFAULT 0,
    success       INTEGER DEFAULT 1,
    quality_score REAL DEFAULT 0.0,
    error         TEXT,
    payload_hash  TEXT
);

-- v2 新增
ALTER TABLE audit_log ADD COLUMN encrypted_payload TEXT;    -- Fernet 加密
ALTER TABLE audit_log ADD COLUMN prev_hash         TEXT;    -- hash chain
ALTER TABLE audit_log ADD COLUMN trace_id          TEXT;    -- OpenTelemetry

CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log(task_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_date ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_log(trace_id);
```

### 15.2 Checkpoint DB (`data/checkpoints.db`)

```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL,
    step           INTEGER NOT NULL,
    node_name      TEXT NOT NULL,
    state_json     TEXT NOT NULL,          -- GraphState 序列化
    metadata_json  TEXT,
    created_at     REAL NOT NULL,
    parent_id      TEXT,                   -- fork 用
    UNIQUE(thread_id, step)
);

CREATE INDEX IF NOT EXISTS idx_ckpt_thread ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_ckpt_parent ON checkpoints(parent_id);

-- 清理任務
-- DELETE FROM checkpoints WHERE created_at < (now - 30 days)
```

### 15.3 Cache DB (`data/cache/llm_cache.db`)

```sql
CREATE TABLE IF NOT EXISTS llm_cache (
    query_hash       TEXT PRIMARY KEY,
    query_text       TEXT NOT NULL,
    query_embedding  BLOB NOT NULL,        -- numpy array bytes
    response_json    TEXT NOT NULL,         -- LLMResponse 序列化
    model_used       TEXT NOT NULL,
    created_at       REAL NOT NULL,
    ttl_seconds      REAL NOT NULL,
    hit_count        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cache_model ON llm_cache(model_used);
CREATE INDEX IF NOT EXISTS idx_cache_created ON llm_cache(created_at);
```

---

## Part 16: 狀態機與生命週期

### 16.1 Task 生命週期

```
                              ┌─────────┐
                              │ PENDING  │
                              └────┬─────┘
                                   │ dispatch()
                              ┌────▼─────┐
                  ┌───────────│ RUNNING  │───────────┐
                  │           └────┬─────┘           │
                  │ pause()        │                  │ error
             ┌────▼─────┐         │             ┌────▼─────┐
             │ PAUSED   │         │             │ FAILED   │
             └────┬─────┘         │             └──────────┘
                  │ resume()      │ complete
                  │               │
                  └───────►┌──────▼─────┐
                           │ COMPLETED  │
                           └────────────┘

                           ┌────────────┐
        cancel() from any → │ CANCELLED  │
                           └────────────┘
```

### 16.2 Circuit Breaker 狀態機

```
              success                failure
                │                      │
     ┌──────────▼──────────┐           │
     │      CLOSED         │───────────┘
     │  (failure_count=0)  │   failure_count++
     └──────────┬──────────┘
                │ failure_count >= threshold
     ┌──────────▼──────────┐
     │       OPEN          │
     │  (skip all calls)   │
     └──────────┬──────────┘
                │ recovery_timeout expired
     ┌──────────▼──────────┐
     │     HALF_OPEN       │──── failure ──→ OPEN
     │  (try one call)     │
     └──────────┬──────────┘
                │ success
     ┌──────────▼──────────┐
     │      CLOSED         │
     └─────────────────────┘
```

### 16.3 Workflow Graph 執行

```
                    ┌────────────────┐
                    │  Initial State │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  entry_point   │
                    └───────┬────────┘
                            │
                ┌───────────▼───────────┐
                │   Save Checkpoint     │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Execute Node        │
                │   (agent.run())       │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Evaluate Edge       │──── conditional? ──→ condition(state)
                │   (next node)         │                        │
                └───────────┬───────────┘                   target node
                            │
                       next == END?
                     ┌──────┴──────┐
                     no           yes
                     │             │
              ┌──────▼──────┐ ┌───▼────┐
              │ loop back   │ │  END   │
              │ to "Save    │ │ return │
              │ Checkpoint" │ │ state  │
              └─────────────┘ └────────┘
```

### 16.4 Handoff 流程

```
Agent A executing
  │
  │ discovers: "I need ML expertise"
  │
  ├─► handoff_manager.transfer(
  │     from="code_agent",
  │     to="ml_pipeline_agent",
  │     reason="ML model training required"
  │   )
  │
  │   ┌──────────────────────────────┐
  │   │ HandoffManager:              │
  │   │ 1. Check depth < max_depth   │
  │   │ 2. Log to audit              │
  │   │ 3. Create tracing span       │
  │   │ 4. Transfer context          │
  │   │ 5. Execute target agent      │
  │   │ 6. Return result             │
  │   └──────────────────────────────┘
  │
  ◄── HandoffResult (contains AgentResult from ml_pipeline_agent)
  │
  │ continues with ML result...
```

---

## Part 17: 資料流

### 17.1 End-to-End: 「分析台灣半導體產業趨勢」

```
1. CLI Input
   └─► TriggerManager.submit(TaskEvent{message="分析台灣半導體產業趨勢"})

2. PerceptionEngine.analyze()
   ├─► Fast-path: domain=research, complexity=medium
   ├─► PII scan: 無 PII → privacy=INTERNAL
   ├─► Attachment: 無
   ├─► LLM (qwen2.5, local):
   │     intent="產業趨勢分析"
   │     required_agents=["web_agent", "rag_agent", "writer_agent"]
   │     required_capabilities=[]
   │     guardrail_rules=["default"]
   └─► PerceivedTask{domain=research, complexity=medium, privacy=INTERNAL}

3. MasterOrchestratorV2.dispatch()
   ├─► EU AI Act: classify("semiconductor analysis") → MINIMAL_RISK ✅
   ├─► GuardrailsEngine.check_input() → all passed ✅
   ├─► _plan_workflow() → research_swarm, FEEDBACK_LOOP
   ├─► _build_feedback_loop_graph():
   │
   │     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │     │web_agent │ →  │rag_agent │ →  │writer    │ →  │critic    │
   │     │(search)  │    │(retrieve)│    │(draft)   │    │(review)  │
   │     └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
   │                                          ▲              │
   │                                          │   quality<0.75│
   │                                          └───────────────┘
   │                                          quality≥0.75 → END
   │
   ├─► CompiledGraph.run():
   │     Step 1: web_agent
   │       ├─► LLMRouter: INTERNAL + MEDIUM → claude-sonnet-4-6
   │       ├─► cache check → miss
   │       ├─► llm.chat() → 搜尋結果
   │       ├─► checkpoint.save(step=1)
   │       └─► tracing: span{agent=web_agent, cost=$0.003, latency=2100ms}
   │
   │     Step 2: rag_agent
   │       ├─► RAGEngine.retrieve(privacy_tier=INTERNAL)
   │       ├─► 找到 3 條相關知識
   │       ├─► checkpoint.save(step=2)
   │       └─► tracing: span{agent=rag_agent, cost=$0.001}
   │
   │     Step 3: writer_agent
   │       ├─► LLMRouter → claude-sonnet-4-6
   │       ├─► llm.chat(system=writer_prompt, context=search+rag)
   │       ├─► checkpoint.save(step=3)
   │       └─► tracing: span{agent=writer_agent, cost=$0.005}
   │
   │     Step 4: critic_agent
   │       ├─► quality_score = 0.82 ≥ 0.75 → END
   │       ├─► checkpoint.save(step=4)
   │       └─► tracing: span{agent=critic_agent, cost=$0.002}
   │
   ├─► GuardrailsEngine.check_output() → all passed ✅
   └─► total_cost = $0.011, quality = 0.82

4. Memory Write
   ├─► RAGEngine.upsert(episodic_record)
   ├─► DataLineageTracker.record_task(inputs=[web_results], output=report)
   └─► AuditLogger.log(4 LLM calls, total $0.011)

5. Return to CLI
   └─► 顯示完整報告 + quality + cost
```

### 17.2 資料流: Handoff 場景

```
Task: "用 scikit-learn 訓練一個分類模型"

1. Perception → domain=engineering, agents=[code_agent]

2. code_agent starts
   ├─► 讀取需求
   ├─► 發現需要 ML pipeline（hyperparameter tuning, cross-validation）
   ├─► handoff_manager.transfer(
   │     from="code_agent",
   │     to="ml_pipeline_agent",
   │     reason="ML training requires specialized pipeline"
   │   )
   │
   └─► ml_pipeline_agent executes
       ├─► 建立 training pipeline
       ├─► 執行 cross-validation
       └─► 回傳 HandoffResult{model_accuracy=0.91, code="..."}

3. code_agent resumes
   ├─► 整合 ML 結果
   └─► 產出完整程式碼 + 報告
```

### 17.3 資料流: Crash Recovery

```
1. 任務執行到 Step 3 時系統 crash
   ├─► Steps 1, 2, 3 的 checkpoint 已存
   └─► Step 4 未執行

2. 系統重啟
   └─► nexus.py resume <task_id>

3. MasterOrchestratorV2.resume(thread_id)
   ├─► checkpoint_store.load_latest(thread_id)
   │     → Checkpoint{step=3, state={...}}
   ├─► CompiledGraph.run(resume_from_step=3)
   │     → 從 Step 4 繼續
   └─► 完成任務
```

---

## Part 18: 錯誤處理策略

### 18.1 錯誤分類

| 錯誤類型 | 可恢復 | 處理方式 |
|---------|--------|----------|
| LLM API timeout | ✅ | Retry with exponential backoff (1s, 2s, 4s) |
| LLM API rate limit | ✅ | Wait for Retry-After header, then retry |
| LLM API 500 | ✅ | Retry once → fallback model |
| LLM provider down | ✅ | Circuit breaker → fallback provider |
| Agent logic error | ❌ | Log error → continue workflow (skip agent) |
| PII detected in output | ✅ | Scrub and continue |
| Budget exceeded | ✅ | Auto-downgrade model chain |
| MCP server crash | ✅ | Restart subprocess → retry once |
| Checkpoint write fail | ⚠️ | Warn → continue without checkpoint |
| Guardrail blocked | ❌ | Return error with explanation |
| EU AI Act blocked | ❌ | Return error with legal reference |
| Handoff loop detected | ❌ | Break loop → return partial result |

### 18.2 錯誤傳播規則

```
Agent error → Harness catches
  │
  ├─ retryable? → retry (max 2 attempts)
  │
  └─ not retryable? →
       ├─ critical agent? → workflow FAILED
       └─ non-critical?   → skip, continue with remaining agents
```

---

## Part 19: 測試策略

### 19.1 測試金字塔

```
        ╱╲
       ╱  ╲       E2E Tests (2-3)
      ╱    ╲      - Full pipeline flow
     ╱──────╲     - A2A federation
    ╱        ╲
   ╱  Integ   ╲   Integration Tests (8-10)
  ╱   Tests    ╲  - Checkpoint recovery
 ╱──────────────╲ - MCP HTTP transport
╱                ╲ - LLM routing chains
╱   Unit Tests    ╲ Unit Tests (20+)
╱  (fast, isolated) ╲ - Router, Cache, CircuitBreaker
╱────────────────────╲ - Graph, Checkpoint, Handoff
                        - Guardrails, Embeddings
```

### 19.2 測試策略

| 層級 | 框架 | Mock 策略 | 數量 |
|------|------|-----------|------|
| Unit | pytest + pytest-asyncio | Mock LLM, Mock MCP | 20+ |
| Integration | pytest | 真實 SQLite, Mock LLM | 8-10 |
| E2E | pytest | 真實 Ollama (local) | 2-3 |

### 19.3 關鍵測試案例

```
Unit:
  - test_router_capability_matching
  - test_router_privacy_filter
  - test_circuit_breaker_state_transitions
  - test_semantic_cache_hit_miss
  - test_graph_conditional_edges
  - test_checkpoint_save_load
  - test_checkpoint_fork
  - test_handoff_max_depth
  - test_guardrail_regex_scrub
  - test_guardrail_block
  - test_embedding_fallback_chain
  - test_pii_scrub_v2_patterns
  - test_audit_hash_chain

Integration:
  - test_checkpoint_crash_recovery
  - test_feedback_loop_graph_execution
  - test_mcp_http_transport
  - test_handoff_code_to_ml
  - test_guardrail_full_pipeline
  - test_cost_auto_downgrade

E2E:
  - test_research_task_full_pipeline
  - test_a2a_delegate_and_receive
```

---

## Part 20: 效能需求與限制

| 指標 | 目標值 | 測量方式 |
|------|--------|----------|
| Perception 延遲 | < 500ms (local LLM) | P95 latency |
| Checkpoint save | < 50ms | P95 latency |
| Checkpoint load | < 30ms | P95 latency |
| LLM cache lookup | < 10ms | P95 latency |
| Guardrail check | < 100ms (regex) | P95 latency |
| MCP tool call | < 5s | P95 latency |
| Memory retrieval | < 200ms | P95 latency |
| A2A task delegation | < 30s | P95 end-to-end |
| Concurrent agents | 10 | 資源池上限 |
| Concurrent LLM calls | 5 | 資源池上限 |
| Max task cost | $1.00 | 可設定 |
| Max task timeout | 600s | 可設定 |
| Checkpoint retention | 30 days | 自動清理 |
| LLM cache hit rate | > 25% | Prometheus gauge |

---

## Part 21: 遷移計畫

### Phase 1: Foundation (第 1-2 週)

**目標：** 建立 v2 基礎，不影響 v1 功能。

| 任務 | 檔案 | 依賴 |
|------|------|------|
| CheckpointStore (SQLite) | `core/orchestrator/checkpoint.py` | 無 |
| WorkflowGraph + CompiledGraph | `core/orchestrator/graph.py` | CheckpointStore |
| EmbeddingManager | `knowledge/rag/embeddings.py` | 無 |
| 移除 SHA256 fallback | `knowledge/rag/engine.py` 修改 | EmbeddingManager |
| privacy_tier 必填 | `knowledge/rag/engine.py` 修改 | 無 |
| Audit 加密 (Fernet) | `core/governance.py` 修改 | 無 |
| Unit tests | `tests/unit/test_checkpoint.py`, `test_graph.py`, `test_embeddings.py` | 上述全部 |

### Phase 2: Routing & Transport (第 3-4 週)

| 任務 | 檔案 | 依賴 |
|------|------|------|
| CircuitBreaker | `core/llm/circuit_breaker.py` | 無 |
| LLMRouterV2 | `core/llm/router_v2.py` | CircuitBreaker |
| LLMSemanticCache | `core/llm/cache.py` | EmbeddingManager |
| LLMClientV2 | `core/llm/client_v2.py` | RouterV2, Cache, CircuitBreaker |
| MCPTransport base | `mcp/transport/base.py` | 無 |
| StdioTransport | `mcp/transport/stdio.py` | MCPTransport |
| StreamableHTTPTransport | `mcp/transport/http.py` | MCPTransport |
| MCPClientV2 | `mcp/client_v2.py` | Transport layer |
| MCP servers.yaml | `config/mcp/servers.yaml` | 無 |
| Unit tests | 各對應測試 | 上述全部 |

### Phase 3: Orchestration (第 5-6 週)

| 任務 | 檔案 | 依賴 |
|------|------|------|
| HandoffManager | `core/orchestrator/handoff.py` | SwarmRegistry |
| GuardrailsEngine | `core/orchestrator/guardrails.py` | 無 |
| Guardrail YAML | `config/guardrails/*.yaml` | 無 |
| MasterOrchestratorV2 | `core/orchestrator/master_v2.py` | Graph, Checkpoint, Handoff, Guardrails |
| BaseAgentV2 | `core/agents/base_v2.py` | HandoffManager |
| PerceivedTask v2 | `core/orchestrator/perception.py` 修改 | Guardrails |
| Integration tests | `tests/integration/test_checkpoint_recovery.py` | Phase 1-3 全部 |

### Phase 4: Observability & A2A (第 7-8 週)

| 任務 | 檔案 | 依賴 |
|------|------|------|
| TracingExporter | `observability/tracing.py` | 無 |
| Metrics 定義 | `observability/metrics.py` | 無 |
| Structured logging | `observability/logging.py` | 無 |
| A2A endpoints | `api/a2a_endpoints.py` | MasterOrchestratorV2 |
| WebSocket handler | `api/ws_handler.py` | MasterOrchestratorV2 |
| A2A streaming | `core/agents/a2a_agent.py` 修改 | A2A endpoints |
| CostIntelligenceEngine | `core/agents/cost_optimizer_agent.py` 修改 | Metrics |
| Grafana dashboards | `config/grafana/` | Metrics |

### Phase 5: Plugins & Polish (第 9-10 週)

| 任務 | 檔案 | 依賴 |
|------|------|------|
| PluginManifest | `plugins/manifest.py` | 無 |
| PluginLoader | `plugins/loader.py` | PluginManifest |
| SandboxedExecutor | `plugins/sandbox.py` | 無 |
| MemoryConsolidator | `knowledge/rag/consolidator.py` | RAGEngineV2 |
| PromptInjectionDetector | guardrail rule | GuardrailsEngine |
| PII Scrubber v2 | `core/governance.py` 修改 | 無 |
| E2E tests | `tests/e2e/` | Phase 1-5 全部 |
| nexus.py init_system v2 | `nexus.py` 修改 | 全部 |

### 向後相容性保證

```
✅ 所有 v1 agent 繼續運作（BaseAgent 不變）
✅ 所有 v1 swarm YAML 繼續有效
✅ 所有 v1 MCP servers 繼續使用 stdio
✅ v1 .env 設定不需修改
✅ v1 audit.db 資料保留（schema 用 ALTER TABLE）
✅ v1 LLMRouter 可透過 NEXUS_ROUTER_VERSION=v1 使用
✅ v1 nexus.py CLI 指令不變
```

---

## Part 22: 技術選型理由

| 選擇 | 替代方案 | 選擇理由 |
|------|----------|----------|
| **SQLite** for checkpoints | Redis, PostgreSQL | 零依賴，開發友好，可擴展到 Redis/PG |
| **WorkflowGraph** (自建) | LangGraph library | 避免額外依賴，掌控核心邏輯，API 更簡潔 |
| **YAML guardrails** | Python code | 非工程師可編輯，可審計，熱載入 |
| **Fernet** for audit encryption | AES-GCM, ChaCha20 | Python 標準庫 (`cryptography`)，足夠安全 |
| **sentence-transformers** fallback | TF-IDF, Word2Vec | 品質遠高於 bag-of-words，無需 GPU |
| **Streamable HTTP** for MCP | gRPC, WebSocket | MCP spec 標準，業界共識 |
| **OpenTelemetry** for tracing | Jaeger SDK, Zipkin | 廠商中立，支援所有主流 backend |
| **structlog** for logging | loguru, standard logging | JSON output，自動 context binding |
| **Pydantic** for structured output | manual JSON parsing | Type safety，自動驗證 |
| **watchdog** for hot-reload | inotify, polling | 跨平台，已在 requirements.txt |

---

## 附錄 A: 模組依賴圖

```
nexus.py (entry point)
  │
  ├─► Layer 0
  │   ├─ LLMRouterV2 ←── CircuitBreaker
  │   ├─ LLMClientV2 ←── RouterV2, Cache, CircuitBreaker, Governance, Tracer
  │   ├─ LLMSemanticCache ←── EmbeddingManager
  │   ├─ GovernanceManager ←── PIIScrubber, AuditLogger, QualityOptimizer
  │   ├─ EUAIActClassifier
  │   ├─ DataLineageTracker
  │   ├─ SkillRegistry
  │   ├─ MCPClientV2 ←── StdioTransport | HTTPTransport
  │   ├─ EmbeddingManager
  │   └─ TracingExporter
  │
  ├─► Layer 1
  │   └─ TriggerManager
  │
  ├─► Layer 2
  │   └─ PerceptionEngine ←── LLMClientV2, PIIScrubber
  │
  ├─► Layer 3
  │   ├─ SwarmRegistry ←── Agent factory
  │   ├─ CheckpointStore
  │   ├─ HandoffManager ←── SwarmRegistry, Tracer, Audit
  │   ├─ GuardrailsEngine
  │   └─ MasterOrchestratorV2 ←── 以上全部
  │
  ├─► Layer 4
  │   ├─ 19 Agents (BaseAgent | BaseAgentV2)
  │   ├─ 10 Skills (via SkillRegistry)
  │   └─ 14+ MCP Servers (via MCPClientV2)
  │
  ├─► Layer 5
  │   ├─ RAGEngineV2 ←── EmbeddingManager, VectorStoreAdapter
  │   └─ MemoryConsolidator ←── RAGEngineV2, EmbeddingManager
  │
  └─► Layer 6
      └─ A2AAgent ←── A2AClient, MasterOrchestratorV2
```

## 附錄 B: init_system() v2 啟動順序

```python
async def init_system() -> dict:
    """v2 完整啟動順序"""

    # ── Phase 0: Observability (first, so everything is traced) ──
    tracer  = TracingExporter(backend=env.TRACING_BACKEND)
    setup_logging(level=env.LOG_LEVEL, format=env.LOG_FORMAT)

    # ── Phase 1: Foundation ──
    circuit_breaker = CircuitBreaker()
    embedder        = EmbeddingManager()
    cache           = LLMSemanticCache(embedder=embedder)
    router          = LLMRouterV2(circuit_breaker=circuit_breaker)
    governance      = GovernanceManager()
    llm_client      = LLMClientV2(
        governance=governance, cache=cache,
        circuit_breaker=circuit_breaker, tracer=tracer,
    )

    rag_engine      = RAGEngineV2(embedder=embedder)
    lineage         = DataLineageTracker()
    skills          = SkillRegistry(); skills.load_all()
    classifier      = EUAIActClassifier()

    # ── Phase 2: MCP ──
    mcp_client = MCPClientV2()
    mcp_config = load_yaml("config/mcp/servers.yaml")
    for name, cfg in mcp_config["servers"].items():
        if cfg.get("auto_start"):
            await mcp_client.connect(name, cfg)

    # ── Phase 3: Orchestration ──
    checkpoint_store = CheckpointStore()
    guardrails       = GuardrailsEngine()
    guardrails.load_rules()

    swarm_registry   = SwarmRegistry()
    shared_deps = {
        "router": router, "llm_client": llm_client,
        "rag_engine": rag_engine, "governance": governance,
        "skills": skills, "mcp_client": mcp_client,
        "lineage": lineage, "classifier": classifier,
        "tracer": tracer,
    }
    swarm_registry.load_all(shared_deps)

    handoff_manager = HandoffManager(
        swarm_registry=swarm_registry, tracer=tracer,
        audit=governance.audit,
    )

    orchestrator = MasterOrchestratorV2(
        resource_pool=ResourcePool(),
        swarm_registry=swarm_registry,
        checkpoint_store=checkpoint_store,
        handoff_manager=handoff_manager,
        guardrails=guardrails,
        quality_optimizer=governance.optimizer,
        tracer=tracer,
    )

    # ── Phase 4: Triggers ──
    trigger_manager = TriggerManager()

    # ── Phase 5: Perception ──
    perception = PerceptionEngine(llm_client=llm_client, governance=governance)

    # ── Phase 6: Background tasks ──
    consolidator = MemoryConsolidator(rag_engine=rag_engine, embedder=embedder)
    # Start periodic consolidation (every 6 hours)

    # ── Phase 7: Plugin loading ──
    plugin_loader = PluginLoader()
    plugin_loader.load_all()

    return {
        "router": router,
        "llm_client": llm_client,
        "rag_engine": rag_engine,
        "governance": governance,
        "skills": skills,
        "mcp_client": mcp_client,
        "trigger_manager": trigger_manager,
        "perception": perception,
        "orchestrator": orchestrator,
        "checkpoint_store": checkpoint_store,
        "handoff_manager": handoff_manager,
        "guardrails": guardrails,
        "lineage": lineage,
        "tracer": tracer,
        "consolidator": consolidator,
        "plugin_loader": plugin_loader,
    }
```

---

*文件版本: 2.0.0-plan*
*建立日期: 2026-02-27*
*基於: v1 完整程式碼審計 + 業界研究 (Anthropic / Google / OpenAI / Top 50 GitHub)*
