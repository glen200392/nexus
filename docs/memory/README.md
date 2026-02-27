# NEXUS Memory System

NEXUS implements a 5-tier memory architecture inspired by cognitive science, plus a graph-based data lineage system.

---

## Memory Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS MEMORY                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Episodic   │  │   Semantic   │  │     Procedural       │ │
│  │   Memory     │  │   Memory     │  │     Memory           │ │
│  │              │  │              │  │                      │ │
│  │ Raw events,  │  │ Distilled    │  │ Learned strategies,  │ │
│  │ task results │  │ facts, rules │  │ prompt templates     │ │
│  │ (ChromaDB)   │  │ (ChromaDB)   │  │ (config/prompts/)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘ │
│         │                 ↑                                     │
│         └─── nightly ─────┘                                    │
│               memory_agent.consolidate()                        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────────────────────────┐   │
│  │   Working    │  │            External                  │   │
│  │   Memory     │  │            Memory                    │   │
│  │              │  │                                      │   │
│  │ Current task │  │ Knowledge base (docs/PDFs/URLs)      │   │
│  │ context,     │  │ ingested by rag_agent, retrieved     │   │
│  │ agent state  │  │ by BM25 + semantic hybrid search     │   │
│  │ (in-process) │  │ (ChromaDB + BM25 index)              │   │
│  └──────────────┘  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Type 1: Episodic Memory

**What**: Raw task execution records — every time an agent executes, a record is stored.

**Storage**: ChromaDB collection `nexus_episodic`

**Record structure**:
```json
{
  "task_id": "task_abc123",
  "timestamp": "2025-08-15T14:32:10Z",
  "agent_id": "web_agent",
  "user_message": "Research LLM inference optimization",
  "result_summary": "Found 5 key papers on speculative decoding...",
  "quality_score": 0.87,
  "domain": "research",
  "privacy_tier": "INTERNAL"
}
```

**Retrieval**: Vector similarity search — new tasks auto-retrieve relevant past experiences and inject into `AgentContext.retrieved_memory`

**Retention**: Raw episodic records decay over time. Records older than 90 days have weight reduced by 50% per consolidation cycle.

---

## Memory Type 2: Semantic Memory

**What**: Distilled facts and patterns extracted from episodic records by `memory_agent`.

**Storage**: ChromaDB collection `nexus_semantic`

**How it grows**:
1. `memory_agent.consolidate()` runs nightly at 03:00
2. Retrieves last N episodic records
3. Local LLM (PRIVATE — never cloud) identifies patterns:
   - "Tasks involving pandas DataFrames often fail with encoding errors on Windows"
   - "User prefers technical tone for research reports, executive tone for summaries"
   - "The churn prediction dataset requires log-transform on TotalCharges before training"
4. Upserts distilled semantic facts to ChromaDB with `memory_type=semantic` tag

**Why local-only**: Semantic memory may contain patterns extracted from sensitive episodic data. Privacy tier is always PRIVATE.

---

## Memory Type 3: Procedural Memory

**What**: Learned behaviors encoded in versioned system prompts.

**Storage**: YAML files in `config/prompts/versions/<agent_id>/`

**Structure**:
```yaml
# config/prompts/versions/code_agent/v1.3.yaml
version: "1.3.0"
agent_id: code_agent
content: |
  You are an expert software engineer...
  [Full system prompt content]
created_at: "2025-08-01T00:00:00Z"
is_active: true
quality_stats:
  avg_score: 0.847
  task_count: 234
  last_updated: "2025-08-14T03:00:00Z"
```

**How it improves**: `prompt_optimizer_agent` generates variants weekly → A/B tested via real task execution → `promote_best` activates the winner if improvement > 2%.

**Rollback**: Any previous version can be activated via `prompt_versioning.rollback()`.

---

## Memory Type 4: Working Memory

**What**: In-process state for the current task execution — not persisted between tasks.

**Contents** (the `AgentContext` object):
```python
@dataclass
class AgentContext:
    task_id:          str           # unique task identifier
    user_message:     str           # original user request
    history:          list[dict]    # conversation turns so far
    privacy_tier:     PrivacyTier   # PRIVATE | INTERNAL | PUBLIC
    complexity:       TaskComplexity # LOW | MEDIUM | HIGH | CRITICAL
    domain:           TaskDomain    # RESEARCH | ENGINEERING | ...
    retrieved_memory: list[dict]    # auto-injected RAG results
    metadata:         dict          # agent-specific operation params
```

**Lifetime**: Created by `MasterOrchestrator`, passed through the entire agent chain, discarded on task completion.

---

## Memory Type 5: External Memory (Knowledge Base)

**What**: The RAG knowledge base — structured documents, PDFs, web pages ingested and made searchable.

**Storage**: ChromaDB collection `nexus_knowledge` + BM25 index (`data/bm25_index.pkl`)

**Ingestion**:
```python
# Via rag_agent ingest operation
result = await rag_agent.execute(AgentContext(
    metadata={"operation": "ingest", "source": "path/to/document.pdf"}
))
```

**Retrieval — Hybrid BM25 + Semantic**:

```
Query: "What are the EU AI Act high-risk categories?"
        ↓
BM25 search (keyword)          Semantic search (embedding)
  → exact term matches         → conceptual similarity
  → score_bm25                 → score_cosine
        ↓                              ↓
              Reciprocal Rank Fusion
              final_score = RRF(bm25_rank, semantic_rank)
                    ↓
              Top-k results returned with citations
```

**Privacy filtering**: Documents tagged `privacy_tier=PRIVATE` are only retrieved when `context.privacy_tier == PRIVATE`.

**Temporal decay**: Older documents receive lower retrieval weight unless explicitly pinned.

---

## Memory Consolidation Pipeline

The full nightly cycle (03:00 daily, `maintenance_swarm`):

```
memory_agent.consolidate()
    │
    ├─ 1. retrieve_episodic(last_n=100)
    │      ChromaDB similarity search across recent records
    │
    ├─ 2. identify_patterns(local_llm)
    │      Prompt: "What recurring patterns do you see in these task records?"
    │      Returns: list of semantic facts
    │
    ├─ 3. upsert_semantic(facts)
    │      Add/update ChromaDB semantic records
    │      Deduplication: cosine similarity > 0.95 → merge, not duplicate
    │
    ├─ 4. apply_temporal_decay()
    │      episodic records > 30 days: weight × 0.9
    │      episodic records > 90 days: weight × 0.5
    │
    └─ 5. return MemoryHealthReport
           {
             "consolidated_count": 47,
             "new_semantic_facts": 12,
             "decayed_records": 8,
             "total_episodic": 2341,
             "total_semantic": 189
           }
```

---

## RAG Engine Details

**File**: `core/rag_engine.py`

**Retrieval pipeline**:

1. **Query expansion**: LLM generates 3 query variants to broaden recall
2. **BM25 retrieval**: TF-IDF keyword matching (fast, no embedding cost)
3. **Semantic retrieval**: Embedding similarity via ChromaDB
4. **Reciprocal Rank Fusion**: Merge ranked lists without needing calibrated scores
5. **Privacy filter**: Drop documents above current privacy tier
6. **Temporal reranking**: Newer documents boosted by small factor
7. **Deduplication**: Remove duplicate chunks from same source document
8. **Return top-k**: With `doc_id`, `score`, `snippet`, `source_metadata`

**Embedding models**:
- Default: `text-embedding-3-small` (OpenAI, if API key present)
- Fallback: `sentence-transformers/all-MiniLM-L6-v2` (local, zero cost)

---

## Data Lineage

Beyond memory, NEXUS tracks the full provenance of every data artifact.

### Graph Model

```
task_abc ──CONSUMED──→ file_sales_q3.csv
task_abc ──CONSUMED──→ knowledge_doc_xyz
task_abc ──PRODUCED──→ report_board.md
report_board.md ──STORED_IN──→ knowledge_doc_abc
```

**Node types**: `data_source`, `agent_task`, `transformation`, `model`, `output`, `knowledge_doc`

**Edge types**: `PRODUCED`, `CONSUMED`, `DERIVED_FROM`, `TRAINED_ON`, `STORED_IN`, `TRIGGERED_BY`

### Backends

**Neo4j** (if `NEO4J_URI` configured): Full Cypher query engine, enterprise-grade graph DB.

**NetworkX + JSON** (fallback): Pure Python, persisted to `data/lineage_graph.json`. Identical API surface — zero code changes needed when switching backends.

### Queries

```python
from nexus.core.lineage import get_tracker

tracker = get_tracker()

# Where did this output come from?
upstream = tracker.trace_back("report_board.md", depth=5)
# Returns: list of nodes from source → current

# What depends on this data source?
downstream = tracker.get_downstream("customer_data.csv", depth=5)
# Returns: all artifacts derived from this source

# Full provenance graph
lineage = tracker.get_lineage("report_board.md", depth=3)
# Returns: {nodes: [...], edges: [...]}

# Export as Graphviz DOT for visualization
dot = tracker.export_dot()
# Returns: DOT string — pipe to `dot -Tsvg` to render
```

### Example: Tracing a Report's Origin

```
User: "Where did the Q3 churn report come from?"

tracker.trace_back("report_q3_churn.md", depth=5)
→ [
    {node: "report_q3_churn.md",        type: "output"},
    {node: "task_ml_train_456",         type: "agent_task",    agent: "ml_pipeline_agent"},
    {node: "gradient_boosting_1234.pkl",type: "model",         produced_at: "2025-08-01"},
    {node: "task_data_clean_123",       type: "agent_task",    agent: "data_agent"},
    {node: "customers_raw.csv",         type: "data_source",   ingested_at: "2025-07-28"}
  ]
```

---

## Memory Privacy Rules

| Memory Type | Privacy Tier | Allowed LLMs |
|-------------|-------------|--------------|
| Episodic | PRIVATE | Local only |
| Semantic | PRIVATE | Local only |
| Procedural | INTERNAL | Small cloud OK |
| Working | Inherited from task | Follows task routing |
| External (RAG) | Per-document tag | Follows document tag |
| Data Lineage | PRIVATE | Local only (metadata) |

The `memory_agent` enforces PRIVATE tier on all memory operations — it will **never** use cloud LLMs regardless of system configuration.
