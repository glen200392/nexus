# NEXUS Agents Reference

19 specialized agents organized by domain. All agents inherit `BaseAgent` and are registered in `core/orchestrator/swarm.py`.

---

## Agent Contract

Every agent implements:

```python
class BaseAgent:
    agent_id:           str            # unique identifier
    agent_name:         str            # human-readable name
    description:        str            # used by orchestrator for routing
    domain:             TaskDomain     # RESEARCH|ENGINEERING|OPERATIONS|ANALYSIS|CREATIVE|ORCHESTRATION
    default_complexity: TaskComplexity # LOW|MEDIUM|HIGH|CRITICAL
    default_privacy:    PrivacyTier    # PRIVATE|INTERNAL|PUBLIC

    async def execute(context: AgentContext) -> AgentResult
```

**`AgentContext`** — what every agent receives:
- `task_id`, `user_message`, `history` (conversation so far)
- `privacy_tier`, `complexity`, `domain`
- `retrieved_memory` (RAG results auto-injected)
- `metadata` (operation-specific parameters)

**`AgentResult`** — what every agent returns:
- `success: bool`
- `output: Any` (agent-specific structure)
- `quality_score: float` (0.0–1.0)
- `cost_usd: float`
- `error: Optional[str]`

---

## Research Domain

### `web_agent`
**Purpose**: Search the internet and fetch web content.

**LLM routing**: INTERNAL tier (MEDIUM complexity → sonnet; PRIVATE → qwen2.5:32b)

**Tools used**:
- DuckDuckGo search API
- Direct HTTP fetch + HTML→Markdown conversion

**Output format**:
```json
{
  "summary": "...",
  "key_findings": ["finding 1", "finding 2"],
  "sources": [{"title": "...", "url": "...", "snippet": "..."}]
}
```

---

### `rag_agent`
**Purpose**: Query the local knowledge base using hybrid BM25 + semantic retrieval.

**LLM routing**: PRIVATE tier preferred (knowledge base may contain sensitive data)

**Output format**:
```json
{
  "answer": "...",
  "confidence": 0.85,
  "citations": [{"doc_id": "...", "score": 0.91, "snippet": "..."}],
  "knowledge_gaps": ["topic not in knowledge base"]
}
```

---

### `critic_agent`
**Purpose**: Quality-gate other agents' outputs. Used in FEEDBACK_LOOP and ADVERSARIAL patterns.

**LLM routing**: LOW complexity (haiku/7b — fast and cheap for scoring)

**Default rubric** (customizable per swarm YAML):
```
accuracy:     35%  — Is the information correct?
completeness: 25%  — Does it cover all required aspects?
clarity:      20%  — Is it understandable?
usefulness:   20%  — Is it actionable?
```

**Code rubric** (used in engineering_swarm):
```
correctness:    40%
test_coverage:  25%
security:       20%
readability:    15%
```

**Verdict thresholds**: score ≥ 0.75 → `accept`; 0.5–0.75 → `revise`; < 0.5 → `reject`

---

## Engineering Domain

### `code_agent`
**Purpose**: Write, modify, debug, and execute code.

**Languages**: Python, JavaScript, TypeScript, Bash, SQL

**Execution**: Sandboxed subprocess, 30s timeout, restricted to current directory

**Output format**:
```json
{
  "code": "...",
  "language": "python",
  "execution_result": {"stdout": "...", "stderr": "", "exit_code": 0},
  "artifacts": ["path/to/created_file.py"],
  "explanation": "..."
}
```

---

### `ml_pipeline_agent`
**Purpose**: Train, evaluate, and interpret ML models using scikit-learn.

**Runs in**: `asyncio.run_in_executor()` — never blocks the event loop

**Supported models**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Ridge

**`auto_ml` mode**: Tests all models, selects best by f1_weighted (classification) or r2 (regression)

**Output format**:
```json
{
  "best_model": "GradientBoostingClassifier",
  "metrics": {"f1_weighted": 0.847, "accuracy": 0.876, "auc_roc": 0.923},
  "feature_importances": [{"feature": "tenure", "importance": 0.189}],
  "model_path": "data/models/gradient_boosting_1740614423.pkl",
  "interpretation": "..."
}
```

---

### `prompt_optimizer_agent`
**Purpose**: DSPy-style systematic prompt improvement.

**Operations** (via `context.metadata["operation"]`):

| Operation | Description |
|-----------|-------------|
| `full_cycle` | Analyze weaknesses + generate 3–5 complete variants (default) |
| `analyze` | Diagnose weaknesses only |
| `recommend` | Compare quality stats across versions (requires ≥3 tasks/version) |
| `promote_best` | Activate the highest-quality version if improvement > 2% |

**How it works**:
1. Loads current active prompt via `prompt_versioning` skill
2. Sends to LLM with `prompt_optimizer.md` system prompt
3. Receives complete rewritten variants (not suggestions)
4. Saves each as a new version (minor bump) via `prompt_versioning`
5. Quality accumulates naturally as real tasks execute
6. `promote_best` runs weekly via scheduler

---

## Operations Domain

### `shell_agent`
**Purpose**: Execute shell commands with safety guardrails.

**Privacy**: Always PRIVATE (shell output may contain secrets)

**Blocked commands** (hard stop):
```
rm -rf /  •  dd if=...  •  :(){ :|:& };:  •  chmod 777 /
sudo su   •  mkfs       •  iptables -F   •  curl | bash
```

**Requires confirmation** (interactive prompt):
```
rm / trash / git push / git reset --hard / docker rm -f
kill / pkill / DROP TABLE / DELETE without WHERE
```

**Output format**:
```json
{
  "stdout": "...",
  "stderr": "",
  "exit_code": 0,
  "command": "ls -la",
  "duration_ms": 45
}
```

---

### `browser_agent`
**Purpose**: Automate browser interactions using Playwright.

**Browser**: Local Chromium (via Playwright)

**Fallback**: httpx + HTML parsing for simple fetch operations

**Actions**: navigate, click, fill_form, extract_text, screenshot, scroll, wait_for_element

**Output format**:
```json
{
  "action": "screenshot",
  "url": "https://example.com",
  "screenshot_b64": "...",
  "extracted_text": "...",
  "success": true
}
```

---

### `memory_agent`
**Purpose**: Consolidate episodic memory into semantic knowledge.

**Privacy**: Always PRIVATE — memory content is never sent to cloud models

**Operations**: `consolidate`, `distill`, `forget`, `reflect`, `report`

**`consolidate` workflow**:
1. Retrieve last N episodic records from ChromaDB
2. LLM (local): identify patterns and generate semantic facts
3. Upsert distilled facts to semantic memory
4. Apply temporal decay to old records
5. Return `MemoryHealthReport`

---

### `email_agent`
**Purpose**: Draft, review, and send emails.

**Privacy**: INTERNAL (may contain professional personal data)

**Safety**: Always shows draft for review. Requires explicit `send=True` in metadata to actually send.

**Transport**: SMTP (Gmail default) or Resend API

**Operations**: `draft`, `send`, `reply`, `summarize`

---

### `a2a_agent`
**Purpose**: Connect NEXUS to the Google Agent-to-Agent (A2A) federation protocol.

**AgentCard published at**: `GET /.well-known/agent.json`

**Operations**: `delegate` (send task to remote), `discover` (find agents by capability), `broadcast`

---

### `cost_optimizer_agent`
**Purpose**: Monitor LLM spending and enforce budget constraints.

**Budget config**: `config/governance/budgets.yaml`

**Auto-downgrade chain**:
```
claude-opus-4-6   → claude-sonnet-4-6  → claude-haiku-4-5
gpt-4o            → gpt-4o-mini
gemini-1.5-pro    → gemini-1.5-flash
```

**Operations**: `check_budget`, `report`, `optimize`, `set_budget`, `alert_check`

---

### `bias_auditor_agent`
**Purpose**: Detect demographic bias and representational harm in AI outputs.

**Two-layer detection**:
1. **Rule scan** (17 regex, 0 token cost): stereotyping, exclusionary language, demographic profiling
2. **LLM analysis** (`bias_auditor.md` system prompt): deep fairness assessment

**Risk levels**: `low` → `medium` → `high` → `critical`

High/critical results automatically saved to `data/bias_reports/` and flagged for human review.

**Operations**: `audit_output`, `batch_audit`, `generate_report`, `flag_for_review`

---

## Analysis Domain

### `data_agent`
**Purpose**: Load structured data files and generate statistical summaries.

**Input formats**: CSV, Excel, JSON, Parquet, SQL query results

**Operations**: descriptive stats, correlation analysis, anomaly detection, matplotlib chart generation

**Output format**:
```json
{
  "row_count": 10000,
  "column_stats": {...},
  "key_findings": ["Churn rate: 14.2%", "Missing TotalCharges: 11 rows"],
  "charts": ["base64_png..."],
  "recommendations": ["Address class imbalance before training"]
}
```

---

## Creative Domain

### `writer_agent`
**Purpose**: Synthesize multi-source findings into polished output.

**Role in workflows**: Always the last agent in a chain. Receives all previous agents' outputs and produces the final human-readable result.

**Adapts to**: Tone (technical/executive/conversational), format (Markdown/HTML/plain), language (zh-TW/en)

---

## Orchestration Domain

### `planner_agent`
**Purpose**: Decompose complex tasks into executable DAGs.

**Output**: `ExecutionPlan` with subtasks and `depends_on` dependency graph

---

## YAML Aliases

The following agent IDs appear in swarm YAML files and map to existing implementations:

| YAML Agent ID | Maps To | Rationale |
|---------------|---------|-----------|
| `analyst_agent` | `WriterAgent` | Synthesis + analysis role |
| `test_agent` | `CodeAgent` | Code execution for testing |
| `review_agent` | `CriticAgent` | Code review = quality critique |
| `proposal_agent` | `PlannerAgent` | Structured proposals |
| `judge_agent` | `CriticAgent` | Quality judgment between proposals |
