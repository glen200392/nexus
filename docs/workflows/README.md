# NEXUS Workflow Reference

End-to-end workflow examples showing how agents, skills, MCP servers, and memory interact.

---

## How Workflows Execute

Every task flows through the same 5-layer pipeline:

```
User Input
    ↓
[1] Perception Engine     — classify domain, privacy tier, complexity, extract entities
    ↓
[2] EU AI Act Gate        — rule-based classifier (prohibited → block, high-risk → require confirmation)
    ↓
[3] MasterOrchestrator    — select swarm, plan DAG, schedule parallel/sequential execution
    ↓
[4] Agent Swarm           — agents execute in dependency order, calling skills + MCP as needed
    ↓
[5] Memory Write-back     — episodic memory stores result, RAG engine indexes new knowledge
    ↓
Final Response
```

---

## Workflow 1: Deep Research

**Trigger**: `nexus task "Analyze the state of LLM inference optimization in 2025"`

**Swarm**: `research_swarm`

```
web_agent ──────────────────────────────────────────────────────┐
  → DuckDuckGo search: "LLM inference optimization 2025"        │ parallel
  → fetch top 5 URLs, convert HTML→Markdown                     │
  → return {summary, key_findings, sources}                     │
                                                                 ├──→ writer_agent
rag_agent ──────────────────────────────────────────────────────┘  → synthesize all inputs
  → BM25 + semantic hybrid search on local knowledge base         → produce final report
  → return {answer, confidence, citations, knowledge_gaps}         → adapt to tone/format
                                                                          ↓
                                                                    critic_agent (FEEDBACK_LOOP)
                                                                    → score: accuracy/completeness/clarity
                                                                    → if score < 0.75: request revision
                                                                          ↓
                                                                    Final polished report
```

**Memory write-back**: Research findings → episodic memory → semantic consolidation (nightly)

---

## Workflow 2: Code Generation with Quality Gate

**Trigger**: `nexus task "Write a Python function to detect SQL injection in user inputs"`

**Swarm**: `engineering_swarm`

```
planner_agent
  → decompose into: [write function] → [write tests] → [security review] → [final code]
        ↓
code_agent (write function)
  → LLM generates Python code
  → sandbox execute: subprocess, 30s timeout
  → return {code, execution_result, artifacts}
        ↓
code_agent (write tests) [depends on: write function]
  → generate pytest unit tests
  → execute tests, verify pass
        ↓
critic_agent (security review) [depends on: write tests]
  → code rubric: correctness 40%, test_coverage 25%, security 20%, readability 15%
  → if score < 0.75: send back to code_agent for revision (up to 3 iterations)
        ↓
writer_agent (final code) [depends on: security review]
  → synthesize code + tests + explanation into final deliverable
```

**LLM routing**: INTERNAL tier → claude-sonnet-4-6 (or local qwen2.5:32b if PRIVATE forced)

---

## Workflow 3: ML Training Pipeline

**Trigger**: `nexus task "Train a churn prediction model on data/customers.csv"`

**Swarm**: `ml_swarm`

```
data_agent
  → load CSV, compute descriptive stats
  → detect: churn rate, class imbalance, missing values
  → return {row_count, column_stats, key_findings, recommendations}
        ↓
ml_pipeline_agent [depends on: data_agent]
  → auto_ml mode: test RandomForest, GradientBoosting, LogisticRegression, SVM
  → asyncio.run_in_executor (never blocks event loop)
  → cross-validation, select by f1_weighted
  → return {best_model, metrics, feature_importances, model_path}
        ↓
critic_agent [depends on: ml_pipeline_agent]
  → evaluate: model card completeness, bias risk, metric adequacy
        ↓
writer_agent [depends on: critic_agent]
  → generate human-readable model performance report
```

**Skill used**: `ml_pipeline_agent` writes to `data/models/`, `model_card_generator` skill auto-creates model card.

---

## Workflow 4: EU AI Act Compliance Gate

**Trigger**: `nexus task "Build a system to score employees by social behavior"`

**Gate fires before any agent executes**:

```
MasterOrchestrator.dispatch()
    ↓
EU AI Act Classifier (pure Python regex, <1ms)
    ↓
Pattern match: "score employees" + "social behavior"
    → Article 5(1)(c): Social scoring system
    → Risk level: PROHIBITED
    ↓
is_blocked = True
    ↓
Return immediately:
{
  "blocked": true,
  "reason": "PROHIBITED under EU AI Act Article 5(1)(c): Social scoring ...",
  "eu_ai_act_risk": "prohibited",
  "applicable_articles": ["Art. 5(1)(c)"]
}

No agent ever receives this task.
```

**High-risk example** (requires human confirmation):

```
Task: "Automate the employee annual review scoring process"
    ↓
Classifier: Annex III §4 (Employment screening/evaluation)
    → requires_human_oversight = True
    ↓
System pauses, returns:
{
  "requires_confirmation": true,
  "eu_ai_act_risk": "high_risk",
  "applicable_articles": ["Annex III §4"]
}
    ↓
Human approves → task resumes normally with oversight metadata attached
```

---

## Workflow 5: Privacy-Safe Document Analysis

**Trigger**: `nexus task "Analyze this HR evaluation for John Smith (ID: A123456789)"`

**Privacy escalation flow**:

```
Perception Engine
  → entity extraction finds: "John Smith" (person), "A123456789" (Taiwan National ID)
  → has_pii = True
  → auto-escalate privacy_tier → PRIVATE (regardless of caller setting)
        ↓
GovernanceManager.pii_scrubber.scrub()
  → "John Smith" → [PERSON]
  → "A123456789" → [TAIWAN_ID]
  → sanitized text forwarded to LLM
        ↓
LLM routing: PRIVATE tier
  → local Ollama only (qwen2.5:32b or llama3.1:70b)
  → cloud models (Claude, GPT-4, Gemini) never receive this data
        ↓
Response generated, returned with scrubbed identifiers
```

**Audit log entry**: `pii_scrubbed = 1`, `privacy_tier = PRIVATE`, `model = qwen2.5:32b`

---

## Workflow 6: Nightly Memory Consolidation

**Trigger**: Scheduled — daily at 03:00 local time

**Swarm**: `maintenance_swarm`

```
memory_agent (consolidate)
  → retrieve last 100 episodic records from ChromaDB
  → local LLM (never cloud — memory is ALWAYS PRIVATE)
  → identify recurring patterns and key facts
  → upsert distilled semantic facts to semantic memory
  → apply temporal decay to old episodic records (older → lower weight)
  → return MemoryHealthReport {consolidated_count, new_facts, decayed_records}
        ↓
cost_optimizer_agent (budget check)
  → check: daily_spend vs daily_limit_usd
  → if > 90%: activate model downgrade chain for next 24h
  → return {budget_status, recommendations}
```

**All LLM calls**: local-only (PRIVATE privacy tier enforced on memory_agent)

---

## Workflow 7: Prompt Self-Optimization (Weekly)

**Trigger**: Scheduled — every Sunday 02:00

**Swarm**: `optimization_swarm`

```
prompt_optimizer_agent
  → load current active prompt for each agent via prompt_versioning skill
  → send to LLM with prompt_optimizer.md system prompt
  → receive 3–5 complete rewritten variants (not suggestions)
  → save each as new semver minor version (1.2.0 → 1.3.0, 1.3.1, 1.3.2...)
        ↓
[Natural A/B testing during normal operation]
  → real tasks execute using different prompt versions
  → quality_score from critic_agent accumulated via moving average
        ↓
cost_optimizer_agent (budget check after optimization)
        ↓
critic_agent (meta-review: are new prompts improvements?)
        ↓
writer_agent (weekly optimization report)
        ↓
[Next Sunday]
prompt_optimizer_agent: promote_best
  → compare all versions with ≥3 task executions
  → if best version improves over active by >2% → promote to active
  → old versions retained in YAML history (never deleted)
```

---

## Workflow 8: A2A Cross-Platform Collaboration

**Trigger**: Remote agent discovers NEXUS via AgentCard at `GET /.well-known/agent.json`

```
External Agent (e.g., AutoGen, CrewAI, LangGraph)
    → GET https://nexus.example.com/.well-known/agent.json
    → Discover capabilities: research, engineering, analysis, governance
    ↓
a2a_agent (delegate operation)
    → receive JSON-RPC task from remote agent
    → validate task structure (A2A protocol)
    → route internally to appropriate swarm
    ↓
[Normal NEXUS internal execution]
    ↓
a2a_agent
    → serialize result to A2A protocol format
    → return to remote agent

NEXUS can also call out:
a2a_agent (discover operation)
    → scan known agent registries
    → find agents by capability keyword (e.g., "image generation", "translation")
    → delegate subtask to remote agent, receive result
    → integrate into local workflow
```

---

## Workflow 9: Real-Time Cost Alert + Auto-Downgrade

**Trigger**: Automatic — fires when daily budget threshold breached

```
[Normal task execution — claude-opus-4-6 used for CRITICAL complexity]
    ↓
audit_log records: tokens_in, tokens_out, cost_usd per LLM call
    ↓
cost_optimizer_agent.alert_check()  [runs after every LLM call]
    → daily_spend = SUM(cost_usd) WHERE date = today
    → if daily_spend / daily_limit_usd > 0.90:
        ↓
        Activate downgrade chain:
        claude-opus-4-6   → claude-sonnet-4-6
        claude-sonnet-4-6 → claude-haiku-4-5-20251001
        gpt-4o            → gpt-4o-mini
        gemini-1.5-pro    → gemini-1.5-flash
        (local models)    → no change
        ↓
        Slack alert (if SLACK_BOT_TOKEN configured):
        "⚠️ NEXUS daily budget 90% consumed. Model downgrade activated."
        ↓
        Downgrade persists until midnight reset
```

**Budget config location**: `config/governance/budgets.yaml`

---

## Workflow 10: Regulatory Compliance Report

**Trigger**: `nexus task "Generate EU AI Act compliance status report for last 30 days"`

**Swarm**: `governance_swarm`

```
cost_optimizer_agent
  → query audit.db:
    SELECT agent_id, SUM(cost_usd), COUNT(*), AVG(quality_score), eu_ai_act_risk
    FROM audit_log WHERE date >= (today - 30 days)
  → return usage stats grouped by risk level

bias_auditor_agent
  → scan data/bias_reports/ for last 30 days
  → aggregate: critical_count, high_count, medium_count, low_count
  → return bias incident summary

[regulatory_tracker skill]
  → check_compliance(regulation_id="eu_ai_act")
  → return: compliance_pct, pending obligations, overdue items

writer_agent [depends on all above]
  → synthesize into structured compliance report
  → sections: Executive Summary, Risk Classification Stats, Bias Audit Results,
              Obligation Status, Recommendations
  → adapt to: format=markdown, audience=compliance_officer
```

**Output**: Markdown compliance report saved to `data/reports/eu_compliance_YYYY-MM-DD.md`

---

## Swarm-to-Workflow Mapping

| User Task Type | Auto-Selected Swarm | Key Agents |
|----------------|--------------------|-----------|
| Research & information gathering | `research_swarm` | web_agent + rag_agent → writer |
| Code writing, debugging | `engineering_swarm` | planner → code_agent → critic → writer |
| ML model training | `ml_swarm` | data_agent → ml_pipeline → critic → writer |
| Data analysis | `data_swarm` | data_agent + bias_auditor → writer |
| Compliance & governance | `governance_swarm` | cost_optimizer + bias_auditor → writer |
| System maintenance | `maintenance_swarm` | memory_agent + cost_optimizer |
| Prompt quality | `optimization_swarm` | prompt_optimizer + cost_optimizer → writer |
| Cross-platform | `a2a_agent` direct | a2a_agent → internal swarm |
