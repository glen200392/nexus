# NEXUS AI Governance

NEXUS has first-class AI governance at every architectural layer. This document covers the full governance stack.

---

## EU AI Act Compliance

### Overview

Every task entering `MasterOrchestrator.dispatch()` is classified by `EUAIActClassifier` before any workflow planning or agent execution begins.

**Classification is:**
- Rule-based (pure Python regex)
- Zero LLM cost
- Sub-millisecond latency
- Deterministic (same input → same output always)
- Cannot be bypassed by agent code

### Risk Levels

| Level | Trigger | System Response |
|-------|---------|-----------------|
| `PROHIBITED` | Art. 5 patterns | `is_blocked=True` → task rejected immediately |
| `HIGH_RISK` | Annex III §1–§8 | `requires_human_oversight=True` → human confirmation required |
| `LIMITED_RISK` | Art. 52 transparency | Continue, add transparency obligation metadata |
| `MINIMAL_RISK` | Default | Continue normally |
| `GPAI` | General-purpose AI deployment | Art. 53 documentation obligations added |

### Prohibited Practices (Article 5)

Tasks matching any of these patterns are **permanently blocked**:

| Pattern | Article | Example Trigger |
|---------|---------|-----------------|
| Social scoring | Art. 5(1)(c) | "score citizens by social behavior" |
| Subliminal manipulation | Art. 5(1)(a) | "subliminal messages to influence behavior" |
| Real-time public biometric surveillance | Art. 5(1)(h) | "real-time biometric identification in public" |
| Exploiting vulnerable groups | Art. 5(1)(b) | "exploit cognitive disability to manipulate" |
| Biometric criminal prediction | Art. 5(1)(e) | "predict criminal likelihood from biometrics" |
| Emotion recognition at work/school | Art. 5(1)(f) | "emotion recognition in workplace" |

### High-Risk Categories (Annex III)

Tasks requiring human oversight:

| Category | Annex III | Trigger Examples |
|----------|-----------|-----------------|
| Biometric identification | §1 | "biometric categorization system" |
| Critical infrastructure | §2 | "power grid automated control" |
| Education assessment | §3 | "student admission scoring by AI" |
| Employment screening | §4 | "recruit candidates using AI", "automated employee evaluation" |
| Essential services | §5 | "credit scoring model", "insurance risk AI" |
| Law enforcement | §6 | "crime prediction system" |
| Migration/border | §7 | "asylum decision automation" |
| Administration of justice | §8 | "judicial decision prediction" |
| Medical diagnosis | §1 (medical) | "automated clinical diagnosis" |

### Integration Point

```python
# In MasterOrchestrator.dispatch()
check = classifier.classify(
    task_type=perceived_task.get("task_type", ""),
    domain=perceived_task.get("domain", ""),
    description=perceived_task.get("user_message", ""),
)
allowed, explanation = classifier.gate_task(check)

if not allowed:
    # Return blocked result — task never reaches any agent
    return blocked_result(explanation)

if check.requires_human_oversight:
    # Pause and require human "confirm" before continuing
    perceived_task["requires_confirmation"] = True
```

### Obligation Tracking

Use the `regulatory_tracker` skill to track EU AI Act compliance obligations:

```python
result = await skill.run(
    operation="check_compliance",
    regulation_id="eu_ai_act"
)
# Returns: compliance_pct, pending obligations, overdue items
```

8 pre-loaded EU AI Act obligations covering Art. 9–15 and Art. 49.

---

## Privacy Protection

### Three-Tier Privacy Model

| Tier | LLM Routing | PII Handling | Use Cases |
|------|-------------|--------------|-----------|
| `PRIVATE` | Local Ollama only | All PII scrubbed before any processing | Personal data, HR records, financial data |
| `INTERNAL` | Small cloud LLMs OK | PII scrubbed before cloud calls | Business data, internal reports |
| `PUBLIC` | Any model | Minimal scrubbing | Public information, research |

### PII Detection

`GovernanceManager.pii_scrubber.scrub()` detects and replaces:

| Data Type | Example | Replacement |
|-----------|---------|-------------|
| Taiwan National ID | `A123456789` | `[TAIWAN_ID]` |
| Credit card | `4111-1111-1111-1111` | `[CREDIT_CARD]` |
| Email address | `user@example.com` | `[EMAIL]` |
| Phone number | `+886-912-345-678` | `[PHONE]` |
| API key / credential | `sk-ant-...` | `[CREDENTIAL]` |

**Privacy escalation**: If `has_pii=True` is detected by the Perception Engine, the task's privacy_tier is automatically escalated to `PRIVATE`, regardless of what the caller set.

---

## Bias Auditing

### Two-Layer Detection

**Layer 1: Rule scan** (fast, zero token cost)

17 regex patterns covering:

| Category | Patterns |
|----------|----------|
| Stereotyping | Demographic generalizations, essentialist claims, gender stereotypes, demographic profiling |
| Exclusionary language | Gendered terms (mankind, stewardess), ableist language (retarded, psycho), othering language |

**Layer 2: LLM deep analysis** (`bias_auditor.md` system prompt)

Three fairness dimensions assessed:
- **Demographic Parity**: Are outcomes distributed equally across demographic groups?
- **Representational Harm**: Does the output reinforce harmful stereotypes?
- **Allocational Harm**: Does it disadvantage certain groups in resource allocation?

### Automatic Actions

| Risk Level | Automatic Action |
|------------|-----------------|
| `low` | Continue, log to episodic memory |
| `medium` | Continue, log warning |
| `high` | Continue, save JSON report to `data/bias_reports/`, flag for human review |
| `critical` | Continue, save report, send Slack alert (if configured), require human review |

### Running Bias Audits

```python
# In governance_swarm, bias_auditor_agent runs automatically
# Can also be called directly:
context.metadata = {
    "operation": "audit_output",
    "output_text": "The text to audit..."
}
result = await bias_auditor_agent.execute(context)
```

---

## Cost Management

### Budget Configuration

```yaml
# config/governance/budgets.yaml (auto-created, not in git)
daily_limit_usd: 5.00
monthly_limit_usd: 100.00
per_agent_daily:
  code_agent: 1.00
  ml_pipeline_agent: 2.00
  # others: 0.50 default
```

### Automatic Downgrade Chains

When daily spend exceeds 90% of budget:

```
claude-opus-4-6   → claude-sonnet-4-6  → claude-haiku-4-5-20251001
gpt-4o            → gpt-4o-mini
gemini-1.5-pro    → gemini-1.5-flash
(local models)    → no change (already free)
```

### Cost Visibility

```bash
# CLI cost report
python nexus.py costs

# Audit log query
sqlite3 data/audit.db "
  SELECT
    agent_id,
    SUM(cost_usd) as total_cost,
    COUNT(*) as task_count,
    AVG(quality_score) as avg_quality
  FROM audit_log
  WHERE date(timestamp) = date('now')
  GROUP BY agent_id
  ORDER BY total_cost DESC
"
```

---

## Audit Trail

### Structure

Every LLM call is logged to `data/audit.db` (SQLite, append-only):

```sql
CREATE TABLE audit_log (
    id              INTEGER PRIMARY KEY,
    timestamp       TEXT,
    event_type      TEXT,    -- llm_call | task_start | task_end | cost_alert | bias_flag
    task_id         TEXT,
    agent_id        TEXT,
    model           TEXT,
    privacy_tier    TEXT,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    cost_usd        REAL,
    quality_score   REAL,
    eu_ai_act_risk  TEXT,    -- null | minimal | limited | high_risk | prohibited
    pii_scrubbed    INTEGER  -- 0 | 1
);
```

### Compliance Reporting

The `governance_swarm` + `writer_agent` can generate compliance reports from audit data:

```
nexus task "Generate EU AI Act compliance status report for last 30 days"
→ cost_optimizer_agent: pull audit stats
→ bias_auditor_agent: pull bias report summary
→ writer_agent: synthesize into compliance report
```

---

## Data Lineage

### What is Tracked

Every task execution records a lineage entry:

```
task_abc  --CONSUMED-->  file_sales_q3.csv
task_abc  --CONSUMED-->  knowledge_doc_xyz
task_abc  --PRODUCED-->  report_board.md
report_board.md  --STORED_IN-->  knowledge_doc_abc
```

**Node types**: `data_source`, `agent_task`, `transformation`, `model`, `output`, `knowledge_doc`

**Edge types**: `PRODUCED`, `CONSUMED`, `DERIVED_FROM`, `TRAINED_ON`, `STORED_IN`, `TRIGGERED_BY`

### Queries

```python
tracker = get_tracker()

# Where did this output come from?
upstream = tracker.trace_back("report_board.md", depth=5)

# What depends on this data source?
downstream = tracker.get_downstream("customer_data.csv", depth=5)

# Full provenance graph
lineage = tracker.get_lineage("report_board.md", depth=3)

# Export as Graphviz DOT
dot = tracker.export_dot()
```

### Backend Selection

**Neo4j** (if `NEO4J_URI` is set and reachable): Full Cypher query support, enterprise-grade graph database.

**NetworkX + JSON** (fallback): Zero infrastructure, pure Python, persisted to `data/lineage_graph.json`. Same API as Neo4j backend.
