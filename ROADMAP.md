# NEXUS Roadmap

> Planned features, improvements, and long-term vision.

---

## Current Status: v0.1 — Foundation Complete

The core platform is built and validated:
- ✅ 5-layer architecture (Trigger → Perception → Orchestration → Execution → Memory)
- ✅ 19 Agents across 7 domains
- ✅ 10 Skills (I/O, ML, governance, data)
- ✅ 14 MCP Servers (local, cloud, DB, MLOps)
- ✅ EU AI Act compliance gate (Art. 5 + Annex III)
- ✅ Bias auditing (rule + LLM dual layer)
- ✅ Cost optimizer with auto-downgrade chains
- ✅ DSPy-style prompt optimization loop
- ✅ Data lineage (Neo4j + NetworkX)
- ✅ RAG Engine (BM25 + semantic hybrid)
- ✅ 10 Swarm configurations

---

## v0.2 — Quality & Observability (Next)

### Testing Infrastructure
- [ ] Integration test suite for all 19 agents
- [ ] EU AI Act classifier unit tests (all 18 patterns)
- [ ] Swarm end-to-end tests with mock LLMs
- [ ] Skill unit tests for all 10 implementations

### Observability
- [ ] Grafana dashboard templates (pre-built panels for cost, quality, latency)
- [ ] Prometheus metrics for agent execution time and token usage
- [ ] Structured logging with `structlog` (JSON output for log aggregation)
- [ ] Real-time quality trend charts in web dashboard

### Developer Experience
- [ ] `nexus agent create <name>` CLI scaffolding command
- [ ] `nexus skill create <name>` CLI scaffolding command
- [ ] `nexus swarm validate <yaml>` — YAML linting + agent existence check
- [ ] Hot-reload for swarm YAML changes (no restart required)

---

## v0.3 — Memory & Knowledge

### Advanced RAG
- [ ] Multi-modal memory (images, PDFs, audio transcripts)
- [ ] Cross-session memory consolidation improvements
- [ ] Knowledge graph enrichment (entity extraction → Neo4j)
- [ ] Memory privacy tagging (PRIVATE memories never surface to INTERNAL queries)

### Document Ingestion Pipeline
- [ ] Bulk document ingestion CLI (`nexus ingest ./docs/`)
- [ ] Web crawl ingestion (sitemap → ChromaDB)
- [ ] GitHub repository ingestion (code + docs → knowledge base)
- [ ] Scheduled re-ingestion for changed documents

### Retrieval Improvements
- [ ] Reciprocal Rank Fusion (RRF) for hybrid scoring
- [ ] Query expansion with local LLM
- [ ] Personalized retrieval (user-specific memory spaces)

---

## v0.4 — Agent Intelligence

### New Agents
- [ ] `vision_agent` — image understanding via multi-modal LLMs
- [ ] `sql_agent` — natural language → SQL → results + charts
- [ ] `document_agent` — large document analysis (chunked reasoning)
- [ ] `scheduler_agent` — manages and creates new scheduled tasks at runtime
- [ ] `alert_agent` — monitors conditions and sends proactive notifications

### Agent Capabilities
- [ ] Agent-to-agent direct communication (not only via orchestrator)
- [ ] Agent memory specialization (each agent has its own memory namespace)
- [ ] Confidence-based routing (low-confidence results trigger second opinion)
- [ ] Agent capability discovery (agents advertise what they can do)

---

## v0.5 — Governance & Compliance

### EU AI Act Compliance
- [ ] Automated compliance report generation (full Art. 9–17 checklist)
- [ ] EU AI Act obligation tracking integration with regulatory_tracker skill
- [ ] GPAI model documentation automation (Art. 53)
- [ ] Conformity assessment workflow (Art. 43)

### Extended Regulations
- [ ] CCPA/CPRA (California) compliance module
- [ ] Taiwan Personal Data Protection Act (PDPA) integration
- [ ] ISO 42001 (AI Management System) checklist

### Audit & Explainability
- [ ] Decision explanation API (why did NEXUS choose this agent/model?)
- [ ] Audit log visualization (timeline of task execution + costs)
- [ ] Exportable compliance reports (PDF via writer_agent)

---

## v1.0 — Production Hardening

### Reliability
- [ ] Multi-node deployment (distributed task queue with Celery/Redis)
- [ ] Agent crash recovery (resume from checkpoint without user intervention)
- [ ] Circuit breaker for external MCP servers (auto-disable failing servers)
- [ ] Graceful degradation (system works with partial infrastructure)

### Security
- [ ] Role-Based Access Control (RBAC) for multi-user deployments
- [ ] Task-level permission gates (users can only see their own tasks)
- [ ] API key rotation automation
- [ ] Penetration testing checklist

### Performance
- [ ] Response streaming (stream agent outputs as they complete)
- [ ] Batch task processing (queue 100 tasks, process in parallel)
- [ ] Model caching (KV cache sharing for Claude, OpenAI prompt caching)
- [ ] Embedding cache (avoid re-embedding the same text)

---

## Long-term Vision (v2.0+)

### Autonomous Operation
- NEXUS identifies and fixes its own performance issues without human intervention
- Self-healing: detects agent failures, spawns replacement agents
- Proactive monitoring: alerts on anomalies before users notice

### Multi-tenancy
- Full organizational deployment with team namespaces
- Budget allocation per team/department
- Shared knowledge base with access controls

### Plugin Ecosystem
- Public marketplace for community-built agents, skills, and MCP servers
- Agent versioning and rollback (like Docker images for agents)
- Cross-organization agent sharing (federated NEXUS network)

### Research Frontier
- Integration with OpenAI o3, Claude's extended thinking for complex reasoning
- Mixture-of-Agents routing (combine multiple cheap models instead of one expensive)
- Automated dataset generation for fine-tuning domain-specific models

---

## Contributing to the Roadmap

Have a feature request or use case not covered here? Open an issue on GitHub:
- Describe the use case (not just the feature)
- Describe what you've tried with the current system
- Label it `roadmap` or `feature-request`

Contributions are welcome for any item marked `[ ]` above. See [CONTRIBUTING.md](CONTRIBUTING.md).
