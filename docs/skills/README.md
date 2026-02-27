# NEXUS Skills Reference

10 reusable skill modules. Any agent can call any skill via `self.skill_registry.get(name).run(**kwargs)`.

Skills are auto-discovered from `skills/implementations/<name>/__init__.py`. No registration required.

---

## Skill Contract

```python
SKILL_META = SkillMeta(
    name="skill_name",
    description="Short description",
    version="1.0.0",
    domains=["research", "engineering"],
    triggers=["keyword", "trigger phrase"],  # used for automatic skill selection
)

class Skill(BaseSkill):
    async def run(self, operation: str = "default", **kwargs) -> dict:
        ...
```

---

## I/O Skills

### `excel_designer`
Read, create, and style Excel workbooks.

| Operation | Description |
|-----------|-------------|
| `read` | Read workbook: returns sheets, headers, data as list of dicts |
| `create` | Create new workbook with data and optional styling |
| `write` | Write data to existing workbook/sheet |
| `style` | Apply font, fill, border, alignment to cell ranges |
| `charts` | Insert bar/line/pie chart from data range |

**Dependencies**: `openpyxl`

---

### `pdf_reader`
Extract text, tables, and metadata from PDF files.

| Operation | Description |
|-----------|-------------|
| `extract_text` | Full text extraction with page markers |
| `extract_tables` | Structured table extraction as list of dicts |
| `metadata` | Author, title, creation date, page count |
| `search` | Find pages containing a keyword |

**Dependencies**: `pdfplumber` (primary), `pypdf` (fallback)

---

### `pptx_builder`
Create and modify PowerPoint presentations.

| Operation | Description |
|-----------|-------------|
| `create` | New presentation with title slide |
| `add_slide` | Append slide with layout (title/content/blank) |
| `add_content` | Add text, bullets, or image to existing slide |
| `save` | Save to file path |

**Dependencies**: `python-pptx`

---

### `shell_executor`
Execute shell commands with safety filtering.

| Operation | Description |
|-----------|-------------|
| `execute_command` | Run command, return stdout/stderr/exit_code |
| `parse_output` | Parse structured output (CSV, JSON, key=value) |

**Note**: Used by `shell_agent` internally. Use `shell_agent` for complex orchestration; use this skill for simple one-off command execution from other agents.

---

## ML & Data Skills

### `notebook_executor`
Execute parameterized Jupyter notebooks.

| Operation | Description |
|-----------|-------------|
| `execute` | Run notebook, optionally inject parameters |
| `list` | List notebooks in a directory |
| `get_outputs` | Extract text + images + JSON metrics from executed notebook |
| `validate` | Check notebook is valid (can be executed) |

**Primary**: `papermill` (parameter injection)
**Fallback**: `nbformat` + `ExecutePreprocessor`

**Parameter injection example**:
```python
result = await skill.run(
    operation="execute",
    notebook_path="analysis/churn_analysis.ipynb",
    parameters={"data_path": "data/q3.csv", "n_samples": 1000},
    output_path="outputs/churn_q3_executed.ipynb",
)
```

**Dependencies**: `nbformat`, `nbconvert`, `papermill`

---

### `synthetic_data_generator`
Generate statistically faithful synthetic datasets.

| Operation | Description |
|-----------|-------------|
| `generate` | Produce synthetic dataset from source file or records |
| `profile` | Statistical profile of a dataset |
| `validate` | Compare synthetic vs real distributions (KS/chi-squared) |
| `generate_text` | Generate fake text records (names, emails, descriptions) |
| `save_dataset` | Persist generated dataset to disk |

**Generation strategy** (in order of preference):
1. **Gaussian copula** (scipy) — preserves inter-column correlation structure
2. **Column-by-column** — per-column distribution fitting (numpy only)
3. **Bootstrap resampling** — fallback if data is too small

**`generate_text` supported fields**: `name`, `email`, `phone`, `id`, `date`, `description`, `text`, `comment`

**Dependencies**: `scipy` (optional, for copula), `numpy`

---

### `rag_evaluator`
Evaluate RAG pipeline quality with standard IR metrics.

| Operation | Description |
|-----------|-------------|
| `add_ground_truth` | Add query + expected document IDs to evaluation set |
| `evaluate_retrieval` | Compute Precision@k, Recall@k, NDCG@k, MRR for a query |
| `evaluate_generation` | Compute ROUGE-L for generated answers vs reference |
| `run_benchmark` | Full benchmark across entire ground truth set |
| `compare_configs` | Compare two retrieval configs side-by-side |
| `get_report` | Retrieve a saved benchmark report |
| `list_benchmarks` | List all benchmark reports |

**Zero ML dependencies** — all metrics implemented in pure Python:
- NDCG: `sum(rel_i / log2(i+2) for i in range(k))` normalized by ideal DCG
- ROUGE-L: LCS-based (longest common subsequence)
- MRR: `1 / rank_of_first_relevant_result`

**Ground truth storage**: JSONL at `data/rag_evals/ground_truth.jsonl`

---

## Governance & Compliance Skills

### `prompt_versioning`
Semantic version control for system prompts with quality tracking.

| Operation | Description |
|-----------|-------------|
| `save` | Save prompt as new version (semver: major.minor.patch) |
| `load` | Load a specific version |
| `list` | List all versions for an agent (with quality stats) |
| `compare` | Unified diff between two versions |
| `rollback` | Activate a previous version |
| `set_active` | Promote a specific version to active |
| `get_active` | Return currently active prompt content |
| `record_quality` | Update quality stats (moving average) after task execution |

**Version storage**: YAML files in `config/prompts/versions/<agent_id>/`

**Quality tracking**: Moving average updated after each task:
```python
new_avg = (old_avg * n + new_score) / (n + 1)
```

---

### `model_card_generator`
Generate NIST AI RMF + EU AI Act-aligned model documentation.

| Operation | Description |
|-----------|-------------|
| `generate` | Create new model card with blank template |
| `get_card` | Load existing model card |
| `update` | Update specific fields |
| `validate` | Check completeness (required fields coverage %) |
| `export_md` | Export model card as readable Markdown |
| `list_cards` | List all model cards |
| `add_eval` | Append evaluation result to model card history |

**Storage**: YAML files in `config/model_cards/`

**Required fields** (for 100% completeness score):
- `model_details.name`, `model_details.version`, `model_details.type`
- `intended_use.use_cases`, `intended_use.out_of_scope`
- `eu_ai_act.risk_level`, `eu_ai_act.applicable_articles`
- `nist_ai_rmf.govern`, `nist_ai_rmf.map`
- `evaluation.metrics`, `limitations.known_limitations`

---

### `regulatory_tracker`
Track AI/data regulation compliance deadlines and obligations.

**Pre-seeded regulations**:
- EU AI Act (2024/1689) — 8 obligations, enforcement 2026-08-01
- EU GDPR (2016/679) — 5 obligations
- NIST AI RMF 1.0 — 4 obligations (voluntary)
- EU Data Act (2023/2854) — 2 obligations, enforcement 2025-09-12
- Canada AIDA (Bill C-27) — 2 obligations (estimated 2026-06)
- Singapore PDPA + Model AI Governance — 2 obligations

| Operation | Description |
|-----------|-------------|
| `list` | All regulations (filter by status or tag) |
| `get` | Detail for one regulation |
| `add` | Add new regulation to registry |
| `update` | Update fields of existing regulation |
| `check_upcoming` | Regulations with deadlines within N days |
| `add_obligation` | Add compliance obligation/checklist item |
| `check_compliance` | Obligation completion summary for a regulation |
| `search` | Full-text search across regulations |
| `export_csv` | Export full registry with obligations as CSV |

**Compliance status levels**: `COMPLIANT` (100%) → `ON_TRACK` (≥75%) → `AT_RISK` (≥50%) → `NON_COMPLIANT`

**Urgency levels** (from `check_upcoming`): `CRITICAL` (≤30 days) → `HIGH` (≤90 days) → `MEDIUM`

**Storage**: YAML registry at `config/regulatory/registry.yaml` (auto-seeded on first run)
