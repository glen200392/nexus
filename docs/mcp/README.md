# NEXUS MCP Servers Reference

14 MCP (Model Context Protocol) servers. Each runs as a subprocess communicating via JSON-RPC 2.0 over stdio.

---

## How MCP Works in NEXUS

```python
# Agent calling an MCP tool:
result = await self.mcp_client.call(
    server_name="arxiv_monitor",
    tool_name="run_monitor",
    arguments={"keywords": ["LLM inference"]}
)
```

**Protocol**: JSON-RPC 2.0 over stdin/stdout
**Discovery**: `tools/list` method returns all available tools
**Connection**: Managed by `mcp/client.py` — servers start on `init_system()` and restart on failure

---

## Local Tools

### `filesystem`
**File**: `mcp/servers/filesystem_server.py`
**External service**: Local file system

| Tool | Description |
|------|-------------|
| `read_file` | Read file content (text) |
| `write_file` | Write content to file |
| `list_directory` | List directory contents |
| `create_directory` | Create directory (with parents) |
| `delete_file` | Delete file (requires explicit confirmation flag) |
| `file_info` | Get size, modified time, permissions |

---

### `git`
**File**: `mcp/servers/git_server.py`
**External service**: Local git repository

| Tool | Description |
|------|-------------|
| `git_status` | Show working tree status |
| `git_diff` | Show diff (staged or unstaged) |
| `git_log` | Recent commit history |
| `git_add` | Stage files |
| `git_commit` | Create commit |
| `git_push` | Push to remote (requires confirmation) |

---

### `sqlite`
**File**: `mcp/servers/sqlite_server.py`
**External service**: SQLite database
**Config**: `SQLITE_DB_PATH`, `SQLITE_ALLOW_WRITE`

| Tool | Description |
|------|-------------|
| `list_tables` | Show all tables |
| `describe_table` | Column names, types, constraints |
| `execute_query` | Run SELECT queries |
| `execute_write` | Run INSERT/UPDATE/DELETE (requires SQLITE_ALLOW_WRITE=true) |
| `create_table` | Create new table |
| `insert_row` | Insert single row |

**Blocked operations**: `DROP`, `TRUNCATE`, `ATTACH`, `PRAGMA`

---

### `sequential_thinking`
**File**: `mcp/servers/sequential_thinking_server.py`
**External service**: None (internal structured reasoning)

Provides structured step-by-step reasoning chains for complex analysis. Useful for multi-step problem decomposition before execution.

---

### `fetch`
**File**: `mcp/servers/fetch_server.py`
**External service**: HTTP/HTTPS endpoints

| Tool | Description |
|------|-------------|
| `fetch_url` | Fetch URL, return HTML |
| `fetch_markdown` | Fetch URL, convert HTML→Markdown |
| `fetch_json` | Fetch URL, return parsed JSON |
| `fetch_links` | Extract all hyperlinks from page |

**HTML→Markdown conversion**: Regex-based (no external deps). Removes: `<script>`, `<nav>`, `<footer>`, ads. Preserves: headings, lists, tables, code blocks.

---

## Cloud API Servers

### `github`
**File**: `mcp/servers/github_server.py`
**External service**: GitHub API v3
**Config**: `GITHUB_TOKEN`

| Tool | Description |
|------|-------------|
| `create_issue` | Create new issue |
| `list_issues` | List open/closed issues |
| `get_pull_request` | Get PR details |
| `list_repositories` | List repos for user/org |
| `create_comment` | Add comment to issue or PR |
| `get_file_contents` | Read file from repository |

---

### `slack`
**File**: `mcp/servers/slack_server.py`
**External service**: Slack API
**Config**: `SLACK_BOT_TOKEN`, `SLACK_DEFAULT_CHANNEL`

| Tool | Description |
|------|-------------|
| `send_message` | Post message to channel |
| `list_channels` | List accessible channels |
| `get_thread` | Get thread replies |
| `upload_file` | Upload file to channel |

---

## Database Servers

### `postgres`
**File**: `mcp/servers/postgres_server.py`
**External service**: PostgreSQL
**Config**: `POSTGRES_URL`

| Tool | Description |
|------|-------------|
| `execute_query` | Run SQL SELECT |
| `execute_write` | Run INSERT/UPDATE/DELETE |
| `list_tables` | List all tables in database |
| `get_schema` | Column definitions for a table |
| `list_databases` | List available databases |

**Dependencies**: `psycopg2-binary`

---

### `chroma`
**File**: `mcp/servers/chroma_server.py`
**External service**: ChromaDB
**Config**: `CHROMA_HOST`, `CHROMA_PORT`

| Tool | Description |
|------|-------------|
| `upsert` | Add or update documents with embeddings |
| `search` | Semantic similarity search |
| `delete` | Remove documents by ID |
| `list_collections` | List all collections |
| `get_collection_info` | Document count, metadata |

---

## Monitoring

### `prometheus`
**File**: `mcp/servers/prometheus_server.py`
**External service**: Prometheus
**Config**: `PROMETHEUS_URL` (default: `http://localhost:9090`)

| Tool | Description |
|------|-------------|
| `query_metrics` | Instant PromQL query |
| `range_query` | Range query with start/end/step |
| `list_metrics` | List all available metric names |
| `get_alerts` | List active alerts |

---

## Browser Automation

### `playwright`
**File**: `mcp/servers/playwright_server.py`
**External service**: Local Chromium browser
**Config**: `PLAYWRIGHT_HEADLESS` (default: `true`)

| Tool | Description |
|------|-------------|
| `navigate` | Go to URL |
| `click` | Click element by CSS selector |
| `fill` | Fill input field |
| `screenshot` | Capture page screenshot (returns base64 PNG) |
| `extract_text` | Get text content from selector |
| `scroll` | Scroll page |
| `wait_for` | Wait for element to appear |

**Dependencies**: `playwright` (run `playwright install chromium`)

---

## AI Research

### `arxiv_monitor`
**File**: `mcp/servers/arxiv_monitor_server.py`
**External service**: arXiv.org Atom API
**Config**: None required (public API)

| Tool | Description |
|------|-------------|
| `search_papers` | Search arXiv by query string |
| `get_paper` | Get full details for arXiv ID |
| `get_recent` | Get most recent papers in category |
| `add_keyword` | Add keyword to persistent watchlist |
| `list_keywords` | Show active watchlist |
| `remove_keyword` | Remove keyword from watchlist |
| `run_monitor` | Check watchlist for NEW papers (incremental, deduped) |

**Categories monitored**: `cs.AI`, `cs.LG`, `cs.CL`

**Incremental deduplication**: Seen paper IDs stored in `data/arxiv_monitor.json`. `run_monitor` returns only papers not previously seen (bounded at 5000 IDs).

---

### `rss_aggregator`
**File**: `mcp/servers/rss_aggregator_server.py`
**External service**: RSS/Atom feeds

**Pre-configured feeds** (seeded on first run):
- Anthropic Blog
- OpenAI Blog
- HuggingFace Blog
- Google DeepMind Blog
- NIST AI Updates
- MIT Technology Review (AI)
- Ars Technica (AI section)
- LangChain Blog

| Tool | Description |
|------|-------------|
| `add_feed` | Add RSS/Atom feed URL |
| `list_feeds` | Show all configured feeds |
| `remove_feed` | Remove a feed |
| `fetch_feed` | Fetch items from one feed |
| `fetch_all` | Fetch items from all feeds |
| `search_items` | Full-text search across cached items |
| `get_stats` | Feed count, item count, cache age |

**Cache**: 1-hour TTL in `data/rss_cache.json`. Both RSS 2.0 and Atom 1.0 supported.

---

## MLOps

### `mlflow`
**File**: `mcp/servers/mlflow_server.py`
**External service**: MLflow Tracking Server
**Config**: `MLFLOW_TRACKING_URI`, optional `MLFLOW_USERNAME` / `MLFLOW_PASSWORD`

| Tool | Description |
|------|-------------|
| `mlflow_list_experiments` | List all experiments |
| `mlflow_create_experiment` | Create new experiment |
| `mlflow_create_run` | Start new run in experiment |
| `mlflow_log_params` | Log parameters to run (batch) |
| `mlflow_log_metrics` | Log metrics to run (batch) |
| `mlflow_end_run` | Finish run (status: FINISHED/FAILED) |
| `mlflow_get_run` | Get run details and metrics |
| `mlflow_list_runs` | List runs in experiment |
| `mlflow_get_best_run` | Get run with best metric value |
| `mlflow_compare_runs` | Side-by-side comparison of multiple runs |

**Zero extra dependencies**: Uses stdlib `urllib` to call MLflow REST API v2.
