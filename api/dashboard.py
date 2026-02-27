"""
NEXUS Web Dashboard — Local monitoring UI
Serves a single-page dashboard showing:
  - Active tasks in real-time (SSE)
  - Task history with quality scores
  - Cost breakdown by model
  - Agent quality trends
  - Submit tasks via HTTP

Run:
    uvicorn nexus.api.dashboard:app --host 0.0.0.0 --port 7800

Access: http://localhost:7800
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="NEXUS Dashboard", version="1.0.0")

# Injected at startup by nexus.py
_orchestrator = None
_governance   = None
_cli_trigger  = None

def inject(orchestrator, governance, cli_trigger):
    global _orchestrator, _governance, _cli_trigger
    _orchestrator = orchestrator
    _governance   = governance
    _cli_trigger  = cli_trigger


# ── A2A AgentCard endpoint ────────────────────────────────────────────────────

@app.get("/.well-known/agent.json")
async def agent_card():
    """
    Google A2A spec: AgentCard endpoint.
    External A2A-compatible agents discover NEXUS capabilities here.
    """
    from nexus.core.agents.a2a_agent import build_nexus_agent_card
    from fastapi.responses import JSONResponse
    return JSONResponse(build_nexus_agent_card().to_dict())


# ── API endpoints ─────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    prompt: str
    priority: str = "NORMAL"
    domain: Optional[str] = None


@app.post("/api/task")
async def submit_task(req: TaskRequest):
    if _cli_trigger is None:
        return {"error": "System not initialized"}
    event_id = await _cli_trigger.submit(
        prompt=req.prompt,
        domain=req.domain,
    )
    return {"event_id": event_id, "status": "queued"}


@app.get("/api/status")
async def get_status():
    if _orchestrator is None:
        return {"error": "Not initialized"}
    return _orchestrator.status()


@app.get("/api/history")
async def get_history(limit: int = 20):
    if _orchestrator is None:
        return []
    tasks = _orchestrator._completed_tasks[-limit:]
    return [
        {
            "task_id":       t.task_id[:8],
            "domain":        t.domain,
            "pattern":       t.workflow_pattern.value,
            "status":        t.status.value,
            "quality_score": round(t.quality_score, 2),
            "cost_usd":      round(t.total_cost_usd, 4),
            "duration_s":    round((t.completed_at or time.time()) - t.started_at, 1)
                             if t.started_at else 0,
            "started_at":    t.started_at,
        }
        for t in reversed(tasks)
    ]


@app.get("/api/costs")
async def get_costs():
    if _governance is None:
        return {}
    since_24h = time.time() - 86400
    return _governance.audit.get_cost_summary(since_timestamp=since_24h)


@app.get("/api/quality")
async def get_quality():
    if _governance is None:
        return {}
    return _governance.optimizer.report()


@app.get("/api/active-tasks")
async def get_active():
    if _orchestrator is None:
        return []
    return [
        {
            "task_id":  tid[:8],
            "domain":   t.domain,
            "status":   t.status.value,
            "elapsed_s": round(time.time() - t.started_at, 1) if t.started_at else 0,
        }
        for tid, t in _orchestrator._active_tasks.items()
    ]


@app.get("/api/stream")
async def stream_events():
    """Server-Sent Events for real-time task updates."""
    async def event_generator():
        while True:
            active = []
            if _orchestrator:
                active = [
                    {"task_id": tid[:8], "domain": t.domain, "status": t.status.value}
                    for tid, t in _orchestrator._active_tasks.items()
                ]
            costs = {}
            if _governance:
                costs = _governance.audit.get_cost_summary()

            data = json.dumps({"active_tasks": active, "costs": costs})
            yield f"data: {data}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Single-page Dashboard HTML ────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEXUS Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; min-height: 100vh; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.2rem; color: #58a6ff; letter-spacing: 0.1em; }
  .badge { background: #238636; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
  .card h2 { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px; }
  .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }
  .stat:last-child { border-bottom: none; }
  .stat .label { color: #8b949e; font-size: 0.85rem; }
  .stat .value { color: #c9d1d9; font-weight: 600; }
  .value.green  { color: #3fb950; }
  .value.yellow { color: #d29922; }
  .value.red    { color: #f85149; }
  .value.blue   { color: #58a6ff; }
  .task-list { max-height: 300px; overflow-y: auto; }
  .task-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 0.85rem; }
  .task-row:last-child { border-bottom: none; }
  .status-badge { padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
  .status-completed { background: #1a4731; color: #3fb950; }
  .status-running   { background: #1c2a4a; color: #58a6ff; }
  .status-failed    { background: #3d1f1f; color: #f85149; }
  .status-pending   { background: #2d2a1f; color: #d29922; }
  .quality-bar { height: 6px; background: #21262d; border-radius: 3px; margin-top: 4px; }
  .quality-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .submit-form { display: flex; gap: 8px; padding: 24px; }
  .submit-form input { flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; color: #c9d1d9; font-family: inherit; font-size: 0.9rem; outline: none; }
  .submit-form input:focus { border-color: #58a6ff; }
  .submit-form button { background: #238636; color: #fff; border: none; border-radius: 6px; padding: 10px 20px; cursor: pointer; font-family: inherit; }
  .submit-form button:hover { background: #2ea043; }
  #live-dot { width: 8px; height: 8px; background: #3fb950; border-radius: 50%; animation: pulse 2s infinite; display: inline-block; margin-right: 8px; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .model-tag { background: #1c2a4a; color: #58a6ff; padding: 1px 6px; border-radius: 3px; font-size: 0.7rem; margin-right: 4px; }
</style>
</head>
<body>
<header>
  <div id="live-dot"></div>
  <h1>NEXUS Agent Dashboard</h1>
  <span class="badge" id="active-count">0 active</span>
</header>

<div class="submit-form">
  <input id="task-input" type="text" placeholder="Submit a task... (e.g. 研究最新的 LLM 論文)" />
  <button onclick="submitTask()">Submit</button>
</div>

<div class="grid">
  <!-- Active Tasks -->
  <div class="card" style="grid-column: span 2;">
    <h2>Active Tasks</h2>
    <div class="task-list" id="active-tasks"><p style="color:#8b949e;font-size:.85rem;">No active tasks</p></div>
  </div>

  <!-- System Status -->
  <div class="card">
    <h2>System</h2>
    <div id="system-stats">
      <div class="stat"><span class="label">Active Tasks</span><span class="value blue" id="stat-active">—</span></div>
      <div class="stat"><span class="label">Completed Today</span><span class="value green" id="stat-completed">—</span></div>
      <div class="stat"><span class="label">Registered Swarms</span><span class="value" id="stat-swarms">—</span></div>
      <div class="stat"><span class="label">LLM Slots</span><span class="value" id="stat-llm">—</span></div>
    </div>
  </div>

  <!-- 24h Cost -->
  <div class="card">
    <h2>Cost (Last 24h)</h2>
    <div class="task-list" id="cost-list"><p style="color:#8b949e;font-size:.85rem;">Loading...</p></div>
  </div>

  <!-- Quality Trends -->
  <div class="card">
    <h2>Agent Quality</h2>
    <div id="quality-list"><p style="color:#8b949e;font-size:.85rem;">Loading...</p></div>
  </div>

  <!-- Task History -->
  <div class="card" style="grid-column: span 2;">
    <h2>Recent Tasks</h2>
    <div class="task-list" id="history-list"><p style="color:#8b949e;font-size:.85rem;">Loading...</p></div>
  </div>
</div>

<script>
const api = async (path) => { try { const r = await fetch(path); return r.json(); } catch { return {}; } };

async function submitTask() {
  const input = document.getElementById('task-input');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  await fetch('/api/task', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt})
  });
}

document.getElementById('task-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') submitTask();
});

function qualityColor(score) {
  if (score >= 0.8) return '#3fb950';
  if (score >= 0.6) return '#d29922';
  return '#f85149';
}

function statusClass(s) {
  return 'status-' + s;
}

async function refreshHistory() {
  const history = await api('/api/history?limit=15');
  const el = document.getElementById('history-list');
  if (!history.length) { el.innerHTML = '<p style="color:#8b949e;font-size:.85rem;">No tasks yet</p>'; return; }
  el.innerHTML = history.map(t => `
    <div class="task-row">
      <span style="color:#8b949e;font-size:.8rem;">#${t.task_id}</span>
      <span style="flex:1;padding:0 12px;font-size:.85rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${t.domain} · ${t.pattern}</span>
      <span style="color:${qualityColor(t.quality_score)};margin-right:12px;">${(t.quality_score*100).toFixed(0)}%</span>
      <span style="color:#8b949e;font-size:.8rem;margin-right:8px;">${t.duration_s}s</span>
      <span style="color:#8b949e;font-size:.8rem;margin-right:8px;">$${t.cost_usd}</span>
      <span class="status-badge ${statusClass(t.status)}">${t.status}</span>
    </div>
  `).join('');
}

async function refreshQuality() {
  const quality = await api('/api/quality');
  const el = document.getElementById('quality-list');
  const entries = Object.entries(quality);
  if (!entries.length) { el.innerHTML = '<p style="color:#8b949e;font-size:.85rem;">No data yet</p>'; return; }
  el.innerHTML = entries.map(([agent, data]) => `
    <div class="stat">
      <span class="label">${agent}</span>
      <span>
        <span class="value" style="color:${qualityColor(data.avg_score)}">${(data.avg_score*100).toFixed(0)}% ${data.trend}</span>
        <span style="color:#8b949e;font-size:.75rem;margin-left:8px;">(${data.task_count} tasks)</span>
      </span>
    </div>
    <div class="quality-bar">
      <div class="quality-fill" style="width:${data.avg_score*100}%;background:${qualityColor(data.avg_score)};"></div>
    </div>
  `).join('');
}

async function refreshCosts() {
  const costs = await api('/api/costs');
  const rows = costs.rows || [];
  const el = document.getElementById('cost-list');
  if (!rows.length) { el.innerHTML = '<p style="color:#8b949e;font-size:.85rem;">No cost data</p>'; return; }
  const total = rows.reduce((s, r) => s + r.total_cost, 0);
  el.innerHTML = `
    <div class="stat" style="margin-bottom:8px;">
      <span class="label">Total 24h</span>
      <span class="value yellow">$${total.toFixed(4)}</span>
    </div>
    ${rows.slice(0, 6).map(r => `
      <div class="stat">
        <span><span class="model-tag">${(r.model_used||'local').split('/').pop()}</span><span style="color:#8b949e;font-size:.75rem;">${r.agent_id}</span></span>
        <span class="value">$${r.total_cost.toFixed(4)}</span>
      </div>
    `).join('')}
  `;
}

// Real-time updates via SSE
const evtSource = new EventSource('/api/stream');
evtSource.onmessage = (e) => {
  const data = JSON.parse(e.data);
  const active = data.active_tasks || [];
  document.getElementById('active-count').textContent = active.length + ' active';
  document.getElementById('stat-active').textContent = active.length;

  const el = document.getElementById('active-tasks');
  if (!active.length) { el.innerHTML = '<p style="color:#8b949e;font-size:.85rem;">No active tasks</p>'; return; }
  el.innerHTML = active.map(t => `
    <div class="task-row">
      <span style="color:#8b949e;font-size:.8rem;">#${t.task_id}</span>
      <span style="flex:1;padding:0 12px;">${t.domain}</span>
      <span style="color:#8b949e;font-size:.8rem;margin-right:8px;">${t.elapsed_s}s</span>
      <span class="status-badge ${statusClass(t.status)}">${t.status}</span>
    </div>
  `).join('');
};

// Poll slower endpoints every 5s
async function refreshAll() {
  const status = await api('/api/status');
  if (status.resource_pool) {
    const rp = status.resource_pool;
    document.getElementById('stat-llm').textContent = `${rp.active_llm_calls} / ${5}`;
  }
  if (status.completed_tasks !== undefined) {
    document.getElementById('stat-completed').textContent = status.completed_tasks;
  }
  if (status.registered_swarms) {
    document.getElementById('stat-swarms').textContent = status.registered_swarms.join(', ') || '—';
  }
  await refreshHistory();
  await refreshQuality();
  await refreshCosts();
}

refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)
