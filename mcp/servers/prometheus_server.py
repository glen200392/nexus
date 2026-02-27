"""
NEXUS MCP Server — Prometheus Metrics
Query Prometheus via its HTTP API (PromQL) over stdlib urllib.

Tools:
  - prom_query          Instant PromQL query (current value)
  - prom_query_range    Range PromQL query (time-series)
  - prom_list_metrics   Enumerate all metric names
  - prom_get_labels     Get label names for a metric
  - prom_get_alerts     List currently firing alerts
  - prom_get_targets    List scrape targets and their health
  - prom_get_rules      List recording and alerting rules

Environment:
  PROMETHEUS_URL   (default: http://localhost:9090)
  PROMETHEUS_TOKEN (optional, for bearer auth)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

PROMETHEUS_URL   = os.environ.get("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
PROMETHEUS_TOKEN = os.environ.get("PROMETHEUS_TOKEN", "")


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _get(path: str, params: Optional[dict] = None) -> dict:
    qs  = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"{PROMETHEUS_URL}{path}{qs}"
    headers = {}
    if PROMETHEUS_TOKEN:
        headers["Authorization"] = f"Bearer {PROMETHEUS_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            return body
    except urllib.error.HTTPError as exc:
        return {"status": "error", "errorType": "http", "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "errorType": "connection", "error": str(exc)}


def _data(resp: dict) -> tuple[bool, object]:
    """Unpack Prometheus API response. Returns (ok, data_or_error)."""
    if resp.get("status") == "success":
        return True, resp.get("data", {})
    return False, resp.get("error", "Unknown error")


# ── Tool Implementations ───────────────────────────────────────────────────────

def prom_query(query: str, time: Optional[str] = None) -> dict:
    """
    Instant PromQL query. Returns the current value(s) for the expression.
    time: RFC3339 or Unix timestamp (default: now)
    """
    params: dict = {"query": query}
    if time:
        params["time"] = time
    ok, data = _data(_get("/api/v1/query", params))
    if not ok:
        return {"error": data}
    result_type = data.get("resultType", "")
    results     = data.get("result", [])
    formatted   = []
    for r in results[:50]:   # cap at 50 series
        labels = r.get("metric", {})
        value  = r.get("value", [None, None])
        formatted.append({
            "labels": labels,
            "value":  value[1] if len(value) > 1 else None,
            "ts":     value[0] if value else None,
        })
    return {
        "query":       query,
        "result_type": result_type,
        "results":     formatted,
        "count":       len(formatted),
    }


def prom_query_range(
    query: str,
    start: str,
    end: str,
    step: str = "1m",
    max_points: int = 200,
) -> dict:
    """
    Range PromQL query. Returns time-series data.
    start/end: RFC3339 or Unix timestamp
    step: duration string (e.g., "30s", "5m", "1h")
    """
    ok, data = _data(_get("/api/v1/query_range", {
        "query": query,
        "start": start,
        "end":   end,
        "step":  step,
    }))
    if not ok:
        return {"error": data}
    results = data.get("result", [])
    series  = []
    for r in results[:20]:   # cap at 20 series
        labels = r.get("metric", {})
        values = r.get("values", [])[-max_points:]
        series.append({
            "labels": labels,
            "points": [[v[0], v[1]] for v in values],
            "count":  len(values),
        })
    return {
        "query":  query,
        "start":  start,
        "end":    end,
        "step":   step,
        "series": series,
    }


def prom_list_metrics(match: Optional[str] = None, limit: int = 500) -> dict:
    """List all metric names, optionally filtered by a substring."""
    ok, data = _data(_get("/api/v1/label/__name__/values"))
    if not ok:
        return {"error": data}
    names = data if isinstance(data, list) else []
    if match:
        names = [n for n in names if match.lower() in n.lower()]
    names = names[:limit]
    return {"metrics": names, "count": len(names)}


def prom_get_labels(metric: str) -> dict:
    """Get label names and sample values for a metric."""
    ok, data = _data(_get("/api/v1/series", {"match[]": metric}))
    if not ok:
        return {"error": data}
    label_map: dict[str, set] = {}
    for series in (data if isinstance(data, list) else []):
        for k, v in series.items():
            if k == "__name__":
                continue
            label_map.setdefault(k, set()).add(v)
    # Convert sets to sorted lists (cap value examples at 10)
    result = {k: sorted(v)[:10] for k, v in label_map.items()}
    return {"metric": metric, "labels": result}


def prom_get_alerts() -> dict:
    """List currently firing alerts."""
    ok, data = _data(_get("/api/v1/alerts"))
    if not ok:
        return {"error": data}
    alerts = data.get("alerts", [])
    firing = [
        {
            "name":        a.get("labels", {}).get("alertname", ""),
            "state":       a.get("state", ""),
            "severity":    a.get("labels", {}).get("severity", ""),
            "summary":     a.get("annotations", {}).get("summary", ""),
            "labels":      a.get("labels", {}),
            "active_at":   a.get("activeAt", ""),
            "value":       a.get("value", ""),
        }
        for a in alerts
    ]
    firing.sort(key=lambda x: x["state"] == "firing", reverse=True)
    return {"alerts": firing, "total": len(firing), "firing": sum(1 for a in firing if a["state"] == "firing")}


def prom_get_targets() -> dict:
    """List all scrape targets with their health status."""
    ok, data = _data(_get("/api/v1/targets"))
    if not ok:
        return {"error": data}
    active = data.get("activeTargets", [])
    targets = [
        {
            "job":          t.get("labels", {}).get("job", ""),
            "instance":     t.get("labels", {}).get("instance", ""),
            "health":       t.get("health", ""),
            "last_scrape":  t.get("lastScrape", ""),
            "error":        t.get("lastError", ""),
            "scrape_url":   t.get("scrapeUrl", ""),
        }
        for t in active
    ]
    return {
        "targets": targets,
        "total":   len(targets),
        "healthy": sum(1 for t in targets if t["health"] == "up"),
    }


def prom_get_rules() -> dict:
    """List all recording and alerting rules with their state."""
    ok, data = _data(_get("/api/v1/rules"))
    if not ok:
        return {"error": data}
    groups  = data.get("groups", [])
    summary = []
    for g in groups:
        rules = []
        for r in g.get("rules", []):
            rules.append({
                "name":    r.get("name", ""),
                "type":    r.get("type", ""),
                "state":   r.get("state", ""),
                "query":   r.get("query", r.get("expr", "")),
                "health":  r.get("health", ""),
            })
        summary.append({"group": g.get("name", ""), "rules": rules})
    return {"groups": summary, "total_groups": len(summary)}


# ── MCP Schema ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "prom_query",
        "description": "Instant PromQL query — returns current metric value(s).",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "PromQL expression"},
                "time":  {"type": "string", "description": "RFC3339 or Unix timestamp (default: now)"},
            },
        },
    },
    {
        "name": "prom_query_range",
        "description": "Range PromQL query — returns time-series data.",
        "inputSchema": {
            "type": "object",
            "required": ["query", "start", "end"],
            "properties": {
                "query":      {"type": "string"},
                "start":      {"type": "string", "description": "RFC3339 or Unix timestamp"},
                "end":        {"type": "string"},
                "step":       {"type": "string", "default": "1m", "description": "Resolution: 30s, 5m, 1h…"},
                "max_points": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "prom_list_metrics",
        "description": "List all metric names, optionally filtered by substring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "match": {"type": "string", "description": "Substring filter"},
                "limit": {"type": "integer", "default": 500},
            },
        },
    },
    {
        "name": "prom_get_labels",
        "description": "Get label names and sample values for a metric.",
        "inputSchema": {
            "type": "object",
            "required": ["metric"],
            "properties": {"metric": {"type": "string"}},
        },
    },
    {
        "name": "prom_get_alerts",
        "description": "List all currently firing or pending alerts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "prom_get_targets",
        "description": "List all Prometheus scrape targets and their health.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "prom_get_rules",
        "description": "List all recording and alerting rules with their state.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_MAP = {
    "prom_query":        prom_query,
    "prom_query_range":  prom_query_range,
    "prom_list_metrics": prom_list_metrics,
    "prom_get_labels":   prom_get_labels,
    "prom_get_alerts":   prom_get_alerts,
    "prom_get_targets":  prom_get_targets,
    "prom_get_rules":    prom_get_rules,
}


# ── MCP Stdio Loop ─────────────────────────────────────────────────────────────

def _respond(req_id, result: dict) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
    sys.stdout.flush()

def _error(req_id, code: int, message: str) -> None:
    sys.stdout.write(json.dumps({
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": code, "message": message},
    }) + "\n")
    sys.stdout.flush()


def main():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue
        req_id = req.get("id")
        method = req.get("method", "")

        if method == "initialize":
            _respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities":    {"tools": {}},
                "serverInfo":      {"name": "prometheus", "version": "1.0"},
            })
        elif method == "tools/list":
            _respond(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            params    = req.get("params", {})
            tool_name = params.get("name", "")
            args      = params.get("arguments", {})
            fn        = TOOL_MAP.get(tool_name)
            if fn is None:
                _error(req_id, -32601, f"Unknown tool: {tool_name}")
                continue
            try:
                result = fn(**{k: v for k, v in args.items()})
                _respond(req_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})
            except Exception as exc:
                _respond(req_id, {"content": [{"type": "text", "text": json.dumps({"error": str(exc)})}]})
        elif method == "notifications/initialized":
            pass
        else:
            _error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
