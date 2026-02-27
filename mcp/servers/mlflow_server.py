"""
NEXUS MCP Server — MLflow Experiment Tracking
Query and log to MLflow Tracking Server via its REST API.
Uses Python stdlib urllib only.

Tools:
  mlflow_create_experiment  Create a new experiment
  mlflow_list_experiments   List all experiments
  mlflow_start_run          Start a new run in an experiment
  mlflow_log_params         Log hyperparameters to a run
  mlflow_log_metrics        Log metrics (with optional step)
  mlflow_end_run            Mark a run as finished
  mlflow_get_run            Get run details and metrics
  mlflow_search_runs        Search runs by experiment + filter expression
  mlflow_get_best_run       Find the run with best metric value
  mlflow_compare_runs       Side-by-side metric comparison for N runs

Environment:
  MLFLOW_TRACKING_URI  (default: http://localhost:5000)
  MLFLOW_USERNAME      (default: empty — for basic auth)
  MLFLOW_PASSWORD      (default: empty)
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip("/")
MLFLOW_USER     = os.environ.get("MLFLOW_USERNAME", "")
MLFLOW_PASSWORD = os.environ.get("MLFLOW_PASSWORD", "")


# ── HTTP Helpers ───────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if MLFLOW_USER:
        creds = base64.b64encode(f"{MLFLOW_USER}:{MLFLOW_PASSWORD}".encode()).decode()
        h["Authorization"] = f"Basic {creds}"
    return h


def _get(path: str, params: Optional[dict] = None) -> dict:
    qs  = ("?" + urllib.parse.urlencode(params)) if params else ""
    req = urllib.request.Request(f"{MLFLOW_URI}{path}{qs}", headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode()[:200]}"}
    except Exception as exc:
        return {"error": str(exc)}


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(f"{MLFLOW_URI}{path}", data=data, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode()[:200]}"}
    except Exception as exc:
        return {"error": str(exc)}


def _patch(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(f"{MLFLOW_URI}{path}", data=data,
                                   headers=_headers(), method="PATCH")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc)}


# ── Tool Implementations ───────────────────────────────────────────────────────

def mlflow_create_experiment(
    name: str,
    tags: Optional[dict] = None,
    artifact_location: Optional[str] = None,
) -> dict:
    """Create a new MLflow experiment."""
    body: dict = {"name": name}
    if tags:
        body["tags"] = [{"key": k, "value": v} for k, v in tags.items()]
    if artifact_location:
        body["artifact_location"] = artifact_location
    result = _post("/api/2.0/mlflow/experiments/create", body)
    if "error" in result:
        return result
    return {"experiment_id": result.get("experiment_id"), "name": name}


def mlflow_list_experiments(view_type: str = "ACTIVE_ONLY", max_results: int = 100) -> dict:
    """List MLflow experiments."""
    result = _get("/api/2.0/mlflow/experiments/search",
                  {"view_type": view_type, "max_results": max_results})
    if "error" in result:
        return result
    exps = [
        {
            "experiment_id": e.get("experiment_id"),
            "name":          e.get("name"),
            "lifecycle_stage": e.get("lifecycle_stage"),
            "artifact_location": e.get("artifact_location"),
        }
        for e in result.get("experiments", [])
    ]
    return {"experiments": exps, "count": len(exps)}


def mlflow_start_run(
    experiment_id: str,
    run_name: Optional[str] = None,
    tags: Optional[dict] = None,
) -> dict:
    """Start a new MLflow run."""
    body: dict = {
        "experiment_id": experiment_id,
        "start_time":    int(time.time() * 1000),
    }
    if run_name:
        body.setdefault("tags", []).append({"key": "mlflow.runName", "value": run_name})
    if tags:
        body.setdefault("tags", [])
        body["tags"].extend([{"key": k, "value": str(v)} for k, v in tags.items()])
    result = _post("/api/2.0/mlflow/runs/create", body)
    if "error" in result:
        return result
    run = result.get("run", {})
    return {
        "run_id":        run.get("info", {}).get("run_id"),
        "experiment_id": run.get("info", {}).get("experiment_id"),
        "status":        run.get("info", {}).get("status"),
    }


def mlflow_log_params(run_id: str, params: dict) -> dict:
    """Log multiple hyperparameters to a run."""
    body = {
        "run_id": run_id,
        "params": [{"key": k, "value": str(v)} for k, v in params.items()],
    }
    result = _post("/api/2.0/mlflow/runs/log-batch", body)
    if "error" in result:
        return result
    return {"logged": True, "run_id": run_id, "param_count": len(params)}


def mlflow_log_metrics(
    run_id: str,
    metrics: dict,
    step: Optional[int] = None,
) -> dict:
    """Log multiple metrics to a run (optionally at a specific step)."""
    ts   = int(time.time() * 1000)
    body = {
        "run_id":  run_id,
        "metrics": [
            {"key": k, "value": float(v), "timestamp": ts, "step": step or 0}
            for k, v in metrics.items()
        ],
    }
    result = _post("/api/2.0/mlflow/runs/log-batch", body)
    if "error" in result:
        return result
    return {"logged": True, "run_id": run_id, "metric_count": len(metrics)}


def mlflow_end_run(run_id: str, status: str = "FINISHED") -> dict:
    """Mark a run as FINISHED, FAILED, or KILLED."""
    result = _post("/api/2.0/mlflow/runs/update", {
        "run_id":   run_id,
        "status":   status,
        "end_time": int(time.time() * 1000),
    })
    if "error" in result:
        return result
    return {"ended": True, "run_id": run_id, "status": status}


def mlflow_get_run(run_id: str) -> dict:
    """Get full details for a run including metrics and params."""
    result = _get("/api/2.0/mlflow/runs/get", {"run_id": run_id})
    if "error" in result:
        return result
    run  = result.get("run", {})
    info = run.get("info", {})
    data = run.get("data", {})
    return {
        "run_id":        info.get("run_id"),
        "status":        info.get("status"),
        "start_time":    info.get("start_time"),
        "end_time":      info.get("end_time"),
        "experiment_id": info.get("experiment_id"),
        "metrics":       {m["key"]: m["value"] for m in data.get("metrics", [])},
        "params":        {p["key"]: p["value"] for p in data.get("params", [])},
        "tags":          {t["key"]: t["value"] for t in data.get("tags", [])},
    }


def mlflow_search_runs(
    experiment_ids: list[str],
    filter_string: str = "",
    order_by: Optional[list[str]] = None,
    max_results: int = 50,
) -> dict:
    """Search runs with optional filter expression."""
    body = {
        "experiment_ids": experiment_ids,
        "filter":         filter_string,
        "run_view_type":  "ACTIVE_ONLY",
        "max_results":    max_results,
    }
    if order_by:
        body["order_by"] = order_by
    result = _post("/api/2.0/mlflow/runs/search", body)
    if "error" in result:
        return result
    runs = []
    for r in result.get("runs", []):
        info = r.get("info", {})
        data = r.get("data", {})
        runs.append({
            "run_id":   info.get("run_id"),
            "status":   info.get("status"),
            "metrics":  {m["key"]: m["value"] for m in data.get("metrics", [])},
            "params":   {p["key"]: p["value"] for p in data.get("params", [])},
        })
    return {"runs": runs, "count": len(runs)}


def mlflow_get_best_run(
    experiment_id: str,
    metric: str,
    mode: str = "max",   # max | min
) -> dict:
    """Find the run with the best value of a specific metric."""
    result = mlflow_search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"metrics.{metric} > -1e10",
        order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
        max_results=1,
    )
    if "error" in result:
        return result
    runs = result.get("runs", [])
    if not runs:
        return {"error": f"No runs with metric '{metric}' found in experiment {experiment_id}"}
    best = runs[0]
    return {
        "run_id":       best["run_id"],
        "best_value":   best["metrics"].get(metric),
        "metric":       metric,
        "mode":         mode,
        "all_metrics":  best["metrics"],
        "params":       best["params"],
    }


def mlflow_compare_runs(run_ids: list[str]) -> dict:
    """Side-by-side metric + param comparison for a list of run IDs."""
    runs = []
    for rid in run_ids:
        run = mlflow_get_run(rid)
        if "error" not in run:
            runs.append(run)
    if not runs:
        return {"error": "No valid runs found"}

    # Collect all metric keys
    all_metrics = set()
    for r in runs:
        all_metrics.update(r.get("metrics", {}).keys())

    comparison = []
    for r in runs:
        row: dict = {"run_id": r["run_id"], "status": r.get("status")}
        for m in sorted(all_metrics):
            row[f"metric:{m}"] = r.get("metrics", {}).get(m)
        row["params"] = r.get("params", {})
        comparison.append(row)

    return {"runs": comparison, "metrics_compared": sorted(all_metrics)}


# ── MCP Schema ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "mlflow_create_experiment",
        "description": "Create a new MLflow experiment.",
        "inputSchema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name":               {"type": "string"},
                "tags":               {"type": "object"},
                "artifact_location":  {"type": "string"},
            },
        },
    },
    {
        "name": "mlflow_list_experiments",
        "description": "List MLflow experiments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "view_type":   {"type": "string", "default": "ACTIVE_ONLY"},
                "max_results": {"type": "integer", "default": 100},
            },
        },
    },
    {
        "name": "mlflow_start_run",
        "description": "Start a new MLflow run in an experiment.",
        "inputSchema": {
            "type": "object",
            "required": ["experiment_id"],
            "properties": {
                "experiment_id": {"type": "string"},
                "run_name":      {"type": "string"},
                "tags":          {"type": "object"},
            },
        },
    },
    {
        "name": "mlflow_log_params",
        "description": "Log hyperparameters to a run.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id", "params"],
            "properties": {
                "run_id": {"type": "string"},
                "params": {"type": "object"},
            },
        },
    },
    {
        "name": "mlflow_log_metrics",
        "description": "Log metrics (with optional step) to a run.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id", "metrics"],
            "properties": {
                "run_id":  {"type": "string"},
                "metrics": {"type": "object"},
                "step":    {"type": "integer"},
            },
        },
    },
    {
        "name": "mlflow_end_run",
        "description": "Mark a run as FINISHED, FAILED, or KILLED.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id"],
            "properties": {
                "run_id": {"type": "string"},
                "status": {"type": "string", "enum": ["FINISHED", "FAILED", "KILLED"], "default": "FINISHED"},
            },
        },
    },
    {
        "name": "mlflow_get_run",
        "description": "Get all details for a specific run.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id"],
            "properties": {"run_id": {"type": "string"}},
        },
    },
    {
        "name": "mlflow_search_runs",
        "description": "Search runs with filter expressions.",
        "inputSchema": {
            "type": "object",
            "required": ["experiment_ids"],
            "properties": {
                "experiment_ids": {"type": "array", "items": {"type": "string"}},
                "filter_string":  {"type": "string", "description": "MLflow filter, e.g. \"metrics.accuracy > 0.8\""},
                "order_by":       {"type": "array", "items": {"type": "string"}},
                "max_results":    {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "mlflow_get_best_run",
        "description": "Find the run with the highest (or lowest) value of a metric.",
        "inputSchema": {
            "type": "object",
            "required": ["experiment_id", "metric"],
            "properties": {
                "experiment_id": {"type": "string"},
                "metric":        {"type": "string"},
                "mode":          {"type": "string", "enum": ["max", "min"], "default": "max"},
            },
        },
    },
    {
        "name": "mlflow_compare_runs",
        "description": "Side-by-side comparison of metrics and params across multiple runs.",
        "inputSchema": {
            "type": "object",
            "required": ["run_ids"],
            "properties": {"run_ids": {"type": "array", "items": {"type": "string"}}},
        },
    },
]

TOOL_MAP = {
    "mlflow_create_experiment": mlflow_create_experiment,
    "mlflow_list_experiments":  mlflow_list_experiments,
    "mlflow_start_run":         mlflow_start_run,
    "mlflow_log_params":        mlflow_log_params,
    "mlflow_log_metrics":       mlflow_log_metrics,
    "mlflow_end_run":           mlflow_end_run,
    "mlflow_get_run":           mlflow_get_run,
    "mlflow_search_runs":       mlflow_search_runs,
    "mlflow_get_best_run":      mlflow_get_best_run,
    "mlflow_compare_runs":      mlflow_compare_runs,
}


# ── MCP Stdio Loop ─────────────────────────────────────────────────────────────

def _respond(req_id, result: dict) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
    sys.stdout.flush()

def _error(req_id, code: int, message: str) -> None:
    sys.stdout.write(json.dumps({
        "jsonrpc": "2.0", "id": req_id,
        "error":   {"code": code, "message": message},
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
                "serverInfo":      {"name": "mlflow", "version": "1.0"},
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
