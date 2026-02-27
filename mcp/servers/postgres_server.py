"""
NEXUS MCP Server — PostgreSQL
Provides SQL query access to PostgreSQL databases via MCP.

Tools:
  - pg_execute_query   Run a SQL query (SELECT by default; writes require ALLOW_WRITE)
  - pg_list_schemas    List all schemas in the database
  - pg_list_tables     List tables in a schema
  - pg_describe_table  Column names, types, constraints for a table
  - pg_get_indexes     Indexes on a table
  - pg_explain         EXPLAIN ANALYZE a query (read-only, safe)
  - pg_get_stats       Table row counts and size estimates (pg_stat_user_tables)

Environment:
  PG_DSN          — Full connection string: postgresql://user:pass@host:5432/db
  PG_HOST         — (default: localhost)
  PG_PORT         — (default: 5432)
  PG_USER         — (default: postgres)
  PG_PASSWORD     — (default: empty)
  PG_DATABASE     — (default: postgres)
  PG_ALLOW_WRITE  — Set to "true" to allow INSERT/UPDATE/DELETE/DDL
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional

# ── Config ─────────────────────────────────────────────────────────────────────
PG_DSN       = os.environ.get("PG_DSN", "")
PG_HOST      = os.environ.get("PG_HOST",     "localhost")
PG_PORT      = int(os.environ.get("PG_PORT", "5432"))
PG_USER      = os.environ.get("PG_USER",     "postgres")
PG_PASSWORD  = os.environ.get("PG_PASSWORD", "")
PG_DATABASE  = os.environ.get("PG_DATABASE", "postgres")
ALLOW_WRITE  = os.environ.get("PG_ALLOW_WRITE", "false").lower() == "true"

# Blocked patterns even in ALLOW_WRITE mode
_BLOCKED = re.compile(
    r"\b(drop\s+database|drop\s+schema|truncate|pg_read_file"
    r"|copy\s+.*\s+to|alter\s+system|pg_terminate_backend)\b",
    re.I,
)
_WRITE_OPS = re.compile(
    r"^\s*(insert|update|delete|create|drop|alter|truncate|grant|revoke|vacuum)\b",
    re.I,
)

# ── Connection pool (single connection, recreate on error) ─────────────────────
_conn = None


def _get_conn():
    global _conn
    if _conn is not None:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise RuntimeError(
            "psycopg2 not installed. Run: pip install psycopg2-binary"
        )

    if PG_DSN:
        _conn = psycopg2.connect(PG_DSN)
    else:
        _conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            user=PG_USER, password=PG_PASSWORD,
            dbname=PG_DATABASE,
        )
    _conn.set_session(autocommit=False)
    return _conn


def _run(sql: str, params=None) -> tuple[list[str], list[list]]:
    """Execute SQL and return (columns, rows)."""
    import psycopg2.extras
    conn   = _get_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(sql, params)
    if cursor.description:
        cols = [d.name for d in cursor.description]
        rows = [list(r.values()) for r in cursor.fetchall()]
    else:
        cols = ["affected_rows"]
        rows = [[cursor.rowcount]]
    conn.commit()
    return cols, rows


# ── Tool Handlers ──────────────────────────────────────────────────────────────

def pg_execute_query(query: str, params: Optional[list] = None, limit: int = 200) -> dict:
    """Execute a SQL query. Writes require PG_ALLOW_WRITE=true."""
    if _BLOCKED.search(query):
        return {"error": "Query contains a blocked operation."}
    if _WRITE_OPS.search(query) and not ALLOW_WRITE:
        return {"error": "Write operations blocked. Set PG_ALLOW_WRITE=true to enable."}

    # Auto-add LIMIT for SELECT queries if not present
    normalized = query.strip().upper()
    if normalized.startswith("SELECT") and "LIMIT" not in normalized:
        query = query.rstrip("; \n") + f" LIMIT {limit}"

    try:
        cols, rows = _run(query, params)
        # Serialize non-JSON-native types (Decimal, datetime, UUID…)
        safe_rows = []
        for row in rows:
            safe_rows.append([str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for v in row])
        return {
            "columns": cols,
            "rows":    safe_rows,
            "count":   len(safe_rows),
        }
    except Exception as exc:
        global _conn
        _conn = None   # Force reconnect on next call
        return {"error": str(exc)}


def pg_list_schemas() -> dict:
    """List all schemas in the current database."""
    try:
        cols, rows = _run(
            "SELECT schema_name, schema_owner "
            "FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast') "
            "ORDER BY schema_name"
        )
        schemas = [{"name": r[0], "owner": r[1]} for r in rows]
        return {"schemas": schemas}
    except Exception as exc:
        return {"error": str(exc)}


def pg_list_tables(schema: str = "public") -> dict:
    """List tables and views in a schema."""
    try:
        cols, rows = _run(
            "SELECT table_name, table_type "
            "FROM information_schema.tables "
            "WHERE table_schema = %s "
            "ORDER BY table_type, table_name",
            [schema],
        )
        tables = [{"name": r[0], "type": r[1]} for r in rows]
        return {"schema": schema, "tables": tables, "count": len(tables)}
    except Exception as exc:
        return {"error": str(exc)}


def pg_describe_table(table: str, schema: str = "public") -> dict:
    """Describe columns, data types, nullability, and defaults."""
    try:
        cols, rows = _run(
            "SELECT column_name, data_type, character_maximum_length, "
            "       is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position",
            [schema, table],
        )
        columns = [
            {
                "name":     r[0],
                "type":     r[1] + (f"({r[2]})" if r[2] else ""),
                "nullable": r[3] == "YES",
                "default":  r[4],
            }
            for r in rows
        ]
        return {"table": f"{schema}.{table}", "columns": columns}
    except Exception as exc:
        return {"error": str(exc)}


def pg_get_indexes(table: str, schema: str = "public") -> dict:
    """Get indexes for a table."""
    try:
        cols, rows = _run(
            "SELECT indexname, indexdef "
            "FROM pg_indexes "
            "WHERE schemaname = %s AND tablename = %s",
            [schema, table],
        )
        indexes = [{"name": r[0], "definition": r[1]} for r in rows]
        return {"table": f"{schema}.{table}", "indexes": indexes}
    except Exception as exc:
        return {"error": str(exc)}


def pg_explain(query: str) -> dict:
    """Run EXPLAIN ANALYZE on a query (read-only, never executes data changes)."""
    if _WRITE_OPS.search(query):
        return {"error": "EXPLAIN only supported for read queries."}
    try:
        cols, rows = _run(f"EXPLAIN ANALYZE {query}")
        plan = "\n".join(str(r[0]) for r in rows)
        return {"plan": plan}
    except Exception as exc:
        return {"error": str(exc)}


def pg_get_stats(schema: str = "public") -> dict:
    """Get row counts and size estimates for all tables in a schema."""
    try:
        cols, rows = _run(
            "SELECT relname AS table_name, "
            "       n_live_tup AS est_rows, "
            "       pg_size_pretty(pg_total_relation_size(quote_ident(relname)::regclass)) AS total_size "
            "FROM pg_stat_user_tables "
            "WHERE schemaname = %s "
            "ORDER BY n_live_tup DESC",
            [schema],
        )
        stats = [{"table": r[0], "est_rows": r[1], "total_size": r[2]} for r in rows]
        return {"schema": schema, "tables": stats}
    except Exception as exc:
        return {"error": str(exc)}


# ── MCP Schema ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "pg_execute_query",
        "description": "Execute a SQL query against PostgreSQL. SELECT by default; set PG_ALLOW_WRITE=true for writes.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query":  {"type": "string"},
                "params": {"type": "array", "description": "Positional parameters ($1, $2…)"},
                "limit":  {"type": "integer", "default": 200, "description": "Auto-LIMIT for SELECT (ignored if LIMIT present)"},
            },
        },
    },
    {
        "name": "pg_list_schemas",
        "description": "List all schemas in the PostgreSQL database.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "pg_list_tables",
        "description": "List tables and views in a schema.",
        "inputSchema": {
            "type": "object",
            "properties": {"schema": {"type": "string", "default": "public"}},
        },
    },
    {
        "name": "pg_describe_table",
        "description": "Get column definitions for a table.",
        "inputSchema": {
            "type": "object",
            "required": ["table"],
            "properties": {
                "table":  {"type": "string"},
                "schema": {"type": "string", "default": "public"},
            },
        },
    },
    {
        "name": "pg_get_indexes",
        "description": "Get index definitions for a table.",
        "inputSchema": {
            "type": "object",
            "required": ["table"],
            "properties": {
                "table":  {"type": "string"},
                "schema": {"type": "string", "default": "public"},
            },
        },
    },
    {
        "name": "pg_explain",
        "description": "Run EXPLAIN ANALYZE on a SELECT query to inspect the query plan.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
    },
    {
        "name": "pg_get_stats",
        "description": "Get estimated row counts and sizes for tables in a schema.",
        "inputSchema": {
            "type": "object",
            "properties": {"schema": {"type": "string", "default": "public"}},
        },
    },
]

TOOL_MAP = {
    "pg_execute_query": pg_execute_query,
    "pg_list_schemas":  pg_list_schemas,
    "pg_list_tables":   pg_list_tables,
    "pg_describe_table": pg_describe_table,
    "pg_get_indexes":   pg_get_indexes,
    "pg_explain":       pg_explain,
    "pg_get_stats":     pg_get_stats,
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
                "serverInfo":      {"name": "postgres", "version": "1.0"},
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
