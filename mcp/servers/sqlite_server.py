"""
NEXUS MCP SQLite Server
SQLite database operations via MCP protocol.
Supports read-only mode by default; write ops require allow_write=True.

Tools:
  list_tables, describe_table, execute_query,
  execute_write, create_table, insert_row, get_schema
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Safety: always block these even in write mode
_BLOCKED_SQL_PATTERNS = [
    "drop database", "drop schema",
    "truncate",   "pragma wal_checkpoint",
    "attach database",
]

ALLOW_WRITE = os.getenv("SQLITE_ALLOW_WRITE", "false").lower() == "true"


def _ok(data: Any) -> dict:
    text = json.dumps(data, indent=2, default=str) if isinstance(data, (dict, list)) else str(data)
    return {"content": [{"type": "text", "text": text}]}

def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "isError": True}

def _get_conn(db_path: str) -> tuple[sqlite3.Connection | None, str]:
    if not db_path:
        return None, "db_path is required"
    if not Path(db_path).exists():
        if ALLOW_WRITE:
            # Create new database
            return sqlite3.connect(db_path), ""
        return None, f"Database not found: {db_path}"
    return sqlite3.connect(db_path), ""


def list_tables(db_path: str) -> dict:
    conn, err = _get_conn(db_path)
    if err: return _err(err)
    try:
        cur = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
        )
        tables = [{"name": r[0], "type": r[1]} for r in cur.fetchall()]
        conn.close()
        return _ok({"tables": tables, "count": len(tables)})
    except Exception as exc:
        return _err(str(exc))


def describe_table(db_path: str, table_name: str) -> dict:
    conn, err = _get_conn(db_path)
    if err: return _err(err)
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        cols = [
            {"cid": r[0], "name": r[1], "type": r[2], "notnull": bool(r[3]),
             "default": r[4], "pk": bool(r[5])}
            for r in cur.fetchall()
        ]
        # Row count
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except Exception:
            count = -1
        conn.close()
        return _ok({"table": table_name, "columns": cols, "row_count": count})
    except Exception as exc:
        return _err(str(exc))


def execute_query(db_path: str, sql: str, params: list = None, limit: int = 100) -> dict:
    """Execute a SELECT query (read-only)."""
    sql_stripped = sql.strip().lower()
    if not sql_stripped.startswith("select"):
        return _err("execute_query is for SELECT only. Use execute_write for modifications.")
    for blocked in _BLOCKED_SQL_PATTERNS:
        if blocked in sql_stripped:
            return _err(f"Blocked SQL pattern: {blocked}")

    conn, err = _get_conn(db_path)
    if err: return _err(err)
    try:
        # Append LIMIT if not present
        if "limit" not in sql_stripped:
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        cur = conn.execute(sql, params or [])
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return _ok({"sql": sql, "columns": cols, "rows": rows, "count": len(rows)})
    except Exception as exc:
        return _err(str(exc))


def execute_write(db_path: str, sql: str, params: list = None) -> dict:
    """Execute a write SQL statement (INSERT/UPDATE/DELETE/CREATE)."""
    if not ALLOW_WRITE:
        return _err(
            "Write operations disabled. "
            "Set SQLITE_ALLOW_WRITE=true env var to enable."
        )
    sql_stripped = sql.strip().lower()
    for blocked in _BLOCKED_SQL_PATTERNS:
        if blocked in sql_stripped:
            return _err(f"Blocked SQL pattern: {blocked}")

    conn, err = _get_conn(db_path)
    if err: return _err(err)
    try:
        cur = conn.execute(sql, params or [])
        conn.commit()
        affected = cur.rowcount
        last_id  = cur.lastrowid
        conn.close()
        return _ok({"sql": sql, "rows_affected": affected, "last_insert_id": last_id})
    except Exception as exc:
        return _err(str(exc))


def create_table(db_path: str, table_name: str, columns: list[dict]) -> dict:
    """
    Create a table.
    columns: [{"name": "id", "type": "INTEGER", "pk": true, "notnull": false}, ...]
    """
    if not ALLOW_WRITE:
        return _err("Write operations disabled. Set SQLITE_ALLOW_WRITE=true to enable.")
    col_defs = []
    for col in columns:
        defn = f"{col['name']} {col.get('type', 'TEXT')}"
        if col.get("pk"):
            defn += " PRIMARY KEY"
        if col.get("autoincrement"):
            defn += " AUTOINCREMENT"
        if col.get("notnull"):
            defn += " NOT NULL"
        if col.get("default") is not None:
            defn += f" DEFAULT {col['default']}"
        col_defs.append(defn)
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
    return execute_write(db_path, sql)


def insert_row(db_path: str, table_name: str, row: dict) -> dict:
    """Insert a single row dict into a table."""
    if not ALLOW_WRITE:
        return _err("Write operations disabled.")
    cols   = list(row.keys())
    vals   = list(row.values())
    params_placeholder = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({params_placeholder})"
    return execute_write(db_path, sql, vals)


def get_schema(db_path: str) -> dict:
    """Get full DDL schema for all tables."""
    conn, err = _get_conn(db_path)
    if err: return _err(err)
    try:
        cur = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
        )
        schema = [{"name": r[0], "ddl": r[1]} for r in cur.fetchall()]
        conn.close()
        return _ok({"schema": schema})
    except Exception as exc:
        return _err(str(exc))


TOOLS = {
    "list_tables":    {"fn": list_tables,   "description": "List all tables and views in the database",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}}, "required": ["db_path"]}},
    "describe_table": {"fn": describe_table,"description": "Show columns and row count for a table",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}, "table_name": {"type": "string"}}, "required": ["db_path", "table_name"]}},
    "execute_query":  {"fn": execute_query, "description": "Execute a SELECT query (read-only)",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}, "sql": {"type": "string"}, "params": {"type": "array"}, "limit": {"type": "integer", "default": 100}}, "required": ["db_path", "sql"]}},
    "execute_write":  {"fn": execute_write, "description": "Execute INSERT/UPDATE/DELETE/CREATE (requires SQLITE_ALLOW_WRITE=true)",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}, "sql": {"type": "string"}, "params": {"type": "array"}}, "required": ["db_path", "sql"]}},
    "create_table":   {"fn": create_table,  "description": "Create a new table with specified columns",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}, "table_name": {"type": "string"}, "columns": {"type": "array"}}, "required": ["db_path", "table_name", "columns"]}},
    "insert_row":     {"fn": insert_row,    "description": "Insert a single row into a table",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}, "table_name": {"type": "string"}, "row": {"type": "object"}}, "required": ["db_path", "table_name", "row"]}},
    "get_schema":     {"fn": get_schema,    "description": "Get full DDL schema for all objects",
                       "inputSchema": {"type": "object", "properties": {"db_path": {"type": "string"}}, "required": ["db_path"]}},
}


def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", ""); rpc_id = msg.get("id")
    if method == "initialize":
        return {"jsonrpc":"2.0","id":rpc_id,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"nexus-sqlite","version":"1.0.0"}}}
    if method == "tools/list":
        return {"jsonrpc":"2.0","id":rpc_id,"result":{"tools":[{"name":n,"description":s["description"],"inputSchema":s["inputSchema"]} for n,s in TOOLS.items()]}}
    if method == "tools/call":
        params=msg.get("params",{}); name=params.get("name",""); args=params.get("arguments",{})
        if name not in TOOLS: return {"jsonrpc":"2.0","id":rpc_id,"result":_err(f"Unknown: {name}")}
        try: return {"jsonrpc":"2.0","id":rpc_id,"result":TOOLS[name]["fn"](**args)}
        except Exception as exc: return {"jsonrpc":"2.0","id":rpc_id,"result":_err(str(exc))}
    if method.startswith("notifications/"): return None
    return {"jsonrpc":"2.0","id":rpc_id,"error":{"code":-32601,"message":f"Unknown: {method}"}}


def main():
    for line in sys.stdin:
        line=line.strip()
        if not line: continue
        try: msg=json.loads(line)
        except: continue
        resp=handle_message(msg)
        if resp: sys.stdout.write(json.dumps(resp)+"\n"); sys.stdout.flush()

if __name__=="__main__": main()
