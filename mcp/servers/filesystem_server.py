"""
NEXUS MCP Filesystem Server
Provides safe file system access via MCP protocol (stdio transport).
Tools: read_file, write_file, list_directory, search_files, get_file_info

Security: all paths are validated against ALLOWED_ROOTS.
Run: python nexus/mcp/servers/filesystem_server.py
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Allowed roots (configure via env or hardcode safe paths) ──────────────────
_raw_roots = os.getenv("MCP_ALLOWED_ROOTS", str(Path.home()))
ALLOWED_ROOTS: list[Path] = [Path(r).resolve() for r in _raw_roots.split(":")]


def _is_allowed(path: str) -> tuple[bool, Path]:
    """Return (ok, resolved_path). ok=False if outside allowed roots."""
    try:
        resolved = Path(path).resolve()
        for root in ALLOWED_ROOTS:
            try:
                resolved.relative_to(root)
                return True, resolved
            except ValueError:
                continue
        return False, resolved
    except Exception:
        return False, Path(path)


def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "isError": True}


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


# ── Tool implementations ───────────────────────────────────────────────────────

def read_file(path: str, encoding: str = "utf-8", max_bytes: int = 500_000) -> dict:
    ok, resolved = _is_allowed(path)
    if not ok:
        return _err(f"Path not in allowed roots: {path}")
    if not resolved.exists():
        return _err(f"File not found: {path}")
    if not resolved.is_file():
        return _err(f"Not a file: {path}")
    size = resolved.stat().st_size
    if size > max_bytes:
        return _err(f"File too large ({size} bytes > {max_bytes} limit). Use offset/limit.")

    mime, _ = mimetypes.guess_type(str(resolved))
    if mime and mime.startswith("image/"):
        return _err("Binary image files not supported in text mode. Use read_media_file.")

    try:
        content = resolved.read_text(encoding=encoding, errors="replace")
        return _ok(content)
    except Exception as exc:
        return _err(str(exc))


def write_file(path: str, content: str, encoding: str = "utf-8") -> dict:
    ok, resolved = _is_allowed(path)
    if not ok:
        return _err(f"Path not in allowed roots: {path}")
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
        return _ok(f"Written {len(content)} chars to {resolved}")
    except Exception as exc:
        return _err(str(exc))


def list_directory(path: str, recursive: bool = False, max_entries: int = 200) -> dict:
    ok, resolved = _is_allowed(path)
    if not ok:
        return _err(f"Path not in allowed roots: {path}")
    if not resolved.exists():
        return _err(f"Directory not found: {path}")
    if not resolved.is_dir():
        return _err(f"Not a directory: {path}")

    entries = []
    try:
        glob_fn = resolved.rglob("*") if recursive else resolved.iterdir()
        for entry in glob_fn:
            if len(entries) >= max_entries:
                entries.append({"name": "... (truncated)", "type": "info"})
                break
            stat = entry.stat()
            entries.append({
                "name":     str(entry.relative_to(resolved)),
                "type":     "dir" if entry.is_dir() else "file",
                "size":     stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    except Exception as exc:
        return _err(str(exc))

    return _ok(json.dumps({"path": str(resolved), "entries": entries}, indent=2))


def search_files(
    root: str,
    pattern: str = "*",
    content_pattern: str = "",
    max_results: int = 50,
) -> dict:
    ok, resolved = _is_allowed(root)
    if not ok:
        return _err(f"Path not in allowed roots: {root}")

    results = []
    try:
        for entry in resolved.rglob(pattern):
            if not entry.is_file():
                continue
            if content_pattern:
                try:
                    text = entry.read_text(encoding="utf-8", errors="ignore")
                    if content_pattern.lower() not in text.lower():
                        continue
                except Exception:
                    continue
            results.append(str(entry))
            if len(results) >= max_results:
                break
    except Exception as exc:
        return _err(str(exc))

    return _ok(json.dumps({"matches": results, "count": len(results)}, indent=2))


def get_file_info(path: str) -> dict:
    ok, resolved = _is_allowed(path)
    if not ok:
        return _err(f"Path not in allowed roots: {path}")
    if not resolved.exists():
        return _err(f"Path not found: {path}")

    stat = resolved.stat()
    mime, _ = mimetypes.guess_type(str(resolved))
    info = {
        "path":     str(resolved),
        "name":     resolved.name,
        "type":     "directory" if resolved.is_dir() else "file",
        "size":     stat.st_size,
        "mime":     mime or "unknown",
        "created":  datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "readable": os.access(resolved, os.R_OK),
        "writable": os.access(resolved, os.W_OK),
    }
    return _ok(json.dumps(info, indent=2))


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = {
    "read_file": {
        "fn": read_file,
        "description": "Read the contents of a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":     {"type": "string", "description": "File path to read"},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_bytes":{"type": "integer", "default": 500000},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "fn": write_file,
        "description": "Write content to a file (creates parent dirs if needed)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
                "encoding":{"type": "string", "default": "utf-8"},
            },
            "required": ["path", "content"],
        },
    },
    "list_directory": {
        "fn": list_directory,
        "description": "List directory contents with metadata",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string"},
                "recursive":  {"type": "boolean", "default": False},
                "max_entries":{"type": "integer", "default": 200},
            },
            "required": ["path"],
        },
    },
    "search_files": {
        "fn": search_files,
        "description": "Search for files by name pattern and/or content",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root":            {"type": "string"},
                "pattern":         {"type": "string", "default": "*"},
                "content_pattern": {"type": "string", "default": ""},
                "max_results":     {"type": "integer", "default": 50},
            },
            "required": ["root"],
        },
    },
    "get_file_info": {
        "fn": get_file_info,
        "description": "Get metadata about a file or directory",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


# ── MCP stdio server loop ──────────────────────────────────────────────────────

def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", "")
    rpc_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "nexus-filesystem", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "tools": [
                    {"name": name, "description": spec["description"],
                     "inputSchema": spec["inputSchema"]}
                    for name, spec in TOOLS.items()
                ]
            },
        }

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": _err(f"Unknown tool: {tool_name}"),
            }
        try:
            result = TOOLS[tool_name]["fn"](**arguments)
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": _err(f"Tool execution error: {exc}"),
            }

    # Notifications (no response needed)
    if method.startswith("notifications/"):
        return None

    return {
        "jsonrpc": "2.0", "id": rpc_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_message(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
