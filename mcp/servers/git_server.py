"""
NEXUS MCP Git Server
Git repository operations via MCP protocol (stdio).
All operations are READ-ONLY by default.
Write operations (commit, push) require explicit flag.

Tools:
  git_status, git_log, git_diff, git_blame, git_show,
  git_branches, git_search_commits, git_file_history
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: str, timeout: int = 15) -> tuple[str, str, int]:
    """Run a git command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True,
            cwd=cwd, timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "git command timed out", -1
    except FileNotFoundError:
        return "", "git not found in PATH", -2


def _validate_repo(path: str) -> tuple[bool, str]:
    """Check if path is a valid git repository."""
    if not path or not Path(path).exists():
        return False, f"Path not found: {path}"
    stdout, stderr, rc = _run_git(["rev-parse", "--git-dir"], path)
    if rc != 0:
        return False, f"Not a git repository: {path}"
    return True, ""


def _ok(data: Any) -> dict:
    text = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (dict, list)) else str(data)
    return {"content": [{"type": "text", "text": text}]}


def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "isError": True}


# ── Tool implementations ───────────────────────────────────────────────────────

def git_status(repo_path: str) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)
    stdout, stderr, rc = _run_git(["status", "--porcelain", "-b"], repo_path)
    if rc != 0: return _err(stderr or "git status failed")
    return _ok({"status": stdout, "repo": repo_path})


def git_log(
    repo_path: str,
    max_count: int = 20,
    branch: str = "HEAD",
    author: str = "",
    since: str = "",
    format: str = "oneline",
) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)

    fmt_map = {
        "oneline": "%h %as %an: %s",
        "detailed": "%H%n%an <%ae>%n%aI%n%s%n%b%n---",
        "json_lines": '{"hash":"%h","date":"%as","author":"%an","message":"%s"}',
    }
    git_format = fmt_map.get(format, fmt_map["oneline"])

    args = ["log", f"--pretty=format:{git_format}", f"-n{max_count}", branch]
    if author:
        args.extend([f"--author={author}"])
    if since:
        args.extend([f"--since={since}"])

    stdout, stderr, rc = _run_git(args, repo_path)
    if rc != 0: return _err(stderr)

    if format == "json_lines":
        entries = []
        for line in stdout.strip().split("\n"):
            try:
                entries.append(json.loads(line))
            except Exception:
                entries.append({"raw": line})
        return _ok({"commits": entries, "count": len(entries)})

    return _ok({"log": stdout, "count": stdout.count("\n") + 1 if stdout else 0})


def git_diff(
    repo_path: str,
    ref1: str = "HEAD",
    ref2: str = "",
    file_path: str = "",
    staged: bool = False,
    stat_only: bool = False,
) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)

    args = ["diff"]
    if staged:
        args.append("--staged")
    if stat_only:
        args.append("--stat")
    if ref2:
        args.extend([ref1, ref2])
    elif ref1 != "HEAD":
        args.append(ref1)
    if file_path:
        args.extend(["--", file_path])

    stdout, stderr, rc = _run_git(args, repo_path)
    if rc != 0 and stderr: return _err(stderr)
    return _ok({"diff": stdout[:20000], "truncated": len(stdout) > 20000})


def git_blame(repo_path: str, file_path: str, start_line: int = 1, end_line: int = 50) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)
    if not file_path: return _err("file_path required")

    args = ["blame", "-L", f"{start_line},{end_line}", "--porcelain", file_path]
    stdout, stderr, rc = _run_git(args, repo_path)
    if rc != 0: return _err(stderr)
    return _ok({"blame": stdout[:10000]})


def git_show(repo_path: str, ref: str = "HEAD", file_path: str = "") -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)

    target = f"{ref}:{file_path}" if file_path else ref
    stdout, stderr, rc = _run_git(["show", target], repo_path)
    if rc != 0: return _err(stderr)
    return _ok({"content": stdout[:20000], "ref": target})


def git_branches(repo_path: str, include_remote: bool = False) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)

    args = ["branch", "-v"]
    if include_remote:
        args.append("-a")
    stdout, stderr, rc = _run_git(args, repo_path)
    if rc != 0: return _err(stderr)

    branches = []
    for line in stdout.strip().split("\n"):
        if line.strip():
            current = line.startswith("*")
            parts = line.lstrip("* ").split()
            branches.append({
                "name":    parts[0] if parts else "",
                "hash":    parts[1] if len(parts) > 1 else "",
                "message": " ".join(parts[2:]) if len(parts) > 2 else "",
                "current": current,
            })
    return _ok({"branches": branches})


def git_search_commits(
    repo_path: str, query: str, max_count: int = 10
) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)
    if not query: return _err("query required")

    stdout, stderr, rc = _run_git(
        ["log", "--all", f"-n{max_count}", f"--grep={query}",
         "--pretty=format:%h %as %an: %s"],
        repo_path,
    )
    if rc != 0: return _err(stderr)
    return _ok({"query": query, "results": stdout, "found": bool(stdout.strip())})


def git_file_history(repo_path: str, file_path: str, max_count: int = 10) -> dict:
    ok, err = _validate_repo(repo_path)
    if not ok: return _err(err)
    if not file_path: return _err("file_path required")

    stdout, stderr, rc = _run_git(
        ["log", f"-n{max_count}", "--follow",
         "--pretty=format:%h %as %an: %s", "--", file_path],
        repo_path,
    )
    if rc != 0: return _err(stderr)
    return _ok({"file": file_path, "history": stdout})


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = {
    "git_status": {
        "fn": git_status,
        "description": "Show working tree status (staged/unstaged/untracked files)",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_path": {"type": "string", "description": "Path to git repository"}},
            "required": ["repo_path"],
        },
    },
    "git_log": {
        "fn": git_log,
        "description": "Show commit log with optional filters",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":  {"type": "string"},
                "max_count":  {"type": "integer", "default": 20},
                "branch":     {"type": "string", "default": "HEAD"},
                "author":     {"type": "string", "default": ""},
                "since":      {"type": "string", "description": "e.g. '2 weeks ago'", "default": ""},
                "format":     {"type": "string", "enum": ["oneline", "detailed", "json_lines"], "default": "oneline"},
            },
            "required": ["repo_path"],
        },
    },
    "git_diff": {
        "fn": git_diff,
        "description": "Show diff between commits, staged changes, or working tree",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":  {"type": "string"},
                "ref1":       {"type": "string", "default": "HEAD"},
                "ref2":       {"type": "string", "default": ""},
                "file_path":  {"type": "string", "default": ""},
                "staged":     {"type": "boolean", "default": False},
                "stat_only":  {"type": "boolean", "default": False},
            },
            "required": ["repo_path"],
        },
    },
    "git_blame": {
        "fn": git_blame,
        "description": "Show who last modified each line of a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":   {"type": "string"},
                "file_path":   {"type": "string"},
                "start_line":  {"type": "integer", "default": 1},
                "end_line":    {"type": "integer", "default": 50},
            },
            "required": ["repo_path", "file_path"],
        },
    },
    "git_show": {
        "fn": git_show,
        "description": "Show content of a commit or file at a specific revision",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":  {"type": "string"},
                "ref":        {"type": "string", "default": "HEAD"},
                "file_path":  {"type": "string", "default": ""},
            },
            "required": ["repo_path"],
        },
    },
    "git_branches": {
        "fn": git_branches,
        "description": "List local (and optionally remote) branches",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":       {"type": "string"},
                "include_remote":  {"type": "boolean", "default": False},
            },
            "required": ["repo_path"],
        },
    },
    "git_search_commits": {
        "fn": git_search_commits,
        "description": "Search commit messages matching a keyword",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":  {"type": "string"},
                "query":      {"type": "string"},
                "max_count":  {"type": "integer", "default": 10},
            },
            "required": ["repo_path", "query"],
        },
    },
    "git_file_history": {
        "fn": git_file_history,
        "description": "Show commit history for a specific file (follows renames)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path":  {"type": "string"},
                "file_path":  {"type": "string"},
                "max_count":  {"type": "integer", "default": 10},
            },
            "required": ["repo_path", "file_path"],
        },
    },
}


# ── MCP stdio loop (same pattern as filesystem_server.py) ────────────────────

def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", "")
    rpc_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "nexus-git", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "tools": [
                    {"name": n, "description": s["description"], "inputSchema": s["inputSchema"]}
                    for n, s in TOOLS.items()
                ]
            },
        }
    if method == "tools/call":
        params    = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name not in TOOLS:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": f"ERROR: Unknown tool {tool_name}"}], "isError": True}}
        try:
            result = TOOLS[tool_name]["fn"](**arguments)
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": f"ERROR: {exc}"}], "isError": True}}
    if method.startswith("notifications/"):
        return None
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
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
