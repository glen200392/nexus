"""
NEXUS MCP Server — Slack Web API
Provides read/write access to Slack workspaces via the Slack Web API.

Tools:
  - slack_post_message        Post a message to a channel
  - slack_post_thread_reply   Reply to a thread
  - slack_list_channels       List public/private channels
  - slack_get_channel_history Read recent messages from a channel
  - slack_add_reaction        Add an emoji reaction to a message
  - slack_list_users          List workspace members
  - slack_search_messages     Search messages by query
  - slack_upload_file         Upload a text snippet or file content

Environment:
  SLACK_BOT_TOKEN   — xoxb-... Bot User OAuth Token (required)
  SLACK_TEAM_ID     — Workspace ID (optional, for multi-workspace)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
_BASE = "https://slack.com/api"


# ── Slack API helper ───────────────────────────────────────────────────────────

def _api(method: str, payload: dict) -> dict:
    """POST to Slack Web API. All methods use JSON body."""
    if not SLACK_BOT_TOKEN:
        return {"ok": False, "error": "SLACK_BOT_TOKEN not set"}
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{_BASE}/{method}",
        data=data,
        headers={
            "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
            "Content-Type":  "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            return result
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _get(method: str, params: dict) -> dict:
    """GET from Slack Web API."""
    if not SLACK_BOT_TOKEN:
        return {"ok": False, "error": "SLACK_BOT_TOKEN not set"}
    qs  = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{_BASE}/{method}?{qs}",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Tool Implementations ───────────────────────────────────────────────────────

def slack_post_message(
    channel: str,
    text: str,
    blocks: Optional[list] = None,
    unfurl_links: bool = True,
) -> dict:
    """Post a message to a Slack channel or DM (channel can be #name, channel_id, or @username)."""
    payload: dict[str, Any] = {"channel": channel, "text": text}
    if blocks:
        payload["blocks"] = blocks
    if not unfurl_links:
        payload["unfurl_links"] = False
    result = _api("chat.postMessage", payload)
    if result.get("ok"):
        return {
            "ts":      result["ts"],
            "channel": result["channel"],
            "text":    text[:200],
        }
    return {"error": result.get("error", "unknown")}


def slack_post_thread_reply(channel: str, thread_ts: str, text: str) -> dict:
    """Reply to a message thread."""
    result = _api("chat.postMessage", {
        "channel":   channel,
        "thread_ts": thread_ts,
        "text":      text,
    })
    if result.get("ok"):
        return {"ts": result["ts"], "thread_ts": thread_ts}
    return {"error": result.get("error", "unknown")}


def slack_list_channels(
    types: str = "public_channel,private_channel",
    limit: int = 100,
    exclude_archived: bool = True,
) -> dict:
    """List channels the bot has access to."""
    result = _get("conversations.list", {
        "types":            types,
        "limit":            limit,
        "exclude_archived": str(exclude_archived).lower(),
    })
    if not result.get("ok"):
        return {"error": result.get("error", "unknown")}
    channels = [
        {
            "id":          c["id"],
            "name":        c.get("name", ""),
            "is_private":  c.get("is_private", False),
            "num_members": c.get("num_members", 0),
            "topic":       c.get("topic", {}).get("value", ""),
        }
        for c in result.get("channels", [])
    ]
    return {"channels": channels, "count": len(channels)}


def slack_get_channel_history(
    channel: str,
    limit: int = 20,
    oldest: Optional[str] = None,
) -> dict:
    """Fetch recent messages from a channel."""
    params: dict = {"channel": channel, "limit": limit}
    if oldest:
        params["oldest"] = oldest
    result = _get("conversations.history", params)
    if not result.get("ok"):
        return {"error": result.get("error", "unknown")}
    messages = [
        {
            "ts":   m.get("ts", ""),
            "user": m.get("user", m.get("bot_id", "unknown")),
            "text": m.get("text", "")[:500],
            "type": m.get("type", "message"),
        }
        for m in result.get("messages", [])
    ]
    return {"channel": channel, "messages": messages, "count": len(messages)}


def slack_add_reaction(channel: str, timestamp: str, name: str) -> dict:
    """Add an emoji reaction to a message. name = emoji name without colons."""
    result = _api("reactions.add", {
        "channel":   channel,
        "timestamp": timestamp,
        "name":      name.strip(":"),
    })
    if result.get("ok"):
        return {"reacted": True, "emoji": name, "ts": timestamp}
    return {"error": result.get("error", "unknown")}


def slack_list_users(limit: int = 200) -> dict:
    """List workspace members."""
    result = _get("users.list", {"limit": limit})
    if not result.get("ok"):
        return {"error": result.get("error", "unknown")}
    users = [
        {
            "id":          u["id"],
            "name":        u.get("name", ""),
            "real_name":   u.get("real_name", ""),
            "email":       u.get("profile", {}).get("email", ""),
            "is_bot":      u.get("is_bot", False),
            "is_admin":    u.get("is_admin", False),
        }
        for u in result.get("members", [])
        if not u.get("deleted")
    ]
    return {"users": users, "count": len(users)}


def slack_search_messages(query: str, count: int = 20, sort: str = "timestamp") -> dict:
    """Search messages across the workspace."""
    result = _get("search.messages", {
        "query": query,
        "count": count,
        "sort":  sort,
    })
    if not result.get("ok"):
        return {"error": result.get("error", "unknown")}
    matches = result.get("messages", {}).get("matches", [])
    hits = [
        {
            "ts":      m.get("ts", ""),
            "channel": m.get("channel", {}).get("name", ""),
            "user":    m.get("username", ""),
            "text":    m.get("text", "")[:400],
            "permalink": m.get("permalink", ""),
        }
        for m in matches
    ]
    return {"query": query, "hits": hits, "total": len(hits)}


def slack_upload_file(
    channels: str,
    content: str,
    filename: str = "output.txt",
    title: str = "",
    filetype: str = "text",
) -> dict:
    """Upload a text snippet to a channel."""
    result = _api("files.uploadV2", {
        "channel": channels,
        "content": content,
        "filename": filename,
        "title": title or filename,
        "filetype": filetype,
    })
    if result.get("ok"):
        f = result.get("file", {})
        return {"file_id": f.get("id", ""), "permalink": f.get("permalink", "")}
    return {"error": result.get("error", "unknown")}


# ── MCP Schema ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "slack_post_message",
        "description": "Post a message to a Slack channel or DM.",
        "inputSchema": {
            "type": "object",
            "required": ["channel", "text"],
            "properties": {
                "channel":      {"type": "string", "description": "#channel-name, channel_id, or @username"},
                "text":         {"type": "string"},
                "blocks":       {"type": "array",  "description": "Optional Block Kit blocks"},
                "unfurl_links": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "slack_post_thread_reply",
        "description": "Reply to an existing message thread.",
        "inputSchema": {
            "type": "object",
            "required": ["channel", "thread_ts", "text"],
            "properties": {
                "channel":   {"type": "string"},
                "thread_ts": {"type": "string", "description": "Timestamp of the parent message"},
                "text":      {"type": "string"},
            },
        },
    },
    {
        "name": "slack_list_channels",
        "description": "List channels the bot can access.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "types":            {"type": "string", "default": "public_channel,private_channel"},
                "limit":            {"type": "integer", "default": 100},
                "exclude_archived": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "slack_get_channel_history",
        "description": "Retrieve recent messages from a channel.",
        "inputSchema": {
            "type": "object",
            "required": ["channel"],
            "properties": {
                "channel": {"type": "string"},
                "limit":   {"type": "integer", "default": 20},
                "oldest":  {"type": "string", "description": "Unix timestamp — only messages after this"},
            },
        },
    },
    {
        "name": "slack_add_reaction",
        "description": "Add an emoji reaction to a message.",
        "inputSchema": {
            "type": "object",
            "required": ["channel", "timestamp", "name"],
            "properties": {
                "channel":   {"type": "string"},
                "timestamp": {"type": "string"},
                "name":      {"type": "string", "description": "Emoji name, e.g. 'thumbsup' or ':white_check_mark:'"},
            },
        },
    },
    {
        "name": "slack_list_users",
        "description": "List all workspace members.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 200}},
        },
    },
    {
        "name": "slack_search_messages",
        "description": "Search messages across the workspace.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 20},
                "sort":  {"type": "string", "enum": ["score", "timestamp"], "default": "timestamp"},
            },
        },
    },
    {
        "name": "slack_upload_file",
        "description": "Upload a text file or code snippet to a channel.",
        "inputSchema": {
            "type": "object",
            "required": ["channels", "content"],
            "properties": {
                "channels": {"type": "string"},
                "content":  {"type": "string"},
                "filename": {"type": "string", "default": "output.txt"},
                "title":    {"type": "string"},
                "filetype": {"type": "string", "default": "text"},
            },
        },
    },
]

TOOL_MAP = {
    "slack_post_message":     slack_post_message,
    "slack_post_thread_reply": slack_post_thread_reply,
    "slack_list_channels":    slack_list_channels,
    "slack_get_channel_history": slack_get_channel_history,
    "slack_add_reaction":     slack_add_reaction,
    "slack_list_users":       slack_list_users,
    "slack_search_messages":  slack_search_messages,
    "slack_upload_file":      slack_upload_file,
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
                "serverInfo":      {"name": "slack", "version": "1.0"},
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
