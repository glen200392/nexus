"""
NEXUS MCP GitHub Server
GitHub REST API v3 integration via MCP protocol.
Requires GITHUB_TOKEN environment variable.

Tools:
  list_repos, get_repo, list_issues, get_issue, create_issue, update_issue,
  list_prs, get_pr, create_pr, list_commits, get_file_content,
  search_code, get_user
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.parse
from typing import Any

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
API_BASE = "https://api.github.com"


def _ok(data: Any) -> dict:
    text = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
    return {"content": [{"type": "text", "text": text}]}

def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "isError": True}


def _request(
    method: str,
    path: str,
    body: dict = None,
    params: dict = None,
) -> tuple[Any, str]:
    """Make a GitHub API request. Returns (data, error_msg)."""
    if not GITHUB_TOKEN:
        return None, "GITHUB_TOKEN not set. Export it: export GITHUB_TOKEN=ghp_..."

    url = f"{API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    headers = {
        "Authorization":         f"Bearer {GITHUB_TOKEN}",
        "Accept":                "application/vnd.github+json",
        "X-GitHub-Api-Version":  "2022-11-28",
        "User-Agent":            "NEXUS-MCP/1.0",
        "Content-Type":          "application/json",
    }

    data_bytes = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            if content:
                return json.loads(content), ""
            return {}, ""
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        try:
            msg = json.loads(error_body).get("message", str(exc))
        except Exception:
            msg = str(exc)
        return None, f"GitHub API error {exc.code}: {msg}"
    except Exception as exc:
        return None, str(exc)


# ── Tool implementations ───────────────────────────────────────────────────────

def get_user(username: str = "") -> dict:
    path = f"/users/{username}" if username else "/user"
    data, err = _request("GET", path)
    if err: return _err(err)
    return _ok({
        "login":       data.get("login"),
        "name":        data.get("name"),
        "email":       data.get("email"),
        "bio":         data.get("bio"),
        "public_repos": data.get("public_repos"),
        "followers":   data.get("followers"),
    })


def list_repos(
    owner: str = "",
    org: str = "",
    type: str = "all",
    sort: str = "updated",
    per_page: int = 20,
) -> dict:
    if org:
        path = f"/orgs/{org}/repos"
    elif owner:
        path = f"/users/{owner}/repos"
    else:
        path = "/user/repos"
    data, err = _request("GET", path, params={"type": type, "sort": sort, "per_page": per_page})
    if err: return _err(err)
    repos = [
        {
            "name":        r.get("name"),
            "full_name":   r.get("full_name"),
            "description": r.get("description"),
            "private":     r.get("private"),
            "language":    r.get("language"),
            "stars":       r.get("stargazers_count"),
            "updated":     r.get("updated_at"),
            "url":         r.get("html_url"),
        }
        for r in (data or [])
    ]
    return _ok({"repos": repos, "count": len(repos)})


def get_repo(owner: str, repo: str) -> dict:
    data, err = _request("GET", f"/repos/{owner}/{repo}")
    if err: return _err(err)
    return _ok({
        "name":          data.get("name"),
        "full_name":     data.get("full_name"),
        "description":   data.get("description"),
        "private":       data.get("private"),
        "language":      data.get("language"),
        "default_branch":data.get("default_branch"),
        "stars":         data.get("stargazers_count"),
        "forks":         data.get("forks_count"),
        "open_issues":   data.get("open_issues_count"),
        "topics":        data.get("topics", []),
        "url":           data.get("html_url"),
        "clone_url":     data.get("clone_url"),
    })


def list_issues(
    owner: str,
    repo: str,
    state: str = "open",
    labels: str = "",
    per_page: int = 20,
    assignee: str = "",
) -> dict:
    params = {"state": state, "per_page": per_page}
    if labels:  params["labels"]   = labels
    if assignee: params["assignee"] = assignee
    data, err = _request("GET", f"/repos/{owner}/{repo}/issues", params=params)
    if err: return _err(err)
    issues = [
        {
            "number":    i.get("number"),
            "title":     i.get("title"),
            "state":     i.get("state"),
            "author":    i.get("user", {}).get("login"),
            "labels":    [l["name"] for l in i.get("labels", [])],
            "assignees": [a["login"] for a in i.get("assignees", [])],
            "comments":  i.get("comments"),
            "created":   i.get("created_at"),
            "updated":   i.get("updated_at"),
            "url":       i.get("html_url"),
        }
        for i in (data or [])
        if not i.get("pull_request")   # exclude PRs from issues list
    ]
    return _ok({"issues": issues, "count": len(issues)})


def get_issue(owner: str, repo: str, issue_number: int) -> dict:
    data, err = _request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}")
    if err: return _err(err)
    # Also get comments
    comments_data, _ = _request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}/comments")
    comments = [
        {"author": c.get("user", {}).get("login"), "body": c.get("body", "")[:500]}
        for c in (comments_data or [])
    ]
    return _ok({
        "number":   data.get("number"),
        "title":    data.get("title"),
        "body":     data.get("body"),
        "state":    data.get("state"),
        "labels":   [l["name"] for l in data.get("labels", [])],
        "comments": comments,
        "url":      data.get("html_url"),
    })


def create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str = "",
    labels: list = None,
    assignees: list = None,
) -> dict:
    payload = {"title": title, "body": body}
    if labels:    payload["labels"]    = labels
    if assignees: payload["assignees"] = assignees
    data, err = _request("POST", f"/repos/{owner}/{repo}/issues", body=payload)
    if err: return _err(err)
    return _ok({"number": data.get("number"), "url": data.get("html_url"), "state": data.get("state")})


def update_issue(
    owner: str,
    repo: str,
    issue_number: int,
    title: str = "",
    body: str = "",
    state: str = "",
    labels: list = None,
) -> dict:
    payload = {}
    if title:  payload["title"]  = title
    if body:   payload["body"]   = body
    if state:  payload["state"]  = state
    if labels is not None: payload["labels"] = labels
    data, err = _request("PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", body=payload)
    if err: return _err(err)
    return _ok({"number": data.get("number"), "state": data.get("state"), "url": data.get("html_url")})


def list_prs(
    owner: str,
    repo: str,
    state: str = "open",
    per_page: int = 20,
) -> dict:
    data, err = _request("GET", f"/repos/{owner}/{repo}/pulls",
                          params={"state": state, "per_page": per_page})
    if err: return _err(err)
    prs = [
        {
            "number": p.get("number"),
            "title":  p.get("title"),
            "state":  p.get("state"),
            "author": p.get("user", {}).get("login"),
            "head":   p.get("head", {}).get("ref"),
            "base":   p.get("base", {}).get("ref"),
            "draft":  p.get("draft"),
            "merged": p.get("merged"),
            "url":    p.get("html_url"),
        }
        for p in (data or [])
    ]
    return _ok({"prs": prs, "count": len(prs)})


def get_pr(owner: str, repo: str, pr_number: int) -> dict:
    data, err = _request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
    if err: return _err(err)
    # Get files changed
    files_data, _ = _request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
    files = [
        {"filename": f.get("filename"), "status": f.get("status"), "changes": f.get("changes")}
        for f in (files_data or [])[:20]
    ]
    return _ok({
        "number":       data.get("number"),
        "title":        data.get("title"),
        "body":         (data.get("body") or "")[:1000],
        "state":        data.get("state"),
        "head":         data.get("head", {}).get("ref"),
        "base":         data.get("base", {}).get("ref"),
        "mergeable":    data.get("mergeable"),
        "draft":        data.get("draft"),
        "files_changed": files,
        "url":          data.get("html_url"),
    })


def create_pr(
    owner: str,
    repo: str,
    title: str,
    head: str,
    base: str = "main",
    body: str = "",
    draft: bool = False,
) -> dict:
    payload = {"title": title, "head": head, "base": base, "body": body, "draft": draft}
    data, err = _request("POST", f"/repos/{owner}/{repo}/pulls", body=payload)
    if err: return _err(err)
    return _ok({"number": data.get("number"), "url": data.get("html_url"), "state": data.get("state")})


def get_file_content(owner: str, repo: str, path: str, ref: str = "") -> dict:
    params = {}
    if ref: params["ref"] = ref
    data, err = _request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)
    if err: return _err(err)
    import base64 as b64
    content_b64 = data.get("content", "").replace("\n", "")
    try:
        content = b64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        content = "[Binary file]"
    return _ok({
        "path":    data.get("path"),
        "size":    data.get("size"),
        "sha":     data.get("sha"),
        "content": content[:20000],
        "url":     data.get("html_url"),
    })


def list_commits(
    owner: str,
    repo: str,
    branch: str = "",
    path: str = "",
    per_page: int = 20,
) -> dict:
    params: dict = {"per_page": per_page}
    if branch: params["sha"]  = branch
    if path:   params["path"] = path
    data, err = _request("GET", f"/repos/{owner}/{repo}/commits", params=params)
    if err: return _err(err)
    commits = [
        {
            "sha":     c.get("sha", "")[:8],
            "message": (c.get("commit", {}).get("message") or "").split("\n")[0][:100],
            "author":  c.get("commit", {}).get("author", {}).get("name"),
            "date":    c.get("commit", {}).get("author", {}).get("date"),
            "url":     c.get("html_url"),
        }
        for c in (data or [])
    ]
    return _ok({"commits": commits, "count": len(commits)})


def search_code(query: str, repo: str = "", language: str = "", per_page: int = 10) -> dict:
    q = query
    if repo:     q += f" repo:{repo}"
    if language: q += f" language:{language}"
    data, err = _request("GET", "/search/code", params={"q": q, "per_page": per_page})
    if err: return _err(err)
    items = [
        {
            "name":       i.get("name"),
            "path":       i.get("path"),
            "repo":       i.get("repository", {}).get("full_name"),
            "url":        i.get("html_url"),
        }
        for i in data.get("items", [])
    ]
    return _ok({"query": query, "total_count": data.get("total_count", 0), "items": items})


TOOLS = {
    "get_user":       {"fn": get_user,       "description": "Get GitHub user profile",
                       "inputSchema": {"type":"object","properties":{"username":{"type":"string","default":""}}}},
    "list_repos":     {"fn": list_repos,     "description": "List repos for a user, org, or authenticated user",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string","default":""},"org":{"type":"string","default":""},"type":{"type":"string","default":"all"},"sort":{"type":"string","default":"updated"},"per_page":{"type":"integer","default":20}}}},
    "get_repo":       {"fn": get_repo,       "description": "Get repository details",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"}},"required":["owner","repo"]}},
    "list_issues":    {"fn": list_issues,    "description": "List issues in a repository",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"state":{"type":"string","default":"open"},"labels":{"type":"string","default":""},"per_page":{"type":"integer","default":20}},"required":["owner","repo"]}},
    "get_issue":      {"fn": get_issue,      "description": "Get a single issue with comments",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"issue_number":{"type":"integer"}},"required":["owner","repo","issue_number"]}},
    "create_issue":   {"fn": create_issue,   "description": "Create a new GitHub issue",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"title":{"type":"string"},"body":{"type":"string","default":""},"labels":{"type":"array","default":[]},"assignees":{"type":"array","default":[]}},"required":["owner","repo","title"]}},
    "update_issue":   {"fn": update_issue,   "description": "Update an existing issue (title, body, state, labels)",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"issue_number":{"type":"integer"},"title":{"type":"string","default":""},"body":{"type":"string","default":""},"state":{"type":"string","default":""},"labels":{"type":"array"}},"required":["owner","repo","issue_number"]}},
    "list_prs":       {"fn": list_prs,       "description": "List pull requests",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"state":{"type":"string","default":"open"},"per_page":{"type":"integer","default":20}},"required":["owner","repo"]}},
    "get_pr":         {"fn": get_pr,         "description": "Get PR details with files changed",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"pr_number":{"type":"integer"}},"required":["owner","repo","pr_number"]}},
    "create_pr":      {"fn": create_pr,      "description": "Create a pull request",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"title":{"type":"string"},"head":{"type":"string"},"base":{"type":"string","default":"main"},"body":{"type":"string","default":""},"draft":{"type":"boolean","default":False}},"required":["owner","repo","title","head"]}},
    "get_file_content":{"fn": get_file_content,"description": "Get file content from a repo",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"path":{"type":"string"},"ref":{"type":"string","default":""}},"required":["owner","repo","path"]}},
    "list_commits":   {"fn": list_commits,   "description": "List commits with optional branch/path filter",
                       "inputSchema": {"type":"object","properties":{"owner":{"type":"string"},"repo":{"type":"string"},"branch":{"type":"string","default":""},"path":{"type":"string","default":""},"per_page":{"type":"integer","default":20}},"required":["owner","repo"]}},
    "search_code":    {"fn": search_code,    "description": "Search code across GitHub",
                       "inputSchema": {"type":"object","properties":{"query":{"type":"string"},"repo":{"type":"string","default":""},"language":{"type":"string","default":""},"per_page":{"type":"integer","default":10}},"required":["query"]}},
}


def handle_message(msg):
    method=msg.get("method",""); rpc_id=msg.get("id")
    if method=="initialize": return {"jsonrpc":"2.0","id":rpc_id,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"nexus-github","version":"1.0.0"}}}
    if method=="tools/list": return {"jsonrpc":"2.0","id":rpc_id,"result":{"tools":[{"name":n,"description":s["description"],"inputSchema":s["inputSchema"]} for n,s in TOOLS.items()]}}
    if method=="tools/call":
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
