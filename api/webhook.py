"""
NEXUS Webhook Trigger — Layer 1 HTTP ingestion
FastAPI server that receives events from:
  - Git hooks (post-commit, post-push)
  - n8n / Zapier / Make.com automation
  - CI/CD pipelines
  - Custom HTTP integrations
  - Scheduled pings

Run:
    uvicorn nexus.api.webhook:app --host 0.0.0.0 --port 7700 --reload

Git hook setup (post-commit):
    echo 'curl -s -X POST http://localhost:7700/hooks/git \
      -H "Content-Type: application/json" \
      -d "{\"event\":\"post-commit\",\"repo\":\"$(pwd)\",\"message\":\"$(git log -1 --format=%s)\"}"' \
    >> .git/hooks/post-commit
    chmod +x .git/hooks/post-commit
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, Header, Depends, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger("nexus.webhook")

# Must be set externally; prevents unauthorized webhook calls
WEBHOOK_SECRET = os.getenv("NEXUS_WEBHOOK_SECRET", "")

# Global event queue — injected at startup by nexus.py
_event_queue = None

def set_event_queue(queue):
    global _event_queue
    _event_queue = queue


# ── Pydantic Models ───────────────────────────────────────────────────────────

class WebhookPayload(BaseModel):
    event:    str            # "post-commit" | "n8n.trigger" | "custom" | ...
    source:   str = "webhook"
    prompt:   str = ""       # Direct task prompt (if known)
    data:     dict = {}      # Raw event data
    priority: str = "NORMAL" # CRITICAL | HIGH | NORMAL | LOW
    domain:   Optional[str] = None
    privacy:  Optional[str] = None
    attachments: list[str] = []


class GitHookPayload(BaseModel):
    event:   str   # post-commit | post-push | post-merge
    repo:    str   # repository path
    branch:  str = "main"
    message: str = ""
    files_changed: list[str] = []
    author:  str = ""


# ── Signature Verification ────────────────────────────────────────────────────

def _verify_signature(body: bytes, signature: str) -> bool:
    """HMAC-SHA256 verification for trusted sources."""
    if not WEBHOOK_SECRET:
        return True  # No secret set = accept all (dev mode)
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature or "")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NEXUS Webhook Receiver", version="1.0.0")


async def _emit_event(
    prompt: str,
    source_name: str,
    priority: str,
    domain: Optional[str],
    privacy: Optional[str],
    payload: dict,
    attachments: list[str] = [],
) -> str:
    """Push a TaskEvent to the NEXUS event queue."""
    if _event_queue is None:
        raise RuntimeError("Event queue not initialized")

    from nexus.core.orchestrator.trigger import (
        TaskEvent, TriggerSource, TriggerPriority
    )
    event = TaskEvent(
        source=TriggerSource.WEBHOOK,
        priority=TriggerPriority[priority],
        raw_input=prompt,
        payload=payload,
        attachments=attachments,
        hint_domain=domain,
        hint_privacy=privacy,
    )
    await _event_queue.put(event)
    logger.info("Webhook event queued: %s [%s]", event.event_id[:8], source_name)
    return event.event_id


# ── Generic webhook endpoint ──────────────────────────────────────────────────

@app.post("/hooks/generic")
async def generic_hook(
    payload: WebhookPayload,
    background: BackgroundTasks,
    x_nexus_signature: str = Header(default=""),
    request: Request = None,
):
    """Generic webhook. Any system can POST here with a task prompt."""
    body = await request.body() if request else b""
    if WEBHOOK_SECRET and not _verify_signature(body, x_nexus_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event_id = await _emit_event(
        prompt=payload.prompt or json.dumps(payload.data),
        source_name=payload.source,
        priority=payload.priority,
        domain=payload.domain,
        privacy=payload.privacy,
        payload=payload.dict(),
        attachments=payload.attachments,
    )
    return {"accepted": True, "event_id": event_id}


# ── Git hook endpoint ─────────────────────────────────────────────────────────

@app.post("/hooks/git")
async def git_hook(payload: GitHookPayload):
    """
    Receives git lifecycle events.
    Auto-generates prompts based on event type.
    """
    prompts = {
        "post-commit": (
            f"A new commit was made to {payload.repo} on branch {payload.branch}.\n"
            f"Commit message: {payload.message}\n"
            f"Files changed: {', '.join(payload.files_changed[:10])}\n"
            "Review this commit for potential issues, documentation needs, "
            "or follow-up tasks."
        ),
        "post-push": (
            f"Code was pushed to {payload.repo} ({payload.branch}).\n"
            "Check if CI should be triggered and if any documentation needs updating."
        ),
        "post-merge": (
            f"Branch merged into {payload.branch} in {payload.repo}.\n"
            f"Merge commit: {payload.message}\n"
            "Summarize what changed and update the project knowledge base."
        ),
    }
    prompt = prompts.get(payload.event, f"Git event: {payload.event} in {payload.repo}")

    event_id = await _emit_event(
        prompt=prompt,
        source_name="git",
        priority="LOW",
        domain="engineering",
        privacy="PRIVATE",   # Git repos may contain sensitive code
        payload=payload.dict(),
    )
    return {"accepted": True, "event_id": event_id, "prompt": prompt[:100]}


# ── n8n / Zapier / Make.com ───────────────────────────────────────────────────

@app.post("/hooks/automation")
async def automation_hook(request: Request):
    """
    Flexible endpoint for automation platforms.
    Expects JSON with at least a "prompt" or "message" field.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    prompt = (
        data.get("prompt") or
        data.get("message") or
        data.get("text") or
        json.dumps(data)[:500]
    )

    event_id = await _emit_event(
        prompt=prompt,
        source_name="automation",
        priority=data.get("priority", "NORMAL"),
        domain=data.get("domain"),
        privacy=data.get("privacy"),
        payload=data,
    )
    return {"accepted": True, "event_id": event_id}


# ── Health / Status ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    queue_size = _event_queue.qsize() if _event_queue else -1
    return {
        "status": "ok",
        "queue_size": queue_size,
        "webhook_secret_set": bool(WEBHOOK_SECRET),
        "timestamp": time.time(),
    }


@app.get("/")
async def root():
    return {
        "service": "NEXUS Webhook Receiver",
        "endpoints": [
            "POST /hooks/generic  — generic task webhook",
            "POST /hooks/git      — git lifecycle hooks",
            "POST /hooks/automation — n8n/Zapier/Make",
            "GET  /health         — health check",
        ],
    }
