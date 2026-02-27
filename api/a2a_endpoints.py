"""
NEXUS A2A (Agent-to-Agent) Federation Endpoints
Implements the Google A2A protocol for inter-agent communication.
Uses in-memory storage for task state (real orchestrator wiring in Phase 5).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class A2ATaskRequest(BaseModel):
    """Incoming A2A task request from an external agent."""
    task_id: str | None = None
    message: str
    sender_agent: str
    capabilities_required: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATaskResponse(BaseModel):
    """Response to an A2A task request."""
    task_id: str
    status: str
    result: str | None = None
    error: str | None = None


class AgentCard(BaseModel):
    """A2A Agent Card — describes this agent's capabilities."""
    name: str = "NEXUS"
    version: str = "2.0"
    capabilities: list[str] = Field(default_factory=lambda: [
        "research",
        "coding",
        "analysis",
        "web_search",
        "file_operations",
        "data_processing",
    ])
    protocols: list[str] = Field(default_factory=lambda: ["a2a-v1"])
    endpoint: str = "/a2a"


# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------

_task_store: dict[str, dict[str, Any]] = {}
_known_agents: list[dict[str, str]] = []


def _get_task_or_404(task_id: str) -> dict[str, Any]:
    """Retrieve task from store or raise 404."""
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/a2a", tags=["a2a"])


@router.get("/.well-known/agent.json", response_model=AgentCard)
async def agent_card() -> AgentCard:
    """Return the NEXUS agent card for A2A discovery."""
    return AgentCard()


@router.post("/tasks/send", response_model=A2ATaskResponse)
async def send_task(request: A2ATaskRequest) -> A2ATaskResponse:
    """
    Accept a new task from an external agent.
    Stores the task in memory and returns a pending status.
    """
    task_id = request.task_id or uuid.uuid4().hex[:12]

    _task_store[task_id] = {
        "task_id": task_id,
        "message": request.message,
        "sender_agent": request.sender_agent,
        "capabilities_required": request.capabilities_required,
        "metadata": request.metadata,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    return A2ATaskResponse(task_id=task_id, status="pending")


@router.get("/tasks/{task_id}", response_model=A2ATaskResponse)
async def get_task(task_id: str) -> A2ATaskResponse:
    """Get the current status of a task."""
    task = _get_task_or_404(task_id)
    return A2ATaskResponse(
        task_id=task["task_id"],
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
    )


@router.post("/tasks/{task_id}/cancel", response_model=A2ATaskResponse)
async def cancel_task(task_id: str) -> A2ATaskResponse:
    """Cancel a pending or running task."""
    task = _get_task_or_404(task_id)
    task["status"] = "cancelled"
    task["updated_at"] = time.time()
    return A2ATaskResponse(
        task_id=task["task_id"],
        status="cancelled",
    )


@router.get("/known-agents")
async def known_agents() -> list[dict[str, str]]:
    """Return the list of known external agent endpoints."""
    return _known_agents
