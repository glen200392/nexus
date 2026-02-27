"""
Integration tests for NEXUS A2A (Agent-to-Agent) endpoints.
Uses FastAPI TestClient for HTTP-level testing.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus.api.a2a_endpoints import router, _task_store


@pytest.fixture(autouse=True)
def _clear_task_store():
    """Clear in-memory task store between tests."""
    _task_store.clear()
    yield
    _task_store.clear()


@pytest.fixture
def client():
    """Create a FastAPI TestClient with the A2A router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestA2AEndpoints:
    def test_agent_card(self, client: TestClient):
        """GET /.well-known/agent.json should return the agent card."""
        resp = client.get("/a2a/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "NEXUS"
        assert data["version"] == "2.0"
        assert "a2a-v1" in data["protocols"]
        assert isinstance(data["capabilities"], list)
        assert len(data["capabilities"]) > 0

    def test_send_task(self, client: TestClient):
        """POST /tasks/send should create a task and return pending status."""
        resp = client.post("/a2a/tasks/send", json={
            "message": "Research quantum computing",
            "sender_agent": "external-agent-1",
            "capabilities_required": ["research"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["task_id"]
        assert data["result"] is None
        assert data["error"] is None

    def test_send_task_with_id(self, client: TestClient):
        """POST /tasks/send with explicit task_id should use that ID."""
        resp = client.post("/a2a/tasks/send", json={
            "task_id": "custom-123",
            "message": "Test task",
            "sender_agent": "agent-x",
        })
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "custom-123"

    def test_get_task(self, client: TestClient):
        """GET /tasks/{id} should return stored task."""
        # Create task first
        create_resp = client.post("/a2a/tasks/send", json={
            "message": "Analyze data",
            "sender_agent": "agent-2",
        })
        task_id = create_resp.json()["task_id"]

        # Retrieve it
        resp = client.get(f"/a2a/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert data["status"] == "pending"

    def test_cancel_task(self, client: TestClient):
        """POST /tasks/{id}/cancel should mark task as cancelled."""
        # Create task
        create_resp = client.post("/a2a/tasks/send", json={
            "message": "Long running task",
            "sender_agent": "agent-3",
        })
        task_id = create_resp.json()["task_id"]

        # Cancel it
        resp = client.post(f"/a2a/tasks/{task_id}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

        # Verify status persists
        get_resp = client.get(f"/a2a/tasks/{task_id}")
        assert get_resp.json()["status"] == "cancelled"

    def test_unknown_task_404(self, client: TestClient):
        """GET /tasks/nonexistent should return 404."""
        resp = client.get("/a2a/tasks/nonexistent-id")
        assert resp.status_code == 404

    def test_cancel_unknown_task_404(self, client: TestClient):
        """POST /tasks/nonexistent/cancel should return 404."""
        resp = client.post("/a2a/tasks/nonexistent-id/cancel")
        assert resp.status_code == 404

    def test_known_agents(self, client: TestClient):
        """GET /known-agents should return a list (empty initially)."""
        resp = client.get("/a2a/known-agents")
        assert resp.status_code == 200
        assert resp.json() == []
