"""
Tests for NEXUS Task Management API.
Uses FastAPI TestClient for HTTP-level testing.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus.api.task_api import router


@pytest.fixture
def client():
    """Create a FastAPI TestClient with the task API router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestTaskLifecycle:
    def test_pause_task(self, client: TestClient):
        """POST /task/{id}/pause should return paused status."""
        resp = client.post("/api/task/task-123/pause")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task-123"
        assert data["status"] == "paused"

    def test_resume_task(self, client: TestClient):
        """POST /task/{id}/resume should return running status."""
        resp = client.post("/api/task/task-456/resume")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task-456"
        assert data["status"] == "running"

    def test_cancel_task(self, client: TestClient):
        """POST /task/{id}/cancel should return cancelled status."""
        resp = client.post("/api/task/task-789/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task-789"
        assert data["status"] == "cancelled"

    def test_get_checkpoints(self, client: TestClient):
        """GET /task/{id}/checkpoints should return empty list."""
        resp = client.get("/api/task/task-123/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task-123"
        assert data["checkpoints"] == []

    def test_fork_execution(self, client: TestClient):
        """POST /task/{id}/fork/{step} should return fork confirmation."""
        resp = client.post("/api/task/task-123/fork/step-5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task-123"
        assert data["forked_from_step"] == "step-5"
        assert data["status"] == "forked"


class TestLLMEndpoints:
    def test_llm_health(self, client: TestClient):
        """GET /llm/health should return health status."""
        resp = client.get("/api/llm/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "circuit_breakers" in data
        assert "openai" in data["circuit_breakers"]

    def test_cache_stats(self, client: TestClient):
        """GET /llm/cache/stats should return cache statistics."""
        resp = client.get("/api/llm/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_entries" in data
        assert "hit_rate" in data
        assert data["hit_rate"] == 0.0


class TestMemoryEndpoints:
    def test_memory_consolidate(self, client: TestClient):
        """POST /memory/consolidate should return success."""
        resp = client.post("/api/memory/consolidate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "entries_processed" in data
