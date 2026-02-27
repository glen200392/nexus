"""
NEXUS A2A Agent — Agent-to-Agent Protocol
Implements Google's A2A (Agent-to-Agent) open protocol so NEXUS can:
  1. Expose its own AgentCard (capability manifest)
  2. Discover remote A2A-compliant agents
  3. Delegate subtasks to remote agents
  4. Stream results back and integrate them into NEXUS workflows

A2A Protocol Reference: https://google.github.io/A2A/

AgentCard is served at: GET <NEXUS_BASE_URL>/.well-known/agent.json
Task lifecycle: tasks/send → tasks/get → tasks/cancel

Operations (context.metadata["operation"]):
  delegate   — send a task to a remote agent and await result
  discover   — fetch and cache the AgentCard of a remote agent
  list       — list all known remote agents
  broadcast  — send the same task to multiple agents, merge results

Environment:
  NEXUS_BASE_URL   — Public URL of this NEXUS instance (for AgentCard)
  A2A_AGENT_URLS   — Comma-separated list of known remote agent base URLs
  A2A_API_KEY      — Bearer token for authenticating outbound A2A calls
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.a2a")

NEXUS_BASE_URL = os.environ.get("NEXUS_BASE_URL", "http://localhost:8080")
A2A_AGENT_URLS = [u.strip() for u in os.environ.get("A2A_AGENT_URLS", "").split(",") if u.strip()]
A2A_API_KEY    = os.environ.get("A2A_API_KEY", "")


# ── A2A Data Structures ────────────────────────────────────────────────────────

@dataclass
class AgentCard:
    """A2A-spec capability manifest for an agent."""
    name:         str
    url:          str
    version:      str = "1.0"
    description:  str = ""
    capabilities: dict = field(default_factory=dict)
    skills:       list[dict] = field(default_factory=list)
    auth_schemes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "url":          self.url,
            "version":      self.version,
            "description":  self.description,
            "capabilities": self.capabilities,
            "skills":       self.skills,
            "authentication": {"schemes": self.auth_schemes},
        }


# ── NEXUS own AgentCard ────────────────────────────────────────────────────────

def build_nexus_agent_card() -> AgentCard:
    """Build the AgentCard that NEXUS publishes about itself."""
    return AgentCard(
        name="NEXUS",
        url=NEXUS_BASE_URL,
        version="1.0",
        description=(
            "NEXUS — Enterprise AI Agent Management Platform. "
            "Multi-agent orchestration with RAG, memory consolidation, "
            "browser automation, code execution, and data analysis."
        ),
        capabilities={
            "streaming":    True,
            "push_notifications": False,
            "state_transition_history": True,
        },
        skills=[
            {"id": "research",    "name": "Research",         "description": "Web search + RAG retrieval"},
            {"id": "code",        "name": "Code Generation",  "description": "Write, review, and execute code"},
            {"id": "data",        "name": "Data Analysis",    "description": "CSV/Excel analysis with charts"},
            {"id": "browser",     "name": "Browser Control",  "description": "Web scraping and automation"},
            {"id": "planning",    "name": "Task Planning",    "description": "DAG-based task decomposition"},
            {"id": "writing",     "name": "Writing",          "description": "Long-form document generation"},
            {"id": "memory",      "name": "Memory",           "description": "Knowledge consolidation and recall"},
        ],
        auth_schemes=["Bearer"],
    )


# ── Remote Agent Client ────────────────────────────────────────────────────────

class A2AClient:
    """HTTP client for the A2A protocol."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self._key     = api_key or A2A_API_KEY

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode()
            return {"error": f"HTTP {exc.code}: {body_text[:300]}"}
        except Exception as exc:
            return {"error": str(exc)}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        qs  = ("?" + urllib.parse.urlencode(params)) if params else ""
        req = urllib.request.Request(
            f"{self.base_url}{path}{qs}",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            return {"error": str(exc)}

    def get_agent_card(self) -> dict:
        """Fetch /.well-known/agent.json from the remote agent."""
        return self._get("/.well-known/agent.json")

    def send_task(
        self,
        message: str,
        skill_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Submit a task via tasks/send."""
        payload = {
            "id": task_id or str(uuid.uuid4()),
            "message": {
                "role":    "user",
                "parts":   [{"type": "text", "text": message}],
            },
        }
        if skill_id:
            payload["skill_id"] = skill_id
        if metadata:
            payload["metadata"] = metadata
        return self._post("/tasks/send", payload)

    def get_task(self, task_id: str) -> dict:
        """Poll task status via tasks/get."""
        return self._post("/tasks/get", {"id": task_id})

    def cancel_task(self, task_id: str) -> dict:
        return self._post("/tasks/cancel", {"id": task_id})

    def poll_until_done(
        self,
        task_id: str,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
    ) -> dict:
        """Poll tasks/get until state is completed/failed/canceled."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_task(task_id)
            if "error" in result:
                return result
            state = result.get("status", {}).get("state", "")
            if state in ("completed", "failed", "canceled"):
                return result
            time.sleep(poll_interval)
        return {"error": f"Task {task_id} timed out after {timeout}s"}


# ── A2A Agent ─────────────────────────────────────────────────────────────────

class A2AAgent(BaseAgent):
    agent_id   = "a2a_agent"
    agent_name = "A2A Collaboration Agent"
    description = (
        "Implements Google's Agent-to-Agent (A2A) protocol. "
        "Discovers remote A2A-compatible agents, delegates subtasks to them, "
        "and merges their results into NEXUS workflows. "
        "Also publishes NEXUS's own AgentCard for external discovery."
    )
    domain             = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.MEDIUM
    default_privacy    = PrivacyTier.INTERNAL

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm     = llm_client or get_client()
        # Local cache: base_url → AgentCard dict
        self._known_agents: dict[str, dict] = {}
        # Pre-seed from env
        for url in A2A_AGENT_URLS:
            self._known_agents[url] = {}

    async def execute(self, context: AgentContext) -> AgentResult:
        operation = context.metadata.get("operation", "delegate")

        if operation == "list":
            return self._list_agents()

        if operation == "discover":
            url = context.metadata.get("agent_url", "")
            if not url:
                return AgentResult(
                    agent_id=self.agent_id, task_id=context.task_id,
                    success=False, output=None,
                    error="agent_url required for discover operation",
                )
            return self._discover(url, context)

        if operation == "broadcast":
            return await self._broadcast(context)

        # Default: delegate to a specific remote agent
        return await self._delegate(context)

    # ── Operations ────────────────────────────────────────────────────────────

    def _list_agents(self) -> AgentResult:
        """Return all known remote agents and their capabilities."""
        agents = []
        for url, card in self._known_agents.items():
            agents.append({
                "url":         url,
                "name":        card.get("name", "(unknown)"),
                "description": card.get("description", ""),
                "skills":      [s.get("id") for s in card.get("skills", [])],
                "discovered":  bool(card),
            })
        # Also include self
        self_card = build_nexus_agent_card().to_dict()
        agents.insert(0, {
            "url":         NEXUS_BASE_URL,
            "name":        self_card["name"],
            "description": self_card["description"],
            "skills":      [s["id"] for s in self_card["skills"]],
            "is_self":     True,
        })
        return AgentResult(
            agent_id=self.agent_id, task_id="",
            success=True,
            output={"agents": agents, "count": len(agents)},
        )

    def _discover(self, url: str, context: AgentContext) -> AgentResult:
        """Fetch and cache the AgentCard of a remote agent."""
        client = A2AClient(url)
        card   = client.get_agent_card()
        if "error" in card:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error=f"Discovery failed for {url}: {card['error']}",
            )
        self._known_agents[url] = card
        logger.info("Discovered A2A agent at %s: %s", url, card.get("name"))
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True,
            output={"discovered": card, "url": url},
            quality_score=0.9,
        )

    async def _delegate(self, context: AgentContext) -> AgentResult:
        """Delegate a task to a single remote agent and return its result."""
        agent_url = context.metadata.get("agent_url", "")
        if not agent_url and self._known_agents:
            agent_url = next(iter(self._known_agents))
        if not agent_url:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error="No remote agent URL provided. Set agent_url in metadata or A2A_AGENT_URLS env.",
            )

        skill_id = context.metadata.get("skill_id")
        timeout  = float(context.metadata.get("timeout", 120))
        client   = A2AClient(agent_url)

        logger.info("A2A delegate → %s (skill=%s)", agent_url, skill_id)
        send_resp = client.send_task(
            message=context.user_message,
            skill_id=skill_id,
            task_id=context.task_id,
            metadata={"source": "nexus", "nexus_task_id": context.task_id},
        )
        if "error" in send_resp:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error=f"tasks/send failed: {send_resp['error']}",
            )

        remote_task_id = send_resp.get("id", "")
        state          = send_resp.get("status", {}).get("state", "")

        # If not immediately done, poll
        if state not in ("completed", "failed", "canceled"):
            final = client.poll_until_done(remote_task_id, timeout=timeout)
        else:
            final = send_resp

        success = final.get("status", {}).get("state") == "completed"
        output  = self._extract_output(final)

        # Store the result in memory
        await self.remember(
            content=(
                f"A2A delegation to {agent_url}\n"
                f"Task: {context.user_message[:200]}\n"
                f"Result: {json.dumps(output)[:400]}"
            ),
            context=context,
            doc_type=DocumentType.EPISODIC,
            tags=["a2a", "delegation", agent_url],
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=success,
            output={
                "remote_task_id": remote_task_id,
                "remote_agent":   agent_url,
                "state":          final.get("status", {}).get("state"),
                "result":         output,
            },
            quality_score=0.8 if success else 0.0,
            error=None if success else final.get("status", {}).get("message", ""),
        )

    async def _broadcast(self, context: AgentContext) -> AgentResult:
        """Send the same task to all known agents and merge results."""
        if not self._known_agents:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error="No remote agents registered. Discover some first.",
            )

        all_results = []
        for url in list(self._known_agents.keys()):
            client  = A2AClient(url)
            task_id = str(uuid.uuid4())
            resp    = client.send_task(context.user_message, task_id=task_id)
            if "error" not in resp:
                final   = client.poll_until_done(task_id, timeout=60)
                output  = self._extract_output(final)
                all_results.append({
                    "agent":  url,
                    "state":  final.get("status", {}).get("state"),
                    "output": output,
                })
            else:
                all_results.append({"agent": url, "error": resp["error"]})

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=any(r.get("state") == "completed" for r in all_results),
            output={"broadcast_results": all_results, "agent_count": len(all_results)},
            quality_score=0.75,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_output(self, task_result: dict) -> Any:
        """
        Extract meaningful text/data from an A2A task result.
        A2A spec: result is in artifacts[].parts or status.message.
        """
        artifacts = task_result.get("artifacts", [])
        texts     = []
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "data":
                    texts.append(json.dumps(part.get("data", {})))
        if texts:
            return "\n\n".join(texts)

        # Fallback to status message
        return task_result.get("status", {}).get("message", "")
