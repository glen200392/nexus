# NEXUS Development Guide

> How to extend NEXUS with new Agents, Skills, and MCP Servers.

---

## Adding a New Agent

Every agent is one Python file in `core/agents/`. Follow this pattern:

### Step 1: Create the agent file

```python
# core/agents/my_agent.py
"""
My Custom Agent
Brief description of what this agent does.

Operations (context.metadata["operation"]):
  default_op  — what it does by default
  op_2        — second operation
"""
from __future__ import annotations
from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier

class MyAgent(BaseAgent):
    agent_id   = "my_agent"
    agent_name = "My Agent"
    description = "One-sentence description for the orchestrator to read."
    domain             = TaskDomain.RESEARCH       # RESEARCH | ENGINEERING | OPERATIONS | ANALYSIS | CREATIVE | ORCHESTRATION
    default_complexity = TaskComplexity.MEDIUM     # LOW | MEDIUM | HIGH | CRITICAL
    default_privacy    = PrivacyTier.INTERNAL      # PRIVATE | INTERNAL | PUBLIC

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    async def execute(self, context: AgentContext) -> AgentResult:
        operation = context.metadata.get("operation", "default_op")

        decision = self.route_llm(context)  # honors privacy tier automatically

        resp = await self._llm.chat(
            messages=[Message("user", context.user_message)],
            model=decision.primary,
            system="You are a helpful assistant.",
            privacy_tier=context.privacy_tier,
            temperature=0.5,
            max_tokens=1000,
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output={"result": resp.content},
            quality_score=0.8,
        )
```

### Step 2: Register in swarm factory

Open `core/orchestrator/swarm.py` and add to the `registry` dict inside `_build_agent()`:

```python
from nexus.core.agents.my_agent import MyAgent

registry = {
    # ... existing agents ...
    "my_agent": MyAgent,
}
```

### Step 3: Add to a swarm YAML

```yaml
# config/agents/research_swarm.yaml
agents:
  my_agent:
    description: "Does something useful"
  # ... existing agents ...
```

That's it. No other files to touch.

---

## Adding a New Skill

Skills live in `skills/implementations/<name>/__init__.py`.

### Step 1: Create the skill directory and file

```python
# skills/implementations/my_skill/__init__.py
"""
My Skill — brief description.

Operations:
  do_thing    — main operation
  list        — list available items
"""
from __future__ import annotations
from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="my_skill",
    description="Short description for agent context injection.",
    version="1.0.0",
    domains=["research", "engineering"],   # which agents can use this
    triggers=["keyword1", "keyword2"],     # phrases that suggest using this skill
)

class MySkill(BaseSkill):
    meta = SKILL_META

    async def run(
        self,
        operation: str = "do_thing",
        param1: str = "",
        param2: int = 10,
        **kwargs,
    ) -> dict:

        if operation == "do_thing":
            return self._do_thing(param1, param2)

        if operation == "list":
            return self._list()

        return {"error": f"Unknown operation: {operation}"}

    def _do_thing(self, param1: str, param2: int) -> dict:
        # Your implementation here
        return {"result": f"Did {param1} {param2} times"}

    def _list(self) -> dict:
        return {"items": []}

# Required: expose as `Skill` for auto-discovery
Skill = MySkill
```

### Step 2: Done — auto-discovered

The `SkillRegistry.load_all()` method automatically discovers any directory containing `__init__.py` with a `Skill` class. No registration required.

### Calling a skill from an agent

```python
async def execute(self, context: AgentContext) -> AgentResult:
    if self.skill_registry:
        skill = self.skill_registry.get("my_skill")
        if skill:
            result = await skill.run(
                operation="do_thing",
                param1="hello",
                param2=5,
            )
```

---

## Adding a New MCP Server

MCP servers are standalone Python scripts in `mcp/servers/`. They run as subprocesses and communicate via JSON-RPC 2.0 over stdio.

### Step 1: Create the server file

```python
# mcp/servers/my_service_server.py
"""
My Service MCP Server
Bridges NEXUS to MyExternalService.

Tools:
  my_tool_1  — description
  my_tool_2  — description

Environment variables:
  MY_SERVICE_API_KEY  — API key for MyService
"""
import json
import os
import sys

MY_API_KEY = os.environ.get("MY_SERVICE_API_KEY", "")

# ── Tool implementations ───────────────────────────────────────────────────────

def my_tool_1(param: str) -> dict:
    """Call MyService and return results."""
    # Implementation using stdlib (urllib) or requests
    return {"result": f"Called with {param}"}

def my_tool_2(items: list) -> dict:
    return {"count": len(items)}

# ── Tool registry ──────────────────────────────────────────────────────────────

TOOLS = {
    "my_tool_1": {
        "description": "Does something with MyService",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Input parameter"}
            },
            "required": ["param"]
        }
    },
    "my_tool_2": {
        "description": "Counts items",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["items"]
        }
    },
}

# ── JSON-RPC handler ───────────────────────────────────────────────────────────

def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    req_id = req.get("id")

    def ok(result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05", "serverInfo": {"name": "my_service_server", "version": "1.0.0"}, "capabilities": {"tools": {}}})

    if method == "tools/list":
        return ok({"tools": [{"name": k, **v} for k, v in TOOLS.items()]})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if name == "my_tool_1":
                result = my_tool_1(**args)
            elif name == "my_tool_2":
                result = my_tool_2(**args)
            else:
                return err(-32601, f"Unknown tool: {name}")
            return ok({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]})
        except Exception as e:
            return err(-32000, str(e))

    return err(-32601, f"Method not found: {method}")

# ── Main stdio loop ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            resp = handle_request(req)
        except json.JSONDecodeError as e:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
        print(json.dumps(resp, ensure_ascii=False), flush=True)
```

### Step 2: Register in nexus.py

```python
# nexus.py — inside init_system()
mcp_servers = {
    # ... existing servers ...
    "my_service": "mcp/servers/my_service_server.py",
}
```

### Step 3: Add env var to .env.example

```bash
# .env.example
MY_SERVICE_API_KEY=your-key-here
```

---

## Adding a New Swarm

No Python required. Create `config/agents/my_swarm.yaml`:

```yaml
swarm_id:    my_swarm
description: What this swarm does
domain:      research    # research | engineering | operations | analysis | creative

agents:
  web_agent:
    description: "Search the web"

  rag_agent:
    description: "Query knowledge base"

  critic_agent:
    description: "Quality gate"
    scoring_rubric:
      accuracy:     0.40
      completeness: 0.30
      clarity:      0.20
      usefulness:   0.10

  writer_agent:
    description: "Synthesize final output"

workflow_defaults:
  pattern:           sequential
  quality_threshold: 0.75
  timeout_seconds:   300
  max_retries:       2
```

Then update `DOMAIN_TO_SWARM` in `core/orchestrator/master.py` if you want this swarm to be auto-routed:

```python
DOMAIN_TO_SWARM = {
    # ... existing ...
    "my_domain": "my_swarm",
}
```

---

## Writing Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_my_agent.py -v

# Run with coverage
pytest tests/ --cov=nexus --cov-report=html
```

### Unit test example

```python
# tests/unit/test_my_agent.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from nexus.core.agents.my_agent import MyAgent
from nexus.core.agents.base import AgentContext
from nexus.core.llm.router import PrivacyTier, TaskComplexity, TaskDomain

@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.chat.return_value = MagicMock(content='{"result": "test"}', cost_usd=0.001)
    return client

@pytest.fixture
def agent(mock_llm):
    return MyAgent(llm_client=mock_llm)

@pytest.fixture
def context():
    return AgentContext(
        task_id="test-123",
        user_message="test input",
        privacy_tier=PrivacyTier.INTERNAL,
        complexity=TaskComplexity.MEDIUM,
        domain=TaskDomain.RESEARCH,
    )

@pytest.mark.asyncio
async def test_execute_returns_success(agent, context):
    result = await agent.execute(context)
    assert result.success is True
    assert result.agent_id == "my_agent"
    assert result.quality_score > 0
```

---

## Code Style

- Python 3.11+
- Type hints on all public functions
- `from __future__ import annotations` at top of every file
- Async all the way down (no `asyncio.run()` inside agents)
- Logging via `logger = logging.getLogger("nexus.agents.my_agent")`
- No hard-coded API keys or paths — use `os.environ.get()`
