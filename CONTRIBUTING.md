# Contributing to NEXUS

Thank you for contributing to NEXUS. This guide covers the contribution process, code standards, and how to get your changes merged.

---

## Getting Started

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/nexus.git
cd nexus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Configure for your local environment
python nexus.py init
```

---

## Contribution Types

### Bug Fixes
- Open an issue first describing the bug
- Reference the issue in your PR: `Fixes #123`
- Include a test that reproduces the bug (and passes after your fix)

### New Agents
- Follow the agent pattern in [DEVELOPMENT.md](DEVELOPMENT.md)
- Every new agent must have:
  - Unit tests with mock LLM
  - Registration in `swarm.py` factory
  - Entry in [docs/agents/README.md](docs/agents/README.md)
  - At least one swarm YAML that uses it

### New Skills
- Follow the skill pattern in [DEVELOPMENT.md](DEVELOPMENT.md)
- Every new skill must have:
  - Unit tests for each operation
  - `SKILL_META` with accurate `triggers` list
  - Entry in [docs/skills/README.md](docs/skills/README.md)

### New MCP Servers
- Follow the MCP pattern in [DEVELOPMENT.md](DEVELOPMENT.md)
- Every new server must:
  - Use stdlib only (no new dependencies) unless absolutely necessary
  - Have graceful failure (exception → error response, not crash)
  - Be registered in `nexus.py`
  - Have entry in [docs/mcp/README.md](docs/mcp/README.md)

---

## Pull Request Process

1. **Branch naming**: `feat/agent-sql`, `fix/swarm-factory-crash`, `docs/deployment-guide`
2. **Commit messages**: imperative mood, present tense: `Add sql_agent`, not `Added sql_agent`
3. **Tests**: All new code must have tests. `pytest tests/ -v` must pass
4. **Documentation**: Update relevant docs (agents, skills, MCP reference)
5. **PR description**: Describe what changed and why, not how

---

## Code Standards

- Python 3.11+
- `from __future__ import annotations` at top of every file
- Type hints on all public methods
- `logging.getLogger("nexus.module.submodule")` for logging
- No hard-coded secrets — use `os.environ.get()`
- Async all the way — no `asyncio.run()` inside agents
- Privacy tier must be respected — never use cloud models for PRIVATE tasks

---

## Security Policy

- Never commit API keys, passwords, or personal data
- PII in examples must be clearly synthetic (`test@example.com`, not real emails)
- New MCP servers must not log sensitive parameters
- Report security vulnerabilities privately via GitHub Security Advisories

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
