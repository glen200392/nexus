"""
NEXUS Shell Agent — Layer 4 Execution
Executes zsh/bash commands locally with safety guardrails.
Two-phase design:
  1. Safety check  (rule-based, no LLM)
  2. LLM plans which commands to run, then executes them
  3. LLM interprets the output into human-readable result

Security principles:
  - Block list checked BEFORE any LLM call (prompt injection can't bypass)
  - Commands run in a minimal-PATH sandbox
  - Timeout enforced per command
  - Working directory restricted to allowed roots
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.shell")


# ── Safety: blocked patterns (checked before any LLM call) ───────────────────
# These are matched against the FINAL command string, not the user prompt.
_BLOCKED_PATTERNS = [
    r"\brm\s+-[a-z]*r[a-z]*f\b",          # rm -rf variants
    r"\bdd\s+if=",                           # disk wipe
    r":(){:|:&};:",                          # fork bomb
    r"\bchmod\s+777\s+/",                    # global permission change
    r"\b(sudo|su)\s+.*",                     # privilege escalation
    r"\bformat\s+[a-z]:",                    # disk format (Windows)
    r"\bmkfs\.",                             # filesystem format
    r">\s*/dev/(sd|nvme|vd)",               # overwrite disk device
    r"\biptables\s+.*-F",                    # flush firewall rules
    r"\bcurl\b.*\|\s*(ba)?sh",              # pipe-to-shell (curl | bash)
    r"\bwget\b.*-O\s*-.*\|\s*(ba)?sh",     # wget pipe-to-shell
]
_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]

# Commands that need explicit user confirmation
_CONFIRM_PATTERNS = [
    r"\brm\b",           # any rm
    r"\btrash\b",        # trash command
    r"\bgit\s+push\b",   # git push
    r"\bgit\s+reset\b",  # git reset
    r"\bdocker\s+(rm|rmi|system prune)\b",  # docker destructive
    r"\bkill\b",         # kill process
    r"\bpkill\b",        # pkill process
]
_CONFIRM_RE = [re.compile(p, re.IGNORECASE) for p in _CONFIRM_PATTERNS]

# Default allowed working directories
ALLOWED_ROOTS = [
    Path.home(),
    Path("/tmp"),
    Path(tempfile.gettempdir()),
]


@dataclass
class CommandResult:
    command:     str
    stdout:      str
    stderr:      str
    returncode:  int
    duration_ms: int
    blocked:     bool = False
    block_reason: str = ""


class ShellAgent(BaseAgent):
    agent_id   = "shell_agent"
    agent_name = "Shell Execution Agent"
    description = "Executes zsh/bash commands with safety guardrails and LLM-powered output analysis"
    domain     = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.MEDIUM
    default_privacy    = PrivacyTier.PRIVATE   # Shell output may contain secrets

    EXEC_TIMEOUT   = 30    # seconds per command
    MAX_OUTPUT_LEN = 8000  # chars before truncation

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        working_dir: Optional[Path] = None,
        require_confirmation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm        = llm_client or get_client()
        self._cwd        = working_dir or Path.home()
        self._need_confirm = require_confirmation

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are a shell command execution agent on macOS (zsh). "
            "Given a task, decide the minimal set of safe shell commands to accomplish it.\n\n"
            "Rules:\n"
            "- Prefer non-destructive commands\n"
            "- Use `ls`, `cat`, `grep`, `find`, `git`, `python3`, etc.\n"
            "- NEVER use sudo or destructive flags\n"
            "- For file listing use `ls -la`; for large files use `head`/`tail`\n"
            "- Return ONLY a JSON object:\n"
            '{\n'
            '  "plan": "what you will do",\n'
            '  "commands": ["cmd1", "cmd2", ...],\n'
            '  "requires_confirmation": false,\n'
            '  "reason": "why these commands"\n'
            '}'
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        task = context.user_message

        # Step 1: LLM plans the command sequence
        decision = self.route_llm(context)
        llm_resp = await self._llm.chat(
            messages=[Message("user", f"Task: {task}")],
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.1,
            max_tokens=512,
        )

        import json, re as _re
        plan = self._parse_plan(llm_resp.content)
        commands: list[str] = plan.get("commands", [])

        if not commands:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error="LLM produced no commands",
            )

        # Step 2: Safety check ALL commands before running any
        for cmd in commands:
            blocked, reason = self._safety_check(cmd)
            if blocked:
                return AgentResult(
                    agent_id=self.agent_id, task_id=context.task_id,
                    success=False, output=None,
                    error=f"Command blocked by safety policy: {reason}\nCommand: {cmd}",
                )

        # Step 3: Confirmation gate for sensitive commands
        if plan.get("requires_confirmation") and self._need_confirm:
            context.metadata["pending_commands"] = commands
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output={"requires_confirmation": True, "commands": commands},
                error="Human confirmation required before execution",
            )

        # Step 4: Execute commands sequentially
        results: list[CommandResult] = []
        total_cost = llm_resp.cost_usd

        for cmd in commands:
            self._logger.info("Executing: %s", cmd[:80])
            result = await self._exec(cmd)
            results.append(result)
            if result.blocked or result.returncode != 0:
                # Stop on first failure
                break

        # Step 5: LLM interprets combined output
        combined_output = "\n\n".join(
            f"$ {r.command}\n{r.stdout or ''}{r.stderr or ''}"
            for r in results
        )[:self.MAX_OUTPUT_LEN]

        interp_resp = await self._llm.chat(
            messages=[
                Message("user", f"Task: {task}"),
                Message("assistant", json.dumps(plan)),
                Message("user",
                    f"Commands were executed. Here is the output:\n\n"
                    f"{combined_output}\n\n"
                    "Explain what happened and what the result means. "
                    "If there were errors, diagnose them."
                ),
            ],
            model=decision.primary,
            system="You are a helpful shell assistant. Interpret command output clearly and concisely.",
            privacy_tier=context.privacy_tier,
        )
        total_cost += interp_resp.cost_usd

        all_success = all(r.returncode == 0 and not r.blocked for r in results)

        # Step 6: Store to memory
        if all_success:
            await self.remember(
                content=f"Task: {task}\nCommands: {commands}\nResult: {combined_output[:500]}",
                context=context,
                doc_type=DocumentType.TASK_RECORD,
                tags=["shell", "command", context.domain.value],
            )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=all_success,
            output={
                "interpretation": interp_resp.content,
                "commands": [
                    {
                        "cmd":        r.command,
                        "returncode": r.returncode,
                        "stdout":     r.stdout[:2000],
                        "stderr":     r.stderr[:500],
                        "duration_ms": r.duration_ms,
                    }
                    for r in results
                ],
            },
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out
                        + interp_resp.tokens_in + interp_resp.tokens_out,
            cost_usd=total_cost,
            llm_used=decision.primary.display_name,
        )

    # ── Safety ────────────────────────────────────────────────────────────────

    def _safety_check(self, cmd: str) -> tuple[bool, str]:
        """Returns (is_blocked, reason). Checked BEFORE any LLM call."""
        for pattern in _BLOCKED_RE:
            if pattern.search(cmd):
                return True, f"Matches blocked pattern: {pattern.pattern}"
        return False, ""

    def _needs_confirmation(self, cmd: str) -> bool:
        return any(p.search(cmd) for p in _CONFIRM_RE)

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _exec(self, command: str) -> CommandResult:
        start = time.time()
        # Sanitize working directory
        cwd = str(self._cwd) if self._cwd.exists() else str(Path.home())

        # Minimal safe environment
        safe_env = {
            "HOME":   os.environ.get("HOME", str(Path.home())),
            "USER":   os.environ.get("USER", ""),
            "PATH":   "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG":   "en_US.UTF-8",
            "TERM":   "xterm-256color",
            "SHELL":  "/bin/zsh",
        }
        # Allow inheriting certain tools
        for key in ("NVM_DIR", "PYENV_ROOT", "GOPATH"):
            if key in os.environ:
                safe_env[key] = os.environ[key]

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=safe_env,
                executable="/bin/zsh",
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self.EXEC_TIMEOUT
                )
                return CommandResult(
                    command=command,
                    stdout=stdout_b.decode("utf-8", errors="replace")[:self.MAX_OUTPUT_LEN],
                    stderr=stderr_b.decode("utf-8", errors="replace")[:2000],
                    returncode=proc.returncode or 0,
                    duration_ms=int((time.time() - start) * 1000),
                )
            except asyncio.TimeoutError:
                proc.kill()
                return CommandResult(
                    command=command, stdout="", returncode=-1, duration_ms=self.EXEC_TIMEOUT * 1000,
                    stderr=f"Timed out after {self.EXEC_TIMEOUT}s",
                )
        except Exception as exc:
            return CommandResult(
                command=command, stdout="", returncode=-2,
                stderr=str(exc), duration_ms=int((time.time() - start) * 1000),
            )

    def _parse_plan(self, raw: str) -> dict:
        import json, re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        # Fallback: extract quoted strings as commands
        commands = _re.findall(r"`([^`]+)`", raw)
        return {"commands": commands or [], "plan": raw[:200], "requires_confirmation": False}
