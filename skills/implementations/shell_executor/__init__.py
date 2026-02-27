"""
Shell Executor Skill
Safe, sandboxed shell command execution as a reusable skill.
Unlike ShellAgent (which plans + interprets), this skill is a direct tool:
  - Input: command string or script path
  - Output: stdout, stderr, returncode

Used by CodeAgent and ShellAgent as their execution primitive.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="shell_executor",
    description="Execute shell commands or scripts in a sandboxed subprocess. "
                "Supports timeout, working-directory control, and environment isolation.",
    version="1.0.0",
    domains=["engineering", "operations"],
    triggers=["execute", "run command", "shell", "bash", "script", "terminal"],
    requires=[],
    is_local=True,
)

# Hard-blocked patterns (immutable safety layer)
import re
_BLOCKED = [
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f\b", re.I),
    re.compile(r"\bdd\s+if=", re.I),
    re.compile(r":(){:|:&};:"),
    re.compile(r"\b(sudo|su)\s+", re.I),
    re.compile(r"\bcurl\b.*\|\s*(ba)?sh", re.I),
]


class Skill(BaseSkill):
    meta = SKILL_META

    async def run(
        self,
        command: str = "",
        script_path: str = "",
        working_dir: str = "",
        timeout: int = 30,
        env_extra: Optional[dict] = None,
        shell: str = "zsh",
        capture_output: bool = True,
        **kwargs,
    ) -> Any:
        """
        Execute a command or script file.

        Returns:
            dict with keys: stdout, stderr, returncode, duration_ms, command
        """
        if script_path:
            p = Path(script_path)
            if not p.exists():
                return {"error": f"Script not found: {script_path}", "returncode": -1}
            command = f"{shell} {script_path}"

        if not command:
            return {"error": "Either command or script_path is required", "returncode": -1}

        # Safety check
        for pattern in _BLOCKED:
            if pattern.search(command):
                return {
                    "error":      f"Command blocked by safety policy: {pattern.pattern}",
                    "returncode": -99,
                    "command":    command,
                }

        cwd = working_dir if working_dir and Path(working_dir).is_dir() else str(Path.home())

        safe_env = {
            "HOME":  os.environ.get("HOME", str(Path.home())),
            "USER":  os.environ.get("USER", ""),
            "PATH":  "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG":  "en_US.UTF-8",
            "SHELL": f"/bin/{shell}",
        }
        if env_extra:
            safe_env.update(env_extra)

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=cwd,
                env=safe_env,
                executable=f"/bin/{shell}",
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return {
                    "command":     command,
                    "stdout":      (stdout_b or b"").decode("utf-8", errors="replace")[:20000],
                    "stderr":      (stderr_b or b"").decode("utf-8", errors="replace")[:5000],
                    "returncode":  proc.returncode or 0,
                    "duration_ms": int((time.time() - start) * 1000),
                    "timed_out":   False,
                }
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "command":     command,
                    "stdout":      "",
                    "stderr":      f"Timed out after {timeout}s",
                    "returncode":  -1,
                    "duration_ms": timeout * 1000,
                    "timed_out":   True,
                }
        except Exception as exc:
            return {
                "command":     command,
                "stdout":      "",
                "stderr":      str(exc),
                "returncode":  -2,
                "duration_ms": int((time.time() - start) * 1000),
                "timed_out":   False,
            }

    async def check_tool(self, tool_name: str) -> bool:
        """Check if a CLI tool is available."""
        result = await self.run(command=f"which {tool_name}", timeout=5)
        return result.get("returncode", -1) == 0

    async def run_python_code(
        self, code: str, timeout: int = 30, working_dir: str = ""
    ) -> Any:
        """Execute Python code inline via python3 -c."""
        # Escape for shell
        escaped = code.replace("'", "'\"'\"'")
        cmd = f"{sys.executable} -c '{escaped}'"
        return await self.run(command=cmd, timeout=timeout, working_dir=working_dir)
