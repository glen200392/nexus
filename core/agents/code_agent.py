"""
NEXUS Code Agent — Layer 4 Execution
Generates, executes, debugs, and refactors code.
Execution runs in a sandboxed subprocess with timeout + resource limits.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.code")


@dataclass
class ExecResult:
    stdout:      str = ""
    stderr:      str = ""
    returncode:  int = 0
    timed_out:   bool = False
    language:    str = "python"
    exec_time_ms: int = 0


class CodeAgent(BaseAgent):
    agent_id   = "code_agent"
    agent_name = "Code Generation & Execution Agent"
    description = "Writes, debugs, executes code in a sandboxed environment"
    domain     = TaskDomain.ENGINEERING
    default_complexity = TaskComplexity.HIGH

    EXEC_TIMEOUT_SECONDS = 30
    ALLOWED_LANGUAGES    = {"python", "javascript", "typescript", "bash", "sh", "sql"}

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are an expert software engineer. When writing code:\n"
            "1. Write clean, readable, production-quality code\n"
            "2. Include error handling for edge cases\n"
            "3. Add brief inline comments only where logic is non-obvious\n"
            "4. Prefer standard library over third-party when possible\n"
            "5. NEVER introduce security vulnerabilities (SQL injection, XSS, etc.)\n"
            "6. Return your response as JSON:\n"
            "{\n"
            '  "explanation": "what the code does",\n'
            '  "language": "python|javascript|bash|...",\n'
            '  "code": "the complete code",\n'
            '  "should_execute": true,\n'
            '  "expected_output": "what running it should produce",\n'
            '  "files_to_create": [{"path": "...", "content": "..."}]\n'
            "}"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        task = context.user_message

        # Pull any existing code from context (for debugging/refactoring tasks)
        existing_code = context.metadata.get("existing_code", "")

        # Step 1: Generate code
        self._logger.info("Generating code for: %s…", task[:60])
        decision = self.route_llm(context)

        user_msg = f"Task: {task}"
        if existing_code:
            user_msg += f"\n\nExisting code to modify/fix:\n```\n{existing_code}\n```"

        messages = [Message("user", user_msg)]
        llm_resp = await self._llm.chat(
            messages=messages,
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.2,   # Low temperature for code
        )

        gen = self._parse_generation(llm_resp.content)
        total_tokens = llm_resp.tokens_in + llm_resp.tokens_out
        total_cost   = llm_resp.cost_usd
        artifacts    = []

        # Step 2: Write files if requested
        for file_spec in gen.get("files_to_create", []):
            path = file_spec.get("path", "")
            content = file_spec.get("content", "")
            if path and content and self._is_safe_path(path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(content, encoding="utf-8")
                artifacts.append({"type": "file", "path": path})
                self._logger.info("Created file: %s", path)

        # Step 3: Execute if requested
        exec_result: Optional[ExecResult] = None
        if gen.get("should_execute", False) and gen.get("code"):
            lang = gen.get("language", "python").lower()
            if lang in self.ALLOWED_LANGUAGES:
                exec_result = await self._execute_code(gen["code"], lang)
                artifacts.append({
                    "type": "execution",
                    "language": lang,
                    "returncode": exec_result.returncode,
                    "stdout": exec_result.stdout[:2000],
                    "stderr": exec_result.stderr[:500],
                })

        # Step 4: If execution failed, ask LLM to fix it
        if exec_result and exec_result.returncode != 0 and not exec_result.timed_out:
            gen, llm_resp2, exec_result2 = await self._debug_loop(
                gen, exec_result, context, decision
            )
            total_tokens += llm_resp2.tokens_in + llm_resp2.tokens_out
            total_cost   += llm_resp2.cost_usd
            if exec_result2:
                exec_result = exec_result2

        # Step 5: Compose final output
        output = {
            "explanation": gen.get("explanation", ""),
            "code": gen.get("code", ""),
            "language": gen.get("language", "python"),
            "execution": {
                "ran": exec_result is not None,
                "success": exec_result.returncode == 0 if exec_result else None,
                "stdout": exec_result.stdout[:2000] if exec_result else "",
                "stderr": exec_result.stderr[:500] if exec_result else "",
                "time_ms": exec_result.exec_time_ms if exec_result else 0,
            } if exec_result else None,
        }

        # Store successful code to memory
        if exec_result is None or exec_result.returncode == 0:
            await self.remember(
                content=f"Task: {task}\nCode:\n{gen.get('code', '')}",
                context=context,
                doc_type=DocumentType.CODE_SNIPPET,
                tags=["code", gen.get("language", "python"), context.domain.value],
            )

        success = exec_result is None or exec_result.returncode == 0
        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=success,
            output=output,
            quality_score=0.8 if success else 0.3,
            tokens_used=total_tokens,
            cost_usd=total_cost,
            llm_used=decision.primary.display_name,
            artifacts=artifacts,
        )

    # ── Sandboxed Execution ───────────────────────────────────────────────────

    async def _execute_code(self, code: str, language: str) -> ExecResult:
        """Execute code in a temp directory with timeout."""
        import time

        with tempfile.TemporaryDirectory(prefix="nexus_exec_") as tmpdir:
            start = time.time()
            result = await self._run_in_sandbox(code, language, tmpdir)
            result.exec_time_ms = int((time.time() - start) * 1000)
            return result

    async def _run_in_sandbox(self, code: str, language: str, tmpdir: str) -> ExecResult:
        """Build command and run with asyncio subprocess."""
        lang = language.lower()

        if lang == "python":
            code_file = os.path.join(tmpdir, "main.py")
            Path(code_file).write_text(code, encoding="utf-8")
            cmd = [sys.executable, "-I", code_file]   # -I = isolated mode

        elif lang in ("bash", "sh"):
            code_file = os.path.join(tmpdir, "script.sh")
            Path(code_file).write_text(code, encoding="utf-8")
            os.chmod(code_file, 0o755)
            cmd = ["/bin/bash", code_file]

        elif lang in ("javascript", "typescript"):
            code_file = os.path.join(tmpdir, "main.js")
            if lang == "typescript":
                # Inline ts-node call (requires ts-node installed)
                cmd = ["npx", "ts-node", "--skip-project", "-e", code]
            else:
                Path(code_file).write_text(code, encoding="utf-8")
                cmd = ["node", code_file]

        else:
            return ExecResult(stderr=f"Language '{language}' execution not supported")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.EXEC_TIMEOUT_SECONDS,
                )
                return ExecResult(
                    stdout=stdout_b.decode("utf-8", errors="replace")[:5000],
                    stderr=stderr_b.decode("utf-8", errors="replace")[:2000],
                    returncode=proc.returncode or 0,
                    language=language,
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ExecResult(
                    stderr=f"Execution timed out after {self.EXEC_TIMEOUT_SECONDS}s",
                    returncode=-1,
                    timed_out=True,
                )
        except FileNotFoundError as exc:
            return ExecResult(stderr=f"Interpreter not found: {exc}", returncode=-2)

    # ── Debug Loop ────────────────────────────────────────────────────────────

    async def _debug_loop(
        self, gen: dict, exec_result: ExecResult, context: AgentContext, decision
    ):
        """One round of LLM self-correction based on execution error."""
        self._logger.info(
            "Execution failed (rc=%d), requesting fix…", exec_result.returncode
        )
        messages = [
            Message("user", context.user_message),
            Message("assistant", json.dumps(gen, ensure_ascii=False)),
            Message("user",
                f"The code failed with:\n"
                f"STDOUT: {exec_result.stdout[:500]}\n"
                f"STDERR: {exec_result.stderr[:500]}\n\n"
                "Please fix the code and return the corrected JSON."
            ),
        ]
        llm_resp = await self._llm.chat(
            messages=messages,
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.1,
        )
        fixed_gen = self._parse_generation(llm_resp.content)
        exec_result2 = None
        if fixed_gen.get("should_execute") and fixed_gen.get("code"):
            lang = fixed_gen.get("language", "python")
            if lang in self.ALLOWED_LANGUAGES:
                exec_result2 = await self._execute_code(fixed_gen["code"], lang)
        return fixed_gen, llm_resp, exec_result2

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_generation(self, raw: str) -> dict:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: extract code block
        code_match = re.search(r"```(?:\w+)?\n(.*?)```", raw, re.DOTALL)
        code = code_match.group(1) if code_match else raw
        return {
            "explanation": "Generated code",
            "language": "python",
            "code": code,
            "should_execute": False,
            "files_to_create": [],
        }

    @staticmethod
    def _is_safe_path(path: str) -> bool:
        """Prevent path traversal attacks."""
        resolved = Path(path).resolve()
        # Only allow writing within current directory tree
        try:
            resolved.relative_to(Path.cwd())
            return True
        except ValueError:
            logger.warning("Blocked unsafe file write path: %s", path)
            return False
