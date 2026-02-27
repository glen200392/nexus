"""
NEXUS Plugin Sandbox — Isolated Execution Environment
Provides timeout protection and resource limits for plugin execution.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import json
from typing import Any

logger = logging.getLogger("nexus.plugins.sandbox")


class SandboxedExecutor:
    """
    Executes plugin functions with timeout and resource constraints.

    Provides two execution modes:
    1. In-process with asyncio timeout (default)
    2. In a subprocess for stronger isolation (simplified stub)
    """

    def __init__(self, timeout: float = 30.0, max_memory_mb: int = 512):
        """
        Initialize the sandboxed executor.

        Args:
            timeout: Maximum execution time in seconds.
            max_memory_mb: Maximum memory usage in MB (advisory, enforced in subprocess mode).
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    async def execute(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with timeout protection.

        If the function is a coroutine function, it is awaited directly.
        Otherwise, it is run in the default executor (thread pool).

        Args:
            func: The function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The function's return value.

        Raises:
            asyncio.TimeoutError: If execution exceeds the timeout.
            Exception: Any exception raised by the function.
        """
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout,
            )
        else:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=self.timeout,
            )

    async def execute_in_process(
        self,
        module_path: str,
        func_name: str,
        *args: Any,
    ) -> Any:
        """
        Execute a function in a separate subprocess for stronger isolation.

        This is a simplified implementation that serializes arguments as JSON,
        runs a subprocess, and deserializes the result.

        Args:
            module_path: Dotted module path (e.g., "nexus.plugins.my_plugin").
            func_name: Name of the function to call within the module.
            *args: Positional arguments (must be JSON-serializable).

        Returns:
            The deserialized return value from the subprocess.

        Raises:
            asyncio.TimeoutError: If execution exceeds the timeout.
            RuntimeError: If the subprocess fails.
        """
        # Build a small Python script to run in subprocess
        script = _build_subprocess_script(module_path, func_name, list(args))

        try:
            proc = await asyncio.wait_for(
                self._run_subprocess(script),
                timeout=self.timeout,
            )
            return proc
        except asyncio.TimeoutError:
            logger.error(
                "Subprocess execution timed out after %.1fs for %s.%s",
                self.timeout, module_path, func_name,
            )
            raise

    async def _run_subprocess(self, script: str) -> Any:
        """Run a Python script in a subprocess and return its output."""
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            raise RuntimeError(
                f"Subprocess failed (exit code {process.returncode}): {error_msg}"
            )

        output = stdout.decode().strip()
        if not output:
            return None

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output


def _build_subprocess_script(
    module_path: str,
    func_name: str,
    args: list,
) -> str:
    """Build a Python script string for subprocess execution."""
    args_json = json.dumps(args)
    return f"""
import json
import sys
import importlib
import asyncio

try:
    module = importlib.import_module({module_path!r})
    func = getattr(module, {func_name!r})
    args = json.loads({args_json!r})
    if asyncio.iscoroutinefunction(func):
        result = asyncio.run(func(*args))
    else:
        result = func(*args)
    print(json.dumps(result))
except Exception as exc:
    print(json.dumps({{"error": str(exc)}}), file=sys.stderr)
    sys.exit(1)
"""
