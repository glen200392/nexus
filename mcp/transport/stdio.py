"""
NEXUS v2 — Stdio MCP Transport.

Communicates with an MCP server via stdin/stdout of a subprocess,
using newline-delimited JSON-RPC messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class StdioTransport:
    """MCP transport over subprocess stdin/stdout."""

    def __init__(
        self,
        command: str | list[str],
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if isinstance(command, str):
            self._cmd = command.split()
        else:
            self._cmd = list(command)
        if args:
            self._cmd.extend(args)

        self._env: dict[str, str] = {**os.environ, **(env or {})}
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._connected = False

    # --- MCPTransport interface ---

    async def connect(self) -> None:
        """Start the subprocess."""
        if self._connected:
            return
        logger.debug("Starting stdio MCP server: %s", self._cmd)
        self._process = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            cwd=self._cwd,
        )
        self._connected = True
        logger.info("Stdio transport connected (pid=%s)", self._process.pid)

    async def send(self, data: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        if not self._connected or self._process is None or self._process.stdin is None:
            raise ConnectionError("Stdio transport is not connected")
        payload = json.dumps(data) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        """Read a JSON-RPC message from the subprocess stdout."""
        if not self._connected or self._process is None or self._process.stdout is None:
            raise ConnectionError("Stdio transport is not connected")
        line = await self._process.stdout.readline()
        if not line:
            self._connected = False
            raise ConnectionError("Stdio transport: server closed stdout")
        return json.loads(line.decode().strip())

    async def close(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._process.kill()
            finally:
                self._connected = False
                logger.info("Stdio transport closed")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.returncode is None
