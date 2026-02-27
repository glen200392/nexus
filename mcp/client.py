"""
NEXUS MCP Client — Model Context Protocol over stdio
Connects to local MCP servers via subprocess, discovers tools,
and executes tool calls using JSON-RPC 2.0.

Usage:
    client = MCPClient()
    await client.connect("filesystem", ["python", "mcp/servers/filesystem_server.py"])
    tools  = await client.list_tools("filesystem")
    result = await client.call("filesystem", "read_file", {"path": "README.md"})
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("nexus.mcp.client")


@dataclass
class MCPTool:
    name:        str
    description: str
    input_schema: dict   # JSON Schema for parameters
    server:      str     # Which server provides this tool


@dataclass
class MCPServerProcess:
    name:    str
    process: asyncio.subprocess.Process
    stdin:   asyncio.StreamWriter
    stdout:  asyncio.StreamReader
    tools:   dict[str, MCPTool] = field(default_factory=dict)


class MCPClient:
    """
    Manages connections to multiple MCP servers.
    Each server runs as a subprocess (stdio transport).
    """

    def __init__(self):
        self._servers: dict[str, MCPServerProcess] = {}

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(
        self,
        server_name: str,
        command: list[str],
        env: Optional[dict] = None,
    ) -> None:
        """
        Start an MCP server subprocess and perform the initialization handshake.
        command: e.g. ["python", "mcp/servers/filesystem_server.py"]
        """
        import os
        proc_env = {**os.environ, **(env or {})}

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )

        server = MCPServerProcess(
            name=server_name,
            process=proc,
            stdin=proc.stdin,
            stdout=proc.stdout,
        )
        self._servers[server_name] = server

        # MCP initialization handshake
        await self._send(server, {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "nexus", "version": "1.0.0"},
            },
        })
        init_resp = await self._recv(server)
        logger.info(
            "MCP server '%s' initialized: %s",
            server_name,
            init_resp.get("result", {}).get("serverInfo", {}),
        )

        # Discover tools
        await self._discover_tools(server)

    async def disconnect(self, server_name: str) -> None:
        server = self._servers.pop(server_name, None)
        if server and server.process.returncode is None:
            server.process.terminate()
            await server.process.wait()

    async def disconnect_all(self) -> None:
        for name in list(self._servers):
            await self.disconnect(name)

    # ── Tool Discovery ────────────────────────────────────────────────────────

    async def _discover_tools(self, server: MCPServerProcess) -> None:
        await self._send(server, {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        })
        resp = await self._recv(server)
        tools_data = resp.get("result", {}).get("tools", [])

        for t in tools_data:
            tool = MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server=server.name,
            )
            server.tools[t["name"]] = tool

        logger.info(
            "Discovered %d tools on server '%s': %s",
            len(server.tools), server.name, list(server.tools.keys()),
        )

    # ── Tool Invocation ───────────────────────────────────────────────────────

    async def call(
        self,
        server: str,
        tool: str,
        params: dict,
    ) -> Any:
        """
        Call a tool on a connected MCP server.
        Returns the content from the tool result.
        """
        srv = self._servers.get(server)
        if srv is None:
            raise RuntimeError(f"MCP server '{server}' not connected")
        if tool not in srv.tools:
            raise ValueError(f"Tool '{tool}' not available on server '{server}'")

        rpc_id = str(uuid.uuid4())
        await self._send(srv, {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": params},
        })
        resp = await self._recv(srv)

        if "error" in resp:
            raise RuntimeError(f"MCP tool error: {resp['error']}")

        result = resp.get("result", {})
        content = result.get("content", [])

        # Extract text content from result
        texts = [c["text"] for c in content if c.get("type") == "text"]
        if texts:
            # Try to parse as JSON for structured data
            combined = "\n".join(texts)
            try:
                return json.loads(combined)
            except json.JSONDecodeError:
                return combined
        return result

    # ── Tool Schema for LLM (function calling format) ────────────────────────

    def get_tools_schema(
        self,
        server: Optional[str] = None,
        format: str = "anthropic",  # "anthropic" | "openai"
    ) -> list[dict]:
        """
        Returns tool definitions in LLM function-calling format.
        Used to inject available tools into LLM context.
        """
        tools: list[MCPTool] = []
        if server:
            srv = self._servers.get(server)
            tools = list(srv.tools.values()) if srv else []
        else:
            for srv in self._servers.values():
                tools.extend(srv.tools.values())

        if format == "anthropic":
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        else:  # openai format
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

    def list_servers(self) -> list[str]:
        return list(self._servers.keys())

    def list_tools(self, server: Optional[str] = None) -> list[MCPTool]:
        if server:
            srv = self._servers.get(server)
            return list(srv.tools.values()) if srv else []
        tools = []
        for srv in self._servers.values():
            tools.extend(srv.tools.values())
        return tools

    # ── JSON-RPC Transport ────────────────────────────────────────────────────

    async def _send(self, server: MCPServerProcess, message: dict) -> None:
        data = json.dumps(message) + "\n"
        server.stdin.write(data.encode("utf-8"))
        await server.stdin.drain()

    async def _recv(self, server: MCPServerProcess, timeout: float = 30.0) -> dict:
        try:
            line = await asyncio.wait_for(server.stdout.readline(), timeout=timeout)
            return json.loads(line.decode("utf-8").strip())
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP server '{server.name}' did not respond in {timeout}s")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MCP server returned invalid JSON: {exc}")


# ─── Global singleton ─────────────────────────────────────────────────────────
_mcp_instance: Optional[MCPClient] = None

def get_mcp_client() -> MCPClient:
    global _mcp_instance
    if _mcp_instance is None:
        _mcp_instance = MCPClient()
    return _mcp_instance
