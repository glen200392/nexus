"""
NEXUS v2 — MCP Client v2.

Manages connections to multiple MCP servers with automatic transport
selection (stdio or HTTP) based on server configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from nexus.mcp.transport.base import MCPTransport
from nexus.mcp.transport.stdio import StdioTransport
from nexus.mcp.transport.http import HTTPTransport

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/mcp/servers.yaml")


class MCPClientV2:
    """Multi-server MCP client with automatic transport selection."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._server_configs: dict[str, dict[str, Any]] = {}
        self._transports: dict[str, MCPTransport] = {}
        self._tool_cache: dict[str, list[dict[str, Any]]] = {}

        self._load_config()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load server configurations from the YAML file."""
        if not self._config_path.exists():
            logger.warning("MCP config not found at %s", self._config_path)
            return
        with open(self._config_path) as f:
            raw = yaml.safe_load(f) or {}
        servers = raw.get("servers", {})
        if isinstance(servers, dict):
            self._server_configs = servers
        elif isinstance(servers, list):
            for entry in servers:
                name = entry.get("name", entry.get("id", ""))
                if name:
                    self._server_configs[name] = entry
        logger.info("Loaded %d MCP server configs", len(self._server_configs))

    # ------------------------------------------------------------------
    # Transport factory
    # ------------------------------------------------------------------

    def _create_transport(self, config: dict[str, Any]) -> MCPTransport:
        """Create the appropriate transport based on config."""
        transport_type = config.get("transport", "stdio")

        if transport_type == "http" or transport_type == "sse":
            url = config.get("url", "")
            headers = config.get("headers", {})
            timeout = config.get("timeout", 30.0)
            return HTTPTransport(url=url, headers=headers, timeout=timeout)

        # Default: stdio
        command = config.get("command", "")
        args = config.get("args", [])
        env = config.get("env", {})
        cwd = config.get("cwd")
        return StdioTransport(command=command, args=args, env=env, cwd=cwd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def server_names(self) -> list[str]:
        """Return names of all configured servers."""
        return list(self._server_configs.keys())

    async def connect(self, server_name: str) -> None:
        """Connect to a named MCP server."""
        if server_name in self._transports:
            transport = self._transports[server_name]
            if transport.is_connected:
                logger.debug("Already connected to %s", server_name)
                return

        config = self._server_configs.get(server_name)
        if config is None:
            raise ValueError(f"Unknown MCP server: {server_name}")

        transport = self._create_transport(config)
        await transport.connect()
        self._transports[server_name] = transport
        logger.info("Connected to MCP server: %s", server_name)

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from a named MCP server."""
        transport = self._transports.pop(server_name, None)
        if transport is not None:
            await transport.close()
            logger.info("Disconnected from MCP server: %s", server_name)
        self._tool_cache.pop(server_name, None)

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        for name in list(self._transports.keys()):
            await self.disconnect(name)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool on the specified MCP server.

        Automatically connects if not already connected.
        """
        if server_name not in self._transports or not self._transports[server_name].is_connected:
            await self.connect(server_name)

        transport = self._transports[server_name]

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params or {},
            },
        }

        await transport.send(request)
        response = await transport.receive()

        if "error" in response:
            raise RuntimeError(
                f"MCP tool error ({server_name}/{tool_name}): "
                f"{response['error'].get('message', response['error'])}"
            )

        return response.get("result", {})

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List available tools on the specified MCP server.

        Results are cached per server until disconnect.
        """
        if server_name in self._tool_cache:
            return self._tool_cache[server_name]

        if server_name not in self._transports or not self._transports[server_name].is_connected:
            await self.connect(server_name)

        transport = self._transports[server_name]

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }

        await transport.send(request)
        response = await transport.receive()

        tools = response.get("result", {}).get("tools", [])
        self._tool_cache[server_name] = tools
        return tools

    async def get_server_status(self, server_name: str) -> dict[str, Any]:
        """Return the connection status of a server."""
        transport = self._transports.get(server_name)
        return {
            "name": server_name,
            "configured": server_name in self._server_configs,
            "connected": transport is not None and transport.is_connected,
            "transport_type": self._server_configs.get(server_name, {}).get("transport", "stdio"),
            "cached_tools": len(self._tool_cache.get(server_name, [])),
        }

    async def get_all_status(self) -> list[dict[str, Any]]:
        """Return connection status for all configured servers."""
        return [await self.get_server_status(name) for name in self._server_configs]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    _id_counter: int = 0

    def _next_id(self) -> int:
        MCPClientV2._id_counter += 1
        return MCPClientV2._id_counter
