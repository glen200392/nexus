"""
NEXUS v2 — MCP Transport base protocol.

Defines the interface that all MCP transports must implement.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MCPTransport(Protocol):
    """Protocol for MCP server transports (stdio, HTTP/SSE, etc.)."""

    async def connect(self) -> None:
        """Establish the transport connection."""
        ...

    async def send(self, data: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the server."""
        ...

    async def receive(self) -> dict[str, Any]:
        """Receive the next JSON-RPC message from the server."""
        ...

    async def close(self) -> None:
        """Close the transport connection and release resources."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True if the transport is currently connected."""
        ...
