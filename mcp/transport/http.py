"""
NEXUS v2 — HTTP/SSE MCP Transport.

Communicates with an MCP server via HTTP POST (JSON-RPC) and
Server-Sent Events for streaming responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HTTPTransport:
    """MCP transport over HTTP with SSE support."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }
        self._timeout = timeout
        self._client: Any = None  # httpx.AsyncClient
        self._connected = False
        self._sse_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sse_task: asyncio.Task | None = None

    # --- MCPTransport interface ---

    async def connect(self) -> None:
        """Initialise the HTTP client and verify the server is reachable."""
        if self._connected:
            return
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx is required for HTTP transport: pip install httpx")

        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers=self._headers,
            timeout=self._timeout,
        )
        # Optionally ping the server
        try:
            resp = await self._client.get("/health")
            logger.debug("HTTP transport health check: %s", resp.status_code)
        except Exception:
            logger.debug("No /health endpoint; assuming server is ready")

        self._connected = True
        logger.info("HTTP transport connected to %s", self._url)

    async def send(self, data: dict[str, Any]) -> None:
        """Send a JSON-RPC request via HTTP POST.

        If the response is an SSE stream, messages are queued for receive().
        """
        if not self._connected or self._client is None:
            raise ConnectionError("HTTP transport is not connected")

        resp = await self._client.post("/rpc", json=data)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Parse SSE events from the body
            for event in self._parse_sse(resp.text):
                await self._sse_queue.put(event)
        else:
            await self._sse_queue.put(resp.json())

    async def receive(self) -> dict[str, Any]:
        """Return the next queued message."""
        if not self._connected:
            raise ConnectionError("HTTP transport is not connected")
        return await self._sse_queue.get()

    async def close(self) -> None:
        """Shut down the HTTP client."""
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        self._connected = False
        logger.info("HTTP transport closed")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --- SSE helpers ---

    async def subscribe_sse(self, path: str = "/events") -> None:
        """Start an SSE listener in the background."""
        if self._sse_task and not self._sse_task.done():
            return
        self._sse_task = asyncio.create_task(self._listen_sse(path))

    async def _listen_sse(self, path: str) -> None:
        """Long-running SSE listener."""
        if self._client is None:
            return
        try:
            import httpx
            async with self._client.stream("GET", path) as resp:
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        raw_event, buffer = buffer.split("\n\n", 1)
                        event = self._parse_single_sse(raw_event)
                        if event:
                            await self._sse_queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("SSE listener error: %s", exc)

    @staticmethod
    def _parse_sse(text: str) -> list[dict[str, Any]]:
        """Parse SSE-formatted text into a list of JSON objects."""
        events: list[dict[str, Any]] = []
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            data_lines: list[str] = []
            for line in block.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    events.append(json.loads(raw))
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE data: %s", raw)
        return events

    @staticmethod
    def _parse_single_sse(raw: str) -> dict[str, Any] | None:
        data_lines: list[str] = []
        for line in raw.strip().splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return None
        try:
            return json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return None
