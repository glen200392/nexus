"""
NEXUS WebSocket Handler
Real-time bidirectional communication for task streaming and status updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("nexus.ws")


class ConnectionManager:
    """Manages active WebSocket connections keyed by session_id."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        self._connections[session_id] = websocket
        logger.info("WebSocket connected: session=%s", session_id)

    async def disconnect(self, session_id: str) -> None:
        """Remove a connection from the manager."""
        ws = self._connections.pop(session_id, None)
        if ws is not None:
            logger.info("WebSocket disconnected: session=%s", session_id)

    async def send_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Send a JSON message to a specific session."""
        ws = self._connections.get(session_id)
        if ws is not None:
            try:
                await ws.send_json(message)
            except Exception:
                logger.warning("Failed to send to session=%s", session_id)
                await self.disconnect(session_id)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected sessions."""
        disconnected: list[str] = []
        for sid, ws in self._connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(sid)
        for sid in disconnected:
            await self.disconnect(sid)

    def get_active_sessions(self) -> list[str]:
        """Return list of active session IDs."""
        return list(self._connections.keys())


# Module-level connection manager
manager = ConnectionManager()


def create_ws_router() -> APIRouter:
    """Create and return a FastAPI APIRouter with WebSocket endpoint."""
    ws_router = APIRouter(tags=["websocket"])

    @ws_router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """
        WebSocket endpoint for real-time communication.

        Receives JSON messages:
            {"type": "task", "message": "...", "session_id": "..."}

        Sends responses:
            {"type": "status|stream|result|error", "data": ...}
        """
        session_id: str | None = None
        try:
            # Use query param or generate session_id
            session_id = websocket.query_params.get("session_id", uuid.uuid4().hex[:12])
            await manager.connect(websocket, session_id)

            # Send connection confirmation
            await manager.send_message(session_id, {
                "type": "status",
                "data": {"connected": True, "session_id": session_id},
            })

            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "data": {"message": "Invalid JSON"},
                    })
                    continue

                msg_type = msg.get("type", "")
                msg_session = msg.get("session_id", session_id)

                if msg_type == "task":
                    # Acknowledge receipt (real orchestrator wiring in Phase 5)
                    task_id = uuid.uuid4().hex[:12]
                    await manager.send_message(session_id, {
                        "type": "status",
                        "data": {
                            "task_id": task_id,
                            "status": "received",
                            "message": msg.get("message", ""),
                        },
                    })
                elif msg_type == "ping":
                    await manager.send_message(session_id, {
                        "type": "status",
                        "data": {"pong": True},
                    })
                else:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "data": {"message": f"Unknown message type: {msg_type}"},
                    })

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)
        finally:
            if session_id:
                await manager.disconnect(session_id)

    return ws_router
