"""
Tests for NEXUS WebSocket Handler.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.api.ws_handler import ConnectionManager


class TestConnectionManager:
    @pytest.fixture
    def manager(self):
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, manager, mock_websocket):
        """Should track connected sessions."""
        await manager.connect(mock_websocket, "session-1")
        assert "session-1" in manager.get_active_sessions()

        await manager.disconnect("session-1")
        assert "session-1" not in manager.get_active_sessions()

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, manager, mock_websocket):
        """Should return all active session IDs."""
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()

        await manager.connect(mock_websocket, "s1")
        await manager.connect(ws2, "s2")

        sessions = manager.get_active_sessions()
        assert "s1" in sessions
        assert "s2" in sessions
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_send_message(self, manager, mock_websocket):
        """Should send JSON to a specific session."""
        await manager.connect(mock_websocket, "s1")
        await manager.send_message("s1", {"type": "test", "data": "hello"})
        mock_websocket.send_json.assert_called_once_with({"type": "test", "data": "hello"})

    @pytest.mark.asyncio
    async def test_send_message_unknown_session(self, manager):
        """Sending to unknown session should not raise."""
        await manager.send_message("nonexistent", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast(self, manager):
        """Should broadcast to all connected sessions."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock()

        await manager.connect(ws1, "s1")
        await manager.connect(ws2, "s2")

        msg = {"type": "broadcast", "data": "all"}
        await manager.broadcast(msg)
        ws1.send_json.assert_called_once_with(msg)
        ws2.send_json.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self, manager):
        """Disconnecting unknown session should not raise."""
        await manager.disconnect("nonexistent")

    @pytest.mark.asyncio
    async def test_send_message_failure_disconnects(self, manager):
        """If send fails, session should be disconnected."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock(side_effect=RuntimeError("connection lost"))

        await manager.connect(ws, "s-fail")
        assert "s-fail" in manager.get_active_sessions()

        await manager.send_message("s-fail", {"type": "test"})
        assert "s-fail" not in manager.get_active_sessions()
