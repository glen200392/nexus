"""
Tests for nexus.mcp.client_v2.MCPClientV2 and MCP transports.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from nexus.mcp.transport.base import MCPTransport
from nexus.mcp.transport.stdio import StdioTransport
from nexus.mcp.transport.http import HTTPTransport
from nexus.mcp.client_v2 import MCPClientV2


# ---------------------------------------------------------------------------
# Transport protocol tests
# ---------------------------------------------------------------------------

class TestMCPTransportProtocol:

    def test_stdio_is_mcp_transport(self):
        t = StdioTransport(command="echo hello")
        assert isinstance(t, MCPTransport)

    def test_http_is_mcp_transport(self):
        t = HTTPTransport(url="http://localhost:8080")
        assert isinstance(t, MCPTransport)


# ---------------------------------------------------------------------------
# StdioTransport tests
# ---------------------------------------------------------------------------

class TestStdioTransport:

    def test_init_string_command(self):
        t = StdioTransport(command="npx -y @server/test")
        assert t._cmd == ["npx", "-y", "@server/test"]

    def test_init_list_command(self):
        t = StdioTransport(command=["node", "server.js"], args=["--port", "8080"])
        assert t._cmd == ["node", "server.js", "--port", "8080"]

    def test_not_connected_initially(self):
        t = StdioTransport(command="echo")
        assert t.is_connected is False

    @pytest.mark.asyncio
    async def test_send_not_connected_raises(self):
        t = StdioTransport(command="echo")
        with pytest.raises(ConnectionError):
            await t.send({"jsonrpc": "2.0", "method": "test"})

    @pytest.mark.asyncio
    async def test_receive_not_connected_raises(self):
        t = StdioTransport(command="echo")
        with pytest.raises(ConnectionError):
            await t.receive()

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        t = StdioTransport(command="cat")
        await t.connect()
        assert t.is_connected is True
        await t.close()
        assert t.is_connected is False

    @pytest.mark.asyncio
    async def test_send_receive_roundtrip(self):
        """Use 'cat' to echo back messages."""
        t = StdioTransport(command="cat")
        await t.connect()
        try:
            msg = {"jsonrpc": "2.0", "id": 1, "method": "test"}
            await t.send(msg)
            received = await t.receive()
            assert received == msg
        finally:
            await t.close()


# ---------------------------------------------------------------------------
# HTTPTransport tests
# ---------------------------------------------------------------------------

class TestHTTPTransport:

    def test_init(self):
        t = HTTPTransport(url="http://localhost:9000/", headers={"X-Key": "abc"})
        assert t._url == "http://localhost:9000"
        assert t._headers["X-Key"] == "abc"

    def test_not_connected_initially(self):
        t = HTTPTransport(url="http://localhost:9000")
        assert t.is_connected is False

    def test_parse_sse(self):
        raw = "data: {\"result\": 42}\n\ndata: {\"result\": 43}\n\n"
        events = HTTPTransport._parse_sse(raw)
        assert len(events) == 2
        assert events[0]["result"] == 42
        assert events[1]["result"] == 43

    def test_parse_sse_multiline_data(self):
        raw = 'data: {"a":\ndata: 1}\n\n'
        events = HTTPTransport._parse_sse(raw)
        assert len(events) == 1
        assert events[0]["a"] == 1

    def test_parse_sse_non_json(self):
        raw = "data: not json\n\n"
        events = HTTPTransport._parse_sse(raw)
        assert len(events) == 0

    def test_parse_single_sse(self):
        result = HTTPTransport._parse_single_sse('data: {"ok": true}')
        assert result == {"ok": True}

    def test_parse_single_sse_empty(self):
        result = HTTPTransport._parse_single_sse("event: ping")
        assert result is None


# ---------------------------------------------------------------------------
# MCPClientV2 tests
# ---------------------------------------------------------------------------

class TestMCPClientV2:

    @pytest.fixture
    def config_yaml(self, tmp_path):
        config = {
            "version": "2.0",
            "servers": {
                "test-stdio": {
                    "transport": "stdio",
                    "command": "echo",
                    "args": [],
                    "description": "Test stdio server",
                },
                "test-http": {
                    "transport": "http",
                    "url": "http://localhost:9999",
                    "description": "Test HTTP server",
                },
            },
        }
        config_file = tmp_path / "servers.yaml"
        import yaml
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_load_config(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        assert "test-stdio" in client.server_names
        assert "test-http" in client.server_names
        assert len(client.server_names) == 2

    def test_missing_config(self, tmp_path):
        client = MCPClientV2(config_path=tmp_path / "nonexistent.yaml")
        assert client.server_names == []

    def test_unknown_server_raises(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        with pytest.raises(ValueError, match="Unknown MCP server"):
            asyncio.get_event_loop().run_until_complete(
                client.connect("nonexistent")
            )

    @pytest.mark.asyncio
    async def test_connect_creates_transport(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        # Mock the transport so we don't actually spawn a process
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        with patch.object(client, "_create_transport", return_value=mock_transport):
            await client.connect("test-stdio")
        assert "test-stdio" in client._transports

    @pytest.mark.asyncio
    async def test_disconnect(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        client._transports["test-stdio"] = mock_transport
        await client.disconnect("test-stdio")
        mock_transport.close.assert_awaited_once()
        assert "test-stdio" not in client._transports

    @pytest.mark.asyncio
    async def test_call_tool(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "hello"}]},
        }
        client._transports["test-stdio"] = mock_transport

        result = await client.call_tool("test-stdio", "read_file", {"path": "/tmp/test"})
        assert "content" in result
        mock_transport.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_tool_error(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        client._transports["test-stdio"] = mock_transport

        with pytest.raises(RuntimeError, match="MCP tool error"):
            await client.call_tool("test-stdio", "unknown_tool", {})

    @pytest.mark.asyncio
    async def test_list_tools(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {"name": "read_file", "description": "Read a file"},
                    {"name": "write_file", "description": "Write a file"},
                ],
            },
        }
        client._transports["test-stdio"] = mock_transport

        tools = await client.list_tools("test-stdio")
        assert len(tools) == 2
        assert tools[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_list_tools_cached(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        client._transports["test-stdio"] = mock_transport
        client._tool_cache["test-stdio"] = [{"name": "cached_tool"}]

        tools = await client.list_tools("test-stdio")
        assert tools[0]["name"] == "cached_tool"
        mock_transport.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_server_status(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        status = await client.get_server_status("test-stdio")
        assert status["name"] == "test-stdio"
        assert status["configured"] is True
        assert status["connected"] is False

    @pytest.mark.asyncio
    async def test_disconnect_all(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        t1 = AsyncMock()
        t2 = AsyncMock()
        client._transports = {"a": t1, "b": t2}
        await client.disconnect_all()
        t1.close.assert_awaited_once()
        t2.close.assert_awaited_once()
        assert len(client._transports) == 0

    def test_create_transport_stdio(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        transport = client._create_transport({"transport": "stdio", "command": "echo", "args": []})
        assert isinstance(transport, StdioTransport)

    def test_create_transport_http(self, config_yaml):
        client = MCPClientV2(config_path=config_yaml)
        transport = client._create_transport({"transport": "http", "url": "http://localhost:8080"})
        assert isinstance(transport, HTTPTransport)
