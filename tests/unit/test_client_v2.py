"""
Tests for nexus.core.llm.client_v2.LLMClientV2.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.core.llm.client_v2 import LLMClientV2, LLMChunk, Attachment
from nexus.core.llm.router_v2 import (
    LLMRouterV2,
    ModelCapability,
    ModelConfigV2,
    RoutingRequestV2,
    RoutingDecisionV2,
)
from nexus.core.llm.cache import LLMSemanticCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def local_model():
    return ModelConfigV2(
        provider="ollama",
        model_id="qwen2.5:7b",
        display_name="Qwen 2.5 7B",
        context_window=32768,
        max_output=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        capabilities=ModelCapability(reasoning=True, code=True),
        privacy_tiers=["public", "internal", "confidential", "restricted"],
        is_local=True,
        avg_latency_ms=200.0,
    )


@pytest.fixture
def mock_router(local_model):
    router = MagicMock(spec=LLMRouterV2)
    router.models = [local_model]
    decision = RoutingDecisionV2(
        primary=local_model,
        fallback=None,
        reason="test",
        estimated_cost_usd=0.0,
        privacy_compliant=True,
    )
    router.route.return_value = decision
    return router


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestLLMChunk:

    def test_defaults(self):
        chunk = LLMChunk()
        assert chunk.content == ""
        assert chunk.is_final is False

    def test_final_chunk(self):
        chunk = LLMChunk(content="done", is_final=True, finish_reason="stop")
        assert chunk.is_final is True
        assert chunk.finish_reason == "stop"


class TestAttachment:

    def test_image_attachment(self):
        att = Attachment(type="image", content="http://example.com/img.png", mime_type="image/png")
        assert att.type == "image"

    def test_audio_attachment(self):
        att = Attachment(type="audio", content="base64data", mime_type="audio/wav")
        assert att.type == "audio"


# ---------------------------------------------------------------------------
# LLMClientV2 tests
# ---------------------------------------------------------------------------

class TestLLMClientV2:

    def test_init_defaults(self):
        client = LLMClientV2()
        assert client.router is not None
        assert client.cache is None
        assert client.circuit_breaker is None

    def test_init_with_deps(self, mock_router):
        cache = MagicMock(spec=LLMSemanticCache)
        client = LLMClientV2(router=mock_router, cache=cache)
        assert client.router is mock_router
        assert client.cache is cache

    @pytest.mark.asyncio
    async def test_chat_calls_provider(self, mock_router, local_model):
        client = LLMClientV2(router=mock_router)
        expected_response = {"content": "Hello!", "model": "qwen2.5:7b", "usage": {}}

        with patch("nexus.core.llm.client_v2._call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = expected_response
            result = await client.chat(
                [{"role": "user", "content": "Hi"}],
                model="qwen2.5:7b",
            )
        assert result["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_uses_cache(self, mock_router, tmp_path):
        cache = LLMSemanticCache(db_path=tmp_path / "test.db")
        client = LLMClientV2(router=mock_router, cache=cache)
        messages = [{"role": "user", "content": "test query"}]

        # First call — provider hit
        with patch("nexus.core.llm.client_v2._call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"content": "cached!", "model": "qwen2.5:7b", "usage": {}}
            await client.chat(messages, model="qwen2.5:7b")

        # Second call — should hit cache (no provider call)
        with patch("nexus.core.llm.client_v2._call_ollama", new_callable=AsyncMock) as mock_call:
            result = await client.chat(messages, model="qwen2.5:7b")
            mock_call.assert_not_called()
        assert result["content"] == "cached!"

    @pytest.mark.asyncio
    async def test_chat_governance_guard(self, mock_router):
        governance = MagicMock()
        governance.redact = AsyncMock(return_value="[REDACTED]")
        # Use a non-local model to trigger governance
        cloud_model = ModelConfigV2(
            provider="anthropic", model_id="claude-haiku-4-5",
            display_name="Haiku", context_window=200000, max_output=8192,
            cost_per_1k_input=0.001, cost_per_1k_output=0.005,
            capabilities=ModelCapability(reasoning=True),
            privacy_tiers=["public"], is_local=False,
        )
        mock_router.models = [cloud_model]
        decision = RoutingDecisionV2(primary=cloud_model, reason="test")
        mock_router.route.return_value = decision

        client = LLMClientV2(router=mock_router, governance=governance)

        with patch("nexus.core.llm.client_v2._call_anthropic", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"content": "resp", "model": "claude-haiku-4-5", "usage": {}}
            await client.chat(
                [{"role": "user", "content": "my SSN is 123-45-6789"}],
                model="claude-haiku-4-5",
            )
            # The call should have received redacted content
            call_args = mock_call.call_args
            sent_messages = call_args[0][0]
            assert sent_messages[0]["content"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_chat_circuit_breaker_fallback(self, mock_router, local_model):
        cb = MagicMock()
        cb.is_available.return_value = True
        cb.record_success = MagicMock()
        cb.record_failure = MagicMock()

        client = LLMClientV2(router=mock_router, circuit_breaker=cb)

        with patch("nexus.core.llm.client_v2._call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"content": "ok", "model": "qwen2.5:7b", "usage": {}}
            await client.chat([{"role": "user", "content": "hi"}], model="qwen2.5:7b")
            cb.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_count_tokens(self):
        client = LLMClientV2()
        count = await client.count_tokens([
            {"role": "user", "content": "Hello, world!"},
        ])
        assert count > 0
        assert isinstance(count, int)

    def test_guess_provider(self):
        assert LLMClientV2._guess_provider("claude-sonnet-4-6") == "anthropic"
        assert LLMClientV2._guess_provider("gpt-4o") == "openai"
        assert LLMClientV2._guess_provider("o4-mini") == "openai"
        assert LLMClientV2._guess_provider("gemini-2.0-flash") == "google"
        assert LLMClientV2._guess_provider("qwen2.5:7b") == "ollama"

    def test_inject_attachments(self):
        messages = [{"role": "user", "content": "Look at this"}]
        attachments = [Attachment(type="image", content="http://img.png", mime_type="image/png")]
        result = LLMClientV2._inject_attachments(messages, attachments)
        assert "[Image:" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_chat_all_fail_raises(self, mock_router):
        client = LLMClientV2(router=mock_router)
        with patch("nexus.core.llm.client_v2._call_ollama", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = RuntimeError("connection refused")
            with pytest.raises(RuntimeError, match="All models failed"):
                await client.chat(
                    [{"role": "user", "content": "hi"}],
                    model="qwen2.5:7b",
                )
