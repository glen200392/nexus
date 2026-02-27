"""
NEXUS v2 — Enhanced LLM Client.

Integrates routing, caching, circuit-breaking, governance, and
multi-provider dispatch into a single unified interface.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional

from nexus.core.llm.client import LLMResponse, Message
from nexus.core.llm.router_v2 import (
    LLMRouterV2,
    ModelConfigV2,
    RoutingRequestV2,
)
from nexus.core.llm.cache import LLMSemanticCache
from nexus.core.llm.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMChunk:
    """A single chunk from a streaming LLM response."""
    content: str = ""
    delta: str = ""
    finish_reason: str | None = None
    model: str = ""
    is_final: bool = False


@dataclass
class Attachment:
    """An attachment (image, audio, file) sent with a chat request."""
    type: str = "image"            # image | audio | file
    content: str = ""              # base64 string or URL
    mime_type: str = "image/png"


# ---------------------------------------------------------------------------
# Provider dispatch helpers
# ---------------------------------------------------------------------------

async def _call_ollama(
    messages: list[dict],
    model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stream: bool = False,
    **kwargs: Any,
) -> dict:
    """Call a local Ollama model via its HTTP API."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx is required for Ollama calls: pip install httpx")

    endpoint = kwargs.get("endpoint", "http://localhost:11434")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{endpoint}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {
        "content": data.get("message", {}).get("content", ""),
        "model": model,
        "usage": {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
        },
    }


async def _call_anthropic(
    messages: list[dict],
    model: str,
    *,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    thinking: bool = False,
    thinking_budget: int | None = None,
    tools: list[dict] | None = None,
    **kwargs: Any,
) -> dict:
    """Call Anthropic's Messages API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic SDK required: pip install anthropic")

    client = anthropic.AsyncAnthropic()
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if temperature is not None and not thinking:
        params["temperature"] = temperature
    if thinking and thinking_budget:
        params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if tools:
        params["tools"] = tools

    resp = await client.messages.create(**params)
    content = ""
    for block in resp.content:
        if hasattr(block, "text"):
            content += block.text
    return {
        "content": content,
        "model": model,
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        },
    }


async def _call_openai(
    messages: list[dict],
    model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    **kwargs: Any,
) -> dict:
    """Call OpenAI's Chat Completions API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai SDK required: pip install openai")

    client = openai.AsyncOpenAI()
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        params["tools"] = tools

    resp = await client.chat.completions.create(**params)
    choice = resp.choices[0]
    return {
        "content": choice.message.content or "",
        "model": model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }


async def _call_google(
    messages: list[dict],
    model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> dict:
    """Call Google's Generative AI API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("google-generativeai SDK required: pip install google-generativeai")

    genai_model = genai.GenerativeModel(model)
    # Convert messages to Gemini format
    contents: list[dict] = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    resp = await asyncio.to_thread(
        genai_model.generate_content,
        contents,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return {
        "content": resp.text if resp.text else "",
        "model": model,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


PROVIDER_DISPATCH = ("ollama", "anthropic", "openai", "google")


def _get_provider_fn(provider: str):
    """Look up the provider dispatch function dynamically.

    Uses globals() so that unittest.mock.patch on the module attribute
    (e.g. ``_call_ollama``) is picked up at call time.
    """
    if provider not in PROVIDER_DISPATCH:
        return None
    return globals().get(f"_call_{provider}")


# ---------------------------------------------------------------------------
# LLMClientV2
# ---------------------------------------------------------------------------

class LLMClientV2:
    """Unified LLM client with routing, caching, circuit-breaking, and governance."""

    def __init__(
        self,
        router: LLMRouterV2 | None = None,
        cache: LLMSemanticCache | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        governance: Any | None = None,
    ) -> None:
        self.router = router or LLMRouterV2()
        self.cache = cache
        self.circuit_breaker = circuit_breaker
        self.governance = governance

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        *,
        system: str = "",
        privacy_tier: str = "public",
        tools: list[dict] | None = None,
        structured_output: dict | None = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,
        thinking_budget: int | None = None,
        cache_ttl: int = 3600,
        attachments: list[Attachment] | None = None,
    ) -> dict | AsyncIterator[LLMChunk]:
        """Send a chat request, routing to the optimal provider.

        Returns a dict with keys: content, model, usage.
        If *stream* is True, returns an async iterator of LLMChunk (not yet implemented — falls back to non-stream).
        """
        # Resolve model via router if not specified
        resolved_model = self._resolve_model(messages, model, privacy_tier, tools, stream)

        # Governance PII guard for cloud models
        if self.governance and not resolved_model.is_local:
            messages = await self._apply_governance(messages)

        # Prepare attachment content into messages
        if attachments:
            messages = self._inject_attachments(messages, attachments)

        # Cache lookup
        cache_key = self._cache_key(messages, resolved_model.model_id)
        if self.cache and not stream:
            cached = await self.cache.get(cache_key, resolved_model.model_id)
            if cached is not None:
                logger.debug("Cache hit for %s", resolved_model.model_id)
                return cached

        # Dispatch to provider
        provider_fn = _get_provider_fn(resolved_model.provider)
        if provider_fn is None:
            raise ValueError(f"Unsupported provider: {resolved_model.provider}")

        call_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if resolved_model.provider == "anthropic":
            call_kwargs.update(system=system, thinking=thinking, thinking_budget=thinking_budget, tools=tools)
        elif resolved_model.provider == "openai":
            call_kwargs["tools"] = tools
        if resolved_model.endpoint:
            call_kwargs["endpoint"] = resolved_model.endpoint

        # Execute with circuit-breaker awareness
        cb_key = f"{resolved_model.provider}:{resolved_model.model_id}"
        result: dict | None = None
        last_error: Exception | None = None

        models_to_try = [resolved_model]
        # Add fallback if available
        if model is None:
            try:
                req = self._build_routing_request(messages, privacy_tier, tools, stream)
                decision = self.router.route(req)
                if decision.fallback:
                    models_to_try.append(decision.fallback)
            except ValueError:
                pass

        for attempt_model in models_to_try:
            attempt_key = f"{attempt_model.provider}:{attempt_model.model_id}"
            if self.circuit_breaker and not self.circuit_breaker.is_available(attempt_key):
                logger.warning("Circuit open for %s, skipping", attempt_key)
                continue
            try:
                fn = _get_provider_fn(attempt_model.provider)
                if fn is None:
                    continue
                kw = dict(call_kwargs)
                if attempt_model.endpoint:
                    kw["endpoint"] = attempt_model.endpoint
                result = await fn(messages, attempt_model.model_id, **kw)
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(attempt_key)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Call to %s failed: %s", attempt_model.model_id, exc)
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(attempt_key)

        if result is None:
            raise RuntimeError(f"All models failed. Last error: {last_error}")

        # Store in cache
        if self.cache and not stream:
            await self.cache.put(cache_key, result, resolved_model.model_id, ttl=cache_ttl)

        return result

    async def count_tokens(self, messages: list[dict], model: str = "") -> int:
        """Rough token count estimate (4 chars ~ 1 token)."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(
        self,
        messages: list[dict],
        model: str | None,
        privacy_tier: str,
        tools: list[dict] | None,
        stream: bool,
    ) -> ModelConfigV2:
        """Resolve a ModelConfigV2 either by name or via the router."""
        if model:
            for m in self.router.models:
                if m.model_id == model:
                    return m
            # Unknown model — create a minimal config
            return ModelConfigV2(
                provider=self._guess_provider(model),
                model_id=model,
                display_name=model,
                context_window=128_000,
                max_output=4096,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                capabilities=__import__("nexus.core.llm.router_v2", fromlist=["ModelCapability"]).ModelCapability(),
            )

        req = self._build_routing_request(messages, privacy_tier, tools, stream)
        decision = self.router.route(req)
        return decision.primary

    def _build_routing_request(
        self,
        messages: list[dict],
        privacy_tier: str,
        tools: list[dict] | None,
        stream: bool,
    ) -> RoutingRequestV2:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens = max(1, total_chars // 4)
        required_caps: list[str] = []
        if tools:
            required_caps.append("tool_use")
        return RoutingRequestV2(
            privacy_tier=privacy_tier,
            required_capabilities=required_caps,
            estimated_input_tokens=est_tokens,
            stream=stream,
        )

    @staticmethod
    def _guess_provider(model_id: str) -> str:
        if "claude" in model_id:
            return "anthropic"
        if "gpt" in model_id or model_id.startswith("o"):
            return "openai"
        if "gemini" in model_id:
            return "google"
        return "ollama"

    @staticmethod
    def _cache_key(messages: list[dict], model: str) -> str:
        import hashlib
        raw = json.dumps(messages, sort_keys=True) + "|" + model
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _apply_governance(self, messages: list[dict]) -> list[dict]:
        """Run governance PII guard on outgoing messages."""
        cleaned: list[dict] = []
        for msg in messages:
            content = msg.get("content", "")
            if hasattr(self.governance, "redact"):
                content = await self.governance.redact(content)
            elif hasattr(self.governance, "guard"):
                content = await self.governance.guard(content)
            cleaned.append({**msg, "content": content})
        return cleaned

    @staticmethod
    def _inject_attachments(messages: list[dict], attachments: list[Attachment]) -> list[dict]:
        """Append attachment information to the last user message."""
        if not attachments or not messages:
            return messages

        messages = [dict(m) for m in messages]  # shallow copy
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return messages

        content = messages[last_user_idx].get("content", "")
        parts: list[str] = [content] if isinstance(content, str) else []
        for att in attachments:
            if att.type == "image" and att.content.startswith("http"):
                parts.append(f"\n[Image: {att.content}]")
            elif att.type == "audio":
                parts.append(f"\n[Audio attachment: {att.mime_type}]")
            else:
                parts.append(f"\n[File attachment: {att.mime_type}]")

        messages[last_user_idx] = {**messages[last_user_idx], "content": "\n".join(parts)}
        return messages
