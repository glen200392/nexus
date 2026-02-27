"""
NEXUS LLM Client — Universal provider adapter
Wraps Ollama / Anthropic / OpenAI / Google behind one async interface.
All callers receive: LLMResponse(content, tokens_in, tokens_out, cost_usd)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from nexus.core.llm.router import ModelConfig, PrivacyTier, get_router

logger = logging.getLogger("nexus.llm.client")


@dataclass
class Message:
    role: str      # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    tokens_in: int  = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_used: str = ""
    finish_reason: str = "stop"
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: int = 0


class LLMClient:
    """
    Unified async LLM client.  Usage:

        client = LLMClient()
        resp = await client.chat(
            messages=[Message("user", "Hello")],
            model=MODEL_REGISTRY["claude-sonnet-4-6"],
            system="You are helpful.",
        )
    """

    def __init__(self, governance=None):
        self._gov = governance   # optional GovernanceManager for PII guard

    async def chat(
        self,
        messages: list[Message],
        model: ModelConfig,
        system: str = "",
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        privacy_tier: PrivacyTier = PrivacyTier.INTERNAL,
    ) -> LLMResponse:
        """Route to the correct provider and return unified LLMResponse."""
        start = time.time()

        # PII guard before any cloud call
        if self._gov and not model.is_local:
            full_text = system + " ".join(m.content for m in messages)
            _, ok = self._gov.guard_cloud_call(full_text, privacy_tier.value, model.model_id)
            if not ok:
                raise PermissionError(
                    f"Privacy policy blocked cloud call to {model.model_id} "
                    f"for PRIVATE tier data"
                )

        try:
            if model.provider == "ollama":
                resp = await self._ollama(messages, model, system, tools, temperature, max_tokens)
            elif model.provider == "anthropic":
                resp = await self._anthropic(messages, model, system, tools, temperature, max_tokens)
            elif model.provider == "openai":
                resp = await self._openai(messages, model, system, tools, temperature, max_tokens)
            elif model.provider == "google":
                resp = await self._google(messages, model, system, tools, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {model.provider}")

            resp.latency_ms = int((time.time() - start) * 1000)
            resp.model_used = model.display_name
            logger.debug(
                "%s | %d+%d tokens | $%.4f | %dms",
                model.display_name, resp.tokens_in, resp.tokens_out,
                resp.cost_usd, resp.latency_ms,
            )
            return resp

        except Exception as exc:
            logger.error("LLM call failed [%s]: %s", model.display_name, exc)
            raise

    # ── Ollama ────────────────────────────────────────────────────────────────

    async def _ollama(
        self,
        messages: list[Message],
        model: ModelConfig,
        system: str,
        tools: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        import httpx

        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            msg_list.append(entry)

        payload: dict = {
            "model": model.model_id,
            "messages": msg_list,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = tools

        endpoint = model.endpoint or "http://localhost:11434"
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{endpoint}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        msg = data.get("message", {})
        content    = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        prompt_tok = data.get("prompt_eval_count", 0)
        out_tok    = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            tokens_in=prompt_tok,
            tokens_out=out_tok,
            cost_usd=0.0,  # local = free
            tool_calls=tool_calls,
        )

    # ── Anthropic ─────────────────────────────────────────────────────────────

    async def _anthropic(
        self,
        messages: list[Message],
        model: ModelConfig,
        system: str,
        tools: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        )

        msg_list = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        kwargs: dict = {
            "model": model.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msg_list,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        r = await client.messages.create(**kwargs)

        content    = ""
        tool_calls = []
        for block in r.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                })

        ti = r.usage.input_tokens
        to = r.usage.output_tokens
        cost = (ti / 1000) * model.cost_per_1k_input + (to / 1000) * model.cost_per_1k_output

        return LLMResponse(
            content=content,
            tokens_in=ti,
            tokens_out=to,
            cost_usd=cost,
            tool_calls=tool_calls,
            finish_reason=r.stop_reason or "stop",
        )

    # ── OpenAI ────────────────────────────────────────────────────────────────

    async def _openai(
        self,
        messages: list[Message],
        model: ModelConfig,
        system: str,
        tools: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        import openai

        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        msg_list = []
        if system:
            msg_list.append({"role": "system", "content": system})
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
                entry["role"] = "tool"
            msg_list.append(entry)

        kwargs: dict = {
            "model": model.model_id,
            "messages": msg_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        r = await client.chat.completions.create(**kwargs)
        choice = r.choices[0]
        msg    = choice.message

        content    = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })

        ti = r.usage.prompt_tokens
        to = r.usage.completion_tokens
        cost = (ti / 1000) * model.cost_per_1k_input + (to / 1000) * model.cost_per_1k_output

        return LLMResponse(
            content=content,
            tokens_in=ti,
            tokens_out=to,
            cost_usd=cost,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )

    # ── Google Gemini ─────────────────────────────────────────────────────────

    async def _google(
        self,
        messages: list[Message],
        model: ModelConfig,
        system: str,
        tools: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

        config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        gmodel = genai.GenerativeModel(
            model_name=model.model_id,
            system_instruction=system or None,
            generation_config=config,
        )

        history = []
        last_user_msg = ""
        for m in messages:
            if m.role == "user":
                history.append({"role": "user", "parts": [m.content]})
                last_user_msg = m.content
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})

        chat    = gmodel.start_chat(history=history[:-1] if history else [])
        r       = await chat.send_message_async(last_user_msg or "")
        content = r.text

        ti   = r.usage_metadata.prompt_token_count if r.usage_metadata else 0
        to   = r.usage_metadata.candidates_token_count if r.usage_metadata else 0
        cost = (ti / 1000) * model.cost_per_1k_input + (to / 1000) * model.cost_per_1k_output

        return LLMResponse(content=content, tokens_in=ti, tokens_out=to, cost_usd=cost)


# ─── Global singleton ────────────────────────────────────────────────────────
_client_instance: Optional[LLMClient] = None

def get_client(governance=None) -> LLMClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient(governance=governance)
    return _client_instance
