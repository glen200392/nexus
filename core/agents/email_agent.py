"""
NEXUS Email Agent — Compose, Review, and Send Email
Drafts email content with LLM, then waits for human confirmation
before sending. Sending is an irreversible external action and is
always gated behind an explicit `send=true` flag in metadata.

Supported operations (context.metadata["operation"]):
  draft   — compose a draft (default); return text for review
  send    — compose + actually send via SMTP (requires send=true confirmation)
  reply   — draft a reply to a provided email thread
  summarize — summarize a long email thread into bullet points

Transport:
  Primary: SMTP (stdlib smtplib)
  Optional: Resend API (https://resend.com) if RESEND_API_KEY is set

Environment:
  SMTP_HOST      (default: smtp.gmail.com)
  SMTP_PORT      (default: 587)
  SMTP_USER      — Sender email address
  SMTP_PASSWORD  — App password or OAuth token
  EMAIL_FROM     — Display name + address, e.g. "NEXUS <nexus@company.com>"
  RESEND_API_KEY — If set, uses Resend instead of SMTP
"""
from __future__ import annotations

import json
import logging
import re
import smtplib
import ssl
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
import os

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity, PrivacyTier
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.email")

SMTP_HOST    = os.environ.get("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT    = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER    = os.environ.get("SMTP_USER",     "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
EMAIL_FROM   = os.environ.get("EMAIL_FROM",    SMTP_USER)
RESEND_KEY   = os.environ.get("RESEND_API_KEY", "")


@dataclass
class EmailDraft:
    to:          list[str] = field(default_factory=list)
    cc:          list[str] = field(default_factory=list)
    subject:     str = ""
    body_text:   str = ""
    body_html:   str = ""
    reply_to_id: str = ""   # Message-ID for threading


class EmailAgent(BaseAgent):
    agent_id   = "email_agent"
    agent_name = "Email Agent"
    description = (
        "Composes professional emails with LLM, previews draft, "
        "then sends via SMTP or Resend API only after explicit confirmation. "
        "Supports draft, send, reply, and summarize operations."
    )
    domain             = TaskDomain.OPERATIONS
    default_complexity = TaskComplexity.LOW
    default_privacy    = PrivacyTier.INTERNAL   # Emails may contain personal data

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        tone = context.metadata.get("tone", "professional")
        lang = context.metadata.get("language", "zh-TW")
        return (
            f"You are an expert email writer. Write {tone} emails in {lang} unless "
            "the user specifies a different language.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "to": ["email@example.com"],\n'
            '  "cc": [],\n'
            '  "subject": "...",\n'
            '  "body": "Full email body in plain text",\n'
            '  "body_html": "<p>Optional HTML version</p>"\n'
            "}\n\n"
            "Rules:\n"
            "- Keep subject concise (< 60 chars)\n"
            "- Use appropriate greeting and closing\n"
            "- Never fabricate email addresses — use only those explicitly provided\n"
            "- If no recipient provided, use 'to': ['PLACEHOLDER@example.com'] as marker"
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        operation = context.metadata.get("operation", "draft")

        if operation == "summarize":
            return await self._summarize_thread(context)

        # For all other operations, first draft the email
        draft = await self._compose_draft(context)
        if draft is None:
            return AgentResult(
                agent_id=self.agent_id,
                task_id=context.task_id,
                success=False,
                output=None,
                error="Failed to compose email draft",
            )

        # Return draft for review — never send without explicit confirmation
        preview = self._format_preview(draft)

        # Only send if explicitly requested AND send flag is set
        actually_send = (
            operation == "send"
            and context.metadata.get("send", False) is True
        )

        sent = False
        send_error = ""
        if actually_send:
            sent, send_error = await self._send(draft)

        await self.remember(
            content=f"Email {'sent' if sent else 'drafted'}: {draft.subject}\nTo: {', '.join(draft.to)}",
            context=context,
            doc_type=DocumentType.EPISODIC,
            tags=["email", operation, "communication"],
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output={
                "draft":       {
                    "to":      draft.to,
                    "cc":      draft.cc,
                    "subject": draft.subject,
                    "body":    draft.body_text,
                },
                "preview":     preview,
                "sent":        sent,
                "send_error":  send_error,
                "requires_confirmation": not actually_send,
            },
            quality_score=0.8,
        )

    # ── Compose ──────────────────────────────────────────────────────────────

    async def _compose_draft(self, context: AgentContext) -> Optional[EmailDraft]:
        """Use LLM to compose a structured email draft."""
        decision = self.route_llm(context)

        extra = ""
        if context.metadata.get("operation") == "reply":
            thread = context.metadata.get("thread", "")
            extra  = f"\n\nOriginal thread to reply to:\n{thread[:2000]}"

        resp = await self._llm.chat(
            messages=[
                Message("user", f"{context.user_message}{extra}")
            ],
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
            temperature=0.4,
            max_tokens=1500,
        )

        data = self._parse_json(resp.content)
        if not data:
            return None

        return EmailDraft(
            to=self._normalize_emails(data.get("to", [])),
            cc=self._normalize_emails(data.get("cc", [])),
            subject=data.get("subject", ""),
            body_text=data.get("body", ""),
            body_html=data.get("body_html", ""),
        )

    async def _summarize_thread(self, context: AgentContext) -> AgentResult:
        """Summarize a long email thread into action items."""
        thread = context.metadata.get("thread", context.user_message)
        decision = self.route_llm(context)
        resp = await self._llm.chat(
            messages=[
                Message("user",
                    f"Summarize this email thread into:\n"
                    "1. Key decisions made\n"
                    "2. Action items with owners\n"
                    "3. Open questions\n\n"
                    f"Thread:\n{thread[:6000]}"
                )
            ],
            model=decision.primary,
            system=(
                "You are an expert at summarizing email threads. "
                "Be concise and action-oriented. "
                "Return JSON: {\"decisions\": [], \"action_items\": [], \"open_questions\": []}"
            ),
            privacy_tier=context.privacy_tier,
        )
        summary = self._parse_json(resp.content)
        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output={"summary": summary, "raw": resp.content},
            quality_score=0.8,
        )

    # ── Send ─────────────────────────────────────────────────────────────────

    async def _send(self, draft: EmailDraft) -> tuple[bool, str]:
        """Send via Resend API or SMTP."""
        if RESEND_KEY:
            return self._send_resend(draft)
        return self._send_smtp(draft)

    def _send_smtp(self, draft: EmailDraft) -> tuple[bool, str]:
        if not SMTP_USER or not SMTP_PASSWORD:
            return False, "SMTP_USER or SMTP_PASSWORD not configured"
        try:
            msg = MIMEMultipart("alternative")
            msg["From"]    = EMAIL_FROM or SMTP_USER
            msg["To"]      = ", ".join(draft.to)
            msg["Subject"] = draft.subject
            if draft.cc:
                msg["Cc"] = ", ".join(draft.cc)
            msg.attach(MIMEText(draft.body_text, "plain", "utf-8"))
            if draft.body_html:
                msg.attach(MIMEText(draft.body_html, "html", "utf-8"))
            all_recipients = draft.to + draft.cc
            ctx = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, all_recipients, msg.as_string())
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _send_resend(self, draft: EmailDraft) -> tuple[bool, str]:
        import urllib.request
        import urllib.error
        payload = json.dumps({
            "from":    EMAIL_FROM or SMTP_USER,
            "to":      draft.to,
            "cc":      draft.cc or None,
            "subject": draft.subject,
            "text":    draft.body_text,
            "html":    draft.body_html or None,
        }).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {RESEND_KEY}",
                "Content-Type":  "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
                return True, result.get("id", "")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            return False, f"HTTP {exc.code}: {body[:200]}"
        except Exception as exc:
            return False, str(exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_preview(self, draft: EmailDraft) -> str:
        lines = [
            f"To:      {', '.join(draft.to) or '(none)'}",
            f"Subject: {draft.subject}",
        ]
        if draft.cc:
            lines.append(f"Cc:      {', '.join(draft.cc)}")
        lines.append("")
        lines.append(draft.body_text[:1500])
        return "\n".join(lines)

    def _normalize_emails(self, raw: list) -> list[str]:
        """Extract valid-looking email addresses, skip placeholders."""
        result = []
        email_re = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
        for item in raw:
            m = email_re.search(str(item))
            if m and "PLACEHOLDER" not in m.group():
                result.append(m.group())
        return result

    def _parse_json(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {}
