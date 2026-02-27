"""
Tests for Governance — PII scrubbing, audit logging, cloud guard.
"""
from __future__ import annotations

import pytest

from nexus.core.governance import (
    PIIScrubber, AuditLogger, AuditRecord, GovernanceManager,
    QualityOptimizer,
)


class TestPIIScrubber:
    """PII detection and scrubbing."""

    def setup_method(self):
        self.scrubber = PIIScrubber()

    def test_scrub_taiwan_id(self):
        text = "His ID is A123456789 and he lives in Taipei."
        scrubbed, found = self.scrubber.scrub(text)
        assert "A123456789" not in scrubbed
        assert "[TAIWAN_ID]" in scrubbed
        assert "TAIWAN_ID" in found

    def test_scrub_credit_card(self):
        text = "Card: 4111-1111-1111-1111"
        scrubbed, found = self.scrubber.scrub(text)
        assert "4111" not in scrubbed
        assert "CREDIT_CARD" in found

    def test_scrub_email(self):
        text = "Contact: user@example.com"
        scrubbed, found = self.scrubber.scrub(text)
        assert "user@example.com" not in scrubbed
        assert "EMAIL" in found

    def test_scrub_phone(self):
        text = "Call 0912345678 for info"
        scrubbed, found = self.scrubber.scrub(text)
        assert "0912345678" not in scrubbed
        assert "PHONE" in found

    def test_scrub_api_key(self):
        text = "Use sk-abcdefghijklmnopqrstuvwxyz123 for auth"
        scrubbed, found = self.scrubber.scrub(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz123" not in scrubbed
        assert "API_KEY" in found

    def test_scrub_credential(self):
        text = "password: my_secret_pass123"
        scrubbed, found = self.scrubber.scrub(text)
        assert "my_secret_pass123" not in scrubbed
        assert "CREDENTIAL" in found

    def test_clean_text_unchanged(self):
        text = "This is a perfectly clean text with no PII."
        scrubbed, found = self.scrubber.scrub(text)
        assert scrubbed == text
        assert found == []

    def test_multiple_pii_types(self):
        text = "ID: A123456789, email: test@mail.com, card: 4111 1111 1111 1111"
        scrubbed, found = self.scrubber.scrub(text)
        assert len(found) >= 2  # At least Taiwan ID and email

    def test_hash_for_audit(self):
        h1 = PIIScrubber.hash_for_audit("test")
        h2 = PIIScrubber.hash_for_audit("test")
        h3 = PIIScrubber.hash_for_audit("different")
        assert h1 == h2  # Deterministic
        assert h1 != h3  # Different inputs


class TestCloudGuard:
    """Bug 7: guard_cloud_call privacy enforcement."""

    def test_blocks_private_to_cloud(self, governance_manager):
        """PRIVATE tier text must be blocked from cloud models."""
        text = "Secret data about A123456789"
        _, ok = governance_manager.guard_cloud_call(
            text, "PRIVATE", "claude-sonnet-4-6", is_local=False
        )
        assert not ok

    def test_allows_private_to_local(self, governance_manager):
        """PRIVATE tier text should be allowed for local models."""
        text = "Secret data"
        _, ok = governance_manager.guard_cloud_call(
            text, "PRIVATE", "qwen2.5:7b", is_local=True
        )
        assert ok

    def test_allows_internal_to_cloud(self, governance_manager):
        """INTERNAL tier text should be allowed for cloud models (after scrubbing)."""
        text = "Company project data"
        scrubbed, ok = governance_manager.guard_cloud_call(
            text, "INTERNAL", "claude-sonnet-4-6", is_local=False
        )
        assert ok

    def test_scrubs_pii_before_cloud(self, governance_manager):
        """PII should be scrubbed even when call is allowed."""
        text = "Contact user@example.com about the project"
        scrubbed, ok = governance_manager.guard_cloud_call(
            text, "INTERNAL", "claude-sonnet-4-6", is_local=False
        )
        assert ok
        assert "user@example.com" not in scrubbed


class TestAuditLogger:
    """Audit logging persistence."""

    def test_log_and_retrieve(self, tmp_audit_db):
        audit = AuditLogger(tmp_audit_db)
        record = AuditRecord(
            event_type="test_event",
            task_id="task-123",
            agent_id="test_agent",
            action="Test action",
            cost_usd=0.01,
        )
        audit.log(record)

        history = audit.get_task_history("task-123")
        assert len(history) == 1
        assert history[0]["event_type"] == "test_event"

    def test_cost_summary(self, tmp_audit_db):
        audit = AuditLogger(tmp_audit_db)
        for i in range(3):
            audit.log(AuditRecord(
                event_type="model_called",
                task_id=f"task-{i}",
                agent_id="agent_a",
                model_used="test-model",
                cost_usd=0.1,
            ))

        summary = audit.get_cost_summary()
        assert len(summary["rows"]) >= 1
        assert summary["rows"][0]["total_cost"] == pytest.approx(0.3)
