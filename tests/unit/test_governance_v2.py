"""
Tests for GovernanceV2 — PIIScrubberV2 and EUAIActClassifierV2.
"""
from __future__ import annotations

import pytest

from nexus.core.governance_v2 import PIIScrubberV2, EUAIActClassifierV2


class TestPIIScrubberV2:

    def setup_method(self):
        self.scrubber = PIIScrubberV2()

    def test_scrub_taiwan_id(self):
        """Taiwan ID should be scrubbed."""
        text = "His ID is A123456789."
        result = self.scrubber.scrub(text)
        assert "A123456789" not in result
        assert "[REDACTED]" in result

    def test_scrub_credit_card(self):
        """Credit card number should be scrubbed."""
        text = "Card: 4111-1111-1111-1111"
        result = self.scrubber.scrub(text)
        assert "4111" not in result
        assert "[REDACTED]" in result

    def test_scrub_ssn(self):
        """US Social Security Number should be scrubbed."""
        text = "SSN: 123-45-6789"
        result = self.scrubber.scrub(text)
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_scrub_medical_record(self):
        """Medical record number should be scrubbed."""
        text = "Patient MRN: 12345678"
        result = self.scrubber.scrub(text)
        assert "MRN: 12345678" not in result
        assert "[REDACTED]" in result

    def test_scrub_email(self):
        """Email should be scrubbed."""
        text = "Contact: user@example.com"
        result = self.scrubber.scrub(text)
        assert "user@example.com" not in result

    def test_scrub_korean_id(self):
        """Korean ID should be scrubbed."""
        text = "Korean ID: 900101-1234567"
        result = self.scrubber.scrub(text)
        assert "900101-1234567" not in result

    def test_scrub_clean_text(self):
        """Clean text should remain unchanged."""
        text = "This is perfectly clean text with no PII."
        result = self.scrubber.scrub(text)
        assert result == text

    def test_tokenize_detokenize(self):
        """Roundtrip tokenization should restore original text."""
        original = "Contact user@example.com about SSN 123-45-6789"
        tokenized, token_map = self.scrubber.tokenize(original)

        # PII should be replaced with tokens
        assert "user@example.com" not in tokenized
        assert "123-45-6789" not in tokenized
        assert "[PII_" in tokenized
        assert len(token_map) > 0

        # Detokenize should restore
        restored = self.scrubber.detokenize(tokenized, token_map)
        assert "user@example.com" in restored
        assert "123-45-6789" in restored

    def test_tokenize_no_pii(self):
        """Tokenizing text without PII should return unchanged text and empty map."""
        text = "No sensitive information here."
        tokenized, token_map = self.scrubber.tokenize(text)
        assert tokenized == text
        assert token_map == {}

    def test_detect(self):
        """Detect should return list of PII types found."""
        text = "Email: test@mail.com, SSN: 123-45-6789"
        types = self.scrubber.detect(text)
        assert "EMAIL" in types
        assert "SSN" in types


class TestEUAIActClassifierV2:

    def setup_method(self):
        self.classifier = EUAIActClassifierV2()

    def test_eu_ai_act_prohibited(self):
        """Social scoring should be classified as prohibited."""
        result = self.classifier.classify(
            "Implement a social scoring system for citizens"
        )
        assert result["risk_level"] == "prohibited"
        assert result["risk_score"] == 1.0
        assert result["compliant"] is False
        assert len(result["flags"]) > 0

    def test_eu_ai_act_prohibited_biometric(self):
        """Real-time biometric identification should be prohibited."""
        result = self.classifier.classify(
            "Deploy real-time biometric surveillance in public spaces"
        )
        assert result["risk_level"] == "prohibited"
        assert result["compliant"] is False

    def test_eu_ai_act_minimal(self):
        """Benign task should be classified as minimal risk."""
        result = self.classifier.classify(
            "Generate a summary of recent technology news"
        )
        assert result["risk_level"] == "minimal"
        assert result["risk_score"] == 0.0
        assert result["compliant"] is True
        assert result["flags"] == []

    def test_eu_ai_act_high_risk(self):
        """Credit scoring should be classified as high risk."""
        result = self.classifier.classify(
            "Build a credit scoring model for loan applications"
        )
        assert result["risk_level"] == "high"
        assert 0.5 <= result["risk_score"] <= 1.0
        assert result["compliant"] is True
        assert len(result["flags"]) > 0

    def test_eu_ai_act_high_risk_employment(self):
        """Employment screening should be high risk."""
        result = self.classifier.classify(
            "Automated employment screening for job candidates"
        )
        assert result["risk_level"] == "high"
        assert result["compliant"] is True

    def test_eu_ai_act_empty_description(self):
        """Empty description should be minimal risk."""
        result = self.classifier.classify("")
        assert result["risk_level"] == "minimal"
        assert result["compliant"] is True

    def test_eu_ai_act_limited_risk(self):
        """Chatbot should be classified as limited risk."""
        result = self.classifier.classify(
            "Build a customer service chatbot"
        )
        assert result["risk_level"] == "limited"
        assert result["compliant"] is True

    def test_eu_ai_act_predictive_policing(self):
        """Predictive policing should be prohibited."""
        result = self.classifier.classify(
            "Develop predictive policing algorithms for city surveillance"
        )
        assert result["risk_level"] == "prohibited"
        assert result["compliant"] is False
