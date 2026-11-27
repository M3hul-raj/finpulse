"""
test_llm_failover.py
Unit tests for LLM cross-provider failover, explicit timeouts, key filtering,
and input validation in src/llm_explainer.py (all external SDK calls mocked).
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llm_explainer


SAMPLE_ALERT = {
    "segment": "Daily Wage",
    "fhs_day1": 49.0,
    "fhs_day30": 48.0,
    "min_lower": 26.6,
    "severity": "CRITICAL",
    "declining": False,
}


def test_primary_gemini_succeeds_without_calling_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.return_value = SimpleNamespace(
        text="Proactively enroll Daily Wage customers in micro-savings auto-sweeps."
    )
    mock_groq_cls = MagicMock()

    with patch("google.genai.Client", return_value=mock_gemini_client) as gemini_cls, \
         patch("groq.Groq", mock_groq_cls):
        meta = llm_explainer.explain_segment_with_meta(SAMPLE_ALERT, timeout=7.5)

    assert meta["provider"] == "gemini"
    assert meta["model"] == "gemini-3.5-flash-lite"
    assert meta["fallback_used"] is False
    assert "micro-savings" in meta["explanation"]
    mock_groq_cls.assert_not_called()

    # Verify Gemini client received millisecond timeout (7.5s -> 7500ms)
    _, kwargs = gemini_cls.call_args
    assert kwargs["http_options"].timeout == 7500


def test_primary_fails_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.side_effect = RuntimeError("429 quota exceeded")

    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Offer short-term emergency liquidity buffer."))]
    )

    with patch("google.genai.Client", return_value=mock_gemini_client), \
         patch("groq.Groq", return_value=mock_groq_client) as groq_cls:
        meta = llm_explainer.explain_segment_with_meta(SAMPLE_ALERT, timeout=6.0)

    assert meta["provider"] == "groq"
    assert meta["model"] == "openai/gpt-oss-120b"
    assert meta["fallback_used"] is False
    assert "liquidity buffer" in meta["explanation"]
    assert meta["attempts"][0]["status"] == "unavailable"
    assert meta["attempts"][1]["status"] == "ok"

    _, kwargs = groq_cls.call_args
    assert kwargs["timeout"] == 6.0


def test_timeout_triggers_failover_to_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.side_effect = TimeoutError("Gemini request timed out")

    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create.side_effect = TimeoutError("Groq request timed out")

    with patch("google.genai.Client", return_value=mock_gemini_client), \
         patch("groq.Groq", return_value=mock_groq_client):
        meta = llm_explainer.explain_segment_with_meta(SAMPLE_ALERT, timeout=2.0)

    assert meta["provider"] == "curated_fallback"
    assert meta["fallback_used"] is True
    assert len(meta["attempts"]) == 4
    assert all(a["status"] == "unavailable" for a in meta["attempts"])
    assert meta["explanation"] == llm_explainer.FALLBACK_RESPONSES["Daily Wage"]


def test_both_providers_fail_unknown_segment_returns_graceful_message(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")

    with patch("google.genai.Client", side_effect=ConnectionError("network down")), \
         patch("groq.Groq", side_effect=ConnectionError("network down")):
        text = llm_explainer.explain_segment({
            **SAMPLE_ALERT,
            "segment": "Custom Segment",
        })

    assert "Custom Segment" in text
    assert "automated fallback" in text


def test_placeholder_keys_ignored_without_sdk_calls(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")

    with patch("google.genai.Client") as gemini_cls, \
         patch("groq.Groq") as groq_cls:
        meta = llm_explainer.explain_segment_with_meta(SAMPLE_ALERT)

    gemini_cls.assert_not_called()
    groq_cls.assert_not_called()
    assert meta["fallback_used"] is True


def test_input_validation_rejects_bad_alerts_and_bad_timeouts(monkeypatch):
    with pytest.raises(ValueError, match="dict"):
        llm_explainer.explain_segment("not-a-dict")

    with pytest.raises(ValueError, match="segment"):
        llm_explainer.explain_segment({**SAMPLE_ALERT, "segment": ""})

    with pytest.raises(ValueError, match="fhs_day1"):
        llm_explainer.explain_segment({**SAMPLE_ALERT, "fhs_day1": "bad"})

    with pytest.raises(ValueError, match="severity"):
        llm_explainer.explain_segment({**SAMPLE_ALERT, "severity": "EXTREME"})

    with pytest.raises(ValueError, match="timeout"):
        llm_explainer.explain_segment(SAMPLE_ALERT, timeout=-1)

    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "not_a_float")
    with pytest.raises(ValueError, match="LLM_TIMEOUT_SECONDS"):
        llm_explainer.explain_segment(SAMPLE_ALERT)
