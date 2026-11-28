"""
llm_explainer.py
Multi-provider AI engine for generating intervention recommendations.

Cross-provider failover waterfall (all calls bounded by explicit timeout):
  1. Google Gemini API   (gemini-3.5-flash-lite)
  2. Groq LPU API        (openai/gpt-oss-120b)
  3. Google Gemini API   (gemini-flash-lite-latest)
  4. Groq LPU API        (llama-3.1-8b-instant)
  5. Curated fallback    (deterministic segment-specific response)

Design notes:
  - Placeholder or blank API keys are ignored before creating client SDK calls.
  - Both providers enforce explicit HTTP timeouts (default 10s; env LLM_TIMEOUT_SECONDS).
  - Never logs secret API keys.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TIMEOUT_SECONDS = 10.0

_PLACEHOLDER_PREFIXES = ("your_", "changeme", "replace_", "example")

# ── Curated Fallback Responses ───────────────────────────────────────────────
FALLBACK_RESPONSES = {
    "Govt/PSU": "The Govt/PSU segment shows a critically declining FHS; the relationship team should immediately schedule financial wellness calls with customers in this segment and offer restructured EMI plans before the next salary cycle.",
    "Gig/Freelance": "With gig workers showing a worsening FHS lower bound, the team should proactively offer a short-term liquidity buffer product or micro-credit line to prevent overdrafts during irregular income months.",
    "Young Professionals": "Young professionals are approaching critical FHS levels; recommend targeted outreach offering budgeting tools and automatic savings nudges to build financial resilience before the 30-day horizon.",
    "Small Business": "The Small Business segment's FHS lower bound signals cash flow stress; the relationship team should offer working capital loan pre-approvals to the most at-risk customers in this segment immediately.",
    "Retirees": "Retirees showing declining FHS may be experiencing pension shortfalls; proactive outreach offering fixed deposit re-structuring or pension advance products is recommended within 7 days.",
    "Daily Wage": "Daily wage workers face the highest income volatility; the team should enroll this segment in NatWest's micro-savings auto-sweep program to build a minimum 2-week expense buffer.",
    "Students": "The student segment's FHS lower bound indicates spending exceeding income; recommend deploying targeted in-app nudges with spending caps and overdraft warnings 5 days before predicted zero-balance dates.",
    "IT Salaried": "The IT Salaried segment is trending toward at-risk levels; recommend offering automated expense-tracking alerts and pre-approved emergency credit lines to hedge against tech layoff-driven income disruption.",
}


def _resolve_timeout(timeout: float | None = None) -> float:
    """Resolve positive timeout in seconds from argument or LLM_TIMEOUT_SECONDS env var."""
    if timeout is None:
        raw = os.getenv("LLM_TIMEOUT_SECONDS")
        if raw is not None and raw.strip():
            try:
                timeout = float(raw)
            except ValueError as exc:
                raise ValueError(f"LLM_TIMEOUT_SECONDS must be a positive number, got '{raw}'") from exc
        else:
            timeout = DEFAULT_TIMEOUT_SECONDS
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")
    return float(timeout)


def _get_valid_key(env_var: str) -> str | None:
    """Return a non-empty, non-placeholder API key from environment variables."""
    val = (os.getenv(env_var) or "").strip()
    if not val:
        return None
    lower = val.lower()
    if any(lower.startswith(p) for p in _PLACEHOLDER_PREFIXES):
        return None
    return val


def _validate_alert(alert: Any) -> dict[str, Any]:
    """Validate required fields and numeric bounds on an alert payload."""
    if not isinstance(alert, dict):
        raise ValueError("alert must be a dict")

    segment = alert.get("segment")
    if not isinstance(segment, str) or not segment.strip():
        raise ValueError("alert['segment'] must be a non-empty string")

    required_nums = ("fhs_day1", "fhs_day30", "min_lower")
    cleaned: dict[str, Any] = {"segment": segment.strip()}
    for field in required_nums:
        val = alert.get(field)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"alert['{field}'] must be numeric")
        cleaned[field] = float(val)

    severity = alert.get("severity", "WARNING")
    if not isinstance(severity, str) or severity not in {"CRITICAL", "WARNING", "INFO"}:
        raise ValueError(f"alert['severity'] must be CRITICAL, WARNING, or INFO, got {severity!r}")
    cleaned["severity"] = severity

    declining = alert.get("declining", False)
    if not isinstance(declining, bool):
        raise ValueError("alert['declining'] must be a boolean")
    cleaned["declining"] = declining

    return cleaned


def _build_prompt(alert: dict[str, Any]) -> str:
    """Build a standardized prompt for any LLM provider."""
    clean = _validate_alert(alert)
    return f"""You are a risk analyst at NatWest bank. A monitoring system has flagged this customer segment:

Segment: {clean['segment']}
Financial Health Score Day 1: {clean['fhs_day1']} / 100
Financial Health Score Day 30: {clean['fhs_day30']} / 100
Worst-case lower bound (next 14 days): {clean['min_lower']} / 100
Severity: {clean['severity']}
FHS Declining: {clean['declining']}

Write exactly 1-2 sentences recommending a specific action for the bank's relationship team. Be direct, professional, and actionable. Do not use bullet points."""


def _try_gemini(prompt: str, model: str = "gemini-3.5-flash-lite", timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str | None:
    """Attempt generation via Google Gemini API with explicit millisecond HTTP timeout."""
    api_key = _get_valid_key("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        timeout_ms = max(1, int(round(timeout * 1000)))
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = (getattr(response, "text", None) or "").strip()
        if text:
            print(f"    [AI] Gemini ({model}) responded successfully")
            return text
    except Exception as e:
        print(f"    [AI] Gemini ({model}) unavailable: {type(e).__name__}")
    return None


def _try_groq(prompt: str, model: str = "openai/gpt-oss-120b", timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str | None:
    """Attempt generation via Groq API with explicit second HTTP timeout."""
    api_key = _get_valid_key("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior risk analyst at NatWest bank. Provide concise, actionable recommendations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        choices = getattr(response, "choices", None) or []
        content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
        text = (content or "").strip()
        if text:
            print(f"    [AI] Groq ({model}) responded successfully")
            return text
    except Exception as e:
        print(f"    [AI] Groq ({model}) unavailable: {type(e).__name__}")
    return None


def _graceful_fallback(segment: str) -> str:
    """Return a deterministic, graceful offline message when all providers fail."""
    return FALLBACK_RESPONSES.get(
        segment,
        f"Proactive outreach recommended for {segment} segment due to declining financial health indicators (automated fallback: AI providers unavailable).",
    )


def explain_segment_with_meta(alert: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    """Generate an explanation with full provider failover metadata."""
    clean = _validate_alert(alert)
    resolved_timeout = _resolve_timeout(timeout)
    prompt = _build_prompt(clean)

    chain = [
        ("gemini", "gemini-3.5-flash-lite", _try_gemini),
        ("groq", "openai/gpt-oss-120b", _try_groq),
        ("gemini", "gemini-flash-lite-latest", _try_gemini),
        ("groq", "llama-3.1-8b-instant", _try_groq),
    ]

    attempts: list[dict[str, str]] = []
    for provider, model, fn in chain:
        result = fn(prompt, model=model, timeout=resolved_timeout)
        if result:
            attempts.append({"provider": provider, "model": model, "status": "ok"})
            return {
                "explanation": result,
                "provider": provider,
                "model": model,
                "fallback_used": False,
                "attempts": attempts,
            }
        attempts.append({"provider": provider, "model": model, "status": "unavailable"})

    print(f"    [AI] Using curated fallback for {clean['segment']}")
    return {
        "explanation": _graceful_fallback(clean["segment"]),
        "provider": "curated_fallback",
        "model": "deterministic",
        "fallback_used": True,
        "attempts": attempts,
    }


def explain_segment(alert: dict[str, Any], timeout: float | None = None) -> str:
    """Generate an AI intervention recommendation with automatic cross-provider failover."""
    return explain_segment_with_meta(alert, timeout=timeout)["explanation"]


def explain_all_alerts(alerts: list[dict[str, Any]], timeout: float | None = None) -> dict[str, str]:
    """Return dict mapping segment name -> explanation string."""
    results = {}
    for alert in alerts:
        clean = _validate_alert(alert)
        print(f"  Generating AI explanation for: {clean['segment']}...")
        results[clean["segment"]] = explain_segment(clean, timeout=timeout)
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from forecaster import forecast_all_segments
    from anomaly import detect_anomalies
    import pandas as pd

    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    print("Running forecasts...")
    results = forecast_all_segments(df)
    alerts = detect_anomalies(results)

    print(f"\nGenerating AI explanations for {len(alerts)} alerts...\n")
    explanations = explain_all_alerts(alerts)
    for seg, text in explanations.items():
        print(f"[{seg}]\n  {text}\n")
