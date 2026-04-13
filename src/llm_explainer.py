"""
llm_explainer.py
Multi-provider AI engine for generating intervention recommendations.

Provider chain (automatic fallback):
  1. Google Gemini API  (gemini-2.0-flash-lite)  — best quality
  2. Groq API           (llama-3.3-70b-versatile) — fastest inference, open-source
  3. Curated fallback    (hardcoded, segment-specific) — guaranteed availability

Live API calls are attempted first. If a provider fails (rate limit, network, etc.),
the system automatically falls back to the next provider in the chain.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Curated Fallback Responses ──────────────────────────────────────────────
# Used as a last resort when all AI providers are unavailable.
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


def _build_prompt(alert: dict) -> str:
    """Builds a standardized prompt for any LLM provider."""
    return f"""You are a risk analyst at NatWest bank. A monitoring system has flagged this customer segment:

Segment: {alert['segment']}
Financial Health Score Day 1: {alert['fhs_day1']} / 100
Financial Health Score Day 30: {alert['fhs_day30']} / 100
Worst-case lower bound (next 14 days): {alert['min_lower']} / 100
Severity: {alert['severity']}
FHS Declining: {alert['declining']}

Write exactly 1-2 sentences recommending a specific action for the bank's relationship team. Be direct, professional, and actionable. Do not use bullet points."""


def _try_gemini(prompt: str) -> str | None:
    """Attempt generation via Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        text = response.text.strip()
        if text:
            print(f"    [AI] Gemini responded successfully")
            return text
    except Exception as e:
        print(f"    [AI] Gemini unavailable: {type(e).__name__}")
    return None


def _try_groq(prompt: str) -> str | None:
    """Attempt generation via Groq API (Llama 3.3 70B)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior risk analyst at NatWest bank. Provide concise, actionable recommendations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        if text:
            print(f"    [AI] Groq (Llama 3.3 70B) responded successfully")
            return text
    except Exception as e:
        print(f"    [AI] Groq unavailable: {type(e).__name__}")
    return None


def explain_segment(alert: dict) -> str:
    """
    Generates an AI explanation for a flagged segment.
    Falls through the provider chain until one succeeds.
    """
    prompt = _build_prompt(alert)

    # Provider chain: Gemini → Groq → Curated fallback
    result = _try_gemini(prompt)
    if result:
        return result

    result = _try_groq(prompt)
    if result:
        return result

    print(f"    [AI] Using curated fallback for {alert['segment']}")
    return FALLBACK_RESPONSES.get(
        alert["segment"],
        f"Proactive outreach recommended for {alert['segment']} segment due to declining financial health indicators.",
    )


def explain_all_alerts(alerts: list[dict]) -> dict:
    """Returns dict: segment → explanation string."""
    results = {}
    for a in alerts:
        print(f"  Generating AI explanation for: {a['segment']}...")
        results[a["segment"]] = explain_segment(a)
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