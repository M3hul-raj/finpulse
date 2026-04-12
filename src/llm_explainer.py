"""
llm_explainer.py
Uses Gemini API (gemini-2.0-flash-lite) to generate intervention recommendations.
Live API call attempted first; falls back to curated responses if unavailable.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

FALLBACK_RESPONSES = {
    "Govt/PSU": "The Govt/PSU segment shows a critically declining FHS; the relationship team should immediately schedule financial wellness calls with customers in this segment and offer restructured EMI plans before the next salary cycle.",
    "Gig/Freelance": "With gig workers showing a worsening FHS lower bound, the team should proactively offer a short-term liquidity buffer product or micro-credit line to prevent overdrafts during irregular income months.",
    "Young Professionals": "Young professionals are approaching critical FHS levels; recommend targeted outreach offering budgeting tools and automatic savings nudges to build financial resilience before the 30-day horizon.",
    "Small Business": "The Small Business segment's FHS lower bound signals cash flow stress; the relationship team should offer working capital loan pre-approvals to the most at-risk customers in this segment immediately.",
    "Retirees": "Retirees showing a WARNING-level FHS may be experiencing pension shortfalls; proactive outreach offering fixed deposit re-structuring or pension advance products is recommended within 7 days.",
    "Daily Wage": "Daily wage workers face the highest income volatility; the team should enroll this segment in NatWest's micro-savings auto-sweep program to build a minimum 2-week expense buffer.",
    "Students": "The student segment's FHS lower bound indicates spending exceeding income; recommend deploying targeted in-app nudges with spending caps and overdraft warnings 5 days before predicted zero-balance dates.",
}


def explain_segment(alert: dict) -> str:
    """
    Attempts live Gemini API call; falls back to curated response if unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"""You are a risk analyst at NatWest bank. A monitoring system has flagged this customer segment:

Segment: {alert['segment']}
Financial Health Score Day 1: {alert['fhs_day1']} / 100
Financial Health Score Day 30: {alert['fhs_day30']} / 100
Worst-case lower bound (next 14 days): {alert['min_lower']} / 100
Severity: {alert['severity']}
FHS Declining: {alert['declining']}

Write exactly 1-2 sentences recommending a specific action for the bank's relationship team. Be direct, professional, and actionable. Do not use bullet points."""
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
            )
            return response.text.strip()
        except Exception:
            pass

    return FALLBACK_RESPONSES.get(
        alert["segment"],
        f"Proactive outreach recommended for {alert['segment']} segment due to declining financial health indicators."
    )


def explain_all_alerts(alerts: list[dict]) -> dict:
    """Returns dict: segment -> explanation string."""
    results = {}
    for a in alerts:
        results[a["segment"]] = explain_segment(a)
        time.sleep(2)
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