"""
llm_explainer.py
Uses Gemini API to generate intervention recommendations
for flagged segments in plain English for bank managers.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def explain_segment(alert: dict) -> str:
    """
    Takes an alert dict and returns a 1-2 sentence
    intervention recommendation for a bank manager.
    """
    prompt = f"""You are a risk analyst at NatWest bank. A monitoring system has flagged this customer segment:

Segment: {alert['segment']}
Financial Health Score Day 1: {alert['fhs_day1']} / 100
Financial Health Score Day 30: {alert['fhs_day30']} / 100
Worst-case lower bound (next 14 days): {alert['min_lower']} / 100
Severity: {alert['severity']}
FHS Declining: {alert['declining']}

Write exactly 1-2 sentences recommending a specific action for the bank's relationship team. Be direct, professional, and actionable. Do not use bullet points."""

    try:
        response = _client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini error: {e}] Proactive outreach recommended for {alert['segment']} segment."


import time

def explain_all_alerts(alerts: list[dict]) -> dict:
    """Returns dict: segment -> explanation string."""
    results = {}
    for a in alerts:
        results[a["segment"]] = explain_segment(a)
        time.sleep(5)  # stay under free tier rate limit
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
        print(f"[{seg}]")
        print(f"  {text}\n")