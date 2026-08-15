"""
anomaly.py
Detects segments at risk based on forecast lower bounds and FHS trends.
"""

import pandas as pd


def detect_anomalies(forecast_results: dict) -> list[dict]:
    """
    Flags segments where:
    - yhat_lower drops below 60 in next 14 days (critical zone)
    - OR FHS is declining over the forecast period
    Returns list of alert dicts.
    """
    alerts = []
    for seg, data in forecast_results.items():
        fc = data["forecast"]
        hist = data["historical"]

        min_lower = float(fc["yhat_lower"].min())
        fhs_start = float(fc["yhat"].iloc[0])
        fhs_end   = float(fc["yhat"].iloc[-1])
        declining = bool(fhs_end < fhs_start - 1.0)

        if min_lower < 35 or declining:
            alerts.append({
                "segment":    seg,
                "min_lower":  round(min_lower, 1),
                "fhs_day1":   round(fhs_start, 1),
                "fhs_day30":  round(fhs_end, 1),
                "declining":  declining,
                "severity":   "CRITICAL" if min_lower < 35 else "WARNING",
            })

    alerts.sort(key=lambda x: x["min_lower"])
    return alerts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from forecaster import forecast_all_segments

    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    results = forecast_all_segments(df)
    alerts = detect_anomalies(results)

    print(f"\n{len(alerts)} segment(s) flagged:\n")
    for a in alerts:
        print(f"[{a['severity']}] {a['segment']}")
        print(f"  FHS: {a['fhs_day1']} → {a['fhs_day30']} | Min lower bound: {a['min_lower']}")
        print(f"  Declining: {a['declining']}\n")