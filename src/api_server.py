"""
api_server.py
Flask REST API wrapping existing FinPulse modules.
Serves JSON data to the custom frontend dashboard.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd

from fhs_calculator import compute_segment_fhs
from forecaster import forecast_all_segments
from anomaly import detect_anomalies
from llm_explainer import explain_all_alerts

app = Flask(__name__, static_folder=None)
CORS(app)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Cache ────────────────────────────────────────────────────────────────────
_cache = {}

def _load_data():
    key = "raw_data"
    if key not in _cache:
        print("Loading customer data...")
        _cache[key] = pd.read_csv(DATA_DIR / "historical.csv", parse_dates=["date"])
    return _cache[key]


def _apply_shock(df, shock_pct):
    if shock_pct <= 0:
        return df
    df = df.copy()
    recent_mask = df["date"] >= (df["date"].max() - pd.Timedelta(days=30))
    df.loc[recent_mask, "balance"] -= df.loc[recent_mask, "balance"] * (shock_pct / 100)
    return df


def _get_forecasts(shock_pct):
    key = f"forecasts_{shock_pct}"
    if key not in _cache:
        print(f"Running forecasts (shock={shock_pct}%)... This may take several minutes.")
        df = _apply_shock(_load_data(), shock_pct)
        _cache[key] = forecast_all_segments(df)
    return _cache[key]


def _get_heatmap(shock_pct):
    key = f"heatmap_{shock_pct}"
    if key not in _cache:
        df = _apply_shock(_load_data(), shock_pct)
        _cache[key] = compute_segment_fhs(df)
    return _cache[key]


# ── Static File Serving ──────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filepath>")
def serve_static(filepath):
    return send_from_directory(FRONTEND_DIR, filepath)


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.route("/api/segments")
def api_segments():
    df = _load_data()
    segments = sorted(df["segment"].unique().tolist())
    return jsonify(segments)


@app.route("/api/portfolio")
def api_portfolio():
    shock = int(request.args.get("shock", 0))
    heatmap_df = _get_heatmap(shock)
    critical = int(len(heatmap_df[heatmap_df["risk_label"] == "RED"]))
    warning = int(len(heatmap_df[heatmap_df["risk_label"] == "YELLOW"]))
    healthy = int(len(heatmap_df[heatmap_df["risk_label"] == "GREEN"]))
    return jsonify({
        "total_customers": 1000,
        "critical": critical,
        "warning": warning,
        "healthy": healthy,
    })


@app.route("/api/heatmap")
def api_heatmap():
    shock = int(request.args.get("shock", 0))
    heatmap_df = _get_heatmap(shock)
    result = []
    for _, row in heatmap_df.iterrows():
        result.append({
            "segment": row["segment"],
            "fhs": float(row["fhs"]),
            "risk_label": row["risk_label"],
        })
    return jsonify(result)


@app.route("/api/forecast")
def api_forecast():
    shock = int(request.args.get("shock", 0))
    segment = request.args.get("segment", "Daily Wage")
    forecasts = _get_forecasts(shock)

    if segment not in forecasts:
        return jsonify({"error": f"Segment '{segment}' not found"}), 404

    data = forecasts[segment]
    hist = data["historical"]
    fc = data["forecast"]

    return jsonify({
        "segment": segment,
        "historical": {
            "dates": [d.isoformat() for d in hist.index],
            "values": [round(float(v), 2) for v in hist.values],
        },
        "forecast": {
            "dates": [d.isoformat() for d in fc["ds"]],
            "yhat": [round(float(v), 2) for v in fc["yhat"]],
            "yhat_lower": [round(float(v), 2) for v in fc["yhat_lower"]],
            "yhat_upper": [round(float(v), 2) for v in fc["yhat_upper"]],
        },
    })


@app.route("/api/alerts")
def api_alerts():
    shock = int(request.args.get("shock", 0))
    forecasts = _get_forecasts(shock)
    alerts = detect_anomalies(forecasts)
    explanations = explain_all_alerts(alerts)

    result = []
    for alert in alerts:
        result.append({
            "segment": str(alert["segment"]),
            "severity": str(alert["severity"]),
            "fhs_day1": float(alert["fhs_day1"]),
            "fhs_day30": float(alert["fhs_day30"]),
            "min_lower": float(alert["min_lower"]),
            "declining": bool(alert["declining"]),
            "explanation": str(explanations.get(alert["segment"], "")),
        })
    return jsonify(result)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FinPulse API Server")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    # Pre-load data
    _load_data()
    app.run(debug=False, port=5000, host="0.0.0.0")
