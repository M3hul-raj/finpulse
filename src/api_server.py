"""
api_server.py
Flask REST API wrapping existing FinPulse modules.
Serves JSON data to the custom frontend dashboard.

Performance features:
  - Thread-safe caching with locks (prevents duplicate computation)
  - Non-blocking endpoints: return HTTP 202 while data is computing
  - Deferred background pre-computation of heatmap + forecasts
  - Frontend polls automatically and resolves when data is ready
  - /api/model-metrics serves precomputed ML pipeline results (static JSON)
"""

import sys
import os
import json
import threading
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
MODEL_DIR = Path(__file__).resolve().parent.parent / "model_results"

# Known segments — hardcoded so /api/segments never touches the CSV.
# Must match customer_generator.SEGMENTS.
KNOWN_SEGMENTS = [
    "Daily Wage", "Gig/Freelance", "Govt/PSU", "IT Salaried",
    "Retirees", "Small Business", "Students", "Young Professionals",
]

# ── Tier-Aware Recommended Actions ───────────────────────────────────────────
# Keyed by (segment, risk_tier) so action text updates when stress slider
# changes a segment's risk level.
SEGMENT_ACTIONS = {
    # RED tier → escalate / intervene
    ("Daily Wage", "RED"): "Escalate to relationship manager; activate emergency liquidity support and defer scheduled debits.",
    ("Gig/Freelance", "RED"): "Escalate for urgent review; offer emergency micro-loan and suspend non-essential fees.",
    ("Students", "RED"): "Flag for welfare check; activate overdraft protection and connect with student financial aid.",
    ("Small Business", "RED"): "Escalate to business banking team; pre-approve emergency working capital line.",
    ("Govt/PSU", "RED"): "Escalate to senior advisor; review pension adequacy and defer premium payments.",
    ("Retirees", "RED"): "Urgent escalation; schedule face-to-face financial wellness review and restructure fixed deposits.",
    ("IT Salaried", "RED"): "Flag for layoff-risk monitoring; pre-approve emergency credit and offer EMI holiday.",
    ("Young Professionals", "RED"): "Escalate; activate spending freeze alerts and offer short-term credit restructuring.",
    # AMBER tier → monitor / offer support
    ("Daily Wage", "AMBER"): "Enrol in micro-savings auto-sweep; schedule proactive relationship manager outreach.",
    ("Gig/Freelance", "AMBER"): "Offer income-smoothing product; activate spending pattern alerts.",
    ("Students", "AMBER"): "Deploy in-app budgeting nudges; pre-approve small overdraft buffer.",
    ("Small Business", "AMBER"): "Schedule quarterly cash flow review; offer seasonal credit facility.",
    ("Govt/PSU", "AMBER"): "Monitor for unusual spending; offer financial planning consultation.",
    ("Retirees", "AMBER"): "Schedule pension adequacy check; offer fixed deposit restructuring options.",
    ("IT Salaried", "AMBER"): "Monitor discretionary spending; offer automated savings recommendations.",
    ("Young Professionals", "AMBER"): "Activate smart budgeting tools; offer salary-linked savings product.",
    # GREEN tier → retain / upsell
    ("Daily Wage", "GREEN"): "Retain with loyalty incentives; introduce savings goal products.",
    ("Gig/Freelance", "GREEN"): "Offer premium freelancer banking package; introduce investment options.",
    ("Students", "GREEN"): "Graduate to Young Professional products; offer first credit card.",
    ("Small Business", "GREEN"): "Upsell business growth credit; offer trade finance products.",
    ("Govt/PSU", "GREEN"): "Cross-sell investment products; offer premium banking upgrade.",
    ("Retirees", "GREEN"): "Offer wealth management consultation; introduce estate planning services.",
    ("IT Salaried", "GREEN"): "Upsell premium banking; offer mortgage and investment products.",
    ("Young Professionals", "GREEN"): "Offer first home loan pre-approval; introduce systematic investment plans.",
}

def _get_action(segment: str, risk_label: str) -> str:
    """Get tier-aware recommended action for a segment."""
    return SEGMENT_ACTIONS.get((segment, risk_label), "Monitor and review at next scheduled assessment.")

# ── Thread-safe Cache ────────────────────────────────────────────────────────
_cache = {}
_locks = {}
_global_lock = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    """Get or create a lock for a specific cache key."""
    with _global_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _load_data():
    key = "raw_data"
    lock = _get_lock(key)
    with lock:
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
    lock = _get_lock(key)
    with lock:
        if key not in _cache:
            print(f"Running forecasts (shock={shock_pct}%)... This may take a few minutes.")
            df = _apply_shock(_load_data(), shock_pct)
            _cache[key] = forecast_all_segments(df)
    return _cache[key]


def _get_heatmap(shock_pct):
    key = f"heatmap_{shock_pct}"
    lock = _get_lock(key)
    with lock:
        if key not in _cache:
            df = _apply_shock(_load_data(), shock_pct)
            _cache[key] = compute_segment_fhs(df)
    return _cache[key]


def _background_precompute():
    """Background thread: pre-compute heatmap + forecasts after server is live."""
    import time
    time.sleep(2)  # Let gunicorn bind the port before doing heavy work
    print("Background: Pre-computing baseline heatmap...")
    _get_heatmap(0)
    print("Background: Heatmap ready!")
    print("Background: Pre-computing baseline forecasts...")
    _get_forecasts(0)
    print("Background: Baseline forecasts ready!")


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
    # Hardcoded — no CSV dependency, instant response for Render health check
    return jsonify(KNOWN_SEGMENTS)


@app.route("/api/portfolio")
def api_portfolio():
    shock = request.args.get("shock", 0, type=int)

    # Non-blocking: return 202 if heatmap not cached yet
    key = f"heatmap_{shock}"
    if key not in _cache:
        threading.Thread(target=_get_heatmap, args=(shock,), daemon=True).start()
        return jsonify({"status": "computing"}), 202

    heatmap_df = _cache[key]
    critical = int(len(heatmap_df[heatmap_df["risk_label"] == "RED"]))
    warning = int(len(heatmap_df[heatmap_df["risk_label"] == "AMBER"]))
    healthy = int(len(heatmap_df[heatmap_df["risk_label"] == "GREEN"]))
    return jsonify({
        "total_customers": 1000,
        "critical": critical,
        "warning": warning,
        "healthy": healthy,
    })


@app.route("/api/heatmap")
def api_heatmap():
    shock = request.args.get("shock", 0, type=int)

    # Non-blocking: return 202 if heatmap not cached yet
    key = f"heatmap_{shock}"
    if key not in _cache:
        threading.Thread(target=_get_heatmap, args=(shock,), daemon=True).start()
        return jsonify({"status": "computing"}), 202

    heatmap_df = _cache[key]
    result = []
    for _, row in heatmap_df.iterrows():
        result.append({
            "segment": row["segment"],
            "fhs": float(row["fhs"]),
            "risk_label": row["risk_label"],
            "subscores": {
                "balance_trend": float(row.get("balance_trend", 50.0)),
                "income_regularity": float(row.get("income_regularity", 50.0)),
                "spending_volatility": float(row.get("spending_volatility", 50.0)),
                "debt_ratio": float(row.get("debt_ratio", 50.0)),
            }
        })
    return jsonify(result)


@app.route("/api/forecast")
def api_forecast():
    shock = request.args.get("shock", 0, type=int)
    segment = request.args.get("segment", "Daily Wage")

    # Non-blocking: if forecasts aren't cached yet, return 202 and let background compute
    key = f"forecasts_{shock}"
    if key not in _cache:
        threading.Thread(target=_get_forecasts, args=(shock,), daemon=True).start()
        return jsonify({"status": "computing"}), 202

    forecasts = _cache[key]

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
    shock = request.args.get("shock", 0, type=int)

    # Non-blocking: if forecasts aren't cached yet, return 202 and let background compute
    key = f"forecasts_{shock}"
    if key not in _cache:
        threading.Thread(target=_get_forecasts, args=(shock,), daemon=True).start()
        return jsonify({"status": "computing"}), 202

    forecasts = _cache[key]
    alerts = detect_anomalies(forecasts)
    explanations = explain_all_alerts(alerts)

    result = []
    for alert in alerts:
        seg = str(alert["segment"])
        sev = str(alert["severity"])
        action_tier = "RED" if sev == "CRITICAL" else "AMBER"
        result.append({
            "segment": seg,
            "severity": sev,
            "fhs_day1": float(alert["fhs_day1"]),
            "fhs_day30": float(alert["fhs_day30"]),
            "min_lower": float(alert["min_lower"]),
            "declining": bool(alert["declining"]),
            "explanation": str(explanations.get(seg, "")),
            "recommended_action": _get_action(seg, action_tier),
        })
    return jsonify(result)


# ── Model Metrics Endpoint ───────────────────────────────────────────────────
@app.route("/api/model-metrics")
def api_model_metrics():
    """Serve precomputed ML pipeline results from model_results/ directory."""
    metrics = {}
    for filename in ["classification_report.json", "clustering.json",
                     "forecast_metrics.json", "statistical_tests.json"]:
        filepath = MODEL_DIR / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                metrics[filename.replace(".json", "")] = json.load(f)
        else:
            metrics[filename.replace(".json", "")] = None

    return jsonify(metrics)


# ── Startup ──────────────────────────────────────────────────────────────────
# All heavy work (CSV read, FHS computation, forecasting) runs in a background
# thread with a 2-second delay, so gunicorn binds the port FIRST and the Render
# health check (/api/segments) passes instantly.
threading.Thread(target=_background_precompute, daemon=True).start()


# ── Main (local development only) ────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser

    print("=" * 60)
    print("  FinPulse API Server")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    # Auto-open browser
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, port=5000, host="0.0.0.0")
