"""
api_server.py
Flask REST API for FinPulse (Synthetic NatWest Segment Risk Intelligence).

Cold-start & reliability features:
  - Instant startup via precomputed model_results/baseline_cache.json (shock=0)
  - Lazy CSV loading: data/historical.csv is loaded only on demand (e.g., shock > 0)
  - Thread-safe per-key locking + alert explanation caching
  - Strict query parameter validation (shock in [0, 100], valid segment names)
  - Dedicated /health and /api/health endpoints (alongside /api/segments)
"""

import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from anomaly import detect_anomalies
from fhs_calculator import compute_segment_fhs
from forecaster import forecast_all_segments
from llm_explainer import explain_all_alerts

app = Flask(__name__, static_folder=None)
CORS(app)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model_results"

# Known segments — hardcoded so /api/segments never touches the CSV.
KNOWN_SEGMENTS = [
    "Daily Wage", "Gig/Freelance", "Govt/PSU", "IT Salaried",
    "Retirees", "Small Business", "Students", "Young Professionals",
]

SEGMENT_ACTIONS = {
    ("Daily Wage", "RED"): "Escalate to relationship manager; activate emergency liquidity support and defer scheduled debits.",
    ("Gig/Freelance", "RED"): "Escalate for urgent review; offer emergency micro-loan and suspend non-essential fees.",
    ("Students", "RED"): "Flag for welfare check; activate overdraft protection and connect with student financial aid.",
    ("Small Business", "RED"): "Escalate to business banking team; pre-approve emergency working capital line.",
    ("Govt/PSU", "RED"): "Escalate to senior advisor; review pension adequacy and defer premium payments.",
    ("Retirees", "RED"): "Urgent escalation; schedule face-to-face financial wellness review and restructure fixed deposits.",
    ("IT Salaried", "RED"): "Flag for layoff-risk monitoring; pre-approve emergency credit and offer EMI holiday.",
    ("Young Professionals", "RED"): "Escalate; activate spending freeze alerts and offer short-term credit restructuring.",
    ("Daily Wage", "AMBER"): "Enrol in micro-savings auto-sweep; schedule proactive relationship manager outreach.",
    ("Gig/Freelance", "AMBER"): "Offer income-smoothing product; activate spending pattern alerts.",
    ("Students", "AMBER"): "Deploy in-app budgeting nudges; pre-approve small overdraft buffer.",
    ("Small Business", "AMBER"): "Schedule quarterly cash flow review; offer seasonal credit facility.",
    ("Govt/PSU", "AMBER"): "Monitor for unusual spending; offer financial planning consultation.",
    ("Retirees", "AMBER"): "Schedule pension adequacy check; offer fixed deposit restructuring options.",
    ("IT Salaried", "AMBER"): "Monitor discretionary spending; offer automated savings recommendations.",
    ("Young Professionals", "AMBER"): "Activate smart budgeting tools; offer salary-linked savings product.",
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
_cache: dict = {}
_locks: dict[str, threading.Lock] = {}
_global_lock = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    """Get or create a lock for a specific cache key."""
    with _global_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _load_precomputed_baseline() -> bool:
    """Load precomputed shock=0 heatmap & forecasts into _cache without touching CSV."""
    cache_path = MODEL_DIR / "baseline_cache.json"
    if not cache_path.exists():
        return False
    try:
        with open(cache_path, "r") as f:
            payload = json.load(f)
        heatmap_df = pd.DataFrame(payload["heatmap"])
        forecasts = {}
        for seg, item in payload["forecasts"].items():
            hist = pd.Series(
                item["historical"]["values"],
                index=pd.DatetimeIndex(pd.to_datetime(item["historical"]["dates"]), freq="D"),
            )
            fc = pd.DataFrame({
                "ds": pd.to_datetime(item["forecast"]["dates"]),
                "yhat": item["forecast"]["yhat"],
                "yhat_lower": item["forecast"]["yhat_lower"],
                "yhat_upper": item["forecast"]["yhat_upper"],
            })
            forecasts[seg] = {"historical": hist, "forecast": fc}

        with _get_lock("heatmap_0"):
            _cache.setdefault("heatmap_0", heatmap_df)
        with _get_lock("forecasts_0"):
            _cache.setdefault("forecasts_0", forecasts)
        return True
    except Exception as e:
        print(f"Warning: failed to load baseline_cache.json: {type(e).__name__}")
        return False


def _ensure_baseline_cached(key: str) -> bool:
    """Hydrate baseline shock=0 cache lazily from precomputed artifact if missing."""
    if key in _cache:
        return True
    if key in ("heatmap_0", "forecasts_0"):
        _load_precomputed_baseline()
    return key in _cache


def _load_data() -> pd.DataFrame:
    """Lazily load synthetic historical.csv only when needed (e.g. shock > 0)."""
    key = "raw_data"
    lock = _get_lock(key)
    with lock:
        if key not in _cache:
            csv_path = DATA_DIR / "historical.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Synthetic dataset missing at {csv_path}")
            print("Lazy-loading synthetic customer data...")
            _cache[key] = pd.read_csv(csv_path, parse_dates=["date"])
    return _cache[key]


def _apply_shock(df: pd.DataFrame, shock_pct: int) -> pd.DataFrame:
    """Apply an N% balance reduction over the trailing 30-day window."""
    if shock_pct <= 0:
        return df
    df = df.copy()
    recent_mask = df["date"] >= (df["date"].max() - pd.Timedelta(days=30))
    df.loc[recent_mask, "balance"] -= df.loc[recent_mask, "balance"] * (shock_pct / 100.0)
    return df


def _get_forecasts(shock_pct: int) -> dict:
    key = f"forecasts_{shock_pct}"
    lock = _get_lock(key)
    with lock:
        if key not in _cache:
            if shock_pct == 0 and _load_precomputed_baseline() and key in _cache:
                return _cache[key]
            print(f"Running forecasts (shock={shock_pct}%)...")
            df = _apply_shock(_load_data(), shock_pct)
            _cache[key] = forecast_all_segments(df)
    return _cache[key]


def _get_heatmap(shock_pct: int) -> pd.DataFrame:
    key = f"heatmap_{shock_pct}"
    lock = _get_lock(key)
    with lock:
        if key not in _cache:
            if shock_pct == 0 and _load_precomputed_baseline() and key in _cache:
                return _cache[key]
            df = _apply_shock(_load_data(), shock_pct)
            _cache[key] = compute_segment_fhs(df)
    return _cache[key]


def _parse_shock_param():
    """Validate ?shock= query param as integer in [0, 100]."""
    raw = request.args.get("shock")
    if raw is None or raw == "":
        return 0, None
    try:
        shock = int(raw)
    except ValueError:
        return None, (jsonify({"error": "shock must be an integer between 0 and 100"}), 400)
    if shock < 0 or shock > 100:
        return None, (jsonify({"error": "shock must be an integer between 0 and 100"}), 400)
    return shock, None


# Hydrate precomputed baseline artifacts immediately (instantaneous, no CSV read)
_load_precomputed_baseline()


# ── Static File Serving ──────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filepath>")
def serve_static(filepath):
    return send_from_directory(FRONTEND_DIR, filepath)


# ── Health & API Endpoints ───────────────────────────────────────────────────
@app.route("/health")
@app.route("/api/health")
def api_health():
    """Instant health/readiness check with zero CSV dependency."""
    return jsonify({
        "status": "ok",
        "synthetic_data": True,
        "baseline_ready": ("heatmap_0" in _cache and "forecasts_0" in _cache),
        "csv_present": (DATA_DIR / "historical.csv").exists(),
        "raw_data_loaded": "raw_data" in _cache,
    })


@app.route("/api/segments")
def api_segments():
    """Instant segment list (no CSV dependency)."""
    return jsonify(KNOWN_SEGMENTS)


@app.route("/api/portfolio")
def api_portfolio():
    shock, err = _parse_shock_param()
    if err:
        return err

    key = f"heatmap_{shock}"
    if not _ensure_baseline_cached(key):
        threading.Thread(target=_get_heatmap, args=(shock,), daemon=True).start()
        return jsonify({"status": "computing"}), 202

    heatmap_df = _cache[key]
    critical = int(len(heatmap_df[heatmap_df["risk_label"] == "RED"]))
    warning = int(len(heatmap_df[heatmap_df["risk_label"] == "AMBER"]))
    healthy = int(len(heatmap_df[heatmap_df["risk_label"] == "GREEN"]))
    return jsonify({
        "total_customers": 1000,
        "synthetic": True,
        "critical": critical,
        "warning": warning,
        "healthy": healthy,
    })


@app.route("/api/heatmap")
def api_heatmap():
    shock, err = _parse_shock_param()
    if err:
        return err

    key = f"heatmap_{shock}"
    if not _ensure_baseline_cached(key):
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
            },
        })
    return jsonify(result)


@app.route("/api/forecast")
def api_forecast():
    shock, err = _parse_shock_param()
    if err:
        return err

    segment = request.args.get("segment", "Daily Wage")
    if segment not in KNOWN_SEGMENTS:
        return jsonify({"error": f"Segment '{segment}' not found"}), 404

    key = f"forecasts_{shock}"
    if not _ensure_baseline_cached(key):
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
        "synthetic": True,
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
    shock, err = _parse_shock_param()
    if err:
        return err

    key = f"forecasts_{shock}"
    if not _ensure_baseline_cached(key):
        threading.Thread(target=_get_forecasts, args=(shock,), daemon=True).start()
        return jsonify({"status": "computing"}), 202

    alert_cache_key = f"alerts_payload_{shock}"
    with _get_lock(alert_cache_key):
        if alert_cache_key in _cache:
            return jsonify(_cache[alert_cache_key])

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
        _cache[alert_cache_key] = result
        return jsonify(result)


# ── Model Metrics Endpoint ───────────────────────────────────────────────────
@app.route("/api/model-metrics")
def api_model_metrics():
    """Serve precomputed ML pipeline results from model_results/ directory."""
    metrics = {}
    for filename in [
        "classification_report.json",
        "clustering.json",
        "forecast_metrics.json",
        "statistical_tests.json",
        "sampling_benchmark.json",
    ]:
        filepath = MODEL_DIR / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                metrics[filename.replace(".json", "")] = json.load(f)
        else:
            metrics[filename.replace(".json", "")] = None

    return jsonify(metrics)


# ── Main (local development only) ────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser

    print("=" * 60)
    print("  FinPulse API Server (Synthetic Simulation)")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, port=5000, host="0.0.0.0")
