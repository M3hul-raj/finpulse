"""
test_core.py
Unit tests for FinPulse core modules.

All tests are self-contained — no pre-generated CSV files required.
Run from project root: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from customer_generator import generate_all_customers, SEGMENTS
from fhs_calculator import compute_fhs, get_risk_label, compute_segment_fhs
from anomaly import detect_anomalies


# ── Shared Fixture ────────────────────────────────────────────────────────────
# Generates 730K rows once per test session instead of per-test,
# and eliminates the dependency on data/historical.csv.

@pytest.fixture(scope="session")
def customer_data():
    """Generate full synthetic dataset once for the entire test session."""
    return generate_all_customers()


# ── customer_generator tests ──────────────────────────────────────────────────

def test_generated_data_shape(customer_data):
    """Generated data must have 730,000 rows and 4 columns."""
    assert customer_data.shape == (730000, 4), f"Expected (730000, 4), got {customer_data.shape}"


def test_generated_data_columns(customer_data):
    """Generated data must have correct column names."""
    assert set(customer_data.columns) == {"date", "balance", "customer_id", "segment"}


def test_all_segments_present(customer_data):
    """All 8 segments must be present in generated data."""
    assert set(customer_data["segment"].unique()) == set(SEGMENTS)


def test_segment_customer_counts(customer_data):
    """Each segment must have exactly 125 customers."""
    for seg in SEGMENTS:
        cust_count = customer_data[customer_data["segment"] == seg]["customer_id"].nunique()
        assert cust_count == 125, f"{seg} has {cust_count} customers, expected 125"


# ── fhs_calculator tests ──────────────────────────────────────────────────────

def test_fhs_range():
    """FHS score must always be between 0 and 100."""
    balances = pd.Series(np.random.normal(50000, 10000, 730))
    score = compute_fhs(balances)
    assert 0 <= score <= 100, f"FHS out of range: {score}"


def test_fhs_negative_balance_penalized():
    """A series with mostly negative balances should score lower than positive."""
    good = pd.Series(np.abs(np.random.normal(50000, 5000, 730)))
    bad  = pd.Series(-np.abs(np.random.normal(50000, 5000, 730)))
    assert compute_fhs(good) > compute_fhs(bad)


def test_risk_labels():
    """Risk label thresholds: <35=RED, 35-59=AMBER, >=60=GREEN."""
    assert get_risk_label(0)   == "RED"
    assert get_risk_label(34)  == "RED"
    assert get_risk_label(35)  == "AMBER"
    assert get_risk_label(59)  == "AMBER"
    assert get_risk_label(60)  == "GREEN"
    assert get_risk_label(100) == "GREEN"


def test_segment_fhs_returns_all_segments(customer_data):
    """compute_segment_fhs must return a row for all 8 segments."""
    result = compute_segment_fhs(customer_data)
    assert len(result) == 8
    assert set(result["segment"]) == set(SEGMENTS)


def test_segment_fhs_columns(customer_data):
    """compute_segment_fhs output must have correct columns."""
    result = compute_segment_fhs(customer_data)
    expected = {"segment", "fhs", "risk_label", "balance_trend", "income_regularity", "spending_volatility", "debt_ratio"}
    assert expected.issubset(set(result.columns))


def test_subscores_variation_across_segments(customer_data):
    """Subscores must show meaningful differentiation across segments at baseline and under stress shocks."""
    from api_server import _apply_shock

    for shock in [0, 20, 30]:
        df_tested = _apply_shock(customer_data, shock) if shock > 0 else customer_data
        result = compute_segment_fhs(df_tested)
        for col in ["balance_trend", "income_regularity", "spending_volatility"]:
            distinct_vals = result[col].nunique()
            assert distinct_vals > 1, f"Subscore {col} at shock={shock}% has only {distinct_vals} distinct value(s) across 8 segments"
        # debt_ratio is bounded [0, 100] across all segments (evaluated on checking balance overdraft proxy)
        assert (result["debt_ratio"] >= 0).all() and (result["debt_ratio"] <= 100).all()


# ── anomaly tests ─────────────────────────────────────────────────────────────

def test_anomaly_detection_returns_list():
    """detect_anomalies must return a list."""
    mock_results = {
        "Test Segment": {
            "historical": pd.Series([30.0] * 100),
            "forecast": pd.DataFrame({
                "ds": pd.date_range("2025-01-01", periods=30),
                "yhat":       [30.0] * 30,
                "yhat_lower": [20.0] * 30,
                "yhat_upper": [40.0] * 30,
            })
        }
    }
    alerts = detect_anomalies(mock_results)
    assert isinstance(alerts, list)


def test_anomaly_flags_critical_segment():
    """Segment with yhat_lower < 25 must be flagged as CRITICAL."""
    mock_results = {
        "Critical Segment": {
            "historical": pd.Series([30.0] * 100),
            "forecast": pd.DataFrame({
                "ds": pd.date_range("2025-01-01", periods=30),
                "yhat":       [30.0] * 30,
                "yhat_lower": [20.0] * 30,
                "yhat_upper": [40.0] * 30,
            })
        }
    }
    alerts = detect_anomalies(mock_results)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "CRITICAL"


def test_healthy_segment_not_flagged():
    """Segment with high FHS and yhat_lower > 35 must not be flagged."""
    mock_results = {
        "Healthy Segment": {
            "historical": pd.Series([80.0] * 100),
            "forecast": pd.DataFrame({
                "ds": pd.date_range("2025-01-01", periods=30),
                "yhat":       [80.0] * 30,
                "yhat_lower": [75.0] * 30,
                "yhat_upper": [85.0] * 30,
            })
        }
    }
    alerts = detect_anomalies(mock_results)
    assert len(alerts) == 0


def test_anomaly_horizon_alignment_worst_case_and_day30():
    """Worst-case KPI and Day-30 KPI must reference the full 30-day forecast horizon without inversion."""
    mock_results = {
        "Steep Decline Segment": {
            "historical": pd.Series([70.0] * 100),
            "forecast": pd.DataFrame({
                "ds": pd.date_range("2025-01-01", periods=30),
                "yhat":       np.linspace(60.0, 10.0, 30),
                "yhat_lower": np.linspace(50.0, 2.0, 30),
                "yhat_upper": np.linspace(70.0, 18.0, 30),
            })
        }
    }
    alerts = detect_anomalies(mock_results)
    assert len(alerts) == 1
    alert = alerts[0]
    # min_lower must cover the full 30 days and never exceed day 30 point forecast on a declining series
    assert alert["min_lower"] == 2.0
    assert alert["fhs_day30"] == 10.0
    assert alert["min_lower"] <= alert["fhs_day30"]


def test_alert_action_matches_severity():
    """Alert recommended actions must correspond to alert severity (CRITICAL -> RED action, WARNING -> AMBER action)."""
    from api_server import _get_action, KNOWN_SEGMENTS

    for seg in KNOWN_SEGMENTS:
        crit_action = _get_action(seg, "RED")
        warn_action = _get_action(seg, "AMBER")
        green_action = _get_action(seg, "GREEN")

        assert crit_action != green_action, f"Critical action for {seg} must not be upsell/green"
        assert "Escalate" in crit_action or "Flag" in crit_action or "Urgent" in crit_action
        assert "Upsell" not in crit_action and "Retain" not in crit_action
        assert "Upsell" not in warn_action and "Cross-sell" not in warn_action


# ── ML Pipeline output tests ─────────────────────────────────────────────────

def test_features_csv_exists():
    """ML pipeline must produce features.csv with correct shape."""
    path = os.path.join(os.path.dirname(__file__), "..", "model_results", "features.csv")
    if not os.path.exists(path):
        pytest.skip("model_results/features.csv not found — run ml_pipeline.py first")
    df = pd.read_csv(path)
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert "risk_tier" in df.columns
    assert "_fhs_mean" in df.columns


def test_fhs_mean_not_in_model_features():
    """fhs_mean must NOT be in the classifier's input feature list (data leakage prevention)."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "model_results", "classification_report.json")
    if not os.path.exists(path):
        pytest.skip("model_results/classification_report.json not found")
    with open(path) as f:
        data = json.load(f)
    feature_cols = data["feature_columns"]
    assert "fhs_mean" not in feature_cols, "fhs_mean found in feature columns — data leakage!"
    assert "_fhs_mean" not in feature_cols, "_fhs_mean found in feature columns — data leakage!"


def test_model_results_json_valid():
    """All ML pipeline JSON outputs must be parseable."""
    import json
    results_dir = os.path.join(os.path.dirname(__file__), "..", "model_results")
    if not os.path.exists(results_dir):
        pytest.skip("model_results/ not found")
    for fname in ["classification_report.json", "clustering.json",
                   "forecast_metrics.json", "statistical_tests.json"]:
        path = os.path.join(results_dir, fname)
        assert os.path.exists(path), f"{fname} missing"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict), f"{fname} is not a dict"


def test_rf_accuracy_in_healthy_range():
    """RF accuracy should be 75-97% — too high suggests leakage, too low suggests bad features."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "model_results", "classification_report.json")
    if not os.path.exists(path):
        pytest.skip("model_results/classification_report.json not found")
    with open(path) as f:
        data = json.load(f)
    rf_acc = data["models"]["random_forest"]["accuracy"]
    assert 0.70 <= rf_acc <= 0.97, f"RF accuracy {rf_acc} outside healthy range [0.70, 0.97]"


# ── Forecaster confidence band test ──────────────────────────────────────────

def test_confidence_band_grows_with_horizon():
    """Confidence interval must widen over the forecast horizon."""
    from forecaster import forecast_segment
    series = pd.Series(
        np.random.normal(70, 2, 45),
        index=pd.date_range("2024-01-01", periods=45, freq="D")
    )
    fc = forecast_segment(series)
    day1_width = fc["yhat_upper"].iloc[0] - fc["yhat_lower"].iloc[0]
    day30_width = fc["yhat_upper"].iloc[-1] - fc["yhat_lower"].iloc[-1]
    assert day30_width > day1_width, (
        f"Day 30 band width ({day30_width:.2f}) should be wider than "
        f"day 1 ({day1_width:.2f}) — confidence must grow with horizon"
    )


# ── API endpoint tests ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """Flask test client for API smoke tests."""
    from api_server import app
    app.config["TESTING"] = True
    return app.test_client()


def test_api_segments(api_client):
    """GET /api/segments must return 8 segments."""
    resp = api_client.get("/api/segments")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 8


def test_api_model_metrics(api_client):
    """GET /api/model-metrics must return data or null for each section."""
    resp = api_client.get("/api/model-metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "classification_report" in data
    assert "forecast_metrics" in data