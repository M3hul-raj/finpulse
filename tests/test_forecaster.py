"""
test_forecaster.py
Unit tests for FinPulse core modules.
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


# ── customer_generator tests ──────────────────────────────────────────────────

def test_generated_data_shape():
    """Generated CSV must have 730,000 rows and 4 columns."""
    df = generate_all_customers()
    assert df.shape == (730000, 4), f"Expected (730000, 4), got {df.shape}"


def test_generated_data_columns():
    """Generated data must have correct column names."""
    df = generate_all_customers()
    assert set(df.columns) == {"date", "balance", "customer_id", "segment"}


def test_all_segments_present():
    """All 8 segments must be present in generated data."""
    df = generate_all_customers()
    assert set(df["segment"].unique()) == set(SEGMENTS)


def test_segment_customer_counts():
    """Each segment must have exactly 125 customers."""
    df = generate_all_customers()
    for seg in SEGMENTS:
        cust_count = df[df["segment"] == seg]["customer_id"].nunique()
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
    """Risk label thresholds must be correct."""
    assert get_risk_label(30)  == "RED"
    assert get_risk_label(59)  == "RED"
    assert get_risk_label(60)  == "YELLOW"
    assert get_risk_label(74)  == "YELLOW"
    assert get_risk_label(75)  == "GREEN"
    assert get_risk_label(90)  == "GREEN"


def test_segment_fhs_returns_all_segments():
    """compute_segment_fhs must return a row for all 8 segments."""
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    result = compute_segment_fhs(df)
    assert len(result) == 8
    assert set(result["segment"]) == set(SEGMENTS)


def test_segment_fhs_columns():
    """compute_segment_fhs output must have correct columns."""
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    result = compute_segment_fhs(df)
    assert set(result.columns) == {"segment", "fhs", "risk_label"}


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