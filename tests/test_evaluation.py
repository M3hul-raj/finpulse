"""
test_evaluation.py
Integrity tests verifying the ML evaluation pipeline is leakage-free,
properly grouped and time-aware, includes required baselines and CIs,
and performs rolling-origin forecast backtesting against naive & seasonal-naive.
"""

import json
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "model_results")


def _load_json(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found — run python src/ml_pipeline.py first")
    with open(path, "r") as f:
        return json.load(f)


def test_classification_excludes_all_label_defining_features():
    cr = _load_json("classification_report.json")
    forbidden = {
        "fhs_mean", "_fhs_mean", "fhs_std", "bal_trend",
        "income_regularity", "pct_negative_days", "vel_mean",
        "bal_mean", "bal_std", "bal_cv",
    }
    used = set(cr["feature_columns"])
    overlap = used & forbidden
    assert not overlap, f"Label-defining features present in primary classifier: {overlap}"


def test_classification_includes_required_baselines_and_cis():
    cr = _load_json("classification_report.json")
    for name in ("majority_class", "threshold_rule", "logistic_regression", "random_forest"):
        assert name in cr["models"], f"Missing baseline/model: {name}"
        m = cr["models"][name]
        assert "accuracy" in m and "macro_f1" in m
        assert len(m["accuracy_ci95"]) == 2
        assert len(m["macro_f1_ci95"]) == 2
        assert m["accuracy_ci95"][0] <= m["accuracy"] <= m["accuracy_ci95"][1]


def test_leakage_audit_and_forward_decline_task_present():
    cr = _load_json("classification_report.json")
    audit = cr["leakage_audit"]
    assert audit["same_window_leakage_detected"] is True
    assert len(audit["feature_audit"]) == 11
    assert audit["same_window_benchmark"]["threshold_rule"]["accuracy"] > 0.90

    fwd = cr["forward_decline_task"]["next_30d_fhs_decline_le_2"]
    assert fwd["train_t_range"][1] < fwd["test_t_range"][0]
    assert fwd["train_customers"] + fwd["test_customers"] == 1000
    for name in ("majority_class", "threshold_rule", "logistic_regression", "random_forest"):
        assert name in fwd["models"]
        assert "macro_f1_ci95" in fwd["models"][name]


def test_forecast_metrics_has_rolling_origin_and_seasonal_naive():
    fm = _load_json("forecast_metrics.json")
    assert "Rolling-origin" in fm["summary"]["evaluation_method"]
    for seg, metrics in fm["per_segment"].items():
        for model in ("naive", "seasonal_naive_7", "seasonal_naive_30", "holt_prod_45d", "holt_winters_m30"):
            assert model in metrics, f"{model} missing for {seg}"
            assert "mae" in metrics[model] and "mape" in metrics[model] and "rmse" in metrics[model]
    wins_losses = fm["summary"]["wins_and_losses"]
    assert "holt_prod_45d_vs_naive" in wins_losses
    assert "hw_m30_vs_naive" in wins_losses
    assert "hw_m30_vs_snaive_30" in wins_losses


def test_sampling_benchmark_json_has_measured_numbers():
    sb = _load_json("sampling_benchmark.json")
    assert sb["sampling_config"]["sampled_per_segment"] == 30
    assert sb["sampling_config"]["full_per_segment"] == 125
    assert sb["heatmap"]["scalar_sampling_speedup"] > 1.0
    assert sb["forecaster"]["scalar_sampling_speedup"] > 1.0
    assert sb["forecaster"]["vectorization_vs_scalar_30c_speedup"] > 1.0
