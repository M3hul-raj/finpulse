"""
benchmark.py
Measures the actual runtime speedup and accuracy impact of customer sampling
(30 vs 125 customers/segment) and vectorized FHS calculation.

Run: python src/benchmark.py
Saves results to model_results/sampling_benchmark.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from anomaly import detect_anomalies
from fhs_calculator import compute_fhs, compute_segment_fhs, get_risk_label, sample_segment_customers
from forecaster import FORECAST_DAYS, LOOKBACK_DAYS, ROLLING_WINDOW, forecast_all_segments, forecast_segment

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model_results"


def _scalar_compute_segment_fhs(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    """Original per-customer scalar loop implementation for apples-to-apples sampling timing."""
    results = []
    for segment, seg_df in df.groupby("segment"):
        pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
        pivot.index = pd.to_datetime(pivot.index)
        selected = sample_segment_customers(list(pivot.columns), segment, sample_size)
        pivot = pivot[selected]
        fhs_vals = [compute_fhs(pivot[cid]) for cid in pivot.columns]
        avg_fhs = float(np.mean(fhs_vals))
        results.append({
            "segment": segment,
            "fhs": round(avg_fhs, 2),
            "risk_label": get_risk_label(avg_fhs),
        })
    return pd.DataFrame(results).sort_values("segment").reset_index(drop=True)


def _scalar_forecast_all_segments(
    df: pd.DataFrame, sample_customers: int | None, lookback_days: int = LOOKBACK_DAYS
) -> dict:
    """Original scalar loop implementation for apples-to-apples sampling timing."""
    results = {}
    for seg in df["segment"].unique():
        seg_df = df[df["segment"] == seg]
        pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()
        selected = sample_segment_customers(list(pivot.columns), seg, sample_customers)
        pivot = pivot[selected]
        dates = pivot.index[-lookback_days:]
        fhs_vals = []
        for end_date in dates:
            start_date = end_date - pd.Timedelta(days=ROLLING_WINDOW)
            window_df = pivot.loc[start_date:end_date]
            cust_fhs = [compute_fhs(window_df[col]) for col in window_df.columns]
            fhs_vals.append(float(np.mean(cust_fhs)))
        series = pd.Series(fhs_vals, index=pd.DatetimeIndex(dates, freq="D"))
        results[seg] = {"historical": series, "forecast": forecast_segment(series, FORECAST_DAYS)}
    return results


def run_sampling_benchmark(df: pd.DataFrame, repeats: int = 3, save: bool = True) -> dict:
    """Measure full (125/segment) vs sampled (30/segment) runtime and accuracy impact."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Customer Sampling (30 vs 125/segment) & Vectorization")
    print("=" * 70)

    # 1. Heatmap runtime: scalar & vectorized
    t_sca_30, t_sca_125 = [], []
    t_vec_30, t_vec_125 = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        hm_sca_30 = _scalar_compute_segment_fhs(df, 30)
        t_sca_30.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        hm_sca_125 = _scalar_compute_segment_fhs(df, None)
        t_sca_125.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        hm_vec_30 = compute_segment_fhs(df, sample_size=30).sort_values("segment").reset_index(drop=True)
        t_vec_30.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        hm_vec_125 = compute_segment_fhs(df, sample_size=None).sort_values("segment").reset_index(drop=True)
        t_vec_125.append(time.perf_counter() - t0)

    hm_diff = np.abs(hm_vec_30["fhs"].to_numpy() - hm_vec_125["fhs"].to_numpy())
    hm_label_agree = int((hm_vec_30["risk_label"] == hm_vec_125["risk_label"]).sum())

    # 2. Forecaster runtime: scalar vs vectorized (30c/45d vs 125c/45d)
    t0 = time.perf_counter()
    _ = _scalar_forecast_all_segments(df, 30, LOOKBACK_DAYS)
    dt_fc_sca_30 = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = _scalar_forecast_all_segments(df, None, LOOKBACK_DAYS)
    dt_fc_sca_125 = time.perf_counter() - t0

    t0 = time.perf_counter()
    fc_vec_30 = forecast_all_segments(df, sample_customers=30, lookback_days=LOOKBACK_DAYS)
    dt_fc_vec_30 = time.perf_counter() - t0

    t0 = time.perf_counter()
    fc_vec_125 = forecast_all_segments(df, sample_customers=None, lookback_days=LOOKBACK_DAYS)
    dt_fc_vec_125 = time.perf_counter() - t0

    per_seg_fc = {}
    hist_maes, fc_maes = [], []
    for seg in sorted(fc_vec_125.keys()):
        h30 = fc_vec_30[seg]["historical"].to_numpy()
        h125 = fc_vec_125[seg]["historical"].to_numpy()
        y30 = fc_vec_30[seg]["forecast"]["yhat"].to_numpy()
        y125 = fc_vec_125[seg]["forecast"]["yhat"].to_numpy()
        h_mae = float(np.mean(np.abs(h30 - h125)))
        f_mae = float(np.mean(np.abs(y30 - y125)))
        hist_maes.append(h_mae)
        fc_maes.append(f_mae)
        per_seg_fc[seg] = {
            "hist_fhs_mae": round(h_mae, 4),
            "forecast_yhat_mae": round(f_mae, 4),
            "day30_sampled": round(float(y30[-1]), 2),
            "day30_full": round(float(y125[-1]), 2),
            "day30_diff": round(float(y30[-1] - y125[-1]), 2),
        }

    alerts_30 = detect_anomalies(fc_vec_30)
    alerts_125 = detect_anomalies(fc_vec_125)
    alert_segs_30 = sorted([a["segment"] for a in alerts_30])
    alert_segs_125 = sorted([a["segment"] for a in alerts_125])

    report = {
        "dataset": "synthetic (seed=42, 1000 customers = 125/segment, 730 days)",
        "sampling_config": {"sampled_per_segment": 30, "full_per_segment": 125, "lookback_days": LOOKBACK_DAYS},
        "heatmap": {
            "scalar_30c_median_s": round(float(np.median(t_sca_30)), 4),
            "scalar_125c_median_s": round(float(np.median(t_sca_125)), 4),
            "scalar_sampling_speedup": round(float(np.median(t_sca_125) / np.median(t_sca_30)), 2),
            "vectorized_30c_median_s": round(float(np.median(t_vec_30)), 4),
            "vectorized_125c_median_s": round(float(np.median(t_vec_125)), 4),
            "vectorized_sampling_speedup": round(float(np.median(t_vec_125) / np.median(t_vec_30)), 2),
            "vectorization_vs_scalar_30c_speedup": round(float(np.median(t_sca_30) / np.median(t_vec_30)), 2),
            "fhs_mae_vs_full": round(float(np.mean(hm_diff)), 4),
            "fhs_max_abs_diff": round(float(np.max(hm_diff)), 4),
            "risk_label_agreement": f"{hm_label_agree}/{len(hm_vec_30)}",
            "per_segment": [
                {
                    "segment": hm_vec_30.loc[i, "segment"],
                    "fhs_sampled_30": float(hm_vec_30.loc[i, "fhs"]),
                    "fhs_full_125": float(hm_vec_125.loc[i, "fhs"]),
                    "diff": round(float(hm_vec_30.loc[i, "fhs"] - hm_vec_125.loc[i, "fhs"]), 2),
                    "risk_sampled": hm_vec_30.loc[i, "risk_label"],
                    "risk_full": hm_vec_125.loc[i, "risk_label"],
                }
                for i in range(len(hm_vec_30))
            ],
        },
        "forecaster": {
            "lookback_days": LOOKBACK_DAYS,
            "scalar_30c_s": round(float(dt_fc_sca_30), 3),
            "scalar_125c_s": round(float(dt_fc_sca_125), 3),
            "scalar_sampling_speedup": round(float(dt_fc_sca_125 / dt_fc_sca_30), 2),
            "vectorized_30c_s": round(float(dt_fc_vec_30), 3),
            "vectorized_125c_s": round(float(dt_fc_vec_125), 3),
            "vectorized_sampling_speedup": round(float(dt_fc_vec_125 / dt_fc_vec_30), 2),
            "vectorization_vs_scalar_30c_speedup": round(float(dt_fc_sca_30 / dt_fc_vec_30), 2),
            "mean_hist_fhs_mae": round(float(np.mean(hist_maes)), 4),
            "mean_forecast_yhat_mae": round(float(np.mean(fc_maes)), 4),
            "max_forecast_yhat_mae": round(float(np.max(fc_maes)), 4),
            "alert_segments_sampled": alert_segs_30,
            "alert_segments_full": alert_segs_125,
            "alerts_exact_match": alert_segs_30 == alert_segs_125,
            "per_segment": per_seg_fc,
        },
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODEL_DIR / "sampling_benchmark.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  [OK] {out_path} saved")

    print(
        f"  Heatmap:   scalar 125c={report['heatmap']['scalar_125c_median_s']}s vs 30c={report['heatmap']['scalar_30c_median_s']}s "
        f"({report['heatmap']['scalar_sampling_speedup']}x); vectorized 30c={report['heatmap']['vectorized_30c_median_s']}s; "
        f"MAE={report['heatmap']['fhs_mae_vs_full']} pts, agree={report['heatmap']['risk_label_agreement']}"
    )
    print(
        f"  Forecast ({LOOKBACK_DAYS}d): scalar 125c={report['forecaster']['scalar_125c_s']}s vs 30c={report['forecaster']['scalar_30c_s']}s "
        f"({report['forecaster']['scalar_sampling_speedup']}x); vectorized 30c={report['forecaster']['vectorized_30c_s']}s; "
        f"mean fc MAE={report['forecaster']['mean_forecast_yhat_mae']} pts, alerts match={report['forecaster']['alerts_exact_match']}"
    )
    return report


if __name__ == "__main__":
    csv_path = DATA_DIR / "historical.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run src/customer_generator.py first.")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    run_sampling_benchmark(df, repeats=3, save=True)
