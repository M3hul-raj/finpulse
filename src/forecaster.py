"""
forecaster.py
Forecasts FHS for each segment over the next 30 days on synthetic balance data.

Model note:
  The production path fits ExponentialSmoothing(trend="add", seasonal=None), which is
  Holt's linear trend method (unseasonal), with SES+trend and linear-trend fallbacks.
  Because LOOKBACK_DAYS=45 is shorter than 2 full monthly cycles (60 days), seasonal
  period=30 Holt-Winters cannot be fit on the 45-day production window and is evaluated
  separately over full 551-day history in ml_pipeline.py / docs/EVALUATION.md.

Performance notes (measured via python src/benchmark.py):
  - Deterministic sampling of 30/125 customers per segment via zlib.crc32 yields a
    measured 4.61x speedup on scalar heatmap (2.90s -> 0.63s, MAE 0.33 FHS pts, 8/8 tiers)
    and 3.01x speedup on scalar forecasting (69.93s -> 23.24s, mean fc MAE 0.94 pts, 1/1 alert match).
  - Vectorizing FHS across customers via compute_fhs_matrix() yields a 21.53x additional
    forecast speedup (23.24s -> 1.08s for 30c/45d; 1.20s for full 125c/45d) with 0.00 parity diff.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from fhs_calculator import compute_fhs_matrix, get_risk_label, sample_segment_customers

FORECAST_DAYS = 30
LOOKBACK_DAYS = 45       # Days of historical FHS to build production time series
SAMPLE_CUSTOMERS = 30    # Customers sampled per segment (None uses all customers)
ROLLING_WINDOW = 180     # Rolling window size for FHS computation (days of balance data)


def build_daily_fhs_series(
    df: pd.DataFrame,
    segment: str,
    sample_customers: int | None = SAMPLE_CUSTOMERS,
    lookback_days: int | None = LOOKBACK_DAYS,
    rolling_window: int = ROLLING_WINDOW,
) -> pd.Series:
    """
    Compute daily average FHS for a segment using deterministic customer sampling
    and a vectorized rolling-window calculation.

    Pass sample_customers=None to include all customers in the segment.
    Pass lookback_days=None to build the full valid historical FHS series (from day rolling_window-1 onward).
    """
    seg_df = df[df["segment"] == segment]
    pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    selected = sample_segment_customers(list(pivot.columns), segment, sample_customers)
    pivot = pivot[selected]

    if lookback_days is None:
        start_pos = min(max(0, rolling_window - 1), max(0, len(pivot) - 1))
        dates = pivot.index[start_pos:]
    else:
        dates = pivot.index[-lookback_days:]

    fhs_vals = []
    for end_date in dates:
        start_date = end_date - pd.Timedelta(days=rolling_window)
        window_df = pivot.loc[start_date:end_date]
        cust_fhs, *_ = compute_fhs_matrix(window_df)
        fhs_vals.append(float(np.mean(cust_fhs)))

    return pd.Series(fhs_vals, index=pd.DatetimeIndex(dates, freq="D"))


def forecast_segment(series: pd.Series, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Forecast using Holt's linear trend (ExponentialSmoothing with trend="add", seasonal=None).
    Fallback chain: Holt -> SES + linear trend -> linear extrapolation -> flat persistence.
    Confidence intervals scale with horizon: 1.96 * residual_std * sqrt(t).
    """
    series = series.dropna()
    if len(series) == 0:
        raise ValueError("Cannot forecast an empty FHS series.")

    forecast_values = None

    # Attempt 1: Holt (additive trend, no seasonal component)
    if len(series) >= 4:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast_values = model.forecast(forecast_days).values
        except Exception as e:
            print(f"    Holt failed for segment: {e}")

    # Attempt 2: Simple Exponential Smoothing + linear slope correction
    if forecast_values is None and len(series) >= 4:
        try:
            model = SimpleExpSmoothing(
                series, initialization_method="estimated"
            ).fit(optimized=True, remove_bias=True)
            base_forecast = model.forecast(forecast_days).values
            recent = series.iloc[-30:]
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent.values, 1)[0]
            trend_correction = np.arange(1, forecast_days + 1) * slope
            forecast_values = base_forecast + trend_correction
        except Exception as e:
            print(f"    SES failed for segment: {e}")

    # Attempt 3: Linear extrapolation or persistence fallback
    if forecast_values is None:
        if len(series) >= 2:
            recent = series.iloc[-30:]
            x = np.arange(len(recent))
            slope, intercept = np.polyfit(x, recent.values, 1)
            forecast_values = intercept + slope * np.arange(
                len(recent), len(recent) + forecast_days
            )
        else:
            forecast_values = np.full(forecast_days, float(series.iloc[-1]))

    yhat = np.clip(forecast_values, 0.0, 100.0)

    residual_std = float(np.std(series.diff().dropna())) if len(series) >= 2 else 2.0
    if np.isnan(residual_std) or residual_std < 0.5:
        residual_std = 2.0

    # Random-walk horizon uncertainty approximation: 1.96 * sigma * sqrt(t)
    horizon = np.arange(1, forecast_days + 1)
    margin = 1.96 * residual_std * np.sqrt(horizon)

    last_date = series.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )

    return pd.DataFrame({
        "ds": future_dates,
        "yhat": np.round(yhat, 2),
        "yhat_lower": np.round(np.clip(yhat - margin, 0.0, 100.0), 2),
        "yhat_upper": np.round(np.clip(yhat + margin, 0.0, 100.0), 2),
    })


def forecast_all_segments(
    df: pd.DataFrame,
    sample_customers: int | None = SAMPLE_CUSTOMERS,
    lookback_days: int | None = LOOKBACK_DAYS,
) -> dict:
    """Returns dict: segment -> {historical: Series, forecast: DataFrame}."""
    results = {}
    for seg in df["segment"].unique():
        print(f"  Forecasting: {seg}")
        series = build_daily_fhs_series(
            df, seg, sample_customers=sample_customers, lookback_days=lookback_days
        )
        fc = forecast_segment(series)
        results[seg] = {"historical": series, "forecast": fc}
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    print("Forecasting all segments (synthetic data)...")
    results = forecast_all_segments(df)
    for seg, data in results.items():
        fc = data["forecast"]
        print(f"\n{seg}:")
        print(f"  Day 1:  {fc['yhat'].iloc[0]:.1f} [{fc['yhat_lower'].iloc[0]:.1f} - {fc['yhat_upper'].iloc[0]:.1f}]")
        print(f"  Day 30: {fc['yhat'].iloc[-1]:.1f} [{fc['yhat_lower'].iloc[-1]:.1f} - {fc['yhat_upper'].iloc[-1]:.1f}]")
        print(f"  Risk:   {get_risk_label(fc['yhat'].iloc[-1])}")
