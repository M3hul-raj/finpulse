"""
forecaster.py
Forecasts FHS for each segment over next 30 days.
Uses SimpleExpSmoothing + linear trend for robustness.

Performance optimizations:
  - Samples 30 customers per segment (CLT: statistically representative)
  - Uses 45-day lookback with 180-day rolling window (vs 90-day expanding)
  - Reduces total FHS calculations from 90,000 to ~10,800 (8× speedup)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import zlib
from fhs_calculator import compute_fhs, get_risk_label

FORECAST_DAYS = 30
LOOKBACK_DAYS = 45       # Days of historical FHS to build time series
SAMPLE_CUSTOMERS = 30    # Customers sampled per segment (CLT: 30 is sufficient)
ROLLING_WINDOW = 180     # Rolling window size for FHS computation (days of balance data)




def build_daily_fhs_series(df: pd.DataFrame, segment: str) -> pd.Series:
    """
    Compute daily average FHS for a segment using sampled customers and
    a rolling window approach for performance.
    """
    seg_df = df[df["segment"] == segment].copy()

    # Pivot: rows=date, cols=customer_id
    pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
    pivot.index = pd.to_datetime(pivot.index)

    # Sample customers deterministically (use stable seed per segment)
    all_customers = sorted(list(pivot.columns))
    seed_val = zlib.crc32(segment.encode("utf-8"))
    rng = np.random.RandomState(seed_val)
    sample_size = min(SAMPLE_CUSTOMERS, len(all_customers))
    sampled = rng.choice(all_customers, size=sample_size, replace=False)
    pivot = pivot[sampled]

    # Use last LOOKBACK_DAYS for the time series
    dates = pivot.index[-LOOKBACK_DAYS:]
    fhs_vals = []

    for end_date in dates:
        # Rolling window: use only last ROLLING_WINDOW days (not all history)
        start_date = end_date - pd.Timedelta(days=ROLLING_WINDOW)
        window_df = pivot.loc[start_date:end_date]
        cust_fhs = [compute_fhs(window_df[col]) for col in window_df.columns]
        fhs_vals.append(np.mean(cust_fhs))

    return pd.Series(fhs_vals, index=pd.DatetimeIndex(dates, freq="D"))


def forecast_segment(series: pd.Series) -> pd.DataFrame:
    """Forecast using SimpleExpSmoothing + linear extrapolation for trend."""
    series = series.dropna()

    try:
        model = SimpleExpSmoothing(series, initialization_method="estimated").fit(
            optimized=True, remove_bias=True
        )
        base_forecast = model.forecast(FORECAST_DAYS)
    except Exception:
        # Fallback: use last value repeated
        base_forecast = pd.Series(
            [series.iloc[-1]] * FORECAST_DAYS
        )

    # Add linear trend from last 30 days
    recent = series.iloc[-30:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent.values, 1)[0]
    trend_correction = np.arange(1, FORECAST_DAYS + 1) * slope

    yhat = np.clip(base_forecast.values + trend_correction, 0, 100)
    residual_std = float((series - series.mean()).std())
    if np.isnan(residual_std) or residual_std == 0:
        residual_std = 2.0

    last_date = series.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D"
    )

    return pd.DataFrame({
        "ds": future_dates,
        "yhat":       np.round(yhat, 2),
        "yhat_lower": np.round(np.clip(yhat - 1.96 * residual_std, 0, 100), 2),
        "yhat_upper": np.round(np.clip(yhat + 1.96 * residual_std, 0, 100), 2),
    })


def forecast_all_segments(df: pd.DataFrame) -> dict:
    """Returns dict: segment -> {historical: Series, forecast: DataFrame}"""
    results = {}
    for seg in df["segment"].unique():
        print(f"  Forecasting: {seg}")
        series = build_daily_fhs_series(df, seg)
        fc = forecast_segment(series)
        results[seg] = {"historical": series, "forecast": fc}
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    print("Forecasting all segments...")
    results = forecast_all_segments(df)
    for seg, data in results.items():
        fc = data["forecast"]
        print(f"\n{seg}:")
        print(f"  Day 1:  {fc['yhat'].iloc[0]:.1f} [{fc['yhat_lower'].iloc[0]:.1f} - {fc['yhat_upper'].iloc[0]:.1f}]")
        print(f"  Day 30: {fc['yhat'].iloc[-1]:.1f} [{fc['yhat_lower'].iloc[-1]:.1f} - {fc['yhat_upper'].iloc[-1]:.1f}]")
        print(f"  Risk:   {get_risk_label(fc['yhat'].iloc[-1])}")