"""
forecaster.py
Forecasts FHS for each segment over next 30 days.
Uses SimpleExpSmoothing + linear trend for robustness.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from fhs_calculator import compute_fhs, get_risk_label

FORECAST_DAYS = 30


def build_daily_fhs_series(df: pd.DataFrame, segment: str) -> pd.Series:
    """Compute daily average FHS per segment to match Heatmap logic exactly."""
    seg_df = df[df["segment"] == segment].copy()
    
    # Pivot: rows=date, cols=customer_id
    pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
    pivot.index = pd.to_datetime(pivot.index)

    # Use expanding window for the last 90 days to match Heatmap logic
    dates = pivot.index[-90:]
    fhs_vals = []
    
    for end_date in dates:
        window_df = pivot.loc[:end_date]
        cust_fhs = [compute_fhs(window_df[col]) for col in window_df.columns]
        fhs_vals.append(np.mean(cust_fhs))

    # freq="D" added to prevent statsmodels Holt-Winters warnings
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