"""
forecaster.py
Forecasts FHS for each segment over the next 30 days on synthetic balance data.

Model architecture:
  - Production fits seasonal damped Holt-Winters:
      ExponentialSmoothing(series, trend="add", damped_trend=True,
                           seasonal="add", seasonal_periods=30)
    over a 180-day lookback window (6 full 30-day salary/rent cycles).
  - Fallback chain when len(series) < 60 or optimization fails:
      seasonal damped HW (m=30) -> damped Holt -> SES + linear slope -> linear / flat.
  - Note on backtest trade-offs: in 23-origin rolling-origin evaluation, seasonal
    HW (m=30) beats Naive and 30-day Seasonal-Naive at 30-day horizons, while Naive
    still achieves lower MAE at short 1-7 day horizons on noisy/stable segments.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from fhs_calculator import compute_fhs_matrix, get_risk_label, sample_segment_customers

FORECAST_DAYS = 30
SEASONAL_PERIODS = 30
LOOKBACK_DAYS = 180      # 6 monthly cycles; satisfies >= 2*seasonal_periods (60)
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


def forecast_segment_with_model(
    series: pd.Series,
    forecast_days: int = FORECAST_DAYS,
) -> tuple[pd.DataFrame, str]:
    """
    Forecast using seasonal damped Holt-Winters (m=30) when >= 60 days exist.
    Returns (forecast_df, model_used).
    """
    series = series.dropna()
    if len(series) == 0:
        raise ValueError("Cannot forecast an empty FHS series.")

    forecast_values = None
    model_used = None

    # Attempt 1: Seasonal damped Holt-Winters (requires >= 2 full seasonal cycles)
    if len(series) >= 2 * SEASONAL_PERIODS:
        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=SEASONAL_PERIODS,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast_values = model.forecast(forecast_days).values
            model_used = "holt_winters_m30"
        except Exception as e:
            print(f"    Seasonal HW (m={SEASONAL_PERIODS}) failed for segment: {e}")

    # Attempt 2: Unseasonal damped Holt (fallback for short series < 60 days)
    if forecast_values is None and len(series) >= 4:
        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast_values = model.forecast(forecast_days).values
            model_used = "holt_damped"
        except Exception as e:
            print(f"    Holt damped failed for segment: {e}")

    # Attempt 3: Simple Exponential Smoothing + linear slope correction
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
            model_used = "ses_trend"
        except Exception as e:
            print(f"    SES failed for segment: {e}")

    # Attempt 4: Linear extrapolation or persistence fallback
    if forecast_values is None:
        if len(series) >= 2:
            recent = series.iloc[-30:]
            x = np.arange(len(recent))
            slope, intercept = np.polyfit(x, recent.values, 1)
            forecast_values = intercept + slope * np.arange(
                len(recent), len(recent) + forecast_days
            )
            model_used = "linear"
        else:
            forecast_values = np.full(forecast_days, float(series.iloc[-1]))
            model_used = "persistence"

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

    fc_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.round(yhat, 2),
        "yhat_lower": np.round(np.clip(yhat - margin, 0.0, 100.0), 2),
        "yhat_upper": np.round(np.clip(yhat + margin, 0.0, 100.0), 2),
    })
    return fc_df, model_used


def forecast_segment(series: pd.Series, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Forecast FHS over forecast_days using seasonal damped Holt-Winters (m=30)."""
    fc_df, _ = forecast_segment_with_model(series, forecast_days=forecast_days)
    return fc_df


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
    print("Forecasting all segments (synthetic data, seasonal damped HW m=30, 180d window)...")
    results = forecast_all_segments(df)
    for seg, data in results.items():
        fc = data["forecast"]
        print(f"\n{seg}:")
        print(f"  Day 1:  {fc['yhat'].iloc[0]:.1f} [{fc['yhat_lower'].iloc[0]:.1f} - {fc['yhat_upper'].iloc[0]:.1f}]")
        print(f"  Day 30: {fc['yhat'].iloc[-1]:.1f} [{fc['yhat_lower'].iloc[-1]:.1f} - {fc['yhat_upper'].iloc[-1]:.1f}]")
        print(f"  Risk:   {get_risk_label(fc['yhat'].iloc[-1])}")
