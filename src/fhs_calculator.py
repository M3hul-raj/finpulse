"""
fhs_calculator.py
Computes Financial Health Score (FHS) for individual customers and segments
on FinPulse's synthetic balance data.

FHS = 0.4*balance_trend + 0.3*income_regularity + 0.2*spending_volatility + 0.1*debt_ratio
Score range: 0-100. Display tiers: <35=RED, 35-59=AMBER, >=60=GREEN.
"""

import zlib
import numpy as np
import pandas as pd


def compute_balance_trend(balances: pd.Series) -> float:
    """Score 0-100: linear slope of daily balances over time."""
    n = len(balances)
    if n < 2:
        return 50.0
    x = np.arange(n, dtype=np.float64)
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    if denom == 0.0:
        return 50.0
    slope = float(np.dot(xm, balances.to_numpy(dtype=np.float64)) / denom)
    # Normalize: slope > 500/day = perfect, slope < -500/day = zero
    score = np.clip((slope + 500.0) / 10.0, 0.0, 100.0)
    return round(float(score), 2)


def compute_income_regularity(balances: pd.Series) -> float:
    """Score 0-100: predictability of monthly net balance deltas via exponential CV decay.

    Evaluated prior to the final 30 days when len > 60 so scenario expense shocks
    applied to the last 30 days remain decoupled from long-run income regularity.
    """
    if len(balances) > 60 and isinstance(balances.index, pd.DatetimeIndex):
        cutoff = balances.index.max() - pd.Timedelta(days=30)
        base_slice = balances[balances.index < cutoff]
    else:
        base_slice = balances
    monthly_deltas = base_slice.resample("ME").last().diff().dropna()
    if len(monthly_deltas) < 2:
        return 50.0
    cv = monthly_deltas.std() / (abs(monthly_deltas.mean()) + 1e-9)
    score = np.clip(100.0 * np.exp(-cv / 1.2), 0.0, 100.0)
    return round(float(score), 2)


def compute_spending_volatility(balances: pd.Series) -> float:
    """Score 0-100: lower relative daily volatility (std of daily diffs / |mean balance|) = higher score."""
    daily_changes = balances.diff().dropna()
    if len(daily_changes) < 2:
        return 50.0
    rel_vol = daily_changes.std() / (abs(balances.mean()) + 1e-9)
    score = np.clip(100.0 * np.exp(-rel_vol / 0.06), 0.0, 100.0)
    return round(float(score), 2)


def compute_debt_ratio(balances: pd.Series) -> float:
    """Score 0-100: penalizes fraction of days spent in negative checking balance."""
    if len(balances) == 0:
        return 50.0
    pct_negative = (balances < 0).mean()
    score = np.clip(100.0 - pct_negative * 200.0, 0.0, 100.0)
    return round(float(score), 2)


def compute_fhs(balances: pd.Series) -> float:
    """Compute composite FHS score for a single customer balance time series."""
    balances = balances.copy()
    balances.index = pd.to_datetime(balances.index)

    trend = compute_balance_trend(balances)
    reg = compute_income_regularity(balances)
    vol = compute_spending_volatility(balances)
    debt = compute_debt_ratio(balances)

    fhs = 0.4 * trend + 0.3 * reg + 0.2 * vol + 0.1 * debt
    return round(float(np.clip(fhs, 0.0, 100.0)), 2)


def compute_fhs_matrix(bal_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized FHS + subscore calculation across all customer columns of a wide DataFrame.

    Returns (fhs, trend, reg, vol, debt) arrays matching compute_fhs() to exact 2-decimal parity.
    """
    arr = bal_df.to_numpy(dtype=np.float64)
    n_rows, n_cols = arr.shape
    if n_cols == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    if n_rows < 2:
        mid = np.full(n_cols, 50.0, dtype=np.float64)
        return mid, mid, mid, mid, mid

    # 1. Balance trend
    x = np.arange(n_rows, dtype=np.float64)
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    slopes = (xm @ arr) / denom if denom != 0.0 else np.zeros(n_cols)
    t_raw = np.clip((slopes + 500.0) / 10.0, 0.0, 100.0)
    trend = np.array([round(float(v), 2) for v in t_raw], dtype=np.float64)

    # 2. Income regularity
    if n_rows > 60 and isinstance(bal_df.index, pd.DatetimeIndex):
        cutoff = bal_df.index.max() - pd.Timedelta(days=30)
        base = bal_df[bal_df.index < cutoff]
    else:
        base = bal_df
    monthly_deltas = base.resample("ME").last().diff().iloc[1:]
    if len(monthly_deltas) < 2:
        reg = np.full(n_cols, 50.0, dtype=np.float64)
    else:
        cv = monthly_deltas.std() / (monthly_deltas.mean().abs() + 1e-9)
        r_raw = np.clip(100.0 * np.exp(-cv.to_numpy(dtype=np.float64) / 1.2), 0.0, 100.0)
        reg = np.array([round(float(v), 2) for v in r_raw], dtype=np.float64)

    # 3. Spending volatility
    daily_changes = bal_df.diff().iloc[1:]
    if len(daily_changes) < 2:
        vol = np.full(n_cols, 50.0, dtype=np.float64)
    else:
        rel_vol = daily_changes.std() / (bal_df.mean().abs() + 1e-9)
        v_raw = np.clip(100.0 * np.exp(-rel_vol.to_numpy(dtype=np.float64) / 0.06), 0.0, 100.0)
        vol = np.array([round(float(v), 2) for v in v_raw], dtype=np.float64)

    # 4. Debt ratio (negative checking balance frequency proxy)
    pct_neg = np.mean(arr < 0, axis=0)
    d_raw = np.clip(100.0 - pct_neg * 200.0, 0.0, 100.0)
    debt = np.array([round(float(v), 2) for v in d_raw], dtype=np.float64)

    f_raw = np.clip(0.4 * trend + 0.3 * reg + 0.2 * vol + 0.1 * debt, 0.0, 100.0)
    fhs = np.array([round(float(v), 2) for v in f_raw], dtype=np.float64)
    return fhs, trend, reg, vol, debt


def get_risk_label(fhs: float) -> str:
    """Map composite FHS [0, 100] to dashboard tier (<35=RED, 35-59=AMBER, >=60=GREEN)."""
    if fhs < 35:
        return "RED"
    elif fhs < 60:
        return "AMBER"
    return "GREEN"


def sample_segment_customers(all_customers: list, segment: str, sample_size: int | None) -> list:
    """Deterministically sample customers for a segment using crc32 seeding."""
    ordered = sorted(all_customers)
    if sample_size is None or sample_size >= len(ordered):
        return ordered
    seed_val = zlib.crc32(segment.encode("utf-8"))
    rng = np.random.RandomState(seed_val)
    return list(rng.choice(ordered, size=sample_size, replace=False))


def compute_segment_fhs(df: pd.DataFrame, sample_size: int | None = 30) -> pd.DataFrame:
    """
    Compute average FHS and sub-scores per segment.
    Pass sample_size=None to evaluate all customers in each segment.
    Returns DataFrame with columns: segment, fhs, risk_label, and 4 sub-scores.
    """
    results = []
    for segment, seg_df in df.groupby("segment"):
        pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
        pivot.index = pd.to_datetime(pivot.index)

        selected = sample_segment_customers(list(pivot.columns), segment, sample_size)
        pivot = pivot[selected]

        fhs_vals, trends, regs, vols, debts = compute_fhs_matrix(pivot)
        avg_fhs = float(np.mean(fhs_vals))
        results.append({
            "segment": segment,
            "fhs": round(avg_fhs, 2),
            "risk_label": get_risk_label(avg_fhs),
            "balance_trend": round(float(np.mean(trends)), 1),
            "income_regularity": round(float(np.mean(regs)), 1),
            "spending_volatility": round(float(np.mean(vols)), 1),
            "debt_ratio": round(float(np.mean(debts)), 1),
        })

    return pd.DataFrame(results).sort_values("fhs", ascending=True).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    print("Computing FHS per segment (synthetic data)...")
    seg_fhs = compute_segment_fhs(df)
    print(seg_fhs)
