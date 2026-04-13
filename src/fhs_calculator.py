"""
fhs_calculator.py
Computes Financial Health Score (FHS) for each customer segment.
FHS = 0.4*balance_trend + 0.3*income_regularity + 0.2*spending_volatility + 0.1*debt_ratio
Score range: 0-100. <60=RED, 60-75=YELLOW, >75=GREEN
"""

import numpy as np
import pandas as pd
import zlib

np.random.seed(42)  # Reproducible sampling


def compute_balance_trend(balances: pd.Series) -> float:
    """Score 0-100: how consistently the balance grows over time."""
    x = np.arange(len(balances))
    slope = np.polyfit(x, balances, 1)[0]
    # Normalize: slope > 500/day = perfect, slope < -500/day = zero
    score = np.clip((slope + 500) / 10, 0, 100)
    return round(float(score), 2)


def compute_income_regularity(balances: pd.Series) -> float:
    """Score 0-100: how regular/predictable the income is."""
    monthly_deltas = balances.resample("ME").last().diff().dropna()
    if len(monthly_deltas) == 0:
        return 50.0
    cv = monthly_deltas.std() / (abs(monthly_deltas.mean()) + 1e-9)
    score = np.clip(100 - cv * 50, 0, 100)
    return round(float(score), 2)


def compute_spending_volatility(balances: pd.Series) -> float:
    """Score 0-100: lower daily volatility = higher score."""
    daily_changes = balances.diff().dropna()
    vol = daily_changes.std()
    score = np.clip(100 - vol / 200, 0, 100)
    return round(float(score), 2)


def compute_debt_ratio(balances: pd.Series) -> float:
    """Score 0-100: penalizes time spent in negative balance."""
    pct_negative = (balances < 0).mean()
    score = np.clip(100 - pct_negative * 200, 0, 100)
    return round(float(score), 2)


def compute_fhs(balances: pd.Series) -> float:
    """Compute composite FHS score for a balance time series."""
    balances = balances.copy()
    balances.index = pd.to_datetime(balances.index)

    trend   = compute_balance_trend(balances)
    reg     = compute_income_regularity(balances)
    vol     = compute_spending_volatility(balances)
    debt    = compute_debt_ratio(balances)

    fhs = 0.4 * trend + 0.3 * reg + 0.2 * vol + 0.1 * debt
    return round(float(np.clip(fhs, 0, 100)), 2)


def get_risk_label(fhs: float) -> str:
    if fhs < 60:
        return "RED"
    elif fhs < 75:
        return "YELLOW"
    return "GREEN"


def compute_segment_fhs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily average FHS per segment.
    Returns DataFrame with columns: date, segment, fhs, risk_label
    """
    results = []
    for segment, seg_df in df.groupby("segment"):
        # Pivot: rows=date, cols=customer_id
        pivot = seg_df.pivot(index="date", columns="customer_id", values="balance")
        pivot.index = pd.to_datetime(pivot.index)

        # Sample customers deterministically for stable scenario deltas
        all_customers = sorted(list(pivot.columns))
        seed_val = zlib.crc32(segment.encode("utf-8"))
        rng = np.random.RandomState(seed_val) # Stable seed per segment
        sample_size = min(30, len(all_customers))
        sampled = rng.choice(all_customers, size=sample_size, replace=False)
        pivot = pivot[sampled]

        # Compute FHS for each sampled customer, then average
        customer_fhs = {}
        for cid in pivot.columns:
            customer_fhs[cid] = compute_fhs(pivot[cid])

        avg_fhs = np.mean(list(customer_fhs.values()))
        results.append({
            "segment": segment,
            "fhs": round(avg_fhs, 2),
            "risk_label": get_risk_label(avg_fhs),
        })

    return pd.DataFrame(results).sort_values("fhs", ascending=True).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    print("Computing FHS per segment...")
    seg_fhs = compute_segment_fhs(df)
    print(seg_fhs)