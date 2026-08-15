"""
fhs_calculator.py
Computes Financial Health Score (FHS) for each customer segment.
FHS = 0.4*balance_trend + 0.3*income_regularity + 0.2*spending_volatility + 0.1*debt_ratio
Score range: 0-100. <35=RED, 35-59=AMBER, >=60=GREEN
"""

import numpy as np
import pandas as pd
import zlib




def compute_balance_trend(balances: pd.Series) -> float:
    """Score 0-100: how consistently the balance grows over time."""
    x = np.arange(len(balances))
    slope = np.polyfit(x, balances, 1)[0]
    # Normalize: slope > 500/day = perfect, slope < -500/day = zero
    score = np.clip((slope + 500) / 10, 0, 100)
    return round(float(score), 2)


def compute_income_regularity(balances: pd.Series) -> float:
    """Score 0-100: how regular/predictable the income is using smooth exponential CV decay.
    Evaluated over the structural pre-shock historical window (prior to final 30 days) to keep
    customer income stability decoupled from macroeconomic scenario expense shocks.
    """
    if len(balances) > 60 and isinstance(balances.index, pd.DatetimeIndex):
        cutoff = balances.index.max() - pd.Timedelta(days=30)
        base_slice = balances[balances.index < cutoff]
    else:
        base_slice = balances
    monthly_deltas = base_slice.resample("ME").last().diff().dropna()
    if len(monthly_deltas) == 0:
        return 50.0
    cv = monthly_deltas.std() / (abs(monthly_deltas.mean()) + 1e-9)
    # Smooth exponential decay: preserves ordinal differentiation without hard floor clipping
    score = np.clip(100 * np.exp(-cv / 1.2), 0, 100)
    return round(float(score), 2)


def compute_spending_volatility(balances: pd.Series) -> float:
    """Score 0-100: lower relative daily volatility (normalized CV) = higher score."""
    daily_changes = balances.diff().dropna()
    if len(daily_changes) == 0:
        return 50.0
    rel_vol = daily_changes.std() / (abs(balances.mean()) + 1e-9)
    # Smooth exponential decay of relative daily volatility (std / mean_balance)
    score = np.clip(100 * np.exp(-rel_vol / 0.06), 0, 100)
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
    if fhs < 35:
        return "RED"
    elif fhs < 60:
        return "AMBER"
    return "GREEN"


def compute_segment_fhs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average FHS per segment.
    Returns DataFrame with columns: segment, fhs, risk_label
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

        # Compute FHS and sub-scores for each sampled customer, then average
        customer_fhs = []
        customer_trends = []
        customer_regs = []
        customer_vols = []
        customer_debts = []
        for cid in pivot.columns:
            b = pivot[cid]
            t = compute_balance_trend(b)
            r = compute_income_regularity(b)
            v = compute_spending_volatility(b)
            d = compute_debt_ratio(b)
            f = 0.4 * t + 0.3 * r + 0.2 * v + 0.1 * d
            customer_trends.append(t)
            customer_regs.append(r)
            customer_vols.append(v)
            customer_debts.append(d)
            customer_fhs.append(f)

        avg_fhs = float(np.mean(customer_fhs))
        results.append({
            "segment": segment,
            "fhs": round(avg_fhs, 2),
            "risk_label": get_risk_label(avg_fhs),
            "balance_trend": round(float(np.mean(customer_trends)), 1),
            "income_regularity": round(float(np.mean(customer_regs)), 1),
            "spending_volatility": round(float(np.mean(customer_vols)), 1),
            "debt_ratio": round(float(np.mean(customer_debts)), 1),
        })

    return pd.DataFrame(results).sort_values("fhs", ascending=True).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/historical.csv", parse_dates=["date"])
    print("Computing FHS per segment...")
    seg_fhs = compute_segment_fhs(df)
    print(seg_fhs)