"""
customer_generator.py
Generates DETERMINISTIC SYNTHETIC financial data for 1,000 simulated customers
across 8 retail banking segments. Saves output to data/historical.csv.

NOTE: All balances, income events, expenses, and shocks are 100% synthetic
(seeded simulation, default seed=42) for demonstration and benchmarking only.
No real customer or NatWest production data is used anywhere in FinPulse.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEGMENTS = [
    "IT Salaried",
    "Gig/Freelance",
    "Students",
    "Small Business",
    "Govt/PSU",
    "Retirees",
    "Daily Wage",
    "Young Professionals",
]

# Segment config: (monthly_income_mean, monthly_expense_ratio, income_stability)
SEGMENT_CONFIG = {
    "IT Salaried":         (85000, 0.35, 0.99),
    "Gig/Freelance":       (45000, 0.70, 0.55),
    "Students":            (12000, 0.85, 0.80),
    "Small Business":      (60000, 0.65, 0.60),
    "Govt/PSU":            (55000, 0.30, 0.99),
    "Retirees":            (30000, 0.60, 0.92),
    "Daily Wage":          (18000, 0.90, 0.45),
    "Young Professionals": (40000, 0.75, 0.85),
}

CUSTOMERS_PER_SEGMENT = 125  # 125 * 8 = 1,000 synthetic customers
DAYS = 730                   # 2 years of daily synthetic balances
START_DATE = "2023-01-01"
DEFAULT_SEED = 42


def generate_customer(
    customer_id: int,
    segment: str,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a single synthetic customer's 730-day balance trajectory."""
    if segment not in SEGMENT_CONFIG:
        raise ValueError(f"Unknown segment '{segment}'. Expected one of {SEGMENTS}.")

    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED + int(customer_id))

    income_mean, expense_ratio, stability = SEGMENT_CONFIG[segment]
    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    balance = np.zeros(DAYS, dtype=np.float64)
    balance[0] = income_mean * stability * rng.uniform(1.0, 3.0)

    actual_income = income_mean * stability
    monthly_expense_budget = actual_income * expense_ratio

    for i, date in enumerate(dates):
        if i == 0:
            continue

        balance[i] = balance[i - 1]

        # Monthly salary on 1st
        if date.day == 1:
            income = actual_income * rng.normal(1.0, (1 - stability) * 0.3 + 0.05)
            balance[i] += max(income, actual_income * 0.1)

        # Rent/EMI on 2nd (35-45% of monthly expense budget)
        if date.day == 2:
            rent = monthly_expense_budget * rng.uniform(0.35, 0.45)
            balance[i] -= rent

        # Weekly groceries on Friday (15% of monthly budget / 4 weeks)
        if date.weekday() == 4:
            groceries = (monthly_expense_budget * 0.15 / 4) * rng.uniform(0.8, 1.2)
            balance[i] -= groceries

        # Daily small expenses (25% of monthly budget / 30 days)
        daily_expense = (monthly_expense_budget * 0.25 / 30) * rng.uniform(0.5, 1.5)
        balance[i] -= daily_expense

        # Weekend extra spending (10% of monthly budget / 8 weekend days)
        if date.weekday() in (5, 6):
            weekend_expense = (monthly_expense_budget * 0.10 / 8) * rng.uniform(0.5, 1.5)
            balance[i] -= weekend_expense

        # Independent random expense shocks (~1/year expected at p=0.0027/day)
        if rng.random() < 0.0027:
            shock = actual_income * rng.uniform(0.1, 0.4)
            balance[i] -= shock

    return pd.DataFrame({
        "date": dates,
        "balance": np.round(balance, 2),
        "customer_id": customer_id,
        "segment": segment,
    })


def generate_all_customers(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Deterministically generate synthetic daily balance data for all 1,000 customers."""
    rng = np.random.default_rng(seed)
    all_data = []
    cid = 1
    for segment in SEGMENTS:
        for _ in range(CUSTOMERS_PER_SEGMENT):
            all_data.append(generate_customer(cid, segment, rng=rng))
            cid += 1
    return pd.concat(all_data, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FinPulse synthetic customer balance dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for deterministic generation.")
    parser.add_argument("--out", type=Path, default=Path("data/historical.csv"), help="Output CSV path.")
    args = parser.parse_args()

    print(f"Generating 1,000 SYNTHETIC customers across {len(SEGMENTS)} segments (seed={args.seed})...")
    df = generate_all_customers(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Done. Shape: {df.shape} -> {args.out}")
    print(df["segment"].value_counts())


if __name__ == "__main__":
    main()
