"""
customer_generator.py
Generates synthetic financial data for 1,000 customers across 8 segments.
Saves output to data/historical.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

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
    "IT Salaried":         (85000, 0.55, 0.95),
    "Gig/Freelance":       (45000, 0.70, 0.55),
    "Students":            (12000, 0.85, 0.80),
    "Small Business":      (60000, 0.65, 0.60),
    "Govt/PSU":            (55000, 0.50, 0.98),
    "Retirees":            (30000, 0.60, 0.92),
    "Daily Wage":          (18000, 0.90, 0.45),
    "Young Professionals": (40000, 0.75, 0.85),
}

CUSTOMERS_PER_SEGMENT = 125  # 125 * 8 = 1000
DAYS = 730  # 2 years
START_DATE = "2023-01-01"
np.random.seed(42)


def generate_customer(customer_id: int, segment: str) -> pd.DataFrame:
    """Generate 2 years of daily balance data for a single customer."""
    income_mean, expense_ratio, stability = SEGMENT_CONFIG[segment]

    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    balance = np.zeros(DAYS)
    balance[0] = income_mean * np.random.uniform(0.5, 2.0)  # random starting balance

    for i, date in enumerate(dates):
        if i == 0:
            continue

        # Monthly salary on 1st (with stability noise)
        if date.day == 1:
            income = income_mean * np.random.normal(stability, 1 - stability + 0.05)
            balance[i] = balance[i-1] + max(income, 0)
        else:
            balance[i] = balance[i-1]

        # Rent/EMI on 2nd
        if date.day == 2:
            rent = income_mean * expense_ratio * np.random.uniform(0.35, 0.45)
            balance[i] -= rent

        # Weekly groceries (Friday)
        if date.weekday() == 4:
            balance[i] -= np.random.uniform(1500, 4000)

        # Daily small expenses
        balance[i] -= np.random.uniform(200, 800)

        # Weekend extra spending
        if date.weekday() in (5, 6):
            balance[i] -= np.random.uniform(500, 2000)

        # Inject shocks (anomalies) ~4 times per year
        if np.random.random() < 0.011:
            shock = income_mean * np.random.uniform(0.3, 0.8)
            balance[i] -= shock

    df = pd.DataFrame({
        "date": dates,
        "balance": np.round(balance, 2),
        "customer_id": customer_id,
        "segment": segment,
    })
    return df


def generate_all_customers() -> pd.DataFrame:
    """Generate data for all 1,000 customers."""
    all_data = []
    cid = 1
    for segment in SEGMENTS:
        for _ in range(CUSTOMERS_PER_SEGMENT):
            all_data.append(generate_customer(cid, segment))
            cid += 1
    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    print("Generating 1,000 customers...")
    df = generate_all_customers()
    out_path = Path("data/historical.csv")
    df.to_csv(out_path, index=False)
    print(f"Done. Shape: {df.shape}")
    print(df.head())
    print(df["segment"].value_counts())