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
    "IT Salaried":         (85000, 0.35, 0.99), # Lowered expense to 35%, maxed stability
    "Gig/Freelance":       (45000, 0.70, 0.55),
    "Students":            (12000, 0.85, 0.80),
    "Small Business":      (60000, 0.65, 0.60),
    "Govt/PSU":            (55000, 0.30, 0.99), # Lowered expense to 30%, maxed stability
    "Retirees":            (30000, 0.60, 0.92),
    "Daily Wage":          (18000, 0.90, 0.45),
    "Young Professionals": (40000, 0.75, 0.85),
}

CUSTOMERS_PER_SEGMENT = 125  # 125 * 8 = 1000
DAYS = 730  # 2 years
START_DATE = "2023-01-01"
np.random.seed(42)


def generate_customer(customer_id: int, segment: str) -> pd.DataFrame:
    income_mean, expense_ratio, stability = SEGMENT_CONFIG[segment]
    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    balance = np.zeros(DAYS)
    balance[0] = income_mean * stability * np.random.uniform(1.0, 3.0)

    # Base actual income and expenses scaled by stability
    actual_income = income_mean * stability
    monthly_expense_budget = actual_income * expense_ratio

    for i, date in enumerate(dates):
        if i == 0:
            continue

        balance[i] = balance[i - 1]

        # Monthly salary on 1st
        if date.day == 1:
            income = actual_income * np.random.normal(1.0, (1 - stability) * 0.3 + 0.05)
            balance[i] += max(income, actual_income * 0.1)

        # Rent/EMI on 2nd (35-45% of monthly expense budget)
        if date.day == 2:
            rent = monthly_expense_budget * np.random.uniform(0.35, 0.45)
            balance[i] -= rent

        # Weekly groceries on Friday (15% of monthly budget / 4 weeks)
        if date.weekday() == 4:
            groceries = (monthly_expense_budget * 0.15 / 4) * np.random.uniform(0.8, 1.2)
            balance[i] -= groceries

        # Daily small expenses (25% of monthly budget / 30 days)
        daily_expense = (monthly_expense_budget * 0.25 / 30) * np.random.uniform(0.5, 1.5)
        balance[i] -= daily_expense

        # Weekend extra spending (10% of monthly budget / 8 weekend days)
        if date.weekday() in (5, 6):
            weekend_expense = (monthly_expense_budget * 0.10 / 8) * np.random.uniform(0.5, 1.5)
            balance[i] -= weekend_expense

        # Shocks: ~2 per year (was 8), smaller magnitude
        if np.random.random() < 0.0027:
            shock = actual_income * np.random.uniform(0.1, 0.4)
            balance[i] -= shock

    return pd.DataFrame({
        "date": dates,
        "balance": np.round(balance, 2),
        "customer_id": customer_id,
        "segment": segment,
    })

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