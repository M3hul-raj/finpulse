# FinPulse — NatWest Segment Risk Intelligence

> **Live Demo:** [finpulse-natwest.onrender.com](https://finpulse-natwest.onrender.com)
> *Free tier — first load after inactivity may take ~50 seconds to wake up.*

## Problem & Solution

**The Problem:** Retail banking interventions typically occur *after* a default. This reactive approach leads to elevated bad-debt provisioning, permanent loss of customer trust, and costly recovery operations.

**The Solution:** FinPulse is a predictive risk intelligence system that shifts NatWest's operations from reactive recovery to proactive intervention. By continuously forecasting the **Financial Health Score (FHS)** of 1,000 customers across 8 banking segments, FinPulse predicts financial distress up to 30 days before it materializes — with validated model performance, baseline comparisons, and statistical testing to back every prediction.

---

## Architecture

```text
                    ┌─────────────────────────────────────┐
                    │         Browser (Frontend)          │
                    │  HTML5 + CSS3 + JS + Chart.js       │
                    └──────────────┬──────────────────────┘
                                   │ HTTP / JSON
                    ┌──────────────▼──────────────────────┐
                    │       Flask API Server              │
                    │       (src/api_server.py)           │
                    └──┬──────┬──────┬──────┬─────────────┘
                       │      │      │      │
          ┌────────────▼┐ ┌───▼────┐ ┌▼─────┐ ┌▼──────────────┐
          │ FHS         │ │Forecast│ │Anomaly│ │ LLM Explainer │
          │ Calculator  │ │  Engine│ │Detect.│ │(Gemini + Groq)│
          │ (stats/math)│ │ (Holt- │ │       │ │ (GPT OSS 120B)│
          │             │ │ Winters)│ │       │ │               │
          └──────┬──────┘ └───┬────┘ └┬─────┘ └┬──────────────┘
                 │            │       │        │
          ┌──────▼────────────▼───────▼────────▼──┐
          │    ML Pipeline (precomputed offline)   │
          │  Classification · Clustering · Stats   │
          └──────────────────┬────────────────────┘
                             │
          ┌──────────────────▼────────────────────┐
          │        data/historical.csv             │
          │  730,000 rows · 1,000 customers        │
          └────────────────────────────────────────┘
```

---

## ML Methodology

### FHS Formula

```
FHS = (0.4 × Balance Trend) + (0.3 × Income Regularity)
    + (0.2 × Spending Volatility) + (0.1 × Debt Ratio)

Dashboard display: < 35 = RED | 35–59 = AMBER | ≥ 60 = GREEN
```

### Feature Engineering

11 behavioral features engineered from 730 days of daily balance data per customer:

| Feature | What it measures |
|---------|-----------------|
| `bal_mean` | Average daily balance level |
| `bal_std` | Balance volatility |
| `bal_trend` | Direction of balance over time (linear slope) |
| `bal_cv` | Coefficient of variation (normalized volatility) |
| `fhs_std` | FHS volatility across rolling windows |
| `income_regularity` | CV of monthly balance deltas |
| `vel_mean` | Mean absolute daily balance change |
| `vel_max` | Maximum single-day balance swing |
| `pct_negative_days` | Fraction of days with negative balance |
| `max_drawdown` | Largest peak-to-trough decline |
| `runway_mean` | Days of expenses the balance can cover |

### Risk Classification

**Models:** Logistic Regression (baseline) vs Random Forest

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 94.5% | 0.932 |
| Random Forest | 96.5% | 0.960 |

**Top 3 features by importance:** `bal_trend` (0.24), `bal_std` (0.19), `bal_mean` (0.15), `income_regularity` (0.11)

**Data leakage prevention:** `fhs_mean` is deliberately excluded from the classifier's input features. The classification target `risk_tier` is defined as `bucket(fhs_mean, [60, 75])` — including `fhs_mean` as an input would let the model trivially re-derive the threshold rule, producing near-perfect accuracy that means nothing.

**Why high accuracy for LR and RF:** The decision boundary between risk tiers is largely monotonic and well-conditioned, which is expected — the tiers are derived from a weighted composite formulation, so a linear model and an ensemble model both achieve strong separation without data leakage.

**Classification vs Display thresholds:** The ML classifier uses `<60 / 60–75 / ≥75` thresholds (High/Medium/Low risk), while the live dashboard displays `<35 / 35–59 / ≥60` (RED/AMBER/GREEN). This is intentional: the 90-day-averaged FHS from 2 years of synthetic data never drops below 35 for any of the 1,000 customers — the data generator creates customers who occasionally face shocks but maintain positive long-run trajectories. Using `<35` as the "High risk" classification boundary would produce zero High-risk training examples, making classification meaningless. The stricter `<60` boundary gives three populated classes (High: 222, Medium: 226, Low: 552) that the classifier can learn from. In production with real banking data, the display and classification thresholds would converge.

### Forecasting Comparison

Three models compared per segment with 80/20 train/test split:

| Model | Description |
|-------|-------------|
| **Naive** | Carry forward last observed value |
| **7-day SMA** | Rolling average of last 7 days |
| **Holt-Winters** | Exponential smoothing with additive trend |

**Result:** Holt-Winters beats the naive baseline on 5/8 segments with an average 3.8% RMSE improvement. Confidence intervals scale with forecast horizon using the standard `SE × √t` approximation — day 30 uncertainty is 5.5× wider than day 1.

### Statistical Testing

| Test | Result | Interpretation |
|------|--------|----------------|
| **One-way ANOVA** | F=2736, p≈0, η²=0.951 | Segment membership explains 95.1% of FHS variance. Large effect. |
| **Spearman correlation** | ρ=0.958, p≈0 | Strong positive relationship between liquidity runway and FHS. |

**Why eta-squared alongside p-value:** With n=1,000 customers, achieving p < 0.05 is nearly guaranteed regardless of actual effect size. Eta-squared (η²) shows the magnitude — 95.1% of variance explained is a large effect, confirming the 8 segments are genuinely distinct in financial behavior.

**Why Spearman uses `runway_mean` (not `balance_trend`):** `balance_trend` is 40% of the FHS formula by weight. Correlating it with FHS would be circular — validating arithmetic, not discovering a relationship. `runway_mean` is computed independently of the FHS sub-scores, so this is a genuine validation that FHS captures liquidity risk.

### Clustering Validation

K-Means clustering (k=2 to k=10) validates whether the 8 predefined banking segments correspond to natural behavioral clusters. Best silhouette score: 0.526 at k=2. PCA explains 72.1% of variance in 2 components (62.0% + 10.1%).

---

## How to Read These Results

All data in FinPulse is synthetic — generated by `customer_generator.py` with segment-specific parameters (income levels, expense ratios, income stability). The ML pipeline results therefore validate the system's internal consistency: that the FHS formula, feature engineering, and models behave as designed. They do not represent a discovery about real customer behavior.

The remaining RF features (`bal_trend`, `income_regularity`, `vel_mean`, etc.) are decomposed components of the same signal that built the FHS composite score. The model is largely learning to reconstruct a known weighted formula from its parts, which is expected behavior for a synthetic-data proof of concept. With real banking data, the classifier would be predicting genuinely independent outcomes (e.g., actual overdraft events) from observed behavioral features.

---

## Known Limitations & Future Work

- **Single train/test split:** The current 80/20 split is a single evaluation. Walk-forward cross-validation (expanding window, retrain-and-test iteratively) would provide more robust error estimates and catch overfitting to a specific time window.
- **Feature importance vs SHAP:** The RF's `.feature_importances_` measures mean decrease in impurity, which can be biased toward high-cardinality features. SHAP values provide per-prediction explanations and handle feature correlations more rigorously. Not implemented due to scope, but would be the next step for model interpretability.
- **Score-derived target:** `risk_tier` is a bucketed version of FHS, not an independent forward-looking event. In production, the classifier should predict actual outcomes — overdraft occurrence, missed payments, or account closure — from the behavioral features, breaking the circularity between input features and target.
- **Data leakage awareness:** `fhs_mean` was excluded from classifier inputs because it directly determines the target label. The remaining features still echo the FHS formula's structure (e.g., `bal_trend` maps to 40% of the formula weight), so high accuracy is partially a reconstruction of a known formula.
- **Debt Ratio Proxy:** The dataset models a single checking account balance time series rather than multi-product liabilities. `debt_ratio` proxies credit distress via overdraft frequency (fraction of days with negative balance), which evaluates to 100 across segments in baseline conditions where overdrafts are rare.

---

## Features

- **Portfolio Dashboard:** Real-time KPI overview showing total customers monitored and critical/at-risk/healthy segment distributions.
- **Risk Heatmap:** 8 segment cards color-coded RED / AMBER / GREEN based on current Financial Health Scores.
- **30-Day FHS Forecast:** Holt-Winters forecast with confidence intervals that grow with horizon. Includes baseline comparison.
- **Anomaly Detection:** Automatic flagging of segments where forecast lower bound drops below the critical threshold.
- **Stress Testing:** Interactive expense shock slider (0–50%). Models inflation spikes or economic crises in real-time.
- **Tier-Aware Recommended Actions:** Each alert includes a specific recommended action that changes based on the segment's current risk tier — escalate for RED, monitor for AMBER, retain/upsell for GREEN.
- **GenAI Intervention Plans:** Multi-provider AI engine (Google Gemini 3.5 → Groq GPT OSS 120B → Gemini Latest → Groq LTS → Curated Fallback) generating actionable intervention recommendations.
- **Model Performance Dashboard:** Interactive section showing classification accuracy, forecast comparison tables, clustering validation, and statistical test results.

---

## Setup & Run

**Prerequisites:** Python 3.10+, Git

```bash
# 1. Clone
git clone https://github.com/M3hul-raj/FinPulse-NatWest.git
cd FinPulse-NatWest

# 2. Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. AI Keys (optional — system uses curated fallbacks if absent)
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux

# 4. Generate data
python src/customer_generator.py

# 5. Run ML pipeline (precomputes all model results)
python src/ml_pipeline.py

# 6. Start server
python src/api_server.py
```

> Dashboard loads at http://localhost:5000. Risk data populates within ~30 seconds on first load; forecasts follow in background.

**Run Tests:**
```bash
pytest tests/ -v
```

---

## Deployment

FinPulse is deployed on [Render](https://render.com) as a single Python web service.

| Setting | Value |
|---------|-------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `bash start.sh` |
| **Health Check Path** | `/api/segments` |

The startup script generates synthetic data on first deploy, then launches gunicorn. All heavy computation runs in background threads — endpoints return HTTP 202 while processing and the frontend polls automatically.

**Environment Variables:** Set `GEMINI_API_KEY` and/or `GROQ_API_KEY` in Render's Environment settings.

---

*Built for NatWest Code for Purpose Hackathon 2026*
*Open-source under Apache License 2.0*
