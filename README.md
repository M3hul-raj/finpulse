# 🏦 FinPulse — NatWest Segment Risk Intelligence

## i. Problem & Solution

**The Problem:** NatWest relationship and risk teams currently react to financial defaults *after* they happen. Interventions occur too late, leading to write-offs and poor customer outcomes.

**The Solution:** FinPulse is a bank-grade AI risk intelligence system that shifts operations from reactive to predictive. It forecasts the **Financial Health Score (FHS)** of banked populations across 8 Indian banking segments. By predicting population-level financial distress up to 30 days in advance, FinPulse enables NatWest relationship managers to take proactive action—offering micro-loans, restructured EMIs, or financial wellness outreach—*before* segments default.

---

## ii. Architecture

FinPulse is built on a lightweight, high-performance Flask REST backend and a zero-dependency vanilla JS frontend, incorporating statistical forecasting and multi-provider Generative AI.

```text
                    ┌─────────────────────────────────────┐
                    │         Browser (Frontend)          │
                    │  HTML5 + CSS3 + JS + Chart.js       │
                    └──────────────┬──────────────────────┘
                                   │ HTTP / JSON
                    ┌──────────────▼──────────────────────┐
                    │       Flask API Server              │
                    │       (src/api_server.py)           │
                    └──┬──────┬──────┬──────┬──────────────┘
                       │      │      │      │
          ┌────────────▼┐ ┌───▼────┐ ┌▼─────┐ ┌▼──────────────┐
          │ FHS         │ │Forecast│ │Anomaly│ │ LLM Explainer │
          │ Calculator  │ │  Engine│ │Detect.│ │(Gemini + Groq)│
          │ (stats/math)│ │ (Holt- │ │       │ │ (Llama 3.3)   │
          │             │ │ Winters) │       │                 │
          └──────┬──────┘ └───┬────┘ └┬─────┘ └┬──────────────┘
                 │            │       │        │
          ┌──────▼────────────▼───────▼────────▼──┐
          │        data/historical.csv            │
          │  730,000 rows · 1,000 customers       │
          └───────────────────────────────────────┘
```

**FHS Formula:**
```
FHS = (0.4 × Balance Trend) + (0.3 × Income Regularity)
    + (0.2 × Spending Volatility) + (0.1 × Debt Ratio)

Score: 0–100 | < 60 = RED | 60–75 = YELLOW | > 75 = GREEN
```

---

## iii. Features

- **Portfolio Dashboard:** Real-time KPI overview showing total customers monitored and critical/at-risk/healthy segment distributions.
- **Risk Heatmap:** 8 interactive segment cards color-coded RED / YELLOW / GREEN based on current Financial Health Scores.
- **30-Day FHS Forecast:** AI-powered statistical forecast using Holt-Winters Exponential Smoothing. Displays population spread, 95% uncertainty bands, and a 30-day SMA baseline.
- **Anomaly Detection:** Automatic flagging of segments where the worst-case statistical lower bound drops below the critical 60 threshold within 14 days.
- **Scenario Testing:** Interactive slider to simulate macroeconomic expense shocks (0–50%) across all segments and instantly observe the FHS impact.
- **GenAI Intervention Plans:** Multi-provider AI engine (Google Gemini → Groq/Llama 3.3 70B) generating actionable, 1–2 sentence intervention recommendations for flagged segments. Features automatic failover clustering to ensure enterprise uptime.

---

## iv. Setup & Run Instructions

**Prerequisites:** Python 3.10+, Git

**1. Clone the repository**
```bash
git clone https://github.com/M3hul-raj/FinPulse-NatWest.git
cd FinPulse-NatWest
```

**2. Create environment & install dependencies**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**3. Configure AI Keys**
```bash
# Copy the example environment variables
copy .env.example .env   # (Windows)
cp .env.example .env     # (Mac/Linux)
```
Add your free API keys to `.env` (Options: Google Gemini or Groq console). *Note: System will use offline curated fallbacks if API keys are absent to preserve demo integrity.*

**4. Generate synthetic data & Start Server**
```bash
# Build the 730k rows of historical data
python src/customer_generator.py

# Boot the API server (loads on http://localhost:5000)
python src/api_server.py
```
> *First startup requires ~45 seconds for initial background forecast caching; subsequent API requests operate with <50ms latency.*

---

## v. Limitations

FinPulse models a production implementation but adheres to hackathon scale limitations:
- **Data Source:** FinPulse currently employs a rigorous synthetic data engine. Production deployments would connect securely to NatWest's Open Banking API via OAuth2.
- **Model Calibration:** FHS component mathematical weights (0.4, 0.3, 0.2, 0.1) are currently evaluated based on generalized domain logic. Production engines would ingest historical NatWest default data to train ML-optimized weight arrays.
- **AI Rate Limits:** The pipeline utilizes free-tier rate limitations (Gemini → Groq). Aggressive rapid-fire testing may temporarily engage the offline curated fallback nodes.

---

## vi. Impact & Future Expansion

By deploying FinPulse, NatWest achieves the ability to detect segmented portfolio distress 30 days ahead of the event horizon. Predictive intelligence allows relationship units to convert defaults into restructured financial lifelines, drastically decreasing bad-debt provisioning while strengthening the underlying customer loyalty base.

**Future roadmap:**
- Direct integration with NatWest transaction data lake.
- Per-customer (granular) FHS tracking and forecasting execution.
- Automated pipeline SMS/Email alerts routing directly to relationship managers.

---

*Built for NatWest Code for Purpose Hackathon 2026 · Team BIT Mesra*
*Open-source under Apache License 2.0*
