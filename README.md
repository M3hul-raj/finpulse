# 🏦 FinPulse — NatWest Segment Risk Intelligence

## i. Problem & Solution

**The Problem:** Retail banking interventions typically occur *after* a default. This reactive approach leads to elevated bad-debt provisioning, permanent loss of customer trust, and costly recovery operations.

**The Solution:** FinPulse is a predictive AI risk intelligence core that shifts NatWest's operations from reactive recovery to proactive intervention. By continuously forecasting the **Financial Health Score (FHS)** of banked populations across 8 target segments, FinPulse predicts financial distress up to 30 days before it materializes. This advance warning allows relationship managers to deploy tailored interventions—such as micro-loans, EMI restructuring, or automated savings sweeps—protecting the bank's balance sheet while preserving customer financial wellbeing.

---

## ii. Architecture

FinPulse combines robust statistical modeling with Generative AI, deployed via a resilient Flask REST backend and a zero-dependency vanilla JS frontend. The forecasting core uses Holt-Winters Exponential Smoothing to capture both trend and seasonality in financial behaviors, providing reliable confidence intervals and avoiding the 'black box' phenomenon of pure neural networks in regulated banking environments.

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
- **Anomaly Detection:** Automatic flagging of segments where the worst-case statistical lower bound drops below the critical 60 threshold within 14 days, enabling preemptive customer outreach rather than retrospective collection.
- **Micro/Macro Scenario Testing:** Interactive macroeconomic stress-testing slider (0–50% expense shock). Allows risk teams to instantly model inflation spikes or housing crises and observe portfolio resilience in real-time.
- **GenAI Intervention Plans:** Multi-provider AI engine (Google Gemini → Groq/Llama 3.3 70B) generating actionable, 1–2 sentence intervention recommendations for flagged segments. Features automatic failover clustering to ensure demonstration and enterprise uptime.

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

**5. Run Tests (optional)**
```bash
pytest tests/ -v
```
> *All 12 tests are fully self-contained — no pre-generated data files required. Tests generate synthetic data internally via a shared session fixture.*

---

## v. Limitations

FinPulse models a production-grade architecture with intentional boundaries for hackathon scoping:
- **Data Source:** FinPulse currently employs a rigorous synthetic data engine. Production deployments would connect securely to NatWest's Open Banking API via OAuth2.
- **Model Calibration:** FHS component mathematical weights (0.4, 0.3, 0.2, 0.1) are currently evaluated based on generalized domain logic. Production engines would ingest historical NatWest default data to train ML-optimized weight arrays.
- **AI Rate Limits:** The pipeline utilizes free-tier rate limitations (Gemini → Groq). Aggressive rapid-fire testing may temporarily engage the offline curated fallback nodes.

---

## vi. Impact & Future Expansion

By deploying FinPulse, NatWest gains the capability to detect segmented portfolio distress up to 30 days ahead of the event horizon. This predictive window allows relationship units to convert impending defaults into restructured financial lifelines. The business impact is twofold: drastically reduced bad-debt provisioning costs, and significantly strengthened customer loyalty.

**Future roadmap:**
- Direct integration with NatWest transaction data lake.
- Per-customer (granular) FHS tracking and forecasting execution.
- Automated pipeline SMS/Email alerts routing directly to relationship managers.

---

*Built for NatWest Code for Purpose Hackathon 2026 · Team BIT Mesra*
*Open-source under Apache License 2.0*
