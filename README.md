# 🏦 FinPulse — NatWest Segment Risk Intelligence

## i. Overview

FinPulse is a bank-grade AI risk intelligence system built for NatWest's retail banking operations. It forecasts the **Financial Health Score (FHS)** of 1,000 customers across 8 Indian banking segments, predicting population-level financial distress up to 30 days in advance. The system enables NatWest's relationship managers to take proactive action — offering micro-loans, restructured EMIs, or financial wellness outreach — before customers default.

**Problem solved:** NatWest relationship teams currently react to defaults after they happen. FinPulse shifts this to predictive, proactive intervention.

**Intended users:** NatWest retail risk managers and relationship teams.

---

## ii. Features

- **Portfolio Dashboard:** Real-time KPI overview — total customers monitored, critical/at-risk/healthy segment counts with animated counters
- **Risk Heatmap:** 8 interactive segment cards color-coded RED / YELLOW / GREEN based on Financial Health Score, with glow effects and click-to-inspect
- **30-Day FHS Forecast:** AI-powered forecast using Holt-Winters Exponential Smoothing with population spread, uncertainty bands, and 30-day SMA baseline comparison
- **Anomaly Detection:** Automatic flagging of segments where the worst-case lower bound drops below critical thresholds within 14 days
- **Scenario Testing:** Interactive slider to simulate expense shocks (0–50%) across all segments and instantly observe FHS impact on the entire dashboard
- **GenAI Intervention Plans:** Multi-provider AI engine (Gemini + Groq/Llama 3.3 70B) generating actionable, 1–2 sentence recommendations per flagged segment with automatic provider fallback
- **Synthetic Data Engine:** Realistic 2-year daily balance data for 1,000 customers with salary cycles, rent, groceries, and random shock events

> **Note on LLM integration:** The system uses a multi-provider AI fallback chain: Google Gemini API → Groq (Llama 3.3 70B) → curated responses. Live API calls are attempted on each run through both providers. If both free-tier quotas are exhausted, segment-specific curated responses ensure demo stability. At least one provider will be live at any given time.

---

## iii. Install and Run Instructions

**Prerequisites:** Python 3.10+, Git

### 1. Clone the repository
```bash
git clone https://github.com/M3hul-raj/FinPulse-NatWest.git
cd FinPulse-NatWest
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Edit `.env` and add your API keys (at least one AI key recommended):
```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```
- Gemini: free key at https://aistudio.google.com/app/apikey
- Groq: free key at https://console.groq.com (recommended — generous free tier)

### 5. Generate synthetic data (first time only)
```bash
python src/customer_generator.py
```
This generates `data/historical.csv` with 730,000 rows (~27 MB). Takes ~30 seconds.

### 6. Run the dashboard
```bash
python src/api_server.py
```
Dashboard opens at **http://localhost:5000**. First load takes ~45 seconds (computing optimized FHS data points across 8 segments using a rolling window). Subsequent loads use cached results and are near-instant.

### 7. Run tests
```bash
pytest tests/test_forecaster.py -v
```
Expected: 12 passed.

---

## iv. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Backend API | Flask 3.1 |
| Frontend | HTML5, CSS3, JavaScript (ES6+) |
| Charts | Chart.js 4.4 |
| Forecasting | statsmodels 0.14 (Holt-Winters Exponential Smoothing) |
| Baseline | pandas rolling SMA (30-day) |
| AI/LLM | Multi-provider: Google Gemini (gemini-2.0-flash-lite) + Groq (Llama 3.3 70B) with automatic fallback |
| Data Generation | pandas + numpy (synthetic, no real customer data used) |
| Testing | pytest 9.0 (12 unit tests) |
| Env Management | python-dotenv |

---

## v. Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Browser (Frontend)          │
                    │  HTML5 + CSS3 + JS + Chart.js       │
                    └──────────────┬──────────────────────┘
                                   │ HTTP / JSON
                    ┌──────────────▼──────────────────────┐
                    │       Flask API Server              │
                    │       (src/api_server.py)            │
                    │                                      │
                    │  GET /api/portfolio                   │
                    │  GET /api/heatmap?shock=N             │
                    │  GET /api/forecast?segment=X&shock=N  │
                    │  GET /api/alerts?shock=N              │
                    │  GET /api/segments                    │
                    └──┬──────┬──────┬──────┬──────────────┘
                       │      │      │      │
          ┌────────────▼┐ ┌───▼────┐ ┌▼─────┐ ┌▼──────────────┐
          │ FHS         │ │Forecast│ │Anomaly│ │ LLM Explainer │
          │ Calculator  │ │  Engine│ │Detect.│ │(Gemini + Groq)│
          │ (fhs_calc.) │ │(forec.)│ │(anom.)│ │ (llm_expl.)   │
          └──────┬──────┘ └───┬────┘ └┬─────┘ └┬──────────────┘
                 │            │       │        │
          ┌──────▼────────────▼───────▼────────▼──┐
          │        data/historical.csv             │
          │  730,000 rows · 1,000 customers        │
          │  Generated by customer_generator.py    │
          └────────────────────────────────────────┘
```

**FHS Formula:**
```
FHS = (0.4 × Balance Trend) + (0.3 × Income Regularity)
    + (0.2 × Spending Volatility) + (0.1 × Debt Ratio)

Score: 0–100 | < 60 = RED | 60–75 = YELLOW | > 75 = GREEN
```

---

## vi. Project Structure

```
FinPulse-NatWest/
├── frontend/                  # Custom web dashboard
│   ├── index.html             # Main dashboard page
│   ├── css/styles.css         # Design system (dark mode, glassmorphism)
│   └── js/app.js              # Client-side logic, Chart.js, API integration
├── src/                       # Python backend
│   ├── api_server.py          # Flask REST API (serves frontend + JSON APIs)
│   ├── fhs_calculator.py      # Financial Health Score computation
│   ├── forecaster.py          # Holt-Winters 30-day forecast engine
│   ├── anomaly.py             # Risk anomaly detection
│   ├── llm_explainer.py       # Multi-provider AI intervention recommendations
│   └── customer_generator.py  # Synthetic data generation (1,000 customers)
├── data/                      # Generated data (not tracked in git)
│   └── historical.csv         # 730,000 rows of synthetic balance data
├── tests/
│   └── test_forecaster.py     # 12 unit tests covering all core modules
├── docs/
│   └── architecture.png       # System architecture diagram
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
├── LICENSE                    # Apache License 2.0
└── README.md                  # This file
```

---

## vii. Usage

### Dashboard
After running `python src/api_server.py` and opening http://localhost:5000:
1. View the **Portfolio Overview** — 4 KPI cards showing customer counts and risk distribution
2. Inspect the **Risk Heatmap** — 8 segment cards with FHS scores and risk levels. Click any card to jump to its forecast
3. Explore the **30-Day Forecast Chart** — select any segment from the dropdown to see historical FHS, AI forecast, uncertainty bands, and SMA baseline
4. Read the **Forecast Summary** — auto-generated natural-language summary below the chart explaining the forecast in plain English
5. Review **Early Warning Alerts** — expand flagged segments to see FHS metrics and GenAI-generated intervention recommendations

### Scenario Testing
Use the sidebar slider **"Simulate expense shock"** to model events:
- **10% shock** → mild inflation impact
- **25% shock** → housing cost spike
- **50% shock** → economic crisis scenario

When a shock is applied, a **Scenario Impact Analysis** table appears below the heatmap showing a side-by-side comparison of baseline FHS vs shocked FHS with delta values for each segment.

### API Endpoints

All data is available via REST API. Example calls using `curl`:

**1. Get portfolio KPIs:**
```bash
curl http://localhost:5000/api/portfolio?shock=0
```
```json
{
  "total_customers": 1000,
  "critical": 1,
  "warning": 1,
  "healthy": 6
}
```

**2. Get segment risk heatmap:**
```bash
curl http://localhost:5000/api/heatmap?shock=0
```
```json
[
  {"segment": "Daily Wage", "fhs": 56.23, "risk_label": "RED"},
  {"segment": "Students", "fhs": 69.36, "risk_label": "YELLOW"},
  {"segment": "Gig/Freelance", "fhs": 77.01, "risk_label": "GREEN"},
  ...
]
```

**3. Get 30-day forecast for a segment:**
```bash
curl "http://localhost:5000/api/forecast?segment=Daily%20Wage&shock=0"
```
```json
{
  "segment": "Daily Wage",
  "historical": {
    "dates": ["2024-10-03T00:00:00", ...],
    "values": [56.45, 56.32, ...]
  },
  "forecast": {
    "dates": ["2025-01-01T00:00:00", ...],
    "yhat": [56.2, 56.1, ...],
    "yhat_lower": [55.4, 55.3, ...],
    "yhat_upper": [57.0, 56.9, ...]
  }
}
```

**4. Get anomaly alerts with GenAI explanations:**
```bash
curl http://localhost:5000/api/alerts?shock=0
```
```json
[
  {
    "segment": "Daily Wage",
    "severity": "CRITICAL",
    "fhs_day1": 56.2,
    "fhs_day30": 55.9,
    "min_lower": 55.5,
    "declining": true,
    "explanation": "Daily wage workers face the highest income volatility; the team should enroll this segment in NatWest's micro-savings auto-sweep program..."
  }
]
```

**5. Scenario comparison (apply 25% shock):**
```bash
curl http://localhost:5000/api/heatmap?shock=25
```
Compare response with `?shock=0` to see the FHS impact on each segment.

---

## viii. Forecast Validation

FinPulse uses two built-in validation mechanisms to ensure forecast reliability:

1. **Baseline comparison (30-day SMA):** Every forecast is plotted alongside a Simple Moving Average baseline. If the AI forecast significantly deviates from the SMA without good reason, it signals potential overfitting.
2. **Uncertainty bands (95% CI):** Forecasts include `yhat_lower` and `yhat_upper` bounds computed from residual standard deviation (±1.96σ), providing a statistically grounded range rather than a single-point prediction.

These ensure the system communicates uncertainty honestly rather than producing overconfident predictions.

---

## ix. Limitations

- **Data Source Verification:** The current deployment uses synthetic data to demonstrate AI capabilities; production deployment would connect to NatWest's Open Banking API via secure OAuth2.
- **Model Calibration:** FHS component weights (0.4, 0.3, 0.2, 0.1) are currently calibrated using domain expert input — production deployment would use ML-optimized weights trained on historical default data.
- **AI Rate Limits:** LLM recommendations use a multi-provider fallback chain (Gemini → Groq → curated); free-tier rate limits may occasionally require fallback to curated responses during high-volume demo usage.
- **Cold Start Latency:** Server startup requires ~45 seconds for initial background forecast computation; subsequent API requests operate with <50ms latency via thread-safe caching.

---

## x. Future Improvements

- Real transaction data ingestion via NatWest Open Banking API
- Per-customer FHS tracking (not just segment averages)
- ML-optimized FHS weight calibration using historical default data
- Automated email/SMS alerts to relationship managers when segments enter RED zone
- Multi-bank segment comparison dashboard
- WebSocket integration for real-time FHS streaming

---

*Built for NatWest Code for Purpose Hackathon 2026 · Team BIT Mesra*

*Open-source under Apache License 2.0*

