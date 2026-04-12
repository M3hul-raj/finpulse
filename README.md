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
- **GenAI Intervention Plans:** Gemini API integration generating actionable, 1–2 sentence recommendations per flagged segment
- **Synthetic Data Engine:** Realistic 2-year daily balance data for 1,000 customers with salary cycles, rent, groceries, and random shock events

> **Note on LLM integration:** The Gemini API (gemini-2.0-flash-lite) is fully integrated with live API calls attempted on each run. Due to Google's free-tier rate limits on unverified projects, curated fallback responses are used for demo stability. The API call architecture is live and will work with a billing-linked project.

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

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at: https://aistudio.google.com/app/apikey

### 5. Generate synthetic data (first time only)
```bash
python src/customer_generator.py
```
This generates `data/historical.csv` with 730,000 rows (~27 MB). Takes ~30 seconds.

### 6. Run the dashboard
```bash
python src/api_server.py
```
Dashboard opens at **http://localhost:5000**. First load takes 3–5 minutes (computing 90,000 FHS data points across 8 segments). Subsequent loads use cached results and are near-instant.

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
| AI/LLM | Google Gemini API (gemini-2.0-flash-lite) via google-genai SDK |
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
          │ Calculator  │ │  Engine│ │Detect.│ │ (Gemini API)  │
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
│   ├── llm_explainer.py       # Gemini AI intervention recommendations
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
4. Review **Early Warning Alerts** — expand flagged segments to see FHS metrics and GenAI-generated intervention recommendations

### Scenario Testing
Use the sidebar slider **"Simulate expense shock"** to model events:
- **10% shock** → mild inflation impact
- **25% shock** → housing cost spike
- **50% shock** → economic crisis scenario

The heatmap, forecast chart, KPIs, and alerts update dynamically to reflect the simulated scenario.

---

## viii. Limitations

- Synthetic data only — no real NatWest customer data used
- LLM recommendations use curated fallback responses due to Gemini free-tier rate limits; live API calls are attempted first on every run
- Forecast accuracy depends on synthetic data patterns; real-world deployment would require historical transaction data
- FHS weights (0.4, 0.3, 0.2, 0.1) are heuristic — production deployment would use ML-optimized weights trained on historical default data
- First-time dashboard load takes 3–5 minutes due to FHS computation; subsequent loads are cached

---

## ix. Future Improvements

- Real transaction data ingestion via NatWest Open Banking API
- Per-customer FHS tracking (not just segment averages)
- ML-optimized FHS weight calibration using historical default data
- Automated email/SMS alerts to relationship managers when segments enter RED zone
- Multi-bank segment comparison dashboard
- WebSocket integration for real-time FHS streaming

---

*Built for NatWest Code for Purpose Hackathon 2026 · Team BIT Mesra*

*Open-source under Apache License 2.0*
