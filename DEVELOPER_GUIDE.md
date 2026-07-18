# FinPulse — Developer Guide & Knowledge Base

> **Purpose**: A comprehensive reference document for returning to the FinPulse project after time away. Contains architecture details, file-by-file breakdowns, data flows, and interview talking points.

---

## Part 1: Full File-by-File Breakdown

### Root Files

| File | Size | Purpose |
|---|---|---|
| `README.md` | 8.3 KB | Project documentation. Contains problem statement, architecture diagram, features, setup instructions, limitations, impact, and deployment config. |
| `requirements.txt` | 380 B | Python dependencies: Flask, NumPy, Pandas, statsmodels, google-genai, groq, pytest, gunicorn. Pinned (`==`). |
| `start.sh` | 464 B | Render deployment entrypoint. Creates `data/` dir, generates CSV if missing, launches gunicorn with 1 worker + 4 threads. |
| `.gitignore` | 255 B | Ignores `.env`, `venv/`, `__pycache__/`, `*.csv`, IDE folders, pytest cache, etc. |
| `.env.example` | 509 B | Template for API keys. Documents both providers (Gemini + Groq) with signup links. Notes offline fallback behavior. |
| `LICENSE` | 11.5 KB | Apache License 2.0 — full standard text. |
| `.github/workflows/tests.yml` | 545 B | GitHub Actions CI workflow. Runs `pytest tests/ -v` on push/PR to `main` using Python 3.14. |

---

### Backend: `src/` (6 Python modules)

#### `customer_generator.py` — Synthetic Data Engine
- **What it does**: Generates 730,000 rows of daily balance data for 1,000 customers across 8 banking segments.
- **How it works**:
  - 8 segments, each with unique financial profiles (income mean, expense ratio, income stability).
  - 125 customers per segment × 730 days (2 years) = 730,000 rows.
  - Simulates realistic patterns: monthly salary (1st), rent/EMI (2nd), weekly groceries (Fridays), daily expenses, weekend spending, random shocks (~2/year).
- **Output**: `data/historical.csv` with columns: `date`, `balance`, `customer_id`, `segment`.

#### `fhs_calculator.py` — Financial Health Score Engine
- **What it does**: Computes a 0–100 Financial Health Score for each customer/segment.
- **The FHS Formula**:
  ```
  FHS = 0.4 × Balance Trend + 0.3 × Income Regularity + 0.2 × Spending Volatility + 0.1 × Debt Ratio
  ```
- **4 sub-scores** (each 0–100):
  - `compute_balance_trend()` — linear regression slope of balance over time.
  - `compute_income_regularity()` — coefficient of variation of monthly balance deltas.
  - `compute_spending_volatility()` — standard deviation of daily balance changes.
  - `compute_debt_ratio()` — percentage of days in negative balance.
- **Risk labels**: `< 60 = RED` · `60–75 = YELLOW` · `> 75 = GREEN`.
- **Performance**: Samples 30 customers per segment using deterministic `zlib.crc32` seeding (not random — same result every restart).

#### `forecaster.py` — Time-Series Forecasting Engine
- **What it does**: Forecasts FHS 30 days into the future for each segment.
- **Algorithm**: Holt-Winters Simple Exponential Smoothing + linear trend correction.
- **Performance optimizations**:
  - 30 customers sampled per segment (Central Limit Theorem: statistically representative).
  - 45-day lookback window with 180-day rolling FHS computation.
  - Reduces total FHS calculations from ~90,000 to ~10,800 (8× speedup).
- **Output per segment**: `{ds, yhat, yhat_lower, yhat_upper}` — point forecast + 95% confidence interval.
- **Fallback**: If Holt-Winters fitting fails, uses last-value repeat.

#### `anomaly.py` — Alert Detection Engine
- **What it does**: Scans forecast results and flags at-risk segments.
- **Trigger conditions** (either):
  - `yhat_lower < 60` within next 14 days → `CRITICAL`.
  - FHS declining by > 1.0 point over 30 days → `WARNING`.
- **Output**: List of alert dicts sorted by severity (most critical first).

#### `llm_explainer.py` — GenAI Intervention Engine
- **What it does**: Generates actionable 1–2 sentence intervention plans for flagged segments.
- **Provider chain** (automatic fallback):
  1. **Google Gemini** (`gemini-2.0-flash-lite`) — primary.
  2. **Groq** (`llama-3.3-70b-versatile`) — secondary.
  3. **Curated fallback** — hardcoded segment-specific responses (8 segments covered).
- **Prompt engineering**: Structured risk analyst prompt with segment data, severity, and FHS trajectory.
- **Resilience**: If all API providers fail, the system always produces output via curated fallbacks.

#### `api_server.py` — Flask REST API
- **What it does**: Wraps all modules into a REST API and serves the frontend.
- **Key endpoints**:
  | Endpoint | Response | Notes |
  |---|---|---|
  | `GET /api/segments` | JSON array of 8 segment names | **Instant** — hardcoded, no CSV read (Render health check) |
  | `GET /api/portfolio?shock=N` | KPI counts (critical/warning/healthy) | Returns 202 if computing |
  | `GET /api/heatmap?shock=N` | FHS + risk label per segment | Returns 202 if computing |
  | `GET /api/forecast?segment=X&shock=N` | Historical + forecast data | Returns 202 if computing |
  | `GET /api/alerts?shock=N` | Alert list with AI explanations | Returns 202 if computing |
- **Architecture patterns**:
  - **Non-blocking 202**: Heavy endpoints return HTTP 202 immediately while background threads compute. Frontend polls.
  - **Thread-safe caching**: Per-key locks prevent duplicate computation.
  - **Deferred startup**: Background thread pre-computes baseline heatmap + forecasts with 2s delay so gunicorn binds the port first.
  - **Shock simulation**: `_apply_shock()` reduces balances in the last 30 days by the shock percentage.

---

### Frontend: `frontend/` (3 files)

#### `index.html` — Dashboard Shell
- **Structure**: Single-page app with 5 sections:
  1. Loading overlay with spinner
  2. Sidebar (scenario slider, FHS formula, risk legend, theme toggle)
  3. Portfolio Overview (4 KPI cards)
  4. Segment Risk Heatmap (8 dynamic cards + scenario comparison)
  5. 30-Day Forecast (Chart.js canvas + 4 KPI sub-cards + text summary)
  6. Early Warning Alerts (expandable cards with GenAI recommendations)
- **SEO**: `<title>`, `<meta description>`, semantic HTML5 (`<main>`, `<aside>`, `<header>`, `<footer>`, `<section>`)
- **Dependencies**: Chart.js 4.4.4 + chartjs-adapter-date-fns 3.0.0 (CDN, pinned versions)

#### `css/styles.css` — Design System
- **Theme system**: `:root` (dark mode default) + `[data-theme="light"]` override — 30+ CSS variables.
- **Design tokens**: backgrounds, accents, risk colors, typography, borders, radii, transitions, shadows.
- **Typography**: Inter (UI text) + JetBrains Mono (numbers/data).
- **Responsive**: 3 breakpoints (1100px, 1024px, 768px) — sidebar hides on mobile.
- **Visual effects**: animated background gradients, scroll animations, hover transitions, glassmorphism.

#### `js/app.js` — Application Logic
- **State management**: Single `state` object (no global sprawl).
- **Data loading**: `fetchJSONWithRetry()` handles 202 polling (60 retries × 3s = 3 min timeout).
- **Chart.js config**: 
  - 6 datasets: Historical FHS, SMA Baseline, Population Spread, Forecast line, Uncertainty Band.
  - Unified X-axis alignment with `null` padding — prevents hover sync bugs.
  - Custom plugin: threshold lines drawn at FHS 60 (Critical) and 75 (Healthy).
  - Tooltip: `mode: 'nearest'`, `intersect: true` — non-intrusive.
- **Interactions**: heatmap card clicks → scroll to forecast, segment dropdown, debounced shock slider (500ms).

---

### Tests: `tests/`

#### `test_core.py` — 12 Unit Tests

| Module | Tests | What's Verified |
|---|---|---|
| `customer_generator` | 4 | Data shape (730K×4), column names, all 8 segments present, 125 customers each |
| `fhs_calculator` | 5 | FHS range [0,100], negative balance penalized, risk label thresholds, segment output completeness |
| `anomaly` | 3 | Returns list, flags CRITICAL for low FHS, doesn't flag healthy segments |

- **Self-contained**: Session-scoped fixture generates data internally — no CSV dependency.
- **Run**: `pytest tests/ -v` from project root.

---

## Part 2: Data Flow Diagram

```text
User opens browser
  │
  ▼
index.html loads → app.js init()
  │
  ├─ GET /api/segments ──────────────► Hardcoded list (instant)
  │   └─ Populates <select> dropdown
  │
  ├─ GET /api/heatmap?shock=0 ──────► fhs_calculator.compute_segment_fhs()
  │   └─ Renders 8 heatmap cards       └─ Returns {segment, fhs, risk_label} × 8
  │
  ├─ GET /api/portfolio?shock=0 ────► Same heatmap data → counts RED/YELLOW/GREEN
  │   └─ Animates 4 KPI counters
  │
  ├─ GET /api/forecast?segment=X ───► forecaster.build_daily_fhs_series()
  │   └─ Renders Chart.js canvas        + forecaster.forecast_segment()
  │       + 4 forecast KPI cards         └─ Returns {historical, forecast}
  │       + text summary
  │
  └─ GET /api/alerts?shock=0 ───────► anomaly.detect_anomalies()
      └─ Renders alert cards              + llm_explainer.explain_all_alerts()
          with GenAI explanations          └─ Returns [{segment, severity, explanation}]
```

**Shock slider**: When the user moves the slider, all endpoints are re-called with `?shock=N`. The API applies an N% balance reduction to the last 30 days before recomputing.

---

## Part 3: Resume/CV Guide

### Project Title
**FinPulse — AI-Powered Segment Risk Intelligence Dashboard**

---

### One-liner (for project list)
> Built a predictive risk intelligence dashboard that forecasts financial distress across 8 banking segments 30 days ahead, using Holt-Winters time-series modeling and multi-provider GenAI to generate intervention plans.

---

### Detailed Resume Bullets (Pick 4-6)

#### For Software Engineering / Full-Stack roles:
- Architected and shipped a **full-stack risk intelligence dashboard** (Python/Flask backend, vanilla JS/Chart.js frontend) deployed on Render, processing **730K synthetic transaction records** across 1,000 customers in 8 segments.
- Designed a **non-blocking API architecture** using HTTP 202 polling and thread-safe caching to handle compute-heavy forecasting workloads (~3 min cold start) without blocking the frontend or failing cloud health checks.
- Implemented a **dual-theme design system** with 30+ CSS custom properties, responsive breakpoints, and glassmorphism effects — delivering a production-grade UI without any CSS framework.

#### For Data Science / ML roles:
- Developed a **composite Financial Health Score (FHS)** combining 4 statistical indicators (balance trend, income regularity, spending volatility, debt ratio) with domain-weighted aggregation to classify customer risk across 8 retail banking segments.
- Built a **30-day forecasting pipeline** using Holt-Winters Exponential Smoothing with linear trend correction, achieving 8× speedup over naive approaches through deterministic sampling (CLT-based, 30 customers/segment) and rolling-window computation.
- Engineered an **early warning anomaly detection system** that flags segments where the 95% confidence lower bound crosses the critical threshold within 14 days, enabling proactive intervention.

#### For AI/GenAI roles:
- Integrated a **multi-provider GenAI pipeline** (Google Gemini → Groq/Llama 3.3 70B → curated fallback) with automatic failover, generating actionable intervention recommendations for at-risk customer segments with zero single-point-of-failure risk.
- Designed structured **prompt engineering** for banking risk analysis, constraining LLM output to 1–2 sentence professional recommendations with temperature 0.4 for deterministic outputs.

#### For Product / Impact-focused roles:
- Built a **macroeconomic stress-testing simulator** with an interactive 0–50% expense shock slider, enabling risk teams to model inflation spikes and housing crises in real-time and observe portfolio resilience instantly.
- Delivered a dashboard that shifts operations from **reactive recovery to proactive intervention**, providing a 30-day early warning window to deploy micro-loans, EMI restructuring, or savings sweeps before customer default.

---

### Skills/Technologies to List
`Python`, `Flask`, `NumPy`, `Pandas`, `statsmodels`, `Chart.js`, `HTML5/CSS3`, `JavaScript (ES6+)`, `Google Gemini API`, `Groq API`, `Llama 3.3 70B`, `REST APIs`, `Gunicorn`, `Render`, `Time-Series Forecasting`, `Exponential Smoothing`, `Anomaly Detection`, `Prompt Engineering`, `Thread-Safe Caching`, `CI/CD`

---

## Part 4: Interview Talking Points

These are the "why" stories behind the technical choices — interviewers love these:

1. **"Why Holt-Winters instead of Prophet/LSTM?"** — Holt-Winters is interpretable (critical for regulated banking), lightweight (no heavy ML dependencies), and produces native confidence intervals. It avoids the "black box" problem of neural networks in financial compliance settings.
2. **"Why a multi-provider GenAI chain?"** — Free-tier APIs have rate limits. By chaining Gemini → Groq → curated fallback, the system never fails to produce output, even during live demos. This mimics enterprise resilience patterns.
3. **"Why HTTP 202 polling instead of WebSockets?"** — The forecasting computation takes 1–3 minutes on cold start. WebSockets would require persistent connections and complicate deployment on Render's free tier. 202 polling is stateless, cloud-native, and the frontend handles it transparently.
4. **"Why deterministic sampling with zlib.crc32?"** — To ensure the same segment always samples the same customers across API calls and server restarts. This prevents "jitter" in the heatmap when users compare scenarios — a subtle UX detail that makes stress-testing reliable.
5. **"Why vanilla JS instead of React?"** — For a single-page dashboard with no routing, React adds build complexity and bundle size without proportional benefit. Vanilla JS with a clear state object keeps the codebase auditable and the frontend zero-dependency.
