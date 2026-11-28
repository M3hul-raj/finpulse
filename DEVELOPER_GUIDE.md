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

### Backend: `src/` (7 Python modules)

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
- **Risk labels**: `< 35 = RED` · `35–59 = AMBER` · `>= 60 = GREEN` (classifier uses separate severity bands `< 60 / 60-74.99 / >= 75`).
- **Performance**: Vectorized `compute_fhs_matrix()` + deterministic sampling (`30/125` customers via `zlib.crc32` seeding).

#### `forecaster.py` — Time-Series Forecasting Engine
- **What it does**: Forecasts FHS 30 days into the future for each segment.
- **Algorithm**: Production fits seasonal damped Holt-Winters (`trend="add", damped_trend=True, seasonal="add", seasonal_periods=30`) on a `180d` lookback window (falling back to damped Holt -> SES -> linear for `< 60` points). Full 23-origin backtests in `ml_pipeline.py` show `8/8` wins vs Naive at 30d (`+20.3%` MAE), while Naive wins at `1–7` day short horizons on `4/8` segments.
- **Measured performance (`src/benchmark.py`)**:
  - Scalar customer sampling (`30` vs `125` customers/segment): `3.16×` heatmap speedup (`2.99s -> 0.95s`), `7.71×` `180d` forecast speedup (`676.59s -> 87.70s`).
  - Vectorized `compute_fhs_matrix()`: `17.60×` `180d` forecast speedup over scalar 30c (`87.70s -> 4.98s` for 30c; `5.75s` for full 125c) with `0.00` numerical diff.
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
- **Quality-first cross-provider waterfall** (automatic failover):
  1. **Google Gemini** (`gemini-3.5-flash-lite`) — Tier 1 flagship frontier reasoning.
  2. **Groq LPU** (`openai/gpt-oss-120b`) — Tier 2 120B parameter powerhouse on sub-second LPU.
  3. **Google Gemini** (`gemini-flash-lite-latest`) — Auto-updating Google alias for deprecation immunity.
  4. **Groq LPU** (`llama-3.1-8b-instant`) — Permanent Groq LTS high-throughput baseline.
  5. **Curated fallback** — hardcoded segment-specific responses (8 segments covered).
- **Prompt engineering**: Structured risk analyst prompt with segment data, severity, and FHS trajectory.
- **Resilience**: 4 levels of live cloud failover + offline banking matrix ensure zero single-point-of-failure risk.

#### `ml_pipeline.py` — Classification & Statistical Analysis
- **What it does**: Trains Logistic Regression and Random Forest classifiers on segment-level features, runs ANOVA/Spearman statistical tests, performs K-Means clustering with PCA, and evaluates Holt-Winters vs. naive forecasting baselines.
- **Output**: `model_results/*.json` (classification_report, clustering, forecast_metrics, statistical_tests) + `data/features.csv`.
- **Key detail**: Excludes `fhs_mean` from classifier inputs to prevent data leakage (FHS is the target, not a feature).

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

See [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/RESUME_FACTS.md](docs/RESUME_FACTS.md) for audited ML metrics, rolling-origin forecast backtests, and verified claim-to-command mappings.

#### Automated Pytest Suite (`tests/test_core.py`, `tests/test_llm_failover.py`, `tests/test_api_reliability.py`, `tests/test_evaluation.py`)
- **Self-contained**: Core tests generate deterministic synthetic data internally (`seed=42`); LLM tests mock external Gemini/Groq SDK calls.
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

## Part 3: Verified Resume/CV Reference

For strictly verified, reproducible resume statements backed by executable commands, see **[docs/RESUME_FACTS.md](docs/RESUME_FACTS.md)** and **[docs/EVALUATION.md](docs/EVALUATION.md)**.

### Skills/Technologies to List
`Python`, `Flask`, `NumPy`, `Pandas`, `scikit-learn`, `statsmodels`, `Chart.js`, `HTML5/CSS3`, `JavaScript (ES6+)`, `Google Gemini API`, `Groq LPU API`, `REST APIs`, `Gunicorn`, `Docker`, `Render`, `Time-Series Backtesting`, `Exponential Smoothing`, `Anomaly Detection`, `Thread-Safe Caching`, `pytest`, `GitHub Actions`

---

## Part 4: Interview Talking Points

1. **"Why audit the risk classifier for leakage?"** — Because `risk_tier` was derived from FHS while feeding FHS's components over the same 730-day window, a threshold rule got `96.0%` accuracy. Separating Year-1 → Year-2 on disjoint customers and removing direct formula inputs dropped RF to `76.5%` (with `78.1%` 5-fold CV), and evaluating a genuinely forward 30-day decline task (`AUC 0.846`) vs disjoint 180-day decline (`mean-reversion rule AUC 0.806 > RF 0.722`) exposed the limits of i.i.d. synthetic shocks.
2. **"How did rolling-origin forecasting change your view?"** — Replacing a cherry-picked 36d/9d split with 23 expanding origins over 551 days revealed that the legacy 45-day unseasonal Holt model lost `0/8` to Naive (`-31.3%` MAE) because day-1 salary / day-2 rent cashflows create a 30-day cycle. Switching live forecasting to seasonal damped Holt-Winters (`m=30`, `180d`) wins `8/8` vs Naive (`+20.3%` MAE) at 30d while still losing to Naive at `h=1..7` short horizons on `4/8` segments.
3. **"How did you eliminate Render cold-start pain?"** — Precomputing baseline `shock=0` heatmap/forecasts into `model_results/baseline_cache.json` (`54.6 KB`), lazy-loading the `27.6 MB` CSV only for `shock > 0`, adding `/health`, and vectorizing `compute_fhs_matrix()` (`17.60×` forecast speedup: `87.70s -> 4.98s`).
4. **"Why deterministic sampling with `zlib.crc32`?"** — Ensures identical customer subsets per segment across restarts and shock scenarios, yielding `3.16×` scalar heatmap / `7.71×` scalar `180d` forecast speedup with `0.33` / `0.65` FHS MAE vs all 125 customers.
5. **"How is LLM failover tested without secrets?"** — Mocked SDK tests verify Gemini-to-Groq failover, explicit client timeouts (`10s`), placeholder key rejection, input validation, and deterministic fallback responses.
