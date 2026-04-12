/* ══════════════════════════════════════════════════════════════
   FinPulse — Application Logic
   NatWest Segment Risk Intelligence Dashboard
   ══════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── State ──────────────────────────────────────────────────────
let state = {
  shock: 0,
  selectedSegment: 'Daily Wage',
  segments: [],
  chart: null,
  forecastLoaded: false,
  baselineHeatmap: null,
};

// ── DOM References ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── API Helpers ────────────────────────────────────────────────
async function fetchJSON(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`API Error: ${res.status}`);
  return res.json();
}

// ── Loading Overlay ────────────────────────────────────────────
function showLoading(text) {
  const overlay = $('#loadingOverlay');
  overlay.classList.remove('hidden');
  if (text) $('#loadingStatus').textContent = text;
}

function hideLoading() {
  const overlay = $('#loadingOverlay');
  overlay.classList.add('hidden');
  $('#appLayout').style.opacity = '1';
}

// ── Animated Number Counter ────────────────────────────────────
function animateValue(el, start, end, duration = 1200) {
  const range = end - start;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + range * eased);

    if (Number.isInteger(end)) {
      el.textContent = current.toLocaleString();
    } else {
      el.textContent = (start + range * eased).toFixed(1);
    }

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// ══════════════════════════════════════════════════════════════
// PORTFOLIO OVERVIEW (KPIs)
// ══════════════════════════════════════════════════════════════
async function loadPortfolio() {
  const data = await fetchJSON(`/api/portfolio?shock=${state.shock}`);

  animateValue($('#kpiTotal'), 0, data.total_customers);
  animateValue($('#kpiCritical'), 0, data.critical);
  animateValue($('#kpiWarning'), 0, data.warning);
  animateValue($('#kpiHealthy'), 0, data.healthy);

  $('#kpiCriticalDelta').textContent = `⚠ ${data.critical} action required`;
}

// ══════════════════════════════════════════════════════════════
// RISK HEATMAP
// ══════════════════════════════════════════════════════════════
async function loadHeatmap() {
  const data = await fetchJSON(`/api/heatmap?shock=${state.shock}`);
  const grid = $('#heatmapGrid');
  grid.innerHTML = '';

  data.forEach((seg, i) => {
    const riskClass = `risk-${seg.risk_label.toLowerCase()}`;
    const card = document.createElement('div');
    card.className = `heatmap-card ${riskClass}`;
    card.style.animationDelay = `${i * 0.08}s`;
    card.style.animation = `fadeInUp 0.5s ease ${i * 0.08}s both`;

    const badgeIcon = seg.risk_label === 'RED' ? '🔴' :
                      seg.risk_label === 'YELLOW' ? '🟡' : '🟢';

    card.innerHTML = `
      <div class="heatmap-segment-name">${seg.segment}</div>
      <div class="heatmap-fhs">${seg.fhs.toFixed(2)}</div>
      <div class="heatmap-sub">/ 100 FHS</div>
      <div class="heatmap-badge">${badgeIcon} ${seg.risk_label}</div>
    `;

    // Click to select segment for forecast
    card.addEventListener('click', () => {
      if (!state.forecastLoaded) return; // don't try if forecasts aren't ready
      state.selectedSegment = seg.segment;
      $('#segmentSelect').value = seg.segment;
      loadForecast();
      // Smooth scroll to forecast
      $('#forecastSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    grid.appendChild(card);
  });

  // ── Scenario Comparison (Gap 2 fix) ──
  const compEl = $('#scenarioComparison');
  if (state.shock > 0 && state.baselineHeatmap) {
    compEl.style.display = 'block';
    $('#scenarioComparisonTag').textContent = `${state.shock}% shock vs baseline`;
    const body = $('#scenarioComparisonBody');
    body.innerHTML = data.map(seg => {
      const baseline = state.baselineHeatmap.find(b => b.segment === seg.segment);
      if (!baseline) return '';
      const delta = seg.fhs - baseline.fhs;
      const deltaClass = delta < -2 ? 'negative' : delta > 2 ? 'positive' : 'neutral';
      const riskColor = seg.risk_label === 'RED' ? 'var(--risk-red)' :
                        seg.risk_label === 'YELLOW' ? 'var(--risk-yellow)' : 'var(--risk-green)';
      return `
        <div class="scenario-comparison-item">
          <div class="scenario-comparison-segment">${seg.segment}</div>
          <div class="scenario-comparison-values">
            <span class="scenario-comparison-baseline">${baseline.fhs.toFixed(1)}</span>
            <span class="scenario-comparison-shocked" style="color: ${riskColor}">${seg.fhs.toFixed(1)}</span>
            <span class="scenario-comparison-delta ${deltaClass}">${delta >= 0 ? '+' : ''}${delta.toFixed(1)}</span>
          </div>
        </div>
      `;
    }).join('');
  } else {
    compEl.style.display = 'none';
  }
}

// ══════════════════════════════════════════════════════════════
// FORECAST CHART
// ══════════════════════════════════════════════════════════════
async function loadForecast() {
  const segment = state.selectedSegment;
  $('#chartTitleSegment').textContent = segment;

  const data = await fetchJSON(`/api/forecast?segment=${encodeURIComponent(segment)}&shock=${state.shock}`);
  const hist = data.historical;
  const fc = data.forecast;

  // Build Chart.js datasets
  const histData = hist.dates.map((d, i) => ({ x: new Date(d), y: hist.values[i] }));
  const fcData = fc.dates.map((d, i) => ({ x: new Date(d), y: fc.yhat[i] }));
  const fcUpper = fc.dates.map((d, i) => ({ x: new Date(d), y: fc.yhat_upper[i] }));
  const fcLower = fc.dates.map((d, i) => ({ x: new Date(d), y: fc.yhat_lower[i] }));

  // SMA baseline from historical
  const smaWindow = 30;
  const smaData = [];
  for (let i = smaWindow - 1; i < hist.values.length; i++) {
    const sum = hist.values.slice(i - smaWindow + 1, i + 1).reduce((a, b) => a + b, 0);
    smaData.push({ x: new Date(hist.dates[i]), y: +(sum / smaWindow).toFixed(2) });
  }

  // Population spread band (±8%)
  const spreadUpper = hist.dates.map((d, i) => ({ x: new Date(d), y: +(hist.values[i] * 1.08).toFixed(2) }));
  const spreadLower = hist.dates.map((d, i) => ({ x: new Date(d), y: +(hist.values[i] * 0.92).toFixed(2) }));

  if (state.chart) {
    state.chart.destroy();
  }

  const ctx = $('#forecastChart').getContext('2d');

  // Gradient for historical line
  const histGrad = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
  histGrad.addColorStop(0, '#3b82f6');
  histGrad.addColorStop(1, '#06b6d4');

  state.chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        // Population Spread (fill)
        {
          label: 'Population Spread',
          data: spreadUpper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(59, 130, 246, 0.06)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.3,
          order: 6,
        },
        {
          label: '_spreadLower',
          data: spreadLower,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.3,
          order: 7,
        },
        // Historical FHS
        {
          label: 'Avg Historical FHS',
          data: histData,
          borderColor: histGrad,
          borderWidth: 2.5,
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#3b82f6',
          tension: 0.3,
          order: 3,
        },
        // SMA Baseline
        {
          label: 'Baseline (30-day SMA)',
          data: smaData,
          borderColor: 'rgba(148, 163, 184, 0.4)',
          borderWidth: 1.5,
          borderDash: [6, 4],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.3,
          order: 4,
        },
        // Forecast Uncertainty Band
        {
          label: 'Uncertainty Band',
          data: fcUpper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.3,
          order: 5,
        },
        {
          label: '_fcLower',
          data: fcLower,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.3,
          order: 5,
        },
        // AI Forecast line
        {
          label: 'AI Forecast (Holt-Winters)',
          data: fcData,
          borderColor: '#f59e0b',
          borderWidth: 2.5,
          borderDash: [6, 3],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#f59e0b',
          tension: 0.3,
          order: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#94a3b8',
            font: { family: 'Inter', size: 11, weight: '500' },
            padding: 16,
            usePointStyle: true,
            pointStyle: 'circle',
            filter: (item) => !item.text.startsWith('_'),
          },
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(124, 58, 237, 0.3)',
          borderWidth: 1,
          titleFont: { family: 'Inter', size: 12, weight: '600' },
          bodyFont: { family: 'JetBrains Mono', size: 12 },
          padding: 14,
          cornerRadius: 10,
          displayColors: true,
          filter: (item) => !item.dataset.label.startsWith('_'),
        },
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'month',
            displayFormats: { month: 'MMM yyyy' },
          },
          grid: {
            color: 'rgba(148, 163, 184, 0.06)',
            drawBorder: false,
          },
          ticks: {
            color: '#64748b',
            font: { family: 'Inter', size: 11 },
            maxTicksLimit: 8,
          },
          title: {
            display: true,
            text: 'Date',
            color: '#64748b',
            font: { family: 'Inter', size: 12, weight: '500' },
          },
        },
        y: {
          min: 0,
          max: 100,
          grid: {
            color: 'rgba(148, 163, 184, 0.06)',
            drawBorder: false,
          },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 11 },
            stepSize: 20,
          },
          title: {
            display: true,
            text: 'Financial Health Score (0–100)',
            color: '#64748b',
            font: { family: 'Inter', size: 12, weight: '500' },
          },
        },
      },
      animation: {
        duration: 800,
        easing: 'easeOutQuart',
      },
    },
    plugins: [
      // Custom plugin: threshold lines
      {
        id: 'thresholdLines',
        afterDraw(chart) {
          const { ctx, chartArea, scales } = chart;
          const yScale = scales.y;

          // RED threshold at 60
          const y60 = yScale.getPixelForValue(60);
          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([8, 4]);
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
          ctx.lineWidth = 1;
          ctx.moveTo(chartArea.left, y60);
          ctx.lineTo(chartArea.right, y60);
          ctx.stroke();

          // Label
          ctx.fillStyle = 'rgba(239, 68, 68, 0.7)';
          ctx.font = '10px Inter';
          ctx.textAlign = 'right';
          ctx.fillText('RED threshold (60)', chartArea.right - 4, y60 - 6);

          // GREEN threshold at 75
          const y75 = yScale.getPixelForValue(75);
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)';
          ctx.moveTo(chartArea.left, y75);
          ctx.lineTo(chartArea.right, y75);
          ctx.stroke();

          ctx.fillStyle = 'rgba(34, 197, 94, 0.7)';
          ctx.fillText('GREEN threshold (75)', chartArea.right - 4, y75 - 6);

          ctx.restore();
        },
      },
    ],
  });

  // Update forecast KPIs
  const day1 = fc.yhat[0];
  const day30 = fc.yhat[fc.yhat.length - 1];
  const delta = day30 - day1;
  const minLower = Math.min(...fc.yhat_lower);
  const maxUpper = Math.max(...fc.yhat_upper);

  $('#fcDay1').textContent = day1.toFixed(1);
  $('#fcDay30').textContent = day30.toFixed(1);
  $('#fcLower').textContent = minLower.toFixed(1);
  $('#fcUpper').textContent = maxUpper.toFixed(1);

  const deltaEl = $('#fcDelta');
  deltaEl.textContent = `${delta >= 0 ? '↑' : '↓'} ${delta.toFixed(1)}`;
  deltaEl.className = `forecast-kpi-delta ${delta >= 0 ? 'up' : 'down'}`;

  // ── Forecast Text Summary (Gap 1 fix) ──
  generateForecastSummary(segment, day1, day30, delta, minLower, maxUpper, fc);
}

function generateForecastSummary(segment, day1, day30, delta, minLower, maxUpper, fc) {
  const summaryEl = $('#forecastSummary');
  const textEl = $('#forecastSummaryText');
  summaryEl.style.display = 'flex';

  const pctChange = ((delta / day1) * 100).toFixed(1);
  const direction = delta >= 0 ? 'increase' : 'decline';
  const riskLabel = day30 < 60 ? 'RED' : day30 < 75 ? 'YELLOW' : 'GREEN';
  const riskClass = riskLabel.toLowerCase();
  const riskWord = riskLabel === 'RED' ? 'Critical' : riskLabel === 'YELLOW' ? 'At-Risk' : 'Healthy';

  // Check for declining trend in forecast
  const midpoint = Math.floor(fc.yhat.length / 2);
  const firstHalf = fc.yhat.slice(0, midpoint).reduce((a, b) => a + b, 0) / midpoint;
  const secondHalf = fc.yhat.slice(midpoint).reduce((a, b) => a + b, 0) / (fc.yhat.length - midpoint);
  const trendNote = secondHalf < firstHalf - 0.5 ? ' A downward trend is observed in the second half of the forecast window.' :
                    secondHalf > firstHalf + 0.5 ? ' A recovery trend is detected in the latter forecast period.' : '';

  const shockNote = state.shock > 0 ? ` Under a <strong>${state.shock}% expense shock</strong> scenario,` : '';

  textEl.innerHTML = `
    <strong>30-Day Forecast Summary — ${segment}:</strong>${shockNote}
    FHS is projected to ${direction} from <strong>${day1.toFixed(1)}</strong> to <strong>${day30.toFixed(1)}</strong>
    (${delta >= 0 ? '+' : ''}${pctChange}%) over the next 30 days.
    Range: <strong>${minLower.toFixed(1)}</strong> (worst-case) to <strong>${maxUpper.toFixed(1)}</strong> (best-case).
    The segment remains in <span class="summary-risk ${riskClass}">${riskWord} (${riskLabel})</span> zone.${trendNote}
  `;
}

// ══════════════════════════════════════════════════════════════
// ALERTS
// ══════════════════════════════════════════════════════════════
async function loadAlerts() {
  const alerts = await fetchJSON(`/api/alerts?shock=${state.shock}`);
  const banner = $('#alertsBanner');
  const list = $('#alertsList');

  if (alerts.length === 0) {
    banner.innerHTML = `
      <div class="alert-banner success">
        ✅ No segments flagged for immediate intervention. All segments are healthy.
      </div>
    `;
    list.innerHTML = '';
    return;
  }

  banner.innerHTML = `
    <div class="alert-banner danger">
      🚨 <strong>${alerts.length} segment(s)</strong> require attention from the relationship team
    </div>
  `;

  list.innerHTML = alerts.map((alert, i) => `
    <div class="alert-card" id="alertCard${i}" style="animation-delay: ${i * 0.1}s">
      <div class="alert-card-header" onclick="toggleAlert(${i})">
        <div class="alert-card-left">
          <span class="alert-severity-badge ${alert.severity.toLowerCase()}">${alert.severity}</span>
          <span class="alert-segment-name">${alert.segment}</span>
        </div>
        <button class="alert-card-toggle" aria-label="Toggle details">▾</button>
      </div>
      <div class="alert-card-body">
        <div class="alert-card-inner">
          <div class="alert-metrics">
            <div class="alert-metric">
              <div class="alert-metric-label">FHS Day 1</div>
              <div class="alert-metric-value">${alert.fhs_day1}<span class="alert-metric-unit">/100</span></div>
            </div>
            <div class="alert-metric">
              <div class="alert-metric-label">FHS Day 30</div>
              <div class="alert-metric-value">${alert.fhs_day30}<span class="alert-metric-unit">/100</span></div>
            </div>
            <div class="alert-metric">
              <div class="alert-metric-label">Worst-Case</div>
              <div class="alert-metric-value">${alert.min_lower}<span class="alert-metric-unit">/100</span></div>
            </div>
          </div>
          <div class="alert-ai-card">
            <div class="alert-ai-icon">🤖</div>
            <div class="alert-ai-content">
              <strong>GenAI Intervention Plan:</strong> ${alert.explanation}
            </div>
          </div>
        </div>
      </div>
    </div>
  `).join('');

  // Auto-expand first alert
  if (alerts.length > 0) {
    setTimeout(() => toggleAlert(0), 300);
  }
}

function toggleAlert(index) {
  const card = $(`#alertCard${index}`);
  if (card) card.classList.toggle('expanded');
}

// ══════════════════════════════════════════════════════════════
// SEGMENT SELECTOR
// ══════════════════════════════════════════════════════════════
async function loadSegments() {
  const segments = await fetchJSON('/api/segments');
  state.segments = segments;
  const select = $('#segmentSelect');
  select.innerHTML = segments.map(s =>
    `<option value="${s}" ${s === state.selectedSegment ? 'selected' : ''}>${s}</option>`
  ).join('');
}

// ══════════════════════════════════════════════════════════════
// SCENARIO SLIDER
// ══════════════════════════════════════════════════════════════
function setupSlider() {
  const slider = $('#shockSlider');
  const valueEl = $('#shockValue');
  const warningEl = $('#shockWarning');
  const warningTextEl = $('#shockWarningText');

  let debounceTimer;

  slider.addEventListener('input', () => {
    const val = parseInt(slider.value);
    valueEl.textContent = `${val}%`;

    if (val > 0) {
      warningEl.classList.add('visible');
      if (val <= 15) {
        warningTextEl.textContent = `${val}% shock — mild inflation impact`;
      } else if (val <= 30) {
        warningTextEl.textContent = `${val}% shock — housing cost spike`;
      } else {
        warningTextEl.textContent = `${val}% shock — economic crisis scenario`;
      }
    } else {
      warningEl.classList.remove('visible');
    }

    // Debounce the API call
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      state.shock = val;
      refreshDashboard();
    }, 500);
  });
}

// ══════════════════════════════════════════════════════════════
// REFRESH ALL DATA
// ══════════════════════════════════════════════════════════════
async function refreshDashboard() {
  try {
    // Load fast data first (portfolio + heatmap), show dashboard immediately
    await Promise.all([
      loadPortfolio(),
      loadHeatmap(),
    ]);

    // Then load slow data (forecast + alerts) in background
    showForecastLoading(true);
    showAlertsLoading(true);

    loadForecast().then(() => {
      showForecastLoading(false);
      state.forecastLoaded = true;
    }).catch(err => {
      console.error('Forecast load failed:', err);
      showForecastLoading(false);
    });

    loadAlerts().then(() => {
      showAlertsLoading(false);
    }).catch(err => {
      console.error('Alerts load failed:', err);
      showAlertsLoading(false);
    });
  } catch (err) {
    console.error('Dashboard refresh failed:', err);
  }
}

function showForecastLoading(show) {
  const container = $('.chart-container');
  if (show) {
    container.classList.add('loading-state');
    if (!container.querySelector('.inline-loader')) {
      const loader = document.createElement('div');
      loader.className = 'inline-loader';
      loader.innerHTML = '<div class="inline-spinner"></div><span>Running AI forecasts across all segments... This takes 3-5 minutes on first load</span>';
      container.prepend(loader);
    }
  } else {
    container.classList.remove('loading-state');
    const loader = container.querySelector('.inline-loader');
    if (loader) loader.remove();
  }
}

function showAlertsLoading(show) {
  const banner = $('#alertsBanner');
  if (show) {
    banner.innerHTML = `
      <div class="alert-banner" style="background: var(--accent-purple-soft); border: 1px solid rgba(124,58,237,0.2); color: var(--accent-purple);">
        <div class="inline-spinner" style="width:18px;height:18px;border-width:2px;"></div>
        Analysing risk across 8 segments...
      </div>
    `;
  }
}

// ══════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════
async function init() {
  showLoading('Connecting to FinPulse API...');

  try {
    // Load segments first (fast)
    await loadSegments();
    $('#loadingStatus').textContent = 'Loading portfolio data...';

    // Store baseline heatmap (no shock) for scenario comparison
    state.baselineHeatmap = await fetchJSON('/api/heatmap?shock=0');

    // Load fast data → show dashboard → then lazy-load slow data
    await Promise.all([loadPortfolio(), loadHeatmap()]);

    // Hide loading and show dashboard
    hideLoading();

    // Now load the slow stuff (forecast & alerts) with inline spinners
    showForecastLoading(true);
    showAlertsLoading(true);

    // Fire forecast and alerts in parallel (these will take minutes)
    loadForecast().then(() => {
      showForecastLoading(false);
      state.forecastLoaded = true;
    }).catch(err => {
      console.error('Forecast error:', err);
      showForecastLoading(false);
    });

    loadAlerts().then(() => {
      showAlertsLoading(false);
    }).catch(err => {
      console.error('Alerts error:', err);
      showAlertsLoading(false);
    });

  } catch (err) {
    console.error('Initialization failed:', err);
    $('#loadingStatus').textContent = `Error: ${err.message}. Ensure the API server is running on port 5000.`;
  }
}

// ── Event Listeners ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupSlider();

  // Segment selector change
  $('#segmentSelect').addEventListener('change', (e) => {
    state.selectedSegment = e.target.value;
    loadForecast();
  });

  // Initialize
  init();
});

// ── Intersection Observer for scroll animations ────────────────
const observerOptions = {
  threshold: 0.1,
  rootMargin: '0px 0px -50px 0px',
};

const scrollObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.animationPlayState = 'running';
    }
  });
}, observerOptions);

document.addEventListener('DOMContentLoaded', () => {
  $$('.section').forEach(section => {
    scrollObserver.observe(section);
  });
});
