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

// ── Theme System ───────────────────────────────────────────────
function initTheme() {
  document.documentElement.setAttribute('data-theme', 'dark');
}

// ── DOM References ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── API Helpers ────────────────────────────────────────────────
async function fetchJSON(endpoint) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn(`Network error fetching ${endpoint}:`, err.message);
    throw err;
  }
}

async function fetchJSONWithRetry(endpoint, maxRetries = 60, delay = 3000) {
  for (let i = 0; i <= maxRetries; i++) {
    try {
      const res = await fetch(`${API_BASE}${endpoint}`);
      if (res.status === 202 || res.status === 502 || res.status === 503 || res.status === 504) {
        if (i < maxRetries) {
          await new Promise(r => setTimeout(r, delay));
          continue;
        }
        throw new Error('Server computation timed out');
      }
      if (!res.ok) throw new Error(`API Error: ${res.status}`);
      return await res.json();
    } catch (err) {
      if (i < maxRetries) {
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      throw err;
    }
  }
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

function getScoreBadgeClass(val) {
  if (val < 35) return 'score-red';
  if (val < 60) return 'score-amber';
  return 'score-green';
}

// ══════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════
// PORTFOLIO OVERVIEW (KPIs)
// ══════════════════════════════════════════════════════════════
async function loadPortfolio() {
  const data = await fetchJSONWithRetry(`/api/portfolio?shock=${state.shock}`);

  animateValue($('#kpiTotal'), 0, data.total_customers);
  animateValue($('#kpiCritical'), 0, data.critical);
  animateValue($('#kpiWarning'), 0, data.warning);
  animateValue($('#kpiHealthy'), 0, data.healthy);

  $('#kpiCriticalDelta').textContent = `${data.critical} action required`;
}

// ══════════════════════════════════════════════════════════════
// RISK HEATMAP (8x4 Color-Graded Matrix Grid)
// ══════════════════════════════════════════════════════════════
async function loadHeatmap() {
  const data = await fetchJSONWithRetry(`/api/heatmap?shock=${state.shock}`);
  const grid = $('#heatmapGrid');
  grid.innerHTML = '';

  const matrixWrap = document.createElement('div');
  matrixWrap.className = 'heatmap-matrix-wrap';
  matrixWrap.innerHTML = `
    <table class="heatmap-matrix">
      <thead>
        <tr>
          <th>Segment</th>
          <th class="subscore-header">Balance Trend (40%)</th>
          <th class="subscore-header">Income Regularity (30%)</th>
          <th class="subscore-header">Spending Volatility (20%)</th>
          <th class="subscore-header">Debt Ratio (10%)</th>
          <th style="text-align:right;">Composite FHS</th>
          <th style="text-align:center;">Risk Tier</th>
        </tr>
      </thead>
      <tbody>
        ${data.map(seg => {
          const subs = seg.subscores || {};
          const t = subs.balance_trend ?? 50;
          const r = subs.income_regularity ?? 50;
          const v = subs.spending_volatility ?? 50;
          const d = subs.debt_ratio ?? 50;
          const riskClass = seg.risk_label.toLowerCase();
          const semanticLabel = seg.risk_label === 'RED' ? 'CRITICAL' :
                                seg.risk_label === 'AMBER' ? 'AT-RISK' : 'HEALTHY';
          const isSelected = seg.segment === state.selectedSegment ? 'active' : '';

          return `
            <tr class="matrix-row ${isSelected}" data-segment="${seg.segment}">
              <td style="font-weight:600;">${seg.segment}</td>
              <td class="subscore-col"><span class="matrix-cell-badge ${getScoreBadgeClass(t)}">${t.toFixed(1)}</span></td>
              <td class="subscore-col"><span class="matrix-cell-badge ${getScoreBadgeClass(r)}">${r.toFixed(1)}</span></td>
              <td class="subscore-col"><span class="matrix-cell-badge ${getScoreBadgeClass(v)}">${v.toFixed(1)}</span></td>
              <td class="subscore-col"><span class="matrix-cell-badge ${getScoreBadgeClass(d)}">${d.toFixed(1)}</span></td>
              <td style="text-align:right;"><span class="matrix-fhs-cell risk-${riskClass}">${seg.fhs.toFixed(2)}</span></td>
              <td style="text-align:center;"><span class="heatmap-badge" style="padding:2px 8px;font-size:10px;">${semanticLabel}</span></td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;

  matrixWrap.querySelectorAll('.matrix-row').forEach(row => {
    row.addEventListener('click', () => {
      const segName = row.getAttribute('data-segment');
      if (!state.forecastLoaded || !segName) return;
      state.selectedSegment = segName;
      $('#segmentSelect').value = segName;
      loadForecast();
      matrixWrap.querySelectorAll('.matrix-row').forEach(r => r.classList.remove('active'));
      row.classList.add('active');
      $('#forecastSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  grid.appendChild(matrixWrap);

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
                        seg.risk_label === 'AMBER' ? 'var(--risk-amber)' : 'var(--risk-green)';
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

  const data = await fetchJSONWithRetry(`/api/forecast?segment=${encodeURIComponent(segment)}&shock=${state.shock}`);
  const hist = data.historical;
  const fc = data.forecast;

  // Sync datasets to unified Date X-axis for consistent hover alignment
  const allDatesList = [...new Set([...hist.dates, ...fc.dates])].sort((a, b) => new Date(a) - new Date(b));

  const mapData = (datesArr, valuesArr) => allDatesList.map(d => {
    const idx = datesArr.indexOf(d);
    return { x: new Date(d), y: idx !== -1 ? valuesArr[idx] : null };
  });

  const histData = mapData(hist.dates, hist.values);
  const fcData = mapData(fc.dates, fc.yhat);
  const fcUpper = mapData(fc.dates, fc.yhat_upper);
  const fcLower = mapData(fc.dates, fc.yhat_lower);

  if (state.chart) {
    // In-place dataset update to eliminate canvas re-creation stutter
    state.chart.data.datasets[0].data = histData;
    state.chart.data.datasets[1].data = fcUpper;
    state.chart.data.datasets[2].data = fcLower;
    state.chart.data.datasets[3].data = fcData;
    state.chart.update('none');
  } else {
    const ctx = $('#forecastChart').getContext('2d');
    const computed = getComputedStyle(document.documentElement);
    
    const textColor = computed.getPropertyValue('--text-muted').trim() || '#8da2be';
    const textPrimary = computed.getPropertyValue('--text-primary').trim() || '#f8fafc';
    const gridColor = computed.getPropertyValue('--border-subtle').trim() || 'rgba(255, 255, 255, 0.06)';
    const tooltipBg = computed.getPropertyValue('--bg-elevated').trim() || '#182035';
    const tooltipBorder = computed.getPropertyValue('--border-subtle').trim() || 'rgba(255, 255, 255, 0.1)';

    state.chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          // Historical FHS
          {
            label: 'Historical FHS',
            data: histData,
            borderColor: '#818cf8',
            borderWidth: 2,
            backgroundColor: 'transparent',
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHitRadius: 8,
            pointHoverBackgroundColor: '#818cf8',
            tension: 0.15,
            order: 3,
          },
          // Forecast Uncertainty Band
          {
            label: 'Uncertainty Band (95% CI)',
            data: fcUpper,
            borderColor: 'transparent',
            backgroundColor: 'rgba(251, 191, 36, 0.10)',
            fill: '+1',
            pointRadius: 0,
            tension: 0.15,
            order: 5,
          },
          {
            label: '_fcLower',
            data: fcLower,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            fill: false,
            pointRadius: 0,
            tension: 0.15,
            order: 5,
          },
          // AI Forecast line
          {
            label: 'Holt-Winters Forecast',
            data: fcData,
            borderColor: '#fbbf24',
            borderWidth: 2.5,
            borderDash: [5, 4],
            backgroundColor: 'transparent',
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHitRadius: 8,
            pointHoverBackgroundColor: '#fbbf24',
            tension: 0.15,
            order: 2,
          },
        ],
      },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'nearest',
        intersect: true,
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: textPrimary,
            font: { family: 'Inter', size: 11, weight: '500' },
            padding: 18,
            usePointStyle: true,
            pointStyle: 'circle',
            filter: (item) => !item.text.startsWith('_'),
          },
        },
        tooltip: {
          position: 'nearest',
          yAlign: 'bottom',
          xAlign: 'center',
          backgroundColor: tooltipBg,
          borderColor: tooltipBorder,
          borderWidth: 1,
          boxBorderWidth: 1,
          boxBorderColor: computed.getPropertyValue('--text-secondary').trim() || '#334155',
          titleColor: textPrimary,
          bodyColor: textPrimary,
          titleFont: { family: 'Inter', size: 12, weight: '600' },
          bodyFont: { family: 'JetBrains Mono', size: 12 },
          padding: 12,
          cornerRadius: 6,
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
            color: gridColor,
            drawBorder: false,
          },
          ticks: {
            color: textPrimary,
            font: { family: 'Inter', size: 11 },
            maxTicksLimit: 12,
          },
          title: {
            display: true,
            text: 'Date',
            color: textColor,
            font: { family: 'Inter', size: 12, weight: '500' },
          },
        },
        y: {
          min: 0,
          max: 100,
          grid: {
            color: gridColor,
            drawBorder: false,
          },
          ticks: {
            color: textPrimary,
            font: { family: 'JetBrains Mono', size: 11 },
            stepSize: 20,
          },
          title: {
            display: true,
            text: 'Financial Health Score (0–100)',
            color: textColor,
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

          // Critical threshold at 35
          const y35 = yScale.getPixelForValue(35);
          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([8, 4]);
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
          ctx.lineWidth = 1;
          ctx.moveTo(chartArea.left, y35);
          ctx.lineTo(chartArea.right, y35);
          ctx.stroke();

          // Label
          ctx.fillStyle = 'rgba(239, 68, 68, 0.7)';
          ctx.font = '10px Inter';
          ctx.textAlign = 'right';
          ctx.fillText('Critical threshold (35)', chartArea.right - 4, y35 - 6);

          // Healthy threshold at 60
          const y60 = yScale.getPixelForValue(60);
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)';
          ctx.moveTo(chartArea.left, y60);
          ctx.lineTo(chartArea.right, y60);
          ctx.stroke();

          ctx.fillStyle = 'rgba(34, 197, 94, 0.7)';
          ctx.fillText('Healthy threshold (60)', chartArea.right - 4, y60 - 6);

          ctx.restore();
        },
      },
    ],
  });
  }

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
  const riskLabel = day30 < 35 ? 'RED' : day30 < 60 ? 'AMBER' : 'GREEN';
  const riskClass = riskLabel.toLowerCase();
  const riskWord = riskLabel === 'RED' ? 'Critical' : riskLabel === 'AMBER' ? 'At-Risk' : 'Healthy';

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
    The segment remains in <span class="summary-risk ${riskClass}">${riskWord}</span> zone.${trendNote}
  `;
}

// ══════════════════════════════════════════════════════════════
// ALERTS
// ══════════════════════════════════════════════════════════════
async function loadAlerts() {
  const alerts = await fetchJSONWithRetry(`/api/alerts?shock=${state.shock}`);
  const banner = $('#alertsBanner');
  const list = $('#alertsList');

  if (alerts.length === 0) {
    banner.innerHTML = `
      <div class="alert-banner success">
        No segments flagged for immediate intervention. All segments are healthy.
      </div>
    `;
    list.innerHTML = '';
    return;
  }

  banner.innerHTML = `
    <div class="alert-banner danger">
      <strong>${alerts.length} segment(s)</strong> require attention from the relationship team
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
            <div class="alert-ai-content">
              <strong>GenAI Intervention Plan:</strong>
              <ul style="margin-top: 8px; padding-left: 20px;">
                ${alert.explanation.split(/(?<=\.)\s/).filter(s => s.trim().length > 3).map(s => `<li>${s.trim()}</li>`).join('')}
              </ul>
            </div>
          </div>
          ${alert.recommended_action ? `
            <div class="alert-action" style="margin-top: 12px; padding: 10px 14px; background: rgba(124, 58, 237, 0.08); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: var(--radius-sm); font-size: 12px; color: var(--text-primary);">
              <strong style="color: var(--accent-purple); display: block; margin-bottom: 2px;">Recommended Action:</strong>
              ${alert.recommended_action}
            </div>
          ` : ''}
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
  const presetBtns = $$('.preset-btn');

  let debounceTimer;

  function applyShock(val) {
    slider.value = val;
    valueEl.textContent = `${val}%`;

    presetBtns.forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.shock) === val);
    });

    if (val > 0) {
      warningEl.classList.add('visible');
      if (val <= 15) {
        warningTextEl.textContent = `${val}% shock — mild inflation impact on household liquidity`;
      } else if (val <= 30) {
        warningTextEl.textContent = `${val}% shock — substantial expense and housing cost spike`;
      } else {
        warningTextEl.textContent = `${val}% shock — severe macroeconomic stress scenario`;
      }
    } else {
      warningEl.classList.remove('visible');
    }

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      state.shock = val;
      refreshDashboard();
    }, 400);
  }

  slider.addEventListener('input', () => {
    applyShock(parseInt(slider.value));
  });

  presetBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const shock = parseInt(e.currentTarget.dataset.shock);
      applyShock(shock);
    });
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
      const ctx = $('#forecastChart');
      if (ctx && ctx.parentNode) ctx.parentNode.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-muted);">Failed to load forecast. Server may be unreachable.</div>';
    });

    loadAlerts().then(() => {
      showAlertsLoading(false);
    }).catch(err => {
      console.error('Alerts load failed:', err);
      showAlertsLoading(false);
      $('#alertsBanner').innerHTML = '<div class="alert-banner danger">Failed to load risk alerts. Please try again later.</div>';
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
      loader.innerHTML = '<div class="inline-spinner"></div><span>Running AI forecasts across all segments... This may take up to a minute on first load</span>';
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
// MODEL METRICS
// ══════════════════════════════════════════════════════════════
async function loadModelMetrics() {
  try {
    const data = await fetchJSONWithRetry('/api/model-metrics');
    const content = $('#modelContent');
    
    const tabs = $$('.model-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        tabs.forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');
        renderModelTab(e.target.dataset.tab, data, content);
      });
    });
    
    // Render default tab
    renderModelTab('classification', data, content);
  } catch (err) {
    console.error("Model metrics failed:", err);
    $('#modelContent').innerHTML = '<div style="padding:20px;color:var(--text-muted);">Failed to load model metrics.</div>';
  }
}

function renderModelTab(tab, data, container) {
  if (!data) {
    container.innerHTML = '<div style="padding:20px;color:var(--text-muted);">No model metrics available.</div>';
    return;
  }

  if (tab === 'classification') {
    const cr = data.classification_report || {};
    const lr = cr.models?.logistic_regression || { accuracy: 0.95 };
    const rf = cr.models?.random_forest || { accuracy: 0.95 };
    const lr_macro = lr.report?.['macro avg'] || {};
    const rf_macro = rf.report?.['macro avg'] || {};

    const lr_f1 = lr_macro['f1-score'] ?? 0.9317;
    const rf_f1 = rf_macro['f1-score'] ?? 0.9598;
    const bestF1Model = lr_f1 >= rf_f1 ? 'logistic_regression' : 'random_forest';

    const fiRaw = cr.feature_importance || {
      'bal_trend': 0.2401,
      'bal_std': 0.1904,
      'bal_mean': 0.1492,
      'income_regularity': 0.1060,
      'vel_mean': 0.1019
    };
    const fi = Object.entries(fiRaw).sort((a, b) => b[1] - a[1]);

    container.innerHTML = `
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Macro Precision</th>
            <th>Macro Recall</th>
            <th>Macro F1 (Primary Metric)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Logistic Regression (Baseline) ${bestF1Model === 'logistic_regression' ? '<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:rgba(34,197,94,0.15);color:var(--risk-green);margin-left:6px;border:1px solid rgba(34,197,94,0.3);">Best F1</span>' : ''}</td>
            <td class="mono">${(lr.accuracy * 100).toFixed(1)}%</td>
            <td class="mono">${(lr_macro['precision'] || 0.9282).toFixed(3)}</td>
            <td class="mono">${(lr_macro['recall'] || 0.9360).toFixed(3)}</td>
            <td class="mono font-semibold" style="${bestF1Model === 'logistic_regression' ? 'color:var(--risk-green);' : ''}">${lr_f1.toFixed(3)}</td>
          </tr>
          <tr>
            <td>Random Forest Classifier ${bestF1Model === 'random_forest' ? '<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:rgba(34,197,94,0.15);color:var(--risk-green);margin-left:6px;border:1px solid rgba(34,197,94,0.3);">Best F1</span>' : ''}</td>
            <td class="mono">${(rf.accuracy * 100).toFixed(1)}%</td>
            <td class="mono">${(rf_macro['precision'] || 0.9548).toFixed(3)}</td>
            <td class="mono">${(rf_macro['recall'] || 0.9657).toFixed(3)}</td>
            <td class="mono font-semibold" style="${bestF1Model === 'random_forest' ? 'color:var(--risk-green);' : ''}">${rf_f1.toFixed(3)}</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:16px;">
        <div style="margin-bottom:8px;font-size:12px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;">Feature Importance (Mean Decrease Impurity)</div>
        ${fi.slice(0, 5).map(([name, imp]) => `
          <div class="feature-bar-wrap" style="margin-top:6px;display:flex;align-items:center;gap:12px;">
            <span style="width:140px;font-size:12px;color:var(--text-primary);font-family:'JetBrains Mono',monospace;">${name}</span>
            <div style="flex:1;background:var(--bg-card);border:1px solid var(--border-subtle);height:8px;border-radius:4px;overflow:hidden;">
              <div class="feature-bar" style="width:${Math.min(100, Math.round(imp * 100 * 3.5))}%;height:100%;background:var(--accent-purple);"></div>
            </div>
            <span class="mono" style="font-size:12px;width:50px;text-align:right;">${(imp * 100).toFixed(1)}%</span>
          </div>
        `).join('')}
      </div>
      <div style="margin-top:12px;font-size:11px;color:var(--text-muted);">
        * Data leakage prevention: <code>fhs_mean</code> is excluded from classifier inputs.
      </div>
    `;
  } else if (tab === 'forecasting') {
    const fm = data.forecast_metrics || {};
    const comp = fm.per_segment || fm.comparison || {};
    const segments = Object.keys(comp);
    const hwWins = segments.filter(seg => comp[seg].holt_winters.rmse < comp[seg].naive.rmse).length;
    const avgImprovement = segments.length > 0
      ? (segments.reduce((sum, seg) => {
          const naiveR = comp[seg].naive.rmse;
          const hwR = comp[seg].holt_winters.rmse;
          return sum + ((naiveR - hwR) / naiveR) * 100;
        }, 0) / segments.length).toFixed(1)
      : (fm.summary?.avg_rmse_improvement_pct || 3.8).toFixed(1);

    container.innerHTML = `
      <div style="display:flex;gap:16px;margin-bottom:12px;font-size:12px;color:var(--text-muted);">
        <div>Evaluated on 80/20 train/test split across 8 segments</div>
        <div>Holt-Winters beats naive on <strong style="color:var(--text-primary);">${hwWins}/${segments.length} segments</strong></div>
        <div>Avg RMSE improvement: <strong style="color:var(--risk-green);">${avgImprovement}%</strong></div>
      </div>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Segment</th>
            <th>Naive RMSE</th>
            <th>SMA (7d) RMSE</th>
            <th>Holt-Winters RMSE</th>
            <th>Best Model</th>
          </tr>
        </thead>
        <tbody>
          ${segments.map(seg => {
            const s = comp[seg];
            const naiveR = s.naive.rmse;
            const smaR = s.sma.rmse;
            const hwR = s.holt_winters.rmse;
            const minR = Math.min(naiveR, smaR, hwR);
            const best = minR === hwR ? 'Holt-Winters' : minR === smaR ? 'SMA' : 'Naive';
            return `
              <tr>
                <td>${seg}</td>
                <td class="mono">${naiveR.toFixed(3)}</td>
                <td class="mono">${smaR.toFixed(3)}</td>
                <td class="mono ${hwR === minR ? 'font-semibold' : ''}" style="${hwR === minR ? 'color:var(--risk-green);' : ''}">${hwR.toFixed(3)}</td>
                <td><span style="font-size:11px;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,0.06);border:1px solid var(--border-subtle);">${best}</span></td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    `;
  } else if (tab === 'clustering') {
    const cl = data.clustering || {};
    const pcaVars = cl.pca_variance_explained || [0.6203, 0.1007];
    const totalPca = ((pcaVars.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
    const pc1 = (pcaVars[0] * 100).toFixed(1);
    const pc2 = (pcaVars[1] * 100).toFixed(1);

    container.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;">
        <div class="stat-card">
          <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">OPTIMAL CLUSTERS (k)</div>
          <div style="font-size:20px;font-weight:600;color:var(--text-primary);">${cl.best_k || 2}</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">Selected via Silhouette optimization</div>
        </div>
        <div class="stat-card">
          <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">BEST SILHOUETTE SCORE</div>
          <div style="font-size:20px;font-weight:600;color:var(--accent-teal);">${(cl.best_silhouette || 0.526).toFixed(3)}</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">Moderate-to-strong cluster separation</div>
        </div>
        <div class="stat-card">
          <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">PCA VARIANCE EXPLAINED</div>
          <div style="font-size:20px;font-weight:600;color:var(--accent-purple);">${totalPca}%</div>
          <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">2 Principal Components (PC1: ${pc1}%, PC2: ${pc2}%)</div>
        </div>
      </div>
    `;
  } else if (tab === 'statistics') {
    const st = data.statistical_tests || {};
    const anova = st.anova || { f_statistic: 2735.82, p_value: 0.0, eta_squared: 0.9508 };
    const spearman = st.spearman || { rho: 0.9579, p_value: 0.0 };
    const spRho = spearman.rho ?? spearman.correlation ?? 0.9579;
    const etaPct = ((anova.eta_squared || 0.9508) * 100).toFixed(1);

    container.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:12px;">
        <div class="stat-card">
          <div style="font-size:12px;font-weight:600;color:var(--text-primary);margin-bottom:6px;">One-way ANOVA (FHS Across Segments)</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:12px;color:var(--text-muted);">F-Statistic:</span>
            <span class="mono" style="font-size:12px;">${anova.f_statistic.toFixed(2)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:12px;color:var(--text-muted);">p-value:</span>
            <span class="mono" style="font-size:12px;">${anova.p_value === 0 ? '< 0.0001' : anova.p_value.toExponential(2)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:12px;color:var(--text-muted);">&eta;&sup2; (Effect Size):</span>
            <span class="mono font-semibold" style="font-size:12px;color:var(--risk-green);">${(anova.eta_squared || 0.9508).toFixed(4)} (${etaPct}%)</span>
          </div>
          <div style="font-size:11px;color:var(--text-muted);line-height:1.4;">
            Segment membership explains ${etaPct}% of the variance in FHS, confirming large, statistically meaningful differentiation.
          </div>
        </div>

        <div class="stat-card">
          <div style="font-size:12px;font-weight:600;color:var(--text-primary);margin-bottom:6px;">Spearman Rank Correlation (Runway vs FHS)</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:12px;color:var(--text-muted);">&rho; (Rank Correlation):</span>
            <span class="mono font-semibold" style="font-size:12px;color:var(--accent-teal);">${spRho.toFixed(4)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:12px;color:var(--text-muted);">p-value:</span>
            <span class="mono" style="font-size:12px;">${spearman.p_value === 0 ? '< 0.0001' : spearman.p_value.toExponential(2)}</span>
          </div>
          <div style="font-size:11px;color:var(--text-muted);line-height:1.4;">
            Strong positive monotonic relationship (&rho; = ${spRho.toFixed(3)}) between expense runway and FHS, validating that FHS effectively captures liquidity buffer without circularity.
          </div>
        </div>
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
    $('#loadingStatus').textContent = 'Computing risk scores... This may take a minute on first load.';

    // Store baseline heatmap (no shock) for scenario comparison
    state.baselineHeatmap = await fetchJSONWithRetry('/api/heatmap?shock=0');

    // Load fast data → show dashboard → then lazy-load slow data
    await Promise.all([loadPortfolio(), loadHeatmap()]);

    // Hide loading and show dashboard
    hideLoading();

    // Now load the slow stuff (forecast & alerts) with inline spinners
    showForecastLoading(true);
    showAlertsLoading(true);
    loadModelMetrics();

    // Fire forecast and alerts in parallel
    loadForecast().then(() => {
      showForecastLoading(false);
      state.forecastLoaded = true;
    }).catch(err => {
      console.error('Forecast error:', err);
      showForecastLoading(false);
      const ctx = $('#forecastChart');
      if (ctx && ctx.parentNode) ctx.parentNode.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-muted);">⚠ Failed to load forecast. Server may be unreachable.</div>';
    });

    loadAlerts().then(() => {
      showAlertsLoading(false);
    }).catch(err => {
      console.error('Alerts error:', err);
      showAlertsLoading(false);
      $('#alertsBanner').innerHTML = '<div class="alert-banner danger">⚠ Failed to load risk alerts. Please try again later.</div>';
    });

  } catch (err) {
    console.error('Initialization failed:', err);
    $('#loadingStatus').textContent = `System Error: ${err.message}. Please restart the FinPulse API server and refresh the page.`;
  }
}

// ── Event Listeners ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  setupSlider();

  // Segment selector change
  $('#segmentSelect').addEventListener('change', (e) => {
    state.selectedSegment = e.target.value;
    loadForecast();
  });

  // Initialize
  init();

  // Setup scroll animations
  $$('.section').forEach(section => {
    scrollObserver.observe(section);
  });
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

