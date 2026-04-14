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
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeUI(savedTheme);

  const toggleBtn = document.getElementById('themeToggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateThemeUI(next);
      
      if (state.chart) {
        loadForecast(); // Re-render to pick up CSS variable changes
      }
    });
  }
}

function updateThemeUI(theme) {
  const icon = document.getElementById('themeToggleIcon');
  const text = document.getElementById('themeToggleText');
  if (icon && text) {
    icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    text.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
  }
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
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (res.status === 202) {
      if (i < maxRetries) {
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      throw new Error('Server computation timed out');
    }
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return await res.json();
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

// ══════════════════════════════════════════════════════════════
// PORTFOLIO OVERVIEW (KPIs)
// ══════════════════════════════════════════════════════════════
async function loadPortfolio() {
  const data = await fetchJSONWithRetry(`/api/portfolio?shock=${state.shock}`);

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
  const data = await fetchJSONWithRetry(`/api/heatmap?shock=${state.shock}`);
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

    const semanticLabel = seg.risk_label === 'RED' ? 'CRITICAL' :
                          seg.risk_label === 'YELLOW' ? 'AT-RISK' : 'HEALTHY';

    card.innerHTML = `
      <div class="heatmap-segment-name">${seg.segment}</div>
      <div class="heatmap-fhs">${seg.fhs.toFixed(2)}</div>
      <div class="heatmap-sub">/ 100 FHS</div>
      <div class="heatmap-badge">${badgeIcon} ${semanticLabel}</div>
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

  // SMA baseline from historical
  const smaWindow = 30;
  const smaData = allDatesList.map(d => {
    const idx = hist.dates.indexOf(d);
    if (idx >= smaWindow - 1) {
      const sum = hist.values.slice(idx - smaWindow + 1, idx + 1).reduce((a, b) => a + b, 0);
      return { x: new Date(d), y: +(sum / smaWindow).toFixed(2) };
    }
    return { x: new Date(d), y: null };
  });

  // Population spread band (±8%)
  const spreadUpper = allDatesList.map(d => {
    const idx = hist.dates.indexOf(d);
    return { x: new Date(d), y: idx !== -1 ? +(hist.values[idx] * 1.08).toFixed(2) : null };
  });
  const spreadLower = allDatesList.map(d => {
    const idx = hist.dates.indexOf(d);
    return { x: new Date(d), y: idx !== -1 ? +(hist.values[idx] * 0.92).toFixed(2) : null };
  });

  if (state.chart) {
    state.chart.destroy();
  }

  const ctx = $('#forecastChart').getContext('2d');
  
  const computed = getComputedStyle(document.documentElement);
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  
  const textColor = computed.getPropertyValue('--text-muted').trim() || '#64748b';
  const textPrimary = computed.getPropertyValue('--text-primary').trim() || '#0f172a';
  const gridColor = isLight ? 'rgba(0, 0, 0, 0.12)' : (computed.getPropertyValue('--border-subtle').trim() || 'rgba(148, 163, 184, 0.06)');
  const tooltipBg = computed.getPropertyValue('--bg-card').trim() || 'rgba(15, 23, 42, 0.95)';
  const tooltipBorder = computed.getPropertyValue('--border-subtle').trim() || 'rgba(124, 58, 237, 0.3)';

  // Gradient for historical line
  const histGrad = ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
  histGrad.addColorStop(0, computed.getPropertyValue('--accent-purple').trim() || '#3b82f6');
  histGrad.addColorStop(1, computed.getPropertyValue('--accent-teal').trim() || '#06b6d4');

  state.chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        // Population Spread (fill)
        {
          label: 'Population Spread',
          data: spreadUpper,
          borderColor: 'transparent',
          backgroundColor: isLight ? 'rgba(59, 130, 246, 0.12)' : 'rgba(59, 130, 246, 0.06)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.2,
          order: 6,
        },
        {
          label: '_spreadLower',
          data: spreadLower,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.2,
          order: 7,
        },
        // Historical FHS
        {
          label: 'Avg Historical FHS',
          data: histData,
          borderColor: histGrad,
          borderWidth: 2,
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHitRadius: 10,
          pointHoverBackgroundColor: '#3b82f6',
          tension: 0.2,
          order: 3,
        },
        // SMA Baseline
        {
          label: 'Baseline (30-day SMA)',
          data: smaData,
          borderColor: isLight ? 'rgba(100, 116, 139, 0.7)' : 'rgba(148, 163, 184, 0.4)',
          borderWidth: 1.5,
          borderDash: [6, 4],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.2,
          order: 4,
        },
        // Forecast Uncertainty Band
        {
          label: 'Uncertainty Band',
          data: fcUpper,
          borderColor: 'transparent',
          backgroundColor: isLight ? 'rgba(245, 158, 11, 0.3)' : 'rgba(245, 158, 11, 0.15)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.2,
          order: 5,
        },
        {
          label: '_fcLower',
          data: fcLower,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.2,
          order: 5,
        },
        // AI Forecast line
        {
          label: 'AI Forecast (Holt-Winters)',
          data: fcData,
          borderColor: computed.getPropertyValue('--risk-yellow').trim() || '#f59e0b',
          borderWidth: 3,
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHitRadius: 10,
          pointHoverBackgroundColor: computed.getPropertyValue('--risk-yellow').trim() || '#f59e0b',
          tension: 0.2,
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

          // Critical threshold at 60
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
          ctx.fillText('Critical threshold (60)', chartArea.right - 4, y60 - 6);

          // Healthy threshold at 75
          const y75 = yScale.getPixelForValue(75);
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(34, 197, 94, 0.5)';
          ctx.moveTo(chartArea.left, y75);
          ctx.lineTo(chartArea.right, y75);
          ctx.stroke();

          ctx.fillStyle = 'rgba(34, 197, 94, 0.7)';
          ctx.fillText('Healthy threshold (75)', chartArea.right - 4, y75 - 6);

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
  const semanticLabel = riskLabel === 'RED' ? 'CRITICAL' : riskLabel === 'YELLOW' ? 'AT-RISK' : 'HEALTHY';

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
    The segment remains in <span class="summary-risk ${riskClass}">${riskWord} (${semanticLabel})</span> zone.${trendNote}
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
              <strong>GenAI Intervention Plan:</strong>
              <ul style="margin-top: 8px; padding-left: 20px;">
                ${alert.explanation.split(/(?<=\.)\s/).filter(s => s.trim().length > 3).map(s => `<li>${s.trim()}</li>`).join('')}
              </ul>
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
      const ctx = $('#forecastChart');
      if (ctx && ctx.parentNode) ctx.parentNode.innerHTML = '<div style="padding:40px;text-align:center;color:var(--text-muted);">⚠ Failed to load forecast. Server may be unreachable.</div>';
    });

    loadAlerts().then(() => {
      showAlertsLoading(false);
    }).catch(err => {
      console.error('Alerts load failed:', err);
      showAlertsLoading(false);
      $('#alertsBanner').innerHTML = '<div class="alert-banner danger">⚠ Failed to load risk alerts. Please try again later.</div>';
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
