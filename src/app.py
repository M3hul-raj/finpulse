"""
app.py
FinPulse - NatWest Risk Intelligence Dashboard
Streamlit frontend for segment FHS forecasting and anomaly detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from fhs_calculator import compute_segment_fhs, get_risk_label
from forecaster import forecast_all_segments
from anomaly import detect_anomalies
from llm_explainer import explain_all_alerts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinPulse | NatWest Risk Intelligence",
    page_icon="🏦",
    layout="wide",
)

RISK_COLORS = {"RED": "#e74c3c", "YELLOW": "#f39c12", "GREEN": "#27ae60"}

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading customer data...")
def load_data():
    return pd.read_csv("data/historical.csv", parse_dates=["date"])

@st.cache_data(show_spinner="Running AI forecasts (first load takes ~3 min)...")
def run_forecasts(_df_hash, shock_pct):
    df = load_data()
    if shock_pct > 0:
        df = df.copy()
        df["balance"] = df["balance"] * (1 - shock_pct / 100)
    return forecast_all_segments(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 FinPulse — NatWest Segment Risk Intelligence")
st.caption("AI-powered Financial Health Score forecasting · 1,000 customers · 8 segments · 30-day horizon")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Scenario Testing")
shock_pct = st.sidebar.slider(
    "Simulate expense shock (%)",
    min_value=0, max_value=50, value=0, step=5,
    help="Simulates a sudden increase in expenses across all segments"
)
if shock_pct > 0:
    st.sidebar.warning(f"⚠️ {shock_pct}% expense shock applied")

st.sidebar.markdown("---")
st.sidebar.markdown("**How FHS is calculated**")
st.sidebar.markdown("""
- 40% Balance Trend
- 30% Income Regularity  
- 20% Spending Volatility
- 10% Debt Ratio

🔴 < 40 · 🟡 40–65 · 🟢 > 65
""")

# ── Load data ─────────────────────────────────────────────────────────────────
df_base = load_data()
if shock_pct > 0:
    df_sim = df_base.copy()
    df_sim["balance"] = df_sim["balance"] * (1 - shock_pct / 100)
else:
    df_sim = df_base

# ── Section 1: Risk Heatmap ───────────────────────────────────────────────────
st.subheader("📊 Current Segment Risk Heatmap")
seg_fhs = compute_segment_fhs(df_sim)

cols = st.columns(4)
for i, row in seg_fhs.iterrows():
    color = RISK_COLORS[row["risk_label"]]
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:{color}18;border-left:5px solid {color};
                    padding:14px;border-radius:8px;margin-bottom:12px">
            <div style="font-weight:600;font-size:15px">{row['segment']}</div>
            <div style="font-size:28px;font-weight:bold;color:{color}">{row['fhs']}</div>
            <div style="color:#888;font-size:12px">/ 100 FHS</div>
            <div style="margin-top:4px">
                <span style="background:{color};color:white;padding:2px 8px;
                             border-radius:10px;font-size:12px;font-weight:bold">
                    {row['risk_label']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Section 2: Forecast Chart ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 30-Day FHS Forecast by Segment")

forecast_results = run_forecasts(shock_pct, shock_pct)
alerts = detect_anomalies(forecast_results)

selected_seg = st.selectbox("Select segment to inspect:", list(forecast_results.keys()))
data = forecast_results[selected_seg]
hist = data["historical"]
fc   = data["forecast"]

fig = go.Figure()

# Historical line
fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values,
    name="Historical FHS",
    line=dict(color="#3498db", width=2),
))

# 30-day SMA baseline
sma = hist.rolling(30).mean()
fig.add_trace(go.Scatter(
    x=sma.index, y=sma.values,
    name="Baseline (30-day SMA)",
    line=dict(color="#95a5a6", width=1, dash="dash"),
))

# Uncertainty band
fig.add_trace(go.Scatter(
    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(230,126,34,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Uncertainty Band",
    showlegend=True,
))

# Forecast line
fig.add_trace(go.Scatter(
    x=fc["ds"], y=fc["yhat"],
    name="AI Forecast (Holt-Winters)",
    line=dict(color="#e67e22", width=2, dash="dot"),
))

# Threshold lines
fig.add_hline(y=40, line_dash="dash", line_color="red", line_width=1,
              annotation_text="RED threshold (40)")
fig.add_hline(y=65, line_dash="dash", line_color="green", line_width=1,
              annotation_text="GREEN threshold (65)")

fig.update_layout(
    title=f"FHS Forecast — {selected_seg}",
    xaxis_title="Date",
    yaxis_title="Financial Health Score (0–100)",
    yaxis=dict(range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=480,
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# Forecast summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("FHS Day 1",  f"{fc['yhat'].iloc[0]:.1f}")
c2.metric("FHS Day 30", f"{fc['yhat'].iloc[-1]:.1f}",
          delta=f"{fc['yhat'].iloc[-1]-fc['yhat'].iloc[0]:.1f}")
c3.metric("Lower Bound (worst)", f"{fc['yhat_lower'].min():.1f}")
c4.metric("Upper Bound (best)",  f"{fc['yhat_upper'].max():.1f}")

# ── Section 3: Anomaly Alerts ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Early Warning Alerts")

if not alerts:
    st.success("✅ No segments flagged for immediate intervention.")
else:
    st.error(f"**{len(alerts)} segment(s) require attention from the relationship team**")
    explanations = explain_all_alerts(alerts)

    for alert in alerts:
        color = "#e74c3c" if alert["severity"] == "CRITICAL" else "#f39c12"
        with st.expander(
            f"[{alert['severity']}] {alert['segment']} — "
            f"FHS {alert['fhs_day1']} → {alert['fhs_day30']} "
            f"| Worst-case: {alert['min_lower']}/100"
        ):
            m1, m2, m3 = st.columns(3)
            m1.metric("FHS Day 1", f"{alert['fhs_day1']}/100")
            m2.metric("FHS Day 30", f"{alert['fhs_day30']}/100",
                      delta=f"{round(alert['fhs_day30']-alert['fhs_day1'],1)}")
            m3.metric("Min Lower Bound", f"{alert['min_lower']}/100")

            st.info(f"🤖 **AI Recommendation:** {explanations[alert['segment']]}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("FinPulse · NatWest Code for Purpose Hackathon · Team BIT Mesra · Built with Streamlit + statsmodels + Gemini AI")