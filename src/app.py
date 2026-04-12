"""
app.py
Legacy Streamlit dashboard (alternate interface).
Primary dashboard: run `python src/api_server.py` and open http://localhost:5000.
To use this instead: `streamlit run src/app.py`
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

st.set_page_config(
    page_title="FinPulse | NatWest Risk Intelligence",
    page_icon="🏦",
    layout="wide",
)

RISK_COLORS = {"RED": "#e74c3c", "YELLOW": "#f39c12", "GREEN": "#27ae60"}

@st.cache_data(show_spinner="Loading customer data...")
def load_data():
    return pd.read_csv("data/historical.csv", parse_dates=["date"])

@st.cache_data(show_spinner="Running AI forecasts (may take 5–8 min on first load)...")
def run_forecasts(shock_pct):
    df = load_data()
    if shock_pct > 0:
        df = df.copy()
        recent_mask = df['date'] >= (df['date'].max() - pd.Timedelta(days=30))
        df.loc[recent_mask, 'balance'] -= df.loc[recent_mask, 'balance'] * (shock_pct / 100)
    return forecast_all_segments(df)

st.title("🏦 FinPulse — NatWest Segment Risk Intelligence")
st.caption("AI-powered Financial Health Score forecasting · 1,000 customers · 8 segments · 30-day horizon")
st.markdown("---")

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

🔴 < 60 · 🟡 60–75 · 🟢 > 75
""")

df_base = load_data()
if shock_pct > 0:
    df_sim = df_base.copy()
    recent_mask = df_sim['date'] >= (df_sim['date'].max() - pd.Timedelta(days=30))
    df_sim.loc[recent_mask, 'balance'] -= df_sim.loc[recent_mask, 'balance'] * (shock_pct / 100)
else:
    df_sim = df_base

st.markdown("### 🌐 Portfolio Overview")
base_fhs = compute_segment_fhs(df_sim)
critical_count = len(base_fhs[base_fhs['risk_label'] == 'RED'])
warning_count = len(base_fhs[base_fhs['risk_label'] == 'YELLOW'])
healthy_count = len(base_fhs[base_fhs['risk_label'] == 'GREEN'])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers Monitored", "1,000", "+12 this week")
k2.metric("Critical Segments (RED)", critical_count, f"{critical_count} action required", delta_color="inverse")
k3.metric("At-Risk Segments (YELLOW)", warning_count)
k4.metric("Healthy Segments (GREEN)", healthy_count)
st.markdown("---")

st.subheader("📊 Current Segment Risk Heatmap")

cols = st.columns(4)
for i, row in base_fhs.iterrows():
    color = RISK_COLORS[row["risk_label"]]
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:{color}18;border-left:5px solid {color};padding:14px;border-radius:8px;margin-bottom:12px">
            <div style="font-weight:600;font-size:15px">{row['segment']}</div>
            <div style="font-size:28px;font-weight:bold;color:{color}">{row['fhs']}</div>
            <div style="color:#888;font-size:12px">/ 100 FHS</div>
            <div style="margin-top:4px">
                <span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:bold">
                    {row['risk_label']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("📈 30-Day FHS Forecast by Segment")

forecast_results = run_forecasts(shock_pct)
alerts = detect_anomalies(forecast_results)

segment_list = list(forecast_results.keys())
default_index = segment_list.index("Daily Wage") if "Daily Wage" in segment_list else 0

selected_seg = st.selectbox(
    "Select segment to inspect:", 
    segment_list,
    index=default_index
)
data = forecast_results[selected_seg]
hist = data["historical"]
fc   = data["forecast"]

fig = go.Figure()

spread_upper = hist.values + (hist.values * 0.08)
spread_lower = hist.values - (hist.values * 0.08)

fig.add_trace(go.Scatter(
    x=pd.concat([hist.index.to_series(), hist.index.to_series().iloc[::-1]]),
    y=np.concatenate([spread_upper, spread_lower[::-1]]),
    fill="toself",
    fillcolor="rgba(52, 152, 219, 0.1)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Population Spread",
    showlegend=True,
))

fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values,
    name="Avg Historical FHS",
    line=dict(color="#3498db", width=3),
))

sma = hist.rolling(30).mean()
fig.add_trace(go.Scatter(
    x=sma.index, y=sma.values,
    name="Baseline (30-day SMA)",
    line=dict(color="#95a5a6", width=1, dash="dash"),
))

fig.add_trace(go.Scatter(
    x=pd.concat([fc["ds"], fc["ds"][::-1]]),
    y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(230,126,34,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Uncertainty Band",
    showlegend=True,
))

fig.add_trace(go.Scatter(
    x=fc["ds"], y=fc["yhat"],
    name="AI Forecast (Holt-Winters)",
    line=dict(color="#e67e22", width=2, dash="dot"),
))

fig.add_hline(y=60, line_dash="dash", line_color="red", line_width=1, annotation_text="RED threshold (60)")
fig.add_hline(y=75, line_dash="dash", line_color="green", line_width=1, annotation_text="GREEN threshold (75)")

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("FHS Day 1",  f"{fc['yhat'].iloc[0]:.1f}")
c2.metric("FHS Day 30", f"{fc['yhat'].iloc[-1]:.1f}", delta=f"{fc['yhat'].iloc[-1]-fc['yhat'].iloc[0]:.1f}")
c3.metric("Lower Bound (worst)", f"{fc['yhat_lower'].min():.1f}")
c4.metric("Upper Bound (best)",  f"{fc['yhat_upper'].max():.1f}")

st.markdown("---")
st.subheader("🚨 Early Warning Alerts")

if not alerts:
    st.success("✅ No segments flagged for immediate intervention.")
else:
    st.error(f"**{len(alerts)} segment(s) require attention from the relationship team**")
    explanations = explain_all_alerts(alerts)

    for alert in alerts:
        color = "#e74c3c" if alert["severity"] == "CRITICAL" else "#f39c12"
        with st.container(border=True):
            st.markdown(f"#### <span style='color:{color}'>[{alert['severity']}]</span> {alert['segment']}", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("FHS Day 1", f"{alert['fhs_day1']}/100")
            m2.metric("FHS Day 30", f"{alert['fhs_day30']}/100", delta=f"{round(alert['fhs_day30']-alert['fhs_day1'],1)}")
            m3.metric("Worst-Case Scenario", f"{alert['min_lower']}/100")

            st.info(f"🤖 **GenAI Intervention Plan:** {explanations[alert['segment']]}")

st.markdown("---")
st.caption("FinPulse · NatWest Code for Purpose Hackathon · Team BIT Mesra · Built with Streamlit + statsmodels + Gemini AI")