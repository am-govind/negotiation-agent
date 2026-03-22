"""
Streamlit Analytics Dashboard: Arena results visualization.

Usage:
    streamlit run src/app/dashboard.py
"""
import streamlit as st
from src.evaluation.metrics import (
    load_latest_results,
    compute_summary_metrics,
    compute_persona_metrics,
    create_dashboard_figures,
)

st.set_page_config(
    page_title="Negotiation Arena Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    h1, h2, h3 { color: #e8e8e8 !important; }
    p, span, label { color: #ccc !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Multi-Agent Arena Dashboard")
st.markdown("*Automated negotiation simulation results*")

# Load results
df = load_latest_results()

if df is None or df.empty:
    st.warning("No arena results found. Run the arena first:")
    st.code("python -m src.evaluation.arena --runs 10", language="bash")
    st.stop()

# ── Summary Metrics ───────────────────────────────────────────
summary = compute_summary_metrics(df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Negotiations", summary["total_negotiations"])
with col2:
    st.metric("Win Rate", f"{summary['win_rate_pct']}%")
with col3:
    st.metric("Avg Margin Retained", f"{summary['avg_margin_retained_pct']}%")
with col4:
    st.metric("Avg Rounds", summary["avg_rounds"])

st.markdown("---")

# ── Per-Persona Table ─────────────────────────────────────────
st.markdown("## Persona Performance Breakdown")
persona_df = compute_persona_metrics(df)
st.dataframe(persona_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────
figures = create_dashboard_figures(df)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(figures["win_rate"], use_container_width=True)
with col2:
    st.plotly_chart(figures["margin"], use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(figures["rounds"], use_container_width=True)
with col4:
    st.plotly_chart(figures["outcomes"], use_container_width=True)

# ── Raw Data ──────────────────────────────────────────────────
with st.expander("📋 Raw Results Data"):
    st.dataframe(df, use_container_width=True, hide_index=True)
