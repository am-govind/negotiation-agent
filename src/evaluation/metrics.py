"""
Metrics Calculator + Visualization for the Arena Dashboard.

Processes arena results and generates charts for the Streamlit dashboard.
"""
import logging
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import ARENA_DIR

logger = logging.getLogger(__name__)


def load_latest_results() -> pd.DataFrame | None:
    """Load the most recent arena results CSV."""
    csv_files = sorted(ARENA_DIR.glob("arena_results_*.csv"), reverse=True)
    if not csv_files:
        logger.warning("No arena results found")
        return None
    df = pd.read_csv(csv_files[0])
    logger.info(f"Loaded {len(df)} results from {csv_files[0].name}")
    return df


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics across all runs."""
    total = len(df)
    accepted = (df["outcome"] == "accepted").sum()

    return {
        "total_negotiations": total,
        "deals_closed": int(accepted),
        "win_rate_pct": round(accepted / total * 100, 1) if total > 0 else 0,
        "avg_margin_retained_pct": round(df["margin_retained_pct"].mean(), 1),
        "avg_rounds": round(df["rounds"].mean(), 1),
        "avg_discount_pct": round(df.get("discount_given_pct", pd.Series([0])).mean(), 1),
        "total_value_adds": int(df.get("num_value_adds", pd.Series([0])).sum()),
    }


def compute_persona_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-persona metrics."""
    metrics = []
    for persona in df["persona"].unique():
        pdf = df[df["persona"] == persona]
        metrics.append({
            "Persona": persona,
            "Win Rate (%)": round((pdf["outcome"] == "accepted").mean() * 100, 1),
            "Avg Margin (%)": round(pdf["margin_retained_pct"].mean(), 1),
            "Avg Rounds": round(pdf["rounds"].mean(), 1),
            "Avg Discount (%)": round(pdf.get("discount_given_pct", pd.Series([0])).mean(), 1),
            "Total Runs": len(pdf),
        })
    return pd.DataFrame(metrics)


# ── Plotly Charts ─────────────────────────────────────────────

def create_win_rate_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of win rate by persona."""
    persona_metrics = compute_persona_metrics(df)
    fig = px.bar(
        persona_metrics,
        x="Persona",
        y="Win Rate (%)",
        color="Persona",
        color_discrete_sequence=["#667eea", "#38ef7d", "#f093fb"],
        title="Win Rate by Buyer Persona",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        showlegend=False,
    )
    return fig


def create_margin_chart(df: pd.DataFrame) -> go.Figure:
    """Box plot of margin retained by persona."""
    fig = px.box(
        df,
        x="persona",
        y="margin_retained_pct",
        color="persona",
        color_discrete_sequence=["#667eea", "#38ef7d", "#f093fb"],
        title="Margin Retained Distribution by Persona",
        labels={"margin_retained_pct": "Margin Retained (%)", "persona": "Buyer Persona"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
        showlegend=False,
    )
    return fig


def create_rounds_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of negotiation rounds by outcome."""
    fig = px.histogram(
        df,
        x="rounds",
        color="outcome",
        barmode="group",
        color_discrete_map={
            "accepted": "#38ef7d",
            "rejected": "#f04e4e",
            "timeout": "#ffa726",
            "error": "#888",
        },
        title="Negotiation Rounds Distribution",
        labels={"rounds": "Number of Rounds", "outcome": "Outcome"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    return fig


def create_outcome_sunburst(df: pd.DataFrame) -> go.Figure:
    """Sunburst chart showing outcome breakdown by persona."""
    fig = px.sunburst(
        df,
        path=["persona", "outcome"],
        color="outcome",
        color_discrete_map={
            "accepted": "#38ef7d",
            "rejected": "#f04e4e",
            "timeout": "#ffa726",
        },
        title="Negotiation Outcomes Breakdown",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    return fig


def create_dashboard_figures(df: pd.DataFrame) -> dict[str, go.Figure]:
    """Create all dashboard figures at once."""
    return {
        "win_rate": create_win_rate_chart(df),
        "margin": create_margin_chart(df),
        "rounds": create_rounds_chart(df),
        "outcomes": create_outcome_sunburst(df),
    }
