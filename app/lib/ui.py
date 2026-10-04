"""Shared look: palette (validated colour blind safe reference palette), chart styling, small components."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
INK, INK2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="none", height=height, margin=dict(l=8, r=8, t=28, b=8),
        font=dict(family=FONT, size=13, color=INK2), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK), bordercolor=GRID),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED), zeroline=False)
    return fig


def funnel(stages: list[tuple[str, int]], color: str = BLUE, height: int | None = None) -> go.Figure:
    """Horizontal funnel bars. Label: count, % of previous stage, % of first stage."""
    names = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    text = []
    for i, c in enumerate(counts):
        if i == 0:
            text.append(f"{c:,}")
        else:
            prev = counts[i - 1] or 1
            text.append(f"{c:,}  ·  {c / prev:.1%} of previous  ·  {c / (counts[0] or 1):.1%} of start")
    fig = go.Figure(go.Bar(x=counts, y=names, orientation="h", marker=dict(color=color, cornerradius=4), text=text,
                           textposition="outside", cliponaxis=False, textfont=dict(color=INK2),
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False, range=[0, max(counts) * 1.9])
    return style(fig, height or 70 + 52 * len(stages), legend=False)


def sidebar_controls():
    """As-of date and reveal toggle shared by every page (kept in session state)."""
    st.sidebar.markdown("### Replay controls")
    st.session_state.setdefault("asof", data.DEFAULT_ASOF.date())
    st.session_state.setdefault("reveal", False)
    st.sidebar.date_input("As-of date", key="asof", min_value=data.FIRST_ASOF.date(), max_value=data.LAST_ASOF.date(),
                          help="The console shows only what was known on this date. Models were trained on earlier data only.")
    st.sidebar.toggle("Reveal what actually happened", key="reveal",
                      help="Adds the real outcomes next to each prediction, so you can check the system against the future.")
    st.sidebar.caption("Data: Olist Brazilian marketplace, Jan 2017 to Aug 2018 (public, Kaggle).")
    return pd.Timestamp(st.session_state.asof) + pd.Timedelta(hours=23, minutes=59), st.session_state.reveal


def stat(col, label: str, value: str, note: str = ""):
    col.metric(label, value, help=note or None)


def pct(x, digits=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"
