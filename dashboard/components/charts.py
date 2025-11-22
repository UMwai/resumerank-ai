"""
Chart components using Plotly for interactive visualizations.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# Color scheme for consistent styling
COLORS = {
    "bullish": "#10b981",
    "bearish": "#ef4444",
    "neutral": "#6b7280",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
}


def score_gauge(
    score: float,
    title: str = "Score",
    min_val: float = 0,
    max_val: float = 1,
    height: int = 250,
) -> None:
    """
    Display a gauge chart for a score value.

    Args:
        score: Score value to display
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        height: Chart height in pixels
    """
    # Determine color based on score
    if score >= 0.7:
        bar_color = COLORS["bullish"]
    elif score <= 0.3:
        bar_color = COLORS["bearish"]
    else:
        bar_color = COLORS["warning"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16}},
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1},
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [min_val, 0.3 * max_val], "color": "#fef2f2"},
                {"range": [0.3 * max_val, 0.7 * max_val], "color": "#fffbeb"},
                {"range": [0.7 * max_val, max_val], "color": "#ecfdf5"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def timeline_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    color_col: Optional[str] = None,
    title: str = "Timeline",
    height: int = 400,
) -> None:
    """
    Display a timeline chart for events or values over time.

    Args:
        df: DataFrame with timeline data
        date_col: Column name for dates
        value_col: Column name for values
        color_col: Optional column for color grouping
        title: Chart title
        height: Chart height in pixels
    """
    if df.empty:
        st.info("No data available for timeline.")
        return

    fig = px.scatter(
        df,
        x=date_col,
        y=value_col,
        color=color_col,
        title=title,
        color_discrete_sequence=[
            COLORS["primary"],
            COLORS["bullish"],
            COLORS["bearish"],
            COLORS["warning"],
        ],
    )

    fig.update_traces(marker=dict(size=10))

    fig.update_layout(
        height=height,
        xaxis_title="Date",
        yaxis_title=value_col.replace("_", " ").title(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    orientation: str = "v",
    title: str = "Chart",
    height: int = 400,
    color_discrete_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Display a bar chart.

    Args:
        df: DataFrame with chart data
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Optional column for color grouping
        orientation: "v" for vertical, "h" for horizontal
        title: Chart title
        height: Chart height in pixels
        color_discrete_map: Optional mapping of values to colors
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    fig = px.bar(
        df,
        x=x_col if orientation == "v" else y_col,
        y=y_col if orientation == "v" else x_col,
        color=color_col,
        orientation=orientation,
        title=title,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=[
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["bullish"],
            COLORS["warning"],
        ],
    )

    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def pie_chart(
    df: pd.DataFrame,
    values_col: str,
    names_col: str,
    title: str = "Distribution",
    height: int = 350,
    hole: float = 0.4,
) -> None:
    """
    Display a pie/donut chart.

    Args:
        df: DataFrame with chart data
        values_col: Column for values
        names_col: Column for segment names
        title: Chart title
        height: Chart height in pixels
        hole: Hole size for donut chart (0 for pie)
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        title=title,
        hole=hole,
        color_discrete_sequence=[
            COLORS["primary"],
            COLORS["bullish"],
            COLORS["warning"],
            COLORS["bearish"],
            COLORS["info"],
            COLORS["secondary"],
        ],
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def calendar_heatmap(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "Calendar Heatmap",
    height: int = 300,
) -> None:
    """
    Display a calendar heatmap for events over time.

    Args:
        df: DataFrame with date and value data
        date_col: Column name for dates
        value_col: Column name for values
        title: Chart title
        height: Chart height in pixels
    """
    if df.empty:
        st.info("No data available for calendar.")
        return

    # Ensure date column is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Aggregate by date
    daily = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()
    daily.columns = ["date", "value"]

    # Create date range
    date_range = pd.date_range(
        start=daily["date"].min(),
        end=daily["date"].max(),
        freq="D"
    )

    # Fill missing dates
    full_df = pd.DataFrame({"date": date_range})
    full_df["date"] = full_df["date"].dt.date
    full_df = full_df.merge(daily, on="date", how="left").fillna(0)

    # Add week and day info
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df["week"] = full_df["date"].dt.isocalendar().week
    full_df["day"] = full_df["date"].dt.dayofweek

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        x=full_df["week"],
        y=full_df["day"],
        z=full_df["value"],
        colorscale=[
            [0, "#f0f9ff"],
            [0.5, COLORS["info"]],
            [1, COLORS["primary"]],
        ],
        showscale=True,
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Week",
        yaxis_title="Day",
        yaxis=dict(
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            tickvals=list(range(7)),
        ),
        margin=dict(l=60, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def multi_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "Trend",
    height: int = 400,
    colors: Optional[List[str]] = None,
) -> None:
    """
    Display a multi-line chart for comparing trends.

    Args:
        df: DataFrame with chart data
        x_col: Column for x-axis
        y_cols: List of columns for y-axis lines
        title: Chart title
        height: Chart height in pixels
        colors: Optional list of colors for each line
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    if colors is None:
        colors = [
            COLORS["primary"],
            COLORS["bullish"],
            COLORS["bearish"],
            COLORS["warning"],
            COLORS["info"],
        ]

    fig = go.Figure()

    for i, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines+markers",
                name=col.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)]),
            ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def score_comparison_chart(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    scores: Dict[str, str] = None,
    title: str = "Score Comparison",
    height: int = 400,
) -> None:
    """
    Display a comparison chart for multiple score types.

    Args:
        df: DataFrame with score data
        ticker_col: Column for ticker/company identifier
        scores: Dictionary mapping display names to column names
        title: Chart title
        height: Chart height in pixels
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    if scores is None:
        scores = {
            "Clinical": "clinical_score",
            "Patent": "patent_score",
            "Insider": "insider_score",
        }

    # Create grouped bar chart
    fig = go.Figure()

    colors = [COLORS["primary"], COLORS["bullish"], COLORS["warning"]]

    for i, (label, col) in enumerate(scores.items()):
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=df[ticker_col],
                y=df[col],
                marker_color=colors[i % len(colors)],
            ))

    fig.update_layout(
        title=title,
        barmode="group",
        height=height,
        xaxis_title="Company",
        yaxis_title="Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def event_timeline(
    events: List[Dict[str, Any]],
    title: str = "Upcoming Events",
    height: int = 300,
) -> None:
    """
    Display a timeline of upcoming events.

    Args:
        events: List of event dictionaries with 'date', 'title', 'type' keys
        title: Chart title
        height: Chart height in pixels
    """
    if not events:
        st.info("No events to display.")
        return

    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])

    # Create color mapping for event types
    type_colors = {
        "clinical": COLORS["primary"],
        "patent": COLORS["warning"],
        "regulatory": COLORS["info"],
        "earnings": COLORS["secondary"],
    }

    df["color"] = df["type"].map(lambda x: type_colors.get(x, COLORS["neutral"]))

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["date"]],
            y=[1],
            mode="markers+text",
            marker=dict(size=20, color=row["color"]),
            text=[row["title"]],
            textposition="top center",
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
