"""
Chart components using Plotly for interactive visualizations.

Features:
- Score gauges and comparisons
- Interactive timelines for trials/patents
- Heatmaps for signal clustering
- Correlation matrices
- Performance attribution charts
- Portfolio composition pie charts
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
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

# Dark mode color scheme
DARK_COLORS = {
    "bullish": "#34d399",
    "bearish": "#f87171",
    "neutral": "#9ca3af",
    "warning": "#fbbf24",
    "info": "#60a5fa",
    "primary": "#818cf8",
    "secondary": "#a78bfa",
}


def get_theme_colors(dark_mode: bool = False) -> Dict[str, str]:
    """Get color scheme based on theme."""
    return DARK_COLORS if dark_mode else COLORS


def get_layout_theme(dark_mode: bool = False) -> Dict[str, Any]:
    """Get Plotly layout theme settings."""
    if dark_mode:
        return {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#e5e7eb"},
            "xaxis": {"gridcolor": "#374151", "zerolinecolor": "#4b5563"},
            "yaxis": {"gridcolor": "#374151", "zerolinecolor": "#4b5563"},
        }
    return {
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"color": "#1f2937"},
        "xaxis": {"gridcolor": "#e5e7eb"},
        "yaxis": {"gridcolor": "#e5e7eb"},
    }


def score_gauge(
    score: float,
    title: str = "Score",
    min_val: float = 0,
    max_val: float = 1,
    height: int = 250,
    dark_mode: bool = False,
) -> None:
    """
    Display a gauge chart for a score value.

    Args:
        score: Score value to display
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    # Determine color based on score
    if score >= 0.7:
        bar_color = colors["bullish"]
    elif score <= 0.3:
        bar_color = colors["bearish"]
    else:
        bar_color = colors["warning"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16}},
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1},
            "bar": {"color": bar_color},
            "bgcolor": "rgba(128,128,128,0.1)" if dark_mode else "white",
            "borderwidth": 2,
            "bordercolor": colors["neutral"],
            "steps": [
                {"range": [min_val, 0.3 * max_val], "color": "#fef2f2" if not dark_mode else "#450a0a"},
                {"range": [0.3 * max_val, 0.7 * max_val], "color": "#fffbeb" if not dark_mode else "#422006"},
                {"range": [0.7 * max_val, max_val], "color": "#ecfdf5" if not dark_mode else "#022c22"},
            ],
            "threshold": {
                "line": {"color": colors["neutral"], "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def timeline_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    color_col: Optional[str] = None,
    title: str = "Timeline",
    height: int = 400,
    dark_mode: bool = False,
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
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for timeline.")
        return

    colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    fig = px.scatter(
        df,
        x=date_col,
        y=value_col,
        color=color_col,
        title=title,
        color_discrete_sequence=[
            colors["primary"],
            colors["bullish"],
            colors["bearish"],
            colors["warning"],
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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def interactive_timeline(
    df: pd.DataFrame,
    start_col: str,
    end_col: Optional[str] = None,
    label_col: str = "label",
    category_col: Optional[str] = None,
    title: str = "Interactive Timeline",
    height: int = 500,
    dark_mode: bool = False,
) -> None:
    """
    Display an interactive Gantt-style timeline.

    Args:
        df: DataFrame with timeline data
        start_col: Column for start dates
        end_col: Column for end dates (optional, uses start_col + 30 days if not provided)
        label_col: Column for event labels
        category_col: Column for categorization
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for timeline.")
        return

    colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    # Prepare data
    timeline_df = df.copy()
    timeline_df[start_col] = pd.to_datetime(timeline_df[start_col])

    if end_col and end_col in timeline_df.columns:
        timeline_df[end_col] = pd.to_datetime(timeline_df[end_col])
    else:
        timeline_df['_end'] = timeline_df[start_col] + timedelta(days=30)
        end_col = '_end'

    # Create Gantt chart
    if category_col and category_col in timeline_df.columns:
        fig = px.timeline(
            timeline_df,
            x_start=start_col,
            x_end=end_col,
            y=label_col,
            color=category_col,
            title=title,
            color_discrete_sequence=[
                colors["primary"],
                colors["bullish"],
                colors["warning"],
                colors["bearish"],
            ],
        )
    else:
        fig = px.timeline(
            timeline_df,
            x_start=start_col,
            x_end=end_col,
            y=label_col,
            title=title,
        )
        fig.update_traces(marker_color=colors["primary"])

    fig.update_layout(
        height=height,
        xaxis_title="Date",
        yaxis_title="",
        showlegend=bool(category_col),
        margin=dict(l=40, r=40, t=60, b=40),
        **theme,
    )

    # Add today marker
    fig.add_vline(
        x=datetime.now(),
        line_dash="dash",
        line_color=colors["warning"],
        annotation_text="Today",
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
    dark_mode: bool = False,
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
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    fig = px.bar(
        df,
        x=x_col if orientation == "v" else y_col,
        y=y_col if orientation == "v" else x_col,
        color=color_col,
        orientation=orientation,
        title=title,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=[
            colors["primary"],
            colors["secondary"],
            colors["bullish"],
            colors["warning"],
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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def pie_chart(
    df: pd.DataFrame,
    values_col: str,
    names_col: str,
    title: str = "Distribution",
    height: int = 350,
    hole: float = 0.4,
    dark_mode: bool = False,
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
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        title=title,
        hole=hole,
        color_discrete_sequence=[
            colors["primary"],
            colors["bullish"],
            colors["warning"],
            colors["bearish"],
            colors["info"],
            colors["secondary"],
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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str = "Heatmap",
    height: int = 400,
    color_scale: str = "RdYlGn",
    dark_mode: bool = False,
) -> None:
    """
    Display a heatmap visualization.

    Args:
        df: DataFrame with heatmap data
        x_col: Column for x-axis categories
        y_col: Column for y-axis categories
        value_col: Column for cell values
        title: Chart title
        height: Chart height in pixels
        color_scale: Plotly color scale name
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for heatmap.")
        return

    theme = get_layout_theme(dark_mode)

    # Pivot data for heatmap
    pivot_df = df.pivot(index=y_col, columns=x_col, values=value_col)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=color_scale,
        showscale=True,
        hoverongaps=False,
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        margin=dict(l=60, r=20, t=60, b=40),
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def signal_heatmap(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    score_cols: List[str] = None,
    title: str = "Signal Strength Heatmap",
    height: int = 400,
    dark_mode: bool = False,
) -> None:
    """
    Display a heatmap of signal strengths across companies and signal types.

    Args:
        df: DataFrame with score data
        ticker_col: Column for company tickers
        score_cols: List of score columns to display
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for heatmap.")
        return

    theme = get_layout_theme(dark_mode)

    if score_cols is None:
        score_cols = ['clinical_score', 'patent_score', 'insider_score']

    # Filter to available columns
    available_cols = [c for c in score_cols if c in df.columns]
    if not available_cols:
        st.info("No score columns available for heatmap.")
        return

    # Prepare data
    heatmap_df = df[[ticker_col] + available_cols].set_index(ticker_col)
    heatmap_df = heatmap_df.fillna(0)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=[c.replace("_", " ").title() for c in available_cols],
        y=heatmap_df.index,
        colorscale=[
            [0, "#ef4444"],    # Red for low scores
            [0.3, "#fbbf24"],  # Yellow
            [0.5, "#d1d5db"],  # Gray for neutral
            [0.7, "#34d399"],  # Light green
            [1, "#10b981"],    # Green for high scores
        ],
        showscale=True,
        zmin=0,
        zmax=1,
        text=np.round(heatmap_df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Signal Type",
        yaxis_title="Company",
        margin=dict(l=100, r=20, t=60, b=40),
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    height: int = 400,
    dark_mode: bool = False,
) -> None:
    """
    Display a correlation matrix heatmap.

    Args:
        df: DataFrame with numeric data
        columns: List of columns to include (defaults to all numeric)
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for correlation matrix.")
        return

    theme = get_layout_theme(dark_mode)

    # Select numeric columns
    if columns:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or len(numeric_df.columns) < 2:
        st.info("Not enough numeric columns for correlation matrix.")
        return

    # Calculate correlation
    corr_matrix = numeric_df.corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[c.replace("_", " ").title() for c in corr_matrix.columns],
        y=[c.replace("_", " ").title() for c in corr_matrix.index],
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        showscale=True,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=100, r=20, t=60, b=40),
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def calendar_heatmap(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "Calendar Heatmap",
    height: int = 300,
    dark_mode: bool = False,
) -> None:
    """
    Display a calendar heatmap for events over time.

    Args:
        df: DataFrame with date and value data
        date_col: Column name for dates
        value_col: Column name for values
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for calendar.")
        return

    colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

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
            [0, "#f0f9ff" if not dark_mode else "#1e3a5f"],
            [0.5, colors["info"]],
            [1, colors["primary"]],
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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def multi_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "Trend",
    height: int = 400,
    colors: Optional[List[str]] = None,
    dark_mode: bool = False,
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
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    theme_colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    if colors is None:
        colors = [
            theme_colors["primary"],
            theme_colors["bullish"],
            theme_colors["bearish"],
            theme_colors["warning"],
            theme_colors["info"],
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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def score_comparison_chart(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    scores: Dict[str, str] = None,
    title: str = "Score Comparison",
    height: int = 400,
    dark_mode: bool = False,
) -> None:
    """
    Display a comparison chart for multiple score types.

    Args:
        df: DataFrame with score data
        ticker_col: Column for ticker/company identifier
        scores: Dictionary mapping display names to column names
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    theme_colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    if scores is None:
        scores = {
            "Clinical": "clinical_score",
            "Patent": "patent_score",
            "Insider": "insider_score",
        }

    # Create grouped bar chart
    fig = go.Figure()

    chart_colors = [theme_colors["primary"], theme_colors["bullish"], theme_colors["warning"]]

    for i, (label, col) in enumerate(scores.items()):
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=label,
                x=df[ticker_col],
                y=df[col],
                marker_color=chart_colors[i % len(chart_colors)],
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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def performance_attribution_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str = "Performance Attribution",
    height: int = 400,
    dark_mode: bool = False,
) -> None:
    """
    Display a waterfall chart for performance attribution.

    Args:
        df: DataFrame with attribution data
        category_col: Column for categories
        value_col: Column for values
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for chart.")
        return

    theme_colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    # Calculate totals
    values = df[value_col].tolist()
    categories = df[category_col].tolist()

    # Add total
    categories.append("Total")
    values.append(sum(values))

    # Determine colors
    bar_colors = []
    for i, v in enumerate(values[:-1]):
        bar_colors.append(theme_colors["bullish"] if v >= 0 else theme_colors["bearish"])
    bar_colors.append(theme_colors["primary"])  # Total

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        x=categories,
        y=values,
        connector={"line": {"color": theme_colors["neutral"]}},
        increasing={"marker": {"color": theme_colors["bullish"]}},
        decreasing={"marker": {"color": theme_colors["bearish"]}},
        totals={"marker": {"color": theme_colors["primary"]}},
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Category",
        yaxis_title="Contribution",
        margin=dict(l=40, r=40, t=60, b=40),
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def radar_chart(
    df: pd.DataFrame,
    categories: List[str],
    values_col: str,
    name_col: Optional[str] = None,
    title: str = "Radar Chart",
    height: int = 400,
    dark_mode: bool = False,
) -> None:
    """
    Display a radar/spider chart for multi-dimensional comparison.

    Args:
        df: DataFrame with radar data
        categories: List of category names for radar axes
        values_col: Column containing values (or list of columns)
        name_col: Column for trace names
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if df.empty:
        st.info("No data available for radar chart.")
        return

    theme_colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    fig = go.Figure()

    chart_colors = [
        theme_colors["primary"],
        theme_colors["bullish"],
        theme_colors["warning"],
        theme_colors["bearish"],
    ]

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row.get(cat, 0) for cat in categories]
        values.append(values[0])  # Close the polygon

        name = row[name_col] if name_col and name_col in row else f"Series {i+1}"

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            line_color=chart_colors[i % len(chart_colors)],
            opacity=0.7,
        ))

    fig.update_layout(
        title=title,
        height=height,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            ),
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40),
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def event_timeline(
    events: List[Dict[str, Any]],
    title: str = "Upcoming Events",
    height: int = 300,
    dark_mode: bool = False,
) -> None:
    """
    Display a timeline of upcoming events.

    Args:
        events: List of event dictionaries with 'date', 'title', 'type' keys
        title: Chart title
        height: Chart height in pixels
        dark_mode: Use dark mode colors
    """
    if not events:
        st.info("No events to display.")
        return

    theme_colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])

    # Create color mapping for event types
    type_colors = {
        "clinical": theme_colors["primary"],
        "patent": theme_colors["warning"],
        "regulatory": theme_colors["info"],
        "earnings": theme_colors["secondary"],
        "insider": theme_colors["bullish"],
    }

    df["color"] = df["type"].map(lambda x: type_colors.get(x, theme_colors["neutral"]))

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
        **theme,
    )

    st.plotly_chart(fig, use_container_width=True)


def sparkline(
    values: List[float],
    title: Optional[str] = None,
    height: int = 60,
    color: Optional[str] = None,
    dark_mode: bool = False,
) -> None:
    """
    Display a compact sparkline chart.

    Args:
        values: List of numeric values
        title: Optional title
        height: Chart height in pixels
        color: Line color
        dark_mode: Use dark mode colors
    """
    if not values:
        return

    theme_colors = get_theme_colors(dark_mode)
    theme = get_layout_theme(dark_mode)

    if color is None:
        # Color based on trend
        color = theme_colors["bullish"] if values[-1] >= values[0] else theme_colors["bearish"]

    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        **theme,
    )

    if title:
        st.caption(title)

    st.plotly_chart(fig, use_container_width=True)
