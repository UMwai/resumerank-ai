"""
Analytics Dashboard Page

Comprehensive analytics including ROI calculator, risk metrics,
historical performance charts, and correlation analysis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.cards import metric_card
from components.charts import (
    bar_chart,
    pie_chart,
    multi_line_chart,
    correlation_matrix,
    performance_attribution_chart,
    radar_chart,
    get_theme_colors,
    get_layout_theme,
)
from utils.accuracy_tracker import (
    AccuracyTracker,
    get_accuracy_tracker,
    generate_demo_predictions,
    AccuracyMetrics,
)


def render_analytics_page(
    accuracy_tracker: Optional[AccuracyTracker] = None,
) -> None:
    """
    Render the analytics dashboard page.

    Args:
        accuracy_tracker: AccuracyTracker instance
    """
    st.title("Analytics Dashboard")
    st.markdown("Track performance, analyze signals, and calculate risk metrics")

    # Get tracker
    tracker = accuracy_tracker or get_accuracy_tracker()
    dark_mode = st.session_state.get("chart_dark_mode", False)

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance Overview",
        "ROI Calculator",
        "Risk Metrics",
        "Signal Analysis",
        "Correlation Analysis",
    ])

    with tab1:
        render_performance_overview(tracker, dark_mode)

    with tab2:
        render_roi_calculator(dark_mode)

    with tab3:
        render_risk_metrics(tracker, dark_mode)

    with tab4:
        render_signal_analysis(tracker, dark_mode)

    with tab5:
        render_correlation_analysis(tracker, dark_mode)


def render_analytics_demo() -> None:
    """Render analytics page with demo data."""
    st.title("Analytics Dashboard")
    st.warning("Running in demo mode - displaying sample analytics")

    # Initialize tracker with demo data
    tracker = get_accuracy_tracker()

    # Check if we need to generate demo data
    metrics = tracker.get_accuracy_metrics()
    if metrics.total_predictions == 0:
        with st.spinner("Generating demo data..."):
            generate_demo_predictions(tracker, count=100)
        st.rerun()

    dark_mode = st.session_state.get("chart_dark_mode", False)

    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance Overview",
        "ROI Calculator",
        "Risk Metrics",
        "Signal Analysis",
        "Correlation Analysis",
    ])

    with tab1:
        render_performance_overview(tracker, dark_mode)

    with tab2:
        render_roi_calculator(dark_mode)

    with tab3:
        render_risk_metrics(tracker, dark_mode)

    with tab4:
        render_signal_analysis(tracker, dark_mode)

    with tab5:
        render_correlation_analysis(tracker, dark_mode)


def render_performance_overview(tracker: AccuracyTracker, dark_mode: bool = False) -> None:
    """Render performance overview section."""
    st.subheader("Performance Overview")

    # Time period selector
    period = st.selectbox(
        "Time Period",
        options=["All Time", "Last 90 Days", "Last 30 Days", "Last 7 Days"],
        key="perf_period",
    )

    days = None
    if period == "Last 90 Days":
        days = 90
    elif period == "Last 30 Days":
        days = 30
    elif period == "Last 7 Days":
        days = 7

    # Get overall metrics
    metrics = tracker.get_accuracy_metrics(days=days)

    # Summary cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        metric_card("Total Predictions", metrics.total_predictions)

    with col2:
        st.metric(
            "Win Rate",
            f"{metrics.win_rate:.1%}",
            delta=f"{metrics.wins} wins" if metrics.wins > 0 else None,
        )

    with col3:
        st.metric(
            "Total ROI",
            f"{metrics.total_roi:.1%}",
            delta=f"Avg: {metrics.average_roi:.1%}" if metrics.average_roi else None,
        )

    with col4:
        metric_card("Best Trade", f"{metrics.best_roi:.1%}" if metrics.best_roi else "N/A")

    with col5:
        metric_card("Worst Trade", f"{metrics.worst_roi:.1%}" if metrics.worst_roi else "N/A")

    st.divider()

    # Win/Loss breakdown
    col1, col2 = st.columns(2)

    with col1:
        # Win/Loss pie chart
        outcome_df = pd.DataFrame({
            "outcome": ["Wins", "Losses", "Breakeven", "Pending"],
            "count": [metrics.wins, metrics.losses, metrics.breakeven, metrics.pending],
        })
        outcome_df = outcome_df[outcome_df["count"] > 0]

        if not outcome_df.empty:
            st.markdown("**Outcome Distribution**")
            colors = get_theme_colors(dark_mode)
            theme = get_layout_theme(dark_mode)

            fig = px.pie(
                outcome_df,
                values="count",
                names="outcome",
                color="outcome",
                color_discrete_map={
                    "Wins": colors["bullish"],
                    "Losses": colors["bearish"],
                    "Breakeven": colors["neutral"],
                    "Pending": colors["warning"],
                },
                hole=0.4,
            )
            fig.update_layout(height=350, **theme)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Performance by source
        st.markdown("**Performance by Source**")
        source_metrics = tracker.get_metrics_by_source(days=days)

        source_df = pd.DataFrame([
            {
                "source": source.replace("_", " ").title(),
                "predictions": m.total_predictions,
                "win_rate": m.win_rate * 100,
                "avg_roi": m.average_roi,
            }
            for source, m in source_metrics.items()
            if m.total_predictions > 0
        ])

        if not source_df.empty:
            colors = get_theme_colors(dark_mode)
            theme = get_layout_theme(dark_mode)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=source_df["source"],
                y=source_df["win_rate"],
                name="Win Rate (%)",
                marker_color=colors["bullish"],
            ))
            fig.update_layout(
                title="Win Rate by Source",
                height=350,
                yaxis_title="Win Rate (%)",
                **theme,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Performance over time
    st.markdown("**Cumulative ROI Over Time**")

    perf_df = tracker.get_performance_over_time(period="weekly")
    if not perf_df.empty:
        colors = get_theme_colors(dark_mode)
        theme = get_layout_theme(dark_mode)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Cumulative ROI line
        fig.add_trace(
            go.Scatter(
                x=perf_df["date"],
                y=perf_df["cumulative_roi"],
                name="Cumulative ROI (%)",
                line=dict(color=colors["primary"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(colors['primary'][1:3], 16)}, {int(colors['primary'][3:5], 16)}, {int(colors['primary'][5:7], 16)}, 0.1)",
            ),
            secondary_y=False,
        )

        # Weekly predictions bar
        fig.add_trace(
            go.Bar(
                x=perf_df["date"],
                y=perf_df["predictions"],
                name="Predictions",
                marker_color=colors["neutral"],
                opacity=0.5,
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Performance Timeline",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **theme,
        )
        fig.update_yaxes(title_text="Cumulative ROI (%)", secondary_y=False)
        fig.update_yaxes(title_text="Predictions", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)


def render_roi_calculator(dark_mode: bool = False) -> None:
    """Render ROI calculator section."""
    st.subheader("ROI Calculator")
    st.markdown("Calculate potential returns based on signal parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trade Parameters**")

        position_size = st.number_input(
            "Position Size ($)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            key="roi_position",
        )

        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            key="roi_entry",
        )

        target_price = st.number_input(
            "Target Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=115.0,
            step=1.0,
            key="roi_target",
        )

        stop_price = st.number_input(
            "Stop Loss ($)",
            min_value=0.01,
            max_value=10000.0,
            value=95.0,
            step=1.0,
            key="roi_stop",
        )

        win_probability = st.slider(
            "Win Probability (%)",
            min_value=0,
            max_value=100,
            value=60,
            key="roi_prob",
        ) / 100

    with col2:
        st.markdown("**Calculated Metrics**")

        # Calculations
        shares = position_size / entry_price
        profit_per_share = target_price - entry_price
        loss_per_share = entry_price - stop_price

        potential_profit = shares * profit_per_share
        potential_loss = shares * loss_per_share

        roi_win = (potential_profit / position_size) * 100
        roi_loss = (potential_loss / position_size) * 100

        expected_value = (win_probability * potential_profit) - ((1 - win_probability) * potential_loss)
        expected_roi = (expected_value / position_size) * 100

        risk_reward = profit_per_share / loss_per_share if loss_per_share > 0 else 0

        # Display metrics
        st.metric("Shares", f"{shares:.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Potential Profit",
                f"${potential_profit:,.2f}",
                delta=f"+{roi_win:.1f}%",
            )
        with col_b:
            st.metric(
                "Potential Loss",
                f"-${potential_loss:,.2f}",
                delta=f"-{roi_loss:.1f}%",
                delta_color="inverse",
            )

        st.metric(
            "Expected Value",
            f"${expected_value:,.2f}",
            delta=f"{expected_roi:.1f}% expected ROI",
        )

        st.metric("Risk/Reward Ratio", f"{risk_reward:.2f}:1")

        # Kelly Criterion
        b = risk_reward
        p = win_probability
        q = 1 - p
        kelly = ((b * p) - q) / b if b > 0 else 0
        kelly_pct = max(0, kelly * 100)

        st.metric(
            "Kelly Criterion",
            f"{kelly_pct:.1f}%",
            help="Optimal position size as % of portfolio",
        )

    st.divider()

    # Position sizing table
    st.markdown("**Position Sizing Scenarios**")

    scenarios = []
    for prob in [0.4, 0.5, 0.6, 0.7, 0.8]:
        ev = (prob * potential_profit) - ((1 - prob) * potential_loss)
        ev_roi = (ev / position_size) * 100

        b = risk_reward
        kelly = ((b * prob) - (1 - prob)) / b if b > 0 else 0
        kelly_size = max(0, kelly) * position_size

        scenarios.append({
            "Win Probability": f"{prob:.0%}",
            "Expected Value": f"${ev:,.2f}",
            "Expected ROI": f"{ev_roi:.1f}%",
            "Kelly Size": f"${kelly_size:,.0f}",
        })

    st.dataframe(pd.DataFrame(scenarios), use_container_width=True)


def render_risk_metrics(tracker: AccuracyTracker, dark_mode: bool = False) -> None:
    """Render risk metrics section."""
    st.subheader("Risk Metrics Dashboard")

    metrics = tracker.get_accuracy_metrics()
    perf_df = tracker.get_performance_over_time(period="weekly")

    # Risk metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sharpe = metrics.sharpe_ratio
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}" if sharpe else "N/A",
            help="Risk-adjusted return (higher is better)",
        )

    with col2:
        max_dd = metrics.max_drawdown
        st.metric(
            "Max Drawdown",
            f"{max_dd:.1f}%" if max_dd else "N/A",
            help="Largest peak-to-trough decline",
        )

    with col3:
        pf = metrics.profit_factor
        st.metric(
            "Profit Factor",
            f"{pf:.2f}" if pf else "N/A",
            help="Gross profit / Gross loss (>1 is profitable)",
        )

    with col4:
        st.metric(
            "Win/Loss Ratio",
            f"{metrics.wins}:{metrics.losses}",
            delta=f"{metrics.win_rate:.1%} win rate",
        )

    st.divider()

    # Confidence vs Accuracy
    st.markdown("**Accuracy by Confidence Level**")

    conf_data = []
    if metrics.high_confidence_accuracy is not None:
        conf_data.append({"level": "High (>80%)", "accuracy": metrics.high_confidence_accuracy * 100})
    if metrics.medium_confidence_accuracy is not None:
        conf_data.append({"level": "Medium (50-80%)", "accuracy": metrics.medium_confidence_accuracy * 100})
    if metrics.low_confidence_accuracy is not None:
        conf_data.append({"level": "Low (<50%)", "accuracy": metrics.low_confidence_accuracy * 100})

    if conf_data:
        conf_df = pd.DataFrame(conf_data)
        colors = get_theme_colors(dark_mode)
        theme = get_layout_theme(dark_mode)

        fig = px.bar(
            conf_df,
            x="level",
            y="accuracy",
            color="accuracy",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        )
        fig.update_layout(
            title="Win Rate by Confidence Level",
            height=350,
            xaxis_title="Confidence Level",
            yaxis_title="Win Rate (%)",
            showlegend=False,
            **theme,
        )
        fig.add_hline(y=50, line_dash="dash", line_color=colors["neutral"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Drawdown chart
    st.markdown("**Drawdown Analysis**")

    if not perf_df.empty and "cumulative_roi" in perf_df.columns:
        colors = get_theme_colors(dark_mode)
        theme = get_layout_theme(dark_mode)

        cumulative = perf_df["cumulative_roi"].values
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=perf_df["date"],
            y=drawdown,
            fill="tozeroy",
            fillcolor=f"rgba({int(colors['bearish'][1:3], 16)}, {int(colors['bearish'][3:5], 16)}, {int(colors['bearish'][5:7], 16)}, 0.3)",
            line=dict(color=colors["bearish"]),
            name="Drawdown",
        ))
        fig.update_layout(
            title="Drawdown from Peak",
            height=300,
            yaxis_title="Drawdown (%)",
            **theme,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_signal_analysis(tracker: AccuracyTracker, dark_mode: bool = False) -> None:
    """Render signal analysis section."""
    st.subheader("Signal Analysis")

    # ROI by signal type
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ROI by Signal Type**")

        signal_roi = tracker.get_roi_by_signal_type()
        if signal_roi:
            signal_df = pd.DataFrame([
                {
                    "signal_type": st.replace("_", " ").title(),
                    "count": data["count"],
                    "win_rate": data["win_rate"] * 100,
                    "avg_roi": data["avg_roi"],
                    "total_roi": data["total_roi"],
                }
                for st, data in signal_roi.items()
            ])

            colors = get_theme_colors(dark_mode)
            theme = get_layout_theme(dark_mode)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=signal_df["signal_type"],
                y=signal_df["avg_roi"],
                marker_color=[colors["bullish"] if "Bullish" in t else colors["bearish"] for t in signal_df["signal_type"]],
            ))
            fig.update_layout(
                title="Average ROI by Signal Type",
                height=350,
                yaxis_title="Average ROI (%)",
                **theme,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Predictions by Source**")

        source_metrics = tracker.get_metrics_by_source()
        source_df = pd.DataFrame([
            {"source": s.replace("_", " ").title(), "count": m.total_predictions}
            for s, m in source_metrics.items()
            if m.total_predictions > 0
        ])

        if not source_df.empty:
            pie_chart(
                source_df,
                values_col="count",
                names_col="source",
                title="Predictions by Source",
                dark_mode=dark_mode,
            )

    st.divider()

    # Recent predictions table
    st.markdown("**Recent Predictions**")

    pred_df = tracker.get_predictions_df(limit=20)
    if not pred_df.empty:
        display_cols = ["ticker", "source", "signal_type", "score", "confidence", "entry_price", "exit_price", "outcome", "roi"]
        display_cols = [c for c in display_cols if c in pred_df.columns]

        # Format columns
        styled_df = pred_df[display_cols].copy()
        if "score" in styled_df.columns:
            styled_df["score"] = styled_df["score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        if "confidence" in styled_df.columns:
            styled_df["confidence"] = styled_df["confidence"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
        if "roi" in styled_df.columns:
            styled_df["roi"] = styled_df["roi"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

        st.dataframe(
            styled_df,
            column_config={
                "ticker": "Ticker",
                "source": "Source",
                "signal_type": "Type",
                "score": "Score",
                "confidence": "Confidence",
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
                "outcome": "Outcome",
                "roi": "ROI",
            },
            use_container_width=True,
            height=400,
        )


def render_correlation_analysis(tracker: AccuracyTracker, dark_mode: bool = False) -> None:
    """Render correlation analysis section."""
    st.subheader("Correlation Analysis")

    pred_df = tracker.get_predictions_df(limit=500)

    if pred_df.empty:
        st.info("Not enough data for correlation analysis.")
        return

    # Prepare correlation data
    numeric_cols = ["score", "confidence", "roi"]
    available_cols = [c for c in numeric_cols if c in pred_df.columns]

    if len(available_cols) >= 2:
        # Add outcome numeric
        if "outcome" in pred_df.columns:
            pred_df["outcome_score"] = pred_df["outcome"].map({
                "win": 1, "breakeven": 0, "loss": -1, "pending": np.nan
            })
            available_cols.append("outcome_score")

        corr_df = pred_df[available_cols].dropna()

        if not corr_df.empty:
            st.markdown("**Signal Attribute Correlations**")
            correlation_matrix(
                corr_df,
                title="Correlation Matrix",
                dark_mode=dark_mode,
            )

    st.divider()

    # Score vs ROI scatter
    st.markdown("**Score vs ROI Analysis**")

    if "score" in pred_df.columns and "roi" in pred_df.columns:
        valid_df = pred_df.dropna(subset=["score", "roi"])

        if not valid_df.empty:
            colors = get_theme_colors(dark_mode)
            theme = get_layout_theme(dark_mode)

            fig = px.scatter(
                valid_df,
                x="score",
                y="roi",
                color="source" if "source" in valid_df.columns else None,
                size="confidence" if "confidence" in valid_df.columns else None,
                hover_data=["ticker"] if "ticker" in valid_df.columns else None,
                color_discrete_sequence=[colors["primary"], colors["bullish"], colors["warning"]],
            )
            fig.update_layout(
                title="Signal Score vs Return",
                height=400,
                xaxis_title="Signal Score",
                yaxis_title="ROI (%)",
                **theme,
            )
            fig.add_hline(y=0, line_dash="dash", line_color=colors["neutral"])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Multi-system comparison radar
    st.markdown("**System Comparison**")

    source_metrics = tracker.get_metrics_by_source()

    radar_data = []
    for source, metrics in source_metrics.items():
        if metrics.total_predictions > 0:
            radar_data.append({
                "source": source.replace("_", " ").title(),
                "Win Rate": metrics.win_rate,
                "Avg ROI": min(1, max(0, (metrics.average_roi + 20) / 40)),  # Normalize
                "Predictions": min(1, metrics.total_predictions / 50),  # Normalize
                "Accuracy": metrics.accuracy,
                "Confidence Accuracy": metrics.high_confidence_accuracy or 0,
            })

    if radar_data:
        radar_df = pd.DataFrame(radar_data)
        categories = ["Win Rate", "Avg ROI", "Predictions", "Accuracy", "Confidence Accuracy"]

        radar_chart(
            radar_df,
            categories=categories,
            values_col="source",
            name_col="source",
            title="System Performance Comparison",
            dark_mode=dark_mode,
        )
