"""
Home/Overview Page

Displays combined opportunities from all three systems,
scored by confidence and providing a dashboard overview.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from components.cards import metric_card, opportunity_card, alert_card
from components.charts import score_gauge, bar_chart, score_comparison_chart
from components.tables import opportunity_table
from components.filters import score_filter, recommendation_filter, confidence_filter


def render_home_page(
    combined_fetcher: Any,
    clinical_fetcher: Any,
    patent_fetcher: Any,
    insider_fetcher: Any,
) -> None:
    """
    Render the home/overview page.

    Args:
        combined_fetcher: CombinedSignalFetcher instance
        clinical_fetcher: ClinicalTrialFetcher instance
        patent_fetcher: PatentIntelligenceFetcher instance
        insider_fetcher: InsiderHiringFetcher instance
    """
    st.title("Investment Intelligence Dashboard")
    st.markdown("Combined signals from Clinical Trials, Patent Intelligence, and Insider/Hiring data")

    # Refresh button
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("Refresh Data", key="home_refresh"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        st.caption(f"Updated: {datetime.now().strftime('%H:%M')}")

    st.divider()

    # Get combined data
    try:
        opportunities_df = combined_fetcher.get_combined_opportunities_df()
        alerts = combined_fetcher.get_alerts()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        opportunities_df = pd.DataFrame()
        alerts = []

    # Key Metrics Row
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_signals = len(opportunities_df) if not opportunities_df.empty else 0
        metric_card(
            "Total Signals",
            total_signals,
            help_text="Total number of companies with active signals"
        )

    with col2:
        bullish_count = (
            len(opportunities_df[opportunities_df['recommendation'].isin(['STRONG BUY', 'BUY'])])
            if not opportunities_df.empty else 0
        )
        metric_card(
            "Bullish Signals",
            bullish_count,
            delta=f"+{bullish_count}" if bullish_count > 0 else None,
            delta_color="normal" if bullish_count > 0 else "off",
            help_text="Companies with BUY or STRONG BUY recommendation"
        )

    with col3:
        bearish_count = (
            len(opportunities_df[opportunities_df['recommendation'].isin(['STRONG SELL', 'SELL'])])
            if not opportunities_df.empty else 0
        )
        metric_card(
            "Bearish Signals",
            bearish_count,
            delta=f"-{bearish_count}" if bearish_count > 0 else None,
            delta_color="inverse" if bearish_count > 0 else "off",
            help_text="Companies with SELL or STRONG SELL recommendation"
        )

    with col4:
        high_confidence = (
            len(opportunities_df[opportunities_df['confidence'] >= 0.7])
            if not opportunities_df.empty else 0
        )
        metric_card(
            "High Confidence",
            high_confidence,
            help_text="Signals with confidence >= 70%"
        )

    st.divider()

    # Alerts Section
    if alerts:
        st.subheader("Active Alerts")

        bullish_alerts = [a for a in alerts if a['type'] == 'BULLISH']
        bearish_alerts = [a for a in alerts if a['type'] == 'BEARISH']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Bullish Alerts**")
            if bullish_alerts:
                for alert in bullish_alerts[:5]:
                    alert_card(
                        alert_type=alert['type'],
                        ticker=alert['ticker'],
                        company_name=alert['company_name'],
                        message=alert['message'],
                        score=alert['combined_score'],
                        confidence=alert['confidence'],
                    )
            else:
                st.info("No bullish alerts at this time.")

        with col2:
            st.markdown("**Bearish Alerts**")
            if bearish_alerts:
                for alert in bearish_alerts[:5]:
                    alert_card(
                        alert_type=alert['type'],
                        ticker=alert['ticker'],
                        company_name=alert['company_name'],
                        message=alert['message'],
                        score=alert['combined_score'],
                        confidence=alert['confidence'],
                    )
            else:
                st.info("No bearish alerts at this time.")

        st.divider()

    # Top Opportunities Section
    st.subheader("Top Opportunities")

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            score_range = score_filter(
                key="home_score",
                label="Combined Score Range",
                default_range=(0, 1)
            )

        with col2:
            min_confidence = confidence_filter(
                key="home_confidence",
                label="Minimum Confidence",
                default=0.0
            )

        with col3:
            recommendations = recommendation_filter(
                key="home_recommendations",
                label="Recommendations"
            )

    # Apply filters
    if not opportunities_df.empty:
        filtered_df = opportunities_df[
            (opportunities_df['combined_score'] >= score_range[0]) &
            (opportunities_df['combined_score'] <= score_range[1]) &
            (opportunities_df['confidence'] >= min_confidence)
        ]

        if recommendations:
            # Handle both formats of recommendations
            rec_filter = recommendations + [r.replace(' ', '_') for r in recommendations]
            filtered_df = filtered_df[filtered_df['recommendation'].isin(rec_filter)]
    else:
        filtered_df = opportunities_df

    # Display opportunities
    if not filtered_df.empty:
        # Top opportunities cards
        st.markdown("**Top 6 Combined Opportunities**")

        top_6 = filtered_df.head(6)
        cols = st.columns(3)

        for i, (_, row) in enumerate(top_6.iterrows()):
            with cols[i % 3]:
                opportunity_card(
                    ticker=row.get('ticker', 'N/A'),
                    company_name=row.get('company_name', 'N/A'),
                    combined_score=row.get('combined_score', 0),
                    confidence=row.get('confidence', 0),
                    recommendation=row.get('recommendation', 'HOLD'),
                    clinical_score=row.get('clinical_score'),
                    patent_score=row.get('patent_score'),
                    insider_score=row.get('insider_score'),
                )

        st.divider()

        # Full table
        st.markdown("**All Opportunities**")
        opportunity_table(filtered_df, show_scores=True)

        # Score comparison chart
        if len(filtered_df) > 0:
            st.markdown("**Score Comparison**")
            chart_df = filtered_df.head(15)  # Top 15 for readability
            score_comparison_chart(
                chart_df,
                ticker_col='ticker',
                title="Score Breakdown by Company"
            )
    else:
        st.info("No opportunities match the current filters.")

    st.divider()

    # System Status Section
    st.subheader("Data Sources Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Clinical Trials**")
        try:
            trials_df = clinical_fetcher.get_monitored_trials()
            trial_count = len(trials_df) if not trials_df.empty else 0
            st.metric("Monitored Trials", trial_count)
        except Exception:
            st.warning("Unable to connect to Clinical Trials database")

    with col2:
        st.markdown("**Patent Intelligence**")
        try:
            calendar_df = patent_fetcher.get_patent_calendar(months=12)
            patent_count = len(calendar_df) if not calendar_df.empty else 0
            st.metric("Patent Events (12mo)", patent_count)
        except Exception:
            st.warning("Unable to connect to Patent Intelligence database")

    with col3:
        st.markdown("**Insider/Hiring**")
        try:
            scores_df = insider_fetcher.get_signal_scores()
            score_count = len(scores_df) if not scores_df.empty else 0
            st.metric("Active Signals", score_count)
        except Exception:
            st.warning("Unable to connect to Insider/Hiring database")


def render_home_demo() -> None:
    """Render home page with demo data when databases are unavailable."""
    st.title("Investment Intelligence Dashboard")
    st.markdown("Combined signals from Clinical Trials, Patent Intelligence, and Insider/Hiring data")

    st.warning("Running in demo mode - displaying sample data")

    # Sample data for demonstration
    demo_opportunities = pd.DataFrame({
        'ticker': ['MRNA', 'VRTX', 'ABBV', 'PFE', 'GILD'],
        'company_name': ['Moderna Inc.', 'Vertex Pharmaceuticals', 'AbbVie Inc.', 'Pfizer Inc.', 'Gilead Sciences'],
        'combined_score': [0.82, 0.75, 0.45, 0.38, 0.28],
        'confidence': [0.85, 0.72, 0.65, 0.78, 0.71],
        'signal_count': [8, 6, 4, 5, 3],
        'recommendation': ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'],
        'clinical_score': [0.85, 0.70, 0.50, 0.40, 0.30],
        'patent_score': [0.75, 0.80, 0.45, 0.35, 0.25],
        'insider_score': [0.86, 0.75, 0.40, 0.38, 0.28],
    })

    # Key Metrics
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Signals", 5)
    with col2:
        metric_card("Bullish Signals", 2, delta="+2")
    with col3:
        metric_card("Bearish Signals", 2, delta="-2", delta_color="inverse")
    with col4:
        metric_card("High Confidence", 4)

    st.divider()

    # Demo opportunities
    st.subheader("Top Opportunities (Demo)")

    cols = st.columns(3)
    for i, (_, row) in enumerate(demo_opportunities.head(3).iterrows()):
        with cols[i]:
            opportunity_card(
                ticker=row['ticker'],
                company_name=row['company_name'],
                combined_score=row['combined_score'],
                confidence=row['confidence'],
                recommendation=row['recommendation'],
                clinical_score=row['clinical_score'],
                patent_score=row['patent_score'],
                insider_score=row['insider_score'],
            )

    st.divider()
    opportunity_table(demo_opportunities)
