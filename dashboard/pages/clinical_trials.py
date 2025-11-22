"""
Clinical Trials Page

Displays live trial signals, upcoming readouts, and top opportunities
from the Clinical Trial Signal Detection system.
"""

from datetime import datetime, date, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st

from components.cards import metric_card, signal_card
from components.charts import timeline_chart, bar_chart, pie_chart
from components.tables import signal_table, styled_dataframe
from components.filters import (
    ticker_filter,
    date_range_filter,
    phase_filter,
    recommendation_filter,
    search_filter,
)


def render_clinical_trials_page(fetcher: Any) -> None:
    """
    Render the clinical trials page.

    Args:
        fetcher: ClinicalTrialFetcher instance
    """
    st.title("Clinical Trial Signals")
    st.markdown("Monitor biotech clinical trials for investment signals")

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Refresh", key="ct_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Recent Signals",
        "Upcoming Readouts",
        "Top Opportunities",
        "All Trials"
    ])

    with tab1:
        render_recent_signals_tab(fetcher)

    with tab2:
        render_upcoming_readouts_tab(fetcher)

    with tab3:
        render_top_opportunities_tab(fetcher)

    with tab4:
        render_all_trials_tab(fetcher)


def render_recent_signals_tab(fetcher: Any) -> None:
    """Render the recent signals tab."""
    st.subheader("Recent Trial Signals")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox(
            "Time Period",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days",
            key="ct_signal_days"
        )

    # Get data
    try:
        signals_df = fetcher.get_recent_signals(days=days)
    except Exception as e:
        st.error(f"Error fetching signals: {e}")
        signals_df = pd.DataFrame()

    if signals_df.empty:
        st.info("No signals found for the selected time period.")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Signals", len(signals_df))

    with col2:
        positive = len(signals_df[signals_df['signal_weight'] > 0])
        metric_card("Bullish Signals", positive, delta=f"+{positive}" if positive else None)

    with col3:
        negative = len(signals_df[signals_df['signal_weight'] < 0])
        metric_card("Bearish Signals", negative, delta=f"-{negative}" if negative else None, delta_color="inverse")

    with col4:
        unique_trials = signals_df['trial_id'].nunique()
        metric_card("Trials Affected", unique_trials)

    st.divider()

    # Signal type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Signal Type Distribution**")
        if 'signal_type' in signals_df.columns:
            type_counts = signals_df['signal_type'].value_counts().reset_index()
            type_counts.columns = ['signal_type', 'count']
            pie_chart(
                type_counts,
                values_col='count',
                names_col='signal_type',
                title=""
            )

    with col2:
        st.markdown("**Signals by Company**")
        if 'company_ticker' in signals_df.columns:
            company_counts = signals_df['company_ticker'].value_counts().head(10).reset_index()
            company_counts.columns = ['company_ticker', 'count']
            bar_chart(
                company_counts,
                x_col='company_ticker',
                y_col='count',
                title=""
            )

    st.divider()

    # Signals table
    st.markdown("**All Recent Signals**")
    signal_table(signals_df, source="clinical_trials")


def render_upcoming_readouts_tab(fetcher: Any) -> None:
    """Render the upcoming readouts tab."""
    st.subheader("Upcoming Trial Readouts")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox(
            "Lookahead Period",
            options=[30, 60, 90, 180, 365],
            index=2,
            format_func=lambda x: f"Next {x} days",
            key="ct_readout_days"
        )

    with col2:
        phases = phase_filter(key="ct_readout_phases")

    # Get data
    try:
        readouts_df = fetcher.get_upcoming_readouts(days=days)
    except Exception as e:
        st.error(f"Error fetching readouts: {e}")
        readouts_df = pd.DataFrame()

    if readouts_df.empty:
        st.info("No upcoming readouts found for the selected period.")
        return

    # Apply phase filter
    if phases and 'phase' in readouts_df.columns:
        readouts_df = readouts_df[readouts_df['phase'].isin(phases)]

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Readouts", len(readouts_df))

    with col2:
        next_30 = len(readouts_df[
            pd.to_datetime(readouts_df['primary_completion_date']) <=
            datetime.now() + timedelta(days=30)
        ]) if 'primary_completion_date' in readouts_df.columns else 0
        metric_card("Next 30 Days", next_30)

    with col3:
        phase3 = len(readouts_df[readouts_df['phase'] == 'Phase 3']) if 'phase' in readouts_df.columns else 0
        metric_card("Phase 3 Trials", phase3)

    with col4:
        avg_score = readouts_df['composite_score'].mean() if 'composite_score' in readouts_df.columns else 0
        metric_card("Avg Score", f"{avg_score:.1f}" if avg_score else "N/A")

    st.divider()

    # Timeline view
    st.markdown("**Readout Timeline**")
    if 'primary_completion_date' in readouts_df.columns and 'composite_score' in readouts_df.columns:
        readouts_df['primary_completion_date'] = pd.to_datetime(readouts_df['primary_completion_date'])
        timeline_chart(
            readouts_df,
            date_col='primary_completion_date',
            value_col='composite_score',
            color_col='phase' if 'phase' in readouts_df.columns else None,
            title="Upcoming Readouts by Date"
        )

    st.divider()

    # Readouts table
    st.markdown("**All Upcoming Readouts**")

    display_cols = [
        'trial_id', 'drug_name', 'indication', 'phase',
        'company_ticker', 'company_name', 'primary_completion_date',
        'enrollment_current', 'enrollment_target',
        'composite_score', 'recommendation'
    ]
    display_cols = [c for c in display_cols if c in readouts_df.columns]

    column_config = {
        'primary_completion_date': st.column_config.DateColumn("Readout Date", format="YYYY-MM-DD"),
        'composite_score': st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f"),
        'enrollment_current': st.column_config.NumberColumn("Enrolled", format="%d"),
        'enrollment_target': st.column_config.NumberColumn("Target", format="%d"),
    }

    styled_dataframe(
        readouts_df[display_cols] if display_cols else readouts_df,
        column_config={k: v for k, v in column_config.items() if k in display_cols}
    )


def render_top_opportunities_tab(fetcher: Any) -> None:
    """Render the top opportunities tab."""
    st.subheader("Top Trial Opportunities")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Number of Opportunities", min_value=5, max_value=50, value=20, key="ct_opp_limit")

    with col2:
        recommendations = recommendation_filter(key="ct_opp_recs")

    # Get data
    try:
        opportunities_df = fetcher.get_top_opportunities(limit=limit)
    except Exception as e:
        st.error(f"Error fetching opportunities: {e}")
        opportunities_df = pd.DataFrame()

    if opportunities_df.empty:
        st.info("No actionable opportunities found.")
        return

    # Apply recommendation filter
    if recommendations and 'recommendation' in opportunities_df.columns:
        rec_filter = recommendations + [r.replace(' ', '_') for r in recommendations]
        opportunities_df = opportunities_df[opportunities_df['recommendation'].isin(rec_filter)]

    # Split by bullish/bearish
    if 'composite_score' in opportunities_df.columns:
        bullish = opportunities_df[opportunities_df['composite_score'] >= 7]
        bearish = opportunities_df[opportunities_df['composite_score'] <= 3]
    else:
        bullish = pd.DataFrame()
        bearish = pd.DataFrame()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bullish Opportunities (Score >= 7)**")
        if not bullish.empty:
            for _, row in bullish.head(5).iterrows():
                signal_card(
                    signal_type="clinical_trial",
                    ticker=row.get('company_ticker', 'N/A'),
                    company_name=row.get('company_name', 'N/A'),
                    score=row.get('composite_score', 0) / 10,
                    description=f"{row.get('drug_name', 'N/A')} - {row.get('indication', 'N/A')}",
                    recommendation=row.get('recommendation', 'HOLD'),
                    source=f"Phase {row.get('phase', 'N/A')}"
                )
        else:
            st.info("No bullish opportunities found.")

    with col2:
        st.markdown("**Bearish Opportunities (Score <= 3)**")
        if not bearish.empty:
            for _, row in bearish.head(5).iterrows():
                signal_card(
                    signal_type="clinical_trial",
                    ticker=row.get('company_ticker', 'N/A'),
                    company_name=row.get('company_name', 'N/A'),
                    score=row.get('composite_score', 0) / 10,
                    description=f"{row.get('drug_name', 'N/A')} - {row.get('indication', 'N/A')}",
                    recommendation=row.get('recommendation', 'HOLD'),
                    source=f"Phase {row.get('phase', 'N/A')}"
                )
        else:
            st.info("No bearish opportunities found.")

    st.divider()

    # Full opportunities table
    st.markdown("**All Opportunities**")
    signal_table(opportunities_df, source="clinical_trials")


def render_all_trials_tab(fetcher: Any) -> None:
    """Render the all trials tab."""
    st.subheader("Monitored Trials")

    # Search and filters
    col1, col2, col3 = st.columns(3)

    with col1:
        search = search_filter(key="ct_trial_search", placeholder="Search by drug or company...")

    with col2:
        phases = phase_filter(key="ct_trial_phases")

    # Get data
    try:
        trials_df = fetcher.get_monitored_trials()
    except Exception as e:
        st.error(f"Error fetching trials: {e}")
        trials_df = pd.DataFrame()

    if trials_df.empty:
        st.info("No monitored trials found.")
        return

    # Apply filters
    if search:
        search_lower = search.lower()
        trials_df = trials_df[
            trials_df.apply(
                lambda row: search_lower in str(row.get('drug_name', '')).lower() or
                           search_lower in str(row.get('company_name', '')).lower() or
                           search_lower in str(row.get('company_ticker', '')).lower(),
                axis=1
            )
        ]

    if phases and 'phase' in trials_df.columns:
        trials_df = trials_df[trials_df['phase'].isin(phases)]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Trials", len(trials_df))

    with col2:
        unique_companies = trials_df['company_ticker'].nunique() if 'company_ticker' in trials_df.columns else 0
        metric_card("Companies", unique_companies)

    with col3:
        recruiting = len(trials_df[trials_df['status'] == 'RECRUITING']) if 'status' in trials_df.columns else 0
        metric_card("Recruiting", recruiting)

    with col4:
        with_scores = len(trials_df[trials_df['composite_score'].notna()]) if 'composite_score' in trials_df.columns else 0
        metric_card("With Scores", with_scores)

    st.divider()

    # Trials distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trials by Phase**")
        if 'phase' in trials_df.columns:
            phase_counts = trials_df['phase'].value_counts().reset_index()
            phase_counts.columns = ['phase', 'count']
            pie_chart(phase_counts, values_col='count', names_col='phase', title="")

    with col2:
        st.markdown("**Trials by Status**")
        if 'status' in trials_df.columns:
            status_counts = trials_df['status'].value_counts().reset_index()
            status_counts.columns = ['status', 'count']
            pie_chart(status_counts, values_col='count', names_col='status', title="")

    st.divider()

    # Trials table
    st.markdown("**All Monitored Trials**")

    display_cols = [
        'trial_id', 'drug_name', 'indication', 'phase', 'status',
        'company_ticker', 'company_name', 'enrollment_current', 'enrollment_target',
        'expected_completion', 'composite_score', 'recommendation'
    ]
    display_cols = [c for c in display_cols if c in trials_df.columns]

    styled_dataframe(trials_df[display_cols] if display_cols else trials_df)


def render_clinical_trials_demo() -> None:
    """Render clinical trials page with demo data."""
    st.title("Clinical Trial Signals")
    st.warning("Running in demo mode - displaying sample data")

    demo_signals = pd.DataFrame({
        'trial_id': ['NCT04470427', 'NCT04283461', 'NCT04368728', 'NCT05436457'],
        'drug_name': ['Drug A', 'Drug B', 'Drug C', 'Drug D'],
        'company_ticker': ['MRNA', 'VRTX', 'PFE', 'GILD'],
        'company_name': ['Moderna', 'Vertex', 'Pfizer', 'Gilead'],
        'signal_type': ['ENROLLMENT_SURGE', 'DATE_CHANGE', 'STATUS_CHANGE', 'ENDPOINT_MODIFICATION'],
        'signal_weight': [3, -2, 2, -1],
        'detected_date': pd.date_range(end=datetime.now(), periods=4),
        'phase': ['Phase 3', 'Phase 2', 'Phase 3', 'Phase 2'],
        'indication': ['COVID-19', 'Cystic Fibrosis', 'Cancer', 'HIV'],
    })

    st.subheader("Recent Signals (Demo)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Signals", 4)
    with col2:
        metric_card("Bullish Signals", 2, delta="+2")
    with col3:
        metric_card("Bearish Signals", 2, delta="-2", delta_color="inverse")
    with col4:
        metric_card("Trials Affected", 4)

    st.divider()
    signal_table(demo_signals, source="clinical_trials")
