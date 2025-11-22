"""
Insider Activity + Hiring Signals Page

Displays recent Form 4 filings, 13F institutional changes,
and job posting trends from the Insider/Hiring Signals system.
"""

from datetime import datetime, date, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st

from components.cards import metric_card, signal_card
from components.charts import bar_chart, pie_chart, multi_line_chart
from components.tables import styled_dataframe, signal_table
from components.filters import (
    date_range_filter,
    ticker_filter,
    search_filter,
    recommendation_filter,
)


def render_insider_hiring_page(fetcher: Any) -> None:
    """
    Render the insider activity and hiring signals page.

    Args:
        fetcher: InsiderHiringFetcher instance
    """
    st.title("Insider Activity & Hiring Signals")
    st.markdown("Track insider transactions, institutional holdings, and hiring patterns")

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Refresh", key="ih_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Form 4 Filings",
        "13F Holdings",
        "Hiring Trends",
        "Signal Scores"
    ])

    with tab1:
        render_form4_tab(fetcher)

    with tab2:
        render_13f_tab(fetcher)

    with tab3:
        render_hiring_tab(fetcher)

    with tab4:
        render_signal_scores_tab(fetcher)


def render_form4_tab(fetcher: Any) -> None:
    """Render the Form 4 insider transactions tab."""
    st.subheader("Recent Form 4 Filings")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        days = st.selectbox(
            "Time Period",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days",
            key="ih_form4_days"
        )

    with col2:
        transaction_type = st.selectbox(
            "Transaction Type",
            options=["All", "Purchase", "Sale"],
            key="ih_form4_type"
        )

    with col3:
        exclude_10b5 = st.checkbox("Exclude 10b5-1 Plans", value=True, key="ih_form4_10b5")

    # Get data
    try:
        form4_df = fetcher.get_recent_form4(days=days)
    except Exception as e:
        st.error(f"Error fetching Form 4 data: {e}")
        form4_df = pd.DataFrame()

    if form4_df.empty:
        st.info("No Form 4 filings found for the selected period.")
        return

    # Apply filters
    if transaction_type != "All" and 'transaction_type' in form4_df.columns:
        form4_df = form4_df[form4_df['transaction_type'] == transaction_type]

    if exclude_10b5 and 'is_10b5_1_plan' in form4_df.columns:
        form4_df = form4_df[form4_df['is_10b5_1_plan'] == False]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Transactions", len(form4_df))

    with col2:
        purchases = len(form4_df[form4_df['transaction_type'] == 'Purchase']) if 'transaction_type' in form4_df.columns else 0
        metric_card("Purchases", purchases, delta=f"+{purchases}" if purchases else None)

    with col3:
        sales = len(form4_df[form4_df['transaction_type'] == 'Sale']) if 'transaction_type' in form4_df.columns else 0
        metric_card("Sales", sales, delta=f"-{sales}" if sales else None, delta_color="inverse")

    with col4:
        if 'transaction_value' in form4_df.columns:
            total_value = form4_df['transaction_value'].sum()
            metric_card("Total Value", f"${total_value / 1e6:.1f}M")
        else:
            metric_card("Total Value", "N/A")

    st.divider()

    # Top insider buys
    st.markdown("**Top Insider Purchases**")
    try:
        top_buys_df = fetcher.get_top_insider_buys(days=days, limit=10)
        if not top_buys_df.empty:
            bar_chart(
                top_buys_df,
                x_col='company_ticker',
                y_col='total_value',
                title="Aggregate Insider Purchases by Company"
            )
    except Exception:
        pass

    st.divider()

    # Transaction type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**By Transaction Type**")
        if 'transaction_type' in form4_df.columns:
            type_counts = form4_df['transaction_type'].value_counts().reset_index()
            type_counts.columns = ['transaction_type', 'count']
            pie_chart(type_counts, values_col='count', names_col='transaction_type', title="")

    with col2:
        st.markdown("**By Insider Role**")
        # Create role categories
        if 'is_director' in form4_df.columns and 'is_officer' in form4_df.columns:
            role_data = pd.DataFrame({
                'role': ['Director', 'Officer', 'Other'],
                'count': [
                    form4_df['is_director'].sum(),
                    form4_df['is_officer'].sum(),
                    len(form4_df) - form4_df['is_director'].sum() - form4_df['is_officer'].sum()
                ]
            })
            role_data = role_data[role_data['count'] > 0]
            pie_chart(role_data, values_col='count', names_col='role', title="")

    st.divider()

    # Full transactions table
    st.markdown("**All Form 4 Filings**")
    signal_table(form4_df, source="insider_hiring")


def render_13f_tab(fetcher: Any) -> None:
    """Render the 13F institutional holdings tab."""
    st.subheader("Institutional Holdings (13F)")

    # Get data
    try:
        holdings_df = fetcher.get_institutional_changes()
    except Exception as e:
        st.error(f"Error fetching 13F data: {e}")
        holdings_df = pd.DataFrame()

    if holdings_df.empty:
        st.info("No institutional holding changes found.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Changes", len(holdings_df))

    with col2:
        new_positions = len(holdings_df[holdings_df['is_new_position'] == True]) if 'is_new_position' in holdings_df.columns else 0
        metric_card("New Positions", new_positions, delta=f"+{new_positions}" if new_positions else None)

    with col3:
        exits = len(holdings_df[holdings_df['is_exit'] == True]) if 'is_exit' in holdings_df.columns else 0
        metric_card("Position Exits", exits, delta=f"-{exits}" if exits else None, delta_color="inverse")

    with col4:
        unique_funds = holdings_df['fund_name'].nunique() if 'fund_name' in holdings_df.columns else 0
        metric_card("Unique Funds", unique_funds)

    st.divider()

    # Position changes by company
    st.markdown("**Top Position Increases**")
    if 'pct_change_shares' in holdings_df.columns and 'company_ticker' in holdings_df.columns:
        increases = holdings_df[holdings_df['pct_change_shares'] > 0].nlargest(10, 'pct_change_shares')
        if not increases.empty:
            bar_chart(
                increases,
                x_col='company_ticker',
                y_col='pct_change_shares',
                title="Largest Position Increases (%)"
            )

    st.divider()

    # New positions vs exits
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**New Positions by Company**")
        if 'is_new_position' in holdings_df.columns:
            new_pos = holdings_df[holdings_df['is_new_position'] == True]
            if not new_pos.empty and 'company_ticker' in new_pos.columns:
                company_counts = new_pos['company_ticker'].value_counts().head(10).reset_index()
                company_counts.columns = ['company_ticker', 'count']
                bar_chart(company_counts, x_col='company_ticker', y_col='count', title="")
            else:
                st.info("No new positions found.")

    with col2:
        st.markdown("**Position Exits by Company**")
        if 'is_exit' in holdings_df.columns:
            exits = holdings_df[holdings_df['is_exit'] == True]
            if not exits.empty and 'company_ticker' in exits.columns:
                company_counts = exits['company_ticker'].value_counts().head(10).reset_index()
                company_counts.columns = ['company_ticker', 'count']
                bar_chart(company_counts, x_col='company_ticker', y_col='count', title="")
            else:
                st.info("No position exits found.")

    st.divider()

    # Full holdings table
    st.markdown("**All 13F Changes**")

    display_cols = [
        'fund_name', 'company_ticker', 'company_name', 'quarter_end',
        'shares', 'market_value', 'pct_change_shares', 'pct_portfolio',
        'is_new_position', 'is_exit'
    ]
    display_cols = [c for c in display_cols if c in holdings_df.columns]

    column_config = {
        'quarter_end': st.column_config.DateColumn("Quarter", format="YYYY-MM-DD"),
        'shares': st.column_config.NumberColumn("Shares", format="%,.0f"),
        'market_value': st.column_config.NumberColumn("Value", format="$%,.0f"),
        'pct_change_shares': st.column_config.NumberColumn("Change %", format="%.1f%%"),
        'pct_portfolio': st.column_config.NumberColumn("% Portfolio", format="%.2f%%"),
    }

    styled_dataframe(
        holdings_df[display_cols] if display_cols else holdings_df,
        column_config={k: v for k, v in column_config.items() if k in display_cols}
    )


def render_hiring_tab(fetcher: Any) -> None:
    """Render the hiring trends tab."""
    st.subheader("Job Posting Trends")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        days = st.selectbox(
            "Time Period",
            options=[30, 60, 90, 180],
            index=2,
            format_func=lambda x: f"Last {x} days",
            key="ih_hiring_days"
        )

    # Get data
    try:
        hiring_df = fetcher.get_job_posting_trends(days=days)
    except Exception as e:
        st.error(f"Error fetching hiring data: {e}")
        hiring_df = pd.DataFrame()

    if hiring_df.empty:
        st.info("No job posting data found for the selected period.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_jobs = hiring_df['total_jobs'].sum() if 'total_jobs' in hiring_df.columns else 0
        metric_card("Total Job Postings", total_jobs)

    with col2:
        commercial_jobs = hiring_df['commercial_jobs'].sum() if 'commercial_jobs' in hiring_df.columns else 0
        metric_card("Commercial Roles", commercial_jobs)

    with col3:
        clinical_jobs = hiring_df['clinical_jobs'].sum() if 'clinical_jobs' in hiring_df.columns else 0
        metric_card("Clinical Roles", clinical_jobs)

    with col4:
        senior_jobs = hiring_df['senior_jobs'].sum() if 'senior_jobs' in hiring_df.columns else 0
        metric_card("Senior Roles", senior_jobs)

    st.divider()

    # Top hiring companies
    st.markdown("**Top Hiring Companies**")
    if 'total_jobs' in hiring_df.columns and 'company_ticker' in hiring_df.columns:
        top_hiring = hiring_df.nlargest(15, 'total_jobs')
        bar_chart(
            top_hiring,
            x_col='company_ticker',
            y_col='total_jobs',
            title="Job Postings by Company"
        )

    st.divider()

    # Job category distribution
    st.markdown("**Job Category Breakdown**")
    col1, col2 = st.columns(2)

    with col1:
        # Aggregate by category
        category_data = pd.DataFrame({
            'category': ['Commercial', 'Clinical', 'Manufacturing', 'R&D'],
            'count': [
                hiring_df['commercial_jobs'].sum() if 'commercial_jobs' in hiring_df.columns else 0,
                hiring_df['clinical_jobs'].sum() if 'clinical_jobs' in hiring_df.columns else 0,
                hiring_df['manufacturing_jobs'].sum() if 'manufacturing_jobs' in hiring_df.columns else 0,
                hiring_df['rd_jobs'].sum() if 'rd_jobs' in hiring_df.columns else 0,
            ]
        })
        category_data = category_data[category_data['count'] > 0]
        if not category_data.empty:
            pie_chart(category_data, values_col='count', names_col='category', title="")

    with col2:
        # Companies with commercial buildup (launch signal)
        st.markdown("**Commercial Buildup Signal**")
        if 'commercial_jobs' in hiring_df.columns:
            commercial_buildup = hiring_df[hiring_df['commercial_jobs'] >= 5]
            if not commercial_buildup.empty:
                for _, row in commercial_buildup.head(5).iterrows():
                    signal_card(
                        signal_type="insider_hiring",
                        ticker=row.get('company_ticker', 'N/A'),
                        company_name=row.get('company_name', 'N/A'),
                        score=0.8,
                        description=f"{row.get('commercial_jobs', 0)} commercial roles posted",
                        recommendation="BUY",
                        source="Launch preparation signal"
                    )
            else:
                st.info("No companies with significant commercial buildup.")

    st.divider()

    # Full hiring table
    st.markdown("**All Companies Hiring**")

    display_cols = [
        'company_ticker', 'company_name', 'total_jobs',
        'commercial_jobs', 'clinical_jobs', 'manufacturing_jobs',
        'rd_jobs', 'senior_jobs', 'removed_jobs'
    ]
    display_cols = [c for c in display_cols if c in hiring_df.columns]

    column_config = {
        'total_jobs': st.column_config.NumberColumn("Total", format="%d"),
        'commercial_jobs': st.column_config.NumberColumn("Commercial", format="%d"),
        'clinical_jobs': st.column_config.NumberColumn("Clinical", format="%d"),
        'manufacturing_jobs': st.column_config.NumberColumn("Mfg", format="%d"),
        'rd_jobs': st.column_config.NumberColumn("R&D", format="%d"),
        'senior_jobs': st.column_config.NumberColumn("Senior", format="%d"),
        'removed_jobs': st.column_config.NumberColumn("Removed", format="%d"),
    }

    styled_dataframe(
        hiring_df[display_cols] if display_cols else hiring_df,
        column_config={k: v for k, v in column_config.items() if k in display_cols}
    )


def render_signal_scores_tab(fetcher: Any) -> None:
    """Render the composite signal scores tab."""
    st.subheader("Composite Signal Scores")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        limit = st.slider("Number of Scores", min_value=10, max_value=100, value=50, key="ih_scores_limit")

    with col2:
        recommendations = recommendation_filter(key="ih_scores_recs")

    # Get data
    try:
        scores_df = fetcher.get_signal_scores(limit=limit)
    except Exception as e:
        st.error(f"Error fetching signal scores: {e}")
        scores_df = pd.DataFrame()

    if scores_df.empty:
        st.info("No signal scores found.")
        return

    # Apply recommendation filter
    if recommendations and 'recommendation' in scores_df.columns:
        scores_df = scores_df[scores_df['recommendation'].isin(recommendations)]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Scores", len(scores_df))

    with col2:
        bullish = len(scores_df[scores_df['composite_score'] >= 3]) if 'composite_score' in scores_df.columns else 0
        metric_card("Bullish", bullish)

    with col3:
        bearish = len(scores_df[scores_df['composite_score'] <= -3]) if 'composite_score' in scores_df.columns else 0
        metric_card("Bearish", bearish)

    with col4:
        avg_conf = scores_df['confidence'].mean() if 'confidence' in scores_df.columns else 0
        metric_card("Avg Confidence", f"{avg_conf:.0%}")

    st.divider()

    # Top bullish and bearish
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Bullish Signals**")
        if 'composite_score' in scores_df.columns:
            bullish_df = scores_df[scores_df['composite_score'] >= 3].nlargest(5, 'composite_score')
            for _, row in bullish_df.iterrows():
                score = (row.get('composite_score', 0) + 10) / 20  # Normalize -10 to +10 to 0-1
                signal_card(
                    signal_type="insider_hiring",
                    ticker=row.get('company_ticker', 'N/A'),
                    company_name=row.get('company_name', 'N/A'),
                    score=score,
                    description=f"Insider: {row.get('insider_score', 0):.1f} | Inst: {row.get('institutional_score', 0):.1f} | Hiring: {row.get('hiring_score', 0):.1f}",
                    recommendation=row.get('recommendation', 'NEUTRAL'),
                    source=f"{row.get('signal_count', 0)} signals"
                )

    with col2:
        st.markdown("**Top Bearish Signals**")
        if 'composite_score' in scores_df.columns:
            bearish_df = scores_df[scores_df['composite_score'] <= -3].nsmallest(5, 'composite_score')
            for _, row in bearish_df.iterrows():
                score = (row.get('composite_score', 0) + 10) / 20
                signal_card(
                    signal_type="insider_hiring",
                    ticker=row.get('company_ticker', 'N/A'),
                    company_name=row.get('company_name', 'N/A'),
                    score=score,
                    description=f"Insider: {row.get('insider_score', 0):.1f} | Inst: {row.get('institutional_score', 0):.1f} | Hiring: {row.get('hiring_score', 0):.1f}",
                    recommendation=row.get('recommendation', 'NEUTRAL'),
                    source=f"{row.get('signal_count', 0)} signals"
                )

    st.divider()

    # Score distribution
    st.markdown("**Score Distribution**")
    if 'composite_score' in scores_df.columns and 'company_ticker' in scores_df.columns:
        bar_chart(
            scores_df.head(20),
            x_col='company_ticker',
            y_col='composite_score',
            title="Composite Scores by Company"
        )

    st.divider()

    # Full scores table
    st.markdown("**All Signal Scores**")

    display_cols = [
        'company_ticker', 'company_name', 'score_date',
        'composite_score', 'confidence', 'signal_count',
        'insider_score', 'institutional_score', 'hiring_score',
        'recommendation'
    ]
    display_cols = [c for c in display_cols if c in scores_df.columns]

    column_config = {
        'score_date': st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        'composite_score': st.column_config.NumberColumn("Score", format="%.2f"),
        'confidence': st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.0%%"),
        'signal_count': st.column_config.NumberColumn("Signals", format="%d"),
        'insider_score': st.column_config.NumberColumn("Insider", format="%.1f"),
        'institutional_score': st.column_config.NumberColumn("Institutional", format="%.1f"),
        'hiring_score': st.column_config.NumberColumn("Hiring", format="%.1f"),
    }

    styled_dataframe(
        scores_df[display_cols] if display_cols else scores_df,
        column_config={k: v for k, v in column_config.items() if k in display_cols}
    )


def render_insider_hiring_demo() -> None:
    """Render insider/hiring page with demo data."""
    st.title("Insider Activity & Hiring Signals")
    st.warning("Running in demo mode - displaying sample data")

    demo_form4 = pd.DataFrame({
        'company_ticker': ['MRNA', 'VRTX', 'PFE', 'GILD'],
        'company_name': ['Moderna', 'Vertex', 'Pfizer', 'Gilead'],
        'insider_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
        'insider_title': ['CEO', 'CFO', 'VP R&D', 'Director'],
        'transaction_type': ['Purchase', 'Purchase', 'Sale', 'Purchase'],
        'transaction_value': [500000, 250000, 150000, 100000],
        'transaction_date': pd.date_range(end=datetime.now(), periods=4),
    })

    st.subheader("Form 4 Filings (Demo)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Transactions", 4)
    with col2:
        metric_card("Purchases", 3, delta="+3")
    with col3:
        metric_card("Sales", 1, delta="-1", delta_color="inverse")
    with col4:
        metric_card("Total Value", "$1.0M")

    st.divider()
    signal_table(demo_form4, source="insider_hiring")
