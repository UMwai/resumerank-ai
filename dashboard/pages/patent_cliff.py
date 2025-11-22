"""
Patent Cliff Page

Displays 12-month patent expiration calendar, generic opportunities,
and trade recommendations from the Patent Intelligence system.
"""

from datetime import datetime, date, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st

from components.cards import metric_card, signal_card
from components.charts import timeline_chart, bar_chart, calendar_heatmap
from components.tables import styled_dataframe, signal_table
from components.filters import date_range_filter, score_filter, search_filter


def render_patent_cliff_page(fetcher: Any) -> None:
    """
    Render the patent cliff page.

    Args:
        fetcher: PatentIntelligenceFetcher instance
    """
    st.title("Patent Cliff Intelligence")
    st.markdown("Track patent expirations and generic entry opportunities")

    # Refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Refresh", key="pc_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "Calendar",
        "Generic Opportunities",
        "Trade Recommendations",
        "Litigation Status"
    ])

    with tab1:
        render_calendar_tab(fetcher)

    with tab2:
        render_opportunities_tab(fetcher)

    with tab3:
        render_trade_recommendations_tab(fetcher)

    with tab4:
        render_litigation_tab(fetcher)


def render_calendar_tab(fetcher: Any) -> None:
    """Render the patent expiration calendar tab."""
    st.subheader("Patent Expiration Calendar")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        months = st.selectbox(
            "Time Horizon",
            options=[6, 12, 18, 24],
            index=1,
            format_func=lambda x: f"{x} months",
            key="pc_calendar_months"
        )

    with col2:
        min_revenue = st.selectbox(
            "Minimum Revenue",
            options=[0, 100_000_000, 500_000_000, 1_000_000_000],
            index=1,
            format_func=lambda x: f"${x:,.0f}" if x > 0 else "All",
            key="pc_calendar_revenue"
        )

    # Get data
    try:
        calendar_df = fetcher.get_patent_calendar(months=months)
    except Exception as e:
        st.error(f"Error fetching calendar data: {e}")
        calendar_df = pd.DataFrame()

    if calendar_df.empty:
        st.info("No patent expirations found for the selected period.")
        return

    # Apply revenue filter
    if min_revenue > 0 and 'annual_revenue' in calendar_df.columns:
        calendar_df = calendar_df[calendar_df['annual_revenue'] >= min_revenue]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Patents", len(calendar_df))

    with col2:
        unique_drugs = calendar_df['brand_name'].nunique() if 'brand_name' in calendar_df.columns else 0
        metric_card("Unique Drugs", unique_drugs)

    with col3:
        if 'annual_revenue' in calendar_df.columns:
            total_revenue = calendar_df.groupby('brand_name')['annual_revenue'].first().sum()
            metric_card("Revenue at Risk", f"${total_revenue / 1e9:.1f}B")
        else:
            metric_card("Revenue at Risk", "N/A")

    with col4:
        next_90 = 0
        if 'expiration_date' in calendar_df.columns:
            calendar_df['expiration_date'] = pd.to_datetime(calendar_df['expiration_date'])
            next_90 = len(calendar_df[
                calendar_df['expiration_date'] <= datetime.now() + timedelta(days=90)
            ])
        metric_card("Next 90 Days", next_90)

    st.divider()

    # Calendar visualization
    st.markdown("**Expiration Timeline**")
    if 'expiration_date' in calendar_df.columns and 'annual_revenue' in calendar_df.columns:
        # Aggregate by drug
        drug_df = calendar_df.groupby(['brand_name', 'expiration_date']).agg({
            'annual_revenue': 'first',
            'branded_company': 'first',
            'branded_company_ticker': 'first',
        }).reset_index()

        timeline_chart(
            drug_df,
            date_col='expiration_date',
            value_col='annual_revenue',
            title="Patent Expirations by Date and Revenue"
        )

    st.divider()

    # Monthly calendar heatmap
    st.markdown("**Monthly Expiration Heatmap**")
    if 'expiration_date' in calendar_df.columns:
        monthly_counts = calendar_df.groupby(
            calendar_df['expiration_date'].dt.to_period('M')
        ).size().reset_index()
        monthly_counts.columns = ['month', 'count']
        monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()

        bar_chart(
            monthly_counts,
            x_col='month',
            y_col='count',
            title="Expirations by Month"
        )

    st.divider()

    # Detailed table
    st.markdown("**All Patent Expirations**")

    display_cols = [
        'brand_name', 'generic_name', 'branded_company', 'branded_company_ticker',
        'patent_number', 'expiration_date', 'annual_revenue', 'certainty_score'
    ]
    display_cols = [c for c in display_cols if c in calendar_df.columns]

    column_config = {
        'expiration_date': st.column_config.DateColumn("Expiration", format="YYYY-MM-DD"),
        'annual_revenue': st.column_config.NumberColumn("Revenue", format="$%,.0f"),
        'certainty_score': st.column_config.ProgressColumn("Certainty", min_value=0, max_value=100, format="%.0f%%"),
    }

    styled_dataframe(
        calendar_df[display_cols] if display_cols else calendar_df,
        column_config={k: v for k, v in column_config.items() if k in display_cols}
    )


def render_opportunities_tab(fetcher: Any) -> None:
    """Render the generic opportunities tab."""
    st.subheader("Generic Entry Opportunities")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_revenue = st.selectbox(
            "Minimum Revenue",
            options=[100_000_000, 500_000_000, 1_000_000_000, 5_000_000_000],
            index=0,
            format_func=lambda x: f"${x / 1e6:.0f}M",
            key="pc_opp_revenue"
        )

    with col2:
        min_certainty = st.slider(
            "Minimum Certainty",
            min_value=0,
            max_value=100,
            value=60,
            format="%d%%",
            key="pc_opp_certainty"
        )

    # Get data
    try:
        opportunities_df = fetcher.get_generic_opportunities(min_revenue=min_revenue)
    except Exception as e:
        st.error(f"Error fetching opportunities: {e}")
        opportunities_df = pd.DataFrame()

    if opportunities_df.empty:
        st.info("No generic opportunities found for the selected criteria.")
        return

    # Apply certainty filter
    if min_certainty > 0 and 'certainty_score' in opportunities_df.columns:
        opportunities_df = opportunities_df[opportunities_df['certainty_score'] >= min_certainty]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Opportunities", len(opportunities_df))

    with col2:
        if 'revenue_at_risk' in opportunities_df.columns:
            total_risk = opportunities_df['revenue_at_risk'].sum()
            metric_card("Total Market", f"${total_risk / 1e9:.1f}B")
        else:
            metric_card("Total Market", "N/A")

    with col3:
        high_cert = len(opportunities_df[opportunities_df['certainty_score'] >= 80]) if 'certainty_score' in opportunities_df.columns else 0
        metric_card("High Certainty", high_cert)

    with col4:
        if 'anda_count' in opportunities_df.columns:
            with_anda = len(opportunities_df[opportunities_df['anda_count'] > 0])
            metric_card("With ANDAs", with_anda)
        else:
            metric_card("With ANDAs", "N/A")

    st.divider()

    # Top opportunities cards
    st.markdown("**Top Opportunities**")

    top_opportunities = opportunities_df.head(6)
    cols = st.columns(3)

    for i, (_, row) in enumerate(top_opportunities.iterrows()):
        with cols[i % 3]:
            certainty = row.get('certainty_score', 0) / 100 if row.get('certainty_score') else 0
            signal_card(
                signal_type="patent_cliff",
                ticker=row.get('branded_company_ticker', 'N/A'),
                company_name=row.get('branded_company', 'N/A'),
                score=certainty,
                description=f"{row.get('brand_name', 'N/A')} ({row.get('generic_name', 'N/A')})",
                recommendation=row.get('recommendation', 'MONITOR'),
                source=f"${row.get('annual_revenue', 0) / 1e6:.0f}M revenue"
            )

    st.divider()

    # Revenue distribution chart
    st.markdown("**Market Opportunity Distribution**")
    if 'annual_revenue' in opportunities_df.columns:
        bar_chart(
            opportunities_df.head(15),
            x_col='brand_name',
            y_col='annual_revenue',
            title="Revenue by Drug"
        )

    st.divider()

    # Full table
    st.markdown("**All Opportunities**")
    signal_table(opportunities_df, source="patent_intelligence")


def render_trade_recommendations_tab(fetcher: Any) -> None:
    """Render the trade recommendations tab."""
    st.subheader("Active Trade Recommendations")

    # Get data
    try:
        trades_df = fetcher.get_trade_recommendations()
    except Exception as e:
        st.error(f"Error fetching trade recommendations: {e}")
        trades_df = pd.DataFrame()

    if trades_df.empty:
        st.info("No active trade recommendations found.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Active Recommendations", len(trades_df))

    with col2:
        execute = len(trades_df[trades_df['recommendation'] == 'EXECUTE TRADE']) if 'recommendation' in trades_df.columns else 0
        metric_card("Execute Now", execute)

    with col3:
        initiate = len(trades_df[trades_df['recommendation'] == 'INITIATE POSITION']) if 'recommendation' in trades_df.columns else 0
        metric_card("Initiate Position", initiate)

    with col4:
        if 'market_opportunity' in trades_df.columns:
            total_opp = trades_df['market_opportunity'].sum()
            metric_card("Total Opportunity", f"${total_opp / 1e9:.1f}B")
        else:
            metric_card("Total Opportunity", "N/A")

    st.divider()

    # Trade recommendation cards
    st.markdown("**Recommendations**")

    for _, row in trades_df.iterrows():
        col1, col2 = st.columns([3, 1])

        with col1:
            rec = row.get('recommendation', 'N/A')
            conf = row.get('confidence', 'N/A')
            certainty = row.get('certainty_score', 0)
            days = row.get('days_until_event', 0)

            st.markdown(f"""
            **{row.get('brand_name', 'N/A')}** ({row.get('generic_name', 'N/A')})

            Company: {row.get('branded_company', 'N/A')} ({row.get('branded_company_ticker', 'N/A')})

            - **Recommendation**: {rec}
            - **Confidence**: {conf}
            - **Certainty Score**: {certainty:.0f}%
            - **Days Until Event**: {days}
            - **Revenue at Risk**: ${row.get('revenue_at_risk', 0) / 1e6:.0f}M
            """)

        with col2:
            if rec == 'EXECUTE TRADE':
                st.error("EXECUTE")
            elif rec == 'INITIATE POSITION':
                st.warning("INITIATE")
            else:
                st.info("MONITOR")

        st.divider()

    # Full table
    st.markdown("**All Trade Recommendations**")

    display_cols = [
        'brand_name', 'generic_name', 'branded_company', 'branded_company_ticker',
        'recommendation', 'confidence', 'certainty_score', 'days_until_event',
        'market_opportunity', 'revenue_at_risk'
    ]
    display_cols = [c for c in display_cols if c in trades_df.columns]

    styled_dataframe(trades_df[display_cols] if display_cols else trades_df)


def render_litigation_tab(fetcher: Any) -> None:
    """Render the litigation status tab."""
    st.subheader("Active Patent Litigation")

    # Get data
    try:
        litigation_df = fetcher.get_litigation_status()
    except Exception as e:
        st.error(f"Error fetching litigation data: {e}")
        litigation_df = pd.DataFrame()

    if litigation_df.empty:
        st.info("No active litigation found.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Active Cases", len(litigation_df))

    with col2:
        unique_drugs = litigation_df['brand_name'].nunique() if 'brand_name' in litigation_df.columns else 0
        metric_card("Drugs Involved", unique_drugs)

    with col3:
        unique_companies = litigation_df['generic_company'].nunique() if 'generic_company' in litigation_df.columns else 0
        metric_card("Generic Challengers", unique_companies)

    st.divider()

    # Litigation table
    st.markdown("**All Active Litigation**")

    display_cols = [
        'brand_name', 'branded_company', 'branded_company_ticker',
        'case_name', 'generic_company', 'status', 'filing_date'
    ]
    display_cols = [c for c in display_cols if c in litigation_df.columns]

    column_config = {
        'filing_date': st.column_config.DateColumn("Filed", format="YYYY-MM-DD"),
    }

    styled_dataframe(
        litigation_df[display_cols] if display_cols else litigation_df,
        column_config={k: v for k, v in column_config.items() if k in display_cols}
    )


def render_patent_cliff_demo() -> None:
    """Render patent cliff page with demo data."""
    st.title("Patent Cliff Intelligence")
    st.warning("Running in demo mode - displaying sample data")

    demo_data = pd.DataFrame({
        'brand_name': ['Humira', 'Keytruda', 'Eliquis', 'Revlimid', 'Opdivo'],
        'generic_name': ['adalimumab', 'pembrolizumab', 'apixaban', 'lenalidomide', 'nivolumab'],
        'branded_company': ['AbbVie', 'Merck', 'Bristol-Myers', 'Bristol-Myers', 'Bristol-Myers'],
        'branded_company_ticker': ['ABBV', 'MRK', 'BMY', 'BMY', 'BMY'],
        'expiration_date': pd.date_range(start='2024-01-01', periods=5, freq='3M'),
        'annual_revenue': [20_000_000_000, 17_000_000_000, 10_000_000_000, 9_000_000_000, 8_000_000_000],
        'certainty_score': [92, 78, 85, 70, 65],
    })

    st.subheader("Patent Calendar (Demo)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Patents", 5)
    with col2:
        metric_card("Unique Drugs", 5)
    with col3:
        metric_card("Revenue at Risk", "$64.0B")
    with col4:
        metric_card("Next 90 Days", 2)

    st.divider()
    signal_table(demo_data, source="patent_intelligence")
