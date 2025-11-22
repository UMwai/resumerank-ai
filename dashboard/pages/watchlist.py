"""
Watchlist Page

Allows users to add/remove stocks to monitor and displays
combined signals for watchlist items.
"""

from datetime import datetime
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from components.cards import metric_card, opportunity_card, signal_card
from components.charts import score_comparison_chart, multi_line_chart
from components.tables import opportunity_table, styled_dataframe
from components.filters import ticker_filter, search_filter


# Session state key for watchlist
WATCHLIST_KEY = "user_watchlist"


def get_watchlist() -> List[str]:
    """Get the current watchlist from session state."""
    if WATCHLIST_KEY not in st.session_state:
        st.session_state[WATCHLIST_KEY] = []
    return st.session_state[WATCHLIST_KEY]


def add_to_watchlist(ticker: str) -> bool:
    """
    Add a ticker to the watchlist.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if added, False if already exists
    """
    watchlist = get_watchlist()
    ticker = ticker.upper().strip()

    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        st.session_state[WATCHLIST_KEY] = watchlist
        return True
    return False


def remove_from_watchlist(ticker: str) -> bool:
    """
    Remove a ticker from the watchlist.

    Args:
        ticker: Stock ticker symbol

    Returns:
        True if removed, False if not found
    """
    watchlist = get_watchlist()
    ticker = ticker.upper().strip()

    if ticker in watchlist:
        watchlist.remove(ticker)
        st.session_state[WATCHLIST_KEY] = watchlist
        return True
    return False


def clear_watchlist() -> None:
    """Clear all items from the watchlist."""
    st.session_state[WATCHLIST_KEY] = []


def render_watchlist_page(
    combined_fetcher: Any,
    clinical_fetcher: Any,
    patent_fetcher: Any,
    insider_fetcher: Any,
) -> None:
    """
    Render the watchlist page.

    Args:
        combined_fetcher: CombinedSignalFetcher instance
        clinical_fetcher: ClinicalTrialFetcher instance
        patent_fetcher: PatentIntelligenceFetcher instance
        insider_fetcher: InsiderHiringFetcher instance
    """
    st.title("Watchlist")
    st.markdown("Monitor your selected stocks across all signal systems")

    # Get current watchlist
    watchlist = get_watchlist()

    # Watchlist management section
    st.subheader("Manage Watchlist")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        new_ticker = st.text_input(
            "Add Ticker",
            placeholder="Enter ticker symbol (e.g., MRNA)",
            key="watchlist_add_input"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("Add", key="watchlist_add_btn", use_container_width=True):
            if new_ticker:
                if add_to_watchlist(new_ticker):
                    st.success(f"Added {new_ticker.upper()} to watchlist")
                    st.rerun()
                else:
                    st.warning(f"{new_ticker.upper()} is already in your watchlist")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear All", key="watchlist_clear_btn", type="secondary", use_container_width=True):
            clear_watchlist()
            st.info("Watchlist cleared")
            st.rerun()

    # Display current watchlist
    if watchlist:
        st.markdown("**Current Watchlist:**")

        # Display as removable chips
        cols = st.columns(min(len(watchlist), 8))
        for i, ticker in enumerate(watchlist):
            with cols[i % 8]:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{ticker}**")
                with col_b:
                    if st.button("x", key=f"remove_{ticker}", help=f"Remove {ticker}"):
                        remove_from_watchlist(ticker)
                        st.rerun()
    else:
        st.info("Your watchlist is empty. Add tickers above to start monitoring.")

    st.divider()

    # If watchlist is empty, show instructions
    if not watchlist:
        st.markdown("""
        ### Getting Started

        1. **Add tickers** using the input above
        2. **View combined signals** from all three intelligence systems
        3. **Track performance** of your selected stocks

        #### Suggested Tickers
        """)

        # Quick-add buttons for common biotech tickers
        suggested_tickers = ['MRNA', 'VRTX', 'ABBV', 'PFE', 'GILD', 'REGN', 'BIIB', 'AMGN']

        cols = st.columns(4)
        for i, ticker in enumerate(suggested_tickers):
            with cols[i % 4]:
                if st.button(f"+ {ticker}", key=f"quick_add_{ticker}", use_container_width=True):
                    add_to_watchlist(ticker)
                    st.rerun()

        return

    # Fetch data for watchlist items
    st.subheader("Watchlist Signals")

    # Get combined opportunities
    try:
        all_opportunities_df = combined_fetcher.get_combined_opportunities_df()

        # Filter to watchlist items
        if not all_opportunities_df.empty:
            watchlist_df = all_opportunities_df[
                all_opportunities_df['ticker'].isin(watchlist)
            ]
        else:
            watchlist_df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        watchlist_df = pd.DataFrame()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Watched Stocks", len(watchlist))

    with col2:
        with_signals = len(watchlist_df) if not watchlist_df.empty else 0
        metric_card("With Signals", with_signals)

    with col3:
        bullish = (
            len(watchlist_df[watchlist_df['recommendation'].isin(['STRONG BUY', 'BUY'])])
            if not watchlist_df.empty and 'recommendation' in watchlist_df.columns else 0
        )
        metric_card("Bullish", bullish)

    with col4:
        bearish = (
            len(watchlist_df[watchlist_df['recommendation'].isin(['STRONG SELL', 'SELL'])])
            if not watchlist_df.empty and 'recommendation' in watchlist_df.columns else 0
        )
        metric_card("Bearish", bearish)

    st.divider()

    # Display watchlist items with signals
    if not watchlist_df.empty:
        # Opportunity cards
        st.markdown("**Signal Overview**")

        # Sort by combined score
        watchlist_df = watchlist_df.sort_values('combined_score', ascending=False)

        cols = st.columns(3)
        for i, (_, row) in enumerate(watchlist_df.iterrows()):
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

        # Score comparison chart
        st.markdown("**Score Comparison**")
        score_comparison_chart(
            watchlist_df,
            ticker_col='ticker',
            title="Signal Breakdown by Stock"
        )

        st.divider()

        # Detailed table
        st.markdown("**Detailed View**")
        opportunity_table(watchlist_df, show_scores=True)

    else:
        # Show stocks without signals
        st.info("No signals found for your watchlist stocks.")

        st.markdown("**Watchlist Status:**")
        for ticker in watchlist:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{ticker}**")
            with col2:
                st.markdown("_No active signals_")

    st.divider()

    # Individual stock details
    st.subheader("Stock Details")

    selected_ticker = st.selectbox(
        "Select a stock for detailed view",
        options=watchlist,
        key="watchlist_detail_select"
    )

    if selected_ticker:
        render_stock_details(
            selected_ticker,
            combined_fetcher,
            clinical_fetcher,
            patent_fetcher,
            insider_fetcher,
        )


def render_stock_details(
    ticker: str,
    combined_fetcher: Any,
    clinical_fetcher: Any,
    patent_fetcher: Any,
    insider_fetcher: Any,
) -> None:
    """
    Render detailed view for a specific stock.

    Args:
        ticker: Stock ticker symbol
        combined_fetcher: CombinedSignalFetcher instance
        clinical_fetcher: ClinicalTrialFetcher instance
        patent_fetcher: PatentIntelligenceFetcher instance
        insider_fetcher: InsiderHiringFetcher instance
    """
    st.markdown(f"### {ticker} Details")

    # Get combined score
    try:
        all_opportunities = combined_fetcher.get_all_opportunities()
        combined_result = combined_fetcher.calculate_combined_score(ticker, all_opportunities)
    except Exception as e:
        st.error(f"Error fetching combined score: {e}")
        combined_result = None

    if combined_result:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Combined Score", f"{combined_result['combined_score']:.2f}")
        with col2:
            metric_card("Confidence", f"{combined_result['confidence']:.0%}")
        with col3:
            metric_card("Signal Count", combined_result['signal_count'])
        with col4:
            st.markdown(f"**{combined_result['recommendation']}**")

    # Tab for each data source
    tab1, tab2, tab3 = st.tabs(["Clinical Trials", "Patent Intel", "Insider/Hiring"])

    with tab1:
        try:
            trials_df = clinical_fetcher.get_monitored_trials()
            if not trials_df.empty and 'company_ticker' in trials_df.columns:
                ticker_trials = trials_df[trials_df['company_ticker'] == ticker]
                if not ticker_trials.empty:
                    st.markdown(f"**{len(ticker_trials)} active trials**")
                    display_cols = ['trial_id', 'drug_name', 'indication', 'phase', 'status', 'expected_completion']
                    display_cols = [c for c in display_cols if c in ticker_trials.columns]
                    styled_dataframe(ticker_trials[display_cols])
                else:
                    st.info("No active clinical trials found.")
            else:
                st.info("No clinical trial data available.")
        except Exception as e:
            st.warning(f"Could not fetch clinical trial data: {e}")

    with tab2:
        try:
            patent_df = patent_fetcher.get_generic_opportunities()
            if not patent_df.empty and 'branded_company_ticker' in patent_df.columns:
                ticker_patents = patent_df[patent_df['branded_company_ticker'] == ticker]
                if not ticker_patents.empty:
                    st.markdown(f"**{len(ticker_patents)} patent cliff events**")
                    display_cols = ['brand_name', 'generic_name', 'annual_revenue', 'certainty_score', 'earliest_expiration']
                    display_cols = [c for c in display_cols if c in ticker_patents.columns]
                    styled_dataframe(ticker_patents[display_cols])
                else:
                    st.info("No patent cliff events found.")
            else:
                st.info("No patent data available.")
        except Exception as e:
            st.warning(f"Could not fetch patent data: {e}")

    with tab3:
        try:
            scores_df = insider_fetcher.get_signal_scores()
            if not scores_df.empty and 'company_ticker' in scores_df.columns:
                ticker_scores = scores_df[scores_df['company_ticker'] == ticker]
                if not ticker_scores.empty:
                    score = ticker_scores.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        metric_card("Insider Score", f"{score.get('insider_score', 0):.1f}")
                    with col2:
                        metric_card("Institutional Score", f"{score.get('institutional_score', 0):.1f}")
                    with col3:
                        metric_card("Hiring Score", f"{score.get('hiring_score', 0):.1f}")
                else:
                    st.info("No insider/hiring signals found.")
            else:
                st.info("No insider/hiring data available.")
        except Exception as e:
            st.warning(f"Could not fetch insider/hiring data: {e}")


def render_watchlist_demo() -> None:
    """Render watchlist page with demo data."""
    st.title("Watchlist")
    st.warning("Running in demo mode - displaying sample data")

    # Demo watchlist
    demo_watchlist = ['MRNA', 'VRTX', 'ABBV']

    st.subheader("Demo Watchlist")

    demo_data = pd.DataFrame({
        'ticker': ['MRNA', 'VRTX', 'ABBV'],
        'company_name': ['Moderna', 'Vertex', 'AbbVie'],
        'combined_score': [0.82, 0.75, 0.45],
        'confidence': [0.85, 0.72, 0.65],
        'signal_count': [8, 6, 4],
        'recommendation': ['STRONG BUY', 'BUY', 'HOLD'],
        'clinical_score': [0.85, 0.70, 0.50],
        'patent_score': [0.75, 0.80, 0.45],
        'insider_score': [0.86, 0.75, 0.40],
    })

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Watched Stocks", 3)
    with col2:
        metric_card("With Signals", 3)
    with col3:
        metric_card("Bullish", 2)
    with col4:
        metric_card("Bearish", 0)

    st.divider()

    cols = st.columns(3)
    for i, (_, row) in enumerate(demo_data.iterrows()):
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
    opportunity_table(demo_data)
