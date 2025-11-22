"""
Table components for displaying data in styled DataFrames.
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st


def styled_dataframe(
    df: pd.DataFrame,
    height: int = 400,
    column_config: Optional[Dict[str, Any]] = None,
    hide_index: bool = True,
    use_container_width: bool = True,
) -> None:
    """
    Display a styled DataFrame with standard formatting.

    Args:
        df: DataFrame to display
        height: Table height in pixels
        column_config: Streamlit column configuration
        hide_index: Whether to hide the index
        use_container_width: Whether to use full container width
    """
    if df.empty:
        st.info("No data available.")
        return

    st.dataframe(
        df,
        height=height,
        column_config=column_config,
        hide_index=hide_index,
        use_container_width=use_container_width,
    )


def signal_table(
    df: pd.DataFrame,
    source: str = "clinical_trials",
    height: int = 400,
) -> None:
    """
    Display a table of signals with appropriate formatting.

    Args:
        df: DataFrame with signal data
        source: Source system for appropriate formatting
        height: Table height in pixels
    """
    if df.empty:
        st.info("No signals available.")
        return

    # Define column configurations based on source
    if source == "clinical_trials":
        column_config = {
            "trial_id": st.column_config.TextColumn("Trial ID", width="small"),
            "drug_name": st.column_config.TextColumn("Drug", width="medium"),
            "company_ticker": st.column_config.TextColumn("Ticker", width="small"),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "signal_type": st.column_config.TextColumn("Signal", width="medium"),
            "signal_weight": st.column_config.NumberColumn(
                "Weight",
                format="%d",
                width="small",
            ),
            "detected_date": st.column_config.DateColumn(
                "Date",
                format="YYYY-MM-DD",
                width="small",
            ),
            "indication": st.column_config.TextColumn("Indication", width="medium"),
            "phase": st.column_config.TextColumn("Phase", width="small"),
        }
    elif source == "patent_intelligence":
        column_config = {
            "brand_name": st.column_config.TextColumn("Drug", width="medium"),
            "generic_name": st.column_config.TextColumn("Generic", width="medium"),
            "branded_company_ticker": st.column_config.TextColumn("Ticker", width="small"),
            "expiration_date": st.column_config.DateColumn(
                "Expiration",
                format="YYYY-MM-DD",
                width="small",
            ),
            "annual_revenue": st.column_config.NumberColumn(
                "Revenue",
                format="$%,.0f",
                width="medium",
            ),
            "certainty_score": st.column_config.ProgressColumn(
                "Certainty",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                width="small",
            ),
        }
    elif source == "insider_hiring":
        column_config = {
            "company_ticker": st.column_config.TextColumn("Ticker", width="small"),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "insider_name": st.column_config.TextColumn("Insider", width="medium"),
            "transaction_type": st.column_config.TextColumn("Type", width="small"),
            "transaction_value": st.column_config.NumberColumn(
                "Value",
                format="$%,.0f",
                width="medium",
            ),
            "transaction_date": st.column_config.DateColumn(
                "Date",
                format="YYYY-MM-DD",
                width="small",
            ),
        }
    else:
        column_config = {}

    # Filter columns that exist in the DataFrame
    filtered_config = {
        col: config
        for col, config in column_config.items()
        if col in df.columns
    }

    st.dataframe(
        df,
        height=height,
        column_config=filtered_config,
        hide_index=True,
        use_container_width=True,
    )


def opportunity_table(
    df: pd.DataFrame,
    height: int = 400,
    show_scores: bool = True,
) -> None:
    """
    Display a table of combined opportunities.

    Args:
        df: DataFrame with opportunity data
        height: Table height in pixels
        show_scores: Whether to show individual system scores
    """
    if df.empty:
        st.info("No opportunities available.")
        return

    column_config = {
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "company_name": st.column_config.TextColumn("Company", width="medium"),
        "combined_score": st.column_config.ProgressColumn(
            "Combined Score",
            min_value=0,
            max_value=1,
            format="%.2f",
            width="medium",
        ),
        "confidence": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0,
            max_value=1,
            format="%.0%%",
            width="small",
        ),
        "recommendation": st.column_config.TextColumn(
            "Recommendation",
            width="medium",
        ),
        "signal_count": st.column_config.NumberColumn(
            "Signals",
            format="%d",
            width="small",
        ),
    }

    if show_scores:
        column_config.update({
            "clinical_score": st.column_config.ProgressColumn(
                "Clinical",
                min_value=0,
                max_value=1,
                format="%.2f",
                width="small",
            ),
            "patent_score": st.column_config.ProgressColumn(
                "Patent",
                min_value=0,
                max_value=1,
                format="%.2f",
                width="small",
            ),
            "insider_score": st.column_config.ProgressColumn(
                "Insider",
                min_value=0,
                max_value=1,
                format="%.2f",
                width="small",
            ),
        })

    # Filter columns that exist in the DataFrame
    filtered_config = {
        col: config
        for col, config in column_config.items()
        if col in df.columns
    }

    st.dataframe(
        df,
        height=height,
        column_config=filtered_config,
        hide_index=True,
        use_container_width=True,
    )


def color_recommendation(val: str) -> str:
    """
    Return CSS style for recommendation cell coloring.

    Args:
        val: Recommendation value

    Returns:
        CSS style string
    """
    colors = {
        "STRONG BUY": "background-color: #10b981; color: white",
        "STRONG_BUY": "background-color: #10b981; color: white",
        "BUY": "background-color: #34d399; color: white",
        "HOLD": "background-color: #fbbf24; color: black",
        "NEUTRAL": "background-color: #9ca3af; color: white",
        "SELL": "background-color: #f87171; color: white",
        "STRONG SELL": "background-color: #ef4444; color: white",
        "STRONG_SHORT": "background-color: #ef4444; color: white",
        "SHORT": "background-color: #f87171; color: white",
    }
    return colors.get(val, "")


def color_score(val: float, min_val: float = 0, max_val: float = 1) -> str:
    """
    Return CSS style for score cell coloring based on value.

    Args:
        val: Score value
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        CSS style string
    """
    if pd.isna(val):
        return ""

    # Normalize to 0-1 range
    normalized = (val - min_val) / (max_val - min_val)

    if normalized >= 0.7:
        return "background-color: #10b981; color: white"
    elif normalized >= 0.5:
        return "background-color: #34d399; color: white"
    elif normalized <= 0.3:
        return "background-color: #ef4444; color: white"
    elif normalized <= 0.4:
        return "background-color: #f87171; color: white"
    else:
        return "background-color: #fbbf24; color: black"


def paginated_table(
    df: pd.DataFrame,
    page_size: int = 25,
    key: str = "table",
) -> None:
    """
    Display a paginated table with navigation.

    Args:
        df: DataFrame to display
        page_size: Number of rows per page
        key: Unique key for the table state
    """
    if df.empty:
        st.info("No data available.")
        return

    # Calculate pagination
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1

    # Page selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"{key}_page",
        )

    # Calculate slice
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    # Show info
    st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} rows")

    # Display page
    styled_dataframe(df.iloc[start_idx:end_idx])
