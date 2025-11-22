"""
Table components for displaying data in styled DataFrames.

Features:
- Styled DataFrames with column configurations
- Pagination for large datasets
- Export buttons integration
- Responsive design
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.export import create_export_buttons


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
    show_export: bool = True,
) -> None:
    """
    Display a table of signals with appropriate formatting.

    Args:
        df: DataFrame with signal data
        source: Source system for appropriate formatting
        height: Table height in pixels
        show_export: Whether to show export buttons
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

    # Export buttons
    if show_export:
        create_export_buttons(
            df,
            filename=f"{source}_signals",
            key_prefix=f"{source}_table",
        )


def opportunity_table(
    df: pd.DataFrame,
    height: int = 400,
    show_scores: bool = True,
    show_export: bool = True,
) -> None:
    """
    Display a table of combined opportunities.

    Args:
        df: DataFrame with opportunity data
        height: Table height in pixels
        show_scores: Whether to show individual system scores
        show_export: Whether to show export buttons
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

    # Export buttons
    if show_export:
        create_export_buttons(
            df,
            filename="opportunities",
            key_prefix="opportunities_table",
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
    show_export: bool = True,
    column_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Display a paginated table with navigation.

    Args:
        df: DataFrame to display
        page_size: Number of rows per page
        key: Unique key for the table state
        show_export: Whether to show export buttons
        column_config: Optional column configuration
    """
    if df.empty:
        st.info("No data available.")
        return

    # Calculate pagination
    total_rows = len(df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    # Page navigation controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("First", key=f"{key}_first", use_container_width=True):
            st.session_state[f"{key}_page"] = 1

    with col2:
        if st.button("Prev", key=f"{key}_prev", use_container_width=True):
            current = st.session_state.get(f"{key}_page", 1)
            st.session_state[f"{key}_page"] = max(1, current - 1)

    with col3:
        # Page selector
        current_page = st.session_state.get(f"{key}_page", 1)
        page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=min(current_page - 1, total_pages - 1),
            key=f"{key}_page_select",
            label_visibility="collapsed",
        )
        st.session_state[f"{key}_page"] = page

    with col4:
        if st.button("Next", key=f"{key}_next", use_container_width=True):
            current = st.session_state.get(f"{key}_page", 1)
            st.session_state[f"{key}_page"] = min(total_pages, current + 1)

    with col5:
        if st.button("Last", key=f"{key}_last", use_container_width=True):
            st.session_state[f"{key}_page"] = total_pages

    # Get current page from session state
    current_page = st.session_state.get(f"{key}_page", 1)

    # Calculate slice
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    # Show info
    st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} rows (Page {current_page} of {total_pages})")

    # Display page
    page_df = df.iloc[start_idx:end_idx]

    if column_config:
        filtered_config = {
            col: config
            for col, config in column_config.items()
            if col in page_df.columns
        }
        st.dataframe(
            page_df,
            column_config=filtered_config,
            hide_index=True,
            use_container_width=True,
        )
    else:
        styled_dataframe(page_df)

    # Page size selector
    col1, col2 = st.columns([3, 1])
    with col2:
        new_page_size = st.selectbox(
            "Rows per page",
            options=[10, 25, 50, 100],
            index=[10, 25, 50, 100].index(page_size) if page_size in [10, 25, 50, 100] else 1,
            key=f"{key}_page_size",
        )
        if new_page_size != page_size:
            st.session_state[f"{key}_page"] = 1
            st.rerun()

    # Export buttons
    if show_export:
        st.markdown("---")
        st.markdown("**Export Options**")
        col1, col2 = st.columns([1, 3])
        with col1:
            export_scope = st.radio(
                "Export scope",
                options=["Current page", "All data"],
                key=f"{key}_export_scope",
                horizontal=True,
            )

        export_df = page_df if export_scope == "Current page" else df
        create_export_buttons(
            export_df,
            filename="table_export",
            key_prefix=f"{key}_export",
        )


def sortable_table(
    df: pd.DataFrame,
    default_sort_col: Optional[str] = None,
    default_ascending: bool = False,
    page_size: int = 25,
    key: str = "sortable",
    show_export: bool = True,
) -> None:
    """
    Display a sortable, paginated table.

    Args:
        df: DataFrame to display
        default_sort_col: Default column to sort by
        default_ascending: Default sort direction
        page_size: Number of rows per page
        key: Unique key for the table
        show_export: Whether to show export buttons
    """
    if df.empty:
        st.info("No data available.")
        return

    # Sort controls
    col1, col2 = st.columns([3, 1])

    with col1:
        sort_col = st.selectbox(
            "Sort by",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_sort_col) if default_sort_col in df.columns else 0,
            key=f"{key}_sort_col",
        )

    with col2:
        ascending = st.checkbox(
            "Ascending",
            value=default_ascending,
            key=f"{key}_ascending",
        )

    # Apply sorting
    sorted_df = df.sort_values(by=sort_col, ascending=ascending)

    # Display paginated table
    paginated_table(
        sorted_df,
        page_size=page_size,
        key=key,
        show_export=show_export,
    )


def searchable_table(
    df: pd.DataFrame,
    search_columns: Optional[List[str]] = None,
    page_size: int = 25,
    key: str = "searchable",
    show_export: bool = True,
) -> None:
    """
    Display a searchable, paginated table.

    Args:
        df: DataFrame to display
        search_columns: Columns to search in (defaults to all string columns)
        page_size: Number of rows per page
        key: Unique key for the table
        show_export: Whether to show export buttons
    """
    if df.empty:
        st.info("No data available.")
        return

    # Determine searchable columns
    if search_columns is None:
        search_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # Search input
    search_query = st.text_input(
        "Search",
        placeholder="Type to search...",
        key=f"{key}_search",
    )

    # Apply search filter
    if search_query and search_columns:
        mask = pd.Series([False] * len(df))
        for col in search_columns:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(search_query, case=False, na=False)
        filtered_df = df[mask]
        st.caption(f"Found {len(filtered_df)} results for '{search_query}'")
    else:
        filtered_df = df

    # Display paginated table
    paginated_table(
        filtered_df,
        page_size=page_size,
        key=key,
        show_export=show_export,
    )
