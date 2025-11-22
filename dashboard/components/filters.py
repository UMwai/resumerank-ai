"""
Filter components for data selection and filtering.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


def ticker_filter(
    available_tickers: List[str],
    key: str = "ticker_filter",
    default: Optional[List[str]] = None,
    label: str = "Select Tickers",
    max_selections: Optional[int] = None,
) -> List[str]:
    """
    Ticker multi-select filter.

    Args:
        available_tickers: List of available ticker symbols
        key: Unique key for the widget
        default: Default selected tickers
        label: Filter label
        max_selections: Maximum number of selections allowed

    Returns:
        List of selected tickers
    """
    if not available_tickers:
        st.info("No tickers available.")
        return []

    # Sort tickers alphabetically
    sorted_tickers = sorted(available_tickers)

    selected = st.multiselect(
        label,
        options=sorted_tickers,
        default=default or [],
        key=key,
        max_selections=max_selections,
    )

    return selected


def date_range_filter(
    key: str = "date_range",
    default_days: int = 30,
    label: str = "Date Range",
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
) -> Tuple[date, date]:
    """
    Date range filter with presets.

    Args:
        key: Unique key for the widget
        default_days: Default number of days to look back
        label: Filter label
        min_date: Minimum selectable date
        max_date: Maximum selectable date

    Returns:
        Tuple of (start_date, end_date)
    """
    today = date.today()
    default_start = today - timedelta(days=default_days)

    # Quick select presets
    col1, col2 = st.columns([1, 2])

    with col1:
        preset = st.selectbox(
            "Quick Select",
            options=["Custom", "7 Days", "30 Days", "90 Days", "YTD", "1 Year"],
            key=f"{key}_preset",
        )

    # Calculate dates based on preset
    if preset == "7 Days":
        start_date = today - timedelta(days=7)
        end_date = today
    elif preset == "30 Days":
        start_date = today - timedelta(days=30)
        end_date = today
    elif preset == "90 Days":
        start_date = today - timedelta(days=90)
        end_date = today
    elif preset == "YTD":
        start_date = date(today.year, 1, 1)
        end_date = today
    elif preset == "1 Year":
        start_date = today - timedelta(days=365)
        end_date = today
    else:
        # Custom - use date inputs
        with col2:
            dates = st.date_input(
                label,
                value=(default_start, today),
                min_value=min_date,
                max_value=max_date,
                key=f"{key}_custom",
            )
            if isinstance(dates, tuple) and len(dates) == 2:
                start_date, end_date = dates
            else:
                start_date = dates if isinstance(dates, date) else default_start
                end_date = today

    return start_date, end_date


def score_filter(
    key: str = "score_filter",
    min_score: float = 0,
    max_score: float = 1,
    default_range: Tuple[float, float] = (0, 1),
    label: str = "Score Range",
    step: float = 0.1,
) -> Tuple[float, float]:
    """
    Score range slider filter.

    Args:
        key: Unique key for the widget
        min_score: Minimum possible score
        max_score: Maximum possible score
        default_range: Default (min, max) range
        label: Filter label
        step: Slider step size

    Returns:
        Tuple of (min_selected, max_selected)
    """
    values = st.slider(
        label,
        min_value=min_score,
        max_value=max_score,
        value=default_range,
        step=step,
        key=key,
    )

    return values


def recommendation_filter(
    key: str = "recommendation_filter",
    label: str = "Recommendations",
    include_all: bool = True,
) -> List[str]:
    """
    Recommendation type filter.

    Args:
        key: Unique key for the widget
        label: Filter label
        include_all: Whether to include "All" option

    Returns:
        List of selected recommendations
    """
    options = [
        "STRONG BUY",
        "BUY",
        "HOLD",
        "NEUTRAL",
        "SELL",
        "STRONG SELL",
    ]

    if include_all:
        all_selected = st.checkbox("Select All", value=True, key=f"{key}_all")
        if all_selected:
            return options

    selected = st.multiselect(
        label,
        options=options,
        default=options[:2],  # Default to bullish recommendations
        key=key,
    )

    return selected


def signal_type_filter(
    key: str = "signal_type_filter",
    label: str = "Signal Types",
    system: str = "all",
) -> List[str]:
    """
    Signal type multi-select filter.

    Args:
        key: Unique key for the widget
        label: Filter label
        system: Which system's signal types to show

    Returns:
        List of selected signal types
    """
    signal_types = {
        "clinical_trials": [
            "ENROLLMENT_SURGE",
            "DATE_CHANGE",
            "STATUS_CHANGE",
            "ENDPOINT_MODIFICATION",
            "SEC_FILING",
        ],
        "patent_intelligence": [
            "PATENT_EXPIRING",
            "ANDA_APPROVED",
            "LITIGATION_RESOLVED",
            "GENERIC_ENTRY",
        ],
        "insider_hiring": [
            "CEO_PURCHASE",
            "CFO_PURCHASE",
            "MULTIPLE_INSIDER_BUY",
            "FUND_NEW_POSITION",
            "COMMERCIAL_BUILDUP",
        ],
    }

    if system == "all":
        options = []
        for types in signal_types.values():
            options.extend(types)
    else:
        options = signal_types.get(system, [])

    selected = st.multiselect(
        label,
        options=options,
        key=key,
    )

    return selected


def phase_filter(
    key: str = "phase_filter",
    label: str = "Clinical Phase",
) -> List[str]:
    """
    Clinical trial phase filter.

    Args:
        key: Unique key for the widget
        label: Filter label

    Returns:
        List of selected phases
    """
    phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]

    selected = st.multiselect(
        label,
        options=phases,
        default=["Phase 2", "Phase 3"],  # Most actionable phases
        key=key,
    )

    return selected


def market_cap_filter(
    key: str = "market_cap_filter",
    label: str = "Market Cap",
) -> str:
    """
    Market cap category filter.

    Args:
        key: Unique key for the widget
        label: Filter label

    Returns:
        Selected market cap category
    """
    options = {
        "All": (0, float("inf")),
        "Micro (<$300M)": (0, 300_000_000),
        "Small ($300M-$2B)": (300_000_000, 2_000_000_000),
        "Mid ($2B-$10B)": (2_000_000_000, 10_000_000_000),
        "Large (>$10B)": (10_000_000_000, float("inf")),
    }

    selected = st.selectbox(
        label,
        options=list(options.keys()),
        key=key,
    )

    return selected


def confidence_filter(
    key: str = "confidence_filter",
    label: str = "Minimum Confidence",
    default: float = 0.5,
) -> float:
    """
    Confidence threshold slider.

    Args:
        key: Unique key for the widget
        label: Filter label
        default: Default confidence threshold

    Returns:
        Selected minimum confidence
    """
    confidence = st.slider(
        label,
        min_value=0.0,
        max_value=1.0,
        value=default,
        step=0.1,
        format="%.0f%%",
        key=key,
    )

    return confidence


def apply_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Apply multiple filters to a DataFrame.

    Args:
        df: DataFrame to filter
        filters: Dictionary of filter names to values
        column_mapping: Optional mapping of filter names to column names

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    if column_mapping is None:
        column_mapping = {}

    filtered = df.copy()

    for filter_name, filter_value in filters.items():
        col_name = column_mapping.get(filter_name, filter_name)

        if col_name not in filtered.columns:
            continue

        if isinstance(filter_value, list) and filter_value:
            filtered = filtered[filtered[col_name].isin(filter_value)]
        elif isinstance(filter_value, tuple) and len(filter_value) == 2:
            # Range filter
            min_val, max_val = filter_value
            filtered = filtered[
                (filtered[col_name] >= min_val) &
                (filtered[col_name] <= max_val)
            ]
        elif filter_value is not None:
            filtered = filtered[filtered[col_name] == filter_value]

    return filtered


def search_filter(
    key: str = "search",
    label: str = "Search",
    placeholder: str = "Type to search...",
) -> str:
    """
    Text search input.

    Args:
        key: Unique key for the widget
        label: Input label
        placeholder: Placeholder text

    Returns:
        Search query string
    """
    return st.text_input(
        label,
        placeholder=placeholder,
        key=key,
    )


def sort_options(
    columns: List[str],
    key: str = "sort",
    default_col: Optional[str] = None,
    default_ascending: bool = False,
) -> Tuple[str, bool]:
    """
    Sort column and direction selector.

    Args:
        columns: List of sortable columns
        key: Unique key for the widget
        default_col: Default sort column
        default_ascending: Default sort direction

    Returns:
        Tuple of (column_name, ascending)
    """
    col1, col2 = st.columns(2)

    with col1:
        sort_col = st.selectbox(
            "Sort By",
            options=columns,
            index=columns.index(default_col) if default_col in columns else 0,
            key=f"{key}_col",
        )

    with col2:
        direction = st.radio(
            "Direction",
            options=["Descending", "Ascending"],
            index=0 if not default_ascending else 1,
            horizontal=True,
            key=f"{key}_dir",
        )

    ascending = direction == "Ascending"

    return sort_col, ascending
