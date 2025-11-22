"""Reusable UI components for the Investment Intelligence Dashboard."""

from .cards import (
    metric_card,
    signal_card,
    opportunity_card,
    alert_card,
)
from .tables import (
    styled_dataframe,
    signal_table,
    opportunity_table,
)
from .charts import (
    score_gauge,
    timeline_chart,
    bar_chart,
    pie_chart,
    calendar_heatmap,
)
from .filters import (
    ticker_filter,
    date_range_filter,
    score_filter,
    recommendation_filter,
)

__all__ = [
    # Cards
    "metric_card",
    "signal_card",
    "opportunity_card",
    "alert_card",
    # Tables
    "styled_dataframe",
    "signal_table",
    "opportunity_table",
    # Charts
    "score_gauge",
    "timeline_chart",
    "bar_chart",
    "pie_chart",
    "calendar_heatmap",
    # Filters
    "ticker_filter",
    "date_range_filter",
    "score_filter",
    "recommendation_filter",
]
