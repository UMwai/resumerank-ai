"""
Card components for displaying metrics and signals.
"""

from typing import Any, Dict, Optional

import streamlit as st


def metric_card(
    title: str,
    value: Any,
    delta: Optional[Any] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
) -> None:
    """
    Display a metric card with optional delta indicator.

    Args:
        title: Card title
        value: Main value to display
        delta: Optional change value
        delta_color: Color scheme for delta ("normal", "inverse", "off")
        help_text: Optional tooltip text
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def signal_card(
    signal_type: str,
    ticker: str,
    company_name: str,
    score: float,
    description: str,
    recommendation: str,
    source: str = "",
) -> None:
    """
    Display a signal card with color-coded recommendation.

    Args:
        signal_type: Type of signal (clinical_trial, patent_cliff, insider_hiring)
        ticker: Company ticker symbol
        company_name: Company name
        score: Signal score (0-1)
        description: Signal description
        recommendation: Recommendation text
        source: Source system
    """
    # Determine colors based on recommendation
    if recommendation in ["STRONG BUY", "STRONG_BUY", "BUY"]:
        border_color = "#10b981"  # Green
        bg_color = "#ecfdf5"
    elif recommendation in ["STRONG SELL", "STRONG_SHORT", "SHORT", "SELL"]:
        border_color = "#ef4444"  # Red
        bg_color = "#fef2f2"
    else:
        border_color = "#6b7280"  # Gray
        bg_color = "#f9fafb"

    # Signal type icons
    icons = {
        "clinical_trial": "pill",
        "patent_cliff": "shield",
        "insider_hiring": "person-walking",
    }
    icon = icons.get(signal_type, "chart-line")

    st.markdown(
        f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            background-color: {bg_color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: bold; font-size: 18px;">{ticker}</span>
                <span style="
                    background-color: {border_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                ">{recommendation}</span>
            </div>
            <div style="color: #6b7280; font-size: 14px; margin-bottom: 8px;">{company_name}</div>
            <div style="font-size: 14px; margin-bottom: 8px;">{description}</div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12px; color: #6b7280;">{source}</span>
                <span style="font-weight: bold;">Score: {score:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def opportunity_card(
    ticker: str,
    company_name: str,
    combined_score: float,
    confidence: float,
    recommendation: str,
    clinical_score: Optional[float] = None,
    patent_score: Optional[float] = None,
    insider_score: Optional[float] = None,
) -> None:
    """
    Display a combined opportunity card with scores from all systems.

    Args:
        ticker: Company ticker symbol
        company_name: Company name
        combined_score: Combined weighted score
        confidence: Confidence level
        recommendation: Recommendation text
        clinical_score: Clinical trial score (optional)
        patent_score: Patent intelligence score (optional)
        insider_score: Insider/hiring score (optional)
    """
    # Determine colors based on score
    if combined_score >= 0.7:
        border_color = "#10b981"
        bg_color = "#ecfdf5"
        score_color = "#059669"
    elif combined_score <= 0.3:
        border_color = "#ef4444"
        bg_color = "#fef2f2"
        score_color = "#dc2626"
    else:
        border_color = "#f59e0b"
        bg_color = "#fffbeb"
        score_color = "#d97706"

    # Build score bars
    def score_bar(label: str, value: Optional[float]) -> str:
        if value is None:
            return f'<div style="font-size: 12px; color: #9ca3af;">{label}: N/A</div>'
        width = int(value * 100)
        bar_color = "#10b981" if value >= 0.5 else "#ef4444"
        return f'''
            <div style="margin-bottom: 4px;">
                <div style="font-size: 12px; color: #6b7280;">{label}</div>
                <div style="background-color: #e5e7eb; border-radius: 4px; height: 8px;">
                    <div style="background-color: {bar_color}; width: {width}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
        '''

    st.markdown(
        f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            background-color: {bg_color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <span style="font-weight: bold; font-size: 24px; color: #1f2937;">{ticker}</span>
                    <div style="color: #6b7280; font-size: 14px;">{company_name}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 32px; font-weight: bold; color: {score_color};">{combined_score:.2f}</div>
                    <div style="font-size: 12px; color: #6b7280;">Confidence: {confidence:.0%}</div>
                </div>
            </div>
            <div style="
                background-color: {border_color};
                color: white;
                padding: 6px 16px;
                border-radius: 16px;
                font-size: 14px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 16px;
            ">{recommendation}</div>
            <div style="margin-top: 12px;">
                {score_bar('Clinical Trials', clinical_score)}
                {score_bar('Patent Intel', patent_score)}
                {score_bar('Insider/Hiring', insider_score)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_card(
    alert_type: str,
    ticker: str,
    company_name: str,
    message: str,
    score: float,
    confidence: float,
) -> None:
    """
    Display an alert card.

    Args:
        alert_type: Type of alert (BULLISH, BEARISH, INFO)
        ticker: Company ticker symbol
        company_name: Company name
        message: Alert message
        score: Alert score
        confidence: Confidence level
    """
    colors = {
        "BULLISH": {"border": "#10b981", "bg": "#ecfdf5", "icon": "arrow-up"},
        "BEARISH": {"border": "#ef4444", "bg": "#fef2f2", "icon": "arrow-down"},
        "INFO": {"border": "#3b82f6", "bg": "#eff6ff", "icon": "info-circle"},
    }
    style = colors.get(alert_type, colors["INFO"])

    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {style['border']};
            border-radius: 4px;
            padding: 12px 16px;
            margin-bottom: 8px;
            background-color: {style['bg']};
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; font-size: 16px;">{ticker}</span>
                    <span style="color: #6b7280; margin-left: 8px;">{company_name}</span>
                </div>
                <span style="
                    background-color: {style['border']};
                    color: white;
                    padding: 2px 8px;
                    border-radius: 8px;
                    font-size: 11px;
                    font-weight: bold;
                ">{alert_type}</span>
            </div>
            <div style="margin-top: 8px; font-size: 14px;">{message}</div>
            <div style="margin-top: 4px; font-size: 12px; color: #6b7280;">
                Score: {score:.2f} | Confidence: {confidence:.0%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """
    Return HTML for a status badge.

    Args:
        status: Status text

    Returns:
        HTML string for the badge
    """
    status_colors = {
        "ACTIVE": "#10b981",
        "RECRUITING": "#3b82f6",
        "COMPLETED": "#6b7280",
        "TERMINATED": "#ef4444",
        "SUSPENDED": "#f59e0b",
        "WITHDRAWN": "#9ca3af",
        "APPROVED": "#10b981",
        "PENDING": "#f59e0b",
    }

    color = status_colors.get(status.upper(), "#6b7280")

    return f'''
        <span style="
            background-color: {color};
            color: white;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: bold;
        ">{status}</span>
    '''
