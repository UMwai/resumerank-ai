"""
Alerts Page

Configurable alerts for high-confidence signals across all systems.
"""

from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from components.cards import metric_card, alert_card
from components.tables import styled_dataframe
from components.filters import score_filter, confidence_filter


# Session state keys for alert configuration
ALERT_CONFIG_KEY = "alert_config"
ALERT_HISTORY_KEY = "alert_history"


def get_alert_config() -> Dict[str, Any]:
    """Get the current alert configuration from session state."""
    if ALERT_CONFIG_KEY not in st.session_state:
        st.session_state[ALERT_CONFIG_KEY] = {
            "enabled": True,
            "bullish_threshold": 0.7,
            "bearish_threshold": 0.3,
            "min_confidence": 0.6,
            "clinical_trials": True,
            "patent_intelligence": True,
            "insider_hiring": True,
            "notification_email": "",
        }
    return st.session_state[ALERT_CONFIG_KEY]


def save_alert_config(config: Dict[str, Any]) -> None:
    """Save alert configuration to session state."""
    st.session_state[ALERT_CONFIG_KEY] = config


def get_alert_history() -> List[Dict[str, Any]]:
    """Get alert history from session state."""
    if ALERT_HISTORY_KEY not in st.session_state:
        st.session_state[ALERT_HISTORY_KEY] = []
    return st.session_state[ALERT_HISTORY_KEY]


def add_to_alert_history(alert: Dict[str, Any]) -> None:
    """Add an alert to history."""
    history = get_alert_history()
    alert["timestamp"] = datetime.now().isoformat()
    history.insert(0, alert)
    # Keep only last 100 alerts
    st.session_state[ALERT_HISTORY_KEY] = history[:100]


def render_alerts_page(
    combined_fetcher: Any,
    clinical_fetcher: Any,
    patent_fetcher: Any,
    insider_fetcher: Any,
) -> None:
    """
    Render the alerts page.

    Args:
        combined_fetcher: CombinedSignalFetcher instance
        clinical_fetcher: ClinicalTrialFetcher instance
        patent_fetcher: PatentIntelligenceFetcher instance
        insider_fetcher: InsiderHiringFetcher instance
    """
    st.title("Alerts")
    st.markdown("Configure and monitor high-confidence signal alerts")

    # Tab navigation
    tab1, tab2, tab3 = st.tabs([
        "Active Alerts",
        "Alert Configuration",
        "Alert History"
    ])

    with tab1:
        render_active_alerts_tab(combined_fetcher)

    with tab2:
        render_configuration_tab()

    with tab3:
        render_history_tab()


def render_active_alerts_tab(combined_fetcher: Any) -> None:
    """Render the active alerts tab."""
    st.subheader("Active Alerts")

    # Get current config
    config = get_alert_config()

    if not config["enabled"]:
        st.warning("Alerts are currently disabled. Enable them in the Configuration tab.")
        return

    # Get alerts based on configuration
    try:
        alerts = combined_fetcher.get_alerts(
            score_threshold=config["bullish_threshold"],
            confidence_threshold=config["min_confidence"]
        )
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        alerts = []

    # Filter by enabled systems
    filtered_alerts = []
    for alert in alerts:
        # For now, include all alerts since they come from combined signals
        filtered_alerts.append(alert)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Alerts", len(filtered_alerts))

    with col2:
        bullish_count = len([a for a in filtered_alerts if a.get('type') == 'BULLISH'])
        metric_card("Bullish", bullish_count, delta=f"+{bullish_count}" if bullish_count else None)

    with col3:
        bearish_count = len([a for a in filtered_alerts if a.get('type') == 'BEARISH'])
        metric_card("Bearish", bearish_count, delta=f"-{bearish_count}" if bearish_count else None, delta_color="inverse")

    with col4:
        high_conf = len([a for a in filtered_alerts if a.get('confidence', 0) >= 0.8])
        metric_card("High Confidence", high_conf)

    st.divider()

    # Display alerts
    if filtered_alerts:
        # Separate bullish and bearish
        bullish_alerts = [a for a in filtered_alerts if a.get('type') == 'BULLISH']
        bearish_alerts = [a for a in filtered_alerts if a.get('type') == 'BEARISH']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Bullish Alerts")
            if bullish_alerts:
                for alert in bullish_alerts:
                    alert_card(
                        alert_type=alert.get('type', 'INFO'),
                        ticker=alert.get('ticker', 'N/A'),
                        company_name=alert.get('company_name', 'N/A'),
                        message=alert.get('message', ''),
                        score=alert.get('combined_score', 0),
                        confidence=alert.get('confidence', 0),
                    )
                    # Add to history
                    add_to_alert_history(alert)
            else:
                st.info("No bullish alerts at this time.")

        with col2:
            st.markdown("### Bearish Alerts")
            if bearish_alerts:
                for alert in bearish_alerts:
                    alert_card(
                        alert_type=alert.get('type', 'INFO'),
                        ticker=alert.get('ticker', 'N/A'),
                        company_name=alert.get('company_name', 'N/A'),
                        message=alert.get('message', ''),
                        score=alert.get('combined_score', 0),
                        confidence=alert.get('confidence', 0),
                    )
                    add_to_alert_history(alert)
            else:
                st.info("No bearish alerts at this time.")
    else:
        st.info("No alerts triggered based on current configuration.")

    st.divider()

    # Alert criteria summary
    st.markdown("### Alert Criteria")
    st.markdown(f"""
    **Current thresholds:**
    - Bullish signal threshold: **{config['bullish_threshold']:.0%}**
    - Bearish signal threshold: **{config['bearish_threshold']:.0%}**
    - Minimum confidence: **{config['min_confidence']:.0%}**

    **Data sources:**
    - Clinical Trials: {'Enabled' if config['clinical_trials'] else 'Disabled'}
    - Patent Intelligence: {'Enabled' if config['patent_intelligence'] else 'Disabled'}
    - Insider/Hiring: {'Enabled' if config['insider_hiring'] else 'Disabled'}
    """)


def render_configuration_tab() -> None:
    """Render the alert configuration tab."""
    st.subheader("Alert Configuration")

    # Get current config
    config = get_alert_config()

    # Main enable/disable toggle
    st.markdown("### General Settings")

    enabled = st.toggle(
        "Enable Alerts",
        value=config["enabled"],
        key="config_enabled"
    )

    st.divider()

    # Threshold settings
    st.markdown("### Signal Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        bullish_threshold = st.slider(
            "Bullish Signal Threshold",
            min_value=0.5,
            max_value=1.0,
            value=config["bullish_threshold"],
            step=0.05,
            format="%.0f%%",
            help="Minimum combined score to trigger a bullish alert",
            key="config_bullish"
        )

    with col2:
        bearish_threshold = st.slider(
            "Bearish Signal Threshold",
            min_value=0.0,
            max_value=0.5,
            value=config["bearish_threshold"],
            step=0.05,
            format="%.0f%%",
            help="Maximum combined score to trigger a bearish alert",
            key="config_bearish"
        )

    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.3,
        max_value=1.0,
        value=config["min_confidence"],
        step=0.05,
        format="%.0f%%",
        help="Minimum confidence level required for alerts",
        key="config_confidence"
    )

    st.divider()

    # Data source settings
    st.markdown("### Data Sources")

    col1, col2, col3 = st.columns(3)

    with col1:
        clinical_trials = st.checkbox(
            "Clinical Trials",
            value=config["clinical_trials"],
            key="config_clinical"
        )

    with col2:
        patent_intelligence = st.checkbox(
            "Patent Intelligence",
            value=config["patent_intelligence"],
            key="config_patent"
        )

    with col3:
        insider_hiring = st.checkbox(
            "Insider/Hiring",
            value=config["insider_hiring"],
            key="config_insider"
        )

    st.divider()

    # Notification settings
    st.markdown("### Notifications")

    notification_email = st.text_input(
        "Notification Email (optional)",
        value=config["notification_email"],
        placeholder="your@email.com",
        help="Email address for alert notifications (not yet implemented)",
        key="config_email"
    )

    st.info("Email notifications are not yet implemented. Alerts are displayed on this page only.")

    st.divider()

    # Save configuration
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("Save Configuration", type="primary", use_container_width=True):
            new_config = {
                "enabled": enabled,
                "bullish_threshold": bullish_threshold,
                "bearish_threshold": bearish_threshold,
                "min_confidence": min_confidence,
                "clinical_trials": clinical_trials,
                "patent_intelligence": patent_intelligence,
                "insider_hiring": insider_hiring,
                "notification_email": notification_email,
            }
            save_alert_config(new_config)
            st.success("Configuration saved!")

    with col2:
        if st.button("Reset to Defaults", type="secondary", use_container_width=True):
            default_config = {
                "enabled": True,
                "bullish_threshold": 0.7,
                "bearish_threshold": 0.3,
                "min_confidence": 0.6,
                "clinical_trials": True,
                "patent_intelligence": True,
                "insider_hiring": True,
                "notification_email": "",
            }
            save_alert_config(default_config)
            st.success("Configuration reset to defaults!")
            st.rerun()


def render_history_tab() -> None:
    """Render the alert history tab."""
    st.subheader("Alert History")

    history = get_alert_history()

    if not history:
        st.info("No alert history available. Alerts will be recorded as they are triggered.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Total Recorded", len(history))

    with col2:
        bullish_count = len([a for a in history if a.get('type') == 'BULLISH'])
        metric_card("Bullish Alerts", bullish_count)

    with col3:
        bearish_count = len([a for a in history if a.get('type') == 'BEARISH'])
        metric_card("Bearish Alerts", bearish_count)

    st.divider()

    # Convert to DataFrame for display
    history_df = pd.DataFrame(history)

    if not history_df.empty:
        # Select and order columns
        display_cols = ['timestamp', 'type', 'ticker', 'company_name', 'combined_score', 'confidence', 'message']
        display_cols = [c for c in display_cols if c in history_df.columns]

        # Format timestamp
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

        column_config = {
            'timestamp': st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
            'type': st.column_config.TextColumn("Type"),
            'ticker': st.column_config.TextColumn("Ticker"),
            'company_name': st.column_config.TextColumn("Company"),
            'combined_score': st.column_config.NumberColumn("Score", format="%.2f"),
            'confidence': st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.0%%"),
            'message': st.column_config.TextColumn("Message"),
        }

        styled_dataframe(
            history_df[display_cols] if display_cols else history_df,
            column_config={k: v for k, v in column_config.items() if k in display_cols},
            height=500
        )

    st.divider()

    # Clear history button
    if st.button("Clear Alert History", type="secondary"):
        st.session_state[ALERT_HISTORY_KEY] = []
        st.success("Alert history cleared!")
        st.rerun()


def render_alerts_demo() -> None:
    """Render alerts page with demo data."""
    st.title("Alerts")
    st.warning("Running in demo mode - displaying sample data")

    # Demo alerts
    demo_alerts = [
        {
            "type": "BULLISH",
            "ticker": "MRNA",
            "company_name": "Moderna Inc.",
            "combined_score": 0.82,
            "confidence": 0.85,
            "message": "High confidence bullish signal for MRNA",
        },
        {
            "type": "BULLISH",
            "ticker": "VRTX",
            "company_name": "Vertex Pharmaceuticals",
            "combined_score": 0.75,
            "confidence": 0.72,
            "message": "High confidence bullish signal for VRTX",
        },
        {
            "type": "BEARISH",
            "ticker": "GILD",
            "company_name": "Gilead Sciences",
            "combined_score": 0.28,
            "confidence": 0.71,
            "message": "High confidence bearish signal for GILD",
        },
    ]

    st.subheader("Active Alerts (Demo)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Alerts", 3)
    with col2:
        metric_card("Bullish", 2, delta="+2")
    with col3:
        metric_card("Bearish", 1, delta="-1", delta_color="inverse")
    with col4:
        metric_card("High Confidence", 3)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Bullish Alerts")
        for alert in [a for a in demo_alerts if a['type'] == 'BULLISH']:
            alert_card(
                alert_type=alert['type'],
                ticker=alert['ticker'],
                company_name=alert['company_name'],
                message=alert['message'],
                score=alert['combined_score'],
                confidence=alert['confidence'],
            )

    with col2:
        st.markdown("### Bearish Alerts")
        for alert in [a for a in demo_alerts if a['type'] == 'BEARISH']:
            alert_card(
                alert_type=alert['type'],
                ticker=alert['ticker'],
                company_name=alert['company_name'],
                message=alert['message'],
                score=alert['combined_score'],
                confidence=alert['confidence'],
            )
