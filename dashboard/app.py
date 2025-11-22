"""
Investment Intelligence Dashboard

A Streamlit application that combines signals from three intelligence systems:
1. Clinical Trial Signal Detection
2. Patent/IP Intelligence
3. Insider Activity + Hiring Signals

Run with: streamlit run app.py
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

# Add dashboard directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.database import DatabaseManager, MockDatabaseManager
from utils.data_fetchers import (
    ClinicalTrialFetcher,
    PatentIntelligenceFetcher,
    InsiderHiringFetcher,
    CombinedSignalFetcher,
)
from pages.home import render_home_page, render_home_demo
from pages.clinical_trials import render_clinical_trials_page, render_clinical_trials_demo
from pages.patent_cliff import render_patent_cliff_page, render_patent_cliff_demo
from pages.insider_hiring import render_insider_hiring_page, render_insider_hiring_demo
from pages.watchlist import render_watchlist_page, render_watchlist_demo
from pages.alerts import render_alerts_page, render_alerts_demo
from pages.analytics import render_analytics_page, render_analytics_demo

# Phase 2 imports
from utils.websocket_client import (
    SignalAggregator,
    MockWebSocketClient,
    create_signal_aggregator,
    Signal,
)
from components.live_feed import (
    render_live_badge,
    render_live_feed_sidebar,
    render_signal_summary,
    add_signal_to_queue,
    get_signal_queue,
)
from utils.watchlist_manager import get_watchlist_manager
from utils.alert_engine import get_alert_engine
from utils.accuracy_tracker import get_accuracy_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Investment Intelligence Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/UMwai/investment-dashboard",
        "Report a bug": "https://github.com/UMwai/investment-dashboard/issues",
        "About": "Investment Intelligence Dashboard - Combining Clinical Trial, Patent, and Insider Signals"
    }
)


def get_theme_css(dark_mode: bool = False) -> str:
    """Generate CSS based on theme mode."""
    if dark_mode:
        return """
        <style>
            /* Dark Mode Styling */
            :root {
                --bg-primary: #1a1a2e;
                --bg-secondary: #16213e;
                --text-primary: #e5e7eb;
                --text-secondary: #9ca3af;
                --border-color: #374151;
                --accent-color: #6366f1;
            }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            [data-testid="stSidebar"] {
                background-color: var(--bg-secondary);
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 28px;
                color: var(--text-primary);
            }

            .stMarkdown {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 8px 16px;
                border-radius: 4px;
                background-color: var(--bg-secondary);
            }

            thead tr th {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
            }

            .stSuccess, .stError, .stWarning, .stInfo {
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
            }

            /* Dark mode specific overrides */
            .stDataFrame {
                background-color: var(--bg-secondary);
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light Mode Styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            [data-testid="stSidebar"] {
                background-color: #f8fafc;
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }

            [data-testid="stMetricValue"] {
                font-size: 28px;
            }

            .stMarkdown {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 8px 16px;
                border-radius: 4px;
            }

            thead tr th {
                background-color: #f1f5f9 !important;
            }

            .stSuccess, .stError, .stWarning, .stInfo {
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
            }
        </style>
        """


def load_config() -> dict:
    """
    Load configuration from config.yaml file.

    Returns:
        Configuration dictionary
    """
    config_paths = [
        Path(__file__).parent / "config.yaml",
        Path(__file__).parent / "config.yml",
        Path(__file__).parent.parent / "config.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

    # Return default config if no file found
    return get_default_config()


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        "demo_mode": True,
        "auto_refresh": {
            "enabled": False,
            "interval_seconds": 300,  # 5 minutes
        },
        "databases": {
            "clinical_trials": {
                "host": os.getenv("CT_DB_HOST", "localhost"),
                "port": int(os.getenv("CT_DB_PORT", 5432)),
                "database": os.getenv("CT_DB_NAME", "clinical_trials"),
                "user": os.getenv("CT_DB_USER", "postgres"),
                "password": os.getenv("CT_DB_PASSWORD", ""),
            },
            "patent_intelligence": {
                "host": os.getenv("PI_DB_HOST", "localhost"),
                "port": int(os.getenv("PI_DB_PORT", 5432)),
                "database": os.getenv("PI_DB_NAME", "patent_intelligence"),
                "user": os.getenv("PI_DB_USER", "postgres"),
                "password": os.getenv("PI_DB_PASSWORD", ""),
            },
            "insider_hiring": {
                "host": os.getenv("IH_DB_HOST", "localhost"),
                "port": int(os.getenv("IH_DB_PORT", 5432)),
                "database": os.getenv("IH_DB_NAME", "insider_signals"),
                "user": os.getenv("IH_DB_USER", "postgres"),
                "password": os.getenv("IH_DB_PASSWORD", ""),
            },
        },
    }


@st.cache_resource
def initialize_database_manager(config: dict) -> DatabaseManager:
    """
    Initialize and cache the database manager.

    Args:
        config: Database configuration

    Returns:
        DatabaseManager instance
    """
    if config.get("demo_mode", False):
        logger.info("Running in demo mode - using mock database")
        return MockDatabaseManager(config.get("databases", {}))

    db_manager = DatabaseManager(config.get("databases", {}))
    db_manager.initialize()
    return db_manager


@st.cache_resource
def initialize_fetchers(_db_manager) -> tuple:
    """
    Initialize and cache data fetchers.

    Args:
        _db_manager: DatabaseManager instance

    Returns:
        Tuple of (clinical_fetcher, patent_fetcher, insider_fetcher, combined_fetcher)
    """
    clinical_fetcher = ClinicalTrialFetcher(_db_manager)
    patent_fetcher = PatentIntelligenceFetcher(_db_manager)
    insider_fetcher = InsiderHiringFetcher(_db_manager)
    combined_fetcher = CombinedSignalFetcher(
        clinical_fetcher,
        patent_fetcher,
        insider_fetcher,
    )

    return clinical_fetcher, patent_fetcher, insider_fetcher, combined_fetcher


def render_sidebar() -> str:
    """
    Render the sidebar navigation with dark mode toggle and refresh controls.

    Returns:
        Selected page name
    """
    with st.sidebar:
        st.markdown("## Investment Intelligence")
        st.markdown("---")

        # Dark Mode Toggle
        dark_mode = st.toggle(
            "Dark Mode",
            value=st.session_state.get("dark_mode", False),
            key="dark_mode_toggle",
            help="Toggle dark mode theme"
        )
        st.session_state["dark_mode"] = dark_mode

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            options=[
                "Home",
                "Clinical Trials",
                "Patent Cliff",
                "Insider/Hiring",
                "Watchlist",
                "Alerts",
                "Analytics",
            ],
            key="navigation",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Auto-refresh settings
        st.markdown("### Auto-Refresh")

        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.get("auto_refresh_enabled", False),
            key="auto_refresh_checkbox",
            help="Automatically refresh data at regular intervals"
        )
        st.session_state["auto_refresh_enabled"] = auto_refresh

        if auto_refresh:
            refresh_interval = st.select_slider(
                "Refresh Interval",
                options=[60, 120, 300, 600, 900],
                value=st.session_state.get("refresh_interval", 300),
                format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x} sec",
                key="refresh_interval_slider",
            )
            st.session_state["refresh_interval"] = refresh_interval

            # Display countdown
            if "last_refresh" in st.session_state:
                elapsed = time.time() - st.session_state["last_refresh"]
                remaining = max(0, refresh_interval - elapsed)
                st.caption(f"Next refresh in: {int(remaining)}s")

        st.markdown("---")

        # Live signal status
        st.markdown("### Live Signals")

        # Show LIVE badge
        aggregator = st.session_state.get("signal_aggregator")
        if aggregator:
            is_connected = aggregator.is_any_connected()
            render_live_badge(is_connected)

            # Show recent signals in sidebar
            signals = aggregator.get_recent_signals(limit=5, hours=24)
            if signals:
                render_live_feed_sidebar(signals, max_signals=3)

        st.markdown("---")

        # Connection status
        st.markdown("### Data Sources")

        config = st.session_state.get("config", {})
        demo_mode = config.get("demo_mode", True)

        if demo_mode:
            st.info("Running in Demo Mode")
            st.caption("Configure database connections in config.yaml to use live data")
        else:
            db_manager = st.session_state.get("db_manager")
            if db_manager:
                status = db_manager.get_connection_status()
                for db_name, connected in status.items():
                    icon = "white_check_mark" if connected else "x"
                    st.markdown(f":{icon}: {db_name.replace('_', ' ').title()}")

        st.markdown("---")

        # Quick actions
        st.markdown("### Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True, help="Refresh all data"):
                st.cache_data.clear()
                st.session_state["last_refresh"] = time.time()
                st.rerun()

        with col2:
            if st.button("Clear Cache", use_container_width=True, help="Clear all cached data"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        st.markdown("---")

        # Info
        st.caption("v2.0.0 - Phase 2")
        st.caption("Last updated: " + st.session_state.get("last_update", "N/A"))

    return page


def check_auto_refresh():
    """Check if auto-refresh should trigger and handle it."""
    if not st.session_state.get("auto_refresh_enabled", False):
        return

    refresh_interval = st.session_state.get("refresh_interval", 300)
    last_refresh = st.session_state.get("last_refresh", 0)

    if time.time() - last_refresh >= refresh_interval:
        st.session_state["last_refresh"] = time.time()
        st.cache_data.clear()
        st.rerun()


@st.cache_resource
def initialize_signal_aggregator(demo_mode: bool = True) -> SignalAggregator:
    """
    Initialize and cache the signal aggregator.

    Args:
        demo_mode: Use mock WebSocket clients

    Returns:
        SignalAggregator instance
    """
    aggregator = create_signal_aggregator({}, demo_mode=demo_mode)
    aggregator.start_all()
    return aggregator


def main():
    """Main application entry point."""
    # Initialize session state
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()

    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False

    # Load configuration
    config = load_config()
    st.session_state["config"] = config

    # Apply theme CSS
    dark_mode = st.session_state.get("dark_mode", False)
    st.markdown(get_theme_css(dark_mode), unsafe_allow_html=True)

    # Initialize database manager with spinner
    with st.spinner("Connecting to databases..."):
        db_manager = initialize_database_manager(config)
        st.session_state["db_manager"] = db_manager

    # Initialize fetchers with spinner
    with st.spinner("Initializing data fetchers..."):
        clinical_fetcher, patent_fetcher, insider_fetcher, combined_fetcher = initialize_fetchers(db_manager)

    # Initialize Phase 2 components
    demo_mode = config.get("demo_mode", True)

    # Initialize signal aggregator (WebSocket)
    if "signal_aggregator" not in st.session_state:
        aggregator = initialize_signal_aggregator(demo_mode=demo_mode)
        st.session_state["signal_aggregator"] = aggregator

    # Initialize watchlist manager
    if "watchlist_manager" not in st.session_state:
        st.session_state["watchlist_manager"] = get_watchlist_manager()

    # Initialize alert engine
    if "alert_engine" not in st.session_state:
        st.session_state["alert_engine"] = get_alert_engine()

    # Initialize accuracy tracker
    if "accuracy_tracker" not in st.session_state:
        st.session_state["accuracy_tracker"] = get_accuracy_tracker()

    # Track last update time
    st.session_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Render sidebar and get selected page
    page = render_sidebar()

    # Check for auto-refresh
    check_auto_refresh()

    # Check if in demo mode
    demo_mode = config.get("demo_mode", True)

    # Pass dark_mode to session state for charts
    st.session_state["chart_dark_mode"] = dark_mode

    # Render selected page with loading spinner
    if page == "Home":
        with st.spinner("Loading dashboard..."):
            if demo_mode:
                render_home_demo()
            else:
                render_home_page(
                    combined_fetcher,
                    clinical_fetcher,
                    patent_fetcher,
                    insider_fetcher,
                )

    elif page == "Clinical Trials":
        with st.spinner("Loading clinical trials data..."):
            if demo_mode:
                render_clinical_trials_demo()
            else:
                render_clinical_trials_page(clinical_fetcher)

    elif page == "Patent Cliff":
        with st.spinner("Loading patent intelligence data..."):
            if demo_mode:
                render_patent_cliff_demo()
            else:
                render_patent_cliff_page(patent_fetcher)

    elif page == "Insider/Hiring":
        with st.spinner("Loading insider/hiring signals..."):
            if demo_mode:
                render_insider_hiring_demo()
            else:
                render_insider_hiring_page(insider_fetcher)

    elif page == "Watchlist":
        with st.spinner("Loading watchlist..."):
            if demo_mode:
                render_watchlist_demo()
            else:
                render_watchlist_page(
                    combined_fetcher,
                    clinical_fetcher,
                    patent_fetcher,
                    insider_fetcher,
                )

    elif page == "Alerts":
        with st.spinner("Loading alerts..."):
            if demo_mode:
                render_alerts_demo()
            else:
                render_alerts_page(
                    combined_fetcher,
                    clinical_fetcher,
                    patent_fetcher,
                    insider_fetcher,
                )

    elif page == "Analytics":
        with st.spinner("Loading analytics..."):
            if demo_mode:
                render_analytics_demo()
            else:
                render_analytics_page(st.session_state.get("accuracy_tracker"))


if __name__ == "__main__":
    main()
