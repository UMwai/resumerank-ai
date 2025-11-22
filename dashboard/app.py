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
        "Get Help": "https://github.com/your-repo/dreamers-v2",
        "Report a bug": "https://github.com/your-repo/dreamers-v2/issues",
        "About": "Investment Intelligence Dashboard - Combining Clinical Trial, Patent, and Insider Signals"
    }
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }

    /* Card styling */
    .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }

    /* Table header styling */
    thead tr th {
        background-color: #f1f5f9 !important;
    }

    /* Success/error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


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
    Render the sidebar navigation.

    Returns:
        Selected page name
    """
    with st.sidebar:
        st.markdown("## Investment Intelligence")
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
            ],
            key="navigation",
            label_visibility="collapsed",
        )

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

        if st.button("Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Info
        st.caption("v1.0.0")
        st.caption("Last updated: " + st.session_state.get("last_update", "N/A"))

    return page


def main():
    """Main application entry point."""
    # Load configuration
    config = load_config()
    st.session_state["config"] = config

    # Initialize database manager
    db_manager = initialize_database_manager(config)
    st.session_state["db_manager"] = db_manager

    # Initialize fetchers
    clinical_fetcher, patent_fetcher, insider_fetcher, combined_fetcher = initialize_fetchers(db_manager)

    # Track last update time
    from datetime import datetime
    st.session_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Render sidebar and get selected page
    page = render_sidebar()

    # Check if in demo mode
    demo_mode = config.get("demo_mode", True)

    # Render selected page
    if page == "Home":
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
        if demo_mode:
            render_clinical_trials_demo()
        else:
            render_clinical_trials_page(clinical_fetcher)

    elif page == "Patent Cliff":
        if demo_mode:
            render_patent_cliff_demo()
        else:
            render_patent_cliff_page(patent_fetcher)

    elif page == "Insider/Hiring":
        if demo_mode:
            render_insider_hiring_demo()
        else:
            render_insider_hiring_page(insider_fetcher)

    elif page == "Watchlist":
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
        if demo_mode:
            render_alerts_demo()
        else:
            render_alerts_page(
                combined_fetcher,
                clinical_fetcher,
                patent_fetcher,
                insider_fetcher,
            )


if __name__ == "__main__":
    main()
