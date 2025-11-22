"""
Live Signal Feed Components

Real-time signal feed display with connection status indicators,
toast notifications, and live badge.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from utils.websocket_client import (
    Signal,
    SignalAggregator,
    ConnectionState,
    ConnectionStatus,
)


# Session state keys
SIGNAL_QUEUE_KEY = "live_signal_queue"
TOAST_QUEUE_KEY = "toast_notification_queue"


def get_signal_queue() -> List[Signal]:
    """Get the signal queue from session state."""
    if SIGNAL_QUEUE_KEY not in st.session_state:
        st.session_state[SIGNAL_QUEUE_KEY] = []
    return st.session_state[SIGNAL_QUEUE_KEY]


def add_signal_to_queue(signal: Signal) -> None:
    """Add a signal to the queue."""
    queue = get_signal_queue()
    queue.insert(0, signal)
    # Keep only last 100 signals
    st.session_state[SIGNAL_QUEUE_KEY] = queue[:100]


def get_toast_queue() -> List[Dict[str, Any]]:
    """Get toast notification queue."""
    if TOAST_QUEUE_KEY not in st.session_state:
        st.session_state[TOAST_QUEUE_KEY] = []
    return st.session_state[TOAST_QUEUE_KEY]


def add_toast(message: str, toast_type: str = "info") -> None:
    """Add a toast notification to the queue."""
    queue = get_toast_queue()
    queue.append({
        "message": message,
        "type": toast_type,
        "timestamp": datetime.now().isoformat(),
    })
    st.session_state[TOAST_QUEUE_KEY] = queue[-5:]  # Keep last 5


def render_live_badge(
    connected: bool,
    pulse: bool = True,
) -> None:
    """
    Render a LIVE badge indicating WebSocket connection status.

    Args:
        connected: Whether WebSocket is connected
        pulse: Whether to show pulse animation
    """
    if connected:
        pulse_css = """
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            animation: pulse 2s infinite;
        """ if pulse else ""

        st.markdown(
            f"""
            <div style="
                display: inline-flex;
                align-items: center;
                background-color: #ecfdf5;
                border: 1px solid #10b981;
                border-radius: 16px;
                padding: 4px 12px;
                font-size: 12px;
                font-weight: bold;
                color: #059669;
                {pulse_css}
            ">
                <span style="
                    width: 8px;
                    height: 8px;
                    background-color: #10b981;
                    border-radius: 50%;
                    margin-right: 6px;
                    display: inline-block;
                "></span>
                LIVE
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                display: inline-flex;
                align-items: center;
                background-color: #fef2f2;
                border: 1px solid #ef4444;
                border-radius: 16px;
                padding: 4px 12px;
                font-size: 12px;
                font-weight: bold;
                color: #dc2626;
            ">
                <span style="
                    width: 8px;
                    height: 8px;
                    background-color: #ef4444;
                    border-radius: 50%;
                    margin-right: 6px;
                    display: inline-block;
                "></span>
                OFFLINE
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_connection_status(
    status: Dict[str, ConnectionStatus],
    compact: bool = False,
) -> None:
    """
    Render connection status indicators for all WebSocket clients.

    Args:
        status: Dictionary of connection statuses
        compact: Use compact display
    """
    if compact:
        # Compact inline display
        indicators = []
        for name, conn_status in status.items():
            color = "#10b981" if conn_status.state == ConnectionState.CONNECTED else "#ef4444"
            icon = "O" if conn_status.state == ConnectionState.CONNECTED else "X"
            display_name = name.replace("_", " ").title()
            indicators.append(f'<span style="color: {color};">{icon} {display_name}</span>')

        st.markdown(
            f'<div style="font-size: 12px; color: #6b7280;">{"  |  ".join(indicators)}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Full display with details
        for name, conn_status in status.items():
            state = conn_status.state
            display_name = name.replace("_", " ").title()

            if state == ConnectionState.CONNECTED:
                icon = "white_check_mark"
                color = "#10b981"
            elif state == ConnectionState.CONNECTING:
                icon = "hourglass_flowing_sand"
                color = "#f59e0b"
            elif state == ConnectionState.RECONNECTING:
                icon = "arrows_counterclockwise"
                color = "#f59e0b"
            else:
                icon = "x"
                color = "#ef4444"

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f":{icon}: **{display_name}**")
            with col2:
                st.markdown(
                    f'<span style="color: {color}; font-size: 12px;">{state.value}</span>',
                    unsafe_allow_html=True,
                )

            if conn_status.error_message:
                st.caption(f"Error: {conn_status.error_message}")

            if conn_status.latency_ms:
                st.caption(f"Latency: {conn_status.latency_ms:.0f}ms")


def render_toast_notifications() -> None:
    """Render toast notifications from the queue."""
    queue = get_toast_queue()

    if not queue:
        return

    # Style mapping for toast types
    styles = {
        "success": {"bg": "#ecfdf5", "border": "#10b981", "icon": "check"},
        "error": {"bg": "#fef2f2", "border": "#ef4444", "icon": "x"},
        "warning": {"bg": "#fffbeb", "border": "#f59e0b", "icon": "warning"},
        "info": {"bg": "#eff6ff", "border": "#3b82f6", "icon": "info"},
    }

    # Render notifications container
    st.markdown(
        """
        <style>
        .toast-container {
            position: fixed;
            top: 70px;
            right: 20px;
            z-index: 9999;
            max-width: 350px;
        }
        .toast-notification {
            margin-bottom: 10px;
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render each notification
    toasts_html = '<div class="toast-container">'
    for toast in reversed(queue[-3:]):  # Show last 3
        style = styles.get(toast.get("type", "info"), styles["info"])
        toasts_html += f"""
            <div class="toast-notification" style="
                background-color: {style['bg']};
                border-left: 4px solid {style['border']};
            ">
                <div style="font-size: 14px;">{toast.get('message', '')}</div>
            </div>
        """
    toasts_html += '</div>'

    st.markdown(toasts_html, unsafe_allow_html=True)

    # Clear processed notifications
    st.session_state[TOAST_QUEUE_KEY] = []


def render_signal_card(signal: Signal, compact: bool = False) -> None:
    """
    Render a single signal card.

    Args:
        signal: Signal to display
        compact: Use compact display
    """
    # Determine colors based on signal type
    colors = {
        "bullish": {"bg": "#ecfdf5", "border": "#10b981", "text": "#059669"},
        "bearish": {"bg": "#fef2f2", "border": "#ef4444", "text": "#dc2626"},
        "info": {"bg": "#eff6ff", "border": "#3b82f6", "text": "#2563eb"},
    }
    style = colors.get(signal.signal_type, colors["info"])

    # Source icons
    source_icons = {
        "clinical_trial": "pill",
        "patent": "shield",
        "insider": "person-walking",
    }
    source_name = signal.source.replace("_", " ").title()

    # Time formatting
    time_diff = datetime.now() - signal.timestamp
    if time_diff < timedelta(minutes=1):
        time_str = "Just now"
    elif time_diff < timedelta(hours=1):
        minutes = int(time_diff.total_seconds() / 60)
        time_str = f"{minutes}m ago"
    elif time_diff < timedelta(days=1):
        hours = int(time_diff.total_seconds() / 3600)
        time_str = f"{hours}h ago"
    else:
        time_str = signal.timestamp.strftime("%Y-%m-%d %H:%M")

    if compact:
        st.markdown(
            f"""
            <div style="
                border-left: 3px solid {style['border']};
                padding: 8px 12px;
                margin-bottom: 8px;
                background-color: {style['bg']};
                border-radius: 0 4px 4px 0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: {style['text']};">{signal.ticker}</span>
                    <span style="font-size: 11px; color: #6b7280;">{time_str}</span>
                </div>
                <div style="font-size: 13px; margin-top: 2px;">{signal.title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                border: 1px solid {style['border']};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
                background-color: {style['bg']};
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: bold; font-size: 18px; color: #1f2937;">{signal.ticker}</span>
                        <span style="
                            background-color: {style['border']};
                            color: white;
                            padding: 2px 8px;
                            border-radius: 12px;
                            font-size: 11px;
                            margin-left: 8px;
                        ">{signal.signal_type.upper()}</span>
                    </div>
                    <span style="font-size: 12px; color: #6b7280;">{time_str}</span>
                </div>
                <div style="color: #6b7280; font-size: 13px; margin-bottom: 4px;">{signal.company_name}</div>
                <div style="font-weight: 500; font-size: 15px; margin-bottom: 8px; color: #1f2937;">{signal.title}</div>
                <div style="font-size: 14px; color: #4b5563; margin-bottom: 12px;">{signal.description}</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="
                        background-color: #f3f4f6;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 11px;
                        color: #6b7280;
                    ">{source_name}</span>
                    <div>
                        <span style="font-size: 12px; color: #6b7280;">Score: </span>
                        <span style="font-weight: bold; color: {style['text']};">{signal.score:.2f}</span>
                        <span style="font-size: 12px; color: #6b7280; margin-left: 12px;">Confidence: </span>
                        <span style="font-weight: bold;">{signal.confidence:.0%}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_feed(
    signals: List[Signal],
    title: str = "Live Signal Feed",
    compact: bool = False,
    max_signals: int = 10,
    show_filter: bool = True,
) -> None:
    """
    Render the live signal feed.

    Args:
        signals: List of signals to display
        title: Feed title
        compact: Use compact display
        max_signals: Maximum signals to show
        show_filter: Show source filter
    """
    st.markdown(f"### {title}")

    # Source filter
    if show_filter and signals:
        sources = list(set(s.source for s in signals))
        source_names = {s: s.replace("_", " ").title() for s in sources}

        selected_source = st.selectbox(
            "Filter by Source",
            options=["All"] + sources,
            format_func=lambda x: "All Sources" if x == "All" else source_names.get(x, x),
            key="live_feed_source_filter",
        )

        if selected_source != "All":
            signals = [s for s in signals if s.source == selected_source]

    # Display signals
    if signals:
        for signal in signals[:max_signals]:
            render_signal_card(signal, compact=compact)

        if len(signals) > max_signals:
            st.caption(f"Showing {max_signals} of {len(signals)} signals")
    else:
        st.info("No signals yet. Waiting for new data...")


def render_live_feed_sidebar(
    signals: List[Signal],
    max_signals: int = 5,
) -> None:
    """
    Render a compact live feed for the sidebar.

    Args:
        signals: List of signals to display
        max_signals: Maximum signals to show
    """
    st.markdown("#### Recent Signals")

    if signals:
        for signal in signals[:max_signals]:
            render_signal_card(signal, compact=True)
    else:
        st.caption("No recent signals")


def render_signal_summary(signals: List[Signal]) -> None:
    """
    Render a summary of recent signals.

    Args:
        signals: List of signals to summarize
    """
    if not signals:
        return

    # Count by type
    bullish = len([s for s in signals if s.signal_type == "bullish"])
    bearish = len([s for s in signals if s.signal_type == "bearish"])
    total = len(signals)

    # Count by source
    by_source = {}
    for signal in signals:
        source = signal.source
        by_source[source] = by_source.get(source, 0) + 1

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Signals (24h)", total)

    with col2:
        st.metric(
            "Bullish",
            bullish,
            delta=f"+{bullish}" if bullish > bearish else None,
        )

    with col3:
        st.metric(
            "Bearish",
            bearish,
            delta=f"-{bearish}" if bearish > bullish else None,
            delta_color="inverse",
        )

    # Source breakdown
    st.markdown("**By Source:**")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        source_name = source.replace("_", " ").title()
        st.progress(count / total, text=f"{source_name}: {count}")
