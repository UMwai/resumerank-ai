"""
Live Portfolio Monitoring Dashboard
Streamlit app for real-time portfolio tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="UMTrading Live Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules (adjust path as needed)
import sys
sys.path.append('/Users/waiyang/Desktop/repo/dreamers-v2/src')

from umtrading.risk.simple_circuit_breaker import SimpleCircuitBreaker
from umtrading.risk.simple_position_sizer import SimplePositionSizer
from umtrading.utils.daily_logger import DailyLogger


class PortfolioMonitor:
    """Portfolio monitoring and metrics calculation"""

    def __init__(self):
        self.portfolio_history = []
        self.trades_history = []
        self.spy_history = []

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns"""
        return prices.pct_change().dropna()

    def generate_mock_data(self, days: int = 30) -> Dict:
        """Generate mock portfolio data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=days)

        # Generate portfolio values with some volatility
        initial_value = 100000
        portfolio_returns = np.random.normal(0.001, 0.015, days)
        portfolio_values = initial_value * (1 + portfolio_returns).cumprod()

        # Generate SPY benchmark
        spy_returns = np.random.normal(0.0008, 0.012, days)
        spy_values = 100 * (1 + spy_returns).cumprod()

        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Portfolio': portfolio_values,
            'SPY': spy_values * (initial_value / 100)  # Scale to portfolio size
        })

        # Calculate metrics
        portfolio_returns = self.calculate_returns(df['Portfolio'])
        sharpe = self.calculate_sharpe_ratio(portfolio_returns[-30:])

        # Generate mock holdings
        holdings = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM'],
            'Shares': [100, 50, 30, 25, 40, 60, 35, 80],
            'Price': [175.50, 380.25, 140.75, 170.50, 480.25, 340.50, 240.75, 165.25],
            'Cost_Basis': [170.00, 375.00, 145.00, 165.00, 470.00, 335.00, 250.00, 160.00],
        })
        holdings['Value'] = holdings['Shares'] * holdings['Price']
        holdings['P&L'] = holdings['Shares'] * (holdings['Price'] - holdings['Cost_Basis'])
        holdings['P&L%'] = ((holdings['Price'] - holdings['Cost_Basis']) / holdings['Cost_Basis'] * 100)

        # Generate mock trades
        trade_times = pd.date_range(start=datetime.now() - timedelta(hours=6),
                                     end=datetime.now(), periods=10)
        trades = pd.DataFrame({
            'Time': trade_times,
            'Symbol': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 10),
            'Action': np.random.choice(['BUY', 'SELL'], 10),
            'Shares': np.random.randint(10, 100, 10),
            'Price': np.random.uniform(150, 400, 10),
            'P&L': np.random.uniform(-500, 1000, 10)
        })

        return {
            'portfolio_df': df,
            'current_value': portfolio_values[-1],
            'daily_pnl': portfolio_values[-1] - portfolio_values[-2],
            'daily_pnl_pct': (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2],
            'peak_value': portfolio_values.max(),
            'current_drawdown': (portfolio_values.max() - portfolio_values[-1]) / portfolio_values.max(),
            'sharpe_ratio': sharpe,
            'holdings': holdings,
            'trades': trades,
            'cash_balance': 15000,
            'market_regime': np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
        }


def create_equity_curve(df: pd.DataFrame) -> go.Figure:
    """Create equity curve chart"""
    fig = go.Figure()

    # Portfolio line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Portfolio'],
        mode='lines',
        name='Portfolio',
        line=dict(color='#2E86AB', width=2)
    ))

    # SPY benchmark
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['SPY'],
        mode='lines',
        name='SPY Benchmark',
        line=dict(color='#A23B72', width=1, dash='dash')
    ))

    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        yaxis=dict(tickformat='$,.0f')
    )

    return fig


def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Create drawdown chart"""
    # Calculate drawdown
    rolling_max = df['Portfolio'].expanding().max()
    drawdown = (df['Portfolio'] - rolling_max) / rolling_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=drawdown,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#E63946', width=1),
        fillcolor='rgba(230, 57, 70, 0.3)'
    ))

    # Add threshold lines
    fig.add_hline(y=-5, line_dash="dot", line_color="orange",
                  annotation_text="Warning (-5%)")
    fig.add_hline(y=-10, line_dash="dot", line_color="red",
                  annotation_text="Halt (-10%)")

    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=300,
        yaxis=dict(tickformat='.1f')
    )

    return fig


def main():
    """Main dashboard application"""

    # Title and header
    st.title("🚀 UMTrading Live Portfolio Monitor")
    st.markdown("### Real-time Portfolio Tracking & Risk Management")

    # Initialize components
    if 'circuit_breaker' not in st.session_state:
        st.session_state.circuit_breaker = SimpleCircuitBreaker()
        st.session_state.position_sizer = SimplePositionSizer(portfolio_value=100000)
        st.session_state.monitor = PortfolioMonitor()
        st.session_state.daily_logger = DailyLogger()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # Refresh settings
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)

        st.divider()

        # Circuit breaker controls
        st.subheader("🚨 Circuit Breaker")
        cb_status = st.session_state.circuit_breaker.get_status()

        # Status indicator
        if cb_status['state'] == 'normal':
            st.success(f"State: {cb_status['state'].upper()}")
        elif cb_status['state'] == 'warning':
            st.warning(f"State: {cb_status['state'].upper()}")
        else:
            st.error(f"State: {cb_status['state'].upper()}")

        st.metric("Current Drawdown", cb_status['current_drawdown'])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset CB", type="secondary"):
                st.session_state.circuit_breaker.reset()
                st.rerun()
        with col2:
            if st.button("Force Halt", type="secondary"):
                st.session_state.circuit_breaker.force_halt()
                st.rerun()

        st.divider()

        # Market regime indicator
        st.subheader("🎯 Market Regime")
        regime = st.selectbox("Current Regime",
                              ["BULLISH", "NEUTRAL", "BEARISH"],
                              index=1)

    # Get mock data (replace with real data in production)
    data = st.session_state.monitor.generate_mock_data()

    # Update circuit breaker with current value
    cb_result = st.session_state.circuit_breaker.update_portfolio_value(data['current_value'])

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${data['current_value']:,.0f}",
            f"{data['daily_pnl']:+,.0f} ({data['daily_pnl_pct']:+.2%})"
        )

    with col2:
        st.metric(
            "Daily P&L",
            f"${data['daily_pnl']:+,.0f}",
            f"{data['daily_pnl_pct']:+.2%}"
        )

    with col3:
        st.metric(
            "Current Drawdown",
            f"{data['current_drawdown']:.2%}",
            delta=None
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{data['sharpe_ratio']:.2f}",
            "30-day rolling"
        )

    with col5:
        st.metric(
            "Cash Balance",
            f"${data['cash_balance']:,.0f}",
            f"{data['cash_balance']/data['current_value']*100:.1f}%"
        )

    st.divider()

    # Charts row
    col1, col2 = st.columns([2, 1])

    with col1:
        # Equity curve
        fig_equity = create_equity_curve(data['portfolio_df'])
        st.plotly_chart(fig_equity, use_container_width=True)

    with col2:
        # Drawdown chart
        fig_dd = create_drawdown_chart(data['portfolio_df'])
        st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # Holdings and trades
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📊 Current Holdings")

        # Format holdings table
        holdings_display = data['holdings'].copy()
        holdings_display['Value'] = holdings_display['Value'].apply(lambda x: f"${x:,.0f}")
        holdings_display['Price'] = holdings_display['Price'].apply(lambda x: f"${x:.2f}")
        holdings_display['Cost_Basis'] = holdings_display['Cost_Basis'].apply(lambda x: f"${x:.2f}")

        # Color P&L
        def color_pnl(val):
            if isinstance(val, str):
                return val
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'

        holdings_display['P&L'] = holdings_display['P&L'].apply(
            lambda x: f"{'🟢' if x > 0 else '🔴'} ${x:+,.0f}"
        )
        holdings_display['P&L%'] = holdings_display['P&L%'].apply(
            lambda x: f"{x:+.1f}%"
        )

        st.dataframe(holdings_display, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📝 Recent Trades")

        # Format trades table
        trades_display = data['trades'].copy()
        trades_display['Time'] = trades_display['Time'].dt.strftime('%H:%M:%S')
        trades_display['Price'] = trades_display['Price'].apply(lambda x: f"${x:.2f}")
        trades_display['P&L'] = trades_display['P&L'].apply(
            lambda x: f"{'🟢' if x > 0 else '🔴'} ${x:+,.0f}"
        )

        # Color actions
        trades_display['Action'] = trades_display['Action'].apply(
            lambda x: f"{'🔵 BUY' if x == 'BUY' else '🟠 SELL'}"
        )

        st.dataframe(trades_display[['Time', 'Symbol', 'Action', 'Shares', 'Price', 'P&L']],
                     use_container_width=True, hide_index=True)

    st.divider()

    # Risk metrics section
    st.subheader("⚠️ Risk Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Position sizing status
        ps_status = st.session_state.position_sizer.get_status()
        st.metric(
            "Positions",
            f"{ps_status['position_count']}/{ps_status['max_positions']}",
            f"{ps_status['positions_available']} available"
        )

    with col2:
        st.metric(
            "Max Position Size",
            f"${ps_status['position_size_limits']['max']:,.0f}",
            ps_status['position_size_limits']['target_pct']
        )

    with col3:
        # Win rate
        winning_trades = len(data['trades'][data['trades']['P&L'] > 0])
        total_trades = len(data['trades'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        st.metric(
            "Win Rate (Today)",
            f"{win_rate:.1%}",
            f"{winning_trades}/{total_trades} trades"
        )

    with col4:
        st.metric(
            "Market Regime",
            data['market_regime'],
            "Current detection"
        )

    # Footer with last update time
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()