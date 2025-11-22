"""
Daily Summary Logger
Logs trading activity and performance metrics to CSV
"""

import csv
import os
from datetime import datetime, date
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class DailyLogger:
    """
    Logs daily trading summaries to CSV for analysis
    Format: Date, Portfolio Value, Regime, Trades, P&L, Drawdown
    """

    def __init__(self,
                 log_dir: str = "logs",
                 filename_prefix: str = "daily_summary"):
        """
        Initialize daily logger

        Args:
            log_dir: Directory for log files
            filename_prefix: Prefix for log filenames
        """
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # CSV file path (monthly files)
        self.csv_file = self._get_csv_path()

        # Initialize CSV if new file
        self._init_csv()

        # Track daily metrics
        self.today_start_value = 0
        self.trades_today = []

    def _get_csv_path(self) -> str:
        """Get CSV file path for current month"""
        today = date.today()
        filename = f"{self.filename_prefix}_{today.strftime('%Y_%m')}.csv"
        return os.path.join(self.log_dir, filename)

    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            headers = [
                'Date',
                'Time',
                'Portfolio_Value',
                'Daily_PnL',
                'Daily_PnL_Pct',
                'Cumulative_PnL',
                'Market_Regime',
                'Trades_Count',
                'Winning_Trades',
                'Losing_Trades',
                'Current_Drawdown',
                'Peak_Value',
                'Sharpe_Ratio',
                'Holdings_Count',
                'Cash_Balance',
                'Notes'
            ]

            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_daily_summary(self,
                          portfolio_value: float,
                          market_regime: str,
                          trades: List[Dict],
                          drawdown: float,
                          peak_value: float,
                          sharpe_ratio: float,
                          holdings_count: int,
                          cash_balance: float,
                          notes: str = "") -> Dict:
        """
        Log daily summary to CSV and console

        Args:
            portfolio_value: Current portfolio value
            market_regime: Current market regime
            trades: List of today's trades
            drawdown: Current drawdown percentage
            peak_value: Portfolio peak value
            sharpe_ratio: Current Sharpe ratio
            holdings_count: Number of positions
            cash_balance: Available cash
            notes: Optional notes

        Returns:
            Dict with summary data
        """
        now = datetime.now()

        # Calculate daily P&L
        daily_pnl = portfolio_value - self.today_start_value if self.today_start_value > 0 else 0
        daily_pnl_pct = daily_pnl / self.today_start_value if self.today_start_value > 0 else 0

        # Count winning/losing trades
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)

        # Prepare row data
        row = [
            now.strftime('%Y-%m-%d'),  # Date
            now.strftime('%H:%M:%S'),  # Time
            f"{portfolio_value:.2f}",  # Portfolio Value
            f"{daily_pnl:.2f}",  # Daily P&L
            f"{daily_pnl_pct:.4f}",  # Daily P&L %
            f"{portfolio_value - 100000:.2f}",  # Assume $100k start
            market_regime,  # Market Regime
            len(trades),  # Trades Count
            winning_trades,  # Winning Trades
            losing_trades,  # Losing Trades
            f"{drawdown:.4f}",  # Current Drawdown
            f"{peak_value:.2f}",  # Peak Value
            f"{sharpe_ratio:.4f}",  # Sharpe Ratio
            holdings_count,  # Holdings Count
            f"{cash_balance:.2f}",  # Cash Balance
            notes  # Notes
        ]

        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Create summary dict
        summary = {
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'portfolio_value': portfolio_value,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'market_regime': market_regime,
            'trades_today': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'current_drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'holdings': holdings_count,
            'cash': cash_balance
        }

        # Print to console
        self._print_summary(summary)

        # Log to file
        logger.info(f"Daily summary: {json.dumps(summary)}")

        return summary

    def _print_summary(self, summary: Dict):
        """Print formatted summary to console"""
        print("\n" + "=" * 60)
        print(f"DAILY SUMMARY - {summary['date']} {summary['time']}")
        print("=" * 60)

        # Portfolio metrics
        print(f"Portfolio Value:  ${summary['portfolio_value']:,.2f}")

        # Color code P&L
        pnl = summary['daily_pnl']
        pnl_pct = summary['daily_pnl_pct']
        if pnl >= 0:
            print(f"Daily P&L:        \033[92m+${pnl:,.2f} ({pnl_pct:+.2%})\033[0m")  # Green
        else:
            print(f"Daily P&L:        \033[91m${pnl:,.2f} ({pnl_pct:.2%})\033[0m")  # Red

        print(f"Current Drawdown: {summary['current_drawdown']:.2%}")
        print(f"Sharpe Ratio:     {summary['sharpe_ratio']:.2f}")

        # Trading activity
        print(f"\nMarket Regime:    {summary['market_regime']}")
        print(f"Trades Today:     {summary['trades_today']}")
        if summary['trades_today'] > 0:
            print(f"  Winners:        {summary['winning_trades']}")
            print(f"  Losers:         {summary['losing_trades']}")

        print(f"Holdings:         {summary['holdings']}")
        print(f"Cash Balance:     ${summary['cash']:,.2f}")
        print("=" * 60 + "\n")

    def log_trade(self, trade: Dict):
        """Log individual trade"""
        self.trades_today.append(trade)

        # Also log to separate trades file
        trades_file = os.path.join(self.log_dir, "trades.csv")
        file_exists = os.path.exists(trades_file)

        with open(trades_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade)

    def set_day_start_value(self, value: float):
        """Set starting portfolio value for the day"""
        self.today_start_value = value
        self.trades_today = []

    def log_error(self, error_msg: str, severity: str = "ERROR"):
        """Log error or warning"""
        timestamp = datetime.now().isoformat()
        error_log = os.path.join(self.log_dir, "errors.log")

        with open(error_log, 'a') as f:
            f.write(f"{timestamp} | {severity} | {error_msg}\n")

        logger.error(f"{severity}: {error_msg}")

    def get_today_summary(self) -> Dict:
        """Get summary of today's activity"""
        return {
            'start_value': self.today_start_value,
            'trades_count': len(self.trades_today),
            'trades': self.trades_today,
            'log_file': self.csv_file
        }


# Example usage
if __name__ == "__main__":
    # Initialize logger
    daily_logger = DailyLogger(log_dir="./logs")

    # Set day start value
    daily_logger.set_day_start_value(100000)

    # Simulate some trades
    trades = [
        {'symbol': 'AAPL', 'action': 'BUY', 'shares': 100, 'price': 175.50, 'pnl': 250},
        {'symbol': 'MSFT', 'action': 'SELL', 'shares': 50, 'price': 380.25, 'pnl': -125}
    ]

    for trade in trades:
        daily_logger.log_trade(trade)

    # Log daily summary
    summary = daily_logger.log_daily_summary(
        portfolio_value=101500,
        market_regime="BULLISH",
        trades=trades,
        drawdown=0.02,
        peak_value=103000,
        sharpe_ratio=1.45,
        holdings_count=8,
        cash_balance=15000,
        notes="Test summary"
    )

    print("\nLogged to:", daily_logger.csv_file)