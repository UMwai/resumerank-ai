"""
Simple Position Sizer Module
Fixed allocation strategy with position limits
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionLimits:
    """Position sizing limits"""
    max_position_pct: float = 0.02  # 2% per position
    max_positions: int = 10  # Maximum 10 positions
    min_position_size: float = 1000  # Minimum $1000 per position
    max_position_size: float = 100000  # Maximum $100k per position


class SimplePositionSizer:
    """
    Simple position sizer with fixed 2% allocation
    Maximum 10 positions allowed
    """

    def __init__(self,
                 portfolio_value: float,
                 limits: Optional[PositionLimits] = None):
        """
        Initialize position sizer

        Args:
            portfolio_value: Total portfolio value
            limits: Position limits configuration
        """
        self.portfolio_value = portfolio_value
        self.limits = limits or PositionLimits()
        self.current_positions = {}
        self.position_count = 0

    def calculate_position_size(self,
                                 symbol: str,
                                 current_price: float,
                                 risk_multiplier: float = 1.0) -> Dict:
        """
        Calculate position size for a symbol

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            risk_multiplier: Risk adjustment factor (0.5 = half size, 2.0 = double)

        Returns:
            Dict with position size details
        """
        # Check if we can add more positions
        if self.position_count >= self.limits.max_positions:
            return {
                'symbol': symbol,
                'can_trade': False,
                'reason': f'Maximum positions reached ({self.limits.max_positions})',
                'position_size': 0,
                'shares': 0
            }

        # Calculate base position size (2% of portfolio)
        base_position_size = self.portfolio_value * self.limits.max_position_pct

        # Apply risk multiplier
        adjusted_size = base_position_size * risk_multiplier

        # Apply min/max limits
        final_size = max(
            self.limits.min_position_size,
            min(adjusted_size, self.limits.max_position_size)
        )

        # Calculate number of shares
        shares = int(final_size / current_price) if current_price > 0 else 0

        # Validate position doesn't exceed limits
        actual_size = shares * current_price
        position_pct = actual_size / self.portfolio_value if self.portfolio_value > 0 else 0

        # Final validation
        can_trade = (
            shares > 0 and
            actual_size >= self.limits.min_position_size and
            actual_size <= self.limits.max_position_size and
            position_pct <= self.limits.max_position_pct * 1.1  # 10% buffer
        )

        return {
            'symbol': symbol,
            'can_trade': can_trade,
            'position_size': actual_size if can_trade else 0,
            'shares': shares if can_trade else 0,
            'position_pct': position_pct,
            'current_price': current_price,
            'reason': 'OK' if can_trade else 'Position validation failed'
        }

    def validate_position(self,
                           symbol: str,
                           proposed_size: float) -> Dict:
        """
        Validate if a proposed position meets limits

        Args:
            symbol: Stock symbol
            proposed_size: Proposed position size in dollars

        Returns:
            Dict with validation result
        """
        # Check position count
        if symbol not in self.current_positions:
            if self.position_count >= self.limits.max_positions:
                return {
                    'valid': False,
                    'reason': f'Maximum {self.limits.max_positions} positions reached'
                }

        # Check position size limits
        if proposed_size < self.limits.min_position_size:
            return {
                'valid': False,
                'reason': f'Below minimum position size ${self.limits.min_position_size:,.0f}'
            }

        if proposed_size > self.limits.max_position_size:
            return {
                'valid': False,
                'reason': f'Exceeds maximum position size ${self.limits.max_position_size:,.0f}'
            }

        # Check portfolio percentage
        position_pct = proposed_size / self.portfolio_value if self.portfolio_value > 0 else 0
        if position_pct > self.limits.max_position_pct:
            return {
                'valid': False,
                'reason': f'Exceeds {self.limits.max_position_pct:.0%} portfolio allocation'
            }

        return {
            'valid': True,
            'reason': 'Position within limits',
            'position_pct': position_pct
        }

    def add_position(self, symbol: str, size: float, shares: int):
        """Add or update a position"""
        if symbol not in self.current_positions:
            self.position_count += 1

        self.current_positions[symbol] = {
            'size': size,
            'shares': shares,
            'weight': size / self.portfolio_value if self.portfolio_value > 0 else 0
        }

    def remove_position(self, symbol: str):
        """Remove a position"""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
            self.position_count -= 1

    def update_portfolio_value(self, new_value: float):
        """Update portfolio value and recalculate weights"""
        self.portfolio_value = new_value

        # Update weights for existing positions
        for symbol in self.current_positions:
            pos = self.current_positions[symbol]
            pos['weight'] = pos['size'] / new_value if new_value > 0 else 0

    def get_available_capital(self) -> float:
        """Get available capital for new positions"""
        total_allocated = sum(pos['size'] for pos in self.current_positions.values())
        max_allocation = self.portfolio_value * self.limits.max_position_pct * self.limits.max_positions
        return min(self.portfolio_value - total_allocated, max_allocation - total_allocated)

    def get_status(self) -> Dict:
        """Get current position sizing status"""
        total_allocated = sum(pos['size'] for pos in self.current_positions.values())

        return {
            'portfolio_value': self.portfolio_value,
            'position_count': self.position_count,
            'max_positions': self.limits.max_positions,
            'positions_available': self.limits.max_positions - self.position_count,
            'total_allocated': total_allocated,
            'available_capital': self.get_available_capital(),
            'allocation_pct': total_allocated / self.portfolio_value if self.portfolio_value > 0 else 0,
            'position_size_limits': {
                'min': self.limits.min_position_size,
                'max': self.limits.max_position_size,
                'target_pct': f"{self.limits.max_position_pct:.1%}"
            },
            'current_positions': self.current_positions
        }


# Example usage
if __name__ == "__main__":
    # Initialize position sizer with $100k portfolio
    sizer = SimplePositionSizer(portfolio_value=100000)

    print("Position Sizing Example")
    print("-" * 50)

    # Test position calculations
    test_positions = [
        ("AAPL", 175.50),
        ("MSFT", 380.25),
        ("GOOGL", 140.75),
        ("AMZN", 170.50)
    ]

    for symbol, price in test_positions:
        result = sizer.calculate_position_size(symbol, price)
        if result['can_trade']:
            sizer.add_position(symbol, result['position_size'], result['shares'])
            print(f"{symbol}: ${result['position_size']:,.0f} "
                  f"({result['shares']} shares @ ${price:.2f}) "
                  f"= {result['position_pct']:.2%} of portfolio")

    print("\nPortfolio Status:")
    status = sizer.get_status()
    print(f"Positions: {status['position_count']}/{status['max_positions']}")
    print(f"Allocated: ${status['total_allocated']:,.0f} ({status['allocation_pct']:.1%})")
    print(f"Available: ${status['available_capital']:,.0f}")