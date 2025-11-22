"""
Simple Circuit Breaker Module
Tracks portfolio drawdown and auto-halts trading on breach
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    NORMAL = "normal"
    WARNING = "warning"
    HALT = "halt"
    KILL = "kill"


class SimpleCircuitBreaker:
    """
    Simple circuit breaker with 3 levels:
    - WARNING: 5% drawdown
    - HALT: 10% drawdown
    - KILL: 15% drawdown
    """

    def __init__(self,
                 warning_threshold: float = 0.05,
                 halt_threshold: float = 0.10,
                 kill_threshold: float = 0.15,
                 log_file: str = "circuit_breaker.log"):
        """
        Initialize circuit breaker

        Args:
            warning_threshold: Drawdown % to trigger warning (default 5%)
            halt_threshold: Drawdown % to trigger halt (default 10%)
            kill_threshold: Drawdown % to trigger kill (default 15%)
            log_file: File path for logging alerts
        """
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold
        self.kill_threshold = kill_threshold

        # Track portfolio peak
        self.peak_value = 0.0
        self.current_value = 0.0
        self.current_drawdown = 0.0

        # Current state
        self.state = CircuitState.NORMAL
        self.last_alert_time = None

        # Setup file logging
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)

    def update_portfolio_value(self, current_value: float) -> Dict:
        """
        Update portfolio value and check circuit breakers

        Args:
            current_value: Current portfolio value

        Returns:
            Dict with state, drawdown, action required
        """
        self.current_value = current_value

        # Update peak if new high
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
        else:
            # Calculate drawdown from peak
            if self.peak_value > 0:
                self.current_drawdown = (self.peak_value - current_value) / self.peak_value

        # Check thresholds and update state
        previous_state = self.state
        action = None

        if self.current_drawdown >= self.kill_threshold:
            self.state = CircuitState.KILL
            action = "KILL_ALL_POSITIONS"
            self._alert(f"KILL SWITCH ACTIVATED! Drawdown: {self.current_drawdown:.2%}")

        elif self.current_drawdown >= self.halt_threshold:
            self.state = CircuitState.HALT
            action = "HALT_TRADING"
            self._alert(f"TRADING HALTED! Drawdown: {self.current_drawdown:.2%}")

        elif self.current_drawdown >= self.warning_threshold:
            self.state = CircuitState.WARNING
            action = "REDUCE_RISK"
            if previous_state == CircuitState.NORMAL:
                self._alert(f"WARNING: Drawdown approaching limit: {self.current_drawdown:.2%}")
        else:
            self.state = CircuitState.NORMAL
            action = None

        return {
            'state': self.state.value,
            'drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'current_value': self.current_value,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }

    def _alert(self, message: str):
        """Send alert to console and log file"""
        alert_msg = f"[CIRCUIT BREAKER] {message}"

        # Console output with color coding
        if "KILL" in message:
            print(f"\033[91m{alert_msg}\033[0m")  # Red
            logger.critical(alert_msg)
        elif "HALT" in message:
            print(f"\033[93m{alert_msg}\033[0m")  # Yellow
            logger.error(alert_msg)
        else:
            print(f"\033[94m{alert_msg}\033[0m")  # Blue
            logger.warning(alert_msg)

        self.last_alert_time = datetime.now()

    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.state in [CircuitState.NORMAL, CircuitState.WARNING]

    def get_status(self) -> Dict:
        """Get current circuit breaker status"""
        return {
            'state': self.state.value,
            'can_trade': self.can_trade(),
            'current_drawdown': f"{self.current_drawdown:.2%}",
            'peak_value': self.peak_value,
            'current_value': self.current_value,
            'thresholds': {
                'warning': f"{self.warning_threshold:.1%}",
                'halt': f"{self.halt_threshold:.1%}",
                'kill': f"{self.kill_threshold:.1%}"
            },
            'last_alert': self.last_alert_time.isoformat() if self.last_alert_time else None
        }

    def reset(self):
        """Reset circuit breaker (use with caution)"""
        logger.info("Circuit breaker reset manually")
        self.state = CircuitState.NORMAL
        self.peak_value = self.current_value
        self.current_drawdown = 0.0

    def force_halt(self, reason: str = "Manual intervention"):
        """Manually trigger trading halt"""
        self.state = CircuitState.HALT
        self._alert(f"MANUAL HALT: {reason}")
        return {'action': 'HALT_TRADING', 'reason': reason}


# Example usage
if __name__ == "__main__":
    # Initialize circuit breaker
    cb = SimpleCircuitBreaker()

    # Simulate portfolio values
    portfolio_values = [100000, 98000, 96000, 94000, 92000, 90000, 88000, 85000]

    print("Circuit Breaker Simulation")
    print("-" * 50)

    for value in portfolio_values:
        result = cb.update_portfolio_value(value)
        print(f"Portfolio: ${value:,.0f} | State: {result['state']} | "
              f"Drawdown: {result['drawdown']:.2%} | Action: {result['action']}")

    print("\nFinal Status:")
    print(cb.get_status())