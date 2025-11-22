"""
Circuit Breakers and Kill Switches
Automatic trading halt mechanisms for risk control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import threading
from collections import deque

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    ACTIVE = "active"           # Normal trading
    TRIGGERED = "triggered"     # Halt triggered
    COOLING_DOWN = "cooldown"   # In cooldown period
    MANUAL_OVERRIDE = "manual"  # Manually overridden


class HaltReason(Enum):
    """Reasons for trading halt"""
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    DAILY_LOSS = "daily_loss"
    POSITION_LOSS = "position_loss"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    SYSTEM_ERROR = "system_error"
    MANUAL_HALT = "manual_halt"
    MARGIN_CALL = "margin_call"
    RISK_LIMIT_BREACH = "risk_limit_breach"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration and thresholds"""

    # Drawdown triggers (percentage of portfolio)
    portfolio_drawdown_warning: float = 0.08    # 8% drawdown warning
    portfolio_drawdown_halt: float = 0.12       # 12% drawdown halt
    portfolio_drawdown_kill: float = 0.15       # 15% drawdown kill switch

    # Daily loss triggers
    daily_loss_warning: float = 0.03            # 3% daily loss warning
    daily_loss_halt: float = 0.05               # 5% daily loss halt
    daily_loss_kill: float = 0.08               # 8% daily loss kill switch

    # Position loss triggers (percentage of position)
    position_loss_warning: float = 0.10         # 10% position loss warning
    position_loss_halt: float = 0.15            # 15% position loss halt
    position_loss_kill: float = 0.20            # 20% position loss kill switch

    # Volatility triggers (VIX levels)
    volatility_warning: float = 30              # VIX > 30 warning
    volatility_halt: float = 40                 # VIX > 40 halt
    volatility_kill: float = 50                 # VIX > 50 kill switch

    # Time-based controls
    cooldown_period_minutes: int = 30           # Cooldown after trigger
    max_triggers_per_day: int = 3               # Max triggers before day halt
    halt_duration_minutes: int = 15             # Initial halt duration

    # Escalation settings
    escalation_enabled: bool = True             # Enable progressive escalation
    escalation_multiplier: float = 1.5          # Multiply halt duration on repeated triggers


@dataclass
class CircuitBreakerEvent:
    """Record of circuit breaker events"""
    timestamp: datetime
    state: CircuitBreakerState
    reason: HaltReason
    severity: str  # 'warning', 'halt', 'kill'
    metrics: Dict
    action_taken: str
    resume_time: Optional[datetime] = None


class DrawdownCircuitBreaker:
    """
    Monitor and halt trading based on portfolio drawdown
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.high_water_mark = 0
        self.current_drawdown = 0
        self.state = CircuitBreakerState.ACTIVE
        self.events: deque = deque(maxlen=100)

    def update_portfolio_value(self, current_value: float) -> Dict:
        """
        Update portfolio value and check drawdown triggers

        Returns:
            Dictionary with trigger status and actions
        """
        # Update high water mark
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
            self.current_drawdown = 0
        else:
            # Calculate current drawdown
            if self.high_water_mark > 0:
                self.current_drawdown = (self.high_water_mark - current_value) / self.high_water_mark
            else:
                self.current_drawdown = 0

        # Check triggers
        result = {
            'current_value': current_value,
            'high_water_mark': self.high_water_mark,
            'current_drawdown': self.current_drawdown,
            'triggered': False,
            'severity': None,
            'action': None
        }

        # Kill switch check (highest priority)
        if self.current_drawdown >= self.config.portfolio_drawdown_kill:
            result['triggered'] = True
            result['severity'] = 'kill'
            result['action'] = 'emergency_liquidation'
            self._trigger_halt(HaltReason.PORTFOLIO_DRAWDOWN, 'kill', result)

        # Halt check
        elif self.current_drawdown >= self.config.portfolio_drawdown_halt:
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = 'halt_all_trading'
            self._trigger_halt(HaltReason.PORTFOLIO_DRAWDOWN, 'halt', result)

        # Warning check
        elif self.current_drawdown >= self.config.portfolio_drawdown_warning:
            result['severity'] = 'warning'
            result['action'] = 'reduce_position_sizes'
            logger.warning(f"Drawdown warning: {self.current_drawdown:.2%}")

        return result

    def _trigger_halt(self, reason: HaltReason, severity: str, metrics: Dict):
        """Record halt event and update state"""
        self.state = CircuitBreakerState.TRIGGERED

        event = CircuitBreakerEvent(
            timestamp=datetime.now(),
            state=self.state,
            reason=reason,
            severity=severity,
            metrics=metrics,
            action_taken=metrics['action']
        )

        if severity == 'halt':
            event.resume_time = datetime.now() + timedelta(
                minutes=self.config.halt_duration_minutes
            )
        elif severity == 'kill':
            event.resume_time = None  # Requires manual intervention

        self.events.append(event)
        logger.critical(f"Circuit breaker triggered: {reason.value}, Severity: {severity}")

    def can_resume_trading(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading can resume after halt

        Returns:
            Tuple of (can_resume, reason_if_not)
        """
        if self.state == CircuitBreakerState.ACTIVE:
            return True, None

        if self.state == CircuitBreakerState.MANUAL_OVERRIDE:
            return False, "Manual override in effect"

        if self.events:
            last_event = self.events[-1]

            # Check if kill switch was triggered
            if last_event.severity == 'kill':
                return False, "Kill switch triggered - manual intervention required"

            # Check cooldown period
            if last_event.resume_time and datetime.now() < last_event.resume_time:
                time_remaining = (last_event.resume_time - datetime.now()).seconds / 60
                return False, f"In cooldown period - {time_remaining:.1f} minutes remaining"

            # Check if drawdown has improved
            if self.current_drawdown < self.config.portfolio_drawdown_warning:
                self.state = CircuitBreakerState.ACTIVE
                return True, None
            else:
                return False, f"Drawdown still elevated: {self.current_drawdown:.2%}"

        return True, None


class LossCircuitBreaker:
    """
    Monitor daily and position-level losses
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.daily_pnl = {}  # Track daily P&L
        self.position_pnl = {}  # Track position P&L
        self.state = CircuitBreakerState.ACTIVE
        self.daily_triggers = 0
        self.last_reset = datetime.now().date()

    def update_daily_pnl(self, portfolio_value: float, starting_value: float) -> Dict:
        """
        Update daily P&L and check triggers

        Args:
            portfolio_value: Current portfolio value
            starting_value: Portfolio value at start of day
        """
        if datetime.now().date() != self.last_reset:
            self._reset_daily_counters()

        daily_return = (portfolio_value - starting_value) / starting_value if starting_value > 0 else 0

        result = {
            'daily_return': daily_return,
            'daily_pnl': portfolio_value - starting_value,
            'triggered': False,
            'severity': None,
            'action': None
        }

        # Check kill switch
        if daily_return <= -self.config.daily_loss_kill:
            result['triggered'] = True
            result['severity'] = 'kill'
            result['action'] = 'close_all_positions'
            self.daily_triggers += 1

        # Check halt
        elif daily_return <= -self.config.daily_loss_halt:
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = 'halt_new_positions'
            self.daily_triggers += 1

        # Check warning
        elif daily_return <= -self.config.daily_loss_warning:
            result['severity'] = 'warning'
            result['action'] = 'tighten_stop_losses'

        # Check if too many triggers today
        if self.daily_triggers >= self.config.max_triggers_per_day:
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = 'day_halt_limit_reached'
            logger.critical(f"Maximum daily triggers ({self.config.max_triggers_per_day}) reached")

        return result

    def update_position_pnl(self, symbol: str, entry_price: float, current_price: float, position_size: float) -> Dict:
        """
        Update position P&L and check triggers

        Args:
            symbol: Position symbol
            entry_price: Entry price for position
            current_price: Current price
            position_size: Size of position (negative for short)
        """
        # Calculate position return
        if entry_price > 0:
            if position_size > 0:  # Long position
                position_return = (current_price - entry_price) / entry_price
            else:  # Short position
                position_return = (entry_price - current_price) / entry_price
        else:
            position_return = 0

        self.position_pnl[symbol] = position_return

        result = {
            'symbol': symbol,
            'position_return': position_return,
            'position_pnl': position_return * abs(position_size) * entry_price,
            'triggered': False,
            'severity': None,
            'action': None
        }

        # Check kill switch
        if position_return <= -self.config.position_loss_kill:
            result['triggered'] = True
            result['severity'] = 'kill'
            result['action'] = f'emergency_close_{symbol}'

        # Check halt
        elif position_return <= -self.config.position_loss_halt:
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = f'close_position_{symbol}'

        # Check warning
        elif position_return <= -self.config.position_loss_warning:
            result['severity'] = 'warning'
            result['action'] = f'tighten_stop_loss_{symbol}'

        return result

    def _reset_daily_counters(self):
        """Reset daily counters at start of new day"""
        self.daily_triggers = 0
        self.last_reset = datetime.now().date()
        logger.info("Daily circuit breaker counters reset")


class VolatilityCircuitBreaker:
    """
    Monitor market volatility and correlation breakdowns
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.ACTIVE
        self.vix_history = deque(maxlen=20)
        self.correlation_history = deque(maxlen=20)

    def check_volatility_triggers(self, current_vix: float, historical_vol: float) -> Dict:
        """
        Check volatility-based triggers

        Args:
            current_vix: Current VIX level
            historical_vol: Historical volatility (e.g., 20-day)
        """
        self.vix_history.append(current_vix)

        result = {
            'current_vix': current_vix,
            'historical_vol': historical_vol,
            'vix_change': 0,
            'triggered': False,
            'severity': None,
            'action': None
        }

        # Calculate VIX change if history available
        if len(self.vix_history) > 1:
            result['vix_change'] = (current_vix - self.vix_history[-2]) / self.vix_history[-2]

        # Check kill switch
        if current_vix >= self.config.volatility_kill:
            result['triggered'] = True
            result['severity'] = 'kill'
            result['action'] = 'maximum_defensive_posture'

        # Check halt
        elif current_vix >= self.config.volatility_halt:
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = 'halt_risky_strategies'

        # Check warning
        elif current_vix >= self.config.volatility_warning:
            result['severity'] = 'warning'
            result['action'] = 'reduce_leverage'

        # Check for volatility spike (50% increase in VIX)
        if result['vix_change'] > 0.5:
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = 'volatility_spike_halt'
            logger.warning(f"Volatility spike detected: VIX increased {result['vix_change']:.1%}")

        return result

    def check_correlation_breakdown(self, correlation_matrix: pd.DataFrame, expected_correlations: pd.DataFrame) -> Dict:
        """
        Check for correlation breakdown (regime change)

        Args:
            correlation_matrix: Current correlation matrix
            expected_correlations: Expected/historical correlation matrix
        """
        # Calculate correlation deviation
        correlation_diff = (correlation_matrix - expected_correlations).abs()
        max_deviation = correlation_diff.max().max()
        mean_deviation = correlation_diff.mean().mean()

        result = {
            'max_deviation': max_deviation,
            'mean_deviation': mean_deviation,
            'triggered': False,
            'severity': None,
            'action': None
        }

        # Correlation breakdown thresholds
        if max_deviation > 0.5:  # Correlation changed by more than 0.5
            result['triggered'] = True
            result['severity'] = 'halt'
            result['action'] = 'correlation_breakdown_halt'
            logger.warning(f"Correlation breakdown detected: max deviation {max_deviation:.2f}")

        elif mean_deviation > 0.3:  # Average correlation change > 0.3
            result['severity'] = 'warning'
            result['action'] = 'review_correlations'

        return result


class CircuitBreakerManager:
    """
    Central manager for all circuit breakers
    Coordinates multiple circuit breakers and provides unified interface
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.drawdown_breaker = DrawdownCircuitBreaker(config)
        self.loss_breaker = LossCircuitBreaker(config)
        self.volatility_breaker = VolatilityCircuitBreaker(config)

        self.global_state = CircuitBreakerState.ACTIVE
        self.halt_reasons = []
        self.event_log = []
        self.override_password = None  # For manual override

    def update_all_metrics(self,
                          portfolio_value: float,
                          starting_value: float,
                          positions: Dict,
                          vix: float) -> Dict:
        """
        Update all circuit breakers and return consolidated status

        Returns:
            Dictionary with overall status and any triggered breakers
        """
        triggers = []

        # Check drawdown
        drawdown_result = self.drawdown_breaker.update_portfolio_value(portfolio_value)
        if drawdown_result['triggered']:
            triggers.append({
                'type': 'drawdown',
                'result': drawdown_result
            })

        # Check daily loss
        daily_result = self.loss_breaker.update_daily_pnl(portfolio_value, starting_value)
        if daily_result['triggered']:
            triggers.append({
                'type': 'daily_loss',
                'result': daily_result
            })

        # Check volatility
        vol_result = self.volatility_breaker.check_volatility_triggers(vix, 0)
        if vol_result['triggered']:
            triggers.append({
                'type': 'volatility',
                'result': vol_result
            })

        # Determine overall state
        if triggers:
            severities = [t['result']['severity'] for t in triggers]
            if 'kill' in severities:
                self.global_state = CircuitBreakerState.TRIGGERED
                overall_action = 'EMERGENCY_SHUTDOWN'
            elif 'halt' in severities:
                self.global_state = CircuitBreakerState.TRIGGERED
                overall_action = 'TRADING_HALTED'
            else:
                overall_action = 'WARNING_MODE'
        else:
            self.global_state = CircuitBreakerState.ACTIVE
            overall_action = 'NORMAL_OPERATIONS'

        # Log event
        event = {
            'timestamp': datetime.now().isoformat(),
            'global_state': self.global_state.value,
            'triggers': triggers,
            'action': overall_action,
            'metrics': {
                'portfolio_value': portfolio_value,
                'drawdown': self.drawdown_breaker.current_drawdown,
                'daily_return': daily_result['daily_return'],
                'vix': vix
            }
        }

        self.event_log.append(event)

        return {
            'global_state': self.global_state,
            'can_trade': self.global_state == CircuitBreakerState.ACTIVE,
            'triggers': triggers,
            'action': overall_action,
            'summary': self.get_status_summary()
        }

    def manual_override(self, password: str, action: str) -> bool:
        """
        Manually override circuit breakers (requires password)

        Args:
            password: Override password
            action: 'activate', 'halt', or 'reset'
        """
        # In production, use proper authentication
        if password != self.override_password:
            logger.error("Invalid override password")
            return False

        if action == 'activate':
            self.global_state = CircuitBreakerState.ACTIVE
            logger.info("Manual override: Trading activated")
        elif action == 'halt':
            self.global_state = CircuitBreakerState.MANUAL_OVERRIDE
            logger.info("Manual override: Trading halted")
        elif action == 'reset':
            self._reset_all_breakers()
            logger.info("Manual override: All breakers reset")
        else:
            return False

        return True

    def _reset_all_breakers(self):
        """Reset all circuit breakers to initial state"""
        self.drawdown_breaker = DrawdownCircuitBreaker(self.config)
        self.loss_breaker = LossCircuitBreaker(self.config)
        self.volatility_breaker = VolatilityCircuitBreaker(self.config)
        self.global_state = CircuitBreakerState.ACTIVE
        self.halt_reasons = []

    def get_status_summary(self) -> Dict:
        """Get summary of all circuit breaker states"""
        can_resume, reason = self.drawdown_breaker.can_resume_trading()

        return {
            'global_state': self.global_state.value,
            'can_trade': self.global_state == CircuitBreakerState.ACTIVE,
            'drawdown': {
                'current': self.drawdown_breaker.current_drawdown,
                'state': self.drawdown_breaker.state.value
            },
            'daily_loss': {
                'triggers_today': self.loss_breaker.daily_triggers,
                'max_triggers': self.config.max_triggers_per_day
            },
            'can_resume': can_resume,
            'resume_reason': reason,
            'last_events': list(self.event_log[-5:]) if self.event_log else []
        }

    def export_event_log(self, filepath: str):
        """Export event log to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.event_log, f, indent=2, default=str)
        logger.info(f"Event log exported to {filepath}")