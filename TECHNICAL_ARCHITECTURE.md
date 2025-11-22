# Technical Implementation Architecture
## Regime-Adaptive Trading Platform

### Executive Summary
This document outlines the complete technical architecture for a production-grade regime-adaptive trading platform capable of processing real-time market data, detecting market regimes, executing trades, and providing comprehensive monitoring and backtesting capabilities.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              External Data Sources                               │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────────────┐ │
│  │Yahoo     │  │Alpha Vantage │  │Polygon.io  │  │Broker APIs (IB/Alpaca)│ │
│  │Finance   │  │              │  │            │  │                        │ │
│  └────┬─────┘  └──────┬───────┘  └─────┬──────┘  └───────────┬────────────┘ │
└───────┼───────────────┼────────────────┼─────────────────────┼──────────────┘
        │               │                │                     │
        ▼               ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Data Ingestion Layer                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                    Apache Kafka / Redis Streams                         │    │
│  │  Topics: market-data, news-events, trade-signals, regime-changes       │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Data Pipeline  │     │  Regime Detection    │     │  Trade Execution    │
│    Service      │     │      Service          │     │     Engine          │
│                 │     │                       │     │                     │
│ • Normalization │     │ • State Detection    │     │ • Order Management  │
│ • Validation    │◄────┤ • Signal Generation  │────►│ • Risk Management   │
│ • Enrichment    │     │ • Alert Dispatch     │     │ • Position Tracking │
└────────┬────────┘     └──────────┬──────────┘     └──────────┬──────────┘
         │                         │                            │
         ▼                         ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Storage Layer                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐     │
│  │  TimescaleDB     │  │  PostgreSQL      │  │  Redis Cache              │     │
│  │  (Time Series)   │  │  (Operational)   │  │  (Hot Data)               │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   REST API      │     │   WebSocket Server   │     │  Dashboard UI       │
│   (FastAPI)     │     │   (Real-time feeds)  │     │  (Streamlit)        │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

## 2. Real-Time Market Data Pipeline

### 2.1 Architecture Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Market Data Pipeline Architecture                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Data Sources                    Ingestion                           │
│  ┌──────────┐                   ┌─────────────────┐                │
│  │Yahoo API │───────────────────►│                 │                │
│  └──────────┘    WebSocket      │  Data Collector │                │
│                                  │    Service      │                │
│  ┌──────────┐                   │                 │                │
│  │Alpha     │───────────────────►│ • Rate Limiting │                │
│  │Vantage   │    REST API       │ • Retry Logic   │                │
│  └──────────┘                   │ • Deduplication │                │
│                                  └────────┬────────┘                │
│  ┌──────────┐                            │                         │
│  │Polygon.io│───────────────────►────────┘                         │
│  └──────────┘    WebSocket                                         │
│                                           ▼                         │
│                              ┌───────────────────────┐              │
│                              │   Kafka Producer      │              │
│                              │   Topics:             │              │
│                              │   - raw-market-data   │              │
│                              │   - normalized-data   │              │
│                              └───────────┬───────────┘              │
│                                          ▼                         │
│                              ┌───────────────────────┐              │
│                              │   Stream Processor    │              │
│                              │   (Apache Flink/      │              │
│                              │    Kafka Streams)     │              │
│                              │                       │              │
│                              │ • Data Normalization  │              │
│                              │ • Schema Validation   │              │
│                              │ • Outlier Detection   │              │
│                              │ • Aggregation         │              │
│                              └───────────┬───────────┘              │
│                                          ▼                         │
│                              ┌───────────────────────┐              │
│                              │   TimescaleDB         │              │
│                              │   Hypertables:        │              │
│                              │   - market_ticks      │              │
│                              │   - ohlcv_1min        │              │
│                              │   - ohlcv_5min        │              │
│                              └───────────────────────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Specifications

```python
# /services/market_data/config.py
from pydantic import BaseSettings
from typing import Dict, List

class MarketDataConfig(BaseSettings):
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPICS: Dict[str, str] = {
        "raw": "market-data-raw",
        "normalized": "market-data-normalized",
        "aggregated": "market-data-aggregated"
    }

    # Data Source Configuration
    YAHOO_API_KEY: str
    ALPHA_VANTAGE_API_KEY: str
    POLYGON_API_KEY: str

    # Rate Limiting
    RATE_LIMITS: Dict[str, int] = {
        "yahoo": 2000,      # requests per hour
        "alpha_vantage": 500,  # requests per day
        "polygon": 10000    # requests per minute
    }

    # TimescaleDB Configuration
    TIMESCALE_CONNECTION: str
    RETENTION_POLICY: Dict[str, int] = {
        "ticks": 7,        # days
        "1min": 30,        # days
        "5min": 90,        # days
        "daily": 3650      # days (10 years)
    }

    # Performance Targets
    MAX_LATENCY_MS: int = 1000
    BATCH_SIZE: int = 1000
    FLUSH_INTERVAL_MS: int = 100
```

### 2.3 Data Normalization Schema

```python
# /services/market_data/schemas.py
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, validator
from typing import Optional, Literal

class NormalizedMarketData(BaseModel):
    """Unified schema for all market data sources"""

    # Identifiers
    symbol: str
    exchange: str
    source: Literal["yahoo", "alpha_vantage", "polygon"]

    # Timestamps
    timestamp: datetime
    received_at: datetime
    processed_at: Optional[datetime]

    # Price Data
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    vwap: Optional[Decimal]

    # Additional Metrics
    bid: Optional[Decimal]
    ask: Optional[Decimal]
    spread: Optional[Decimal]

    @validator('spread', always=True)
    def calculate_spread(cls, v, values):
        if v is None and values.get('bid') and values.get('ask'):
            return values['ask'] - values['bid']
        return v

    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }
```

---

## 3. Regime Detection Service

### 3.1 Service Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Regime Detection Service                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐         ┌──────────────────┐                   │
│  │  Kafka Consumer│────────►│  Event Processor │                   │
│  │  (Market Data) │         │                  │                   │
│  └────────────────┘         └────────┬─────────┘                   │
│                                      │                              │
│                                      ▼                              │
│                         ┌───────────────────────┐                   │
│                         │   Regime Detector     │                   │
│                         │                       │                   │
│                         │  • HMM Calculator     │                   │
│                         │  • Volatility Analyzer│                   │
│                         │  • Trend Detector     │                   │
│                         │  • Volume Analyzer    │                   │
│                         └───────────┬───────────┘                   │
│                                     │                               │
│                    ┌────────────────┼────────────────┐              │
│                    ▼                ▼                ▼              │
│         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│         │State Manager │  │Alert Service │  │Signal Publisher│      │
│         │              │  │              │  │              │      │
│         │• PostgreSQL  │  │• Slack       │  │• Kafka       │      │
│         │• Redis Cache │  │• Email       │  │• WebSocket   │      │
│         └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation Specifications

```python
# /services/regime_detection/engine.py
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

class MarketRegime(Enum):
    BULL_QUIET = "bull_quiet"
    BULL_VOLATILE = "bull_volatile"
    BEAR_QUIET = "bear_quiet"
    BEAR_VOLATILE = "bear_volatile"
    TRANSITION = "transition"

@dataclass
class RegimeState:
    """Current regime state with metadata"""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    detected_at: datetime
    indicators: Dict[str, float]
    previous_regime: Optional[MarketRegime]
    duration_hours: float

class RegimeDetectionEngine:
    """
    Main regime detection engine
    Runs as a background service with multiple detection frequencies
    """

    def __init__(self, config: Dict):
        self.config = config
        self.current_state: Optional[RegimeState] = None
        self.state_history: List[RegimeState] = []
        self.detection_intervals = {
            "realtime": timedelta(minutes=5),
            "tactical": timedelta(hours=1),
            "strategic": timedelta(days=1)
        }

    async def run(self):
        """Main execution loop"""
        tasks = [
            self._run_realtime_detection(),
            self._run_tactical_detection(),
            self._run_strategic_detection(),
            self._run_state_persistence()
        ]
        await asyncio.gather(*tasks)

    async def _run_realtime_detection(self):
        """High-frequency regime detection for immediate signals"""
        while True:
            try:
                # Fetch latest 5-min data
                data = await self._fetch_recent_data(minutes=30)

                # Quick regime check
                indicators = self._calculate_realtime_indicators(data)

                # Check for regime change
                if self._has_regime_changed(indicators, threshold=0.7):
                    await self._trigger_regime_change(indicators)

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Realtime detection error: {e}")
                await asyncio.sleep(60)

    async def _run_tactical_detection(self):
        """Hourly comprehensive regime analysis"""
        while True:
            try:
                # Fetch hourly data
                data = await self._fetch_recent_data(hours=24)

                # Comprehensive analysis
                regime = await self._detect_regime_comprehensive(data)

                # Update state
                await self._update_regime_state(regime)

                await asyncio.sleep(3600)  # 1 hour

            except Exception as e:
                logger.error(f"Tactical detection error: {e}")
                await asyncio.sleep(300)

    def _calculate_realtime_indicators(self, data) -> Dict[str, float]:
        """Calculate fast indicators for regime detection"""
        return {
            "volatility_ratio": self._calc_volatility_ratio(data),
            "trend_strength": self._calc_trend_strength(data),
            "volume_surge": self._calc_volume_surge(data),
            "momentum": self._calc_momentum(data),
            "vix_level": self._get_vix_level()
        }
```

### 3.3 State Persistence Strategy

```python
# /services/regime_detection/persistence.py
from sqlalchemy import Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RegimeHistory(Base):
    """Store regime state transitions"""
    __tablename__ = "regime_history"

    id = Column(String, primary_key=True)
    regime = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    detected_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    indicators = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=True)

class RegimeCache:
    """Redis-based caching for current regime"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    async def set_current_regime(self, state: RegimeState):
        key = "regime:current"
        value = {
            "regime": state.regime.value,
            "confidence": state.confidence,
            "detected_at": state.detected_at.isoformat(),
            "indicators": state.indicators
        }
        await self.redis.setex(
            key,
            self.ttl,
            json.dumps(value)
        )

    async def get_current_regime(self) -> Optional[RegimeState]:
        key = "regime:current"
        data = await self.redis.get(key)
        if data:
            return self._deserialize_state(json.loads(data))
        return None
```

---

## 4. Trade Execution Engine

### 4.1 Architecture Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Trade Execution Engine                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐         ┌──────────────────┐                   │
│  │ Signal Receiver│────────►│  Order Validator │                   │
│  │  (Kafka/WS)    │         │                  │                   │
│  └────────────────┘         │ • Risk Checks    │                   │
│                             │ • Position Limits │                   │
│                             │ • Margin Calc     │                   │
│                             └────────┬─────────┘                   │
│                                      │                              │
│                                      ▼                              │
│                         ┌───────────────────────┐                   │
│                         │   Order Router        │                   │
│                         │                       │                   │
│                         │  • Smart Routing      │                   │
│                         │  • Venue Selection    │                   │
│                         │  • Algo Selection     │                   │
│                         └───────────┬───────────┘                   │
│                                     │                               │
│                    ┌────────────────┼────────────────┐              │
│                    ▼                ▼                ▼              │
│         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│         │ IB Gateway   │  │Alpaca API    │  │Paper Trading │      │
│         │              │  │              │  │   Engine     │      │
│         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│                └──────────────────┼──────────────────┘              │
│                                   ▼                                 │
│                         ┌──────────────────┐                        │
│                         │ Order Management │                        │
│                         │     System       │                        │
│                         │                  │                        │
│                         │ • Fill Tracking  │                        │
│                         │ • Reconciliation │                        │
│                         │ • Audit Logging  │                        │
│                         └──────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Order Management System (OMS)

```python
# /services/trade_execution/oms.py
from enum import Enum
from typing import Dict, List, Optional, Union
from decimal import Decimal
from datetime import datetime
import uuid
from dataclasses import dataclass, field

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class Order:
    """Core order object"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # buy/sell
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "DAY"

    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Execution details
    filled_quantity: int = 0
    average_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")

    # Risk management
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    max_position_size: Optional[int] = None

    # Metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    regime: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class OrderManagementSystem:
    """
    Central order management system
    Handles order lifecycle, routing, and execution
    """

    def __init__(self, config: Dict):
        self.config = config
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.position_tracker = PositionTracker()
        self.risk_manager = RiskManager(config["risk_limits"])
        self.brokers = self._initialize_brokers()

    async def submit_order(self, order: Order) -> Order:
        """Submit order through validation and routing"""
        try:
            # Pre-trade validation
            await self._validate_order(order)

            # Risk checks
            risk_approved = await self.risk_manager.check_order(order)
            if not risk_approved:
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reason"] = "Risk limits exceeded"
                return order

            # Route to appropriate broker
            broker = self._select_broker(order)

            # Submit to broker
            broker_response = await broker.submit_order(order)

            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            order.metadata["broker"] = broker.name
            order.metadata["broker_order_id"] = broker_response["order_id"]

            # Track active order
            self.active_orders[order.order_id] = order

            # Audit log
            await self._audit_log("ORDER_SUBMITTED", order)

            return order

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata["error"] = str(e)
            await self._audit_log("ORDER_REJECTED", order)
            raise

    async def _validate_order(self, order: Order):
        """Comprehensive order validation"""
        validations = [
            self._validate_symbol,
            self._validate_quantity,
            self._validate_price_levels,
            self._validate_market_hours,
            self._validate_account_balance
        ]

        for validation in validations:
            result = await validation(order)
            if not result["valid"]:
                raise ValueError(f"Order validation failed: {result['reason']}")

    def _select_broker(self, order: Order) -> 'BrokerInterface':
        """Smart broker selection based on order characteristics"""
        if self.config["mode"] == "paper":
            return self.brokers["paper"]

        # Production broker selection logic
        if order.symbol.startswith("CRYPTO"):
            return self.brokers["alpaca"]
        elif order.order_type in [OrderType.STOP, OrderType.TRAILING_STOP]:
            return self.brokers["interactive_brokers"]
        else:
            # Default to broker with best execution
            return self._get_best_execution_broker(order)
```

### 4.3 Position Tracking & Reconciliation

```python
# /services/trade_execution/position_tracker.py
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import asyncio

@dataclass
class Position:
    """Current position representation"""
    symbol: str
    quantity: int
    average_cost: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    opened_at: datetime
    last_updated: datetime

    @property
    def pnl_percentage(self) -> Decimal:
        if self.average_cost > 0:
            return (self.unrealized_pnl / (self.average_cost * abs(self.quantity))) * 100
        return Decimal("0")

class PositionTracker:
    """
    Real-time position tracking and reconciliation
    """

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.reconciliation_interval = 300  # 5 minutes
        self.last_reconciliation: Optional[datetime] = None

    async def update_position(self, fill: 'Fill'):
        """Update position based on fill"""
        symbol = fill.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                average_cost=Decimal("0"),
                current_price=fill.price,
                market_value=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                opened_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

        position = self.positions[symbol]

        if fill.side == "buy":
            # Update average cost
            total_cost = (position.average_cost * position.quantity) + (fill.price * fill.quantity)
            position.quantity += fill.quantity
            position.average_cost = total_cost / position.quantity if position.quantity > 0 else Decimal("0")
        else:  # sell
            # Calculate realized PnL
            if position.quantity > 0:
                realized = (fill.price - position.average_cost) * min(fill.quantity, position.quantity)
                position.realized_pnl += realized

            position.quantity -= fill.quantity

        # Update market value and unrealized PnL
        position.current_price = fill.price
        position.market_value = position.current_price * position.quantity
        position.unrealized_pnl = (position.current_price - position.average_cost) * position.quantity
        position.last_updated = datetime.utcnow()

        # Remove position if closed
        if position.quantity == 0:
            del self.positions[symbol]

    async def reconcile_with_broker(self, broker_positions: Dict[str, 'BrokerPosition']):
        """Reconcile internal positions with broker"""
        discrepancies = []

        for symbol, internal_pos in self.positions.items():
            broker_pos = broker_positions.get(symbol)

            if not broker_pos:
                discrepancies.append({
                    "symbol": symbol,
                    "type": "missing_at_broker",
                    "internal_quantity": internal_pos.quantity
                })
                continue

            if abs(internal_pos.quantity - broker_pos.quantity) > 0:
                discrepancies.append({
                    "symbol": symbol,
                    "type": "quantity_mismatch",
                    "internal_quantity": internal_pos.quantity,
                    "broker_quantity": broker_pos.quantity,
                    "difference": internal_pos.quantity - broker_pos.quantity
                })

        # Check for positions at broker not in internal tracking
        for symbol, broker_pos in broker_positions.items():
            if symbol not in self.positions:
                discrepancies.append({
                    "symbol": symbol,
                    "type": "missing_internal",
                    "broker_quantity": broker_pos.quantity
                })

        if discrepancies:
            await self._handle_discrepancies(discrepancies)

        self.last_reconciliation = datetime.utcnow()
        return discrepancies
```

---

## 5. Backtesting Infrastructure

### 5.1 Architecture Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Backtesting Infrastructure                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────┐         ┌──────────────────┐                   │
│  │ Strategy Config│────────►│  Backtest Engine │                   │
│  │    (YAML)      │         │                  │                   │
│  └────────────────┘         │ • Event Simulator│                   │
│                             │ • Order Matching │                   │
│                             │ • Slippage Model │                   │
│                             └────────┬─────────┘                   │
│                                      │                              │
│                                      ▼                              │
│                         ┌───────────────────────┐                   │
│                         │  Historical Data      │                   │
│                         │     Provider          │                   │
│                         │                       │                   │
│                         │ • Data Loader         │                   │
│                         │ • Replay Engine       │                   │
│                         │ • Corporate Actions   │                   │
│                         └───────────┬───────────┘                   │
│                                     │                               │
│                                     ▼                               │
│                         ┌───────────────────────┐                   │
│                         │  Walk-Forward         │                   │
│                         │    Validator          │                   │
│                         │                       │                   │
│                         │ • In-Sample Testing   │                   │
│                         │ • Out-Sample Testing  │                   │
│                         │ • Parameter Tuning    │                   │
│                         └───────────┬───────────┘                   │
│                                     │                               │
│                                     ▼                               │
│                         ┌───────────────────────┐                   │
│                         │  Performance Metrics  │                   │
│                         │                       │                   │
│                         │ • Sharpe/Sortino      │                   │
│                         │ • Max Drawdown        │                   │
│                         │ • Win Rate            │                   │
│                         │ • Profit Factor       │                   │
│                         └───────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Backtesting Engine Implementation

```python
# /services/backtesting/engine.py
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    commission: Decimal = Decimal("0.001")  # 0.1%
    slippage: Decimal = Decimal("0.0005")   # 0.05%

    # Walk-forward settings
    in_sample_periods: int = 252  # Trading days
    out_sample_periods: int = 63  # Trading days
    reoptimize_frequency: int = 21  # Trading days

    # Execution settings
    fill_at_close: bool = False
    use_adjusted_close: bool = True
    handle_splits: bool = True
    handle_dividends: bool = True

class BacktestEngine:
    """
    High-performance backtesting engine with walk-forward validation
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_provider = HistoricalDataProvider()
        self.event_queue = asyncio.Queue()
        self.portfolio = Portfolio(config.initial_capital)
        self.performance_tracker = PerformanceTracker()
        self.regime_engine = RegimeDetectionEngine()

    async def run_backtest(
        self,
        strategy: 'Strategy',
        symbols: List[str]
    ) -> 'BacktestResults':
        """Run complete backtest with walk-forward validation"""

        # Load historical data
        data = await self.data_provider.load_data(
            symbols=symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date
        )

        # Initialize walk-forward windows
        windows = self._generate_walk_forward_windows()

        results = []
        for window in windows:
            # Optimize parameters on in-sample data
            if window["optimize"]:
                optimal_params = await self._optimize_parameters(
                    strategy=strategy,
                    data=data[window["in_sample_start"]:window["in_sample_end"]],
                    symbols=symbols
                )
                strategy.update_parameters(optimal_params)

            # Run backtest on out-of-sample data
            window_result = await self._run_window(
                strategy=strategy,
                data=data[window["out_sample_start"]:window["out_sample_end"]],
                symbols=symbols
            )

            results.append(window_result)

        # Aggregate results
        return self._aggregate_results(results)

    async def _run_window(
        self,
        strategy: 'Strategy',
        data: pd.DataFrame,
        symbols: List[str]
    ) -> 'WindowResult':
        """Run backtest on a single window"""

        # Reset portfolio for this window
        self.portfolio.reset()

        # Event-driven simulation
        for timestamp, market_data in data.iterrows():
            # Detect regime
            regime = await self.regime_engine.detect_regime(
                data=data[:timestamp],
                lookback=30
            )

            # Generate signals
            signals = await strategy.generate_signals(
                market_data=market_data,
                regime=regime,
                portfolio=self.portfolio
            )

            # Execute trades
            for signal in signals:
                order = self._create_order_from_signal(signal)
                fill = self._simulate_fill(order, market_data)
                await self.portfolio.process_fill(fill)

            # Update portfolio valuation
            await self.portfolio.mark_to_market(market_data)

            # Track performance
            self.performance_tracker.update(
                timestamp=timestamp,
                portfolio_value=self.portfolio.total_value,
                positions=self.portfolio.positions
            )

        return WindowResult(
            start_date=data.index[0],
            end_date=data.index[-1],
            metrics=self.performance_tracker.calculate_metrics(),
            trades=self.portfolio.trade_history
        )

    def _simulate_fill(self, order: Order, market_data: pd.Series) -> 'Fill':
        """Simulate order fill with slippage and market impact"""

        base_price = market_data[order.symbol]

        # Apply slippage
        if order.side == "buy":
            fill_price = base_price * (1 + self.config.slippage)
        else:
            fill_price = base_price * (1 - self.config.slippage)

        # Apply market impact for large orders
        market_impact = self._calculate_market_impact(
            order_size=order.quantity,
            avg_volume=market_data[f"{order.symbol}_volume"]
        )
        fill_price *= (1 + market_impact)

        # Calculate commission
        commission = abs(order.quantity * fill_price) * self.config.commission

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=market_data.name
        )
```

### 5.3 Performance Metrics Calculation

```python
# /services/backtesting/metrics.py
import numpy as np
from typing import Dict, List, Tuple
from decimal import Decimal

class PerformanceMetrics:
    """Comprehensive performance metrics calculator"""

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        if downside_deviation == 0:
            return float('inf')

        return np.sqrt(252) * excess_returns.mean() / downside_deviation

    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and duration"""
        cumulative = np.cumprod(1 + equity_curve)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()

        # Find start of drawdown
        dd_start = max_dd_idx
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                dd_start = i
                break

        # Find end of drawdown (recovery)
        dd_end = len(drawdown) - 1
        for i in range(max_dd_idx, len(drawdown)):
            if drawdown[i] == 0:
                dd_end = i
                break

        duration = dd_end - dd_start

        return max_dd, dd_start, duration

    @staticmethod
    def calculate_calmar_ratio(
        returns: np.ndarray,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        if max_drawdown == 0:
            return float('inf')
        annual_return = (1 + returns.mean()) ** 252 - 1
        return annual_return / abs(max_drawdown)
```

---

## 6. Monitoring & Observability

### 6.1 Metrics Architecture

```yaml
# /monitoring/prometheus/config.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'market-data-pipeline'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: /metrics

  - job_name: 'regime-detection'
    static_configs:
      - targets: ['localhost:8002']

  - job_name: 'trade-execution'
    static_configs:
      - targets: ['localhost:8003']

  - job_name: 'portfolio-service'
    static_configs:
      - targets: ['localhost:8004']

# Alert rules
rule_files:
  - 'alerts/trading_alerts.yml'
  - 'alerts/infrastructure_alerts.yml'
```

### 6.2 Key Metrics to Track

```python
# /services/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Market Data Metrics
market_data_received = Counter(
    'market_data_received_total',
    'Total market data points received',
    ['source', 'symbol']
)

market_data_latency = Histogram(
    'market_data_latency_seconds',
    'Latency from market to processing',
    ['source'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Regime Detection Metrics
current_regime = Gauge(
    'current_market_regime',
    'Current detected market regime',
    ['regime_type']
)

regime_confidence = Gauge(
    'regime_detection_confidence',
    'Confidence level of current regime detection'
)

regime_changes = Counter(
    'regime_changes_total',
    'Total number of regime changes detected'
)

# Trading Metrics
orders_submitted = Counter(
    'orders_submitted_total',
    'Total orders submitted',
    ['strategy', 'order_type', 'side']
)

orders_filled = Counter(
    'orders_filled_total',
    'Total orders filled',
    ['strategy', 'broker']
)

order_latency = Histogram(
    'order_execution_latency_seconds',
    'Time from signal to fill',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# Portfolio Metrics
portfolio_value = Gauge(
    'portfolio_total_value_usd',
    'Total portfolio value in USD'
)

position_pnl = Gauge(
    'position_pnl_usd',
    'P&L per position',
    ['symbol', 'type']  # type: realized/unrealized
)

portfolio_sharpe = Gauge(
    'portfolio_sharpe_ratio',
    'Rolling 30-day Sharpe ratio'
)

portfolio_drawdown = Gauge(
    'portfolio_drawdown_percentage',
    'Current drawdown from peak'
)

# System Metrics
api_requests = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

api_latency = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['endpoint', 'method']
)

database_connections = Gauge(
    'database_connections_active',
    'Active database connections',
    ['database']
)

kafka_lag = Gauge(
    'kafka_consumer_lag',
    'Kafka consumer lag',
    ['topic', 'partition']
)
```

### 6.3 Alerting Configuration

```yaml
# /monitoring/alerts/trading_alerts.yml
groups:
  - name: trading_alerts
    interval: 30s
    rules:
      - alert: HighOrderRejectionRate
        expr: |
          rate(orders_submitted_total[5m]) > 0 and
          rate(orders_rejected_total[5m]) / rate(orders_submitted_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High order rejection rate detected"
          description: "Order rejection rate is {{ $value | humanizePercentage }}"

      - alert: LargeDrawdown
        expr: portfolio_drawdown_percentage > 10
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Large portfolio drawdown detected"
          description: "Portfolio drawdown is {{ $value }}%"

      - alert: RegimeChangeDetected
        expr: increase(regime_changes_total[1h]) > 0
        labels:
          severity: info
        annotations:
          summary: "Market regime change detected"
          description: "Market regime has changed in the last hour"

      - alert: MarketDataDelayed
        expr: |
          time() - market_data_last_received_timestamp > 60
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Market data feed delayed"
          description: "No market data received for {{ $value }} seconds"

      - alert: PositionLimitExceeded
        expr: |
          abs(position_size) > position_limit
        labels:
          severity: critical
        annotations:
          summary: "Position limit exceeded"
          description: "Position size {{ $value }} exceeds limit"
```

### 6.4 Dashboard Configuration

```python
# /monitoring/dashboards/config.py
from typing import Dict, List

DASHBOARD_CONFIG = {
    "Trading Performance": {
        "refresh_interval": 5,  # seconds
        "panels": [
            {
                "title": "Portfolio Value",
                "type": "timeseries",
                "metrics": ["portfolio_total_value_usd"],
                "period": "24h"
            },
            {
                "title": "P&L by Position",
                "type": "table",
                "metrics": ["position_pnl_usd"],
                "group_by": ["symbol"]
            },
            {
                "title": "Current Regime",
                "type": "stat",
                "metrics": ["current_market_regime", "regime_confidence"]
            },
            {
                "title": "Order Flow",
                "type": "timeseries",
                "metrics": [
                    "rate(orders_submitted_total[5m])",
                    "rate(orders_filled_total[5m])"
                ],
                "period": "1h"
            }
        ]
    },

    "System Health": {
        "refresh_interval": 10,
        "panels": [
            {
                "title": "API Latency",
                "type": "heatmap",
                "metrics": ["api_request_duration_seconds"],
                "buckets": [0.01, 0.05, 0.1, 0.5, 1.0]
            },
            {
                "title": "Market Data Latency",
                "type": "gauge",
                "metrics": ["market_data_latency_seconds"],
                "thresholds": {
                    "green": 0.1,
                    "yellow": 0.5,
                    "red": 1.0
                }
            },
            {
                "title": "Kafka Lag",
                "type": "timeseries",
                "metrics": ["kafka_consumer_lag"],
                "group_by": ["topic"]
            },
            {
                "title": "Database Connections",
                "type": "stat",
                "metrics": ["database_connections_active"],
                "group_by": ["database"]
            }
        ]
    }
}
```

---

## 7. Deployment Strategy

### 7.1 Docker Configuration

```dockerfile
# /docker/services/Dockerfile.market_data
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/market_data /app/services/market_data
COPY common /app/common

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Run service
CMD ["uvicorn", "services.market_data.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 7.2 Docker Compose Configuration

```yaml
# /docker-compose.yml
version: '3.8'

services:
  # Infrastructure Services
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: trading_platform
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader"]
      interval: 10s
      timeout: 5s
      retries: 5

  timescaledb:
    image: timescale/timescaledb:2.11.0-pg14
    environment:
      POSTGRES_DB: market_data
      POSTGRES_USER: market_user
      POSTGRES_PASSWORD: ${TIMESCALE_PASSWORD}
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./init_scripts/timescale_init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5433:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_LOG_RETENTION_HOURS: 168
    volumes:
      - kafka_data:/var/lib/kafka/data
    ports:
      - "9092:9092"

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data

  # Application Services
  market-data:
    build:
      context: .
      dockerfile: docker/services/Dockerfile.market_data
    depends_on:
      - kafka
      - timescaledb
      - redis
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      TIMESCALE_CONNECTION: postgresql://market_user:${TIMESCALE_PASSWORD}@timescaledb:5432/market_data
      REDIS_URL: redis://redis:6379
      YAHOO_API_KEY: ${YAHOO_API_KEY}
      ALPHA_VANTAGE_API_KEY: ${ALPHA_VANTAGE_API_KEY}
      POLYGON_API_KEY: ${POLYGON_API_KEY}
    ports:
      - "8001:8001"
    restart: unless-stopped

  regime-detection:
    build:
      context: .
      dockerfile: docker/services/Dockerfile.regime_detection
    depends_on:
      - kafka
      - postgres
      - redis
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      DATABASE_URL: postgresql://trader:${DB_PASSWORD}@postgres:5432/trading_platform
      REDIS_URL: redis://redis:6379
    ports:
      - "8002:8002"
    restart: unless-stopped

  trade-execution:
    build:
      context: .
      dockerfile: docker/services/Dockerfile.trade_execution
    depends_on:
      - kafka
      - postgres
      - redis
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      DATABASE_URL: postgresql://trader:${DB_PASSWORD}@postgres:5432/trading_platform
      REDIS_URL: redis://redis:6379
      IB_GATEWAY_HOST: ${IB_GATEWAY_HOST}
      IB_GATEWAY_PORT: ${IB_GATEWAY_PORT}
      ALPACA_API_KEY: ${ALPACA_API_KEY}
      ALPACA_SECRET_KEY: ${ALPACA_SECRET_KEY}
      TRADING_MODE: ${TRADING_MODE:-paper}
    ports:
      - "8003:8003"
    restart: unless-stopped

  api-gateway:
    build:
      context: .
      dockerfile: docker/services/Dockerfile.api
    depends_on:
      - market-data
      - regime-detection
      - trade-execution
    environment:
      MARKET_DATA_URL: http://market-data:8001
      REGIME_URL: http://regime-detection:8002
      EXECUTION_URL: http://trade-execution:8003
    ports:
      - "8000:8000"
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: docker/services/Dockerfile.dashboard
    depends_on:
      - api-gateway
    environment:
      API_URL: http://api-gateway:8000
    ports:
      - "8501:8501"
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:v2.45.0
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.0.0
    depends_on:
      - prometheus
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    ports:
      - "3000:3000"
    restart: unless-stopped

volumes:
  postgres_data:
  timescale_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  prometheus_data:
  grafana_data:
```

### 7.3 CI/CD Pipeline

```yaml
# /.github/workflows/deploy.yml
name: Deploy Trading Platform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ --cov=services --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service:
          - market-data
          - regime-detection
          - trade-execution
          - api-gateway
          - dashboard

    steps:
      - uses: actions/checkout@v3

      - name: Log in to Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: docker/services/Dockerfile.${{ matrix.service }}
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.service }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.service }}:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to staging
        run: |
          # SSH into staging server and update services
          ssh -o StrictHostKeyChecking=no ${{ secrets.STAGING_USER }}@${{ secrets.STAGING_HOST }} << 'EOF'
            cd /opt/trading-platform
            git pull origin main
            docker-compose -f docker-compose.staging.yml pull
            docker-compose -f docker-compose.staging.yml up -d --remove-orphans
            docker-compose -f docker-compose.staging.yml exec -T api-gateway python manage.py migrate
          EOF

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://trading.example.com

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        run: |
          # Deploy using blue-green strategy
          ssh -o StrictHostKeyChecking=no ${{ secrets.PROD_USER }}@${{ secrets.PROD_HOST }} << 'EOF'
            cd /opt/trading-platform

            # Pull new images
            docker-compose -f docker-compose.prod.yml pull

            # Start new containers (green)
            docker-compose -f docker-compose.prod.yml up -d --scale api-gateway=2

            # Health check
            sleep 30
            curl -f http://localhost:8000/health || exit 1

            # Switch traffic to new containers
            docker-compose -f docker-compose.prod.yml up -d --remove-orphans

            # Remove old containers
            docker container prune -f
          EOF
```

### 7.4 Environment Configuration

```bash
# /.env.example
# Database
DB_PASSWORD=secure_password_here
TIMESCALE_PASSWORD=secure_password_here

# Market Data APIs
YAHOO_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Broker APIs
IB_GATEWAY_HOST=localhost
IB_GATEWAY_PORT=4001
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here

# Trading Configuration
TRADING_MODE=paper  # paper or live
MAX_POSITION_SIZE=10000
MAX_PORTFOLIO_RISK=0.02

# Monitoring
GRAFANA_PASSWORD=secure_password_here
SENTRY_DSN=https://your-sentry-dsn.ingest.sentry.io/

# AWS (for production)
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
```

---

## 8. Database Schema

### 8.1 TimescaleDB Schema for Market Data

```sql
-- /init_scripts/timescale_init.sql

-- Create market data hypertable
CREATE TABLE market_ticks (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(20) NOT NULL,
    exchange    VARCHAR(20),
    price       DECIMAL(20, 8) NOT NULL,
    volume      BIGINT,
    bid         DECIMAL(20, 8),
    ask         DECIMAL(20, 8),
    source      VARCHAR(20)
);

SELECT create_hypertable('market_ticks', 'time');
CREATE INDEX idx_market_ticks_symbol_time ON market_ticks (symbol, time DESC);

-- Create OHLCV continuous aggregates
CREATE MATERIALIZED VIEW ohlcv_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(volume) AS volume,
    count(*) AS ticks
FROM market_ticks
GROUP BY bucket, symbol;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('ohlcv_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- Create 5-minute aggregates
CREATE MATERIALIZED VIEW ohlcv_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', bucket) AS bucket,
    symbol,
    first(open, bucket) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, bucket) AS close,
    sum(volume) AS volume,
    sum(ticks) AS ticks
FROM ohlcv_1min
GROUP BY time_bucket('5 minutes', bucket), symbol;

-- Add compression policy
ALTER TABLE market_ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('market_ticks', INTERVAL '7 days');
```

### 8.2 PostgreSQL Schema for Trading Data

```sql
-- /migrations/001_trading_schema.sql

-- Regime states table
CREATE TABLE regime_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime VARCHAR(50) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    indicators JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_regime_states_detected_at ON regime_states (detected_at DESC);
CREATE INDEX idx_regime_states_active ON regime_states (ended_at) WHERE ended_at IS NULL;

-- Orders table
CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    limit_price DECIMAL(20, 8),
    stop_price DECIMAL(20, 8),
    time_in_force VARCHAR(10) NOT NULL DEFAULT 'DAY',

    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,

    -- Execution details
    filled_quantity INTEGER DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,

    -- Relationships
    strategy_id VARCHAR(50),
    signal_id UUID,
    regime VARCHAR(50),

    -- Metadata
    broker VARCHAR(20),
    broker_order_id VARCHAR(100),
    metadata JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_orders_symbol ON orders (symbol);
CREATE INDEX idx_orders_status ON orders (status);
CREATE INDEX idx_orders_submitted_at ON orders (submitted_at DESC);
CREATE INDEX idx_orders_strategy ON orders (strategy_id);

-- Trades table (filled orders)
CREATE TABLE trades (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID REFERENCES orders(order_id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    executed_at TIMESTAMPTZ NOT NULL,

    -- P&L tracking
    realized_pnl DECIMAL(20, 8),

    -- Metadata
    broker VARCHAR(20),
    broker_trade_id VARCHAR(100),
    metadata JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_symbol ON trades (symbol);
CREATE INDEX idx_trades_executed_at ON trades (executed_at DESC);
CREATE INDEX idx_trades_order_id ON trades (order_id);

-- Positions table
CREATE TABLE positions (
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    average_cost DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    market_value DECIMAL(20, 8),

    -- P&L tracking
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,

    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(symbol) WHERE closed_at IS NULL
);

CREATE INDEX idx_positions_symbol ON positions (symbol);
CREATE INDEX idx_positions_active ON positions (closed_at) WHERE closed_at IS NULL;

-- Audit log table
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_log_created_at ON audit_log (created_at DESC);
CREATE INDEX idx_audit_log_entity ON audit_log (entity_type, entity_id);
CREATE INDEX idx_audit_log_event_type ON audit_log (event_type);
```

---

## 9. Security Considerations

### 9.1 API Security

```python
# /services/api/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional
import secrets

class SecurityManager:
    """Comprehensive security manager for API"""

    def __init__(self, config: Dict):
        self.config = config
        self.secret_key = config["JWT_SECRET_KEY"]
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)

    def create_access_token(self, user_id: str, scopes: List[str]) -> str:
        """Create JWT access token"""
        payload = {
            "sub": user_id,
            "scopes": scopes,
            "exp": datetime.utcnow() + self.access_token_expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# API key validation for broker integrations
class APIKeyValidator:
    """Validate and encrypt API keys for external services"""

    def __init__(self):
        self.cipher_suite = Fernet(Fernet.generate_key())

    def encrypt_key(self, api_key: str) -> bytes:
        """Encrypt API key for storage"""
        return self.cipher_suite.encrypt(api_key.encode())

    def decrypt_key(self, encrypted_key: bytes) -> str:
        """Decrypt API key for use"""
        return self.cipher_suite.decrypt(encrypted_key).decode()
```

### 9.2 Data Encryption

```python
# /services/common/encryption.py
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import os

class DataEncryption:
    """Encrypt sensitive data at rest"""

    @staticmethod
    def encrypt_sensitive_fields(data: Dict, fields: List[str]) -> Dict:
        """Encrypt specified fields in dictionary"""
        encrypted_data = data.copy()
        for field in fields:
            if field in encrypted_data:
                encrypted_data[field] = DataEncryption.encrypt_string(
                    str(encrypted_data[field])
                )
        return encrypted_data

    @staticmethod
    def encrypt_string(plaintext: str) -> str:
        """Encrypt string using AES-256"""
        # Implementation details...
        pass
```

---

## 10. Performance Optimization

### 10.1 Caching Strategy

```python
# /services/common/caching.py
from functools import lru_cache, wraps
import asyncio
from typing import Optional, Any
import hashlib
import json

class CacheManager:
    """Multi-level caching strategy"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}

    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        ttl: int = 300
    ) -> Any:
        """Get from cache or compute and store"""

        # Check local cache first
        if key in self.local_cache:
            return self.local_cache[key]

        # Check Redis cache
        cached = await self.redis.get(key)
        if cached:
            value = json.loads(cached)
            self.local_cache[key] = value
            return value

        # Compute value
        value = await compute_func()

        # Store in both caches
        await self.redis.setex(key, ttl, json.dumps(value))
        self.local_cache[key] = value

        return value

def cache_result(ttl: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"

            # Try to get from cache
            result = await cache_manager.get_or_compute(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl
            )
            return result
        return wrapper
    return decorator
```

### 10.2 Query Optimization

```python
# /services/common/database_optimization.py
from sqlalchemy import text
from typing import List, Dict

class QueryOptimizer:
    """Database query optimization utilities"""

    @staticmethod
    async def batch_insert(connection, table: str, records: List[Dict]):
        """Batch insert with COPY for PostgreSQL"""
        if not records:
            return

        # Use COPY for large datasets
        if len(records) > 1000:
            # Convert to CSV format
            import io
            import csv

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writerows(records)
            output.seek(0)

            # Use COPY command
            await connection.copy_to_table(
                table,
                source=output,
                columns=list(records[0].keys()),
                format='csv'
            )
        else:
            # Use regular batch insert
            await connection.execute_many(
                f"INSERT INTO {table} VALUES ($1, $2, ...)",
                records
            )

    @staticmethod
    def optimize_timeseries_query(
        start_time: datetime,
        end_time: datetime,
        symbols: List[str]
    ) -> str:
        """Optimize time-series queries for TimescaleDB"""
        return text("""
            SELECT
                time_bucket('5 minutes', time) AS bucket,
                symbol,
                first(price, time) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price, time) AS close,
                sum(volume) AS volume
            FROM market_ticks
            WHERE time >= :start_time
                AND time < :end_time
                AND symbol = ANY(:symbols)
            GROUP BY bucket, symbol
            ORDER BY bucket DESC, symbol
        """)
```

---

## Summary

This comprehensive technical architecture provides:

1. **Real-Time Data Pipeline**: Sub-second latency with Kafka/Redis, TimescaleDB for storage
2. **Regime Detection**: Multi-frequency detection with state persistence and alerting
3. **Trade Execution**: Professional OMS with smart routing and position tracking
4. **Backtesting**: Walk-forward validation with comprehensive metrics
5. **Monitoring**: Prometheus + Grafana with custom dashboards and alerts
6. **Deployment**: Docker-based with CI/CD pipeline and blue-green deployment

The architecture is designed for:
- **Scalability**: Horizontal scaling of all services
- **Reliability**: Health checks, circuit breakers, and failover
- **Performance**: Caching, query optimization, and efficient data structures
- **Security**: JWT auth, API key encryption, and audit logging
- **Observability**: Comprehensive metrics and distributed tracing

This platform can handle production trading workloads while maintaining sub-second latencies and high availability.