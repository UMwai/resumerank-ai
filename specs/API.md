# Investment Intelligence Platform - API Specification

## Overview

This document specifies the APIs and interfaces for the Investment Intelligence Platform, covering signal generation, portfolio management, regime detection, and system integration.

---

## REST API

### Base URL

```
Production: https://api.investintel.io/v1
Development: http://localhost:8000/api/v1
```

### Authentication

All API requests require authentication via API key:

```bash
curl -H "Authorization: Bearer API_KEY" \
     https://api.investintel.io/v1/signals/clinical-trials
```

---

## Signal APIs

### Clinical Trial Signals

#### List Clinical Trial Signals

```http
GET /signals/clinical-trials
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| min_score | float | No | Minimum signal score (0-10) |
| phase | string | No | Trial phase filter (1, 2, 3, 4) |
| status | string | No | Trial status filter |
| limit | int | No | Results per page (default: 50) |
| offset | int | No | Pagination offset |

**Response:**

```json
{
  "signals": [
    {
      "id": "sig-001",
      "nct_id": "NCT12345678",
      "ticker": "MRNA",
      "company": "Moderna Inc",
      "trial_title": "Phase 3 mRNA Vaccine Trial",
      "signal_type": "amendment",
      "signal_score": 8.5,
      "confidence": 0.85,
      "detected_at": "2024-01-15T14:30:00Z",
      "description": "Major protocol amendment - primary endpoint modification",
      "recommendation": "BULLISH",
      "details": {
        "amendment_date": "2024-01-10",
        "changes": ["Primary endpoint modified", "Sample size increased"],
        "historical_pattern": "Similar amendments preceded positive results 78% of time"
      }
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 50
}
```

#### Get Single Trial Signal

```http
GET /signals/clinical-trials/{signal_id}
```

**Response:**

```json
{
  "id": "sig-001",
  "nct_id": "NCT12345678",
  "ticker": "MRNA",
  "company": "Moderna Inc",
  "trial_title": "Phase 3 mRNA Vaccine Trial",
  "phase": "Phase 3",
  "status": "Active, not recruiting",
  "signal_type": "amendment",
  "signal_score": 8.5,
  "confidence": 0.85,
  "detected_at": "2024-01-15T14:30:00Z",
  "description": "Major protocol amendment - primary endpoint modification",
  "recommendation": "BULLISH",
  "expected_catalyst_date": "2024-04-15",
  "options_strategy": {
    "type": "call_spread",
    "strike_1": 120,
    "strike_2": 140,
    "expiry": "2024-05-17",
    "max_risk": 500,
    "max_reward": 1500
  },
  "historical_accuracy": {
    "similar_signals": 45,
    "accurate_predictions": 35,
    "accuracy_rate": 0.78
  }
}
```

### Patent/IP Signals

#### List Patent Cliff Events

```http
GET /signals/patent-cliffs
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| months_ahead | int | No | Months until expiration (default: 18) |
| min_revenue | float | No | Minimum annual drug revenue ($M) |
| ticker | string | No | Filter by company ticker |

**Response:**

```json
{
  "signals": [
    {
      "id": "pat-001",
      "drug_name": "Lipitor",
      "ticker": "PFE",
      "company": "Pfizer Inc",
      "annual_revenue_mm": 2500,
      "patent_expiry_date": "2025-06-15",
      "months_to_expiry": 14,
      "generic_filers": ["TEVA", "MYL", "GILD"],
      "expected_revenue_loss_pct": 85,
      "signal_score": 9.2,
      "trade_recommendation": {
        "type": "pair_trade",
        "long": "TEVA",
        "short": "PFE",
        "expected_return": 0.25,
        "time_horizon_months": 6
      }
    }
  ],
  "total": 25,
  "page": 1,
  "limit": 50
}
```

### Insider Activity Signals

#### List Insider Signals

```http
GET /signals/insider-activity
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| signal_type | string | No | "form4", "13f", "hiring", "cluster" |
| min_score | float | No | Minimum signal score (0-10) |
| ticker | string | No | Filter by ticker |
| insider_type | string | No | "officer", "director", "10%_owner" |

**Response:**

```json
{
  "signals": [
    {
      "id": "ins-001",
      "ticker": "NVDA",
      "company": "NVIDIA Corporation",
      "signal_type": "cluster",
      "signal_score": 8.8,
      "detected_at": "2024-01-15T10:00:00Z",
      "description": "3 executives purchased shares within 5 days",
      "details": {
        "transactions": [
          {
            "insider_name": "Jensen Huang",
            "title": "CEO",
            "transaction_type": "P",
            "shares": 10000,
            "price": 450.25,
            "value": 4502500,
            "date": "2024-01-12"
          }
        ],
        "total_value": 12500000,
        "buyer_count": 3,
        "seller_count": 0
      },
      "recommendation": "BULLISH",
      "confidence": 0.82
    }
  ]
}
```

### Combined Signals

#### Get Combined Signal Score

```http
GET /signals/combined
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ticker | string | No | Filter by ticker |
| min_score | float | No | Minimum combined score |
| sector | string | No | Filter by sector |

**Response:**

```json
{
  "signals": [
    {
      "ticker": "MRNA",
      "company": "Moderna Inc",
      "combined_score": 8.9,
      "recommendation": "STRONG_BUY",
      "confidence": 0.88,
      "component_scores": {
        "clinical_trial": 8.5,
        "patent_ip": null,
        "insider_activity": 7.2,
        "hiring_signals": 6.8
      },
      "active_signals": 3,
      "key_catalysts": [
        {
          "type": "clinical_trial",
          "description": "Phase 3 results expected Q2 2024",
          "expected_date": "2024-05-15"
        }
      ],
      "suggested_position": {
        "direction": "LONG",
        "allocation_pct": 5,
        "entry_price": 95.50,
        "stop_loss": 85.00,
        "target_1": 120.00,
        "target_2": 140.00
      }
    }
  ]
}
```

---

## Portfolio APIs

### Get Current Positions

```http
GET /portfolio/positions
```

**Response:**

```json
{
  "positions": [
    {
      "ticker": "MRNA",
      "shares": 100,
      "avg_cost": 92.50,
      "current_price": 105.25,
      "market_value": 10525.00,
      "unrealized_pnl": 1275.00,
      "unrealized_pnl_pct": 0.138,
      "weight_pct": 10.5,
      "entry_date": "2024-01-10",
      "signal_source": "clinical_trial"
    }
  ],
  "summary": {
    "total_value": 100000.00,
    "cash": 25000.00,
    "invested": 75000.00,
    "unrealized_pnl": 5250.00,
    "unrealized_pnl_pct": 0.07,
    "position_count": 8
  }
}
```

### Get Performance Metrics

```http
GET /portfolio/performance
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| period | string | No | "1d", "1w", "1m", "3m", "ytd", "1y" |

**Response:**

```json
{
  "period": "1m",
  "metrics": {
    "total_return": 0.085,
    "annualized_return": 1.02,
    "sharpe_ratio": 2.15,
    "sortino_ratio": 2.85,
    "max_drawdown": -0.045,
    "volatility": 0.12,
    "beta": 0.85,
    "alpha": 0.15,
    "win_rate": 0.68,
    "profit_factor": 2.3
  },
  "benchmark_comparison": {
    "spy_return": 0.032,
    "alpha_vs_spy": 0.053,
    "correlation_to_spy": 0.65
  },
  "attribution": {
    "clinical_trial_signals": 0.045,
    "patent_cliff_trades": 0.025,
    "insider_activity": 0.015
  }
}
```

### Trigger Rebalance

```http
POST /portfolio/rebalance
```

**Request Body:**

```json
{
  "target_allocations": {
    "MRNA": 0.10,
    "NVDA": 0.08,
    "CASH": 0.25
  },
  "reason": "regime_change",
  "dry_run": false
}
```

**Response:**

```json
{
  "rebalance_id": "reb-001",
  "status": "executed",
  "trades": [
    {
      "ticker": "MRNA",
      "action": "BUY",
      "shares": 25,
      "price": 105.25,
      "value": 2631.25
    },
    {
      "ticker": "NVDA",
      "action": "SELL",
      "shares": 10,
      "price": 520.00,
      "value": 5200.00
    }
  ],
  "new_allocations": {
    "MRNA": 0.10,
    "NVDA": 0.08,
    "CASH": 0.28
  },
  "executed_at": "2024-01-15T14:30:00Z"
}
```

---

## Regime APIs

### Get Current Regime

```http
GET /regime/current
```

**Response:**

```json
{
  "regime": "BULL",
  "confidence": 0.85,
  "indicators": {
    "sma_50": 4850.25,
    "sma_200": 4520.10,
    "vix": 14.5,
    "trend": "up",
    "volatility": "low"
  },
  "detected_at": "2024-01-15T09:30:00Z",
  "duration_days": 45,
  "allocation_adjustments": {
    "equity": 0.80,
    "bonds": 0.10,
    "cash": 0.10
  }
}
```

### Get Regime History

```http
GET /regime/history
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| start_date | string | No | Start date (ISO format) |
| end_date | string | No | End date (ISO format) |

**Response:**

```json
{
  "regimes": [
    {
      "regime": "BULL",
      "start_date": "2023-11-01",
      "end_date": null,
      "duration_days": 76,
      "performance": {
        "portfolio_return": 0.18,
        "benchmark_return": 0.12
      }
    },
    {
      "regime": "SIDEWAYS",
      "start_date": "2023-08-15",
      "end_date": "2023-10-31",
      "duration_days": 77,
      "performance": {
        "portfolio_return": 0.02,
        "benchmark_return": -0.01
      }
    }
  ]
}
```

---

## Alert APIs

### List Alerts

```http
GET /alerts
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| status | string | No | "active", "acknowledged", "all" |
| priority | string | No | "high", "medium", "low" |
| type | string | No | "signal", "risk", "system" |

**Response:**

```json
{
  "alerts": [
    {
      "id": "alert-001",
      "type": "signal",
      "priority": "high",
      "title": "High-Confidence Clinical Trial Signal",
      "message": "MRNA Phase 3 amendment detected - score 8.5",
      "ticker": "MRNA",
      "created_at": "2024-01-15T14:30:00Z",
      "acknowledged": false,
      "action_url": "/signals/clinical-trials/sig-001"
    },
    {
      "id": "alert-002",
      "type": "risk",
      "priority": "high",
      "title": "Portfolio Drawdown Warning",
      "message": "Daily drawdown exceeded 3% threshold",
      "created_at": "2024-01-15T15:45:00Z",
      "acknowledged": false,
      "current_value": -0.035,
      "threshold": -0.03
    }
  ]
}
```

### Acknowledge Alert

```http
POST /alerts/{alert_id}/acknowledge
```

**Response:**

```json
{
  "id": "alert-001",
  "acknowledged": true,
  "acknowledged_at": "2024-01-15T16:00:00Z",
  "acknowledged_by": "user@example.com"
}
```

---

## Backtest API

### Run Backtest

```http
POST /backtest/run
```

**Request Body:**

```json
{
  "strategy": "regime_adaptive",
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 100000,
  "parameters": {
    "max_position_pct": 0.10,
    "stop_loss_pct": 0.08,
    "rebalance_frequency": "monthly"
  }
}
```

**Response:**

```json
{
  "backtest_id": "bt-001",
  "status": "completed",
  "results": {
    "total_return": 1.85,
    "annualized_return": 0.28,
    "sharpe_ratio": 1.65,
    "sortino_ratio": 2.12,
    "max_drawdown": -0.18,
    "calmar_ratio": 1.55,
    "win_rate": 0.62,
    "profit_factor": 1.85,
    "total_trades": 156,
    "avg_trade_return": 0.012
  },
  "benchmark_comparison": {
    "spy_return": 0.95,
    "alpha": 0.90,
    "beta": 0.75,
    "correlation": 0.68
  },
  "monthly_returns": [...],
  "drawdown_series": [...],
  "trade_log_url": "/backtest/bt-001/trades"
}
```

---

## Python SDK

### Installation

```bash
pip install investintel-sdk
```

### Usage

```python
from investintel import InvestIntelClient

# Initialize client
client = InvestIntelClient(api_key="YOUR_API_KEY")

# Get clinical trial signals
signals = client.signals.clinical_trials(
    min_score=7.0,
    phase="3"
)

for signal in signals:
    print(f"{signal.ticker}: {signal.signal_score} - {signal.description}")

# Get combined signals for a ticker
combined = client.signals.combined(ticker="MRNA")
print(f"Combined score: {combined.combined_score}")
print(f"Recommendation: {combined.recommendation}")

# Get current portfolio
portfolio = client.portfolio.positions()
print(f"Total value: ${portfolio.summary.total_value:,.2f}")
print(f"Unrealized P&L: {portfolio.summary.unrealized_pnl_pct:.1%}")

# Get current regime
regime = client.regime.current()
print(f"Current regime: {regime.regime} (confidence: {regime.confidence:.0%})")

# Run backtest
result = client.backtest.run(
    strategy="regime_adaptive",
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_capital=100000
)
print(f"Sharpe ratio: {result.results.sharpe_ratio:.2f}")
```

---

## CLI Interface

### Signal Commands

```bash
# List clinical trial signals
investintel signals clinical --min-score 7

# List patent cliff events
investintel signals patents --months-ahead 12

# Get combined signal for ticker
investintel signals combined --ticker MRNA

# Export signals to CSV
investintel signals export --output signals.csv
```

### Portfolio Commands

```bash
# View current positions
investintel portfolio positions

# View performance
investintel portfolio performance --period 1m

# Trigger rebalance
investintel portfolio rebalance --dry-run
```

### Backtest Commands

```bash
# Run backtest
investintel backtest run --strategy regime_adaptive \
    --start 2020-01-01 --end 2024-01-01 --capital 100000

# View backtest results
investintel backtest results --id bt-001

# Export trade log
investintel backtest trades --id bt-001 --output trades.csv
```

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.investintel.io/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    api_key: 'YOUR_API_KEY'
  }));

  // Subscribe to channels
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['signals', 'portfolio', 'regime']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Message Types

#### Signal Update

```json
{
  "type": "signal",
  "channel": "signals",
  "data": {
    "id": "sig-001",
    "signal_type": "clinical_trial",
    "ticker": "MRNA",
    "score": 8.5,
    "description": "New high-confidence signal detected"
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### Portfolio Update

```json
{
  "type": "portfolio_update",
  "channel": "portfolio",
  "data": {
    "total_value": 105250.00,
    "daily_pnl": 1250.00,
    "daily_pnl_pct": 0.012
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### Regime Change

```json
{
  "type": "regime_change",
  "channel": "regime",
  "data": {
    "old_regime": "SIDEWAYS",
    "new_regime": "BULL",
    "confidence": 0.85,
    "recommended_action": "Increase equity allocation to 80%"
  },
  "timestamp": "2024-01-15T09:30:00Z"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter value",
    "details": {
      "field": "min_score",
      "value": "abc",
      "expected": "float between 0 and 10"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| UNAUTHORIZED | 401 | Invalid or missing API key |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 400 | Invalid request parameters |
| RATE_LIMITED | 429 | Too many requests |
| SERVER_ERROR | 500 | Internal server error |

---

## Rate Limits

| Tier | Requests/Minute | Requests/Day |
|------|-----------------|--------------|
| Free | 10 | 100 |
| Starter | 60 | 1000 |
| Professional | 300 | 10000 |
| Enterprise | Unlimited | Unlimited |

Rate limit headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705334400
```
