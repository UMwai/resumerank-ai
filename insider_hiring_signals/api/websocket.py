"""
WebSocket Real-time Alerts API

Provides real-time alert delivery via WebSocket for:
- Form 4 insider transactions
- Pattern detection alerts
- High-confidence signal notifications
- Slack integration
- Rate limiting and deduplication
"""

import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import WebSocket dependencies
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.warning("websockets package not installed")
    WEBSOCKETS_AVAILABLE = False

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("fastapi package not installed")
    FASTAPI_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class AlertType(Enum):
    """Types of real-time alerts."""
    FORM4_FILING = "form4_filing"
    PATTERN_DETECTED = "pattern_detected"
    HIGH_CONFIDENCE_SIGNAL = "high_confidence_signal"
    PRICE_ALERT = "price_alert"
    INSTITUTIONAL_UPDATE = "institutional_update"
    EXECUTIVE_CHANGE = "executive_change"


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class Alert:
    """Real-time alert message."""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    ticker: str
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'priority': self.priority.value,
            'ticker': self.ticker,
            'title': self.title,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'Alert':
        """Create Alert from dictionary."""
        return cls(
            alert_id=data['alert_id'],
            alert_type=AlertType(data['alert_type']),
            priority=AlertPriority(data['priority']),
            ticker=data['ticker'],
            title=data['title'],
            message=data['message'],
            details=data.get('details', {}),
            timestamp=datetime.fromisoformat(data['timestamp']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
        )


class AlertDeduplicator:
    """Prevents duplicate alerts within a time window."""

    def __init__(self, window_seconds: int = 300):
        """Initialize with deduplication window."""
        self.window = timedelta(seconds=window_seconds)
        self._seen: Dict[str, datetime] = {}

    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate."""
        # Generate content hash
        content = f"{alert.alert_type.value}_{alert.ticker}_{alert.title}"
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

        now = datetime.now()

        # Clean old entries
        self._seen = {
            k: v for k, v in self._seen.items()
            if now - v < self.window
        }

        # Check if seen recently
        if content_hash in self._seen:
            return True

        # Mark as seen
        self._seen[content_hash] = now
        return False


class RateLimiter:
    """Rate limits alerts per client and globally."""

    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_client_minute: int = 20
    ):
        self.max_per_minute = max_per_minute
        self.max_per_client = max_per_client_minute
        self._global_counts: List[datetime] = []
        self._client_counts: Dict[str, List[datetime]] = defaultdict(list)

    def check_limit(self, client_id: Optional[str] = None) -> bool:
        """Check if rate limit allows sending."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self._global_counts = [t for t in self._global_counts if t > minute_ago]

        if len(self._global_counts) >= self.max_per_minute:
            return False

        if client_id:
            self._client_counts[client_id] = [
                t for t in self._client_counts[client_id] if t > minute_ago
            ]
            if len(self._client_counts[client_id]) >= self.max_per_client:
                return False

        return True

    def record_send(self, client_id: Optional[str] = None):
        """Record a sent alert."""
        now = datetime.now()
        self._global_counts.append(now)
        if client_id:
            self._client_counts[client_id].append(now)


class AlertSubscription:
    """Client subscription preferences."""

    def __init__(
        self,
        client_id: str,
        tickers: Optional[Set[str]] = None,
        alert_types: Optional[Set[AlertType]] = None,
        min_priority: AlertPriority = AlertPriority.MEDIUM
    ):
        self.client_id = client_id
        self.tickers = tickers  # None means all tickers
        self.alert_types = alert_types  # None means all types
        self.min_priority = min_priority
        self.created_at = datetime.now()

    def matches(self, alert: Alert) -> bool:
        """Check if alert matches subscription."""
        # Priority check
        if alert.priority.value > self.min_priority.value:
            return False

        # Ticker filter
        if self.tickers and alert.ticker not in self.tickers:
            return False

        # Alert type filter
        if self.alert_types and alert.alert_type not in self.alert_types:
            return False

        return True


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Any] = {}  # client_id -> websocket
        self.subscriptions: Dict[str, AlertSubscription] = {}

    async def connect(self, client_id: str, websocket: Any):
        """Accept and register a new connection."""
        self.active_connections[client_id] = websocket
        # Default subscription
        self.subscriptions[client_id] = AlertSubscription(client_id)
        logger.info(f"Client connected: {client_id}")

    def disconnect(self, client_id: str):
        """Remove a disconnected client."""
        self.active_connections.pop(client_id, None)
        self.subscriptions.pop(client_id, None)
        logger.info(f"Client disconnected: {client_id}")

    def update_subscription(
        self,
        client_id: str,
        tickers: Optional[Set[str]] = None,
        alert_types: Optional[Set[AlertType]] = None,
        min_priority: Optional[AlertPriority] = None
    ):
        """Update client subscription."""
        if client_id not in self.subscriptions:
            return

        sub = self.subscriptions[client_id]
        if tickers is not None:
            sub.tickers = tickers
        if alert_types is not None:
            sub.alert_types = alert_types
        if min_priority is not None:
            sub.min_priority = min_priority

    async def broadcast(self, alert: Alert) -> int:
        """Broadcast alert to all matching subscribers."""
        sent_count = 0

        for client_id, websocket in list(self.active_connections.items()):
            subscription = self.subscriptions.get(client_id)

            if subscription and subscription.matches(alert):
                try:
                    await websocket.send(alert.to_json())
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Failed to send to {client_id}: {e}")
                    self.disconnect(client_id)

        return sent_count

    def get_connected_count(self) -> int:
        """Get number of connected clients."""
        return len(self.active_connections)


class AlertWebSocketServer:
    """
    WebSocket server for real-time alerts.

    Features:
    - Real-time Form 4 alerts
    - Pattern detection notifications
    - High-confidence signal alerts
    - Subscription management
    - Rate limiting and deduplication
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        config_path: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.config = get_config(config_path) if config_path else None

        self.connection_manager = ConnectionManager()
        self.deduplicator = AlertDeduplicator()
        self.rate_limiter = RateLimiter()

        self._alert_queue: asyncio.Queue = None
        self._running = False

    async def start(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets package not installed")
            return

        self._alert_queue = asyncio.Queue()
        self._running = True

        # Start alert processor
        asyncio.create_task(self._process_alerts())

        # Start server
        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port
        ):
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def stop(self):
        """Stop the server."""
        self._running = False
        logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket: Any, path: str = ""):
        """Handle a new WebSocket connection."""
        # Generate client ID
        client_id = f"client_{id(websocket)}"

        try:
            await self.connection_manager.connect(client_id, websocket)

            # Send welcome message
            welcome = {
                'type': 'welcome',
                'client_id': client_id,
                'message': 'Connected to Insider/Hiring Signals Alert Server',
                'timestamp': datetime.now().isoformat(),
            }
            await websocket.send(json.dumps(welcome))

            # Handle messages
            async for message in websocket:
                await self._handle_message(client_id, message)

        except Exception as e:
            logger.error(f"Connection error for {client_id}: {e}")
        finally:
            self.connection_manager.disconnect(client_id)

    async def _handle_message(self, client_id: str, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'subscribe':
                # Update subscription
                tickers = set(data.get('tickers', [])) if data.get('tickers') else None
                alert_types = None
                if data.get('alert_types'):
                    alert_types = {AlertType(t) for t in data['alert_types']}

                min_priority = AlertPriority.MEDIUM
                if data.get('min_priority'):
                    min_priority = AlertPriority(data['min_priority'])

                self.connection_manager.update_subscription(
                    client_id, tickers, alert_types, min_priority
                )

                logger.info(f"Updated subscription for {client_id}")

            elif msg_type == 'ping':
                # Respond to ping
                ws = self.connection_manager.active_connections.get(client_id)
                if ws:
                    await ws.send(json.dumps({'type': 'pong'}))

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {client_id}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")

    async def _process_alerts(self):
        """Process queued alerts."""
        while self._running:
            try:
                # Wait for alert with timeout
                alert = await asyncio.wait_for(
                    self._alert_queue.get(),
                    timeout=1.0
                )

                # Check deduplication
                if self.deduplicator.is_duplicate(alert):
                    continue

                # Check rate limit
                if not self.rate_limiter.check_limit():
                    logger.warning("Global rate limit reached")
                    continue

                # Broadcast
                sent_count = await self.connection_manager.broadcast(alert)
                self.rate_limiter.record_send()

                logger.debug(f"Alert sent to {sent_count} clients: {alert.title}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    async def send_alert(self, alert: Alert):
        """Queue an alert for sending."""
        if self._alert_queue:
            await self._alert_queue.put(alert)

    def create_form4_alert(
        self,
        ticker: str,
        insider_name: str,
        insider_title: str,
        transaction_type: str,
        shares: int,
        value: float,
        is_10b5_1: bool = False
    ) -> Alert:
        """Create a Form 4 filing alert."""
        # Determine priority
        is_c_suite = any(t in insider_title.upper() for t in ['CEO', 'CFO', 'CMO', 'COO'])
        is_purchase = transaction_type.lower() in ['purchase', 'buy']

        if is_c_suite and is_purchase and value > 100000:
            priority = AlertPriority.HIGH
        elif is_c_suite:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW

        # Reduce priority for 10b5-1 plans
        if is_10b5_1:
            priority = AlertPriority(min(priority.value + 1, 4))

        title = f"Form 4: {insider_name} ({ticker})"
        message = (
            f"{insider_title} {transaction_type.lower()} "
            f"{shares:,} shares (${value:,.0f})"
        )

        if is_10b5_1:
            message += " [10b5-1 plan]"

        return Alert(
            alert_id=f"f4_{ticker}_{datetime.now().timestamp():.0f}",
            alert_type=AlertType.FORM4_FILING,
            priority=priority,
            ticker=ticker,
            title=title,
            message=message,
            details={
                'insider_name': insider_name,
                'insider_title': insider_title,
                'transaction_type': transaction_type,
                'shares': shares,
                'value': value,
                'is_10b5_1': is_10b5_1,
            },
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )

    def create_pattern_alert(
        self,
        ticker: str,
        pattern_type: str,
        description: str,
        signal_strength: int,
        details: Dict
    ) -> Alert:
        """Create a pattern detection alert."""
        # Determine priority based on signal strength
        if abs(signal_strength) >= 8:
            priority = AlertPriority.CRITICAL
        elif abs(signal_strength) >= 6:
            priority = AlertPriority.HIGH
        elif abs(signal_strength) >= 4:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW

        direction = "Bullish" if signal_strength > 0 else "Bearish"
        title = f"{direction} Pattern: {pattern_type} ({ticker})"

        return Alert(
            alert_id=f"pat_{ticker}_{datetime.now().timestamp():.0f}",
            alert_type=AlertType.PATTERN_DETECTED,
            priority=priority,
            ticker=ticker,
            title=title,
            message=description,
            details={
                'pattern_type': pattern_type,
                'signal_strength': signal_strength,
                **details
            },
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7)
        )

    def create_signal_alert(
        self,
        ticker: str,
        score: float,
        confidence: float,
        recommendation: str,
        contributing_signals: List[str]
    ) -> Alert:
        """Create a high-confidence signal alert."""
        if score >= 6 and confidence >= 0.7:
            priority = AlertPriority.HIGH
            title = f"Strong Buy Signal: {ticker}"
        elif score <= -6 and confidence >= 0.7:
            priority = AlertPriority.HIGH
            title = f"Strong Sell Signal: {ticker}"
        elif abs(score) >= 4 and confidence >= 0.6:
            priority = AlertPriority.MEDIUM
            direction = "Buy" if score > 0 else "Sell"
            title = f"{direction} Signal: {ticker}"
        else:
            priority = AlertPriority.LOW
            title = f"Signal Update: {ticker}"

        return Alert(
            alert_id=f"sig_{ticker}_{datetime.now().timestamp():.0f}",
            alert_type=AlertType.HIGH_CONFIDENCE_SIGNAL,
            priority=priority,
            ticker=ticker,
            title=title,
            message=f"Score: {score:+.1f}, Confidence: {confidence:.0%}. {recommendation}",
            details={
                'score': score,
                'confidence': confidence,
                'recommendation': recommendation,
                'contributing_signals': contributing_signals,
            },
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(days=1)
        )


class SlackNotifier:
    """
    Sends alerts to Slack channels.

    Integrates with Slack webhooks for team notifications.
    """

    def __init__(self, webhook_url: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize Slack notifier."""
        self.config = get_config(config_path) if config_path else None

        self.webhook_url = webhook_url
        if not self.webhook_url and self.config:
            self.webhook_url = self.config.get_nested('slack', 'webhook_url')

        if not self.webhook_url:
            self.webhook_url = os.environ.get('SLACK_WEBHOOK_URL')

    def _format_alert(self, alert: Alert) -> Dict:
        """Format alert as Slack message."""
        # Color based on priority
        colors = {
            AlertPriority.CRITICAL: '#FF0000',
            AlertPriority.HIGH: '#FF9900',
            AlertPriority.MEDIUM: '#FFCC00',
            AlertPriority.LOW: '#36A64F',
        }

        # Icon based on alert type
        icons = {
            AlertType.FORM4_FILING: ':chart_with_upwards_trend:',
            AlertType.PATTERN_DETECTED: ':warning:',
            AlertType.HIGH_CONFIDENCE_SIGNAL: ':dart:',
            AlertType.PRICE_ALERT: ':moneybag:',
            AlertType.INSTITUTIONAL_UPDATE: ':bank:',
            AlertType.EXECUTIVE_CHANGE: ':bust_in_silhouette:',
        }

        icon = icons.get(alert.alert_type, ':bell:')
        color = colors.get(alert.priority, '#808080')

        return {
            'attachments': [{
                'color': color,
                'fallback': f"{alert.title}: {alert.message}",
                'title': f"{icon} {alert.title}",
                'text': alert.message,
                'fields': [
                    {
                        'title': 'Ticker',
                        'value': alert.ticker,
                        'short': True,
                    },
                    {
                        'title': 'Priority',
                        'value': alert.priority.name,
                        'short': True,
                    },
                ],
                'footer': 'Insider/Hiring Signals',
                'ts': int(alert.timestamp.timestamp()),
            }]
        }

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not self.webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed for Slack notifications")
            return False

        try:
            payload = self._format_alert(alert)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0
                )

            if response.status_code == 200:
                logger.debug(f"Sent Slack alert: {alert.title}")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def send_alert_sync(self, alert: Alert) -> bool:
        """Synchronous version of send_alert."""
        import requests

        if not self.webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False

        try:
            payload = self._format_alert(alert)

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10.0
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


# FastAPI application for REST + WebSocket
def create_app() -> Optional[Any]:
    """Create FastAPI application with WebSocket endpoint."""
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available")
        return None

    app = FastAPI(
        title="Insider/Hiring Signals Alert API",
        description="Real-time alerts for insider activity and hiring signals",
        version="2.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Connection manager
    manager = ConnectionManager()
    deduplicator = AlertDeduplicator()
    rate_limiter = RateLimiter()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "connected_clients": manager.get_connected_count(),
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/alerts/recent")
    async def get_recent_alerts():
        """Get recent alerts (placeholder)."""
        return {"alerts": [], "count": 0}

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """WebSocket endpoint for real-time alerts."""
        await websocket.accept()
        await manager.connect(client_id, websocket)

        try:
            # Send welcome
            await websocket.send_json({
                "type": "welcome",
                "client_id": client_id,
                "message": "Connected to alerts",
            })

            while True:
                # Receive messages
                data = await websocket.receive_json()

                if data.get("type") == "subscribe":
                    tickers = set(data.get("tickers", [])) if data.get("tickers") else None
                    manager.update_subscription(client_id, tickers=tickers)

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(client_id)

    return app


if __name__ == '__main__':
    # Test alert creation
    server = AlertWebSocketServer()

    # Create test alerts
    form4_alert = server.create_form4_alert(
        ticker='MRNA',
        insider_name='Stephane Bancel',
        insider_title='CEO',
        transaction_type='Purchase',
        shares=10000,
        value=250000
    )

    pattern_alert = server.create_pattern_alert(
        ticker='MRNA',
        pattern_type='insider_cluster_buy',
        description='3 insiders bought $500K in 30 days',
        signal_strength=8,
        details={'insider_count': 3}
    )

    signal_alert = server.create_signal_alert(
        ticker='MRNA',
        score=7.5,
        confidence=0.85,
        recommendation='Strong Buy: Multiple bullish signals converging',
        contributing_signals=['CEO purchase', 'Commercial hiring surge', 'Fund accumulation']
    )

    print("Test Alerts Created:")
    print("=" * 50)
    print(f"\nForm 4 Alert:")
    print(json.dumps(form4_alert.to_dict(), indent=2))
    print(f"\nPattern Alert:")
    print(json.dumps(pattern_alert.to_dict(), indent=2))
    print(f"\nSignal Alert:")
    print(json.dumps(signal_alert.to_dict(), indent=2))

    # Test Slack formatting
    slack = SlackNotifier()
    print(f"\nSlack Message Format:")
    print(json.dumps(slack._format_alert(form4_alert), indent=2))
