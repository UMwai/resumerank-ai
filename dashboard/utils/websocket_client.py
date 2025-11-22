"""
WebSocket Client for Real-Time Signal Updates

Provides WebSocket connectivity to Clinical Trial, Patent, and Insider APIs
with automatic reconnection using exponential backoff.

Features:
- Connection pooling for multiple WebSocket endpoints
- Exponential backoff reconnection strategy
- Connection status monitoring
- Message queuing for reliability
- Thread-safe signal broadcasting
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Set
import random

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class Signal:
    """Represents a real-time signal from any data source."""
    id: str
    source: str  # 'clinical_trial', 'patent', 'insider'
    signal_type: str  # 'bullish', 'bearish', 'info'
    ticker: str
    company_name: str
    title: str
    description: str
    score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "signal_type": self.signal_type,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "title": self.title,
            "description": self.description,
            "score": self.score,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create signal from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=data.get("id", ""),
            source=data.get("source", "unknown"),
            signal_type=data.get("signal_type", "info"),
            ticker=data.get("ticker", ""),
            company_name=data.get("company_name", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            score=float(data.get("score", 0)),
            confidence=float(data.get("confidence", 0)),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConnectionStatus:
    """Status of a WebSocket connection."""
    endpoint: str
    state: ConnectionState
    last_connected: Optional[datetime] = None
    last_message: Optional[datetime] = None
    reconnect_attempts: int = 0
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None


class ReconnectionStrategy:
    """
    Exponential backoff reconnection strategy.

    Implements jittered exponential backoff to prevent thundering herd.
    """

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
        max_attempts: int = 10,
    ):
        """
        Initialize reconnection strategy.

        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Delay multiplier per attempt
            jitter: Jitter factor (0-1) to randomize delay
            max_attempts: Maximum reconnection attempts (0 for unlimited)
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.max_attempts = max_attempts
        self._attempt = 0

    def reset(self) -> None:
        """Reset attempt counter."""
        self._attempt = 0

    def get_delay(self) -> float:
        """
        Get next reconnection delay.

        Returns:
            Delay in seconds with jitter applied
        """
        delay = min(
            self.initial_delay * (self.multiplier ** self._attempt),
            self.max_delay
        )

        # Apply jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        self._attempt += 1
        return max(0, delay)

    def should_retry(self) -> bool:
        """Check if another retry should be attempted."""
        if self.max_attempts == 0:
            return True
        return self._attempt < self.max_attempts

    @property
    def attempts(self) -> int:
        """Get current attempt count."""
        return self._attempt


class BaseWebSocketClient(ABC):
    """
    Abstract base class for WebSocket clients.

    Provides common functionality for connection management,
    message handling, and reconnection logic.
    """

    def __init__(
        self,
        endpoint: str,
        name: str,
        reconnection_strategy: Optional[ReconnectionStrategy] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            endpoint: WebSocket endpoint URL
            name: Client name for logging
            reconnection_strategy: Custom reconnection strategy
        """
        self.endpoint = endpoint
        self.name = name
        self.strategy = reconnection_strategy or ReconnectionStrategy()

        self._state = ConnectionState.DISCONNECTED
        self._websocket = None
        self._message_queue: Queue = Queue()
        self._callbacks: List[Callable[[Signal], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._last_connected: Optional[datetime] = None
        self._last_message: Optional[datetime] = None
        self._error_message: Optional[str] = None
        self._latency_ms: Optional[float] = None

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    def get_status(self) -> ConnectionStatus:
        """Get connection status."""
        return ConnectionStatus(
            endpoint=self.endpoint,
            state=self._state,
            last_connected=self._last_connected,
            last_message=self._last_message,
            reconnect_attempts=self.strategy.attempts,
            error_message=self._error_message,
            latency_ms=self._latency_ms,
        )

    def subscribe(self, callback: Callable[[Signal], None]) -> None:
        """
        Subscribe to signal updates.

        Args:
            callback: Function to call when signal is received
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[Signal], None]) -> None:
        """
        Unsubscribe from signal updates.

        Args:
            callback: Function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_subscribers(self, signal: Signal) -> None:
        """Notify all subscribers of a new signal."""
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")

    @abstractmethod
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """Close WebSocket connection."""
        pass

    @abstractmethod
    async def _receive_messages(self) -> None:
        """Receive and process messages."""
        pass

    @abstractmethod
    def _parse_message(self, message: str) -> Optional[Signal]:
        """Parse raw message into Signal."""
        pass

    async def _run(self) -> None:
        """Main connection loop with reconnection logic."""
        self._running = True

        while self._running:
            try:
                self._state = ConnectionState.CONNECTING
                logger.info(f"[{self.name}] Connecting to {self.endpoint}")

                await self._connect()

                self._state = ConnectionState.CONNECTED
                self._last_connected = datetime.now()
                self._error_message = None
                self.strategy.reset()

                logger.info(f"[{self.name}] Connected successfully")

                await self._receive_messages()

            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Connection cancelled")
                break

            except Exception as e:
                self._state = ConnectionState.ERROR
                self._error_message = str(e)
                logger.error(f"[{self.name}] Connection error: {e}")

                if not self._running:
                    break

                if self.strategy.should_retry():
                    delay = self.strategy.get_delay()
                    self._state = ConnectionState.RECONNECTING
                    logger.info(
                        f"[{self.name}] Reconnecting in {delay:.1f}s "
                        f"(attempt {self.strategy.attempts})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[{self.name}] Max reconnection attempts reached"
                    )
                    break

            finally:
                try:
                    await self._disconnect()
                except Exception as e:
                    logger.warning(f"[{self.name}] Error during disconnect: {e}")

        self._state = ConnectionState.CLOSED

    def start(self) -> None:
        """Start WebSocket client in background thread."""
        if self._running:
            return

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run())
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop WebSocket client."""
        self._running = False

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)


class MockWebSocketClient(BaseWebSocketClient):
    """
    Mock WebSocket client for demo mode.

    Generates simulated signals at random intervals.
    """

    def __init__(
        self,
        endpoint: str = "ws://mock/signals",
        name: str = "MockSignals",
        signal_interval: float = 10.0,
    ):
        """
        Initialize mock WebSocket client.

        Args:
            endpoint: Mock endpoint URL
            name: Client name
            signal_interval: Average interval between signals in seconds
        """
        super().__init__(endpoint, name)
        self.signal_interval = signal_interval
        self._signal_counter = 0

    async def _connect(self) -> None:
        """Simulate connection."""
        await asyncio.sleep(0.5)  # Simulate connection delay

    async def _disconnect(self) -> None:
        """Simulate disconnection."""
        pass

    async def _receive_messages(self) -> None:
        """Generate mock signals periodically."""
        while self._running:
            # Random delay around the interval
            delay = self.signal_interval * (0.5 + random.random())
            await asyncio.sleep(delay)

            signal = self._generate_mock_signal()
            if signal:
                self._last_message = datetime.now()
                self._notify_subscribers(signal)

    def _parse_message(self, message: str) -> Optional[Signal]:
        """Parse message (not used in mock)."""
        return None

    def _generate_mock_signal(self) -> Signal:
        """Generate a random mock signal."""
        self._signal_counter += 1

        sources = ["clinical_trial", "patent", "insider"]
        source = random.choice(sources)

        tickers = ["MRNA", "VRTX", "ABBV", "PFE", "GILD", "REGN", "BIIB", "AMGN"]
        ticker = random.choice(tickers)

        companies = {
            "MRNA": "Moderna Inc.",
            "VRTX": "Vertex Pharmaceuticals",
            "ABBV": "AbbVie Inc.",
            "PFE": "Pfizer Inc.",
            "GILD": "Gilead Sciences",
            "REGN": "Regeneron Pharmaceuticals",
            "BIIB": "Biogen Inc.",
            "AMGN": "Amgen Inc.",
        }

        score = random.uniform(0.2, 0.95)
        signal_type = "bullish" if score >= 0.5 else "bearish"

        titles = {
            "clinical_trial": [
                "Phase 3 Trial Results Announced",
                "New Clinical Trial Initiated",
                "Trial Enrollment Complete",
                "FDA Breakthrough Designation",
                "Positive Interim Results",
            ],
            "patent": [
                "Key Patent Expiring Soon",
                "New Patent Filed",
                "Patent Challenge Filed",
                "IP Portfolio Expansion",
                "Generic Entry Expected",
            ],
            "insider": [
                "Large Insider Purchase",
                "CEO Increases Stake",
                "Major Hiring Activity",
                "Executive Compensation Change",
                "Board Changes Announced",
            ],
        }

        title = random.choice(titles.get(source, ["New Signal"]))

        return Signal(
            id=f"mock_{self._signal_counter}_{int(time.time())}",
            source=source,
            signal_type=signal_type,
            ticker=ticker,
            company_name=companies.get(ticker, ticker),
            title=title,
            description=f"{title} detected for {companies.get(ticker, ticker)}",
            score=round(score, 3),
            confidence=round(random.uniform(0.6, 0.95), 3),
            timestamp=datetime.now(),
            metadata={
                "generated": True,
                "demo_mode": True,
            },
        )


class ClinicalTrialWebSocketClient(BaseWebSocketClient):
    """WebSocket client for Clinical Trial signals."""

    def __init__(self, endpoint: str):
        """
        Initialize Clinical Trial WebSocket client.

        Args:
            endpoint: WebSocket endpoint URL
        """
        super().__init__(endpoint, "ClinicalTrials")

    async def _connect(self) -> None:
        """Connect to Clinical Trial WebSocket."""
        try:
            import websockets
            self._websocket = await websockets.connect(
                self.endpoint,
                ping_interval=30,
                ping_timeout=10,
            )
        except ImportError:
            raise ImportError("websockets package required for WebSocket support")

    async def _disconnect(self) -> None:
        """Disconnect from Clinical Trial WebSocket."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def _receive_messages(self) -> None:
        """Receive messages from Clinical Trial WebSocket."""
        import websockets

        async for message in self._websocket:
            self._last_message = datetime.now()
            signal = self._parse_message(message)
            if signal:
                self._notify_subscribers(signal)

    def _parse_message(self, message: str) -> Optional[Signal]:
        """Parse Clinical Trial message."""
        try:
            data = json.loads(message)

            return Signal(
                id=data.get("trial_id", ""),
                source="clinical_trial",
                signal_type="bullish" if data.get("positive", False) else "bearish",
                ticker=data.get("company_ticker", ""),
                company_name=data.get("company_name", ""),
                title=data.get("event_type", "Clinical Trial Update"),
                description=data.get("description", ""),
                score=float(data.get("score", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                timestamp=datetime.fromisoformat(data["timestamp"])
                    if "timestamp" in data else datetime.now(),
                metadata={
                    "phase": data.get("phase"),
                    "indication": data.get("indication"),
                    "drug_name": data.get("drug_name"),
                },
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse clinical trial message: {e}")
            return None


class PatentWebSocketClient(BaseWebSocketClient):
    """WebSocket client for Patent Intelligence signals."""

    def __init__(self, endpoint: str):
        """
        Initialize Patent WebSocket client.

        Args:
            endpoint: WebSocket endpoint URL
        """
        super().__init__(endpoint, "PatentIntel")

    async def _connect(self) -> None:
        """Connect to Patent WebSocket."""
        try:
            import websockets
            self._websocket = await websockets.connect(
                self.endpoint,
                ping_interval=30,
                ping_timeout=10,
            )
        except ImportError:
            raise ImportError("websockets package required for WebSocket support")

    async def _disconnect(self) -> None:
        """Disconnect from Patent WebSocket."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def _receive_messages(self) -> None:
        """Receive messages from Patent WebSocket."""
        import websockets

        async for message in self._websocket:
            self._last_message = datetime.now()
            signal = self._parse_message(message)
            if signal:
                self._notify_subscribers(signal)

    def _parse_message(self, message: str) -> Optional[Signal]:
        """Parse Patent Intelligence message."""
        try:
            data = json.loads(message)

            return Signal(
                id=data.get("patent_id", ""),
                source="patent",
                signal_type=data.get("signal_type", "info"),
                ticker=data.get("company_ticker", ""),
                company_name=data.get("company_name", ""),
                title=data.get("event_type", "Patent Update"),
                description=data.get("description", ""),
                score=float(data.get("score", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                timestamp=datetime.fromisoformat(data["timestamp"])
                    if "timestamp" in data else datetime.now(),
                metadata={
                    "drug_name": data.get("drug_name"),
                    "expiration_date": data.get("expiration_date"),
                    "revenue_at_risk": data.get("revenue_at_risk"),
                },
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse patent message: {e}")
            return None


class InsiderWebSocketClient(BaseWebSocketClient):
    """WebSocket client for Insider/Hiring signals."""

    def __init__(self, endpoint: str):
        """
        Initialize Insider WebSocket client.

        Args:
            endpoint: WebSocket endpoint URL
        """
        super().__init__(endpoint, "InsiderHiring")

    async def _connect(self) -> None:
        """Connect to Insider WebSocket."""
        try:
            import websockets
            self._websocket = await websockets.connect(
                self.endpoint,
                ping_interval=30,
                ping_timeout=10,
            )
        except ImportError:
            raise ImportError("websockets package required for WebSocket support")

    async def _disconnect(self) -> None:
        """Disconnect from Insider WebSocket."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def _receive_messages(self) -> None:
        """Receive messages from Insider WebSocket."""
        import websockets

        async for message in self._websocket:
            self._last_message = datetime.now()
            signal = self._parse_message(message)
            if signal:
                self._notify_subscribers(signal)

    def _parse_message(self, message: str) -> Optional[Signal]:
        """Parse Insider/Hiring message."""
        try:
            data = json.loads(message)

            return Signal(
                id=data.get("transaction_id", ""),
                source="insider",
                signal_type="bullish" if data.get("transaction_type") == "purchase" else "bearish",
                ticker=data.get("company_ticker", ""),
                company_name=data.get("company_name", ""),
                title=data.get("event_type", "Insider Activity"),
                description=data.get("description", ""),
                score=float(data.get("score", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                timestamp=datetime.fromisoformat(data["timestamp"])
                    if "timestamp" in data else datetime.now(),
                metadata={
                    "insider_name": data.get("insider_name"),
                    "position": data.get("position"),
                    "shares": data.get("shares"),
                    "value": data.get("value"),
                },
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse insider message: {e}")
            return None


class SignalAggregator:
    """
    Aggregates signals from multiple WebSocket clients.

    Provides unified interface for managing all signal sources
    and broadcasting signals to subscribers.
    """

    def __init__(self, demo_mode: bool = True):
        """
        Initialize signal aggregator.

        Args:
            demo_mode: Use mock clients for demo
        """
        self.demo_mode = demo_mode
        self._clients: Dict[str, BaseWebSocketClient] = {}
        self._callbacks: List[Callable[[Signal], None]] = []
        self._signal_history: List[Signal] = []
        self._max_history = 100
        self._lock = threading.Lock()

    def add_client(self, name: str, client: BaseWebSocketClient) -> None:
        """
        Add a WebSocket client.

        Args:
            name: Client identifier
            client: WebSocket client instance
        """
        self._clients[name] = client
        client.subscribe(self._on_signal)

    def remove_client(self, name: str) -> None:
        """
        Remove a WebSocket client.

        Args:
            name: Client identifier
        """
        if name in self._clients:
            self._clients[name].unsubscribe(self._on_signal)
            self._clients[name].stop()
            del self._clients[name]

    def subscribe(self, callback: Callable[[Signal], None]) -> None:
        """Subscribe to all signals."""
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable[[Signal], None]) -> None:
        """Unsubscribe from signals."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _on_signal(self, signal: Signal) -> None:
        """Handle incoming signal from any client."""
        with self._lock:
            self._signal_history.insert(0, signal)
            self._signal_history = self._signal_history[:self._max_history]

        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    def get_recent_signals(
        self,
        limit: int = 50,
        source: Optional[str] = None,
        ticker: Optional[str] = None,
        hours: int = 24,
    ) -> List[Signal]:
        """
        Get recent signals with optional filtering.

        Args:
            limit: Maximum number of signals to return
            source: Filter by source ('clinical_trial', 'patent', 'insider')
            ticker: Filter by ticker symbol
            hours: Filter to signals within this many hours

        Returns:
            List of signals
        """
        from datetime import timedelta

        with self._lock:
            signals = self._signal_history.copy()

        cutoff = datetime.now() - timedelta(hours=hours)
        signals = [s for s in signals if s.timestamp >= cutoff]

        if source:
            signals = [s for s in signals if s.source == source]

        if ticker:
            signals = [s for s in signals if s.ticker.upper() == ticker.upper()]

        return signals[:limit]

    def get_connection_status(self) -> Dict[str, ConnectionStatus]:
        """Get status of all clients."""
        return {
            name: client.get_status()
            for name, client in self._clients.items()
        }

    def is_any_connected(self) -> bool:
        """Check if any client is connected."""
        return any(client.is_connected for client in self._clients.values())

    def start_all(self) -> None:
        """Start all clients."""
        for client in self._clients.values():
            client.start()

    def stop_all(self) -> None:
        """Stop all clients."""
        for client in self._clients.values():
            client.stop()


def create_signal_aggregator(
    config: Dict[str, Any],
    demo_mode: bool = True,
) -> SignalAggregator:
    """
    Factory function to create a signal aggregator.

    Args:
        config: Configuration dictionary with WebSocket endpoints
        demo_mode: Use mock clients

    Returns:
        Configured SignalAggregator instance
    """
    aggregator = SignalAggregator(demo_mode=demo_mode)

    if demo_mode:
        # Use mock client in demo mode
        mock_client = MockWebSocketClient(signal_interval=15.0)
        aggregator.add_client("mock", mock_client)
    else:
        # Add real WebSocket clients
        ws_config = config.get("websockets", {})

        if "clinical_trials" in ws_config:
            client = ClinicalTrialWebSocketClient(ws_config["clinical_trials"])
            aggregator.add_client("clinical_trials", client)

        if "patent_intelligence" in ws_config:
            client = PatentWebSocketClient(ws_config["patent_intelligence"])
            aggregator.add_client("patent_intelligence", client)

        if "insider_hiring" in ws_config:
            client = InsiderWebSocketClient(ws_config["insider_hiring"])
            aggregator.add_client("insider_hiring", client)

    return aggregator
