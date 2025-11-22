"""
WebSocket Support for Clinical Trial Signal Detection System.

Provides real-time streaming of signal updates to connected clients:
- FastAPI WebSocket endpoint
- Connection management for multiple clients
- Event broadcasting for new signals
- Subscription filtering by trial/company
- Health monitoring and reconnection support
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class EventType(Enum):
    """WebSocket event types."""
    SIGNAL_DETECTED = "signal_detected"
    SCORE_UPDATED = "score_updated"
    TRIAL_UPDATE = "trial_update"
    PREDICTION_UPDATE = "prediction_update"
    SYSTEM_ALERT = "system_alert"
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    ERROR = "error"


@dataclass
class WebSocketEvent:
    """WebSocket event structure."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps({
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id or f"{self.event_type.value}_{int(self.timestamp.timestamp() * 1000)}"
        })


@dataclass
class Subscription:
    """Client subscription preferences."""
    trial_ids: Set[str] = field(default_factory=set)  # Empty = all trials
    company_tickers: Set[str] = field(default_factory=set)  # Empty = all companies
    event_types: Set[EventType] = field(default_factory=set)  # Empty = all events
    min_score: Optional[float] = None  # Only signals with score >= this
    min_confidence: Optional[float] = None  # Only signals with confidence >= this


class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.

    Handles:
    - Connection tracking
    - Subscription management
    - Message broadcasting with filtering
    - Connection health monitoring
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self._connection_times: Dict[str, datetime] = {}
        self._message_counts: Dict[str, int] = {}

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        subscription: Optional[Subscription] = None
    ) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            subscription: Optional initial subscription
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = subscription or Subscription()
        self._connection_times[client_id] = datetime.utcnow()
        self._message_counts[client_id] = 0

        logger.info(f"WebSocket connected: {client_id} (total: {len(self.active_connections)})")

        # Send confirmation
        await self.send_personal_event(
            client_id,
            WebSocketEvent(
                event_type=EventType.SUBSCRIPTION_CONFIRMED,
                data={
                    "client_id": client_id,
                    "subscription": {
                        "trial_ids": list(self.subscriptions[client_id].trial_ids),
                        "company_tickers": list(self.subscriptions[client_id].company_tickers),
                        "event_types": [e.value for e in self.subscriptions[client_id].event_types],
                    }
                }
            )
        )

    def disconnect(self, client_id: str) -> None:
        """
        Remove a WebSocket connection.

        Args:
            client_id: Client identifier to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self._connection_times:
            del self._connection_times[client_id]
        if client_id in self._message_counts:
            del self._message_counts[client_id]

        logger.info(f"WebSocket disconnected: {client_id} (remaining: {len(self.active_connections)})")

    def update_subscription(self, client_id: str, subscription: Subscription) -> bool:
        """
        Update subscription for a client.

        Args:
            client_id: Client identifier
            subscription: New subscription settings

        Returns:
            True if updated successfully
        """
        if client_id not in self.active_connections:
            return False

        self.subscriptions[client_id] = subscription
        logger.debug(f"Updated subscription for {client_id}")
        return True

    def _should_receive(
        self,
        client_id: str,
        event: WebSocketEvent,
        trial_id: Optional[str] = None,
        ticker: Optional[str] = None,
        score: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """
        Check if a client should receive an event based on subscription.

        Args:
            client_id: Client identifier
            event: Event to check
            trial_id: Associated trial ID (if any)
            ticker: Associated company ticker (if any)
            score: Signal score (if any)
            confidence: Signal confidence (if any)

        Returns:
            True if client should receive the event
        """
        sub = self.subscriptions.get(client_id)
        if not sub:
            return True  # Default: receive all

        # Check event type filter
        if sub.event_types and event.event_type not in sub.event_types:
            return False

        # Check trial filter
        if sub.trial_ids and trial_id and trial_id not in sub.trial_ids:
            return False

        # Check company filter
        if sub.company_tickers and ticker and ticker not in sub.company_tickers:
            return False

        # Check score threshold
        if sub.min_score is not None and score is not None:
            if score < sub.min_score:
                return False

        # Check confidence threshold
        if sub.min_confidence is not None and confidence is not None:
            if confidence < sub.min_confidence:
                return False

        return True

    async def send_personal_event(self, client_id: str, event: WebSocketEvent) -> bool:
        """
        Send event to a specific client.

        Args:
            client_id: Target client identifier
            event: Event to send

        Returns:
            True if sent successfully
        """
        websocket = self.active_connections.get(client_id)
        if not websocket:
            return False

        try:
            await websocket.send_text(event.to_json())
            self._message_counts[client_id] = self._message_counts.get(client_id, 0) + 1
            return True
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")
            return False

    async def broadcast(
        self,
        event: WebSocketEvent,
        trial_id: Optional[str] = None,
        ticker: Optional[str] = None,
        score: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> int:
        """
        Broadcast event to all subscribed clients.

        Args:
            event: Event to broadcast
            trial_id: Associated trial ID for filtering
            ticker: Associated ticker for filtering
            score: Signal score for filtering
            confidence: Signal confidence for filtering

        Returns:
            Number of clients that received the message
        """
        sent_count = 0
        disconnected = []

        for client_id, websocket in self.active_connections.items():
            # Check if client should receive this event
            if not self._should_receive(client_id, event, trial_id, ticker, score, confidence):
                continue

            try:
                await websocket.send_text(event.to_json())
                self._message_counts[client_id] = self._message_counts.get(client_id, 0) + 1
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

        logger.debug(f"Broadcast {event.event_type.value} to {sent_count} clients")
        return sent_count

    async def send_heartbeat(self) -> int:
        """
        Send heartbeat to all connected clients.

        Returns:
            Number of clients that received heartbeat
        """
        event = WebSocketEvent(
            event_type=EventType.HEARTBEAT,
            data={"status": "ok", "connections": len(self.active_connections)}
        )

        return await self.broadcast(event)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": sum(self._message_counts.values()),
            "clients": {
                client_id: {
                    "connected_at": self._connection_times.get(client_id, datetime.utcnow()).isoformat(),
                    "messages_received": self._message_counts.get(client_id, 0),
                    "subscription": {
                        "trial_ids": list(sub.trial_ids),
                        "company_tickers": list(sub.company_tickers),
                    }
                }
                for client_id, sub in self.subscriptions.items()
            }
        }


class SignalWebSocket:
    """
    FastAPI WebSocket server for clinical trial signals.

    Provides:
    - /ws/{client_id} - WebSocket endpoint
    - /health - Health check endpoint
    - /stats - Connection statistics
    """

    def __init__(
        self,
        port: int = 8001,
        host: str = "0.0.0.0",
        heartbeat_interval: int = 30
    ):
        """
        Initialize WebSocket server.

        Args:
            port: Port to run on (default 8001)
            host: Host to bind to
            heartbeat_interval: Seconds between heartbeats
        """
        self.port = port
        self.host = host
        self.heartbeat_interval = heartbeat_interval
        self.manager = ConnectionManager()
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Create FastAPI app
        self.app = FastAPI(
            title="Clinical Trial Signals WebSocket API",
            description="Real-time WebSocket API for clinical trial signal updates",
            version="1.0.0",
            lifespan=self._lifespan
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage app lifespan (startup/shutdown)."""
        # Startup
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"WebSocket server starting on {self.host}:{self.port}")
        yield
        # Shutdown
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket server shutdown")

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeats."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.manager.send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    def _register_routes(self) -> None:
        """Register FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Clinical Trial Signals WebSocket API",
                "version": "1.0.0",
                "websocket_endpoint": f"ws://{self.host}:{self.port}/ws/{{client_id}}",
                "status": "running"
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "connections": len(self.manager.active_connections),
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.get("/stats")
        async def stats():
            """Connection statistics."""
            return self.manager.get_stats()

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            client_id: str,
            trial_ids: Optional[str] = Query(None, description="Comma-separated trial IDs"),
            tickers: Optional[str] = Query(None, description="Comma-separated tickers"),
            min_score: Optional[float] = Query(None, ge=0, le=10),
            min_confidence: Optional[float] = Query(None, ge=0, le=1)
        ):
            """
            WebSocket endpoint for real-time signal updates.

            Connect with optional subscription filters:
            - trial_ids: Filter by specific trial IDs
            - tickers: Filter by company tickers
            - min_score: Minimum score threshold
            - min_confidence: Minimum confidence threshold

            Example: ws://localhost:8001/ws/client1?tickers=BTCH,PHRM&min_score=7
            """
            # Parse subscription parameters
            subscription = Subscription(
                trial_ids=set(trial_ids.split(",")) if trial_ids else set(),
                company_tickers=set(tickers.split(",")) if tickers else set(),
                min_score=min_score,
                min_confidence=min_confidence
            )

            await self.manager.connect(websocket, client_id, subscription)

            try:
                while True:
                    # Receive messages from client
                    data = await websocket.receive_text()

                    try:
                        message = json.loads(data)
                        await self._handle_client_message(client_id, message)
                    except json.JSONDecodeError:
                        await self.manager.send_personal_event(
                            client_id,
                            WebSocketEvent(
                                event_type=EventType.ERROR,
                                data={"error": "Invalid JSON"}
                            )
                        )

            except WebSocketDisconnect:
                self.manager.disconnect(client_id)

    async def _handle_client_message(self, client_id: str, message: Dict) -> None:
        """
        Handle incoming message from client.

        Supported message types:
        - subscribe: Update subscription filters
        - ping: Respond with pong
        """
        msg_type = message.get("type")

        if msg_type == "subscribe":
            # Update subscription
            sub_data = message.get("subscription", {})
            subscription = Subscription(
                trial_ids=set(sub_data.get("trial_ids", [])),
                company_tickers=set(sub_data.get("tickers", [])),
                min_score=sub_data.get("min_score"),
                min_confidence=sub_data.get("min_confidence")
            )
            self.manager.update_subscription(client_id, subscription)

            await self.manager.send_personal_event(
                client_id,
                WebSocketEvent(
                    event_type=EventType.SUBSCRIPTION_CONFIRMED,
                    data={"subscription": sub_data}
                )
            )

        elif msg_type == "ping":
            await self.manager.send_personal_event(
                client_id,
                WebSocketEvent(
                    event_type=EventType.HEARTBEAT,
                    data={"pong": True}
                )
            )

    # Public methods for emitting events

    async def emit_signal(
        self,
        trial_id: str,
        ticker: str,
        signal_type: str,
        signal_weight: int,
        signal_value: str,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
        additional_data: Optional[Dict] = None
    ) -> int:
        """
        Emit a new signal event to subscribed clients.

        Args:
            trial_id: Trial NCT ID
            ticker: Company ticker
            signal_type: Type of signal detected
            signal_weight: Signal weight value
            signal_value: Signal description
            score: Current composite score
            confidence: Confidence level
            additional_data: Additional signal data

        Returns:
            Number of clients notified
        """
        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={
                "trial_id": trial_id,
                "ticker": ticker,
                "signal_type": signal_type,
                "signal_weight": signal_weight,
                "signal_value": signal_value,
                "score": score,
                "confidence": confidence,
                **(additional_data or {})
            }
        )

        return await self.manager.broadcast(
            event,
            trial_id=trial_id,
            ticker=ticker,
            score=score,
            confidence=confidence
        )

    async def emit_score_update(
        self,
        trial_id: str,
        ticker: str,
        score: float,
        confidence: float,
        recommendation: str,
        previous_score: Optional[float] = None
    ) -> int:
        """
        Emit a score update event.

        Args:
            trial_id: Trial NCT ID
            ticker: Company ticker
            score: New composite score
            confidence: Confidence level
            recommendation: Investment recommendation
            previous_score: Previous score (for change tracking)

        Returns:
            Number of clients notified
        """
        event = WebSocketEvent(
            event_type=EventType.SCORE_UPDATED,
            data={
                "trial_id": trial_id,
                "ticker": ticker,
                "score": score,
                "confidence": confidence,
                "recommendation": recommendation,
                "previous_score": previous_score,
                "score_change": score - previous_score if previous_score else None
            }
        )

        return await self.manager.broadcast(
            event,
            trial_id=trial_id,
            ticker=ticker,
            score=score,
            confidence=confidence
        )

    async def emit_prediction_update(
        self,
        trial_id: str,
        ticker: str,
        success_probability: float,
        risk_level: str,
        key_factors: List[Dict]
    ) -> int:
        """
        Emit a prediction update event.

        Args:
            trial_id: Trial NCT ID
            ticker: Company ticker
            success_probability: Predicted success probability
            risk_level: Risk level assessment
            key_factors: Key factors influencing prediction

        Returns:
            Number of clients notified
        """
        event = WebSocketEvent(
            event_type=EventType.PREDICTION_UPDATE,
            data={
                "trial_id": trial_id,
                "ticker": ticker,
                "success_probability": success_probability,
                "risk_level": risk_level,
                "key_factors": key_factors
            }
        )

        return await self.manager.broadcast(event, trial_id=trial_id, ticker=ticker)

    async def emit_system_alert(self, message: str, severity: str = "info") -> int:
        """
        Emit a system-wide alert.

        Args:
            message: Alert message
            severity: Alert severity (info, warning, error, critical)

        Returns:
            Number of clients notified
        """
        event = WebSocketEvent(
            event_type=EventType.SYSTEM_ALERT,
            data={
                "message": message,
                "severity": severity
            }
        )

        return await self.manager.broadcast(event)

    def run(self) -> None:
        """Run the WebSocket server."""
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# Singleton instance for use across the application
_websocket_server: Optional[SignalWebSocket] = None


def get_websocket_server(
    port: int = 8001,
    host: str = "0.0.0.0"
) -> SignalWebSocket:
    """
    Get or create the WebSocket server instance.

    Args:
        port: Port to run on
        host: Host to bind to

    Returns:
        SignalWebSocket instance
    """
    global _websocket_server
    if _websocket_server is None:
        _websocket_server = SignalWebSocket(port=port, host=host)
    return _websocket_server


if __name__ == "__main__":
    # Run the WebSocket server
    logging.basicConfig(level=logging.INFO)

    print("Starting Clinical Trial Signals WebSocket Server...")
    print("=" * 60)
    print(f"WebSocket endpoint: ws://localhost:8001/ws/{{client_id}}")
    print(f"Health check: http://localhost:8001/health")
    print(f"Statistics: http://localhost:8001/stats")
    print("=" * 60)

    server = SignalWebSocket(port=8001)
    server.run()
