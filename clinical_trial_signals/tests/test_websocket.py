"""
Tests for the WebSocket Support module.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.websocket import (
    ConnectionManager,
    SignalWebSocket,
    WebSocketEvent,
    EventType,
    Subscription,
    get_websocket_server,
)


class TestWebSocketEvent:
    """Tests for WebSocketEvent dataclass."""

    def test_event_creation(self):
        """Test creating a WebSocketEvent."""
        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={"trial_id": "NCT12345678", "score": 8.5}
        )

        assert event.event_type == EventType.SIGNAL_DETECTED
        assert event.data["trial_id"] == "NCT12345678"
        assert isinstance(event.timestamp, datetime)

    def test_event_to_json(self):
        """Test converting event to JSON."""
        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={"trial_id": "NCT12345678"}
        )

        json_str = event.to_json()

        assert "signal_detected" in json_str
        assert "NCT12345678" in json_str
        assert "timestamp" in json_str

    def test_event_with_custom_id(self):
        """Test event with custom ID."""
        event = WebSocketEvent(
            event_type=EventType.HEARTBEAT,
            data={},
            event_id="custom_123"
        )

        json_str = event.to_json()
        assert "custom_123" in json_str


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_empty_subscription(self):
        """Test empty subscription (all events)."""
        sub = Subscription()

        assert len(sub.trial_ids) == 0
        assert len(sub.company_tickers) == 0
        assert len(sub.event_types) == 0
        assert sub.min_score is None

    def test_subscription_with_filters(self):
        """Test subscription with filters."""
        sub = Subscription(
            trial_ids={"NCT001", "NCT002"},
            company_tickers={"BTCH", "PHRM"},
            min_score=7.0,
            min_confidence=0.7
        )

        assert "NCT001" in sub.trial_ids
        assert "BTCH" in sub.company_tickers
        assert sub.min_score == 7.0


class TestConnectionManager:
    """Tests for ConnectionManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connecting a client."""
        mock_websocket = AsyncMock()

        await self.manager.connect(mock_websocket, "client1")

        assert "client1" in self.manager.active_connections
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_subscription(self):
        """Test connecting with subscription."""
        mock_websocket = AsyncMock()
        sub = Subscription(trial_ids={"NCT001"})

        await self.manager.connect(mock_websocket, "client1", sub)

        assert "NCT001" in self.manager.subscriptions["client1"].trial_ids

    def test_disconnect(self):
        """Test disconnecting a client."""
        # Manually add connection
        mock_ws = Mock()
        self.manager.active_connections["client1"] = mock_ws
        self.manager.subscriptions["client1"] = Subscription()

        self.manager.disconnect("client1")

        assert "client1" not in self.manager.active_connections
        assert "client1" not in self.manager.subscriptions

    def test_update_subscription(self):
        """Test updating subscription."""
        mock_ws = Mock()
        self.manager.active_connections["client1"] = mock_ws
        self.manager.subscriptions["client1"] = Subscription()

        new_sub = Subscription(trial_ids={"NCT002"})
        result = self.manager.update_subscription("client1", new_sub)

        assert result is True
        assert "NCT002" in self.manager.subscriptions["client1"].trial_ids

    def test_update_subscription_nonexistent(self):
        """Test updating subscription for nonexistent client."""
        new_sub = Subscription()
        result = self.manager.update_subscription("nonexistent", new_sub)

        assert result is False

    def test_should_receive_no_filters(self):
        """Test should_receive with no filters (receives all)."""
        mock_ws = Mock()
        self.manager.active_connections["client1"] = mock_ws
        self.manager.subscriptions["client1"] = Subscription()

        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={}
        )

        result = self.manager._should_receive(
            "client1", event, trial_id="NCT001"
        )

        assert result is True

    def test_should_receive_with_trial_filter(self):
        """Test should_receive with trial filter."""
        mock_ws = Mock()
        self.manager.active_connections["client1"] = mock_ws
        self.manager.subscriptions["client1"] = Subscription(
            trial_ids={"NCT001", "NCT002"}
        )

        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={}
        )

        # Should receive NCT001
        assert self.manager._should_receive("client1", event, trial_id="NCT001") is True

        # Should not receive NCT999
        assert self.manager._should_receive("client1", event, trial_id="NCT999") is False

    def test_should_receive_with_score_filter(self):
        """Test should_receive with score filter."""
        mock_ws = Mock()
        self.manager.active_connections["client1"] = mock_ws
        self.manager.subscriptions["client1"] = Subscription(min_score=7.0)

        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={}
        )

        # Should receive score 8.0
        assert self.manager._should_receive("client1", event, score=8.0) is True

        # Should not receive score 6.0
        assert self.manager._should_receive("client1", event, score=6.0) is False

    @pytest.mark.asyncio
    async def test_send_personal_event(self):
        """Test sending event to specific client."""
        mock_websocket = AsyncMock()
        self.manager.active_connections["client1"] = mock_websocket

        event = WebSocketEvent(
            event_type=EventType.HEARTBEAT,
            data={"status": "ok"}
        )

        result = await self.manager.send_personal_event("client1", event)

        assert result is True
        mock_websocket.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_personal_event_nonexistent(self):
        """Test sending to nonexistent client."""
        event = WebSocketEvent(
            event_type=EventType.HEARTBEAT,
            data={}
        )

        result = await self.manager.send_personal_event("nonexistent", event)

        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting to all clients."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        self.manager.active_connections["client1"] = mock_ws1
        self.manager.active_connections["client2"] = mock_ws2
        self.manager.subscriptions["client1"] = Subscription()
        self.manager.subscriptions["client2"] = Subscription()

        event = WebSocketEvent(
            event_type=EventType.SYSTEM_ALERT,
            data={"message": "Test"}
        )

        sent = await self.manager.broadcast(event)

        assert sent == 2
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_filter(self):
        """Test broadcasting with subscription filters."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        self.manager.active_connections["client1"] = mock_ws1
        self.manager.active_connections["client2"] = mock_ws2
        self.manager.subscriptions["client1"] = Subscription(trial_ids={"NCT001"})
        self.manager.subscriptions["client2"] = Subscription(trial_ids={"NCT002"})

        event = WebSocketEvent(
            event_type=EventType.SIGNAL_DETECTED,
            data={}
        )

        # Broadcast for NCT001 - only client1 should receive
        sent = await self.manager.broadcast(event, trial_id="NCT001")

        assert sent == 1
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_not_called()

    def test_get_stats(self):
        """Test getting connection statistics."""
        mock_ws = Mock()
        self.manager.active_connections["client1"] = mock_ws
        self.manager.subscriptions["client1"] = Subscription()
        self.manager._connection_times["client1"] = datetime.utcnow()
        self.manager._message_counts["client1"] = 5

        stats = self.manager.get_stats()

        assert stats["active_connections"] == 1
        assert "client1" in stats["clients"]
        assert stats["clients"]["client1"]["messages_received"] == 5


class TestSignalWebSocket:
    """Tests for SignalWebSocket server."""

    def test_server_initialization(self):
        """Test server initialization."""
        server = SignalWebSocket(port=8002, host="127.0.0.1")

        assert server.port == 8002
        assert server.host == "127.0.0.1"
        assert server.app is not None

    def test_server_has_routes(self):
        """Test server has required routes."""
        server = SignalWebSocket()

        routes = [route.path for route in server.app.routes]

        assert "/" in routes
        assert "/health" in routes
        assert "/stats" in routes
        assert "/ws/{client_id}" in routes

    @pytest.mark.asyncio
    async def test_emit_signal(self):
        """Test emitting a signal event."""
        server = SignalWebSocket()

        # Add a mock client
        mock_ws = AsyncMock()
        await server.manager.connect(mock_ws, "test_client")

        # Emit signal
        sent = await server.emit_signal(
            trial_id="NCT12345678",
            ticker="BTCH",
            signal_type="insider_buying",
            signal_weight=4,
            signal_value="Insider purchase detected",
            score=8.0,
            confidence=0.85
        )

        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_score_update(self):
        """Test emitting a score update event."""
        server = SignalWebSocket()

        mock_ws = AsyncMock()
        await server.manager.connect(mock_ws, "test_client")

        sent = await server.emit_score_update(
            trial_id="NCT12345678",
            ticker="BTCH",
            score=8.5,
            confidence=0.85,
            recommendation="STRONG_BUY",
            previous_score=7.5
        )

        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_prediction_update(self):
        """Test emitting a prediction update event."""
        server = SignalWebSocket()

        mock_ws = AsyncMock()
        await server.manager.connect(mock_ws, "test_client")

        sent = await server.emit_prediction_update(
            trial_id="NCT12345678",
            ticker="BTCH",
            success_probability=72.5,
            risk_level="moderate",
            key_factors=[{"factor": "enrollment", "impact": "positive"}]
        )

        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_system_alert(self):
        """Test emitting a system alert."""
        server = SignalWebSocket()

        mock_ws = AsyncMock()
        await server.manager.connect(mock_ws, "test_client")

        sent = await server.emit_system_alert(
            message="System maintenance scheduled",
            severity="warning"
        )

        assert sent == 1


class TestGetWebSocketServer:
    """Tests for the server singleton."""

    def test_get_server_creates_instance(self):
        """Test that get_websocket_server creates an instance."""
        # Reset singleton
        import api.websocket as ws_module
        ws_module._websocket_server = None

        server = get_websocket_server(port=8003)

        assert server is not None
        assert server.port == 8003

    def test_get_server_returns_same_instance(self):
        """Test that get_websocket_server returns same instance."""
        import api.websocket as ws_module
        ws_module._websocket_server = None

        server1 = get_websocket_server(port=8004)
        server2 = get_websocket_server(port=8005)  # Should be ignored

        assert server1 is server2
        assert server1.port == 8004  # First call's port


class TestEventTypes:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Test all required event types exist."""
        assert EventType.SIGNAL_DETECTED
        assert EventType.SCORE_UPDATED
        assert EventType.TRIAL_UPDATE
        assert EventType.PREDICTION_UPDATE
        assert EventType.SYSTEM_ALERT
        assert EventType.HEARTBEAT
        assert EventType.SUBSCRIPTION_CONFIRMED
        assert EventType.ERROR
