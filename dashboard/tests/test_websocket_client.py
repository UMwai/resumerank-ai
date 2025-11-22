"""
Tests for WebSocket Client Module

Tests cover:
- Signal dataclass
- ReconnectionStrategy
- MockWebSocketClient
- SignalAggregator
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.websocket_client import (
    Signal,
    ConnectionState,
    ConnectionStatus,
    ReconnectionStrategy,
    MockWebSocketClient,
    SignalAggregator,
    create_signal_aggregator,
)


class TestSignal:
    """Test Signal dataclass."""

    def test_signal_creation(self):
        """Test creating a Signal."""
        signal = Signal(
            id="test_1",
            source="clinical_trial",
            signal_type="bullish",
            ticker="MRNA",
            company_name="Moderna Inc.",
            title="Phase 3 Results",
            description="Positive results announced",
            score=0.85,
            confidence=0.90,
        )

        assert signal.id == "test_1"
        assert signal.source == "clinical_trial"
        assert signal.signal_type == "bullish"
        assert signal.ticker == "MRNA"
        assert signal.score == 0.85
        assert signal.confidence == 0.90

    def test_signal_to_dict(self):
        """Test Signal to_dict conversion."""
        signal = Signal(
            id="test_2",
            source="patent",
            signal_type="bearish",
            ticker="PFE",
            company_name="Pfizer",
            title="Patent Expiry",
            description="Key patent expiring",
            score=0.35,
            confidence=0.80,
        )

        data = signal.to_dict()

        assert data["id"] == "test_2"
        assert data["source"] == "patent"
        assert data["signal_type"] == "bearish"
        assert data["ticker"] == "PFE"
        assert "timestamp" in data

    def test_signal_from_dict(self):
        """Test Signal from_dict creation."""
        data = {
            "id": "test_3",
            "source": "insider",
            "signal_type": "bullish",
            "ticker": "GILD",
            "company_name": "Gilead",
            "title": "Insider Purchase",
            "description": "CEO bought shares",
            "score": 0.75,
            "confidence": 0.85,
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {"insider_name": "John Doe"},
        }

        signal = Signal.from_dict(data)

        assert signal.id == "test_3"
        assert signal.source == "insider"
        assert signal.ticker == "GILD"
        assert signal.metadata["insider_name"] == "John Doe"


class TestReconnectionStrategy:
    """Test ReconnectionStrategy."""

    def test_initial_delay(self):
        """Test initial delay is correct."""
        strategy = ReconnectionStrategy(initial_delay=1.0)
        delay = strategy.get_delay()

        # Should be around 1.0 (with jitter)
        assert 0.9 <= delay <= 1.1

    def test_exponential_backoff(self):
        """Test exponential backoff increases delay."""
        strategy = ReconnectionStrategy(
            initial_delay=1.0,
            multiplier=2.0,
            jitter=0.0,  # Disable jitter for predictable testing
        )

        delays = [strategy.get_delay() for _ in range(4)]

        # Should double each time: 1, 2, 4, 8
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0
        assert delays[3] == 8.0

    def test_max_delay(self):
        """Test max delay cap."""
        strategy = ReconnectionStrategy(
            initial_delay=1.0,
            max_delay=5.0,
            multiplier=2.0,
            jitter=0.0,
        )

        # Get many delays
        for _ in range(10):
            delay = strategy.get_delay()
            assert delay <= 5.0

    def test_reset(self):
        """Test reset resets attempt counter."""
        strategy = ReconnectionStrategy(initial_delay=1.0, jitter=0.0)

        # Get a few delays
        strategy.get_delay()
        strategy.get_delay()
        strategy.get_delay()

        assert strategy.attempts == 3

        strategy.reset()

        assert strategy.attempts == 0
        assert strategy.get_delay() == 1.0

    def test_should_retry_limited(self):
        """Test should_retry with limited attempts."""
        strategy = ReconnectionStrategy(max_attempts=3)

        assert strategy.should_retry() is True
        strategy.get_delay()
        assert strategy.should_retry() is True
        strategy.get_delay()
        assert strategy.should_retry() is True
        strategy.get_delay()
        assert strategy.should_retry() is False

    def test_should_retry_unlimited(self):
        """Test should_retry with unlimited attempts."""
        strategy = ReconnectionStrategy(max_attempts=0)

        for _ in range(100):
            assert strategy.should_retry() is True
            strategy.get_delay()


class TestMockWebSocketClient:
    """Test MockWebSocketClient."""

    def test_client_creation(self):
        """Test creating a mock client."""
        client = MockWebSocketClient(signal_interval=5.0)

        assert client.endpoint == "ws://mock/signals"
        assert client.name == "MockSignals"
        assert client.signal_interval == 5.0

    def test_initial_state(self):
        """Test initial connection state."""
        client = MockWebSocketClient()

        assert client.state == ConnectionState.DISCONNECTED
        assert client.is_connected is False

    def test_get_status(self):
        """Test get_status returns ConnectionStatus."""
        client = MockWebSocketClient()
        status = client.get_status()

        assert isinstance(status, ConnectionStatus)
        assert status.endpoint == "ws://mock/signals"
        assert status.state == ConnectionState.DISCONNECTED

    def test_subscribe_unsubscribe(self):
        """Test subscribe and unsubscribe."""
        client = MockWebSocketClient()

        callback = MagicMock()

        # Subscribe
        client.subscribe(callback)
        assert callback in client._callbacks

        # Don't add duplicate
        client.subscribe(callback)
        assert client._callbacks.count(callback) == 1

        # Unsubscribe
        client.unsubscribe(callback)
        assert callback not in client._callbacks

    def test_generate_mock_signal(self):
        """Test mock signal generation."""
        client = MockWebSocketClient()
        signal = client._generate_mock_signal()

        assert isinstance(signal, Signal)
        assert signal.source in ["clinical_trial", "patent", "insider"]
        assert signal.ticker in ["MRNA", "VRTX", "ABBV", "PFE", "GILD", "REGN", "BIIB", "AMGN"]
        assert 0 <= signal.score <= 1
        assert 0 <= signal.confidence <= 1
        assert signal.signal_type in ["bullish", "bearish"]


class TestSignalAggregator:
    """Test SignalAggregator."""

    def test_aggregator_creation(self):
        """Test creating an aggregator."""
        aggregator = SignalAggregator(demo_mode=True)

        assert aggregator.demo_mode is True
        assert len(aggregator._clients) == 0

    def test_add_remove_client(self):
        """Test adding and removing clients."""
        aggregator = SignalAggregator()
        client = MockWebSocketClient()

        aggregator.add_client("mock", client)
        assert "mock" in aggregator._clients

        aggregator.remove_client("mock")
        assert "mock" not in aggregator._clients

    def test_subscribe_to_signals(self):
        """Test subscribing to aggregated signals."""
        aggregator = SignalAggregator()
        callback = MagicMock()

        aggregator.subscribe(callback)
        assert callback in aggregator._callbacks

        aggregator.unsubscribe(callback)
        assert callback not in aggregator._callbacks

    def test_get_recent_signals_empty(self):
        """Test get_recent_signals when empty."""
        aggregator = SignalAggregator()
        signals = aggregator.get_recent_signals()

        assert signals == []

    def test_get_recent_signals_with_filter(self):
        """Test get_recent_signals with filtering."""
        aggregator = SignalAggregator()

        # Manually add signals to history
        signal1 = Signal(
            id="1", source="clinical_trial", signal_type="bullish",
            ticker="MRNA", company_name="Moderna", title="Test",
            description="Test", score=0.8, confidence=0.9,
        )
        signal2 = Signal(
            id="2", source="patent", signal_type="bearish",
            ticker="PFE", company_name="Pfizer", title="Test",
            description="Test", score=0.3, confidence=0.8,
        )

        aggregator._signal_history = [signal1, signal2]

        # Filter by source
        clinical_signals = aggregator.get_recent_signals(source="clinical_trial")
        assert len(clinical_signals) == 1
        assert clinical_signals[0].ticker == "MRNA"

        # Filter by ticker
        pfizer_signals = aggregator.get_recent_signals(ticker="PFE")
        assert len(pfizer_signals) == 1
        assert pfizer_signals[0].source == "patent"

    def test_get_connection_status(self):
        """Test get_connection_status."""
        aggregator = SignalAggregator()
        client = MockWebSocketClient()
        aggregator.add_client("mock", client)

        status = aggregator.get_connection_status()

        assert "mock" in status
        assert isinstance(status["mock"], ConnectionStatus)

    def test_is_any_connected(self):
        """Test is_any_connected."""
        aggregator = SignalAggregator()
        client = MockWebSocketClient()
        aggregator.add_client("mock", client)

        # Initially not connected
        assert aggregator.is_any_connected() is False


class TestCreateSignalAggregator:
    """Test create_signal_aggregator factory function."""

    def test_demo_mode(self):
        """Test creating aggregator in demo mode."""
        aggregator = create_signal_aggregator({}, demo_mode=True)

        assert "mock" in aggregator._clients
        assert aggregator.demo_mode is True

    def test_production_mode_no_config(self):
        """Test creating aggregator without WebSocket config."""
        aggregator = create_signal_aggregator({}, demo_mode=False)

        # Should have no clients without config
        assert len(aggregator._clients) == 0
